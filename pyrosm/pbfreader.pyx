from pyrosm.exceptions import PBFNotImplemented
from struct import unpack
import warnings
import zlib
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.proto.osmformat_pb2 import HeaderBlock, PrimitiveBlock
from pyrosm.tagparser cimport tounicode, parse_dense_tags, parse_tags, explode_way_tags
from pyrosm._arrays cimport to_clong_array, concatenate_dicts_of_arrays
from pyrosm.delta_compression cimport (
    delta_decode_latitude,
    delta_decode_longitude,
    delta_decode_id,
    delta_decode_timestamp,
    delta_decode_changeset,
)
from pyrosm.data_filter cimport (
    get_nodeid_lookup_khash,
    nodes_for_way_exist_khash,
    get_latest_version,
    clean_empty_values_from_ways,
)
from pyrosm.utils import valid_header_block
from pyrosm.frames import create_df
from pyrosm.node_lookup import NodeLocations
import numpy as np
import pandas as pd
from libc.stdlib cimport malloc, free

_warned_slow_backend = False


cdef _warn_if_slow_protobuf_backend():
    global _warned_slow_backend
    if _warned_slow_backend:
        return
    _warned_slow_backend = True
    from google.protobuf.internal import api_implementation
    if api_implementation.Type() == "python":
        warnings.warn(
            "protobuf is using its pure-Python backend; PBF parsing will be "
            "much slower. Install a protobuf build with the C/upb backend for "
            "full speed.",
            RuntimeWarning,
            stacklevel=2,
        )


cdef get_primitive_blocks_and_string_tables(filepath):
    cdef int msg_len
    cdef bytes blob_data
    cdef str feature

    with open(filepath, 'rb') as f:

        # Check that the data stream is valid OSM
        # =======================================

        buf = f.read(4)
        msg_len = unpack('!L', buf)[0]
        msg = BlobHeader()
        msg.ParseFromString(f.read(msg_len))
        blob_header = msg

        msg = Blob()
        msg.ParseFromString(f.read(blob_header.datasize))
        blob_data = zlib.decompress(msg.zlib_data)
        header_block = HeaderBlock()
        header_block.ParseFromString(blob_data)

        if valid_header_block(header_block):
            pass

        # Gather primitive blocks and string tables
        primitive_blocks = []
        string_tables = []

        while True:
            # Read header
            buf = f.read(4)

            # Stop when the end has been reached
            if len(buf) == 0:
                break

            msg_len = unpack('!L', buf)[0]

            msg = BlobHeader()
            msg.ParseFromString(f.read(msg_len))
            blob_header = msg

            # Get data
            msg = Blob()
            msg.ParseFromString(f.read(blob_header.datasize))
            blob_data = zlib.decompress(msg.zlib_data)

            # Get primite block
            pblock = PrimitiveBlock()
            pblock.ParseFromString(blob_data)

            # Get string table and decode
            str_table = [tounicode(s) for s in pblock.stringtable.s]

            primitive_blocks.append(pblock)
            string_tables.append(str_table)

    return primitive_blocks, string_tables


cdef parse_dense(
        pblock,
        data,
        string_table,
        bounding_box,
        unix_time_filter,
        node_id_filter=None,
        bint keep_metadata=True,
):
    cdef:
        int node_granularity = pblock.granularity
        int timestamp_granularity = pblock.date_granularity
        int lon_offset = pblock.lon_offset
        int lat_offset = pblock.lat_offset

    # History (.osh) parsing filters on timestamp/visible, so the per-node metadata
    # is always decoded for history files regardless of the flag.
    cdef bint keep_meta = keep_metadata or unix_time_filter is not None

    # Get latitudes
    lats = delta_decode_latitude(data, node_granularity, lat_offset)

    # Get longitudes
    lons = delta_decode_longitude(data, node_granularity, lon_offset)

    # Ids
    ids = delta_decode_id(data)

    # Tags
    tags = np.empty(len(data.id), dtype=object)
    parsed = parse_dense_tags(data.keys_vals, string_table)

    # In some cases node-tags are not available at all
    if len(parsed) != 0:
        tags[:] = parsed

    # Visible flags (if visible is False, the element has been deleted)
    visible = np.array(list(data.denseinfo.visible)).astype(bool)
    if visible.shape[0] == 0:
        visible = np.full(len(data.id), False, dtype=bool)

    if keep_meta:
        # Version
        versions = np.array(list(data.denseinfo.version), dtype=np.int64)

        # Timestamp
        timestamps = delta_decode_timestamp(data, timestamp_granularity)

        # Changeset
        changesets = delta_decode_changeset(data)

        # Metadata might not be available, if so add empty
        # This can happen with BBBike data
        if versions.shape[0] == 0:
            versions = np.zeros(len(data.id), dtype=np.int8)
        if changesets.shape[0] == 0:
            changesets = np.zeros(len(data.id), dtype=np.int8)
        if timestamps.shape[0] == 0:
            timestamps = np.zeros(len(data.id), dtype=np.int8)

    # 'node_id_filter' (a second, completeness pass) keeps nodes by id rather than
    # by the bounding box, to fetch a kept way's vertices that lie just outside the box.
    mask = None
    if node_id_filter is not None:
        mask = np.isin(ids, node_id_filter)
    elif bounding_box is not None:
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)

    if mask is not None:
        ids = ids[mask]
        lons = lons[mask]
        lats = lats[mask]
        tags = tags[mask]
        visible = visible[mask]
        if keep_meta:
            versions = versions[mask]
            changesets = changesets[mask]
            timestamps = timestamps[mask]

    if unix_time_filter is not None:
        # Filter out node (versions) based on time
        mask = timestamps <= unix_time_filter
        ids = ids[mask]
        versions = versions[mask]
        changesets = changesets[mask]
        timestamps = timestamps[mask]
        lons = lons[mask]
        lats = lats[mask]
        tags = tags[mask]
        visible = visible[mask]

    if keep_meta:
        return [dict(id=ids,
                     version=versions,
                     changeset=changesets,
                     timestamp=timestamps,
                     lon=lons,
                     lat=lats,
                     tags=tags,
                     visible=visible,
                     )]
    return [dict(id=ids,
                 lon=lons,
                 lat=lats,
                 tags=tags,
                 visible=visible,
                 )]


# Handles the non-dense PBF node encoding (exercised via an osmium-generated
# non-dense fixture in the tests).
cdef parse_nodes(pblock, data, string_table, bounding_box, unix_time_filter=None,
                 node_id_filter=None, bint keep_metadata=True):
    # History (.osh) parsing filters on timestamp/visible, so the per-node metadata
    # is always decoded for history files regardless of the flag.
    cdef bint keep_meta = keep_metadata or unix_time_filter is not None

    ids = []
    versions = []
    changesets = []
    timestamps = []
    lons = []
    lats = []
    visibles = []
    tag_dicts = []

    granularity = pblock.granularity
    lon_offset = pblock.lon_offset
    lat_offset = pblock.lat_offset
    div = 1000000000

    for node in data:
        ids.append(node.id)
        lons.append(node.lon)
        lats.append(node.lat)
        visibles.append(node.info.visible)
        tag_dicts.append(parse_tags(node.keys, node.vals, string_table))
        if keep_meta:
            try:
                changesets.append(int(node.info.changeset))
            except:
                changesets.append(0)
            versions.append(node.info.version)
            timestamps.append(node.info.timestamp)

    id_ = np.array(ids, dtype=np.int64)
    lon = (np.array(lons, dtype=np.int64) * granularity + lon_offset) / div
    lat = (np.array(lats, dtype=np.int64) * granularity + lat_offset) / div
    visible = np.array(visibles, dtype=bool)
    # Keep the same schema as parse_dense (incl. 'tags' and 'visible') so the
    # node frame concatenates with dense nodes and the OSH 'visible' filter works.
    tags = np.empty(len(ids), dtype=object)
    tags[:] = tag_dicts

    if keep_meta:
        version = np.array(versions, dtype=np.int64)
        changeset = np.array(changesets, dtype=np.int64)
        timestamp = np.array(timestamps, dtype=np.int64)

    if node_id_filter is not None:
        # Completeness pass: keep nodes by id rather than by the bounding box.
        mask = np.isin(id_, node_id_filter)
    elif bounding_box is not None:
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax)
    else:
        mask = None

    if mask is not None:
        id_ = id_[mask]
        lon = lon[mask]
        lat = lat[mask]
        visible = visible[mask]
        tags = tags[mask]
        if keep_meta:
            version = version[mask]
            changeset = changeset[mask]
            timestamp = timestamp[mask]

    if unix_time_filter is not None:
        # Drop node versions newer than the requested timestamp (OSH files), so
        # get_latest_version later picks the latest version at/before it.
        time_mask = timestamp <= unix_time_filter
        id_ = id_[time_mask]
        version = version[time_mask]
        changeset = changeset[time_mask]
        timestamp = timestamp[time_mask]
        lon = lon[time_mask]
        lat = lat[time_mask]
        visible = visible[time_mask]
        tags = tags[time_mask]

    if keep_meta:
        return dict(id=id_,
                    version=version,
                    changeset=changeset,
                    timestamp=timestamp,
                    lon=lon,
                    lat=lat,
                    tags=tags,
                    visible=visible,
                    )
    return dict(id=id_,
                lon=lon,
                lat=lat,
                tags=tags,
                visible=visible,
                )


cdef parse_nodeids_from_ref_deltas(refs):
    cdef long long delta, nid = 0
    cdef int i, N = len(refs)
    cdef long long * refs_c = to_clong_array(refs)
    cdef long long *nodes = <long long *>malloc(N*sizeof(long long))

    try:
        for i in range(N):
            nid += refs_c[i]
            nodes[i] = nid
        return [x for x in nodes[:N]]

    finally:
        free(nodes)
        free(refs_c)


cdef parse_ways(
        data,
        string_table,
        node_lookup,
        unix_time_filter,
):
    cdef long long id
    cdef int version, i, timestamp, n=len(data)

    way_set = []
    for i in range(0, n):
        way = data[i]
        if unix_time_filter is not None:
            # Filter way versions according the time filter
            if way.info.timestamp > unix_time_filter:
                continue
        nodes = parse_nodeids_from_ref_deltas(way.refs)
        if node_lookup is not None:
            if nodes_for_way_exist_khash(nodes, node_lookup):
                way_set.append(
                    dict(
                        id=way.id,
                        version=way.info.version,
                        timestamp=way.info.timestamp,
                        visible=way.info.visible,
                        tags=parse_tags(way.keys, way.vals, string_table),
                        nodes=nodes,
                    )
                )
        else:
            way_set.append(
                dict(
                    id=way.id,
                    version=way.info.version,
                    timestamp=way.info.timestamp,
                    visible=way.info.visible,
                    tags=parse_tags(way.keys, way.vals, string_table),
                    nodes=nodes,
                )
            )
    return way_set


cdef get_relation_members(relation, str_table):
    relation_types = {
        0: b"node",
        1: b"way",
        2: b"relation"
        }

    # Member-ids
    member_N = len(relation.memids)
    member_id_deltas = np.zeros(member_N+1, dtype=np.int64)
    member_id_deltas[1:] = list(relation.memids)
    member_ids = np.cumsum(member_id_deltas)[1:]

    # Member types - 0: Node, 1: Way, 2: Relation
    member_types = np.array([relation_types[t]
                             for t in relation.types],
                            dtype=object)

    # Member roles
    member_roles = np.array([str_table[role_id] for role_id in relation.roles_sid],
                            dtype=object)

    return dict(
        member_id=member_ids,
        member_type=member_types,
        member_role=member_roles,
        )


cdef parse_relations(data, string_table, unix_time_filter):
    N = len(data)

    # Version
    versions = np.array([rel.info.version for rel in data], dtype=np.int64)

    ids = np.array([rel.id for rel in data], dtype=np.int64)

    timestamps = np.array([rel.info.timestamp for rel in data], dtype=np.int64)

    # Changeset
    changesets = np.array([rel.info.changeset for rel in data], dtype=np.int64)

    # Visible
    visible = np.array([rel.info.visible for rel in data], dtype=bool)

    # Relation members
    members = np.array([
        get_relation_members(rel, string_table)
        for rel in data],
        dtype=object
        )

    # Tags
    tags = np.array([
            parse_tags(rel.keys, rel.vals, string_table)
            for rel in data],
            dtype=object
            )

    # Filter by time
    if unix_time_filter is not None:
        mask = timestamps <= unix_time_filter
        ids = ids[mask]
        versions = versions[mask]
        changesets = changesets[mask]
        timestamps = timestamps[mask]
        members = members[mask]
        tags = tags[mask]
        visible = visible[mask]

    return [dict(id=ids,
                version=versions,
                changeset=changesets,
                timestamp=timestamps,
                members=members,
                tags=tags,
                visible=visible,
                )]


cpdef parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
):
    _warn_if_slow_protobuf_backend()
    return _parse_osm_data(filepath, bounding_box, exclude_relations,
                           unix_time_filter, keep_metadata)


cdef _parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
):
    all_ways = []
    all_nodes = []
    all_relations = []
    node_lookup_created = False

    primitive_blocks, string_tables = get_primitive_blocks_and_string_tables(filepath)

    for pblock, str_table in zip(primitive_blocks, string_tables):
        for pgroup in pblock.primitivegroup:
            if len(pgroup.dense.id) > 0:
                all_nodes += parse_dense(pblock, pgroup.dense, str_table, bounding_box,
                                         unix_time_filter, None, keep_metadata)
            elif len(pgroup.nodes) > 0:
                all_nodes += [parse_nodes(pblock, pgroup.nodes, str_table,
                                          bounding_box, unix_time_filter, None,
                                          keep_metadata)]
            elif len(pgroup.ways) > 0:
                # Once all the nodes have been parsed comes Ways
                if bounding_box is not None:
                    if not node_lookup_created:
                        node_lookup = get_nodeid_lookup_khash(all_nodes)
                    all_ways += parse_ways(pgroup.ways, str_table, node_lookup, unix_time_filter)
                else:
                    all_ways += parse_ways(pgroup.ways, str_table, None, unix_time_filter)
            elif len(pgroup.relations) > 0:
                if exclude_relations:
                    continue
                all_relations += parse_relations(pgroup.relations, str_table, unix_time_filter)

    # Explode the way tags
    all_ways = explode_way_tags(all_ways)

    # Concatenate nodes and create a DataFrame
    nodes_df = create_df(concatenate_dicts_of_arrays(all_nodes))
    relations_df = create_df(concatenate_dicts_of_arrays(all_relations))

    # If no nodes were parsed the nodes DataFrame is empty and has no 'id' column. Without
    # any nodes the ways cannot be georeferenced anyway, so return an empty result instead
    # of crashing later at nodes_df.set_index("id"). (A swapped/inverted bounding box is
    # rejected earlier with a ValueError at OSM() construction; reaching here means the
    # bounding box is well-formed but does not overlap this PBF's data extent.)
    if "id" not in nodes_df.columns:
        warnings.warn(
            "The given bounding box did not contain any OSM nodes, so no data could be "
            "parsed. It likely does not overlap the data extent of this PBF file.",
            UserWarning,
            stacklevel=2,
        )
        return {}, [], {}, {}

    # Keep the closest record to the timestamp if filter is used
    if unix_time_filter is not None:
        ways_df = pd.DataFrame(all_ways)
        # Drop deleted history items
        nodes_df = nodes_df.loc[nodes_df["visible"]==True].copy()
        ways_df = ways_df.loc[ways_df["visible"]==True].copy()
        relations_df = relations_df[relations_df["visible"]==True].copy()

        # Get latest version
        nodes_df = get_latest_version(nodes_df)
        all_ways = get_latest_version(ways_df).to_dict(orient="records")

        # DataFrame structure produces unnecesary None values that needs to be cleaned
        all_ways = clean_empty_values_from_ways(all_ways)

        relations_df = get_latest_version(relations_df)

    # Keys with numpy arrays
    all_nodes = {col: nodes_df[col].values for col in nodes_df.columns}
    all_relations = {col: relations_df[col].values for col in relations_df.columns}

    # Complete the geometries of ways that straddle the bounding box (#236). A way is
    # kept when >=1 of its nodes is inside the box, but the box-filtered node parse
    # dropped the vertices that lie just outside it, so its polygon/line would otherwise
    # be cut. Fetch ONLY those missing node coordinates with a second pass over the
    # already-in-memory blocks (no extra I/O / decompression) and add them to the
    # coordinate lookup used for geometry building. They are deliberately NOT added to
    # `all_nodes` (the standalone node features), so out-of-box nodes never leak into
    # node-feature results (e.g. POIs). bbox-only.
    coords_df = nodes_df
    if bounding_box is not None and len(all_ways) > 0:
        present_ids = set(nodes_df["id"].tolist())
        missing_ids = set()
        for way in all_ways:
            for node_id in way["nodes"]:
                if node_id not in present_ids:
                    missing_ids.add(node_id)
        if len(missing_ids) > 0:
            node_id_filter = np.array(sorted(missing_ids), dtype=np.int64)
            boundary_nodes = []
            for pblock, str_table in zip(primitive_blocks, string_tables):
                for pgroup in pblock.primitivegroup:
                    if len(pgroup.dense.id) > 0:
                        boundary_nodes += parse_dense(pblock, pgroup.dense, str_table,
                                                      None, unix_time_filter,
                                                      node_id_filter, keep_metadata)
                    elif len(pgroup.nodes) > 0:
                        boundary_nodes += [parse_nodes(pblock, pgroup.nodes, str_table,
                                                       None, unix_time_filter,
                                                       node_id_filter, keep_metadata)]
            if len(boundary_nodes) > 0:
                boundary_df = create_df(concatenate_dicts_of_arrays(boundary_nodes))
                if "id" in boundary_df.columns:
                    # Apply the same OSH latest-version selection to the boundary
                    # nodes as nodes_df received above, so timestamped geometries are
                    # built from the latest non-deleted vertex rather than whichever
                    # historical version drop_duplicates happens to keep first.
                    if unix_time_filter is not None:
                        boundary_df = boundary_df.loc[
                            boundary_df["visible"] == True
                        ].copy()
                        boundary_df = get_latest_version(boundary_df)
                    coords_df = pd.concat(
                        [nodes_df, boundary_df], ignore_index=True
                    ).drop_duplicates(subset="id")

    # Compact node-coordinate store (cykhash id->index map + column arrays),
    # replacing the per-node dict-of-dicts; geometry/graph code reads through it.
    node_coordinates_lookup = NodeLocations(coords_df)

    return all_nodes, all_ways, all_relations, node_coordinates_lookup
