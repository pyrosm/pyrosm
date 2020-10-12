from pyrosm.exceptions import PBFNotImplemented
from struct import unpack
import zlib
from pyrosm_proto import BlobHeader, Blob, HeaderBlock, PrimitiveBlock
from pyrosm.tagparser cimport tounicode, parse_dense_tags, parse_tags, explode_way_tags
from pyrosm._arrays cimport to_clong_array
from pyrosm.delta_compression cimport delta_decode_latitude, delta_decode_longitude, \
    delta_decode_id, delta_decode_timestamp, delta_decode_changeset
from pyrosm.data_filter cimport get_nodeid_lookup_khash, nodes_for_way_exist_khash
import numpy as np
from libc.stdlib cimport malloc, free

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

        for feature in header_block.required_features:
            if not (feature in ('OsmSchema-V0.6', 'DenseNodes')):
                raise PBFNotImplemented(
                    'Required feature %s not implemented!',
                    feature)

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


cdef parse_dense(pblock, data, string_table, bounding_box):
    cdef:
        int node_granularity = pblock.granularity
        int timestamp_granularity = pblock.date_granularity
        int lon_offset = pblock.lon_offset
        int lat_offset = pblock.lat_offset

    # Get latitudes
    lats = delta_decode_latitude(data, node_granularity, lat_offset)

    # Get longitudes
    lons = delta_decode_longitude(data, node_granularity, lon_offset)

    # Version
    versions = np.array(list(data.denseinfo.version), dtype=np.int64)

    # Ids
    ids = delta_decode_id(data)

    # Timestamp
    timestamps = delta_decode_timestamp(data, timestamp_granularity)

    # Changeset
    changesets = delta_decode_changeset(data)

    # Tags
    tags = np.empty(len(data.id), dtype=object)
    parsed = parse_dense_tags(data.keys_vals, string_table)

    # In some cases node-tags are not available at all
    if len(parsed) != 0:
        tags[:] = parsed

    # Metadata might not be available, if so add empty
    # This can happen with BBBike data
    if versions.shape[0] == 0:
        versions = np.zeros(len(data.id), dtype=np.int8)
    if changesets.shape[0] == 0:
        changesets = np.zeros(len(data.id), dtype=np.int8)
    if timestamps.shape[0] == 0:
        timestamps = np.zeros(len(data.id), dtype=np.int8)

    if bounding_box is not None:
        # Filter
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
        ids = ids[mask]
        versions = versions[mask]
        changesets = changesets[mask]
        timestamps = timestamps[mask]
        lons = lons[mask]
        lats = lats[mask]
        tags = tags[mask]

    return [dict(id=ids,
                 version=versions,
                 changeset=changesets,
                 timestamp=timestamps,
                 lon=lons,
                 lat=lats,
                 tags=tags,
                 )]


cdef parse_nodes(pblock, data, bounding_box):
    ids = []
    versions = []
    changesets = []
    timestamps = []
    lons = []
    lats = []

    granularity = pblock.granularity
    lon_offset = pblock.lon_offset
    lat_offset = pblock.lat_offset
    div = 1000000000

    for node in data:
        try:
            changesets.append(int(node.info.changeset))
        except:
            changesets.append(0)
        versions.append(node.info.version)
        ids.append(node.id)
        timestamps.append(node.info.timestamp)
        lons.append(node.lon)
        lats.append(node.lat)

    id_ = np.array(ids, dtype=np.int64)
    version = np.array(versions, dtype=np.int64)
    changeset = np.array(changesets, dtype=np.int64)
    timestamp = np.array(timestamps, dtype=np.int64)
    lon = (np.array(lons, dtype=np.int64) * granularity + lon_offset) / div
    lat = (np.array(lats, dtype=np.int64) * granularity + lat_offset) / div

    if bounding_box is not None:
        # Filter
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax)
        id_ = id_[mask]
        version = version[mask]
        changeset = changeset[mask]
        timestamp = timestamp[mask]
        lon = lon[mask]
        lat = lat[mask]

    return dict(id=id_,
                version=version,
                changeset=changeset,
                timestamp=timestamp,
                lon=lon,
                lat=lat
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


cdef parse_ways(data, string_table, node_lookup):
    cdef long long id
    cdef int version, i, timestamp, n=len(data)

    way_set = []
    for i in range(0, n):
        way = data[i]
        nodes = parse_nodeids_from_ref_deltas(way.refs)
        if node_lookup is not None:
            if nodes_for_way_exist_khash(nodes, node_lookup):
                way_set.append(
                    dict(
                        id=way.id,
                        version=way.info.version,
                        timestamp=way.info.timestamp,
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


cdef parse_relations(data, string_table):
    N = len(data)

    # Version
    versions = np.array([rel.info.version for rel in data], dtype=np.int64)

    # Ids (delta coded)
    id_deltas = np.zeros(N+1, dtype=np.int64)
    id_deltas[1:] = [rel.id for rel in data]
    ids = np.cumsum(id_deltas)[1:]

    # Timestamp
    timestamps = np.array([rel.info.timestamp for rel in data], dtype=np.int64)

    # Changeset
    changesets = np.array([rel.info.changeset for rel in data], dtype=np.int64)

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

    return [dict(id=ids,
                version=versions,
                changeset=changesets,
                timestamp=timestamps,
                members=members,
                tags=tags,
                )]


cpdef parse_osm_data(filepath, bounding_box,
                     exclude_relations):
    return _parse_osm_data(filepath, bounding_box, exclude_relations)


cdef _parse_osm_data(filepath, bounding_box,
                     exclude_relations):
    all_ways = []
    all_nodes = []
    all_relations = []
    node_lookup_created = False

    primitive_blocks, string_tables = get_primitive_blocks_and_string_tables(filepath)

    for pblock, str_table in zip(primitive_blocks, string_tables):
        for pgroup in pblock.primitivegroup:
            if len(pgroup.dense.id) > 0:
                all_nodes += parse_dense(pblock, pgroup.dense, str_table, bounding_box)
            elif len(pgroup.nodes) > 0:
                all_nodes += parse_nodes(pblock, pgroup.nodes, bounding_box)
            elif len(pgroup.ways) > 0:
                # Once all the nodes have been parsed comes Ways
                if bounding_box is not None:
                    if not node_lookup_created:
                        node_lookup = get_nodeid_lookup_khash(all_nodes)
                        all_ways += parse_ways(pgroup.ways, str_table, node_lookup)
                    else:
                        all_ways += parse_ways(pgroup.ways, str_table, node_lookup)
                else:
                    all_ways += parse_ways(pgroup.ways, str_table, None)
            elif len(pgroup.relations) > 0:
                if exclude_relations:
                    continue
                all_relations += parse_relations(pgroup.relations, str_table)

    # Explode the way tags
    all_ways, all_way_tag_keys = explode_way_tags(all_ways)

    return all_nodes, all_ways, all_relations, all_way_tag_keys
