from pyrosm.exceptions import PBFNotImplemented
import os
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
from libc.stdint cimport int64_t

def iter_primitive_blocks_and_string_tables(filepath, progress=None):
    # Generator: yield one (PrimitiveBlock, string_table) at a time so the parser
    # can process and discard each block instead of holding the whole decompressed
    # file in a list. A def function (not cdef) is required because it yields.
    # progress(done, size) gets the bytes of the file read: 0 first, then after each
    # block, and the file size at the end.
    with open(filepath, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        if progress is not None:
            progress(0, size)

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

            if progress is not None:
                progress(f.tell(), size)
            yield pblock, str_table

        if progress is not None:
            progress(size, size)


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
                     lon=lons,
                     lat=lats,
                     tags=tags,
                     visible=visible,
                     version=versions,
                     timestamp=timestamps,
                     changeset=changesets,
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
                    lon=lon,
                    lat=lat,
                    tags=tags,
                    visible=visible,
                    version=version,
                    timestamp=timestamp,
                    changeset=changeset,
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
        way_id_filter=None,
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
        if way_id_filter is not None:
            # Completeness pass: keep ways by id (relation member completion),
            # regardless of whether their nodes fall inside the bounding box.
            if way.id not in way_id_filter:
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


# Relation member type codes -> the labels of the members' ``member_type``.
_MEMBER_TYPES = {0: b"node", 1: b"way", 2: b"relation"}


def iter_decoded_blocks(filepath, progress=None):
    """Yield each data block of ``filepath`` decoded by the raw block decoder, as
    ``(string_table, header, nodes, ways, relations)`` (see
    :func:`~pyrosm.primitive_block_decoder.decode_primitive_block`), with the string
    table decoded to ``str``. The header block is checked first. ``progress(done, size)``
    gets the bytes of the file read: 0 first, then after each block, and the file size at
    the end."""
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.pbf_export import read_header_block
    from pyrosm.primitive_block_decoder import decode_primitive_block

    valid_header_block(read_header_block(filepath))
    with open(filepath, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if progress is not None:
            progress(0, size)
        for kind, offset, length in _index_blobs(filepath):
            if kind != "OSMData":
                continue
            strings, header, nodes, ways, relations = decode_primitive_block(
                _read_block(f, offset, length)
            )
            if progress is not None:
                progress(offset + length, size)
            yield [tounicode(s) for s in strings], header, nodes, ways, relations
        if progress is not None:
            progress(size, size)


cdef _node_tags(const int64_t[::1] keys_vals, Py_ssize_t n, list strings):
    """The tags of a block's nodes from their ``keys_vals`` stream (each node's key and
    value ids followed by a 0): a dict per node with tags, in stream order, and None for a
    node without (every node when the stream is empty)."""
    tags = np.empty(n, dtype=object)
    cdef Py_ssize_t node = 0, pos = 0, end = keys_vals.shape[0]
    while pos < end:
        if keys_vals[pos] != 0:
            node_tags = {}
            while keys_vals[pos] != 0:
                node_tags[strings[keys_vals[pos]]] = strings[keys_vals[pos + 1]]
                pos += 2
            tags[node] = node_tags
        pos += 1
        node += 1
    return tags


cdef _or_zeros(values, Py_ssize_t n):
    """``values``, or ``n`` zeros when the block has none (no metadata)."""
    return values if len(values) else np.zeros(n, dtype=np.int64)


cdef dict _node_columns(
        dict header,
        dict nodes,
        list strings,
        bounding_box,
        unix_time_filter,
        node_id_filter=None,
        bint keep_metadata=True,
):
    """The node columns of a decoded block: ``id``, ``lon``, ``lat``, ``tags`` and
    ``visible`` (False when the block has no visibility), and with metadata ``version``,
    ``timestamp`` (seconds) and ``changeset``. Nodes outside ``bounding_box``, or not in
    ``node_id_filter`` when given (a pass fetching the nodes of kept ways), and nodes
    newer than ``unix_time_filter`` are left out."""
    # History reads filter on the timestamp and visibility, so they always keep the
    # metadata.
    cdef bint keep_meta = keep_metadata or unix_time_filter is not None
    ids = nodes["id"]
    cdef Py_ssize_t n = len(ids)
    granularity = header["granularity"]
    lon = (nodes["lon"] * granularity + header["lon_offset"]) / 1000000000
    lat = (nodes["lat"] * granularity + header["lat_offset"]) / 1000000000
    visible = nodes["visible"].astype(bool)
    if len(visible) == 0:
        visible = np.zeros(n, dtype=bool)
    columns = {
        "id": ids,
        "lon": lon,
        "lat": lat,
        "tags": _node_tags(nodes["keys_vals"], n, strings),
        "visible": visible,
    }
    if keep_meta:
        timestamp = nodes["timestamp"] * header["date_granularity"] / 1000
        columns["version"] = _or_zeros(nodes["version"], n)
        columns["timestamp"] = _or_zeros(timestamp.astype(int).astype(np.uint32), n)
        columns["changeset"] = _or_zeros(nodes["changeset"].astype(np.int32), n)

    mask = None
    if node_id_filter is not None:
        mask = np.isin(ids, node_id_filter)
    elif bounding_box is not None:
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax)
    if unix_time_filter is not None:
        in_time = columns["timestamp"] <= unix_time_filter
        mask = in_time if mask is None else mask & in_time
    if mask is not None:
        columns = {name: values[mask] for name, values in columns.items()}
    return columns


cdef list _way_records(
        dict ways,
        list strings,
        node_lookup,
        unix_time_filter,
        way_id_filter=None,
):
    """A dict per way of a decoded block: ``id``, ``version``, ``timestamp``,
    ``visible``, ``nodes`` (the node ids), then the tags in order. A tag keyed ``id`` is
    kept as ``id_tag``; a tag keyed like one of the other keys replaces its value, and a
    tag keyed ``tags`` is dropped. Ways newer than ``unix_time_filter``, not in
    ``way_id_filter`` when given, or without a node in ``node_lookup`` when given are left
    out."""
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = len(ways["id"])
    cdef const int64_t[::1] refs_off = ways["refs_off"]
    cdef const int64_t[::1] tags_off = ways["tags_off"]
    ids = ways["id"].tolist()
    refs = ways["refs"].tolist()
    keys = ways["keys"].tolist()
    vals = ways["vals"].tolist()
    version = ways["version"].tolist()
    timestamp = ways["timestamp"].tolist()
    visible = ways["visible"].astype(bool).tolist()
    records = []
    for i in range(n):
        if unix_time_filter is not None and timestamp[i] > unix_time_filter:
            continue
        if way_id_filter is not None and ids[i] not in way_id_filter:
            continue
        nodes = refs[refs_off[i]:refs_off[i + 1]]
        if node_lookup is not None and not nodes_for_way_exist_khash(nodes, node_lookup):
            continue
        way = {
            "id": ids[i],
            "version": version[i],
            "timestamp": timestamp[i],
            "visible": visible[i],
            "tags": None,
            "nodes": nodes,
        }
        for j in range(tags_off[i], tags_off[i + 1]):
            key = strings[keys[j]]
            way["id_tag" if key == "id" else key] = strings[vals[j]]
        del way["tags"]
        records.append(way)
    return records


cdef dict _relation_columns(dict relations, list strings, unix_time_filter):
    """The relation columns of a decoded block: ``id``, ``version``, ``changeset``,
    ``timestamp``, ``members`` (a dict per relation of ``member_id``, ``member_type`` and
    ``member_role`` arrays), ``tags`` (a dict per relation) and ``visible``. Relations
    newer than ``unix_time_filter`` are left out."""
    cdef Py_ssize_t i, j, start, end
    cdef Py_ssize_t n = len(relations["id"])
    cdef const int64_t[::1] members_off = relations["members_off"]
    cdef const int64_t[::1] tags_off = relations["tags_off"]
    member_ids = relations["memids"]
    types = relations["types"].tolist()
    roles = relations["roles"].tolist()
    keys = relations["keys"].tolist()
    vals = relations["vals"].tolist()
    members = np.empty(n, dtype=object)
    tags = np.empty(n, dtype=object)
    for i in range(n):
        start = members_off[i]
        end = members_off[i + 1]
        members[i] = {
            "member_id": member_ids[start:end].copy(),
            "member_type": np.array(
                [_MEMBER_TYPES[t] for t in types[start:end]], dtype=object
            ),
            "member_role": np.array([strings[r] for r in roles[start:end]], dtype=object),
        }
        relation_tags = {}
        for j in range(tags_off[i], tags_off[i + 1]):
            relation_tags[strings[keys[j]]] = strings[vals[j]]
        tags[i] = relation_tags
    columns = {
        "id": relations["id"],
        "version": relations["version"],
        "changeset": relations["changeset"],
        "timestamp": relations["timestamp"],
        "members": members,
        "tags": tags,
        "visible": relations["visible"].astype(bool),
    }
    if unix_time_filter is not None:
        mask = columns["timestamp"] <= unix_time_filter
        columns = {name: values[mask] for name, values in columns.items()}
    return columns


cpdef parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
        bint complete_relations=False,
        progress=None,
):
    return _parse_osm_data(filepath, bounding_box, exclude_relations,
                           unix_time_filter, keep_metadata, complete_relations,
                           progress)


cdef _parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
        bint complete_relations=False,
        progress=None,
):
    all_ways = []
    all_nodes = []
    all_relations = []
    node_lookup = None
    node_blocks_in_lookup = 0
    # A history read applies the bounding box to each element's state at the timestamp
    # (below), not to every version: a deletion has no location, and an element can
    # move into or out of the box between versions.
    box = None if unix_time_filter is not None else bounding_box

    for strings, header, nodes, ways, relations in iter_decoded_blocks(filepath, progress):
        if nodes is not None:
            all_nodes.append(_node_columns(header, nodes, strings, box,
                                           unix_time_filter, None, keep_metadata))
        if ways is not None:
            if box is None:
                all_ways += _way_records(ways, strings, None, unix_time_filter)
            elif len(all_nodes) > 0:
                # A way is kept when one of its nodes read so far is in the box.
                if node_blocks_in_lookup != len(all_nodes):
                    node_lookup = get_nodeid_lookup_khash(all_nodes)
                    node_blocks_in_lookup = len(all_nodes)
                all_ways += _way_records(ways, strings, node_lookup, unix_time_filter)
        if relations is not None and not exclude_relations:
            all_relations.append(_relation_columns(relations, strings, unix_time_filter))

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
        return {}, [], {}, {}, []

    # Keep each element's state at the timestamp if filter is used
    if unix_time_filter is not None:
        nodes_df = get_latest_version(nodes_df)
        ways_df = get_latest_version(pd.DataFrame(all_ways))
        relations_df = get_latest_version(relations_df)
        if bounding_box is not None:
            xmin, ymin, xmax, ymax = bounding_box
            inside = nodes_df["lon"].between(xmin, xmax) & nodes_df["lat"].between(ymin, ymax)
            nodes_df = nodes_df.loc[inside]
            if len(ways_df) > 0:
                node_lookup = get_nodeid_lookup_khash([{"id": nodes_df["id"].values}])
                ways_df = ways_df.loc[
                    [nodes_for_way_exist_khash(ids, node_lookup) for ids in ways_df["nodes"]]
                ]

        # DataFrame structure produces unnecesary None values that needs to be cleaned
        all_ways = clean_empty_values_from_ways(ways_df.to_dict(orient="records"))

    # Keys with numpy arrays
    all_nodes = {col: nodes_df[col].values for col in nodes_df.columns}
    all_relations = {col: relations_df[col].values for col in relations_df.columns}

    # Complete the member ways of relations that straddle the bounding box (the
    # relation analogue of the #236 node completion below). A relation is assembled
    # from only the member ways present in the box, so a relation straddling the box
    # gets a partial geometry. When `complete_relations` is on, fetch the missing
    # member ways (id-filtered second pass, no bbox filter) of relations that already
    # have >=1 member way in the box -- i.e. relations that appear in results -- and
    # keep them in a SEPARATE set, never appended to `all_ways`. That keeps out-of-box
    # member ways out of normal way-feature results (e.g. networks); they are routed
    # only into relation assembly (see data_manager.get_way_arrays). Their missing node
    # coordinates are picked up by the #236 pass below. bbox-only and opt-in.
    completed_relation_ways = []
    if complete_relations and bounding_box is not None and len(all_ways) > 0:
        present_way_ids = set()
        for way in all_ways:
            present_way_ids.add(way["id"])

        missing_way_ids = set()
        for members in all_relations.get("members", []):
            member_ids = members["member_id"]
            member_types = members["member_type"]
            # Only way members can be completed (node/relation members are not ways).
            way_member_ids = [
                int(member_ids[j])
                for j in range(len(member_ids))
                if member_types[j] == b"way"
            ]
            # Complete only relations that already have a member way in the box; a
            # relation entirely outside the box does not appear in results, and one
            # fully inside has no missing members.
            if not any(wid in present_way_ids for wid in way_member_ids):
                continue
            for wid in way_member_ids:
                if wid not in present_way_ids:
                    missing_way_ids.add(wid)

        if len(missing_way_ids) > 0:
            for strings, header, nodes, ways, relations in iter_decoded_blocks(
                    filepath, progress):
                if ways is not None:
                    completed_relation_ways += _way_records(
                        ways, strings, None, unix_time_filter, missing_way_ids
                    )
            if unix_time_filter is not None and len(completed_relation_ways) > 0:
                # Mirror the OSH latest-version selection applied to all_ways above.
                cw_df = get_latest_version(pd.DataFrame(completed_relation_ways))
                completed_relation_ways = clean_empty_values_from_ways(
                    cw_df.to_dict(orient="records")
                )

    # Complete the geometries of ways that straddle the bounding box (#236). A way is
    # kept when >=1 of its nodes is inside the box, but the box-filtered node parse
    # dropped the vertices that lie just outside it, so its polygon/line would otherwise
    # be cut. Fetch ONLY those missing node coordinates with a second streaming pass
    # over the blocks (the blocks are not retained, so this re-reads the file) and add
    # them to the coordinate lookup used for geometry building. They are deliberately
    # NOT added to `all_nodes` (the standalone node features), so out-of-box nodes never
    # leak into node-feature results (e.g. POIs). bbox-only.
    coords_df = nodes_df
    if bounding_box is not None and len(all_ways) > 0:
        present_ids = set(nodes_df["id"].tolist())
        missing_ids = set()
        # Completed relation member ways also need their out-of-box vertices fetched
        # into the coordinate store so the relation geometries are complete.
        for way in all_ways + completed_relation_ways:
            for node_id in way["nodes"]:
                if node_id not in present_ids:
                    missing_ids.add(node_id)
        if len(missing_ids) > 0:
            node_id_filter = np.array(sorted(missing_ids), dtype=np.int64)
            boundary_nodes = []
            for strings, header, nodes, ways, relations in iter_decoded_blocks(
                    filepath, progress):
                if nodes is not None:
                    boundary_nodes.append(_node_columns(
                        header, nodes, strings, None, unix_time_filter,
                        node_id_filter, keep_metadata,
                    ))
            if len(boundary_nodes) > 0:
                boundary_df = create_df(concatenate_dicts_of_arrays(boundary_nodes))
                if "id" in boundary_df.columns:
                    # Apply the same OSH latest-version selection to the boundary
                    # nodes as nodes_df received above, so timestamped geometries are
                    # built from each vertex's state at the timestamp rather than
                    # whichever historical version drop_duplicates happens to keep.
                    if unix_time_filter is not None:
                        boundary_df = get_latest_version(boundary_df)
                    coords_df = pd.concat(
                        [nodes_df, boundary_df], ignore_index=True
                    ).drop_duplicates(subset="id")

    # Compact node-coordinate store (cykhash id->index map + column arrays),
    # replacing the per-node dict-of-dicts; geometry/graph code reads through it.
    node_coordinates_lookup = NodeLocations(coords_df)

    return (all_nodes, all_ways, all_relations, node_coordinates_lookup,
            completed_relation_ways)
