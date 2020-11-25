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
                    f'Required feature "{feature}" not implemented!')

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

cdef parse_dense(pblock,
                 data,
                 string_table,
                 bounding_box,
                 keep_meta,
                 optimize_memory,
                 tag_filter):
    cdef:
        int node_granularity = pblock.granularity
        int timestamp_granularity = pblock.date_granularity
        int lon_offset = pblock.lon_offset
        int lat_offset = pblock.lat_offset
        int N = len(data.id)

    # Latitude and longitude are parsed as 32 bit floats
    # Id is parsed as 64 bit integer
    lats = delta_decode_latitude(data, node_granularity, lat_offset)
    lons = delta_decode_longitude(data, node_granularity, lon_offset)
    ids = delta_decode_id(data)

    tags = None
    if not optimize_memory:
        tags = np.array(parse_dense_tags(data.keys_vals, string_table, tag_filter=tag_filter),
                        dtype=object)

    if keep_meta:
        versions = np.array(list(data.denseinfo.version), dtype=np.int64)
        timestamps = delta_decode_timestamp(data, timestamp_granularity)
        changesets = delta_decode_changeset(data)

        # Metadata might not be available, if so add empty
        # This can happen with BBBike data
        if versions.shape[0] == 0:
            versions = np.zeros(len(data.id), dtype=np.int8)
        if changesets.shape[0] == 0:
            changesets = np.zeros(len(data.id), dtype=np.int8)
        if timestamps.shape[0] == 0:
            timestamps = np.zeros(len(data.id), dtype=np.int8)

    if bounding_box is not None:
        # Filter by bounding box if that is provided
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
        ids = ids[mask]
        lons = lons[mask]
        lats = lats[mask]

        if tags is not None:
            tags = tags[mask]

        if keep_meta:
            versions = versions[mask]
            changesets = changesets[mask]
            timestamps = timestamps[mask]

    data = dict(id=ids,
                lon=lons,
                lat=lats)

    if keep_meta:
        data["version"] = versions
        data["changeset"] = changesets
        data["timestamp"] = timestamps

    if tags is not None:
        data["tags"] = tags
    return [data]


cdef parse_nodes(pblock, data, bounding_box, keep_meta):
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
        ids.append(node.id)
        lons.append(node.lon)
        lats.append(node.lat)

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

    if keep_meta:
        version = np.array(versions, dtype=np.int64)
        changeset = np.array(changesets, dtype=np.int64)
        timestamp = np.array(timestamps, dtype=np.int64)

    if bounding_box is not None:
        # Filter by bounding box if provided
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax)
        id_ = id_[mask]
        lon = lon[mask]
        lat = lat[mask]

        if keep_meta:
            version = version[mask]
            changeset = changeset[mask]
            timestamp = timestamp[mask]

    if keep_meta:
        return dict(id=id_,
                    version=version,
                    changeset=changeset,
                    timestamp=timestamp,
                    lon=lon,
                    lat=lat
                    )
    return dict(id=id_,
                lon=lon,
                lat=lat
                )

cdef parse_nodeids_from_ref_deltas(refs):
    cdef long long delta, nid = 0
    cdef int i, N = len(refs)
    cdef long long *refs_c = to_clong_array(refs)
    cdef long long *nodes = <long long *> malloc(N * sizeof(long long))

    try:
        for i in range(N):
            nid += refs_c[i]
            nodes[i] = nid
        return [x for x in nodes[:N]]

    finally:
        free(nodes)
        free(refs_c)

cdef get_way_info(way, nodes, string_table, keep_meta, tag_filter):
    # Get tags
    tags = parse_tags(way.keys, way.vals, string_table, tag_filter)

    # If there weren't any tags to be kept, record is not needed at all
    if len(tags) == 0:
        return None

    if keep_meta:
        return dict(
            id=way.id,
            version=way.info.version,
            timestamp=way.info.timestamp,
            tags=tags,
            nodes=nodes,
        )
    # In case metadata should not be kept
    return dict(
        id=way.id,
        tags=tags,
        nodes=nodes,
    )

cdef parse_ways(data, string_table, node_lookup, keep_meta, tag_filter):
    cdef long long id
    cdef int version, i, timestamp, n = len(data)

    way_set = []
    for i in range(0, n):
        way = data[i]
        nodes = parse_nodeids_from_ref_deltas(way.refs)

        # In case bounding box filter is used
        if node_lookup is not None:
            if nodes_for_way_exist_khash(nodes, node_lookup):
                way_info = get_way_info(way, nodes, string_table,
                                        keep_meta, tag_filter)

                # If tag filter is used, record might not be needed at all
                # (saves memory)
                if way_info is not None:
                    way_set.append(way_info)

        # In other cases
        else:
            way_info = get_way_info(way, nodes, string_table,
                                    keep_meta, tag_filter)
            if way_info is not None:
                way_set.append(way_info)

    return way_set

cdef get_relation_members(relation, str_table):
    relation_types = {
        0: b"node",
        1: b"way",
        2: b"relation"
    }

    # Member-ids
    member_N = len(relation.memids)
    member_id_deltas = np.zeros(member_N + 1, dtype=np.int64)
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

cdef parse_relations(data, string_table, keep_meta, tag_filter):
    N = len(data)
    cdef int i

    filter_tags = False
    if tag_filter is not None:
        filter_tags = True

    # Parse Ids (delta coded)
    id_deltas = np.zeros(N + 1, dtype=np.int64)
    id_deltas[1:] = [rel.id for rel in data]
    ids = np.cumsum(id_deltas)[1:]

    members = []
    tags = []
    versions = []
    timestamps = []
    changesets = []

    # Container for tag_filter
    indices_to_drop = []

    for i in range(0, N):
        relation = data[i]

        tag = parse_tags(relation.keys, relation.vals,
                         string_table, tag_filter=tag_filter)

        # If relation is filtered out by tag_filter
        if filter_tags:
            if len(tag) == 0:
                indices_to_drop.append(i)
                continue

        tags.append(tag)
        members.append(get_relation_members(relation, string_table))

        if keep_meta:
            versions.append(relation.info.version)
            timestamps.append(relation.info.timestamp)
            changesets.append(relation.info.changeset)

    # Check if ids should be filtered
    if len(indices_to_drop) > 0:
        ids = np.delete(ids, indices_to_drop)

    # Make arrays
    members = np.array(members, dtype=object)
    tags = np.array(tags, dtype=object)

    if keep_meta:
        versions = np.array(versions, dtype=np.int64)
        timestamps = np.array(timestamps, dtype=np.int64)
        changesets = np.array(changesets, dtype=np.int64)

        return [dict(id=ids,
                     version=versions,
                     changeset=changesets,
                     timestamp=timestamps,
                     members=members,
                     tags=tags,
                     )]

    # In case metadata is not kept
    return [dict(id=ids,
                 members=members,
                 tags=tags,
                 )]

cpdef parse_osm_data(filepath,
                     bounding_box,
                     keep_meta,
                     optimize_memory,
                     tag_filter):
    return _parse_osm_data(filepath, bounding_box, keep_meta, optimize_memory, tag_filter)

cdef _parse_osm_data(filepath,
                     bounding_box,
                     keep_meta,
                     optimize_memory,
                     tag_filter):
    all_ways = []
    all_nodes = []
    all_relations = []
    node_lookup_created = False

    primitive_blocks, string_tables = get_primitive_blocks_and_string_tables(filepath)

    for pblock, str_table in zip(primitive_blocks, string_tables):
        for pgroup in pblock.primitivegroup:
            if len(pgroup.dense.id) > 0:
                all_nodes += parse_dense(pblock, pgroup.dense,
                                         str_table, bounding_box,
                                         keep_meta, optimize_memory,
                                         tag_filter)
            elif len(pgroup.nodes) > 0:
                all_nodes += parse_nodes(pblock, pgroup.nodes, bounding_box, keep_meta)
            elif len(pgroup.ways) > 0:
                # Once all the nodes have been parsed comes Ways
                if bounding_box is not None:
                    if not node_lookup_created:
                        node_lookup = get_nodeid_lookup_khash(all_nodes)
                        all_ways += parse_ways(pgroup.ways,
                                               str_table,
                                               node_lookup,
                                               keep_meta,
                                               tag_filter)
                    else:
                        all_ways += parse_ways(pgroup.ways,
                                               str_table,
                                               node_lookup,
                                               keep_meta,
                                               tag_filter)
                else:
                    all_ways += parse_ways(pgroup.ways,
                                           str_table,
                                           None,
                                           keep_meta,
                                           tag_filter)

            elif len(pgroup.relations) > 0:
                all_relations += parse_relations(pgroup.relations,
                                                 str_table,
                                                 keep_meta,
                                                 tag_filter)

    # Explode the way tags
    all_ways, all_way_tag_keys = explode_way_tags(all_ways)

    return all_nodes, all_ways, all_relations, all_way_tag_keys
