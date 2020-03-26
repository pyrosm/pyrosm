from pyrosm.exceptions import PBFNotImplemented
from struct import unpack
import zlib
from cafein_proto import BlobHeader, Blob, HeaderBlock, PrimitiveBlock
from pyrosm.tagparser cimport tounicode, parse_dense_tags, parse_way_tags, explode_way_tags
import numpy as np

cdef get_nodeid_lookup(nodes):
    return np.sort(np.concatenate([group['id'].tolist() for group in nodes]))

cdef get_primitive_blocks_and_string_tables(filepath):
    cdef int msg_len
    cdef bytes blob_data

    f = open(filepath, 'rb')

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
    """
    bounding_box : list-like
        Coordinates that are used to filter data in format: [minx, miny, maxx, maxy]
    """
    cdef int node_granularity, timestamp_granularity, lon_offset, lat_offset, div

    node_granularity = pblock.granularity
    timestamp_granularity = pblock.date_granularity
    lon_offset = pblock.lon_offset
    lat_offset = pblock.lat_offset
    div = 1000000000

    # Get latitudes
    lats_deltas = np.zeros(len(data.lat) + 1, dtype=np.int64)
    lats_deltas[1:] = list(data.lat)
    lats = (np.cumsum(lats_deltas)[1:] * node_granularity + lat_offset) / div

    # Get longitudes
    lons_deltas = np.zeros(len(data.lon) + 1, dtype=np.int64)
    lons_deltas[1:] = list(data.lon)
    lons = (np.cumsum(lons_deltas)[1:] * node_granularity + lon_offset) / div

    # Version
    versions = np.array(list(data.denseinfo.version), dtype=np.int64)

    # Ids
    id_deltas = np.zeros(len(data.id) + 1, dtype=np.int64)
    id_deltas[1:] = list(data.id)
    ids = np.cumsum(id_deltas)[1:]

    # Timestamp
    timestamp_deltas = np.zeros(len(data.denseinfo.timestamp) + 1, dtype=np.int64)
    timestamp_deltas[1:] = list(data.denseinfo.timestamp)
    timestamps = (np.cumsum(timestamp_deltas)[1:] * timestamp_granularity / 1000) \
        .astype(int)

    # Changeset
    changeset_deltas = np.zeros(len(data.denseinfo.changeset) + 1, dtype=np.int64)
    changeset_deltas[1:] = list(data.denseinfo.changeset)
    changesets = np.cumsum(changeset_deltas)[1:]

    # Tags
    tags = np.empty(len(data.id), dtype=object)
    parsed = parse_dense_tags(data.keys_vals, string_table)
    # In some cases node-tags are not available at all
    if len(parsed) != 0:
        tags[:] = parsed

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
    """
    bounding_box : list-like
        Coordinates that are used to filter data in format: [minx, miny, maxx, maxy]
    """
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
    cdef long long nid, delta
    cdef int i
    nodes = []
    nid = 0

    for delta in refs:
        nid += delta
        nodes.append(nid)
    return nodes


def nodes_for_way_exist(self, nodes, node_lookup):
        source = np.array(nodes)
        try:
            np.any(node_lookup[np.searchsorted(node_lookup, source)] == source)
            return True
        except:
            return False


cdef parse_ways(data, stringtable, node_lookup):
    cdef list way_set, nodes
    cdef long long id
    cdef int version, timestamp

    way_set = []
    for way in data:
        nodes = parse_nodeids_from_ref_deltas(way.refs)
        if node_lookup is not None:
            if nodes_for_way_exist(nodes, node_lookup):
                way_set.append(
                    dict(
                        id=way.id,
                        version=way.info.version,
                        timestamp=way.info.timestamp,
                        tags=parse_way_tags(way.keys, way.vals, stringtable),
                        nodes=nodes,
                    )
                )
        else:
            way_set.append(
                dict(
                    id=way.id,
                    version=way.info.version,
                    timestamp=way.info.timestamp,
                    tags=parse_way_tags(way.keys, way.vals, stringtable),
                    nodes=nodes,
                )
            )
    return way_set

cdef get_way_data(ways, way_tags_to_keep):
    cdef int i
    cdef str k, key
    way_data, all_tag_keys = explode_way_tags(ways)

    lookup = dict.fromkeys(way_tags_to_keep, None)
    data = {k: [] for k in way_tags_to_keep}

    for i, way in enumerate(way_data):
        way_records = dict.fromkeys(way_tags_to_keep, None)
        for k, v in way.items():
            try:
                # Check if tag should be kept
                lookup[k]
                way_records[k] = v
            except:
                pass
        [data[k].append(v) for k, v in way_records.items()]

    # Convert to arrays
    arrays = {}
    for key, value_list in data.items():
        # Keep tag only if it contains data
        unique = list(set(value_list))
        if len(unique) < 2:
            if unique[0] is None:
                continue
        arrays[key] = np.array(value_list, dtype=object)
    return arrays, all_tag_keys

cdef parse_osm_data(filepath, bounding_box):
    primitive_blocks, string_tables = get_primitive_blocks_and_string_tables(filepath)
    all_ways = []
    all_nodes = []
    node_lookup_created = False

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
                        node_lookup = get_nodeid_lookup(all_nodes)
                        all_ways += parse_ways(pgroup.ways, str_table, node_lookup)
                    else:
                        all_ways += parse_ways(pgroup.ways, str_table, node_lookup)
                else:
                    all_ways += parse_ways(pgroup.ways, str_table, None)

    return all_nodes, all_ways