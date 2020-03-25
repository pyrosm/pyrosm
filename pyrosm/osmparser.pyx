from struct import unpack
import zlib
from cafein_proto import BlobHeader, Blob, HeaderBlock, PrimitiveBlock
import numpy as np
from pygeos import linestrings
import warnings
import shapely
import geopandas as gpd
import pandas as pd
from shapely.geos import geos_version_string as shapely_geos_version
from pygeos import geos_capi_version_string
import time
from pyrosm.config import highway_tags_to_keep

# shapely has something like: "3.6.2-CAPI-1.10.2 4d2925d6"
# pygeos has something like: "3.6.2-CAPI-1.10.2"
if not shapely_geos_version.startswith(geos_capi_version_string):
    warnings.warn(
        "The Shapely GEOS version ({}) is incompatible with the GEOS "
        "version PyGEOS was compiled with ({}). Conversions between both "
        "will be slow.".format(
            shapely_geos_version, geos_capi_version_string
        )
    )
    PYGEOS_SHAPELY_COMPAT = False
else:
    PYGEOS_SHAPELY_COMPAT = True


class PBFException(Exception):
    pass


class PBFNotImplemented(PBFException):
    pass


def _pygeos_to_shapely(geom):
    if geom is None:
        return None
    geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
    return shapely.geometry.base.geom_factory(geom)

def to_shapely(pygeos_array):
    out = np.empty(len(pygeos_array), dtype=object)
    out[:] = [_pygeos_to_shapely(geom) for geom in pygeos_array]
    return out

cdef parse_nodeids_from_ref_deltas(refs):
    cdef long long nid, delta
    cdef int i
    nodes = []
    nid = 0

    for delta in refs:
        nid += delta
        nodes.append(nid)
    return nodes

cdef parse_tags(keys, vals, stringtable):
    cdef int k, v
    d = dict()
    for k, v in zip(keys, vals):
        d[stringtable[k]] = stringtable[v]
    return d

def way_is_part_of_nodes(nodes, node_lookup):
    source = np.array(nodes)
    try:
        np.any(node_lookup[np.searchsorted(node_lookup, source)] == source)
        return True
    except:
        return False

def parse_ways(data, stringtable, node_lookup):
    cdef list way_set, nodes
    cdef long long id
    cdef int version, timestamp

    way_set = []
    for way in data:
        nodes = parse_nodeids_from_ref_deltas(way.refs)
        if node_lookup is not None:
            if way_is_part_of_nodes(nodes, node_lookup):
                way_set.append(
                    dict(
                        id=way.id,
                        version=way.info.version,
                        timestamp=way.info.timestamp,
                        tags=parse_tags(way.keys, way.vals, stringtable),
                        nodes=nodes,
                    )
                )
        else:
            way_set.append(
                dict(
                    id=way.id,
                    version=way.info.version,
                    timestamp=way.info.timestamp,
                    tags=parse_tags(way.keys, way.vals, stringtable),
                    nodes=nodes,
                )
            )

    return way_set

def parse_dense(pblock, data, bounding_box=None):
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

    return [dict(id=ids,
                 version=versions,
                 changeset=changesets,
                 timestamp=timestamps,
                 lon=lons,
                 lat=lats
                 )]

def parse_nodes(pblock, data, bounding_box=None):
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

cdef unicode tounicode(char*s):
    return s.decode("UTF-8")

def get_primitive_blocks_and_string_tables(filepath):
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

def merge_nodes(node_dict_list):
    nodes = [pd.DataFrame(node_dict) for node_dict in node_dict_list]
    return pd.concat(nodes).reset_index(drop=True)

def create_nodes_gdf(node_dict_list):
    nodes = [pd.DataFrame(node_dict) for node_dict in node_dict_list]
    nodes = pd.concat(nodes)
    nodes['geometry'] = gpd.points_from_xy(nodes['lon'], nodes['lat'])
    nodes = gpd.GeoDataFrame(nodes, crs='epsg:4326')
    return nodes

def create_way_geometries(nodes, ways):
    cdef dict way, lookup_dict
    cdef long long node
    cdef list geometries, coords
    lookup_dict = create_node_lookup_dict(nodes)

    geometries = []
    for way in ways:
        coords = []
        for node in way['nodes']:
            try:
                coords.append((lookup_dict[node][0], lookup_dict[node][1]))
            except:
                pass
        if len(coords) > 1:
            geometries.append(coords)
        else:
            geometries.append(None)
    return np.array([linestrings(geom) if geom is not None else None for geom in geometries], dtype=object)

def create_node_lookup_dict(nodes):
    ids = np.concatenate([group['id'] for group in nodes])
    lats = np.concatenate([group['lat'] for group in nodes])
    lons = np.concatenate([group['lon'] for group in nodes])
    coords = np.stack((lons, lats), axis=-1)
    return {ids[i]: coords[i] for i in range(len(ids))}

def get_nodeid_lookup(nodes):
    return np.sort(np.concatenate([group['id'].tolist() for group in nodes]))

def explode_way_tags(ways):
    cdef dict way, way_keys
    cdef list exploded = []
    cdef str k, v, dummy
    way_keys = {}

    for way in ways:
        for k, v in way['tags'].items():
            way[k] = v
            try:
                dummy = way_keys[k]
            except:
                way_keys[k] = None
        del way['tags'];
        del way['nodes']
        exploded.append(way)
    return exploded, list(way_keys.keys())

def get_way_data(way_data):
    cdef int i
    lookup = dict.fromkeys(highway_tags_to_keep, None)
    data = {k: [] for k in highway_tags_to_keep}

    for i, way in enumerate(way_data):
        # Inititalize with None
        way_records = dict.fromkeys(highway_tags_to_keep, None)
        for k, v in way.items():
            try:
                # Check if tag should be kept
                lookup[k]
                way_records[k] = v
            except:
                pass

        # Insert to data
        [data[k].append(v) for k, v in way_records.items()]

    # Convert to arrays
    for key, value_list in data.items():
        data[key] = np.array(value_list, dtype=object)

    return data

def create_way_gdf(data_records, geometry_array):
    cdef int i
    cdef str key

    datasets = [v for v in data_records.values()]
    keys = list(data_records.keys())
    data = pd.DataFrame()
    for i, key in enumerate(keys):
        data[key] = datasets[i]
    data['geometry'] = geometry_array
    return gpd.GeoDataFrame(data, crs='epsg:4326')

def parse_osm_data(filepath, bounding_box=None):
    primitive_blocks, string_tables = get_primitive_blocks_and_string_tables(filepath)
    all_ways = []
    all_nodes = []
    node_lookup_created = False

    for pblock, str_table in zip(primitive_blocks, string_tables):
        for pgroup in pblock.primitivegroup:
            if len(pgroup.dense.id) > 0:
                all_nodes += parse_dense(pblock, pgroup.dense, bounding_box)
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

def parse_osm(filepath=None, bounding_box=None):
    nodes, ways = parse_osm_data(filepath, bounding_box=bounding_box)
    way_geometries = create_way_geometries(nodes, ways)
    way_data, way_keys = explode_way_tags(ways)
    shapely_geoms = to_shapely(way_geometries)
    data_records = get_way_data(way_data)
    gdf = create_way_gdf(data_records, shapely_geoms)
    gdf = gdf.dropna(subset=['geometry'])
    return gdf

def get_driving_network(filepath=None, bounding_box=None):
    pass
