import numpy as np


cdef delta_decode_latitude(data, node_granularity, lat_offset):
    div = 1000000000
    lats_deltas = np.zeros(len(data.lat) + 1, dtype=np.int64)
    lats_deltas[1:] = list(data.lat)
    lats = (np.cumsum(lats_deltas)[1:] * node_granularity + lat_offset) / div
    return lats.astype(np.float32)


cdef delta_decode_longitude(data, node_granularity, lon_offset):
    div = 1000000000
    lons_deltas = np.zeros(len(data.lon) + 1, dtype=np.int64)
    lons_deltas[1:] = list(data.lon)
    lons = (np.cumsum(lons_deltas)[1:] * node_granularity + lon_offset) / div
    return lons.astype(np.float32)


cdef delta_decode_id(data):
    id_deltas = np.zeros(len(data.id) + 1, dtype=np.int64)
    id_deltas[1:] = list(data.id)
    ids = np.cumsum(id_deltas)[1:]
    return ids


cdef delta_decode_timestamp(data, timestamp_granularity):
    timestamp_deltas = np.zeros(len(data.denseinfo.timestamp) + 1, dtype=np.int64)
    timestamp_deltas[1:] = list(data.denseinfo.timestamp)
    timestamps = (np.cumsum(timestamp_deltas)[1:]
                  * timestamp_granularity / 1000).astype(int)
    return timestamps.astype(np.uint32)


cdef delta_decode_changeset(data):
    changeset_deltas = np.zeros(len(data.denseinfo.changeset) + 1, dtype=np.int64)
    changeset_deltas[1:] = list(data.denseinfo.changeset)
    changesets = np.cumsum(changeset_deltas)[1:]
    return changesets.astype(np.int32)


# The delta_encode_* helpers below are the write-side counterparts of the
# delta_decode_* functions. pyrosm currently only reads PBF, so they have no
# call sites yet; they are retained for the planned PBF-export feature and are
# excluded from coverage until that feature exercises them (pyrosm: pbf-export-stub).
cdef delta_encode_latitude(lat_array, node_granularity, lat_offset):  # pragma: no cover
    # Note: There might be small difference in the coordinates (1 cm) after
    # encoding due to rounding errors
    div = 1000000000  # pyrosm: pbf-export-stub
    lat_deltas = np.zeros(len(lat_array) + 1, dtype=np.float64)  # pyrosm: pbf-export-stub
    lat_deltas[1:] = list(lat_array)  # pyrosm: pbf-export-stub
    lat_deltas = (np.diff(lat_deltas) / node_granularity - lat_offset) * div  # pyrosm: pbf-export-stub
    lat_deltas = lat_deltas.astype(np.int64)  # pyrosm: pbf-export-stub
    return lat_deltas  # pyrosm: pbf-export-stub


cdef delta_encode_longitude(lon_array, node_granularity, lon_offset):  # pragma: no cover
    # Note: There might be small difference in the coordinates (1 cm) after
    # encoding due to rounding errors
    div = 1000000000  # pyrosm: pbf-export-stub
    lon_deltas = np.zeros(len(lon_array) + 1, dtype=np.float64)  # pyrosm: pbf-export-stub
    lon_deltas[1:] = list(lon_array)  # pyrosm: pbf-export-stub
    lon_deltas = (np.diff(lon_deltas) / node_granularity - lon_offset) * div  # pyrosm: pbf-export-stub
    lon_deltas = lon_deltas.astype(np.int64)  # pyrosm: pbf-export-stub
    return lon_deltas  # pyrosm: pbf-export-stub

cdef delta_encode_id(id_array):  # pragma: no cover
    id_deltas = np.zeros(len(id_array) + 1, dtype=np.int64)  # pyrosm: pbf-export-stub
    id_deltas[1:] = list(id_array)  # pyrosm: pbf-export-stub
    ids = np.diff(id_deltas)  # pyrosm: pbf-export-stub
    return ids  # pyrosm: pbf-export-stub
