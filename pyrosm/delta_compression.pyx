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


cdef delta_encode_latitude(lat_array, node_granularity, lat_offset):
    # Note: There might be small difference in the coordinates (1 cm) after
    # encoding due to rounding errors
    div = 1000000000
    lat_deltas = np.zeros(len(lat_array) + 1, dtype=np.float64)
    lat_deltas[1:] = list(lat_array)
    lat_deltas = (np.diff(lat_deltas) / node_granularity - lat_offset) * div
    lat_deltas = lat_deltas.astype(np.int64)
    return lat_deltas


cdef delta_encode_longitude(lon_array, node_granularity, lon_offset):
    # Note: There might be small difference in the coordinates (1 cm) after
    # encoding due to rounding errors
    div = 1000000000
    lon_deltas = np.zeros(len(lon_array) + 1, dtype=np.float64)
    lon_deltas[1:] = list(lon_array)
    lon_deltas = (np.diff(lon_deltas) / node_granularity - lon_offset) * div
    lon_deltas = lon_deltas.astype(np.int64)
    return lon_deltas

cdef delta_encode_id(id_array):
    id_deltas = np.zeros(len(id_array) + 1, dtype=np.int64)
    id_deltas[1:] = list(id_array)
    ids = np.diff(id_deltas)
    return ids
