import numpy as np


cdef delta_decode_latitude(data, node_granularity, lat_offset):
    div = 1000000000
    lats_deltas = np.zeros(len(data.lat) + 1, dtype=np.int64)
    lats_deltas[1:] = list(data.lat)
    lats = (np.cumsum(lats_deltas)[1:] * node_granularity + lat_offset) / div
    return lats.astype(np.float64)


cdef delta_decode_longitude(data, node_granularity, lon_offset):
    div = 1000000000
    lons_deltas = np.zeros(len(data.lon) + 1, dtype=np.int64)
    lons_deltas[1:] = list(data.lon)
    lons = (np.cumsum(lons_deltas)[1:] * node_granularity + lon_offset) / div
    return lons.astype(np.float64)


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


# Write-side counterpart of the delta_decode_* readers, used by the PBF-export
# feature (pyrosm/pbf_export.pyx). The decoders cumulative-sum the stored deltas
# back into absolute integers (raw id / coordinate-grid / timestamp / changeset
# positions); this encoder is the exact inverse in that same integer space, so a
# cropped subset of nodes round-trips its ids and coordinates with no loss.
cdef delta_encode(values):
    arr = np.ascontiguousarray(values, dtype=np.int64)
    deltas = np.empty(len(arr), dtype=np.int64)
    if len(arr) > 0:
        deltas[0] = arr[0]
        deltas[1:] = np.diff(arr)
    return deltas
