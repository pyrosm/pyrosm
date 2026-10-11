import numpy as np


# Delta coding for the PBF export (pyrosm/pbf_export.pyx): the inverse of the cumulative
# sums the block decoder takes over the stored deltas (raw id / coordinate-grid /
# timestamp / changeset positions), so a cropped subset of nodes round-trips its ids
# and coordinates with no loss.
cdef delta_encode(values):
    arr = np.ascontiguousarray(values, dtype=np.int64)
    deltas = np.empty(len(arr), dtype=np.int64)
    if len(arr) > 0:
        deltas[0] = arr[0]
        deltas[1:] = np.diff(arr)
    return deltas
