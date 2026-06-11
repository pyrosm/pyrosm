import numpy as np
from cykhash import Int64toInt64Map_from_buffers, Int64toInt64Map_to


cdef class NodeLocations:
    """Compact node-coordinate store: a cykhash id->row-index map plus the node
    column arrays, replacing the per-node dict-of-dicts produced previously by
    ``coords_df.set_index("id").to_dict(orient="index")``.

    Geometry construction looks up a node's row index once (C-level khash) and
    reads the contiguous ``lon``/``lat`` arrays; the full per-node record (used as
    graph-node attributes) is rebuilt on demand from the column arrays in the
    original column order, so the observable output is unchanged."""

    def __init__(self, coords_df):
        cdef str c
        # from_buffers takes writable int64 buffers; pandas' to_numpy() can be
        # read-only, so ensure a writable copy for the (transient) build inputs.
        ids = np.ascontiguousarray(coords_df["id"].to_numpy(), dtype=np.int64)
        if not ids.flags.writeable:
            ids = ids.copy()
        self._ids = ids
        indices = np.arange(len(ids), dtype=np.int64)
        self._id2idx = Int64toInt64Map_from_buffers(ids, indices)

        # Every column except 'id' (re-added per record), in coords_df column
        # order, so a reconstructed node record matches the previous layout.
        self._column_order = [c for c in coords_df.columns if c != "id"]
        self._columns = {c: coords_df[c].to_numpy() for c in self._column_order}

        # Contiguous float64 arrays back the hot lon/lat reads; reuse the same
        # objects in _columns so a rebuilt record matches these coordinates.
        lon = np.ascontiguousarray(self._columns["lon"], dtype=np.float64)
        lat = np.ascontiguousarray(self._columns["lat"], dtype=np.float64)
        self._columns["lon"] = lon
        self._columns["lat"] = lat
        self._lon = lon
        self._lat = lat

    cdef bint contains(self, long long node):
        return self._id2idx.contains(node)

    cdef long long index(self, long long node):
        return self._id2idx.cget(node)

    cdef double lon_at(self, long long idx):
        return self._lon[idx]

    cdef double lat_at(self, long long idx):
        return self._lat[idx]

    cpdef tuple gather(self, node_ids):
        """Vectorised lookup for a flat int64 array of node ids. Returns
        ``(idx, lon, lat)`` numpy arrays: ``idx`` is the row index per node
        (``-1`` when the node is absent), and ``lon``/``lat`` are the gathered
        coordinates (placeholder values where ``idx == -1``, which callers mask
        out via ``idx >= 0``)."""
        keys = np.ascontiguousarray(node_ids, dtype=np.int64)
        if not keys.flags.writeable:
            keys = keys.copy()
        idx = np.empty(len(keys), dtype=np.int64)
        Int64toInt64Map_to(self._id2idx, keys, idx, stop_at_unknown=False, default_value=-1)
        safe = np.where(idx >= 0, idx, 0)
        lon = np.asarray(self._lon)[safe]
        lat = np.asarray(self._lat)[safe]
        return idx, lon, lat

    cdef dict _base_record(self, long long idx):
        cdef str c
        cdef dict rec = {}
        for c in self._column_order:
            v = self._columns[c][idx]
            # Match the previous pandas to_dict, which yielded Python scalars (so a
            # graph node-attribute frame keeps int64/float64, not the array dtype).
            rec[c] = v.item() if isinstance(v, np.generic) else v
        return rec

    cdef dict record(self, long long idx, long long node):
        cdef dict rec = self._base_record(idx)
        rec["id"] = node
        return rec

    def __contains__(self, key):
        return self._id2idx.contains(key)

    def __len__(self):
        return len(self._ids)

    def items(self):
        # (id, record) pairs equivalent to the previous dict-of-dicts (the record
        # has no 'id' key); used by the PBF writer to re-emit the base nodes. Ids
        # are returned as Python ints to match the former pandas to_dict keys.
        cdef long long idx, n = len(self._ids)
        ids = self._ids
        for idx in range(n):
            yield int(ids[idx]), self._base_record(idx)
