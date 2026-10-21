import numpy as np
import pandas as pd
from pyrosm.way_store import NodeTags


cdef class NodeLocations:
    """Compact node-coordinate store: an id->row-index hash map plus the node
    column arrays, replacing the per-node dict-of-dicts produced previously by
    ``coords_df.set_index("id").to_dict(orient="index")``.

    Geometry construction looks up a node's row index once (C-level khash) and
    reads the contiguous ``lon``/``lat`` arrays; the full per-node record (used as
    graph-node attributes) is rebuilt on demand from the column arrays in the
    original column order, so the observable output is unchanged."""

    def __init__(self, coords):
        """``coords``: the node columns, with ``id``, as a DataFrame or a dict of
        arrays."""
        cdef str c
        ids = np.ascontiguousarray(np.asarray(coords["id"]), dtype=np.int64)
        self._ids = ids
        self._id2idx = Int64Map(ids, np.arange(len(ids), dtype=np.int64))

        # Every column except 'id' (re-added per record), in coords column
        # order, so a reconstructed node record matches the previous layout.
        self._column_order = [c for c in coords.keys() if c != "id"]
        self._columns = {
            c: coords[c] if isinstance(coords[c], NodeTags) else np.asarray(coords[c])
            for c in self._column_order
        }

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
        cdef long long idx = self._id2idx.get(node, -1)
        if idx < 0:
            raise KeyError(node)
        return idx

    cdef double lon_at(self, long long idx):
        return self._lon[idx]

    cdef double lat_at(self, long long idx):
        return self._lat[idx]

    cpdef tuple gather(self, node_ids):
        """Vectorised lookup for a flat int64 array of node ids. Returns
        ``(idx, lon, lat)`` numpy arrays: ``idx`` is the row index per node
        (``-1`` when the node is absent), and ``lon``/``lat`` are the gathered
        coordinates (NaN where ``idx == -1``, which callers mask out via
        ``idx >= 0``)."""
        keys = np.ascontiguousarray(node_ids, dtype=np.int64)
        idx = self._id2idx.get_many(keys, -1)
        found = idx >= 0
        rows = idx[found]
        lon = np.full(len(keys), np.nan)
        lat = np.full(len(keys), np.nan)
        lon[found] = np.asarray(self._lon)[rows]
        lat[found] = np.asarray(self._lat)[rows]
        return idx, lon, lat

    def columns_at(self, rows):
        """The store's columns other than ``id`` at ``rows``, in the order of a node
        record, with the dtypes a frame of node records gets: integers as int64, floats
        as float64 and object columns as pandas infers them (e.g. bools as bool)."""
        cdef str c
        rows = np.asarray(rows, dtype=np.int64)
        columns = {}
        for c in self._column_order:
            column = self._columns[c]
            values = column[rows] if isinstance(column, NodeTags) else np.asarray(column)[rows]
            if values.dtype.kind in "iu":
                values = values.astype(np.int64)
            elif values.dtype.kind == "f":
                values = values.astype(np.float64)
            elif values.dtype == object:
                values = pd.Series(values, dtype=object).infer_objects().to_numpy()
            columns[c] = values
        return columns

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

    @property
    def ids(self):
        """The node ids, one per row of the store."""
        return self._ids

    def values_at(self, name, rows):
        """The stored values of the column ``name`` at ``rows``, as they are, or None
        when the store has no such column."""
        column = self._columns.get(name)
        if column is None:
            return None
        return column[rows] if isinstance(column, NodeTags) else np.asarray(column)[rows]

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
