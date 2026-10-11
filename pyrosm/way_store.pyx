"""The in-memory reader's ways, kept as arrays and turned into record dicts on demand."""

import numpy as np

# The keys every way record has; a tag keyed like one of them replaces its value.
_RECORD_KEYS = frozenset({"id", "version", "timestamp", "visible", "nodes"})
# Iterating a store builds this many records at a time.
_CHUNK = 65536


def _ranges(starts, lengths):
    """The indices of the ranges ``starts[i] .. starts[i] + lengths[i] - 1``, joined in
    order."""
    lengths = np.asarray(lengths, dtype=np.int64)
    before = np.cumsum(lengths) - lengths
    return np.repeat(np.asarray(starts, dtype=np.int64) - before, lengths) + np.arange(
        int(lengths.sum()), dtype=np.int64
    )


def any_per_row(hits, offsets):
    """For each row of a CSR layout (``offsets``), whether any of its entries in ``hits``
    is True."""
    counts = np.zeros(len(hits) + 1, dtype=np.int64)
    np.cumsum(hits, out=counts[1:])
    return counts[offsets[1:]] > counts[offsets[:-1]]


def _offsets(counts):
    """CSR offsets of rows with ``counts`` entries each."""
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


cdef class WayStore:
    """The ways of a parse as arrays: ``ids``, ``version``, ``timestamp``, ``visible``,
    the node ids (``refs``, way ``i`` owning ``refs[refs_off[i]:refs_off[i + 1]]``) and
    the tags (``keys`` / ``vals`` ids into ``strings``, with ``tags_off``). ``len()`` is
    the number of ways and iterating yields their records (see :meth:`records`)."""

    cdef readonly object ids, version, timestamp, visible, refs, refs_off
    cdef readonly object keys, vals, tags_off
    cdef readonly list strings
    cdef dict _string_ids

    def __init__(self, ids, version, timestamp, visible, refs, refs_off, keys, vals,
                 tags_off, list strings, dict string_ids):
        self.ids = ids
        self.version = version
        self.timestamp = timestamp
        self.visible = visible
        self.refs = refs
        self.refs_off = refs_off
        self.keys = keys
        self.vals = vals
        self.tags_off = tags_off
        self.strings = strings
        self._string_ids = string_ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return _iter_records(self)

    cpdef list records(self, rows=None):
        """The record of each way at ``rows`` (every way when None), in order: ``id``,
        ``version``, ``timestamp``, ``visible``, ``nodes`` (the node ids), then the tags
        in order. A tag keyed ``id`` is kept as ``id_tag``; a tag keyed like one of the
        other keys replaces its value, and a tag keyed ``tags`` is dropped."""
        cdef Py_ssize_t i, j, n, r = 0, t = 0
        if rows is None:
            rows = np.arange(len(self.ids), dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        ref_starts = self.refs_off[rows]
        ref_counts = self.refs_off[rows + 1] - ref_starts
        tag_starts = self.tags_off[rows]
        tag_counts = self.tags_off[rows + 1] - tag_starts
        tag_rows = _ranges(tag_starts, tag_counts)
        refs = self.refs[_ranges(ref_starts, ref_counts)].tolist()
        keys = self.keys[tag_rows].tolist()
        vals = self.vals[tag_rows].tolist()
        ids = self.ids[rows].tolist()
        version = self.version[rows].tolist()
        timestamp = self.timestamp[rows].tolist()
        visible = self.visible[rows].tolist()
        n_refs = ref_counts.tolist()
        n_tags = tag_counts.tolist()
        strings = self.strings
        records = []
        n = len(ids)
        for i in range(n):
            way = {
                "id": ids[i],
                "version": version[i],
                "timestamp": timestamp[i],
                "visible": visible[i],
                "tags": None,
                "nodes": refs[r:r + n_refs[i]],
            }
            r += n_refs[i]
            for j in range(t, t + n_tags[i]):
                key = strings[keys[j]]
                way["id_tag" if key == "id" else key] = strings[vals[j]]
            t += n_tags[i]
            del way["tags"]
            records.append(way)
        return records

    cpdef candidate_rows(self, osm_keys, relation_way_ids, bint keep_all):
        """The rows, in order, of the ways the key check of
        :func:`~pyrosm.data_filter.filter_osm_records` can keep: the ways whose id is in
        ``relation_way_ids``, and the ways with a tag keyed in ``osm_keys`` (``id_tag``
        also matching an ``id`` tag) or, with ``keep_all``, with any tag. Every row when
        an osm key is a key every record has."""
        n = len(self.ids)
        if not isinstance(osm_keys, list):
            osm_keys = [osm_keys]
        if not keep_all and not _RECORD_KEYS.isdisjoint(osm_keys):
            return np.arange(n, dtype=np.int64)
        keep = np.zeros(n, dtype=bool)
        if relation_way_ids is not None:
            keep |= np.isin(self.ids, np.asarray(relation_way_ids, dtype=np.int64))
        if keep_all:
            keep |= np.diff(self.tags_off) > 0
        else:
            tag_keys = set(osm_keys)
            if "id_tag" in tag_keys:
                tag_keys.add("id")
            key_ids = [self._string_ids[key] for key in tag_keys if key in self._string_ids]
            if key_ids:
                keep |= any_per_row(np.isin(self.keys, key_ids), self.tags_off)
        return np.flatnonzero(keep)


def _iter_records(WayStore store):
    cdef Py_ssize_t start, n = len(store.ids)
    for start in range(0, n, _CHUNK):
        yield from store.records(np.arange(start, min(start + _CHUNK, n), dtype=np.int64))


cdef class WayStoreBuilder:
    """Collects the ways of decoded blocks into a :class:`WayStore`, with one string
    table for all of them."""

    cdef list _parts
    cdef list _strings
    cdef dict _string_ids

    def __init__(self):
        self._parts = []
        self._strings = []
        self._string_ids = {}

    def add(self, dict ways, list strings, keep=None):
        """Add the ways of a decoded block (see
        :func:`~pyrosm.primitive_block_decoder.decode_primitive_block`) that ``keep`` (a
        mask, or None for all) selects; ``strings`` is the block's decoded string
        table."""
        cdef Py_ssize_t i
        ref_counts = np.diff(ways["refs_off"])
        tag_counts = np.diff(ways["tags_off"])
        refs, keys, vals = ways["refs"], ways["keys"], ways["vals"]
        part = {
            "ids": ways["id"],
            "version": ways["version"],
            "timestamp": ways["timestamp"],
            "visible": ways["visible"].astype(bool),
        }
        if keep is not None:
            rows = np.flatnonzero(keep)
            if len(rows) == 0:
                return
            part = {name: values[rows] for name, values in part.items()}
            refs = refs[_ranges(ways["refs_off"][rows], ref_counts[rows])]
            tag_rows = _ranges(ways["tags_off"][rows], tag_counts[rows])
            keys, vals = keys[tag_rows], vals[tag_rows]
            ref_counts, tag_counts = ref_counts[rows], tag_counts[rows]
        # The block's string ids used by these ways -> ids in the store's table.
        string_ids = self._string_ids
        table = self._strings
        mapping = np.zeros(len(strings), dtype=np.int32)
        for i in np.unique(np.concatenate([keys, vals])).tolist():
            string = strings[i]
            j = string_ids.get(string)
            if j is None:
                j = len(table)
                string_ids[string] = j
                table.append(string)
            mapping[i] = j
        part.update(
            refs=refs,
            ref_counts=ref_counts,
            keys=mapping[keys],
            vals=mapping[vals],
            tag_counts=tag_counts,
        )
        self._parts.append(part)

    def build(self):
        """The :class:`WayStore` of the ways added."""
        parts = self._parts
        store = WayStore(
            _joined(parts, "ids", np.int64),
            _joined(parts, "version", np.int64),
            _joined(parts, "timestamp", np.int64),
            _joined(parts, "visible", bool),
            _joined(parts, "refs", np.int64),
            _offsets(_joined(parts, "ref_counts", np.int64)),
            _joined(parts, "keys", np.int32),
            _joined(parts, "vals", np.int32),
            _offsets(_joined(parts, "tag_counts", np.int64)),
            self._strings,
            self._string_ids,
        )
        self._parts = []
        return store


def _joined(list parts, str name, dtype):
    """The ``name`` arrays of ``parts`` joined (empty when there are none)."""
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate([part[name] for part in parts]).astype(dtype, copy=False)
