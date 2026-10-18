"""The in-memory reader's ways and node tags, kept as arrays and turned into dicts on
demand."""

import numpy as np
from libc.stdint cimport int32_t, int64_t

from pyrosm._arrays import _offsets_from_lengths

# The keys every way record has; a tag keyed like one of them replaces its value.
_RECORD_KEYS = frozenset({"id", "version", "timestamp", "visible", "nodes"})
# Iterating a store builds this many records at a time.
_CHUNK = 65536


def _range_indices(starts, lengths):
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


def _map_strings(dict string_ids, list table, list strings, used):
    """The ids in ``table`` of the block ``strings`` at the positions ``used`` (0 at the
    others), adding the strings ``table`` lacks; ``string_ids`` maps ``table``'s strings
    to their ids."""
    mapping = np.zeros(len(strings), dtype=np.int32)
    for i in used.tolist():
        string = strings[i]
        j = string_ids.get(string)
        if j is None:
            j = len(table)
            string_ids[string] = j
            table.append(string)
        mapping[i] = j
    return mapping


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
        tag_rows = _range_indices(tag_starts, tag_counts)
        refs = self.refs[_range_indices(ref_starts, ref_counts)].tolist()
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
            refs = refs[_range_indices(ways["refs_off"][rows], ref_counts[rows])]
            tag_rows = _range_indices(ways["tags_off"][rows], tag_counts[rows])
            keys, vals = keys[tag_rows], vals[tag_rows]
            ref_counts, tag_counts = ref_counts[rows], tag_counts[rows]
        # The block's string ids used by these ways -> ids in the store's table.
        mapping = _map_strings(
            self._string_ids, self._strings, strings,
            np.unique(np.concatenate([keys, vals])),
        )
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
            _offsets_from_lengths(_joined(parts, "ref_counts", np.int64)),
            _joined(parts, "keys", np.int32),
            _joined(parts, "vals", np.int32),
            _offsets_from_lengths(_joined(parts, "tag_counts", np.int64)),
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


cdef class NodeTags:
    """The tags of the parsed nodes: ``rows`` (the nodes with tags, ascending) and their
    tags (``keys`` / ``vals`` ids into ``strings``, with ``tags_off`` over ``rows``).
    ``len()`` is the number of nodes. Indexing with a node row gives its tags as a dict,
    None for a node without tags; with an array of rows (or a mask, or a slice), an object
    array of them."""

    cdef readonly object rows, keys, vals, tags_off
    cdef readonly list strings
    cdef readonly Py_ssize_t n_nodes
    cdef dict _string_ids
    cdef const int64_t[::1] _rows
    cdef const int64_t[::1] _offsets
    cdef const int32_t[::1] _keys
    cdef const int32_t[::1] _vals

    def __init__(self, rows, keys, vals, tags_off, list strings, dict string_ids,
                 Py_ssize_t n_nodes):
        self.rows = rows
        self.keys = keys
        self.vals = vals
        self.tags_off = tags_off
        self.strings = strings
        self._string_ids = string_ids
        self.n_nodes = n_nodes
        self._rows = rows
        self._offsets = tags_off
        self._keys = keys
        self._vals = vals

    def __len__(self):
        return self.n_nodes

    def __iter__(self):
        return _iter_tags(self)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self.tags_at(index)
        if isinstance(index, slice):
            index = np.arange(*index.indices(self.n_nodes))
        index = np.asarray(index)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        return self.dicts(index)

    cpdef object tags_at(self, Py_ssize_t row):
        """The tags of node ``row``: a dict in their order, None for a node without
        tags."""
        cdef Py_ssize_t lo = 0, hi = self._rows.shape[0], mid, j
        if row < 0:
            row += self.n_nodes
        if not 0 <= row < self.n_nodes:
            raise IndexError("node row %d out of range" % row)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._rows[mid] < row:
                lo = mid + 1
            else:
                hi = mid
        if lo == self._rows.shape[0] or self._rows[lo] != row:
            return None
        tags = {}
        for j in range(self._offsets[lo], self._offsets[lo + 1]):
            tags[self.strings[self._keys[j]]] = self.strings[self._vals[j]]
        return tags

    cpdef dicts(self, node_rows):
        """The tags of the nodes at ``node_rows``, in order, as an object array (see
        :meth:`tags_at`)."""
        cdef Py_ssize_t i, n
        node_rows = np.asarray(node_rows, dtype=np.int64)
        n = len(node_rows)
        out = np.empty(n, dtype=object)
        for i in range(n):
            out[i] = self.tags_at(node_rows[i])
        return out

    cpdef candidate_rows(self, osm_keys, bint keep_all):
        """The rows, in order, of the nodes the node filter
        (:func:`~pyrosm.data_filter.record_should_be_kept`) can keep: the nodes with a tag
        keyed in ``osm_keys`` or, with ``keep_all`` (or ``osm_keys`` not a list of keys),
        every node with tags."""
        if keep_all or not isinstance(osm_keys, (list, tuple, set)):
            return self.rows
        key_ids = [self._string_ids[key] for key in osm_keys if key in self._string_ids]
        if not key_ids:
            return np.empty(0, dtype=np.int64)
        return self.rows[any_per_row(np.isin(self.keys, key_ids), self.tags_off)]


def _iter_tags(NodeTags tags):
    cdef Py_ssize_t start, n = tags.n_nodes
    for start in range(0, n, _CHUNK):
        yield from tags.dicts(np.arange(start, min(start + _CHUNK, n), dtype=np.int64))


cdef class NodeTagsBuilder:
    """Collects the tags of decoded blocks' nodes into :class:`NodeTags`, numbering the
    nodes kept in the order they are added, with one string table for all of them."""

    cdef list _parts
    cdef list _strings
    cdef dict _string_ids
    cdef Py_ssize_t _n_nodes

    def __init__(self):
        self._parts = []
        self._strings = []
        self._string_ids = {}
        self._n_nodes = 0

    def add(self, keys_vals, Py_ssize_t n, list strings, keep=None):
        """Add the tags of a decoded block's ``n`` nodes from their ``keys_vals`` stream
        (each node's key and value ids followed by a 0); ``keep`` (a mask, or None for
        all) selects the nodes the parse keeps."""
        keys_vals = np.asarray(keys_vals)
        if len(keys_vals) == 0:
            counts = np.zeros(n, dtype=np.int64)
            pairs = keys_vals
        else:
            ends = np.flatnonzero(keys_vals == 0)
            starts = np.zeros(n, dtype=np.int64)
            starts[1:] = ends[:-1] + 1
            counts = (ends - starts) // 2
            # The ids other than the 0 delimiters are each node's keys and values in turn.
            pairs = keys_vals[keys_vals != 0]
        keys, vals = pairs[0::2], pairs[1::2]
        if keep is not None:
            rows = np.flatnonzero(keep)
            tag_rows = _range_indices(_offsets_from_lengths(counts)[rows], counts[rows])
            keys, vals, counts = keys[tag_rows], vals[tag_rows], counts[rows]
        self.add_pairs(counts, keys, vals, strings)

    def add_pairs(self, counts, keys, vals, list strings):
        """Add the tags of nodes that have ``counts`` tags each, node after node:
        ``keys`` and ``vals`` are the ids of their ``strings``."""
        counts = np.asarray(counts, dtype=np.int64)
        keys = np.asarray(keys, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.int64)
        tagged = np.flatnonzero(counts > 0)
        mapping = _map_strings(
            self._string_ids, self._strings, strings,
            np.unique(np.concatenate([keys, vals])),
        )
        self._parts.append({
            "rows": tagged + self._n_nodes,
            "counts": counts[tagged],
            "keys": mapping[keys],
            "vals": mapping[vals],
        })
        self._n_nodes += len(counts)

    def build(self):
        """The :class:`NodeTags` of the nodes added."""
        parts = self._parts
        tags = NodeTags(
            _joined(parts, "rows", np.int64),
            _joined(parts, "keys", np.int32),
            _joined(parts, "vals", np.int32),
            _offsets_from_lengths(_joined(parts, "counts", np.int64)),
            self._strings,
            self._string_ids,
            self._n_nodes,
        )
        self._parts = []
        return tags
