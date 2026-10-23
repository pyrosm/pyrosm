# cython: boundscheck=False, wraparound=False
"""The ways of a threaded read, column by column, without the GIL.

A way store (:class:`~pyrosm.engine.ways.ElementStore`) holds the candidate ways of a read
as flat arrays: ``ids``, the node refs (``refs`` with ``refs_off``), the metadata, and the
tags as ids into one string buffer (``pair_key`` / ``pair_val`` with ``pairs_off``, the
strings in ``str_data`` at ``str_off``). The functions here select ways, by the value filter
of a read and by a bounding box, find the tag columns the ways have, and build a chunk of the
ways as the columns of a result: those columns and the JSON ``tags`` column as Arrow string
buffers, and the geometries as WKB. They release the GIL for the work.

The rules are those of the reader's Python path (``engine.collect``, ``data_filter.pyx``
and ``geometry.pyx``). A way the value filter here does not cover -- a string that is not
valid UTF-8, a value ``str.strip`` may change beyond ASCII -- is left to that path
("check"), and so are the tag columns of a "special" way (see ``special_ways``).
"""

import numpy as np

from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport (
    int32_t, int64_t, uint8_t, uint32_t, uint64_t, uintptr_t, UINT32_MAX
)
from libc.string cimport memcmp, memcpy
from libc.stdlib cimport free, malloc

from pyrosm._pointers cimport _double_ptr, _int64_ptr, _uint8_ptr
from pyrosm.int64_hash cimport Int64Map, Int64Set
from pyrosm.node_lookup cimport NodeLocations

cdef enum:
    # the kinds of a value-filter entry (see ``value_filter``)
    C_FILTER_VALUES = 0
    C_FILTER_MULTI = 1
    C_FILTER_ACCESS = 2
    C_FILTER_ANY = 3
    # what ``value_filter`` decides for a way
    C_DROP = 0
    C_KEEP = 1
    C_CHECK = 2
    # the geometry types of a chunk, as bits of ``build_chunk``'s ``types``
    C_LINESTRING = 1
    C_POLYGON = 2
    # a string's column as a key, when it is not a column index
    C_LEFTOVER = -1
    C_SPECIAL = -2
    # the rule of a key of the polygon features (see ``area_by_tags``)
    C_NOT_A_FEATURE = 0
    C_ALL = 1
    C_PASSLIST = 2
    C_BLOCKLIST = 3

FILTER_VALUES = C_FILTER_VALUES
FILTER_MULTI = C_FILTER_MULTI
FILTER_ACCESS = C_FILTER_ACCESS
FILTER_ANY = C_FILTER_ANY
KEEP = C_KEEP
CHECK = C_CHECK
LINESTRING = C_LINESTRING
POLYGON = C_POLYGON


def _contiguous(a, dtype):
    """``a`` as a contiguous one-dimensional array of ``dtype``; ``ValueError`` for an array
    of another shape (the loops here index its values one after another)."""
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim != 1:
        raise ValueError("arrays must be one-dimensional")
    return a


def _check_offsets(off, Py_ssize_t n_values):
    """Raise ``ValueError`` unless the CSR offsets ``off`` rise from 0 to ``n_values``."""
    cdef Py_ssize_t n = len(off)
    if n == 0 or off[0] != 0 or off[n - 1] != n_values or (off[1:] < off[: n - 1]).any():
        raise ValueError("offsets must rise from 0 to the number of values")


# --- selecting ways ---------------------------------------------------------------------


def _check_store(pairs_off, pair_key, pair_val, str_off, str_data, rows):
    """Raise ``ValueError`` unless the tag columns of a store fit together: the pair offsets
    and string offsets fit their arrays, every pair's ids are strings of the store, and the
    ``rows`` are ways of it."""
    _check_offsets(pairs_off, pair_key.shape[0])
    _check_offsets(str_off, str_data.shape[0])
    if pair_val.shape[0] != pair_key.shape[0]:
        raise ValueError("pair keys and values must have the same length")
    n_strings = str_off.shape[0] - 1
    for ids in (pair_key, pair_val):
        if ids.shape[0] and (ids.min() < 0 or ids.max() >= n_strings):
            raise ValueError("pair ids must be strings of the store")
    if rows.shape[0] and (rows.min() < 0 or rows.max() >= pairs_off.shape[0] - 1):
        raise ValueError("rows must be ways of the store")


def take(off, values, rows):
    """The rows ``rows`` of a ragged column (``values``, row ``i`` at
    ``values[off[i]:off[i + 1]]``): ``(offsets, values)``."""
    off = _contiguous(off, np.int64)
    values = _contiguous(values, np.int64)
    rows = _contiguous(rows, np.int64)
    _check_offsets(off, values.shape[0])
    if rows.shape[0] and (rows.min() < 0 or rows.max() >= off.shape[0] - 1):
        raise ValueError("rows must be rows of the column")
    cdef Py_ssize_t n = rows.shape[0], i, total = 0, length
    cdef bint too_long = False
    cdef int64_t* o = _int64_ptr(off)
    cdef int64_t* r = _int64_ptr(rows)
    new_off = np.zeros(n + 1, dtype=np.int64)
    cdef int64_t* no = _int64_ptr(new_off)
    with nogil:
        for i in range(n):
            length = o[r[i] + 1] - o[r[i]]
            if length > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(int64_t) - total:
                too_long = True
                break
            total += length
            no[i + 1] = total
    if too_long:
        raise OverflowError("the rows hold more values than an array can")
    new_values = np.empty(total, dtype=np.int64)
    cdef int64_t* v = _int64_ptr(values)
    cdef int64_t* nv = _int64_ptr(new_values)
    with nogil:
        for i in range(n):
            if no[i + 1] > no[i]:
                memcpy(nv + no[i], v + o[r[i]], (no[i + 1] - no[i]) * sizeof(int64_t))
    return new_off, new_values


def in_set(ids, Int64Set wanted):
    """Whether each of ``ids`` is in ``wanted``."""
    ids = _contiguous(ids, np.int64)
    cdef Py_ssize_t n = ids.shape[0], i
    cdef int64_t* v = _int64_ptr(ids)
    mask = np.zeros(n, dtype=np.uint8)
    cdef uint8_t* m = _uint8_ptr(mask)
    with nogil:
        for i in range(n):
            m[i] = wanted.contains(v[i])
    return mask.view(bool)


def with_node_in(refs_off, refs, Int64Set nodes):
    """Whether each way (its node refs ``refs[refs_off[i]:refs_off[i + 1]]``) has a node of
    the set ``nodes``."""
    refs_off = _contiguous(refs_off, np.int64)
    refs = _contiguous(refs, np.int64)
    _check_offsets(refs_off, refs.shape[0])
    cdef Py_ssize_t n = refs_off.shape[0] - 1, i, j
    cdef int64_t* o = _int64_ptr(refs_off)
    cdef int64_t* r = _int64_ptr(refs)
    mask = np.zeros(n, dtype=np.uint8)
    cdef uint8_t* m = _uint8_ptr(mask)
    with nogil:
        for i in range(n):
            for j in range(o[i], o[i + 1]):
                if nodes.contains(r[j]):
                    m[i] = 1
                    break
    return mask.view(bool)


# --- strings --------------------------------------------------------------------------


cdef bint _valid_utf8(const uint8_t* s, Py_ssize_t n) noexcept nogil:
    """Whether ``s[:n]`` is well-formed UTF-8, as Python's strict decoder accepts it."""
    # A lead byte gives the length of a character (0x00-0x7F one byte, 0xC2-0xDF two,
    # 0xE0-0xEF three, 0xF0-0xF4 four), and each following byte is 0b10xxxxxx. Refused:
    # 0x80-0xC1 and 0xF5-0xFF as lead bytes, overlong forms (0xE0 then below 0xA0, 0xF0
    # then below 0x90), surrogates (0xED then 0xA0 or above) and code points above
    # U+10FFFF (0xF4 then 0x90 or above).
    cdef Py_ssize_t i = 0
    cdef uint8_t c, c1
    while i < n:
        c = s[i]
        if c < 0x80:
            i += 1
        elif c < 0xC2:
            return False
        elif c < 0xE0:
            if i + 1 >= n or (s[i + 1] & 0xC0) != 0x80:
                return False
            i += 2
        elif c < 0xF0:
            if i + 2 >= n:
                return False
            c1 = s[i + 1]
            if (c1 & 0xC0) != 0x80 or (s[i + 2] & 0xC0) != 0x80:
                return False
            if (c == 0xE0 and c1 < 0xA0) or (c == 0xED and c1 >= 0xA0):
                return False
            i += 3
        elif c < 0xF5:
            if i + 3 >= n:
                return False
            c1 = s[i + 1]
            if ((c1 & 0xC0) != 0x80 or (s[i + 2] & 0xC0) != 0x80
                    or (s[i + 3] & 0xC0) != 0x80):
                return False
            if (c == 0xF0 and c1 < 0x90) or (c == 0xF4 and c1 >= 0x90):
                return False
            i += 4
        else:
            return False
    return True


cdef uint64_t _hash(const uint8_t* s, Py_ssize_t n) noexcept nogil:
    """The 64-bit FNV-1a hash of ``s[:n]``."""
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(n):
        h = (h ^ s[i]) * 1099511628211ULL
    return h


cdef bint _equals(const uint8_t* s, Py_ssize_t n, const char* text) noexcept nogil:
    """Whether ``s[:n]`` is the NUL-terminated ``text``."""
    cdef Py_ssize_t i
    for i in range(n):
        if text[i] == 0 or <uint8_t>text[i] != s[i]:
            return False
    return text[n] == 0


def string_columns(str_off, str_data, columns):
    """For each string: whether it is valid UTF-8, and its column as a key -- the index in
    ``columns`` (a list of bytes), ``-2`` for a metadata name (``id``, ``version``,
    ``timestamp``, ``visible``), ``-1`` otherwise. Returns ``(valid, column)``."""
    str_off = _contiguous(str_off, np.int64)
    str_data = _contiguous(str_data, np.uint8)
    _check_offsets(str_off, str_data.shape[0])
    cdef Py_ssize_t n = str_off.shape[0] - 1, i, slot, size = 16, c
    cdef Py_ssize_t n_cols = len(columns)
    # The column names go into a hash table with linear probing: ``table`` holds a column
    # index per slot (-1 for an empty slot), and is a power of two of at least four slots
    # per column, so a lookup ends at an empty slot after a few probes.
    while size < 4 * n_cols:
        size *= 2
    names = b"".join(columns)
    name_off = np.zeros(n_cols + 1, dtype=np.int64)
    if n_cols:
        np.cumsum([len(k) for k in columns], out=name_off[1:])
    name_data = np.frombuffer(names + b"\0", dtype=np.uint8)
    table = np.full(size, -1, dtype=np.int64)
    cdef int64_t* t = _int64_ptr(table)
    cdef int64_t* no = _int64_ptr(name_off)
    cdef uint8_t* nd = _uint8_ptr(name_data)
    for c in range(n_cols):
        slot = _hash(nd + no[c], no[c + 1] - no[c]) & (size - 1)
        while t[slot] >= 0:
            slot = (slot + 1) & (size - 1)
        t[slot] = c
    valid = np.empty(n, dtype=np.uint8)
    column = np.empty(n, dtype=np.int32)
    cdef uint8_t* v = _uint8_ptr(valid)
    cdef int32_t* col = <int32_t*><uintptr_t>column.ctypes.data
    cdef int64_t* o = _int64_ptr(str_off)
    cdef uint8_t* d = _uint8_ptr(str_data)
    cdef const uint8_t* s
    cdef Py_ssize_t length
    with nogil:
        # Each string: its validity, then whether it is a metadata name, else its column.
        for i in range(n):
            s = d + o[i]
            length = o[i + 1] - o[i]
            v[i] = _valid_utf8(s, length)
            if (_equals(s, length, "id") or _equals(s, length, "version")
                    or _equals(s, length, "timestamp") or _equals(s, length, "visible")):
                col[i] = C_SPECIAL
                continue
            col[i] = C_LEFTOVER
            slot = _hash(s, length) & (size - 1)
            while t[slot] >= 0:
                c = t[slot]
                if (no[c + 1] - no[c] == length
                        and memcmp(nd + no[c], s, length) == 0):
                    col[i] = c
                    break
                slot = (slot + 1) & (size - 1)
    return valid.view(bool), column


cdef inline bint _same_string(const int64_t* so, const uint8_t* sd, int64_t a,
                              int64_t b) noexcept nogil:
    return a == b or (so[a + 1] - so[a] == so[b + 1] - so[b]
                      and memcmp(sd + so[a], sd + so[b], so[a + 1] - so[a]) == 0)


cdef int _repeats_key(const int64_t* pk, Py_ssize_t p0, Py_ssize_t p1, const int64_t* so,
                      const uint8_t* sd, int64_t* table, Py_ssize_t size) noexcept nogil:
    """Whether a key of the pairs ``p0:p1`` occurs twice: by pairs for a few keys, else in
    the hash ``table`` of ``size`` slots (a power of two, at least twice the pairs)."""
    cdef Py_ssize_t p, q, slot
    cdef int64_t a
    if p1 - p0 <= 16:
        for p in range(p0, p1):
            for q in range(p0, p):
                if _same_string(so, sd, pk[p], pk[q]):
                    return 1
        return 0
    for slot in range(size):
        table[slot] = -1
    for p in range(p0, p1):
        a = pk[p]
        slot = _hash(sd + so[a], so[a + 1] - so[a]) & (size - 1)
        while table[slot] >= 0:
            if _same_string(so, sd, table[slot], a):
                return 1
            slot = (slot + 1) & (size - 1)
        table[slot] = a
    return 0


def special_ways(pairs_off, pair_key, pair_val, str_off, str_data, valid, column):
    """Whether each way is special: a key that is a metadata name, a key that occurs twice,
    or a key or value that is not valid UTF-8 (see ``string_columns``)."""
    pairs_off = _contiguous(pairs_off, np.int64)
    pair_key = _contiguous(pair_key, np.int64)
    pair_val = _contiguous(pair_val, np.int64)
    str_off = _contiguous(str_off, np.int64)
    str_data = _contiguous(str_data, np.uint8)
    valid = _contiguous(valid, np.uint8)
    column = _contiguous(column, np.int32)
    _check_store(pairs_off, pair_key, pair_val, str_off, str_data, np.empty(0, np.int64))
    if valid.shape[0] != str_off.shape[0] - 1 or column.shape[0] != str_off.shape[0] - 1:
        raise ValueError("valid and column must have one entry per string")
    cdef Py_ssize_t n = pairs_off.shape[0] - 1, i, p, most = 0, size = 16
    cdef int64_t* po = _int64_ptr(pairs_off)
    cdef int64_t* pk = _int64_ptr(pair_key)
    cdef int64_t* pv = _int64_ptr(pair_val)
    cdef int64_t* so = _int64_ptr(str_off)
    cdef uint8_t* sd = _uint8_ptr(str_data)
    cdef uint8_t* ok = _uint8_ptr(valid)
    cdef int32_t* col = <int32_t*><uintptr_t>column.ctypes.data
    for i in range(n):
        most = max(most, po[i + 1] - po[i])
    while size < 2 * most:
        size *= 2
    table = np.empty(size, dtype=np.int64)
    cdef int64_t* t = _int64_ptr(table)
    special = np.zeros(n, dtype=np.uint8)
    cdef uint8_t* sp = _uint8_ptr(special)
    cdef int64_t a
    cdef Py_ssize_t slots
    with nogil:
        # A way is special when a key is a metadata name or a string is not valid UTF-8,
        # else when a key repeats (``t`` is the scratch table of ``_repeats_key``).
        for i in range(n):
            for p in range(po[i], po[i + 1]):
                a = pk[p]
                if col[a] == C_SPECIAL or not ok[a] or not ok[pv[p]]:
                    sp[i] = 1
                    break
            if sp[i]:
                continue
            slots = 16
            while slots < 2 * (po[i + 1] - po[i]):
                slots *= 2
            sp[i] = _repeats_key(pk, po[i], po[i + 1], so, sd, t, slots)
    return special.view(bool)


def column_presence(pairs_off, pair_key, column, rows, Py_ssize_t n_columns):
    """For the ways ``rows``: which of the ``n_columns`` columns a key routes to, and whether
    a key routes to no column (a leftover tag). Returns ``(present, leftover)``."""
    pairs_off = _contiguous(pairs_off, np.int64)
    pair_key = _contiguous(pair_key, np.int64)
    column = _contiguous(column, np.int32)
    rows = _contiguous(rows, np.int64)
    _check_offsets(pairs_off, pair_key.shape[0])
    if rows.shape[0] and (rows.min() < 0 or rows.max() >= pairs_off.shape[0] - 1):
        raise ValueError("rows must be ways of the store")
    if pair_key.shape[0] and (pair_key.min() < 0 or pair_key.max() >= column.shape[0]):
        raise ValueError("pair ids must be strings of the store")
    if column.shape[0] and column.max() >= n_columns:
        raise ValueError("a column index must be below n_columns")
    present = np.zeros(max(n_columns, 1), dtype=np.uint8)
    cdef uint8_t* pr = _uint8_ptr(present)
    cdef int64_t* po = _int64_ptr(pairs_off)
    cdef int64_t* pk = _int64_ptr(pair_key)
    cdef int32_t* col = <int32_t*><uintptr_t>column.ctypes.data
    cdef int64_t* r = _int64_ptr(rows)
    cdef Py_ssize_t n = rows.shape[0], i, p
    cdef int c
    cdef bint leftover = False
    with nogil:
        for i in range(n):
            for p in range(po[r[i]], po[r[i] + 1]):
                c = col[pk[p]]
                if c >= 0:
                    pr[c] = 1
                else:
                    leftover = True
    return present[:n_columns].view(bool), leftover


def area_by_tags(pairs_off, pair_key, pair_val, str_off, str_data):
    """Whether the tags of each way make it an area when it is closed, as
    :func:`pyrosm.config.polygon_features.is_area` decides from its tag dict: not when its
    ``area`` tag is ``no``, otherwise when one of its tags is a polygon feature. Of a key
    that occurs twice, its last value counts, as in the tag dict."""
    from pyrosm.config.polygon_features import ALL, BLOCKLIST, PASSLIST, POLYGON_FEATURES

    pairs_off = _contiguous(pairs_off, np.int64)
    pair_key = _contiguous(pair_key, np.int64)
    pair_val = _contiguous(pair_val, np.int64)
    str_off = _contiguous(str_off, np.int64)
    str_data = _contiguous(str_data, np.uint8)
    _check_store(pairs_off, pair_key, pair_val, str_off, str_data, np.empty(0, np.int64))
    # The strings of the rule (its keys, their listed values and "no"), and for each string
    # of the store its index among them, negative for any other string.
    names = sorted(
        set(POLYGON_FEATURES).union(*(v for _, v in POLYGON_FEATURES.values()), {"no"})
    )
    index = {name: i for i, name in enumerate(names)}
    _, name_of = string_columns(str_off, str_data, [name.encode() for name in names])
    cdef Py_ssize_t n_names = len(names)
    rules = np.zeros(n_names, dtype=np.uint8)
    listed = np.zeros(n_names * n_names, dtype=np.uint8)  # [key * n_names + value]
    codes = {ALL: C_ALL, PASSLIST: C_PASSLIST, BLOCKLIST: C_BLOCKLIST}
    for key, (rule, values) in POLYGON_FEATURES.items():
        rules[index[key]] = codes[rule]
        for value in values:
            listed[index[key] * n_names + index[value]] = 1
    cdef int32_t area_name = index["area"], no_name = index["no"]
    cdef Py_ssize_t n = pairs_off.shape[0] - 1, i, p, q
    cdef int64_t* po = _int64_ptr(pairs_off)
    cdef int64_t* pk = _int64_ptr(pair_key)
    cdef int64_t* pv = _int64_ptr(pair_val)
    cdef int32_t* nm = <int32_t*><uintptr_t>name_of.ctypes.data
    cdef uint8_t* ru = _uint8_ptr(rules)
    cdef uint8_t* li = _uint8_ptr(listed)
    flags = np.empty(n, dtype=np.uint8)
    cdef uint8_t* out = _uint8_ptr(flags)
    cdef int32_t k, v
    cdef bint polygon, area_no, later, on_list
    with nogil:
        for i in range(n):
            polygon = False
            area_no = False
            for p in range(po[i], po[i + 1]):
                k = nm[pk[p]]
                if k < 0 or ru[k] == C_NOT_A_FEATURE:
                    continue
                later = False
                for q in range(p + 1, po[i + 1]):
                    if nm[pk[q]] == k:
                        later = True
                        break
                if later:
                    continue
                v = nm[pv[p]]
                if k == area_name:
                    if v == no_name:
                        area_no = True
                    else:
                        polygon = True
                elif ru[k] == C_ALL:
                    polygon = True
                else:
                    on_list = v >= 0 and li[k * n_names + v]
                    if (ru[k] == C_PASSLIST and on_list) or (
                        ru[k] == C_BLOCKLIST and not on_list
                    ):
                        polygon = True
            out[i] = polygon and not area_no
    return flags.view(bool)


# --- the value filter -----------------------------------------------------------------


cdef struct _Entry:
    int kind
    Py_ssize_t keys_start, keys_end       # into the key list
    Py_ssize_t values_start, values_end   # into the value list


cdef struct _Filter:
    bint keep
    Py_ssize_t n_entries
    _Entry* entries
    Py_ssize_t n_osm_keys
    const int64_t* osm_off                # osm key k: text[osm_off[k]:osm_off[k + 1]]
    const uint8_t* osm_text
    const int64_t* key_off                # entry keys
    const uint8_t* key_text
    const int64_t* value_off              # entry values
    const uint8_t* value_text


cdef inline bint _is_space(uint8_t c) noexcept nogil:
    """Whether ``c`` is an ASCII character ``str.strip`` removes."""
    return c == 32 or (9 <= c <= 13) or (28 <= c <= 31)


cdef int _in_values(const _Filter* f, const _Entry* e, const uint8_t* s,
                    Py_ssize_t n) noexcept nogil:
    """1 when ``s[:n]`` is one of the entry's values, else 0."""
    cdef Py_ssize_t v, length
    for v in range(e.values_start, e.values_end):
        length = f.value_off[v + 1] - f.value_off[v]
        if length == n and memcmp(f.value_text + f.value_off[v], s, n) == 0:
            return 1
    return 0


cdef int _in_multi(const _Filter* f, const _Entry* e, const uint8_t* s,
                   Py_ssize_t n) noexcept nogil:
    """``MultiValue.__contains__``: 1 when ``s[:n]`` is one of the values, or one of its
    ``;``-separated parts is once stripped; 0 when not; 2 when a part ends in a non-ASCII
    character (which ``str.strip`` may remove)."""
    cdef Py_ssize_t start = 0, end, a, b
    cdef bint semicolon = False
    cdef int found
    if _in_values(f, e, s, n):
        return 1
    for a in range(n):
        if s[a] == c';':
            semicolon = True
            break
    if not semicolon:
        return 0
    # Each part between semicolons, stripped of ASCII whitespace at both ends.
    while start <= n:
        end = start
        while end < n and s[end] != c';':
            end += 1
        a = start
        b = end
        while a < b and _is_space(s[a]):
            a += 1
        while b > a and _is_space(s[b - 1]):
            b -= 1
        if b > a and (s[a] >= 0x80 or s[b - 1] >= 0x80):
            return 2
        if _in_values(f, e, s + a, b - a):
            return 1
        start = end + 1
    return 0


cdef Py_ssize_t _value_of(const uint8_t* key, Py_ssize_t key_len, const int64_t* pk,
                          const int64_t* pv, Py_ssize_t p0, Py_ssize_t p1,
                          const int64_t* so, const uint8_t* sd) noexcept nogil:
    """The pair of the way's last tag keyed ``key``, or -1."""
    cdef Py_ssize_t p, found = -1
    cdef int64_t k
    for p in range(p0, p1):
        k = pk[p]
        if so[k + 1] - so[k] == key_len and memcmp(sd + so[k], key, key_len) == 0:
            found = p
    return found


cdef int _filter_decision(const _Filter* f, const int64_t* pk, const int64_t* pv,
                          Py_ssize_t p0, Py_ssize_t p1, const int64_t* so,
                          const uint8_t* sd, const uint8_t* valid) noexcept nogil:
    """``record_should_be_kept`` for the way with the pairs ``p0:p1``: KEEP, DROP or CHECK
    (decided by the Python path, which compares the strings decoded with replacement
    characters; so for a way with a string that is not valid UTF-8)."""
    cdef Py_ssize_t k, p, kk
    cdef const _Entry* e
    cdef int r
    cdef int64_t v
    cdef bint found = False
    # 1. A way with a string that is not valid UTF-8 is left to the Python path.
    for p in range(p0, p1):
        if not valid[pk[p]] or not valid[pv[p]]:
            return C_CHECK
    # 2. A way without any of the read's keys is dropped.
    for k in range(f.n_osm_keys):
        if _value_of(f.osm_text + f.osm_off[k], f.osm_off[k + 1] - f.osm_off[k], pk, pv,
                     p0, p1, so, sd) >= 0:
            found = True
            break
    if not found:
        return C_DROP
    # 3. Without filter entries, the filter type decides.
    if f.n_entries == 0:
        return C_KEEP if f.keep else C_DROP
    # 4. The first entry the way matches decides: an access rule by the value of the first
    # of its keys the way has, the others by the value of their key (any value for
    # FILTER_ANY). An entry the way does not match passes to the next.
    for k in range(f.n_entries):
        e = &f.entries[k]
        if e.kind == C_FILTER_ACCESS:
            r = 0
            for kk in range(e.keys_start, e.keys_end):
                p = _value_of(f.key_text + f.key_off[kk], f.key_off[kk + 1] - f.key_off[kk],
                              pk, pv, p0, p1, so, sd)
                if p >= 0:
                    v = pv[p]
                    r = _in_multi(f, e, sd + so[v], so[v + 1] - so[v])
                    break
            if r == 2:
                return C_CHECK
            if r == 1:
                return C_KEEP if f.keep else C_DROP
            continue
        kk = e.keys_start
        p = _value_of(f.key_text + f.key_off[kk], f.key_off[kk + 1] - f.key_off[kk], pk, pv,
                      p0, p1, so, sd)
        if p < 0:
            continue
        if e.kind == C_FILTER_ANY:
            return C_KEEP if f.keep else C_DROP
        v = pv[p]
        if e.kind == C_FILTER_VALUES:
            r = _in_values(f, e, sd + so[v], so[v + 1] - so[v])
        else:
            r = _in_multi(f, e, sd + so[v], so[v + 1] - so[v])
        if r == 2:
            return C_CHECK
        if r == 1:
            return C_KEEP if f.keep else C_DROP
    # 5. A way no entry matches gets the opposite of the filter type.
    return C_DROP if f.keep else C_KEEP


def _offsets_and_text(items):
    """``(offsets, text)`` arrays of a list of bytes."""
    off = np.zeros(len(items) + 1, dtype=np.int64)
    if items:
        np.cumsum([len(x) for x in items], out=off[1:])
    return off, np.frombuffer(b"".join(items) + b"\0", dtype=np.uint8)


def value_filter(pairs_off, pair_key, pair_val, str_off, str_data, valid, rows, osm_keys,
                 entries, bint keep):
    """The value filter of a read for the ways ``rows``: KEEP, DROP or CHECK (the Python
    path decides) per way, as ``record_should_be_kept`` decides for a dict filter.
    ``osm_keys`` are the keys a way must have one of (bytes); ``entries`` the filter's
    entries in order, ``(kind, keys, values)`` with ``kind`` one of the FILTER_ values,
    ``keys`` the entry's key (an access rule's keys in order) and ``values`` its values, all
    bytes; ``keep`` is the filter type; ``valid`` whether each string is valid UTF-8
    (``string_columns``)."""
    pairs_off = _contiguous(pairs_off, np.int64)
    pair_key = _contiguous(pair_key, np.int64)
    pair_val = _contiguous(pair_val, np.int64)
    str_off = _contiguous(str_off, np.int64)
    str_data = _contiguous(str_data, np.uint8)
    rows = _contiguous(rows, np.int64)
    valid = _contiguous(valid, np.uint8)
    _check_store(pairs_off, pair_key, pair_val, str_off, str_data, rows)
    if valid.shape[0] != str_off.shape[0] - 1:
        raise ValueError("valid must have one flag per string")
    for kind, keys, values in entries:
        if kind not in (C_FILTER_VALUES, C_FILTER_MULTI, C_FILTER_ACCESS, C_FILTER_ANY):
            raise ValueError("an entry's kind must be one of the FILTER_ values")
        if len(keys) == 0 or (kind != C_FILTER_ACCESS and len(keys) != 1):
            raise ValueError("an entry has one key, an access rule one or more")
    osm_off, osm_text = _offsets_and_text(list(osm_keys))
    all_keys, all_values = [], []
    cdef _Filter f
    cdef Py_ssize_t n_entries = len(entries), k
    f.entries = <_Entry*>malloc(max(n_entries, 1) * sizeof(_Entry))
    if f.entries == NULL:
        raise MemoryError()
    try:
        for k, (kind, keys, values) in enumerate(entries):
            f.entries[k].kind = kind
            f.entries[k].keys_start = len(all_keys)
            all_keys.extend(keys)
            f.entries[k].keys_end = len(all_keys)
            f.entries[k].values_start = len(all_values)
            all_values.extend(values)
            f.entries[k].values_end = len(all_values)
        key_off, key_text = _offsets_and_text(all_keys)
        value_off, value_text = _offsets_and_text(all_values)
        f.keep = keep
        f.n_entries = n_entries
        f.n_osm_keys = len(osm_keys)
        f.osm_off = _int64_ptr(osm_off)
        f.osm_text = _uint8_ptr(osm_text)
        f.key_off = _int64_ptr(key_off)
        f.key_text = _uint8_ptr(key_text)
        f.value_off = _int64_ptr(value_off)
        f.value_text = _uint8_ptr(value_text)
        out = np.empty(rows.shape[0], dtype=np.uint8)
        _filter_rows(&f, pairs_off, pair_key, pair_val, str_off, str_data, valid, rows,
                     out)
        return out
    finally:
        free(f.entries)


cdef void _filter_rows(const _Filter* f, pairs_off, pair_key, pair_val, str_off, str_data,
                       valid, rows, out):
    """Write the value filter's decision for each way of ``rows`` to ``out``, without the
    GIL (see ``value_filter``)."""
    cdef Py_ssize_t n = rows.shape[0], i
    cdef int64_t* po = _int64_ptr(pairs_off)
    cdef int64_t* pk = _int64_ptr(pair_key)
    cdef int64_t* pv = _int64_ptr(pair_val)
    cdef int64_t* so = _int64_ptr(str_off)
    cdef uint8_t* sd = _uint8_ptr(str_data)
    cdef int64_t* r = _int64_ptr(rows)
    cdef uint8_t* o = _uint8_ptr(out)
    cdef uint8_t* ok = _uint8_ptr(valid)
    with nogil:
        for i in range(n):
            o[i] = _filter_decision(f, pk, pv, po[r[i]], po[r[i] + 1], so, sd, ok)


# --- tag columns and the JSON ``tags`` column -----------------------------------------


# JSON strings are escaped as ``rapidjson.dumps`` escapes them, output in ASCII only: ``"``
# and ``\`` take a backslash; backspace, tab, newline, form feed and carriage return their
# short escapes (``\b \t \n \f \r``); the other control characters below 0x20 and every
# non-ASCII character a ``\uXXXX`` escape with uppercase hex digits, a character above
# U+FFFF two (a UTF-16 surrogate pair).


cdef inline Py_ssize_t _utf8_width(uint8_t c) noexcept nogil:
    """The number of bytes of the UTF-8 character the non-ASCII lead byte ``c`` begins."""
    return 2 if c < 0xE0 else (3 if c < 0xF0 else 4)


cdef Py_ssize_t _escaped_length(const uint8_t* s, Py_ssize_t n) noexcept nogil:
    """The length of the valid UTF-8 ``s[:n]`` as a JSON string body, escaped as
    ``rapidjson.dumps`` escapes it (ASCII only), or -1 when it exceeds ``PY_SSIZE_T_MAX``.
    For bytes that are not valid UTF-8 it is the length ``_write_escaped`` writes."""
    cdef Py_ssize_t i = 0, length = 0
    cdef uint8_t c
    while i < n:
        if length > PY_SSIZE_T_MAX - 12:
            return -1
        c = s[i]
        if c < 0x80:
            if (c == c'"' or c == c'\\' or c == c'\b' or c == c'\t' or c == c'\n'
                    or c == c'\f' or c == c'\r'):
                length += 2
            elif c < 0x20:
                length += 6
            else:
                length += 1
            i += 1
        else:
            length += 6 if _utf8_width(c) < 4 else 12
            i += _utf8_width(c)
    return length


cdef inline uint8_t _hex_digit(uint32_t v) noexcept nogil:
    """The uppercase hex digit of ``v`` (0-15)."""
    return <uint8_t>(c'0' + v) if v < 10 else <uint8_t>(c'A' + v - 10)


cdef inline Py_ssize_t _write_unicode_escape(uint8_t* out,
                                             uint32_t unit) noexcept nogil:
    """Write the escape ``\\uXXXX`` of the UTF-16 code unit ``unit``; returns 6."""
    out[0] = c'\\'
    out[1] = c'u'
    out[2] = _hex_digit((unit >> 12) & 15)
    out[3] = _hex_digit((unit >> 8) & 15)
    out[4] = _hex_digit((unit >> 4) & 15)
    out[5] = _hex_digit(unit & 15)
    return 6


cdef Py_ssize_t _write_escaped(uint8_t* out, const uint8_t* s, Py_ssize_t n) noexcept nogil:
    """Write the valid UTF-8 ``s[:n]`` escaped (see ``_escaped_length``); returns the
    length written."""
    cdef Py_ssize_t i = 0, w = 0, j, width
    cdef uint8_t c
    cdef uint32_t cp
    while i < n:
        c = s[i]
        if c < 0x80:
            if c == c'"' or c == c'\\':
                out[w] = c'\\'
                out[w + 1] = c
                w += 2
            elif c == c'\b':
                out[w] = c'\\'
                out[w + 1] = c'b'
                w += 2
            elif c == c'\t':
                out[w] = c'\\'
                out[w + 1] = c't'
                w += 2
            elif c == c'\n':
                out[w] = c'\\'
                out[w + 1] = c'n'
                w += 2
            elif c == c'\f':
                out[w] = c'\\'
                out[w + 1] = c'f'
                w += 2
            elif c == c'\r':
                out[w] = c'\\'
                out[w + 1] = c'r'
                w += 2
            elif c < 0x20:
                w += _write_unicode_escape(out + w, c)
            else:
                out[w] = c
                w += 1
            i += 1
            continue
        # A non-ASCII character: its code point from the 2, 3 or 4 bytes its lead byte
        # names (a byte past ``n``, only in bytes that are not valid UTF-8, read as 0). A
        # 4-byte character, above U+FFFF, becomes a UTF-16 surrogate pair.
        width = _utf8_width(c)
        cp = c & (0x1F if width == 2 else (0x0F if width == 3 else 0x07))
        for j in range(1, width):
            cp = (cp << 6) | ((s[i + j] & 0x3F) if i + j < n else 0)
        i += width
        if width < 4:
            w += _write_unicode_escape(out + w, cp)
        else:
            cp -= 0x10000
            w += _write_unicode_escape(out + w, 0xD800 + (cp >> 10))
            w += _write_unicode_escape(out + w, 0xDC00 + (cp & 0x3FF))
    return w


cdef Py_ssize_t _int_length(int64_t v) noexcept nogil:
    """The number of characters of ``v`` in decimal (with the minus sign)."""
    cdef Py_ssize_t n = 1
    cdef uint64_t u
    # The magnitude as unsigned, written so that it also holds for the most negative int64
    # (whose negation does not fit in an int64).
    if v < 0:
        n += 1
        u = <uint64_t>(-(v + 1)) + 1
    else:
        u = <uint64_t>v
    while u >= 10:
        u //= 10
        n += 1
    return n


cdef Py_ssize_t _write_int(uint8_t* out, int64_t v) noexcept nogil:
    """Write ``v`` in decimal; returns the number of characters written."""
    cdef Py_ssize_t n = _int_length(v), w = n - 1
    cdef uint64_t u
    if v < 0:
        out[0] = c'-'
        u = <uint64_t>(-(v + 1)) + 1
    else:
        u = <uint64_t>v
    # The digits from the last one back.
    while True:
        out[w] = <uint8_t>(c'0' + u % 10)
        u //= 10
        w -= 1
        if u == 0:
            break
    return n


cdef Py_ssize_t _write_text(uint8_t* out, const char* text) noexcept nogil:
    """Write the NUL-terminated ``text``; returns its length."""
    cdef Py_ssize_t n = 0
    while text[n] != 0:
        out[n] = <uint8_t>text[n]
        n += 1
    return n


cdef struct _Chunk:
    const int64_t* rows                # the store rows of the chunk
    Py_ssize_t n
    const int64_t* pairs_off
    const int64_t* pair_key
    const int64_t* pair_val
    const int64_t* str_off
    const uint8_t* str_data
    const int32_t* column              # a string's column as a key (see string_columns)
    const int32_t* out_column          # a column's position among the output columns
    const uint8_t* special             # per row: the caller fills its tag columns and JSON
    const int64_t* version
    const int64_t* timestamp
    const int64_t* visible
    Py_ssize_t n_out
    bint json_version, json_timestamp, json_visible
    const int64_t* refs_off
    const int64_t* refs
    const int64_t* node_row            # per ref of the chunk's rows in order: store row or -1
    const int64_t* node_start          # per row: its first entry in node_row
    const double* lon
    const double* lat
    bint network                       # lines only
    const uint8_t* area_by_tags        # per way of the store: an area when closed, or NULL
    uint8_t* kind                      # per row: 0 (no geometry), LINESTRING or POLYGON
    int64_t* n_points
    uint8_t* close                     # per row: the ring repeats its first point
    uint8_t* all_valid                 # per row: every node has valid coordinates


cdef list _point_at_tag_arrays(_Chunk* ch, rows, pairs_off, pair_key, pair_val, str_off,
                               str_data, valid, column, out_column, Py_ssize_t n_out,
                               special, version, timestamp, visible, bint json_version,
                               bint json_timestamp, bint json_visible):
    """Point ``ch``'s row and tag fields at the arrays, made contiguous; returns them, for
    the caller to keep while it uses ``ch``. Raises ``ValueError`` unless the arrays fit
    together and the tags of the rows that are not special are valid UTF-8."""
    arrays = [
        _contiguous(rows, np.int64),
        _contiguous(pairs_off, np.int64),
        _contiguous(pair_key, np.int64),
        _contiguous(pair_val, np.int64),
        _contiguous(str_off, np.int64),
        _contiguous(str_data, np.uint8),
        _contiguous(valid, np.uint8),
        _contiguous(column, np.int32),
        _contiguous(out_column, np.int32),
        _contiguous(special, np.uint8),
        _contiguous(version, np.int64),
        _contiguous(timestamp, np.int64),
        _contiguous(visible, np.int64),
    ]
    (rows, pairs_off, pair_key, pair_val, str_off, str_data, valid, column, out_column,
     special, version, timestamp, visible) = arrays
    _check_store(pairs_off, pair_key, pair_val, str_off, str_data, rows)
    n_strings = str_off.shape[0] - 1
    n_ways = pairs_off.shape[0] - 1
    if valid.shape[0] != n_strings or column.shape[0] != n_strings:
        raise ValueError("valid and column must have one entry per string")
    if column.shape[0] and column.max() >= out_column.shape[0]:
        raise ValueError("a column index must be a position of out_column")
    if n_out < 0 or (out_column.shape[0] and out_column.max() >= n_out):
        raise ValueError("an output position must be below n_out")
    if special.shape[0] != rows.shape[0]:
        raise ValueError("special must have one entry per row")
    if any(a.shape[0] != n_ways for a in (version, timestamp, visible)):
        raise ValueError("the metadata must have one entry per way")
    ch.rows = _int64_ptr(rows)
    ch.n = rows.shape[0]
    ch.pairs_off = _int64_ptr(pairs_off)
    ch.pair_key = _int64_ptr(pair_key)
    ch.pair_val = _int64_ptr(pair_val)
    ch.str_off = _int64_ptr(str_off)
    ch.str_data = _uint8_ptr(str_data)
    ch.column = <int32_t*><uintptr_t>column.ctypes.data
    ch.out_column = <int32_t*><uintptr_t>out_column.ctypes.data
    ch.special = _uint8_ptr(special)
    ch.version = _int64_ptr(version)
    ch.timestamp = _int64_ptr(timestamp)
    ch.visible = _int64_ptr(visible)
    ch.n_out = n_out
    ch.json_version = json_version
    ch.json_timestamp = json_timestamp
    ch.json_visible = json_visible
    cdef const uint8_t* ok = _uint8_ptr(valid)
    cdef Py_ssize_t i, p, w
    cdef bint invalid = False
    with nogil:
        for i in range(ch.n):
            if ch.special[i]:
                continue
            w = ch.rows[i]
            for p in range(ch.pairs_off[w], ch.pairs_off[w + 1]):
                if not ok[ch.pair_key[p]] or not ok[ch.pair_val[p]]:
                    invalid = True
                    break
            if invalid:
                break
    if invalid:
        raise ValueError("the tags of a row that is not special must be valid UTF-8")
    return arrays


def tag_columns(rows, pairs_off, pair_key, pair_val, str_off, str_data, valid, column,
                out_column, Py_ssize_t n_out, special, version, timestamp, visible,
                bint json, bint json_version, bint json_timestamp, bint json_visible):
    """The tag columns and the JSON ``tags`` column of the ways ``rows`` of a store.

    ``valid`` and ``column`` give each string's validity and column as a key
    (``string_columns``), ``out_column`` each column's position among the ``n_out`` output
    tag columns (-1 when not output). The rows marked ``special`` (one entry per row) get no
    tag values or JSON here: the caller fills them in. ``json`` builds the JSON ``tags``
    column, with the metadata the ``json_*`` flags name first (from ``version``,
    ``timestamp`` and ``visible``, one entry per way of the store).

    Returns a dict: ``columns`` (per output tag column ``(offsets, data, valid)``, an Arrow
    string column of the rows) and ``json`` (the same for the JSON column, or None)."""
    cdef _Chunk ch
    arrays = _point_at_tag_arrays(&ch, rows, pairs_off, pair_key, pair_val, str_off,
                                  str_data, valid, column, out_column, n_out, special,
                                  version, timestamp, visible, json_version,
                                  json_timestamp, json_visible)
    kept = np.arange(ch.n, dtype=np.int64)
    return {
        "columns": _tag_columns(&ch, kept),
        "json": _json_column(&ch, kept) if json else None,
    }


cdef list _tag_columns(const _Chunk* ch, kept):
    """``(offsets, data, valid)`` of each output tag column for the rows ``kept`` (positions
    in the chunk's rows)."""
    cdef Py_ssize_t n = kept.shape[0], i, w, p, c, n_out = ch.n_out, length
    cdef int64_t* kp = _int64_ptr(kept)
    sizes = np.zeros(max(n_out, 1), dtype=np.int64)
    cdef int64_t* sz = _int64_ptr(sizes)
    cdef int col
    cdef int64_t v
    cdef bint too_long = False
    # Two passes: the bytes of each output column, then the values copied in. A column
    # is an Arrow string column: ``offsets`` per row, the values back to back, ``valid``.
    with nogil:
        for i in range(n):
            if ch.special[kp[i]]:
                continue
            w = ch.rows[kp[i]]
            for p in range(ch.pairs_off[w], ch.pairs_off[w + 1]):
                col = ch.column[ch.pair_key[p]]
                if col >= 0 and ch.out_column[col] >= 0:
                    c = ch.out_column[col]
                    v = ch.pair_val[p]
                    length = ch.str_off[v + 1] - ch.str_off[v]
                    if length > PY_SSIZE_T_MAX - sz[c]:
                        too_long = True
                        break
                    sz[c] += length
            if too_long:
                break
    if too_long:
        raise OverflowError("a tag column holds more bytes than an array can")
    offsets = np.zeros((max(n_out, 1), n + 1), dtype=np.int64)
    valid = np.zeros((max(n_out, 1), n), dtype=np.uint8)
    datas = [np.empty(max(sizes[c], 1), dtype=np.uint8) for c in range(n_out)]
    pointers = np.array([d.ctypes.data for d in datas] or [0], dtype=np.uint64)
    cdef uint64_t* ptr = <uint64_t*><uintptr_t>pointers.ctypes.data
    cdef int64_t* off = _int64_ptr(offsets)
    cdef uint8_t* ok = _uint8_ptr(valid)
    used = np.zeros(max(n_out, 1), dtype=np.int64)
    cdef int64_t* u = _int64_ptr(used)
    with nogil:
        for i in range(n):
            w = ch.rows[kp[i]]
            if not ch.special[kp[i]]:
                for p in range(ch.pairs_off[w], ch.pairs_off[w + 1]):
                    col = ch.column[ch.pair_key[p]]
                    if col >= 0 and ch.out_column[col] >= 0:
                        c = ch.out_column[col]
                        v = ch.pair_val[p]
                        length = ch.str_off[v + 1] - ch.str_off[v]
                        memcpy(<uint8_t*><uintptr_t>ptr[c] + u[c],
                               ch.str_data + ch.str_off[v], length)
                        u[c] += length
                        ok[c * n + i] = 1
            for c in range(n_out):
                off[c * (n + 1) + i + 1] = u[c]
    return [(offsets[c], datas[c][: sizes[c]], valid[c]) for c in range(n_out)]


cdef tuple _json_column(const _Chunk* ch, kept):
    """``(offsets, data, valid)`` of the JSON ``tags`` column for the rows ``kept``: the
    metadata the chunk puts there, then the tags that route to no output column."""
    cdef Py_ssize_t n = kept.shape[0], i, w, p, size = 0, length, entries
    cdef Py_ssize_t key_len, val_len
    cdef int64_t* kp = _int64_ptr(kept)
    cdef int col
    cdef int64_t k, v
    cdef bint too_long = False
    lengths = np.zeros(n, dtype=np.int64)
    cdef int64_t* ln = _int64_ptr(lengths)
    # Two passes, as in ``_tag_columns``. The lengths: ``"version":`` and ``"visible":``
    # are 10 characters and ``"timestamp":`` 12, ``true``/``false`` 4/5; a tag adds its
    # escaped key and value with 5 characters (``"`` ``":"`` ``"``); the braces and the
    # commas between the entries add ``1 + entries``. A row without entries has no JSON.
    with nogil:
        for i in range(n):
            if ch.special[kp[i]]:
                continue
            w = ch.rows[kp[i]]
            length = 0
            entries = 0
            if ch.json_version:
                length += 10 + _int_length(ch.version[w])
                entries += 1
            if ch.json_timestamp:
                length += 12 + _int_length(ch.timestamp[w])
                entries += 1
            if ch.json_visible:
                length += 10 + (4 if ch.visible[w] else 5)
                entries += 1
            for p in range(ch.pairs_off[w], ch.pairs_off[w + 1]):
                k = ch.pair_key[p]
                col = ch.column[k]
                if col >= 0 and ch.out_column[col] >= 0:
                    continue
                v = ch.pair_val[p]
                key_len = _escaped_length(ch.str_data + ch.str_off[k],
                                          ch.str_off[k + 1] - ch.str_off[k])
                val_len = _escaped_length(ch.str_data + ch.str_off[v],
                                          ch.str_off[v + 1] - ch.str_off[v])
                if (key_len < 0 or val_len < 0 or key_len > PY_SSIZE_T_MAX - 5 - val_len
                        or 5 + key_len + val_len > PY_SSIZE_T_MAX - length):
                    too_long = True
                    break
                length += 5 + key_len + val_len
                entries += 1
            if too_long or (entries and length > PY_SSIZE_T_MAX - 1 - entries - size):
                too_long = True
                break
            if entries:
                ln[i] = length + 1 + entries
            size += ln[i]
    if too_long:
        raise OverflowError("the JSON column holds more bytes than an array can")
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    data = np.empty(max(size, 1), dtype=np.uint8)
    cdef uint8_t* d = _uint8_ptr(data)
    cdef int64_t* off = _int64_ptr(offsets)
    cdef uint8_t* out
    cdef bint comma
    with nogil:
        for i in range(n):
            if ln[i] == 0:
                continue
            w = ch.rows[kp[i]]
            out = d + off[i]
            out[0] = c'{'
            length = 1
            comma = False
            if ch.json_version:
                length += _write_text(out + length, "\"version\":")
                length += _write_int(out + length, ch.version[w])
                comma = True
            if ch.json_timestamp:
                if comma:
                    out[length] = c','
                    length += 1
                length += _write_text(out + length, "\"timestamp\":")
                length += _write_int(out + length, ch.timestamp[w])
                comma = True
            if ch.json_visible:
                if comma:
                    out[length] = c','
                    length += 1
                length += _write_text(out + length, "\"visible\":")
                length += _write_text(out + length, "true" if ch.visible[w] else "false")
                comma = True
            for p in range(ch.pairs_off[w], ch.pairs_off[w + 1]):
                k = ch.pair_key[p]
                col = ch.column[k]
                if col >= 0 and ch.out_column[col] >= 0:
                    continue
                v = ch.pair_val[p]
                if comma:
                    out[length] = c','
                    length += 1
                out[length] = c'"'
                length += 1
                length += _write_escaped(out + length, ch.str_data + ch.str_off[k],
                                         ch.str_off[k + 1] - ch.str_off[k])
                out[length] = c'"'
                out[length + 1] = c':'
                out[length + 2] = c'"'
                length += 3
                length += _write_escaped(out + length, ch.str_data + ch.str_off[v],
                                         ch.str_off[v + 1] - ch.str_off[v])
                out[length] = c'"'
                length += 1
                comma = True
            out[length] = c'}'
    return offsets, data[:size], (lengths > 0).astype(np.uint8)


# --- geometries -----------------------------------------------------------------------


# The ring orientation of GEOS's ``Orientation::isCCW``, which shapely's
# ``orient_polygons`` uses. Its ``orientationIndex`` (the sign of a 2x2 determinant) is
# decided here only when GEOS's double-precision filter is certain of the sign; otherwise
# 2 is returned and the caller asks GEOS.
cdef int _orientation_index(double ax, double ay, double bx, double by, double cx,
                            double cy) noexcept nogil:
    """The orientation of ``c`` to the segment ``a``-``b``: 1 counter-clockwise, -1
    clockwise, 0 collinear, 2 when double precision cannot tell."""
    # det = (a - c) x (b - c). When its two products have opposite signs (or one is 0) the
    # sign of their difference is exact; otherwise it is trusted only when it is larger
    # than 1e-15 times their sum (the bound GEOS uses), else undecided.
    cdef double detsum
    cdef double detleft = (ax - cx) * (by - cy)
    cdef double detright = (ay - cy) * (bx - cx)
    cdef double det = detleft - detright
    if detleft > 0.0:
        if detright <= 0.0:
            return (det > 0.0) - (det < 0.0)
        detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return (det > 0.0) - (det < 0.0)
        detsum = -detleft - detright
    else:
        return (det > 0.0) - (det < 0.0)
    if det >= 1e-15 * detsum or -det >= 1e-15 * detsum:
        return (det > 0.0) - (det < 0.0)
    return 2


cdef int _is_ccw(const double* xy, Py_ssize_t n_points) noexcept nogil:
    """``Orientation::isCCW`` of the closed ring of ``n_points`` points ``xy`` (x, y
    pairs): 1 counter-clockwise, 0 not, 2 when double precision cannot tell."""
    cdef Py_ssize_t n = n_points - 1, i, hi = 0, low_i, down_hi
    cdef double prev_y, py, hx, hy, lx = 0, ly = 0
    cdef int index
    if n < 3:
        return 0
    # The highest point ``hi`` (the first of a run of points at that height, reached going
    # up), with the point before it, ``(lx, ly)``.
    hx = xy[0]
    hy = xy[1]
    prev_y = hy
    for i in range(1, n + 1):
        py = xy[2 * i + 1]
        if py > prev_y and py >= hy:
            hx = xy[2 * i]
            hy = py
            hi = i
            lx = xy[2 * i - 2]
            ly = xy[2 * i - 1]
        prev_y = py
    if hi == 0:
        return 0
    # ``low_i``: the first point after the high run that is lower; ``down_hi``: the last
    # point of the high run (where the ring starts down again).
    low_i = hi
    while True:
        low_i = (low_i + 1) % n
        if low_i == hi or xy[2 * low_i + 1] != hy:
            break
    down_hi = low_i - 1 if low_i > 0 else n - 1
    # A single high point: the ring's turn at it gives the orientation (degenerate,
    # repeated points: not counter-clockwise). A flat top: the direction along it does.
    if hx == xy[2 * down_hi] and hy == xy[2 * down_hi + 1]:
        if ((lx == hx and ly == hy)
                or (xy[2 * low_i] == hx and xy[2 * low_i + 1] == hy)
                or (lx == xy[2 * low_i] and ly == xy[2 * low_i + 1])):
            return 0
        index = _orientation_index(lx, ly, hx, hy, xy[2 * low_i], xy[2 * low_i + 1])
        if index == 2:
            return 2
        return 1 if index == 1 else 0
    return 1 if xy[2 * down_hi] - hx < 0 else 0


cdef inline bint _valid_node(const _Chunk* ch, int64_t row) noexcept nogil:
    """Whether the node store row ``row`` (-1: a node not in the store) has coordinates
    within lon -180..180 and lat -90..90."""
    cdef double x, y
    if row < 0:
        return False
    x = ch.lon[row]
    y = ch.lat[row]
    return x <= 180.0 and x >= -180.0 and y <= 90.0 and y >= -90.0


cdef void _measure_geometry(const _Chunk* ch, Py_ssize_t i) noexcept nogil:
    """The geometry of row ``i`` as geometry.pyx builds it: its kind, its number of points,
    whether the ring repeats its first point and whether all its nodes are valid."""
    cdef Py_ssize_t w = ch.rows[i], j, n, k = 0, p, first = -1, last = -1
    cdef const int64_t* nrow = ch.node_row + ch.node_start[i]
    cdef int64_t r0 = ch.refs_off[w]
    cdef bint is_area = False, all_valid = True
    n = ch.refs_off[w + 1] - r0
    ch.kind[i] = 0
    ch.close[i] = 0
    ch.n_points[i] = 0
    ch.all_valid[i] = 0
    if n == 0:
        return
    # The valid nodes: how many (``k``), the first and the last.
    for j in range(n):
        if _valid_node(ch, nrow[j]):
            if first < 0:
                first = j
            last = j
            k += 1
        else:
            all_valid = False
    ch.all_valid[i] = all_valid
    # A closed way is an area when its tags make it one (``area_by_tags``); a network's
    # ways are lines.
    if not ch.network and ch.refs[r0] == ch.refs[r0 + n - 1]:
        is_area = ch.area_by_tags[w] != 0
    # An area with every node valid and 4 or more of them is a polygon of all its nodes;
    # otherwise of its valid nodes when there are 3 or more, closed by repeating the first
    # when there are only 3 or the first and last differ. A line needs 2 valid nodes.
    if is_area:
        if all_valid and n >= 4:
            ch.kind[i] = C_POLYGON
            ch.n_points[i] = n
        elif k > 2:
            ch.kind[i] = C_POLYGON
            ch.n_points[i] = k
            if k == 3 or (ch.lon[nrow[first]] != ch.lon[nrow[last]]
                          or ch.lat[nrow[first]] != ch.lat[nrow[last]]):
                ch.close[i] = 1
                ch.n_points[i] = k + 1
        return
    if k >= 2:
        ch.kind[i] = C_LINESTRING
        ch.n_points[i] = k


cdef inline void _put_u32(uint8_t* out, uint32_t v) noexcept nogil:
    """Write ``v`` in the machine's byte order."""
    memcpy(out, &v, 4)


cdef inline uint8_t _host_order() noexcept nogil:
    """WKB's byte order marker of this machine (1 little-endian, 0 big-endian): the
    geometries are written in its order, as ``shapely.to_wkb`` writes them."""
    cdef uint32_t one = 1
    return (<uint8_t*>&one)[0]


cdef Py_ssize_t _write_wkb(const _Chunk* ch, Py_ssize_t i, uint8_t* out,
                           double* xy) noexcept nogil:
    """Write row ``i``'s geometry as 2D WKB in the machine's byte order, and its points to
    ``xy``; a polygon's ring counter-clockwise. Returns the bytes written, or -1 (nothing
    written to ``out``) when the ring's orientation needs GEOS."""
    cdef Py_ssize_t w = ch.rows[i], j, k = 0, header, n_points = ch.n_points[i]
    cdef const int64_t* nrow = ch.node_row + ch.node_start[i]
    cdef Py_ssize_t n = ch.refs_off[w + 1] - ch.refs_off[w]
    cdef int ccw
    cdef double tx, ty
    for j in range(n):
        if ch.all_valid[i] or _valid_node(ch, nrow[j]):
            xy[2 * k] = ch.lon[nrow[j]]
            xy[2 * k + 1] = ch.lat[nrow[j]]
            k += 1
    if ch.close[i]:
        xy[2 * k] = xy[0]
        xy[2 * k + 1] = xy[1]
    # WKB: a byte-order byte, the geometry type as uint32 (2 LineString, 3 Polygon), for a
    # polygon its number of rings (1), the number of points, then x and y of each point
    # as doubles.
    if ch.kind[i] == C_POLYGON:
        ccw = _is_ccw(xy, n_points)
        if ccw == 2:
            return -1
        # A ring that is not counter-clockwise is reversed in place.
        if ccw == 0:
            for j in range(n_points // 2):
                tx = xy[2 * j]
                ty = xy[2 * j + 1]
                xy[2 * j] = xy[2 * (n_points - 1 - j)]
                xy[2 * j + 1] = xy[2 * (n_points - 1 - j) + 1]
                xy[2 * (n_points - 1 - j)] = tx
                xy[2 * (n_points - 1 - j) + 1] = ty
        out[0] = _host_order()
        _put_u32(out + 1, 3)
        _put_u32(out + 5, 1)
        _put_u32(out + 9, <uint32_t>n_points)
        header = 13
    else:
        out[0] = _host_order()
        _put_u32(out + 1, 2)
        _put_u32(out + 5, <uint32_t>n_points)
        header = 9
    memcpy(out + header, xy, 16 * n_points)
    return header + 16 * n_points


cdef inline Py_ssize_t _wkb_size(const _Chunk* ch, Py_ssize_t i) noexcept nogil:
    """The WKB size of row ``i``'s geometry (see ``_write_wkb``)."""
    return (13 if ch.kind[i] == C_POLYGON else 9) + 16 * ch.n_points[i]


def build_chunk(rows, refs_off, refs, pairs_off, pair_key, pair_val, str_off, str_data,
                valid, column, out_column, Py_ssize_t n_out, special, version, timestamp,
                visible, NodeLocations nodes not None, bint network, area_by_tags,
                bint json,
                bint json_version, bint json_timestamp, bint json_visible):
    """Build the ways ``rows`` of a store as the columns of a result chunk.

    The tag arguments are those of :func:`tag_columns`. ``nodes`` holds the coordinates of
    the ways' nodes; ``network`` builds lines only; ``area_by_tags`` holds, per way of the
    store, whether its tags make it an area when it is closed (see
    :func:`area_by_tags`), and is not read for a network.

    Returns a dict: ``kept`` (the positions in ``rows`` that have a geometry, the chunk's
    rows), ``wkb_off`` and ``wkb`` (their geometries), ``types`` (bits of
    C_LINESTRING and C_POLYGON), ``bounds``, ``coords`` and ``coords_off`` (each geometry's
    points), ``check`` (rows whose ring orientation needs GEOS; their WKB is left
    unwritten), ``columns`` and ``json`` (those of :func:`tag_columns` for the rows
    ``kept``)."""
    cdef _Chunk ch
    arrays = _point_at_tag_arrays(&ch, rows, pairs_off, pair_key, pair_val, str_off,
                                  str_data, valid, column, out_column, n_out, special,
                                  version, timestamp, visible, json_version,
                                  json_timestamp, json_visible)
    refs_off = _contiguous(refs_off, np.int64)
    refs = _contiguous(refs, np.int64)
    _check_offsets(refs_off, refs.shape[0])
    if refs_off.shape[0] != pairs_off.shape[0]:
        raise ValueError("refs_off and pairs_off must have one entry per way and one more")
    if not network:
        area_by_tags = _contiguous(area_by_tags, np.bool_).view(np.uint8)
        if area_by_tags.shape[0] != refs_off.shape[0] - 1:
            raise ValueError("area_by_tags must have one flag per way")
    lon = np.asarray(nodes._lon)
    lat = np.asarray(nodes._lat)
    if len(nodes._ids) > min(lon.shape[0], lat.shape[0]):
        raise ValueError("the node store must have coordinates for each of its ids")
    cdef Py_ssize_t n = ch.n, i, j, w, total = 0, length
    cdef const int64_t* r = ch.rows
    cdef int64_t* ro = _int64_ptr(refs_off)
    cdef int64_t* rf = _int64_ptr(refs)
    cdef bint too_long = False
    # 1. Each ref of the rows, in order, as its node store row (-1 when the store lacks
    # it): ``node_row``, row ``i``'s refs starting at ``node_start[i]``.
    node_start = np.empty(n + 1, dtype=np.int64)
    cdef int64_t* ns = _int64_ptr(node_start)
    with nogil:
        for i in range(n):
            ns[i] = total
            length = ro[r[i] + 1] - ro[r[i]]
            if length > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(int64_t) - total:
                too_long = True
                break
            total += length
        ns[n] = total
    if too_long:
        raise OverflowError("the rows hold more node refs than an array can")
    node_row = np.empty(max(total, 1), dtype=np.int64)
    cdef int64_t* nr = _int64_ptr(node_row)
    cdef Int64Map index = nodes._id2idx
    with nogil:
        for i in range(n):
            w = r[i]
            for j in range(ro[w], ro[w + 1]):
                nr[ns[i] + j - ro[w]] = index.get(rf[j], -1)
    kind = np.zeros(n, dtype=np.uint8)
    n_points = np.zeros(n, dtype=np.int64)
    close = np.zeros(n, dtype=np.uint8)
    all_valid = np.zeros(n, dtype=np.uint8)
    ch.refs_off = ro
    ch.refs = rf
    ch.node_row = nr
    ch.node_start = ns
    ch.lon = _double_ptr(lon) if lon.shape[0] else NULL
    ch.lat = _double_ptr(lat) if lat.shape[0] else NULL
    ch.network = network
    ch.area_by_tags = NULL if network else _uint8_ptr(area_by_tags)
    ch.kind = _uint8_ptr(kind)
    ch.n_points = _int64_ptr(n_points)
    ch.close = _uint8_ptr(close)
    ch.all_valid = _uint8_ptr(all_valid)

    # 2. Each row's geometry kind and number of points, so that the output arrays can be
    # allocated at their sizes.
    cdef Py_ssize_t n_kept = 0, wkb_bytes = 0, points = 0, size
    with nogil:
        for i in range(n):
            _measure_geometry(&ch, i)
            if not ch.kind[i]:
                continue
            size = _wkb_size(&ch, i)
            if (ch.n_points[i] > UINT32_MAX
                    or ch.n_points[i] > PY_SSIZE_T_MAX // 16 - points
                    or size > PY_SSIZE_T_MAX - wkb_bytes):
                too_long = True
                break
            n_kept += 1
            points += ch.n_points[i]
            wkb_bytes += size
    if too_long:
        raise OverflowError("the geometries hold more points than their arrays can")
    kept = np.empty(n_kept, dtype=np.int64)
    wkb_off = np.zeros(n_kept + 1, dtype=np.int64)
    wkb = np.zeros(max(wkb_bytes, 1), dtype=np.uint8)
    coords = np.empty((max(points, 1), 2), dtype=np.float64)
    coords_off = np.zeros(n_kept + 1, dtype=np.int64)
    check = np.zeros(n_kept, dtype=np.uint8)
    cdef int64_t* kp = _int64_ptr(kept)
    cdef int64_t* wo = _int64_ptr(wkb_off)
    cdef uint8_t* wb = _uint8_ptr(wkb)
    cdef double* xy = _double_ptr(coords)
    cdef int64_t* co = _int64_ptr(coords_off)
    cdef uint8_t* ck = _uint8_ptr(check)
    cdef Py_ssize_t written, at = 0, wat = 0, pat = 0
    cdef int types = 0
    cdef double minx = 0, miny = 0, maxx = 0, maxy = 0, x, y
    cdef bint first = True
    # 3. The WKB, points and bounds of each row with a geometry; a ring whose orientation
    # needs GEOS keeps a zero-filled WKB of its size, marked in ``check``.
    with nogil:
        for i in range(n):
            if not ch.kind[i]:
                continue
            written = _write_wkb(&ch, i, wb + wat, xy + 2 * pat)
            if written < 0:
                ck[at] = 1
                written = _wkb_size(&ch, i)
            types |= ch.kind[i]
            for j in range(ch.n_points[i]):
                x = xy[2 * (pat + j)]
                y = xy[2 * (pat + j) + 1]
                if first:
                    minx = maxx = x
                    miny = maxy = y
                    first = False
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
            kp[at] = i
            wat += written
            pat += ch.n_points[i]
            at += 1
            wo[at] = wat
            co[at] = pat
    return {
        "kept": kept,
        "wkb_off": wkb_off,
        "wkb": wkb[:wkb_bytes],
        "types": types,
        "bounds": (minx, miny, maxx, maxy),
        "coords": coords[:points],
        "coords_off": coords_off,
        "check": np.flatnonzero(check),
        "columns": _tag_columns(&ch, kept),
        "json": _json_column(&ch, kept) if json else None,
    }
