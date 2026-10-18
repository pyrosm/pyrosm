# cython: boundscheck=False, wraparound=False
"""The ways of a threaded read, column by column, without the GIL.

A way store (:class:`~pyrosm.engine.ways.ElementStore`) holds the candidate ways of a read
as flat arrays: ``ids``, the node refs (``refs`` with ``refs_off``), the metadata, and the
tags as ids into one string buffer (``pair_key`` / ``pair_val`` with ``pairs_off``, the
strings in ``str_data`` at ``str_off``). The functions here select ways, by the value filter
of a read and by a bounding box, releasing the GIL for the work.

The rules are those of the reader's Python path (``engine.collect`` and
``data_filter.pyx``). A way the rules here do not cover -- a string that is not valid
UTF-8, a value ``str.strip`` may change beyond ASCII -- is left to that path ("check").
"""

import numpy as np

from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t, uintptr_t
from libc.string cimport memcmp, memcpy
from libc.stdlib cimport free, malloc

from pyrosm._pointers cimport _int64_ptr, _uint8_ptr
from pyrosm.int64_hash cimport Int64Set

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
    # a string's column as a key, when it is not a column index
    C_LEFTOVER = -1
    C_SPECIAL = -2

FILTER_VALUES = C_FILTER_VALUES
FILTER_MULTI = C_FILTER_MULTI
FILTER_ACCESS = C_FILTER_ACCESS
FILTER_ANY = C_FILTER_ANY
KEEP = C_KEEP
CHECK = C_CHECK


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
