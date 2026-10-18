# cython: boundscheck=False, wraparound=False
"""The ways of a threaded read, column by column, without the GIL.

A way store (:class:`~pyrosm.engine.ways.ElementStore`) holds the candidate ways of a read
as flat arrays: ``ids``, the node refs (``refs`` with ``refs_off``), the metadata, and the
tags as ids into one string buffer (``pair_key`` / ``pair_val`` with ``pairs_off``, the
strings in ``str_data`` at ``str_off``). The functions here select ways, releasing the GIL
for the work.
"""

import numpy as np

from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport int64_t, uint8_t, uintptr_t
from libc.string cimport memcpy

from pyrosm._pointers cimport _int64_ptr, _uint8_ptr
from pyrosm.int64_hash cimport Int64Set


def _contiguous(a, dtype):
    """``a`` as a contiguous one-dimensional array of ``dtype``; ``ValueError`` for an array
    of another shape (the loops here index its values one after another)."""
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim != 1:
        raise ValueError("arrays must be one-dimensional")
    return a


def _check_offsets(off, Py_ssize_t n_values):
    """Raise ``ValueError`` unless the CSR offsets ``off`` rise from 0 to ``n_values``."""
    if len(off) == 0 or off[0] != 0 or off[-1] != n_values or (off[1:] < off[:-1]).any():
        raise ValueError("offsets must rise from 0 to the number of values")


# --- selecting ways ---------------------------------------------------------------------


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
