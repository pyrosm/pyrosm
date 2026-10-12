# cython: boundscheck=False, wraparound=False
"""Hash tables for OSM ids: ``Int64Set`` and ``Int64Map`` (int64 -> int64), on klib's
khash (``khash.h``, MIT licence)."""

from libc.stdint cimport int64_t, uint8_t
import numpy as np

# khash grows a table once it is 77 % full; a table holds at most 2**31 buckets.
cdef double _MAX_LOAD = 0.77
cdef khint_t _MAX_BUCKETS = <khint_t>1 << 31


cdef khint_t _buckets_for(Py_ssize_t n) except 0:
    """The number of buckets that holds ``n`` keys without growing."""
    cdef khint_t buckets = 4
    while <khint_t>(buckets * _MAX_LOAD + 0.5) < n:
        if buckets == _MAX_BUCKETS:
            raise MemoryError("too many ids for one hash table: %d" % n)
        buckets *= 2
    return buckets


cdef inline bint _set_has(kh_i64set_t *table, int64_t key) noexcept nogil:
    return kh_get_i64set(table, key) != table.n_buckets


cdef inline int64_t _map_get(kh_i64map_t *table, int64_t key, int64_t default) noexcept nogil:
    cdef khint_t slot = kh_get_i64map(table, key)
    return default if slot == table.n_buckets else table.vals[slot]


cdef _int64_array(values):
    if values is None:
        return np.empty(0, dtype=np.int64)
    return np.ascontiguousarray(values, dtype=np.int64)


cdef class Int64Set:
    """A set of int64 values, built from an array of them."""

    def __cinit__(self, values=None):
        cdef khint_t buckets = _buckets_for(0 if values is None else len(values))
        cdef const int64_t[::1] keys = _int64_array(values)
        cdef Py_ssize_t i
        cdef int ret
        self.table = kh_init_i64set()
        if self.table is NULL or kh_resize_i64set(self.table, buckets) < 0:
            raise MemoryError()
        # A table sized for its keys does not grow, so kh_put cannot fail.
        with nogil:
            for i in range(keys.shape[0]):
                kh_put_i64set(self.table, keys[i], &ret)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_i64set(self.table)

    cdef bint contains(self, int64_t key) noexcept nogil:
        return _set_has(self.table, key)

    def __contains__(self, int64_t key):
        return self.contains(key)

    def __len__(self):
        return self.table.size

    def isin(self, values):
        """A boolean array: whether each of ``values`` is in the set."""
        cdef const int64_t[::1] keys = _int64_array(values)
        cdef Py_ssize_t i, n = keys.shape[0]
        result = np.empty(n, dtype=np.bool_)
        cdef uint8_t[::1] out = result.view(np.uint8)
        with nogil:
            for i in range(n):
                out[i] = _set_has(self.table, keys[i])
        return result

    def contains_any(self, values):
        """Whether any of ``values`` (an iterable of ints) is in the set."""
        cdef int64_t value
        if values is None:
            return False
        for value in values:
            if self.contains(value):
                return True
        return False


cdef class Int64Map:
    """A map from int64 keys to int64 values, built from an array of each; a key given
    twice maps to its last value."""

    def __cinit__(self, keys=None, values=None):
        cdef khint_t buckets = _buckets_for(0 if keys is None else len(keys))
        cdef const int64_t[::1] k = _int64_array(keys)
        cdef const int64_t[::1] v = _int64_array(values)
        cdef Py_ssize_t i
        cdef int ret
        if v.shape[0] != k.shape[0]:
            raise ValueError("%d keys but %d values" % (k.shape[0], v.shape[0]))
        self.table = kh_init_i64map()
        if self.table is NULL or kh_resize_i64map(self.table, buckets) < 0:
            raise MemoryError()
        # A table sized for its keys does not grow, so kh_put cannot fail.
        with nogil:
            for i in range(k.shape[0]):
                self.table.vals[kh_put_i64map(self.table, k[i], &ret)] = v[i]

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_i64map(self.table)

    cdef bint contains(self, int64_t key) noexcept nogil:
        return kh_get_i64map(self.table, key) != self.table.n_buckets

    cdef int64_t get(self, int64_t key, int64_t default) noexcept nogil:
        return _map_get(self.table, key, default)

    def __contains__(self, int64_t key):
        return self.contains(key)

    def __len__(self):
        return self.table.size

    def get_many(self, keys, int64_t default=-1):
        """An int64 array with the value of each of ``keys``, ``default`` for a key that
        is not in the map."""
        cdef const int64_t[::1] k = _int64_array(keys)
        cdef Py_ssize_t i, n = k.shape[0]
        result = np.empty(n, dtype=np.int64)
        cdef int64_t[::1] out = result
        with nogil:
            for i in range(n):
                out[i] = _map_get(self.table, k[i], default)
        return result
