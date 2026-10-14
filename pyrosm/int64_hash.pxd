from libc.stdint cimport int64_t, uint32_t


cdef extern from "int64_hash.h" nogil:
    ctypedef uint32_t khint_t

    ctypedef struct kh_i64set_t:
        khint_t n_buckets
        khint_t size
        int64_t *keys

    ctypedef struct kh_i64map_t:
        khint_t n_buckets
        khint_t size
        int64_t *vals

    kh_i64set_t *kh_init_i64set()
    void kh_destroy_i64set(kh_i64set_t *h)
    int kh_resize_i64set(kh_i64set_t *h, khint_t new_n_buckets)
    khint_t kh_put_i64set(kh_i64set_t *h, int64_t key, int *ret)
    khint_t kh_get_i64set(const kh_i64set_t *h, int64_t key)
    bint kh_exist(const kh_i64set_t *h, khint_t slot)

    kh_i64map_t *kh_init_i64map()
    void kh_destroy_i64map(kh_i64map_t *h)
    int kh_resize_i64map(kh_i64map_t *h, khint_t new_n_buckets)
    khint_t kh_put_i64map(kh_i64map_t *h, int64_t key, int *ret)
    khint_t kh_get_i64map(const kh_i64map_t *h, int64_t key)


cdef class Int64Set:
    cdef kh_i64set_t *table
    cdef bint contains(self, int64_t key) noexcept nogil


cdef class Int64Map:
    cdef kh_i64map_t *table
    cdef bint contains(self, int64_t key) noexcept nogil
    cdef int64_t get(self, int64_t key, int64_t default) noexcept nogil
