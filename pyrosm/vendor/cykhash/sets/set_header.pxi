
"""
Template for sets

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

include "set_init.pxi"


cdef extern from *:

    ctypedef struct kh_int64set_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        #size_t *vals  //dummy

    kh_int64set_t* kh_init_int64set() nogil
    void kh_destroy_int64set(kh_int64set_t*) nogil
    void kh_clear_int64set(kh_int64set_t*) nogil
    khint_t kh_get_int64set(kh_int64set_t*, int64_t) nogil
    void kh_resize_int64set(kh_int64set_t*, khint_t) nogil
    khint_t kh_put_int64set(kh_int64set_t*, int64_t, int*) nogil
    void kh_del_int64set(kh_int64set_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int64set "kh_exist" (kh_int64set_t*, khint_t) nogil


cdef class Int64Set:
    cdef kh_int64set_t *table

    cdef bint contains(self, int64_t key) except *
    cdef Int64SetIterator get_iter(self)
    cdef khint_t size(self) 
    cpdef void add(self, int64_t key) except *
    cpdef void discard(self, int64_t key) except *
    

cdef class Int64SetIterator:
    cdef khint_t   it
    cdef Int64Set  parent

    cdef bint has_next(self) except *
    cdef int64_t next(self) except *
    cdef void __move(self) except *


cpdef Int64Set Int64Set_from_buffer(int64_t[:] buf, double size_hint=*)


from libc.stdint cimport  uint8_t
cpdef void isin_int64(int64_t[:] query, Int64Set db, uint8_t[:] result) except *

cpdef bint all_int64(int64_t[:] query, Int64Set db) except *
cpdef bint all_int64_from_iter(object query, Int64Set db) except *

cpdef bint none_int64(int64_t[:] query, Int64Set db) except *
cpdef bint none_int64_from_iter(object query, Int64Set db) except *

cpdef bint any_int64(int64_t[:] query, Int64Set db) except *
cpdef bint any_int64_from_iter(object query, Int64Set db) except *

cpdef size_t count_if_int64(int64_t[:] query, Int64Set db) except *
cpdef size_t count_if_int64_from_iter(object query, Int64Set db) except *

cpdef void swap_int64(Int64Set a, Int64Set b) except *

# for drop-in replacements:
cpdef bint aredisjoint_int64(Int64Set a, Int64Set b) except *
cpdef bint issubset_int64(Int64Set s, Int64Set sub) except *
cpdef Int64Set copy_int64(Int64Set s)
cpdef void update_int64(Int64Set s, Int64Set other) except *
cpdef Int64Set intersect_int64(Int64Set a, Int64Set b)
cpdef Int64Set difference_int64(Int64Set a, Int64Set b)
cpdef Int64Set symmetric_difference_int64(Int64Set a, Int64Set b)



cdef extern from *:

    ctypedef struct kh_float64set_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float64_t *keys
        #size_t *vals  //dummy

    kh_float64set_t* kh_init_float64set() nogil
    void kh_destroy_float64set(kh_float64set_t*) nogil
    void kh_clear_float64set(kh_float64set_t*) nogil
    khint_t kh_get_float64set(kh_float64set_t*, float64_t) nogil
    void kh_resize_float64set(kh_float64set_t*, khint_t) nogil
    khint_t kh_put_float64set(kh_float64set_t*, float64_t, int*) nogil
    void kh_del_float64set(kh_float64set_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float64set "kh_exist" (kh_float64set_t*, khint_t) nogil


cdef class Float64Set:
    cdef kh_float64set_t *table

    cdef bint contains(self, float64_t key) except *
    cdef Float64SetIterator get_iter(self)
    cdef khint_t size(self) 
    cpdef void add(self, float64_t key) except *
    cpdef void discard(self, float64_t key) except *
    

cdef class Float64SetIterator:
    cdef khint_t   it
    cdef Float64Set  parent

    cdef bint has_next(self) except *
    cdef float64_t next(self) except *
    cdef void __move(self) except *


cpdef Float64Set Float64Set_from_buffer(float64_t[:] buf, double size_hint=*)


from libc.stdint cimport  uint8_t
cpdef void isin_float64(float64_t[:] query, Float64Set db, uint8_t[:] result) except *

cpdef bint all_float64(float64_t[:] query, Float64Set db) except *
cpdef bint all_float64_from_iter(object query, Float64Set db) except *

cpdef bint none_float64(float64_t[:] query, Float64Set db) except *
cpdef bint none_float64_from_iter(object query, Float64Set db) except *

cpdef bint any_float64(float64_t[:] query, Float64Set db) except *
cpdef bint any_float64_from_iter(object query, Float64Set db) except *

cpdef size_t count_if_float64(float64_t[:] query, Float64Set db) except *
cpdef size_t count_if_float64_from_iter(object query, Float64Set db) except *

cpdef void swap_float64(Float64Set a, Float64Set b) except *

# for drop-in replacements:
cpdef bint aredisjoint_float64(Float64Set a, Float64Set b) except *
cpdef bint issubset_float64(Float64Set s, Float64Set sub) except *
cpdef Float64Set copy_float64(Float64Set s)
cpdef void update_float64(Float64Set s, Float64Set other) except *
cpdef Float64Set intersect_float64(Float64Set a, Float64Set b)
cpdef Float64Set difference_float64(Float64Set a, Float64Set b)
cpdef Float64Set symmetric_difference_float64(Float64Set a, Float64Set b)



cdef extern from *:

    ctypedef struct kh_int32set_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int32_t *keys
        #size_t *vals  //dummy

    kh_int32set_t* kh_init_int32set() nogil
    void kh_destroy_int32set(kh_int32set_t*) nogil
    void kh_clear_int32set(kh_int32set_t*) nogil
    khint_t kh_get_int32set(kh_int32set_t*, int32_t) nogil
    void kh_resize_int32set(kh_int32set_t*, khint_t) nogil
    khint_t kh_put_int32set(kh_int32set_t*, int32_t, int*) nogil
    void kh_del_int32set(kh_int32set_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int32set "kh_exist" (kh_int32set_t*, khint_t) nogil


cdef class Int32Set:
    cdef kh_int32set_t *table

    cdef bint contains(self, int32_t key) except *
    cdef Int32SetIterator get_iter(self)
    cdef khint_t size(self) 
    cpdef void add(self, int32_t key) except *
    cpdef void discard(self, int32_t key) except *
    

cdef class Int32SetIterator:
    cdef khint_t   it
    cdef Int32Set  parent

    cdef bint has_next(self) except *
    cdef int32_t next(self) except *
    cdef void __move(self) except *


cpdef Int32Set Int32Set_from_buffer(int32_t[:] buf, double size_hint=*)


from libc.stdint cimport  uint8_t
cpdef void isin_int32(int32_t[:] query, Int32Set db, uint8_t[:] result) except *

cpdef bint all_int32(int32_t[:] query, Int32Set db) except *
cpdef bint all_int32_from_iter(object query, Int32Set db) except *

cpdef bint none_int32(int32_t[:] query, Int32Set db) except *
cpdef bint none_int32_from_iter(object query, Int32Set db) except *

cpdef bint any_int32(int32_t[:] query, Int32Set db) except *
cpdef bint any_int32_from_iter(object query, Int32Set db) except *

cpdef size_t count_if_int32(int32_t[:] query, Int32Set db) except *
cpdef size_t count_if_int32_from_iter(object query, Int32Set db) except *

cpdef void swap_int32(Int32Set a, Int32Set b) except *

# for drop-in replacements:
cpdef bint aredisjoint_int32(Int32Set a, Int32Set b) except *
cpdef bint issubset_int32(Int32Set s, Int32Set sub) except *
cpdef Int32Set copy_int32(Int32Set s)
cpdef void update_int32(Int32Set s, Int32Set other) except *
cpdef Int32Set intersect_int32(Int32Set a, Int32Set b)
cpdef Int32Set difference_int32(Int32Set a, Int32Set b)
cpdef Int32Set symmetric_difference_int32(Int32Set a, Int32Set b)



cdef extern from *:

    ctypedef struct kh_float32set_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float32_t *keys
        #size_t *vals  //dummy

    kh_float32set_t* kh_init_float32set() nogil
    void kh_destroy_float32set(kh_float32set_t*) nogil
    void kh_clear_float32set(kh_float32set_t*) nogil
    khint_t kh_get_float32set(kh_float32set_t*, float32_t) nogil
    void kh_resize_float32set(kh_float32set_t*, khint_t) nogil
    khint_t kh_put_float32set(kh_float32set_t*, float32_t, int*) nogil
    void kh_del_float32set(kh_float32set_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float32set "kh_exist" (kh_float32set_t*, khint_t) nogil


cdef class Float32Set:
    cdef kh_float32set_t *table

    cdef bint contains(self, float32_t key) except *
    cdef Float32SetIterator get_iter(self)
    cdef khint_t size(self) 
    cpdef void add(self, float32_t key) except *
    cpdef void discard(self, float32_t key) except *
    

cdef class Float32SetIterator:
    cdef khint_t   it
    cdef Float32Set  parent

    cdef bint has_next(self) except *
    cdef float32_t next(self) except *
    cdef void __move(self) except *


cpdef Float32Set Float32Set_from_buffer(float32_t[:] buf, double size_hint=*)


from libc.stdint cimport  uint8_t
cpdef void isin_float32(float32_t[:] query, Float32Set db, uint8_t[:] result) except *

cpdef bint all_float32(float32_t[:] query, Float32Set db) except *
cpdef bint all_float32_from_iter(object query, Float32Set db) except *

cpdef bint none_float32(float32_t[:] query, Float32Set db) except *
cpdef bint none_float32_from_iter(object query, Float32Set db) except *

cpdef bint any_float32(float32_t[:] query, Float32Set db) except *
cpdef bint any_float32_from_iter(object query, Float32Set db) except *

cpdef size_t count_if_float32(float32_t[:] query, Float32Set db) except *
cpdef size_t count_if_float32_from_iter(object query, Float32Set db) except *

cpdef void swap_float32(Float32Set a, Float32Set b) except *

# for drop-in replacements:
cpdef bint aredisjoint_float32(Float32Set a, Float32Set b) except *
cpdef bint issubset_float32(Float32Set s, Float32Set sub) except *
cpdef Float32Set copy_float32(Float32Set s)
cpdef void update_float32(Float32Set s, Float32Set other) except *
cpdef Float32Set intersect_float32(Float32Set a, Float32Set b)
cpdef Float32Set difference_float32(Float32Set a, Float32Set b)
cpdef Float32Set symmetric_difference_float32(Float32Set a, Float32Set b)



cdef extern from *:

    ctypedef struct kh_pyobjectset_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        pyobject_t *keys
        #size_t *vals  //dummy

    kh_pyobjectset_t* kh_init_pyobjectset() nogil
    void kh_destroy_pyobjectset(kh_pyobjectset_t*) nogil
    void kh_clear_pyobjectset(kh_pyobjectset_t*) nogil
    khint_t kh_get_pyobjectset(kh_pyobjectset_t*, pyobject_t) nogil
    void kh_resize_pyobjectset(kh_pyobjectset_t*, khint_t) nogil
    khint_t kh_put_pyobjectset(kh_pyobjectset_t*, pyobject_t, int*) nogil
    void kh_del_pyobjectset(kh_pyobjectset_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_pyobjectset "kh_exist" (kh_pyobjectset_t*, khint_t) nogil


cdef class PyObjectSet:
    cdef kh_pyobjectset_t *table

    cdef bint contains(self, object key) except *
    cdef PyObjectSetIterator get_iter(self)
    cdef khint_t size(self) 
    cpdef void add(self, object key) except *
    cpdef void discard(self, object key) except *
    

cdef class PyObjectSetIterator:
    cdef khint_t   it
    cdef PyObjectSet  parent

    cdef bint has_next(self) except *
    cdef object next(self)
    cdef void __move(self) except *


cpdef PyObjectSet PyObjectSet_from_buffer(object[:] buf, double size_hint=*)


from libc.stdint cimport  uint8_t
cpdef void isin_pyobject(object[:] query, PyObjectSet db, uint8_t[:] result) except *

cpdef bint all_pyobject(object[:] query, PyObjectSet db) except *
cpdef bint all_pyobject_from_iter(object query, PyObjectSet db) except *

cpdef bint none_pyobject(object[:] query, PyObjectSet db) except *
cpdef bint none_pyobject_from_iter(object query, PyObjectSet db) except *

cpdef bint any_pyobject(object[:] query, PyObjectSet db) except *
cpdef bint any_pyobject_from_iter(object query, PyObjectSet db) except *

cpdef size_t count_if_pyobject(object[:] query, PyObjectSet db) except *
cpdef size_t count_if_pyobject_from_iter(object query, PyObjectSet db) except *

cpdef void swap_pyobject(PyObjectSet a, PyObjectSet b) except *

# for drop-in replacements:
cpdef bint aredisjoint_pyobject(PyObjectSet a, PyObjectSet b) except *
cpdef bint issubset_pyobject(PyObjectSet s, PyObjectSet sub) except *
cpdef PyObjectSet copy_pyobject(PyObjectSet s)
cpdef void update_pyobject(PyObjectSet s, PyObjectSet other) except *
cpdef PyObjectSet intersect_pyobject(PyObjectSet a, PyObjectSet b)
cpdef PyObjectSet difference_pyobject(PyObjectSet a, PyObjectSet b)
cpdef PyObjectSet symmetric_difference_pyobject(PyObjectSet a, PyObjectSet b)

