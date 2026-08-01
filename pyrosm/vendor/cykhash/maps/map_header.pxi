"""
Template for maps

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


include "map_init.pxi"

cdef extern from *:

    ctypedef struct kh_int64toint64map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        int64_t *vals  

    kh_int64toint64map_t* kh_init_int64toint64map() nogil
    void kh_destroy_int64toint64map(kh_int64toint64map_t*) nogil
    void kh_clear_int64toint64map(kh_int64toint64map_t*) nogil
    khint_t kh_get_int64toint64map(kh_int64toint64map_t*, int64_t) nogil
    void kh_resize_int64toint64map(kh_int64toint64map_t*, khint_t) nogil
    khint_t kh_put_int64toint64map(kh_int64toint64map_t*, int64_t, int* result) nogil
    void kh_del_int64toint64map(kh_int64toint64map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int64toint64map "kh_exist" (kh_int64toint64map_t*, khint_t) nogil

cdef class Int64toInt64Map:
    cdef kh_int64toint64map_t *table
    cdef bint for_int

    cdef bint contains(self, int64_t key) except *
    cdef Int64toInt64MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, int64_t key, int64_t value) except *
    cpdef int64_t cget(self, int64_t key) except *
    cpdef void discard(self, int64_t key) except *
    

cdef struct int64toint64_key_val_pair:
    int64_t key
    int64_t val


cdef class Int64toInt64MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Int64toInt64Map  parent

    cdef bint has_next(self) except *
    cdef int64toint64_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Int64toInt64MapView:
    cdef Int64toInt64Map  parent
    cdef int       view_type

    cdef Int64toInt64MapIterator get_iter(self)


cpdef Int64toInt64Map Int64toInt64Map_from_buffers(int64_t[:] keys, int64_t[:] vals, double size_hint=*)

cpdef size_t Int64toInt64Map_to(Int64toInt64Map map, int64_t[:] keys, int64_t[:] vals, bint stop_at_unknown=*, int64_t default_value=*) except *



# other help functions:
cpdef void swap_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *
cpdef Int64toInt64Map copy_int64toint64map(Int64toInt64Map s)
cpdef bint are_equal_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *
cpdef void update_int64toint64map(Int64toInt64Map a, Int64toInt64Map b) except *

cdef extern from *:

    ctypedef struct kh_int64tofloat64map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        float64_t *vals  

    kh_int64tofloat64map_t* kh_init_int64tofloat64map() nogil
    void kh_destroy_int64tofloat64map(kh_int64tofloat64map_t*) nogil
    void kh_clear_int64tofloat64map(kh_int64tofloat64map_t*) nogil
    khint_t kh_get_int64tofloat64map(kh_int64tofloat64map_t*, int64_t) nogil
    void kh_resize_int64tofloat64map(kh_int64tofloat64map_t*, khint_t) nogil
    khint_t kh_put_int64tofloat64map(kh_int64tofloat64map_t*, int64_t, int* result) nogil
    void kh_del_int64tofloat64map(kh_int64tofloat64map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int64tofloat64map "kh_exist" (kh_int64tofloat64map_t*, khint_t) nogil

cdef class Int64toFloat64Map:
    cdef kh_int64tofloat64map_t *table
    cdef bint for_int

    cdef bint contains(self, int64_t key) except *
    cdef Int64toFloat64MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, int64_t key, float64_t value) except *
    cpdef float64_t cget(self, int64_t key) except *
    cpdef void discard(self, int64_t key) except *
    

cdef struct int64tofloat64_key_val_pair:
    int64_t key
    float64_t val


cdef class Int64toFloat64MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Int64toFloat64Map  parent

    cdef bint has_next(self) except *
    cdef int64tofloat64_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Int64toFloat64MapView:
    cdef Int64toFloat64Map  parent
    cdef int       view_type

    cdef Int64toFloat64MapIterator get_iter(self)


cpdef Int64toFloat64Map Int64toFloat64Map_from_buffers(int64_t[:] keys, float64_t[:] vals, double size_hint=*)

cpdef size_t Int64toFloat64Map_to(Int64toFloat64Map map, int64_t[:] keys, float64_t[:] vals, bint stop_at_unknown=*, float64_t default_value=*) except *



# other help functions:
cpdef void swap_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *
cpdef Int64toFloat64Map copy_int64tofloat64map(Int64toFloat64Map s)
cpdef bint are_equal_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *
cpdef void update_int64tofloat64map(Int64toFloat64Map a, Int64toFloat64Map b) except *

cdef extern from *:

    ctypedef struct kh_float64toint64map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float64_t *keys
        int64_t *vals  

    kh_float64toint64map_t* kh_init_float64toint64map() nogil
    void kh_destroy_float64toint64map(kh_float64toint64map_t*) nogil
    void kh_clear_float64toint64map(kh_float64toint64map_t*) nogil
    khint_t kh_get_float64toint64map(kh_float64toint64map_t*, float64_t) nogil
    void kh_resize_float64toint64map(kh_float64toint64map_t*, khint_t) nogil
    khint_t kh_put_float64toint64map(kh_float64toint64map_t*, float64_t, int* result) nogil
    void kh_del_float64toint64map(kh_float64toint64map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float64toint64map "kh_exist" (kh_float64toint64map_t*, khint_t) nogil

cdef class Float64toInt64Map:
    cdef kh_float64toint64map_t *table
    cdef bint for_int

    cdef bint contains(self, float64_t key) except *
    cdef Float64toInt64MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, float64_t key, int64_t value) except *
    cpdef int64_t cget(self, float64_t key) except *
    cpdef void discard(self, float64_t key) except *
    

cdef struct float64toint64_key_val_pair:
    float64_t key
    int64_t val


cdef class Float64toInt64MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Float64toInt64Map  parent

    cdef bint has_next(self) except *
    cdef float64toint64_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Float64toInt64MapView:
    cdef Float64toInt64Map  parent
    cdef int       view_type

    cdef Float64toInt64MapIterator get_iter(self)


cpdef Float64toInt64Map Float64toInt64Map_from_buffers(float64_t[:] keys, int64_t[:] vals, double size_hint=*)

cpdef size_t Float64toInt64Map_to(Float64toInt64Map map, float64_t[:] keys, int64_t[:] vals, bint stop_at_unknown=*, int64_t default_value=*) except *



# other help functions:
cpdef void swap_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *
cpdef Float64toInt64Map copy_float64toint64map(Float64toInt64Map s)
cpdef bint are_equal_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *
cpdef void update_float64toint64map(Float64toInt64Map a, Float64toInt64Map b) except *

cdef extern from *:

    ctypedef struct kh_float64tofloat64map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float64_t *keys
        float64_t *vals  

    kh_float64tofloat64map_t* kh_init_float64tofloat64map() nogil
    void kh_destroy_float64tofloat64map(kh_float64tofloat64map_t*) nogil
    void kh_clear_float64tofloat64map(kh_float64tofloat64map_t*) nogil
    khint_t kh_get_float64tofloat64map(kh_float64tofloat64map_t*, float64_t) nogil
    void kh_resize_float64tofloat64map(kh_float64tofloat64map_t*, khint_t) nogil
    khint_t kh_put_float64tofloat64map(kh_float64tofloat64map_t*, float64_t, int* result) nogil
    void kh_del_float64tofloat64map(kh_float64tofloat64map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float64tofloat64map "kh_exist" (kh_float64tofloat64map_t*, khint_t) nogil

cdef class Float64toFloat64Map:
    cdef kh_float64tofloat64map_t *table
    cdef bint for_int

    cdef bint contains(self, float64_t key) except *
    cdef Float64toFloat64MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, float64_t key, float64_t value) except *
    cpdef float64_t cget(self, float64_t key) except *
    cpdef void discard(self, float64_t key) except *
    

cdef struct float64tofloat64_key_val_pair:
    float64_t key
    float64_t val


cdef class Float64toFloat64MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Float64toFloat64Map  parent

    cdef bint has_next(self) except *
    cdef float64tofloat64_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Float64toFloat64MapView:
    cdef Float64toFloat64Map  parent
    cdef int       view_type

    cdef Float64toFloat64MapIterator get_iter(self)


cpdef Float64toFloat64Map Float64toFloat64Map_from_buffers(float64_t[:] keys, float64_t[:] vals, double size_hint=*)

cpdef size_t Float64toFloat64Map_to(Float64toFloat64Map map, float64_t[:] keys, float64_t[:] vals, bint stop_at_unknown=*, float64_t default_value=*) except *



# other help functions:
cpdef void swap_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *
cpdef Float64toFloat64Map copy_float64tofloat64map(Float64toFloat64Map s)
cpdef bint are_equal_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *
cpdef void update_float64tofloat64map(Float64toFloat64Map a, Float64toFloat64Map b) except *

cdef extern from *:

    ctypedef struct kh_int32toint32map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int32_t *keys
        int32_t *vals  

    kh_int32toint32map_t* kh_init_int32toint32map() nogil
    void kh_destroy_int32toint32map(kh_int32toint32map_t*) nogil
    void kh_clear_int32toint32map(kh_int32toint32map_t*) nogil
    khint_t kh_get_int32toint32map(kh_int32toint32map_t*, int32_t) nogil
    void kh_resize_int32toint32map(kh_int32toint32map_t*, khint_t) nogil
    khint_t kh_put_int32toint32map(kh_int32toint32map_t*, int32_t, int* result) nogil
    void kh_del_int32toint32map(kh_int32toint32map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int32toint32map "kh_exist" (kh_int32toint32map_t*, khint_t) nogil

cdef class Int32toInt32Map:
    cdef kh_int32toint32map_t *table
    cdef bint for_int

    cdef bint contains(self, int32_t key) except *
    cdef Int32toInt32MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, int32_t key, int32_t value) except *
    cpdef int32_t cget(self, int32_t key) except *
    cpdef void discard(self, int32_t key) except *
    

cdef struct int32toint32_key_val_pair:
    int32_t key
    int32_t val


cdef class Int32toInt32MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Int32toInt32Map  parent

    cdef bint has_next(self) except *
    cdef int32toint32_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Int32toInt32MapView:
    cdef Int32toInt32Map  parent
    cdef int       view_type

    cdef Int32toInt32MapIterator get_iter(self)


cpdef Int32toInt32Map Int32toInt32Map_from_buffers(int32_t[:] keys, int32_t[:] vals, double size_hint=*)

cpdef size_t Int32toInt32Map_to(Int32toInt32Map map, int32_t[:] keys, int32_t[:] vals, bint stop_at_unknown=*, int32_t default_value=*) except *



# other help functions:
cpdef void swap_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *
cpdef Int32toInt32Map copy_int32toint32map(Int32toInt32Map s)
cpdef bint are_equal_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *
cpdef void update_int32toint32map(Int32toInt32Map a, Int32toInt32Map b) except *

cdef extern from *:

    ctypedef struct kh_int32tofloat32map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int32_t *keys
        float32_t *vals  

    kh_int32tofloat32map_t* kh_init_int32tofloat32map() nogil
    void kh_destroy_int32tofloat32map(kh_int32tofloat32map_t*) nogil
    void kh_clear_int32tofloat32map(kh_int32tofloat32map_t*) nogil
    khint_t kh_get_int32tofloat32map(kh_int32tofloat32map_t*, int32_t) nogil
    void kh_resize_int32tofloat32map(kh_int32tofloat32map_t*, khint_t) nogil
    khint_t kh_put_int32tofloat32map(kh_int32tofloat32map_t*, int32_t, int* result) nogil
    void kh_del_int32tofloat32map(kh_int32tofloat32map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_int32tofloat32map "kh_exist" (kh_int32tofloat32map_t*, khint_t) nogil

cdef class Int32toFloat32Map:
    cdef kh_int32tofloat32map_t *table
    cdef bint for_int

    cdef bint contains(self, int32_t key) except *
    cdef Int32toFloat32MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, int32_t key, float32_t value) except *
    cpdef float32_t cget(self, int32_t key) except *
    cpdef void discard(self, int32_t key) except *
    

cdef struct int32tofloat32_key_val_pair:
    int32_t key
    float32_t val


cdef class Int32toFloat32MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Int32toFloat32Map  parent

    cdef bint has_next(self) except *
    cdef int32tofloat32_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Int32toFloat32MapView:
    cdef Int32toFloat32Map  parent
    cdef int       view_type

    cdef Int32toFloat32MapIterator get_iter(self)


cpdef Int32toFloat32Map Int32toFloat32Map_from_buffers(int32_t[:] keys, float32_t[:] vals, double size_hint=*)

cpdef size_t Int32toFloat32Map_to(Int32toFloat32Map map, int32_t[:] keys, float32_t[:] vals, bint stop_at_unknown=*, float32_t default_value=*) except *



# other help functions:
cpdef void swap_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *
cpdef Int32toFloat32Map copy_int32tofloat32map(Int32toFloat32Map s)
cpdef bint are_equal_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *
cpdef void update_int32tofloat32map(Int32toFloat32Map a, Int32toFloat32Map b) except *

cdef extern from *:

    ctypedef struct kh_float32toint32map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float32_t *keys
        int32_t *vals  

    kh_float32toint32map_t* kh_init_float32toint32map() nogil
    void kh_destroy_float32toint32map(kh_float32toint32map_t*) nogil
    void kh_clear_float32toint32map(kh_float32toint32map_t*) nogil
    khint_t kh_get_float32toint32map(kh_float32toint32map_t*, float32_t) nogil
    void kh_resize_float32toint32map(kh_float32toint32map_t*, khint_t) nogil
    khint_t kh_put_float32toint32map(kh_float32toint32map_t*, float32_t, int* result) nogil
    void kh_del_float32toint32map(kh_float32toint32map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float32toint32map "kh_exist" (kh_float32toint32map_t*, khint_t) nogil

cdef class Float32toInt32Map:
    cdef kh_float32toint32map_t *table
    cdef bint for_int

    cdef bint contains(self, float32_t key) except *
    cdef Float32toInt32MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, float32_t key, int32_t value) except *
    cpdef int32_t cget(self, float32_t key) except *
    cpdef void discard(self, float32_t key) except *
    

cdef struct float32toint32_key_val_pair:
    float32_t key
    int32_t val


cdef class Float32toInt32MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Float32toInt32Map  parent

    cdef bint has_next(self) except *
    cdef float32toint32_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Float32toInt32MapView:
    cdef Float32toInt32Map  parent
    cdef int       view_type

    cdef Float32toInt32MapIterator get_iter(self)


cpdef Float32toInt32Map Float32toInt32Map_from_buffers(float32_t[:] keys, int32_t[:] vals, double size_hint=*)

cpdef size_t Float32toInt32Map_to(Float32toInt32Map map, float32_t[:] keys, int32_t[:] vals, bint stop_at_unknown=*, int32_t default_value=*) except *



# other help functions:
cpdef void swap_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *
cpdef Float32toInt32Map copy_float32toint32map(Float32toInt32Map s)
cpdef bint are_equal_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *
cpdef void update_float32toint32map(Float32toInt32Map a, Float32toInt32Map b) except *

cdef extern from *:

    ctypedef struct kh_float32tofloat32map_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float32_t *keys
        float32_t *vals  

    kh_float32tofloat32map_t* kh_init_float32tofloat32map() nogil
    void kh_destroy_float32tofloat32map(kh_float32tofloat32map_t*) nogil
    void kh_clear_float32tofloat32map(kh_float32tofloat32map_t*) nogil
    khint_t kh_get_float32tofloat32map(kh_float32tofloat32map_t*, float32_t) nogil
    void kh_resize_float32tofloat32map(kh_float32tofloat32map_t*, khint_t) nogil
    khint_t kh_put_float32tofloat32map(kh_float32tofloat32map_t*, float32_t, int* result) nogil
    void kh_del_float32tofloat32map(kh_float32tofloat32map_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_float32tofloat32map "kh_exist" (kh_float32tofloat32map_t*, khint_t) nogil

cdef class Float32toFloat32Map:
    cdef kh_float32tofloat32map_t *table
    cdef bint for_int

    cdef bint contains(self, float32_t key) except *
    cdef Float32toFloat32MapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, float32_t key, float32_t value) except *
    cpdef float32_t cget(self, float32_t key) except *
    cpdef void discard(self, float32_t key) except *
    

cdef struct float32tofloat32_key_val_pair:
    float32_t key
    float32_t val


cdef class Float32toFloat32MapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef Float32toFloat32Map  parent

    cdef bint has_next(self) except *
    cdef float32tofloat32_key_val_pair next(self) except *
    cdef void __move(self) except *


cdef class Float32toFloat32MapView:
    cdef Float32toFloat32Map  parent
    cdef int       view_type

    cdef Float32toFloat32MapIterator get_iter(self)


cpdef Float32toFloat32Map Float32toFloat32Map_from_buffers(float32_t[:] keys, float32_t[:] vals, double size_hint=*)

cpdef size_t Float32toFloat32Map_to(Float32toFloat32Map map, float32_t[:] keys, float32_t[:] vals, bint stop_at_unknown=*, float32_t default_value=*) except *



# other help functions:
cpdef void swap_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *
cpdef Float32toFloat32Map copy_float32tofloat32map(Float32toFloat32Map s)
cpdef bint are_equal_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *
cpdef void update_float32tofloat32map(Float32toFloat32Map a, Float32toFloat32Map b) except *


##TODO: unify with others


cdef extern from *:

    ctypedef struct kh_pyobjectmap_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        pyobject_t *keys
        pyobject_t *vals  

    kh_pyobjectmap_t* kh_init_pyobjectmap() nogil
    void kh_destroy_pyobjectmap(kh_pyobjectmap_t*) nogil
    void kh_clear_pyobjectmap(kh_pyobjectmap_t*) nogil
    khint_t kh_get_pyobjectmap(kh_pyobjectmap_t*, pyobject_t) nogil
    void kh_resize_pyobjectmap(kh_pyobjectmap_t*, khint_t) nogil
    khint_t kh_put_pyobjectmap(kh_pyobjectmap_t*, pyobject_t, int* result) nogil
    void kh_del_pyobjectmap(kh_pyobjectmap_t*, khint_t) nogil

    #specializing "kh_exist"-macro 
    bint kh_exist_pyobjectmap "kh_exist" (kh_pyobjectmap_t*, khint_t) nogil


cdef class PyObjectMap:
    cdef kh_pyobjectmap_t *table

    cdef bint contains(self, pyobject_t key) except *
    cdef PyObjectMapIterator get_iter(self, int view_type)
    cdef khint_t size(self) 
    cpdef void cput(self, object key, object value) except *
    cpdef object cget(self, object key)
    cpdef void discard(self, object key) except *
    

cdef struct pyobject_key_val_pair:
    pyobject_t key
    pyobject_t val


cdef class PyObjectMapIterator:
    cdef khint_t   it
    cdef int       view_type
    cdef PyObjectMap  parent

    cdef bint has_next(self) except *
    cdef pyobject_key_val_pair next(self) except *
    cdef void __move(self) except *

cdef class PyObjectMapView:
    cdef PyObjectMap  parent
    cdef int       view_type

    cdef PyObjectMapIterator get_iter(self)

cpdef PyObjectMap PyObjectMap_from_buffers(object[:] keys, object[:] vals, double size_hint=*)

cpdef size_t PyObjectMap_to(PyObjectMap map, object[:] keys, object[:] vals, bint stop_at_unknown=*, object default_value=*) except *

# other help functions:
cpdef void swap_pyobjectmap(PyObjectMap a, PyObjectMap b) except *
cpdef PyObjectMap copy_pyobjectmap(PyObjectMap s)
cpdef bint are_equal_pyobjectmap(PyObjectMap a, PyObjectMap b) except *
cpdef void update_pyobjectmap(PyObjectMap a, PyObjectMap b) except *





