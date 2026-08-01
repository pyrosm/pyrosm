
"""
Template for sets

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions   
    KHASH_INIT(int64set,         int64_t, char, 0,    cykh_int64_hash_func,    cykh_int64_hash_equal)
    KHASH_INIT(int32set,         int32_t, char, 0,    cykh_int32_hash_func,    cykh_int32_hash_equal)
    KHASH_INIT(float64set,   khfloat64_t, char, 0,  cykh_float64_hash_func,  cykh_float64_hash_equal)
    KHASH_INIT(float32set,   khfloat32_t, char, 0,  cykh_float32_hash_func,  cykh_float32_hash_equal)
    KHASH_INIT(pyobjectset, khpyobject_t, char, 0, cykh_pyobject_hash_func, cykh_pyobject_hash_equal) 
    """
    pass

