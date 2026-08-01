"""
Template for maps

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""



###################  INTS:

cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions
    #define CYKHASH_MAP_INIT_INT64(name, khval_t)		\
        KHASH_INIT(name, int64_t, khval_t, 1, cykh_float64_hash_func, cykh_int64_hash_equal)
  
    CYKHASH_MAP_INIT_INT64(int64toint64map, int64_t)
    CYKHASH_MAP_INIT_INT64(int64tofloat64map, float64_t)
    """
    pass

cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions
    #define CYKHASH_MAP_INIT_INT32(name, khval_t)		\
        KHASH_INIT(name, int32_t, khval_t, 1, cykh_float32_hash_func, cykh_int32_hash_equal)
  
    CYKHASH_MAP_INIT_INT32(int32toint32map, int32_t)
    CYKHASH_MAP_INIT_INT32(int32tofloat32map, float32_t)
    """
    pass




################   FLOATS:


# see float_utils.pxi for definitions

cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions
    #define CYKHASH_MAP_INIT_FLOAT64(name, khval_t)								\
	    KHASH_INIT(name, khfloat64_t, khval_t, 1, cykh_float64_hash_func, cykh_float64_hash_equal)

    CYKHASH_MAP_INIT_FLOAT64(float64toint64map, int64_t)
    CYKHASH_MAP_INIT_FLOAT64(float64tofloat64map, float64_t)

    """
    pass


# see float_utils.pxi for definitions

cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions
    #define CYKHASH_MAP_INIT_FLOAT32(name, khval_t)								\
	    KHASH_INIT(name, khfloat32_t, khval_t, 1, cykh_float32_hash_func, cykh_float32_hash_equal)

    CYKHASH_MAP_INIT_FLOAT32(float32toint32map, int32_t)
    CYKHASH_MAP_INIT_FLOAT32(float32tofloat32map, float32_t)

    """
    pass



################   OBJECT:
cdef extern from *:
    """
    // preprocessor creates needed struct-type and all function definitions

    // map with keys of type pyobject -> result pyobject
    #define CYKHASH_MAP_INIT_PYOBJECT(name, khval_t)										\
	    KHASH_INIT(name, khpyobject_t, khval_t, 1, cykh_pyobject_hash_func, cykh_pyobject_hash_equal)

    //preprocessor creates needed struct-type and all function definitions 
    //set with keys of type pyobject -> resulting typename: kh_pyobjectmap_t;
    CYKHASH_MAP_INIT_PYOBJECT(pyobjectmap, pyobject_t)

    """
    pass
