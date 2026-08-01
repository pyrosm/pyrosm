
include "murmurhash.pxi"


# hash functions from khash.h:
cdef extern from *:
    """
        #ifndef CYKHASH_INTEGRAL_HASH_FUNS_PXI
        #define CYKHASH_INTEGRAL_HASH_FUNS_PXI
            CYKHASH_INLINE uint32_t kh_int32_hash_func(uint32_t key){return key;}
            CYKHASH_INLINE int      kh_int32_hash_equal(uint32_t a, uint32_t b) {return a==b;}

            CYKHASH_INLINE uint32_t kh_int64_hash_func(uint64_t key){
                    return (uint32_t)((key)>>33^(key)^(key)<<11);
            }
            CYKHASH_INLINE int      kh_int64_hash_equal(uint64_t a, uint64_t b) {return a==b;}
        #endif
    """
    pass



# handling floats
cdef extern from *:
    """
        #ifndef CYKHASH_FLOATING_HASH_FUNS_PXI
        #define CYKHASH_FLOATING_HASH_FUNS_PXI
            // correct handling for float64/32 <-> int64/32
            #include <string.h>

            typedef double float64_t;
            typedef float float32_t;

            //don't pun and alias:

            CYKHASH_INLINE uint64_t f64_to_ui64(float64_t val){
                  uint64_t res; 
                  memcpy(&res, &val, sizeof(float64_t)); 
                  return res;
            } 

            CYKHASH_INLINE float64_t ui64_to_f64(uint64_t val){
                  float64_t res; 
                  memcpy(&res, &val, sizeof(float64_t)); 
                  return res;
            }

            CYKHASH_INLINE uint32_t f32_to_ui32(float32_t val){
                  uint32_t res; 
                  memcpy(&res, &val, sizeof(float32_t)); 
                  return res;
            } 

            CYKHASH_INLINE float32_t ui32_to_f32(uint32_t val){
                  float32_t res; 
                  memcpy(&res, &val, sizeof(float32_t)); 
                  return res;
            }

            // HASH AND EQUAL FUNCTIONS:
            // 64bit
            //khash has nothing predefined for float/double
            //      in the first stet we add needed functionality
            typedef float64_t khfloat64_t;

            #define ZERO_HASH 0
            #define NAN_HASH  0

            CYKHASH_INLINE uint32_t kh_float64_hash_func(float64_t val){
                  if(val==0.0){
                    return ZERO_HASH;
                  }
                  if(val!=val){
                   return NAN_HASH;
                  }
                  int64_t as_int = f64_to_ui64(val);
                  return murmur2_64to32(as_int);
            }

            //                                                       take care of nans:
            #define kh_float64_hash_equal(a, b) ((a) == (b) || ((b) != (b) && (a) != (a)))

            // 32bit
            typedef float float32_t;
            typedef float32_t khfloat32_t;


            CYKHASH_INLINE uint32_t kh_float32_hash_func(float32_t val){ 
                  if(val==0.0){
                    return ZERO_HASH;
                  }
                  if(val!=val){
                   return NAN_HASH;
                  }    
                  int32_t as_int = f32_to_ui32(val);
                  return murmur2_32to32(as_int);
            }


            //                                                       take care of nans:
            #define kh_float32_hash_equal(a, b) ((a) == (b) || ((b) != (b) && (a) != (a)))
        #endif
    """  
    pass


cdef extern from *:
    """
        #ifndef CYKHASH_PYOBJECT_HASH_FUNS_PXI
        #define CYKHASH_PYOBJECT_HASH_FUNS_PXI
            //khash has nothing predefined for Pyobject
            #include <Python.h>

            typedef PyObject* pyobject_t;
            typedef pyobject_t khpyobject_t;


            CYKHASH_INLINE int floatobject_cmp(PyFloatObject* a, PyFloatObject* b){
                return (
                         Py_IS_NAN(PyFloat_AS_DOUBLE(a)) &&
                         Py_IS_NAN(PyFloat_AS_DOUBLE(b))
                       )
                       ||
                       ( PyFloat_AS_DOUBLE(a) == PyFloat_AS_DOUBLE(b) );
            }


            CYKHASH_INLINE int complexobject_cmp(PyComplexObject* a, PyComplexObject* b){
                return (
                            Py_IS_NAN(a->cval.real) &&
                            Py_IS_NAN(b->cval.real) &&
                            Py_IS_NAN(a->cval.imag) &&
                            Py_IS_NAN(b->cval.imag)
                       )
                       ||
                       (
                            Py_IS_NAN(a->cval.real) &&
                            Py_IS_NAN(b->cval.real) &&
                            a->cval.imag == b->cval.imag
                       )
                       ||
                       (
                            a->cval.real == b->cval.real &&
                            Py_IS_NAN(a->cval.imag) &&
                            Py_IS_NAN(b->cval.imag)
                       )
                       ||
                       (
                            a->cval.real == b->cval.real &&
                            a->cval.imag == b->cval.imag
                       );
            }

            CYKHASH_INLINE int pyobject_cmp(PyObject* a, PyObject* b);


            CYKHASH_INLINE int tupleobject_cmp(PyTupleObject* a, PyTupleObject* b){
                Py_ssize_t i;

                if (Py_SIZE(a) != Py_SIZE(b)) {
                    return 0;
                }

                for (i = 0; i < Py_SIZE(a); ++i) {
                    if (!pyobject_cmp(PyTuple_GET_ITEM(a, i), PyTuple_GET_ITEM(b, i))) {
                        return 0;
                    }
                }
                return 1;
            }


            CYKHASH_INLINE int pyobject_cmp(PyObject* a, PyObject* b) {
                if (a == b) {
                    return 1;
                }
                if (Py_TYPE(a) == Py_TYPE(b)) {
                    // special handling for some built-in types which could have NaNs:
                    if (PyFloat_CheckExact(a)) {
                        return floatobject_cmp((PyFloatObject*)a, (PyFloatObject*)b);
                    }
                    if (PyComplex_CheckExact(a)) {
                        return complexobject_cmp((PyComplexObject*)a, (PyComplexObject*)b);
                    }
                    if (PyTuple_CheckExact(a)) {
                        return tupleobject_cmp((PyTupleObject*)a, (PyTupleObject*)b);
                    }
                    // frozenset isn't yet supported
                }

	            int result = PyObject_RichCompareBool(a, b, Py_EQ);
	            if (result < 0) {
		            PyErr_Clear();
		            return 0;
	            }
	            return result;
            }


            ///hashes:

            CYKHASH_INLINE Py_hash_t _Cykhash_HashDouble(double val) {
                //Since Python3.10, nan is no longer has hash 0
                if (Py_IS_NAN(val)) {
                    return 0;
                }
            #if PY_VERSION_HEX < 0x030A0000
                return _Py_HashDouble(val);
            #else
                return _Py_HashDouble(NULL, val);
            #endif
            }


            CYKHASH_INLINE Py_hash_t floatobject_hash(PyFloatObject* key) {
                return _Cykhash_HashDouble(PyFloat_AS_DOUBLE(key));
            }


            // replaces _Py_HashDouble with _Cykhash_HashDouble
            #define _CykhashHASH_IMAG 1000003UL
            CYKHASH_INLINE Py_hash_t complexobject_hash(PyComplexObject* key) {
                Py_uhash_t realhash = (Py_uhash_t)_Cykhash_HashDouble(key->cval.real);
                Py_uhash_t imaghash = (Py_uhash_t)_Cykhash_HashDouble(key->cval.imag);
                if (realhash == (Py_uhash_t)-1 || imaghash == (Py_uhash_t)-1) {
                    return -1;
                }
                Py_uhash_t combined = realhash + _CykhashHASH_IMAG * imaghash;
                if (combined == (Py_uhash_t)-1) {
                    return -2;
                }
                return (Py_hash_t)combined;
            }


            CYKHASH_INLINE uint32_t pyobject_hash(PyObject* key);

            //we could use any hashing algorithm, this is the original CPython's for tuples

            #if SIZEOF_PY_UHASH_T > 4
            #define _CykhashHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
            #define _CykhashHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
            #define _CykhashHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
            #define _CykhashHASH_XXROTATE(x) ((x << 31) | (x >> 33))  /* Rotate left 31 bits */
            #else
            #define _CykhashHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
            #define _CykhashHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
            #define _CykhashHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
            #define _CykhashHASH_XXROTATE(x) ((x << 13) | (x >> 19))  /* Rotate left 13 bits */
            #endif

            CYKHASH_INLINE Py_hash_t tupleobject_hash(PyTupleObject* key) {
                Py_ssize_t i, len = Py_SIZE(key);
                PyObject **item = key->ob_item;

                Py_uhash_t acc = _CykhashHASH_XXPRIME_5;
                for (i = 0; i < len; i++) {
                    Py_uhash_t lane = pyobject_hash(item[i]);
                    if (lane == (Py_uhash_t)-1) {
                        return -1;
                    }
                    acc += lane * _CykhashHASH_XXPRIME_2;
                    acc = _CykhashHASH_XXROTATE(acc);
                    acc *= _CykhashHASH_XXPRIME_1;
                }

                /* Add input length, mangled to keep the historical value of hash(()). */
                acc += len ^ (_CykhashHASH_XXPRIME_5 ^ 3527539UL);

                if (acc == (Py_uhash_t)-1) {
                    return 1546275796;
                }
                return acc;
            }


            CYKHASH_INLINE uint32_t pyobject_hash(PyObject* key) {
                Py_hash_t hash;
                // For PyObject_Hash holds:
                //    hash(0.0) == 0 == hash(-0.0)
                //    yet for different nan-objects different hash-values
                //    are possible
                if (PyFloat_CheckExact(key)) {
                    // we cannot use kh_float64_hash_func
                    // becase float(k) == k holds for any int-object k
                    // and kh_float64_hash_func doesn't respect it
                    hash = floatobject_hash((PyFloatObject*)key);
                }
                else if (PyComplex_CheckExact(key)) {
                    // we cannot use kh_complex128_hash_func
                    // becase complex(k,0) == k holds for any int-object k
                    // and kh_complex128_hash_func doesn't respect it
                    hash = complexobject_hash((PyComplexObject*)key);
                }
                else if (PyTuple_CheckExact(key)) {
                    hash = tupleobject_hash((PyTupleObject*)key);
                }
                else {
                    hash = PyObject_Hash(key);
                }

	            if (hash == -1) {
		            PyErr_Clear();
		            return 0;
	            }
                #if SIZEOF_PY_HASH_T == 4
                    // it is already 32bit value
                    return (uint32_t)hash;
                #else
                    // for 64bit builds,
                    // we need information of the upper 32bits as well
                    // uints avoid undefined behavior of signed ints
                    return kh_int64_hash_func((uint64_t) hash);
                #endif
            }
        #endif
    """
    pass



cdef extern from *:
    """
        #ifndef CYKHASH_DEFINE_HASH_FUNS_PXI
        #define CYKHASH_DEFINE_HASH_FUNS_PXI


            // used hash-functions
            #define cykh_int32_hash_func murmur2_32to32
            #define cykh_int64_hash_func murmur2_64to32

            #define cykh_float32_hash_func kh_float32_hash_func
            #define cykh_float64_hash_func kh_float64_hash_func

            #define cykh_pyobject_hash_func pyobject_hash

            
            // used equality-functions
            #define cykh_int32_hash_equal kh_int32_hash_equal
            #define cykh_int64_hash_equal kh_int64_hash_equal

            #define cykh_float32_hash_equal kh_float32_hash_equal
            #define cykh_float64_hash_equal kh_float64_hash_equal

            #define cykh_pyobject_hash_equal pyobject_cmp

        #endif
    """
    pass
