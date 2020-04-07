from libc.stdlib cimport malloc
import cython

cdef char** to_cstring_array(list str_list):
    """
    Converts Python byte-string list to an "array" of c-strings. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef int i, N=len(str_list)
    cdef char ** string_array = <char **> malloc(N * sizeof(char *))
    cdef char * txt

    if not string_array:
        raise MemoryError()

    for i in range(0, N):
        txt = str_list[i]
        string_array[i] = txt

    return string_array


cdef int* to_cint_array(list int_list):
    """
    Converts Python list of integers to an "array" of C-integers. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef int *c_ints
    cdef int N = len(int_list)

    c_ints = <int *>malloc(N*cython.sizeof(int))

    if not c_ints:
        raise MemoryError()

    for i in range(0, N):
        c_ints[i] = int_list[i]

    return c_ints


cdef float* to_cfloat_array(list float_list):
    """
    Converts Python list of floats to an "array" of C-floats. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef float *c_floats
    cdef int N = len(float_list)

    c_floats = <float *>malloc(N*cython.sizeof(float))

    if not c_floats:
        raise MemoryError()

    for i in range(0, N):
        c_floats[i] = float_list[i]

    return c_floats

cdef long long* to_clong_array(long_list):
    """
    Converts Python list of integers (long) to an "array" of C-long longs. 
    NOTE: Memory handling needs to be done manually in the main application!
    """

    cdef long long *c_longs
    cdef int N = len(long_list)

    c_longs = <long long *>malloc(N*sizeof(long long))

    if not c_longs:
        raise MemoryError()

    for i in range(0, N):
        c_longs[i] = long_list[i]

    return c_longs