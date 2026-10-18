# Typed C pointers to the data of contiguous numpy arrays, for the loops that run without
# the GIL. Each array must be contiguous and of the named dtype, and stay alive while the
# pointer is used.

from libc.stdint cimport int64_t, uint8_t, uintptr_t


cdef inline int64_t* _int64_ptr(array):
    """The data of the contiguous int64 array ``array``."""
    return <int64_t*><uintptr_t>array.ctypes.data


cdef inline uint8_t* _uint8_ptr(array):
    """The data of the contiguous uint8 array ``array``."""
    return <uint8_t*><uintptr_t>array.ctypes.data


cdef inline double* _double_ptr(array):
    """The data of the contiguous float64 array ``array``."""
    return <double*><uintptr_t>array.ctypes.data
