from libc.stdlib cimport malloc
import cython
import numpy as np


cdef convert_to_array_dict(data):
    # Convert to arrays
    arrays = {}
    for key, value_list in data.items():
        # Nodes are in a list and should always be kept
        if not isinstance(value_list[0], list):
            # Otherwise keep tag only if it contains data
            unique = list(set(value_list))
            if len(unique) < 2:
                if unique[0] is None:
                    continue
        arrays[key] = np.array(value_list, dtype=object)
    return arrays

cdef get_dtype(key):
    dtypes = {"id": np.int64,
              "version": np.int8,
              "changeset": np.int8,
              "timestamp": np.int64,
              "lon": np.float32,
              "lat": np.float32,
              "tags": object,
              "members": object,
              }
    if key in dtypes.keys():
        return dtypes[key]
    return None

cdef concatenate_dicts_or_arrays(dict_list_of_arrays):
    cdef str k

    keys = list(set([k for d in dict_list_of_arrays
                     for k in d.keys()]))
    result_dict = {key: [] for key in keys}

    for dicts in dict_list_of_arrays:
        for k, v in dicts.items():
            result_dict[k] += v.tolist()

    # Convert to arrays
    result_arrays = {}
    for k, v in result_dict.items():
        result_arrays[k] = np.array(v, dtype=get_dtype(k))

    # The length of all arrays must match
    length = None
    for k, array in result_arrays.items():
        if length is None:
            length = array.shape[0]
        else:
            assert length == array.shape[0]

    return result_arrays


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