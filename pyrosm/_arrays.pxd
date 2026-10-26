from libc.stdint cimport int64_t

cdef convert_to_arrays_and_drop_empty(data)
cpdef concatenate_dicts_of_arrays(dict_list_of_arrays)
cdef convert_way_records_to_lists(ways, tags_to_separate_as_arrays)
cdef char** to_cstring_array(list str_list)
cdef int* to_cint_array(list int_list)
cdef float* to_cfloat_array(list float_list)
cdef long long* to_clong_array(long_list)
cpdef _kept_keys_vals(const int64_t[::1] keys_vals, keep)
