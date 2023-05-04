from libc.stdlib cimport malloc
import cython
import numpy as np
from rapidjson import dumps
from geopandas import GeoSeries

cdef get_dtype(key):
    dtypes = {"id": np.int64,
              "version": np.int32,
              "changeset": np.int32,
              "timestamp": np.uint32,
              "lon": np.float32,
              "lat": np.float32,
              "tags": object,
              "members": object,
              "geometry": object,
              }
    if key in dtypes.keys():
        return dtypes[key]
    return object

cdef convert_way_records_to_lists(ways, tags_to_separate_as_arrays):
    """
    Function to convert heterogeneous way dictionaries into harmonized dictionary 
    of value-lists for all OSM keys. If a given OSM-key is not present for a given way 
    record, will add None. This process makes it possible to create same-sized numpy arrays
    for all OSM way tags.  
    """
    cdef int i
    cdef int n = len(ways)

    lookup = dict.fromkeys(tags_to_separate_as_arrays, None)
    data = {k: [] for k in tags_to_separate_as_arrays}
    data["tags"] = []

    for i in range(0, n):
        way = ways[i]
        way_records = dict.fromkeys(tags_to_separate_as_arrays, None)
        other_tags = {}
        for k, v in way.items():
            try:
                # Check if tag should be kept as a column
                lookup[k]
                way_records[k] = v
            except:
                # If not add into tags
                other_tags[k] = v
        [data[k].append(v) for k, v in way_records.items()]
        if len(other_tags) > 0:
            data["tags"].append(dumps(other_tags))
        else:
            data["tags"].append(None)

    return data

cdef convert_to_arrays_and_drop_empty(data):
    """
    Function to convert the harmonized dictionary 
    of key:values to numpy arrays. If a given OSM-key contains
    only None values (i.e. is "empty"), it will be dropped from the 
    resulting dictionary. 
    """
    # Convert to arrays
    arrays = {}
    for key, value_list in data.items():
        # Parse geometry separately for handling multi-geoms correctly
        if key == "geometry":
            arrays[key] = GeoSeries(value_list).values
            continue

        # Nodes are in a list and should always be kept
        elif not isinstance(value_list[0], list):
            # Keep tag only if it contains data
            unique = list(set(value_list))
            if len(unique) < 2:
                if unique[0] is None:
                    continue
        try:
            arrays[key] = np.array(value_list, dtype=get_dtype(key))
        except ValueError as e:
            if "invalid literal for int" in str(e):
                # Try first converting to int via floats
                try:
                    value_list = list(map(int, list(map(float, value_list))))
                    arrays[key] = np.array(value_list, dtype=np.int64)
                except ValueError as e:
                    # If there is a string, keep as is
                    if "convert string" in str(e):
                        arrays[key] = np.array(value_list, dtype=object)
                except Exception as e:
                    raise e
        except Exception as e:
            raise e

    return arrays

cpdef concatenate_dicts_of_arrays(dict_list_of_arrays):
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
        if len(v) > 0:
            result_arrays[k] = np.array(v, dtype=get_dtype(k))

    # The length of all arrays must match
    length = None
    for k, array in result_arrays.items():
        arr_cnt = array.shape[0]
        if length is None:
            length = arr_cnt
        else:
            assert length == arr_cnt, f"The length of '{k}' " \
                                      f"should be {length}, " \
                                      f"got {array.shape[0]}."

    return result_arrays

cdef char** to_cstring_array(list str_list):
    """
    Converts Python byte-string list to an "array" of c-strings. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef int i, N = len(str_list)
    cdef char ** string_array = <char **> malloc(N * sizeof(char *))
    cdef char *txt

    if not string_array:
        raise MemoryError()

    for i in range(0, N):
        txt = str_list[i]
        string_array[i] = txt

    return string_array

cdef int*to_cint_array(list int_list):
    """
    Converts Python list of integers to an "array" of C-integers. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef int *c_ints
    cdef int N = len(int_list)

    c_ints = <int *> malloc(N * cython.sizeof(int))

    if not c_ints:
        raise MemoryError()

    for i in range(0, N):
        c_ints[i] = int_list[i]

    return c_ints

cdef float*to_cfloat_array(list float_list):
    """
    Converts Python list of floats to an "array" of C-floats. 
    NOTE: Memory handling needs to be done manually in the main application!
    """
    cdef float *c_floats
    cdef int N = len(float_list)

    c_floats = <float *> malloc(N * cython.sizeof(float))

    if not c_floats:
        raise MemoryError()

    for i in range(0, N):
        c_floats[i] = float_list[i]

    return c_floats

cdef long long*to_clong_array(long_list):
    """
    Converts Python list of integers (long) to an "array" of C-long longs. 
    NOTE: Memory handling needs to be done manually in the main application!
    """

    cdef long long *c_longs
    cdef int N = len(long_list)

    c_longs = <long long *> malloc(N * sizeof(long long))

    if not c_longs:
        raise MemoryError()

    for i in range(0, N):
        c_longs[i] = long_list[i]

    return c_longs
