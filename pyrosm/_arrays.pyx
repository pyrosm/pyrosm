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
              "lon": np.float64,
              "lat": np.float64,
              "tags": object,
              "members": object,
              "geometry": object,
              }
    if key in dtypes.keys():
        return dtypes[key]
    return object

cdef convert_way_records_to_lists(ways, tags_to_separate_as_arrays):
    """
    Function to convert heterogeneous way dictionaries into a harmonized dictionary
    of value-lists. Only the OSM keys that actually occur in the data are materialized
    as columns: a candidate key absent from every way is skipped here instead of being
    built as an all-None column and dropped downstream. Where a key is absent from a
    given way its value is None, so every column stays the same length. Keys not in
    'tags_to_separate_as_arrays' are collected into the JSON 'tags' column.
    """
    cdef int i, n = len(ways)
    cdef bint any_other = False
    column_keys = list(dict.fromkeys(tags_to_separate_as_arrays))
    column_set = set(column_keys)
    records = []
    other_list = []
    appearing = set()

    for i in range(0, n):
        way = ways[i]
        record = {}
        other_tags = {}
        for k, v in way.items():
            if k in column_set:
                record[k] = v
                appearing.add(k)
            else:
                other_tags[k] = v
        records.append(record)
        if len(other_tags) > 0:
            other_list.append(dumps(other_tags))
            any_other = True
        else:
            other_list.append(None)

    data = {}
    for k in column_keys:
        if k in appearing:
            data[k] = [record.get(k) for record in records]
    # The leftover-tags column is kept only when some way had leftover tags
    # (otherwise it would be all-None and was dropped downstream before).
    if any_other:
        data["tags"] = other_list

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

cpdef columns_to_arrays(data):
    """Python entry to convert a ``{column: value-list}`` dict into the harmonized numpy
    arrays (dropping all-None columns, applying the per-key dtypes) used to build a
    GeoDataFrame -- the same conversion ``way_records_to_arrays`` runs, exposed so an
    alternative reader can feed pre-split columns directly instead of rebuilding per-element
    records first."""
    return convert_to_arrays_and_drop_empty(data)

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

cpdef way_records_to_arrays(records, tags_to_separate_as_arrays):
    """Convert way/relation records (dicts whose tags have been exploded onto the record)
    into the harmonized dict-of-arrays used to build a GeoDataFrame -- splitting the OSM
    keys in ``tags_to_separate_as_arrays`` into their own columns and the rest into the
    JSON ``tags`` column. Python entry point reusing the same conversion the in-memory
    reader uses, so an alternative reader can produce byte-identical columns."""
    return convert_to_arrays_and_drop_empty(
        convert_way_records_to_lists(records, tags_to_separate_as_arrays)
    )

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
