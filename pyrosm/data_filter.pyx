import numpy as np
from cykhash.khashsets cimport Int64Set_from_buffer
from cykhash import isin_int64
from cpython cimport array


class Solver:
    """Solver is used to toggle between exclude / keep checks applied in data filter."""
    def __init__(self, direction):

        if direction == "exclude":
            self.solver = self.isin_check
        elif direction == "keep":
            self.solver = self.notin_check
        else:
            raise ValueError("filter type should be 'keep' or 'exclude'")

    def isin_check(self, value, container):
        if value in container:
            return True
        return False

    def notin_check(self, value, container):
        if value not in container:
            return True
        return False

    def check(self, value, container):
        return self.solver(value, container)


cdef has_osm_data_type(osm_data_type, tag_keys):
    cdef str key
    cdef int i, n = len(tag_keys)
    for i in range(0, n):
        key = tag_keys[i]
        # Allow osm_data_type to match with a slice of string
        # E.g. with buildings the "building" tag-key might be missing
        # (due to incorrect tagging), but it can still be attached
        # as part of other keys, such as "building:levels".
        if osm_data_type in key:
            return True
    return False

cdef way_is_part_of_relation(way_record, lookup_dict):
    try:
        lookup_dict[way_record["id"]]
        return True
    except KeyError as e:
        return False
    except Exception as e:
        raise e

cdef filter_osm(data_records, data_filter, osm_data_type,
                relation_way_ids, filter_type):
    """
    Filter OSM data records by given OSM tag key:value pairs.
    
    Parameters
    ----------
    data_records : list:
        A list of OSM data records. 
        A single record is a dictionary with OSM data, such as: 
           - {"highway": "primary", "maxspeed": 80, "name": "Highway-name-foo", 
              "nodes": [1111,2222,3333,4444]}
              
    data_filter : dict ( {"osm-key": ["list-of-osm-values"]} )
        A dictionary of tag-keys and associated values that will be used to
        filter the OSM data records, e.g. {"building": ["residential"]} filters
        the records where "building" key has a value "residential". 
        The records will be kept or excluded according the <filter_type> parameter.   
              
    osm_data_type : str
        Basic data type is used for filtering, such as:
         - 'highway' (for roads), 
         - 'building' (for buildings), 
         - 'landuse' (for landuse) etc.
         
    relation_way_ids : list (optional)
        A list of way ids that belong to relations. Ways that match with these ids are always kept.
         
    filter_type : str ( 'keep' | 'exclude' ) 
        Whether the given data_filter should 'keep' or 'exclude' the records 
        where given tag:value pair is present in the record.  
    """
    cdef str rec_value
    cdef int i, N = len(data_records)

    solver = Solver(filter_type)

    filtered_data = []
    filter_out = False

    if data_filter is not None:
        filter_keys = list(data_filter.keys())

    relation_check = False
    if relation_way_ids is not None:
        # TODO: This check should probably happen somewhere before coming here
        if type(relation_way_ids) not in [list, np.ndarray]:
            raise ValueError("'relation_way_ids' should be a list or an array.")
        relation_way_lookup = dict.fromkeys(relation_way_ids)
        relation_check = True

    for i in range(0, N):
        record = data_records[i]

        if relation_check:
            if way_is_part_of_relation(record, relation_way_lookup):
                filtered_data.append(record)
                continue

        if not has_osm_data_type(osm_data_type, list(record.keys())):
            continue

        # Check if should be filtered based on given data_filter
        if data_filter is not None:
            for k, v in record.items():
                if k in filter_keys:
                    if solver.check(v, data_filter[k]):
                        filter_out = True
                        break
            if not filter_out:
                filtered_data.append(record)
            filter_out = False
        else:
            filtered_data.append(record)
    return filtered_data

cdef filter_array_dict_by_indices_or_mask(array_dict, indices):
    return {k: v[indices] for k, v in array_dict.items()}

cdef get_lookup_khash_for_int64(int64_id_array):
    return Int64Set_from_buffer(
        memoryview(
            int64_id_array
        )
    )

cdef get_nodeid_lookup_khash(nodes):
    return get_lookup_khash_for_int64(
        np.concatenate([group['id'].tolist()
                        for group in nodes]).astype(np.int64)
    )

cdef nodes_for_way_exist_khash(nodes, node_lookup):
    v = array.array('q', nodes)
    cdef int n = len(v)

    result = array.array('B', [False] * n)
    isin_int64(v, node_lookup, result)
    if True in result:
        return True
    return False