import numpy as np
from cykhash.khashsets cimport any_int64_from_iter, isin_int64, Int64Set_from_buffer
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


cdef has_osm_data_type(osm_data_types, tag_keys):
    cdef str key, osm_data_type
    cdef int i, n = len(tag_keys)
    for i in range(0, n):
        key = tag_keys[i]
        for osm_data_type in osm_data_types:
            if osm_data_type == key:
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


cdef filter_osm_records(data_records,
                        data_filter,
                        osm_data_type,
                        relation_way_ids,
                        filter_type):
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
              
    osm_data_type : str | list
        Basic data type(s) used for filtering, such as:
         - 'highway' (for roads), 
         - 'building' (for buildings), 
         - 'landuse' (for landuse) etc.
         - ['amenity', 'shop', 'craft'] (a combination that can be useful for parsing POIs)
         
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

    if not isinstance(osm_data_type, list):
        osm_data_type = [osm_data_type]

    if data_filter is not None:
        if len(data_filter) == 0:
            data_filter = None

    if data_filter is not None:
        # Check if there are duplicate filter values
        # e.g. a situation where {"route": "tram"} and {"railway": "tram"}
        # filters are present simultaneously.
        overlapping_filter = False

        filter_values = []
        way_filter = {}
        for key, vals in data_filter.items():

            # Check for {osm-key: True} cases
            if vals is True or vals == [True]:
                continue

            filter_values += vals
            way_filter[key] = vals

        # Update data filter
        if len(way_filter) == 0:
            data_filter = None
        else:
            data_filter = {k: v for k, v in way_filter.items()}
            filter_keys = list(data_filter.keys())

        # Check for overlapping filter
        if len(filter_values) > len(list(set(filter_values))):
            overlapping_filter = True

    relation_check = False
    if relation_way_ids is not None:
        relation_way_lookup = dict.fromkeys(relation_way_ids)
        relation_check = True

    for i in range(0, N):
        record = data_records[i]
        record_keys = list(record.keys())
        # If way is part of relation it should be kept
        # (ways that are part of relation might not have any tags)
        if relation_check:
            if way_is_part_of_relation(record, relation_way_lookup):
                filtered_data.append(record)
                continue

        if not has_osm_data_type(osm_data_type, record_keys):
            continue

        # Check if should be filtered based on given data_filter
        if data_filter is not None:
            filter_out = False
            filter_was_in_record = False

            for k, v in record.items():
                if k in filter_keys:
                    filter_was_in_record = True
                    if solver.check(v, data_filter[k]):
                        filter_out = True
                        # If there are identical filter criteria used in multiple OSM-keys
                        # Check that none of them matches, hence do not break the loop
                        # after the first match is found.
                        if not overlapping_filter:
                            break
                    else:
                        filter_out = False
                        break

            # If none of the filter keys are present in the element,
            # it should not be kept
            if not filter_was_in_record:
                continue

            if not filter_out:
                filtered_data.append(record)

        # If data_filter has not been specified,
        # all data for specified osm-keys should be kept.
        else:
            filtered_data.append(record)
    return filtered_data


cpdef _filter_array_dict_by_indices_or_mask(array_dict, indices):
    return filter_array_dict_by_indices_or_mask(array_dict, indices)


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
    return any_int64_from_iter(nodes, node_lookup)


cpdef get_mask_by_osmid(src_array, osm_ids):
    """
    Creates a (boolean) mask for the given source array flagging True
    all items that exist in the 'osm_ids' array. Can be used to filter items
    e.g. from OSM node data arrays.
    """
    n = len(src_array)
    lookup = Int64Set_from_buffer(osm_ids)
    result = np.empty(src_array.size, dtype=np.bool)
    isin_int64(src_array, lookup, result)
    return result


cdef record_should_be_kept(tag, osm_keys, data_filter, filter_type):
    if tag is None:
        return False

    cdef str k, osm_key
    filter_keys = list(data_filter.keys())
    tag_keys = list(tag.keys())

    # Check if OSM key exist for the given element
    osm_key_was_found = False
    for osm_key in osm_keys:
        if osm_key in tag_keys:
            osm_key_was_found = True

    # If not, the element shouldn't be kept
    if not osm_key_was_found:
        return False

    # If there is no filter but the element is correct kind
    if len(filter_keys) == 0:
        if filter_type == "keep":
            return True
        else:
            return False

    # If there is a filter, check if match is found
    for k, v in data_filter.items():
        if k in tag_keys:
            # Check match with data filter
            if tag[k] in v:
                if filter_type == "keep":
                    return True
                else:
                    return False
            # If filter is not defined, check for 'osm_key': True
            elif v == [True] or v == True:
                if filter_type == "keep":
                    return True
                else:
                    return False

    if filter_type == "keep":
        return False
    return True


cdef filter_relation_indices(relations, osm_keys, data_filter, filter_type):
    cdef int i, n = len(relations["tags"])
    indices = []

    # Ensure keys
    if len(data_filter) == 0:
        relation_filter = {}
    else:
        relation_filter = {key: value for key, value in data_filter.items()}

    for i in range(0, n):
        tag = relations["tags"][i]
        if record_should_be_kept(tag, osm_keys, relation_filter, filter_type):
            indices.append(i)
    return indices


cdef filter_node_indices(node_arrays, osm_keys, data_filter, filter_type):
    cdef int i, n = len(node_arrays["tags"])
    indices = []

    if len(data_filter) == 0:
        node_filter = {}
    else:
        node_filter = {key: value for key, value in data_filter.items()}

    for i in range(0, n):
        tag = node_arrays["tags"][i]
        if record_should_be_kept(tag, osm_keys, node_filter, filter_type):
            indices.append(i)

    return indices


cpdef get_latest_version(df):
    # The order of versions is always the same
    # (newest version is the last)
    return df.groupby("id").last().reset_index()


cdef clean_empty_values_from_ways(ways):
    cdef int i, n = len(ways)
    cleaned = []
    for i in range(0, n):
        cleaned.append({x: y for x, y in ways[i].items() if y != None})
    return cleaned
