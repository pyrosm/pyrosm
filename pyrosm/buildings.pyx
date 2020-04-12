from pyrosm.data_filter cimport filter_osm, filter_array_dict_by_indices_or_mask
from pyrosm._arrays cimport convert_to_arrays_and_drop_empty, convert_way_records_to_lists, concatenate_dicts_of_arrays
from pyrosm.geometry cimport get_relation_multipolygon_indices_for_osm_key

cdef filter_buildings(data_records, data_filter, relation_way_ids):
    return filter_osm(data_records, data_filter,
                      osm_data_type="building",
                      relation_way_ids=relation_way_ids,
                      filter_type="keep")

cdef get_building_multipolygon_indices(relations):
    return get_relation_multipolygon_indices_for_osm_key(relations, "building")

cdef get_building_relation_ways(way_records, members_array_dict):
    # Get ways that are buildings and which id is part of building relation
    cdef int i, N = len(way_records)
    member_ids = members_array_dict["member_id"]
    result = []

    for i in range(0, N):
        way = way_records[i]
        if way["id"] in member_ids:
            result.append(way)
    return result

def get_building_relation_arrays(relations):
    # Combine all blocks
    relations = concatenate_dicts_of_arrays(relations)
    # Get indices for building MultiPolygons
    building_indices = get_building_multipolygon_indices(relations)
    # If no building relations were found, return None
    if len(building_indices) == 0:
        return None
    # Otherwise, filter the data accordingly
    return filter_array_dict_by_indices_or_mask(relations, building_indices)

cdef separate_relation_ways(way_records, relation_way_ids):
    cdef int i, n = len(way_records)
    # Create lookup dict
    relation_ids = dict.fromkeys(relation_way_ids, None)
    normal_ways = []
    relation_ways = []
    for i in range(0, n):
        way = way_records[i]
        try:
            # Check if way is part of relation
            relation_ids[way["id"]]
            relation_ways.append(way)
        except Exception as e:
            normal_ways.append(way)

    return normal_ways, relation_ways

cdef get_building_way_arrays(ways, relation_member_ids, tags_to_keep, building_filter):
    # Get all building ways
    ways = filter_buildings(ways, building_filter, relation_member_ids)

    relation_arrays = None
    if relation_member_ids is not None:
        # Separate ways that are part of a relation
        ways, relation_ways = separate_relation_ways(ways, relation_member_ids)
        relation_ways = convert_way_records_to_lists(relation_ways, tags_to_keep)
        relation_arrays = convert_to_arrays_and_drop_empty(relation_ways)

    # Process separated ways
    ways = convert_way_records_to_lists(ways, tags_to_keep)
    way_arrays = convert_to_arrays_and_drop_empty(ways)

    return way_arrays, relation_arrays

cdef _get_building_data(way_records, relations, tags_to_keep, building_filter):
    tags_to_keep += ["id", "nodes", "timestamp", "version"]

    # Get relations
    filtered_relations = get_building_relation_arrays(relations)

    relation_member_ids = None
    if filtered_relations is not None:
        members = concatenate_dicts_of_arrays(filtered_relations['members'])
        relation_member_ids = members["member_id"]

    # Get building ways (separately as "normal" ways and relation_ways)
    ways, relation_ways = get_building_way_arrays(way_records, relation_member_ids,
                                                  tags_to_keep, building_filter)
    return ways, relation_ways, filtered_relations

cpdef get_building_data(way_records, relations, tags_to_keep, building_filter):
    return _get_building_data(way_records, relations, tags_to_keep, building_filter)
