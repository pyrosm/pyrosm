from pyrosm.data_filter cimport filter_osm, get_filtered_data
from pyrosm._arrays cimport convert_to_array_dict


cdef filter_buildings(data_records, data_filter):
    return filter_osm(data_records, data_filter,
                      osm_data_type="building")


cdef _get_building_data(ways, tags_to_keep, building_filter):

    tags_to_keep += ["id", "nodes", "timestamp", "version"]

    # Filter data with given filter
    # (if filter is None, will keep all with tag 'building')
    ways = filter_buildings(ways, building_filter)
    data = get_filtered_data(ways, tags_to_keep)
    arrays = convert_to_array_dict(data)

    return arrays


cpdef get_building_data(ways, tags_to_keep, building_filter):
    return _get_building_data(ways, tags_to_keep, building_filter)