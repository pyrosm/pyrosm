from pyrosm.data_filter cimport filter_osm
from pyrosm._arrays cimport convert_to_arrays_and_drop_empty, convert_way_records_to_lists

cdef filter_network_data(data_records, data_filter):
    return filter_osm(data_records, data_filter,
                      osm_data_type="highway",
                      filter_type="exclude")

cdef _get_way_data(ways, tags_to_keep, network_filter):

    # Tags to keep as separate columns
    tags_to_keep += ["id", "nodes", "timestamp", "version"]

    # Filter data with given filter
    # (if network_filter is None, will keep all with tag 'highway')
    ways = filter_network_data(ways, network_filter)
    data = convert_way_records_to_lists(ways, tags_to_keep)
    arrays = convert_to_arrays_and_drop_empty(data)
    return arrays


cpdef get_way_data(ways, tags_to_keep, network_filter):
    return _get_way_data(ways, tags_to_keep, network_filter)