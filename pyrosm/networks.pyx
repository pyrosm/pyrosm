import numpy as np
from pyrosm.data_filter cimport filter_osm, get_filtered_data
from pyrosm._arrays cimport convert_to_array_dict

cdef filter_network_data(data_records, data_filter):
    return filter_osm(data_records, data_filter,
                      osm_data_type="highway")

cdef _get_way_data(ways, tags_to_keep, network_filter):

    # ID and Node-information is still needed later
    tags_to_keep += ["id", "nodes"]

    # Filter data with given filter
    # (if network_filter is None, will keep all with tag 'highway')
    ways = filter_network_data(ways, network_filter)
    data = get_filtered_data(ways, tags_to_keep)
    arrays = convert_to_array_dict(data)
    return arrays


cpdef get_way_data(ways, tags_to_keep, network_filter):
    return _get_way_data(ways, tags_to_keep, network_filter)