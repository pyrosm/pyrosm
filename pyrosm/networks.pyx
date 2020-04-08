import numpy as np

cdef filter_network_data(data_records, data_filter):
    cdef str rec_value
    cdef int i, N=len(data_records)

    filtered_data = []
    filter_out = False

    if data_filter is not None:
        filter_keys = list(data_filter.keys())

    for i in range(0, N):
        record = data_records[i]
        if "highway" not in record.keys():
            continue
        # Check if should be filtered based on given data_filter
        if data_filter is not None:
            for k, v in record.items():
                if k in filter_keys:
                    if v in data_filter[k]:
                        filter_out = True
                        break
            if not filter_out:
                filtered_data.append(record)
            filter_out = False
        else:
            filtered_data.append(record)
    return filtered_data


cdef _get_way_data(ways, way_tags_to_keep, network_filter):
    cdef int i

    # ID and Node-information is still needed later
    way_tags_to_keep += ["id", "nodes"]

    # Filter data with given filter
    # (if network_filter is None, will keep all with tag 'highway')
    ways = filter_network_data(ways, network_filter)
    cdef int n=len(ways)

    lookup = dict.fromkeys(way_tags_to_keep, None)
    data = {k: [] for k in way_tags_to_keep}

    for i in range(0, n):
        way = ways[i]
        way_records = dict.fromkeys(way_tags_to_keep, None)
        for k, v in way.items():
            try:
                # Check if tag should be kept
                lookup[k]
                way_records[k] = v
            except:
                pass
        [data[k].append(v) for k, v in way_records.items()]

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


cpdef get_way_data(ways, way_tags_to_keep, network_filter):
    return _get_way_data(ways, way_tags_to_keep, network_filter)