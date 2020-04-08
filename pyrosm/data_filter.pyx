
cdef filter_osm(data_records, data_filter, osm_data_type):
    """
    osm_data_type can be: 'highway', 'building', or 'landuse'
    """
    cdef str rec_value
    cdef int i, N=len(data_records)

    filtered_data = []
    filter_out = False

    if data_filter is not None:
        filter_keys = list(data_filter.keys())

    for i in range(0, N):
        record = data_records[i]
        if osm_data_type not in record.keys():
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


cdef get_filtered_data(ways, tags_to_keep):
    cdef int i
    cdef int n=len(ways)

    lookup = dict.fromkeys(tags_to_keep, None)
    data = {k: [] for k in tags_to_keep}
    data["tags"] = []

    for i in range(0, n):
        way = ways[i]
        way_records = dict.fromkeys(tags_to_keep, None)
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
        data["tags"].append(str(other_tags))
    return data
