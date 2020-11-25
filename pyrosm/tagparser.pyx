from rapidjson import dumps

cdef unicode tounicode(char *s):
    return s.decode("UTF-8")

cdef parse_tags(keys, vals, stringtable, tag_filter):
    cdef int i, N=len(keys)

    filter_tags = False
    if tag_filter is not None:
        filter_tags = True

    tags = {}
    for i in range(0, N):
        # Parse key and value as text
        key, value = stringtable[keys[i]], stringtable[vals[i]]

        # Filter tags if requested
        if filter_tags:
            if key not in tag_filter:
                continue
        tags[key] = value

    return tags

cdef parse_dense_tags(keys_vals, string_table, tag_filter):
    cdef int N=len(keys_vals)
    cdef int tag_idx = 0
    tag_list = []

    filter_tags = False
    if tag_filter is not None:
        filter_tags = True

    # Flag for informing whether tags were
    # found/kept for any of the nodes
    tags_kept = False

    while tag_idx < N:
        tags = {}

        while keys_vals[tag_idx] != 0:
            key, value = string_table[keys_vals[tag_idx]], string_table[keys_vals[tag_idx + 1]]
            tag_idx += 2

            if filter_tags:
                if not key in tag_filter:
                    continue

            tags[key] = value

        if len(tags) > 0:
            tag_list.append(tags)
            tags_kept = True

        else:
            tag_list.append(None)
        tag_idx += 1

    if not tags_kept:
        return None

    return tag_list

cdef explode_way_tags(ways):
    exploded = []
    cdef int i, n=len(ways)
    way_keys = {}
    for i in range(0, n):
        way = ways[i]
        for k, v in way['tags'].items():
            way[k] = v
            try:
                dummy = way_keys[k]
            except:
                way_keys[k] = None
        del way['tags']
        exploded.append(way)
    return exploded, list(way_keys.keys())

cdef explode_tag_array(tag_array, tags_as_columns):
    lookup = dict.fromkeys(tags_as_columns, None)
    data = {k: [] for k in tags_as_columns}
    data["tags"] = []
    n = len(tag_array)
    for i in range(0, n):
        tag = tag_array[i]
        tag_records = dict.fromkeys(tags_as_columns, None)
        other_tags = {}
        for k, v in tag.items():
            try:
                # Check if tag should be kept as a column
                lookup[k]
                tag_records[k] = v
            except:
                # If not add into tags
                other_tags[k] = v
        [data[k].append(v) for k, v in tag_records.items()]
        if len(other_tags) > 0:
            data["tags"].append(dumps(other_tags))
        else:
            data["tags"].append(None)
    return data