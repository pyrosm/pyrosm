from rapidjson import dumps

cdef unicode tounicode(char *s):
    return s.decode("UTF-8")

cdef parse_tags(keys, vals, stringtable):
    cdef int i, N=len(keys)
    d = dict()
    for i in range(0, N):
        k, v = keys[i], vals[i]
        d[stringtable[k]] = stringtable[v]
    return d

cdef parse_dense_tags(keys_vals, string_table):
    cdef int N=len(keys_vals)
    cdef int tag_idx = 0
    tag_list = []

    while tag_idx < N:
        tags = dict()
        while keys_vals[tag_idx] != 0:
            k = keys_vals[tag_idx]
            v = keys_vals[tag_idx + 1]
            tag_idx += 2
            tags[string_table[k]] = string_table[v]

        if len(tags) > 0:
            tag_list.append(tags)
        else:
            tag_list.append(None)
        tag_idx += 1
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
    return exploded

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