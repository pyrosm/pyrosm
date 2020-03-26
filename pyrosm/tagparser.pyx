cdef unicode tounicode(char *s):
    return s.decode("UTF-8")

cdef parse_way_tags(keys, vals, stringtable):
    cdef int k, v
    d = dict()
    for k, v in zip(keys, vals):
        d[stringtable[k]] = stringtable[v]
    return d

cdef parse_dense_tags(keys_vals, string_table):
    cdef int tag_idx, k, v
    tag_list = []
    tag_idx = 0
    while tag_idx < len(keys_vals):
        tags = {}
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
    cdef dict way, way_keys
    cdef list exploded = []
    cdef str k, v, dummy
    way_keys = {}

    for way in ways:
        for k, v in way['tags'].items():
            way[k] = v
            try:
                dummy = way_keys[k]
            except:
                way_keys[k] = None
        del way['tags']
        exploded.append(way)
    return exploded, list(way_keys.keys())