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
    return exploded, list(way_keys.keys())