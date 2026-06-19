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

cpdef explode_way_tags(ways):
    exploded = []
    cdef int i, n=len(ways)
    way_keys = {}
    for i in range(0, n):
        way = ways[i]
        for k, v in way['tags'].items():
            if k == "id":
                # An OSM tag literally keyed "id" would otherwise overwrite the
                # element's OSM id; surface it as "id_tag" to keep both.
                way["id_tag"] = v
            else:
                way[k] = v
            try:
                dummy = way_keys[k]
            except:
                way_keys[k] = None
        del way['tags']
        exploded.append(way)
    return exploded

cdef explode_tag_array(tag_array, tags_as_columns):
    # Build only the tag-columns that actually occur (a candidate key with at least
    # one value). Each element's tags are routed in a single pass; the leftover keys
    # form the JSON 'tags' column. Candidate columns that would be entirely empty are
    # never materialised -- they were dropped downstream anyway -- so the result is
    # the same set of non-empty columns the caller kept before, built far more cheaply.
    cdef int i, n = len(tag_array)
    cdef bint any_other = False
    column_keys = list(dict.fromkeys(tags_as_columns))
    column_set = set(column_keys)
    records = []
    other_list = []
    appearing = set()
    for i in range(0, n):
        tag = tag_array[i]
        record = {}
        other_tags = {}
        for k, v in tag.items():
            if k in column_set:
                record[k] = v
                appearing.add(k)
            else:
                other_tags[k] = v
        records.append(record)
        if len(other_tags) > 0:
            other_list.append(dumps(other_tags))
            any_other = True
        else:
            other_list.append(None)
    data = {}
    for k in column_keys:
        if k in appearing:
            data[k] = [record.get(k) for record in records]
    # The leftover-tags column is kept only when some element had leftover tags
    # (otherwise it would be all-None and was dropped before).
    if any_other:
        data["tags"] = other_list
    return data


cpdef explode_node_tag_array(tag_array, tags_as_columns):
    """Python entry point to the node tag-explosion (splits a tag-dict array into the
    occurring ``tags_as_columns`` columns plus a JSON ``tags`` column), so an alternative
    reader can build node features with the same columns as the in-memory reader."""
    return explode_tag_array(tag_array, tags_as_columns)