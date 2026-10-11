from rapidjson import dumps

cdef unicode tounicode(char *s):
    return s.decode("UTF-8")

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