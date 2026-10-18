from rapidjson import dumps

cdef unicode tounicode(char *s):
    return s.decode("UTF-8")

cdef explode_tag_array(tag_array, tags_as_columns):
    """The tag columns of the tag dicts ``tag_array`` that occur (a key of
    ``tags_as_columns`` with at least one value), then, when some element has other tags,
    the JSON ``tags`` column of those. A tag keyed ``id`` is routed as ``id_tag``, so it
    cannot replace the element's id."""
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
            key = "id_tag" if k == "id" else k
            if key in column_set:
                record[key] = v
                appearing.add(key)
            else:
                other_tags[key] = v
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
    occurring ``tags_as_columns`` columns plus a JSON ``tags`` column; a tag keyed ``id``
    as ``id_tag``), so an alternative
    reader can build node features with the same columns as the in-memory reader."""
    return explode_tag_array(tag_array, tags_as_columns)