cdef filter_osm_records(data_records, data_filter, osm_data_type, relation_way_ids, filter_type)
cdef filter_array_dict_by_indices_or_mask(array_dict, indices)
cdef get_lookup_khash_for_int64(int64_id_array)
cdef get_nodeid_lookup_khash(nodes)
cdef nodes_for_way_exist_khash(nodes, node_lookup)
cdef filter_relation_indices(relations, osm_keys, data_filter, filter_type)
cdef filter_node_indices(node_arrays, osm_keys, data_filter, filter_type)
cpdef get_mask_by_osmid(src_array, osm_ids)
cpdef get_latest_version(df)
cdef clean_empty_values_from_ways(ways)

# For debuggin purposes
cpdef _filter_array_dict_by_indices_or_mask(array_dict, indices)