cdef filter_osm(data_records, data_filter, osm_data_type, relation_way_ids, filter_type)
cdef filter_array_dict_by_indices_or_mask(array_dict, indices)
cdef get_lookup_khash_for_int64(int64_id_array)
cdef get_nodeid_lookup_khash(nodes)
cdef nodes_for_way_exist_khash(nodes, node_lookup)