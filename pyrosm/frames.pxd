cpdef create_nodes_gdf(node_dict_list)
cpdef create_gdf(data_records, geometry_array)
cpdef prepare_way_gdf(node_coordinates, ways, parse_network, calculate_seg_lengths)
cpdef prepare_node_gdf(nodes)
cpdef prepare_geodataframe(nodes,
                           node_coordinates,
                           ways,
                           relations,
                           relation_ways,
                           tags_as_columns,
                           bounding_box,
                           parse_network=*,
                           calculate_seg_lengths=*)