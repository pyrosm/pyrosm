cpdef create_point_geometries(xarray, yarray)
cdef _create_point_geometries(xarray, yarray)
cdef _create_way_geometries(node_coordinates, way_elements, parse_network)
cpdef create_way_geometries(node_coordinates, way_elements, parse_network)
cdef create_relation_geometry(node_coordinates, ways,
                             member_roles, force_linestring,
                             make_multipolygon)
cdef create_linear_ring(coordinates)
cdef create_linestring_geometry(nodes, node_coordinates)
cdef get_way_coordinates_for_polygon(node_coordinate_lookup, way_records)
cpdef fix_geometry(geometry, diff_threshold=*)
cdef is_valid_coordinate_pair(lat, lon)
