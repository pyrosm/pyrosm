cpdef create_point_geometries(xarray, yarray)
cdef _create_point_geometries(xarray, yarray)
cdef _create_way_geometries(node_coordinates, way_elements, parse_network)
cpdef create_way_geometries(node_coordinates, way_elements, parse_network)
cpdef create_network_segments(node_coordinates, way_elements)
cpdef segment_lines(start, end)
cdef create_relation_geometry(node_coordinates, ways,
                             member_roles, force_linestring,
                             make_multipolygon,
                             bint drop_if_open=*)
cpdef create_linear_ring(coordinates)
cdef get_way_coordinates_for_polygon(node_coordinate_lookup, way_records)
cpdef fix_geometry(geometry, diff_threshold=*)
cdef is_valid_coordinate_pair(lat, lon)
