import numpy as np
from pygeos import linestrings, polygons, points, linearrings
from pygeos import GEOSException
import shapely


cdef _create_node_coordinates_lookup(nodes):
    cdef int i
    ids = np.concatenate([group['id'] for group in nodes])
    lats = np.concatenate([group['lat'] for group in nodes])
    lons = np.concatenate([group['lon'] for group in nodes])
    coords = np.stack((lons, lats), axis=-1)
    return {ids[i]: coords[i] for i in range(0, len(ids))}


cdef get_relation_multipolygon_indices_for_osm_key(relations, osm_key):
    """
    osm_key is e.g. 'building' which would return all relation indices for buildings.
    Other possible keys are e.g. 'landuse' or 'amenity'
    """
    cdef int i, n = len(relations["tags"])
    indices = []
    for i in range(0, n):
        tag = relations["tags"][i]
        if "type" in tag.keys():
            if tag["type"] in ["multipolygon"]:
                for k, v in tag.items():
                    if osm_key in k:
                        indices.append(i)
                        break
    return indices


cdef pygeos_to_shapely(geom):
    if geom is None:
        return None
    geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
    return shapely.geometry.base.geom_factory(geom)


cdef to_shapely(pygeos_array):
    out = np.empty(len(pygeos_array), dtype=object)
    out[:] = [pygeos_to_shapely(geom) for geom in pygeos_array]
    return out


cdef get_way_coordinates_for_polygon(node_coordinate_lookup, way_elements):
    cdef int i, ii, nn, n = len(way_elements["id"])
    features = []
    for i in range(0, n):
        way_nodes = way_elements["nodes"][i]
        nn = len(way_nodes)
        coords = []
        for ii in range(0, nn):
            node = way_nodes[ii]
            try:
                coords.append((node_coordinate_lookup[node][0],
                               node_coordinate_lookup[node][1]))
            except:
                pass
        features.append(coords)
    return features


cdef create_linear_ring(coordinates):
    try:
        return linearrings(coordinates)
    except GEOSException as e:
        if "Invalid number of points" in str(e):
            return None
        elif "point array must contain" in str(e):
            return None
        raise e
    except Exception as e:
        raise e


cdef _create_point_geometries(xarray, yarray):
    cdef:
        int N = len(xarray)
        float x, y

    geometries = []
    for i in range(0, N):
        coords = (xarray[i], yarray[i])
        geometries.append(coords)

    return to_shapely(np.array(
        [points(geom)
         if geom is not None else None
         for geom in geometries],
        dtype=object))


cdef _create_way_geometries(node_coordinates, way_elements):
    cdef long long node
    cdef list coords, way_nodes
    cdef int i, ii, nn, n = len(way_elements['id'])

    geometries = []
    for i in range(0, n):
        way_nodes = way_elements['nodes'][i]
        coords = []
        nn = len(way_nodes)
        for ii in range(0, nn):
            node = way_nodes[ii]
            try:
                coords.append((node_coordinates[node][0],
                               node_coordinates[node][1]))
            except:
                pass
        if len(coords) > 1:
            geometries.append(coords)
        else:
            geometries.append(None)
    return to_shapely(np.array(
        [linestrings(geom)
         if geom is not None else None
         for geom in geometries],
        dtype=object))


cdef create_pygeos_polygon_from_relation(node_coordinates, relation_ways, member_roles):
    cdef int i, m_cnt
    cdef str role

    # Get coordinates for relation
    coordinates = get_way_coordinates_for_polygon(node_coordinates, relation_ways)

    shell = []
    holes = []
    m_cnt = len(member_roles)
    for i in range(0, m_cnt):
        role = member_roles[i]
        if role == "outer":
            ring = create_linear_ring(coordinates[i])
            if ring is not None:
                shell.append(ring)

        elif role == "inner":
            ring = create_linear_ring(coordinates[i])
            if ring is not None:
                holes.append(ring)

        else:
            raise ValueError("Got invalid member role: " + str(role))

    if len(shell) == 0:
        return None

    elif len(shell) == 1:
        shell = shell[0]

    if len(holes) == 0:
        holes = None

    return polygons(shell, holes)

cdef _create_polygon_geometries(node_coordinates, way_elements):
    cdef long long node
    cdef list coords, nodes_
    cdef int n = len(way_elements['id'])
    cdef int i, ii, nn

    geometries = []

    for i in range(0, n):
        nodes_ = way_elements['nodes'][i]
        coords = []

        nn = len(nodes_)
        for ii in range(0, nn):
            node = nodes_[ii]
            try:
                coords.append((node_coordinates[node][0],
                               node_coordinates[node][1]))
            except:
                pass

        if len(coords) > 2:
            try:
                geometries.append(polygons(coords))
            except GEOSException as e:
                # Some geometries might not be valid for creating a Polygon
                # These might occur e.g. at the edge of the spatial extent
                if "Invalid number of points in LinearRing" in str(e):
                    geometries.append(None)
                else:
                    raise e
            except Exception as e:
                raise e

        else:
            geometries.append(None)

    return to_shapely(geometries)


cpdef create_node_coordinates_lookup(nodes):
    return _create_node_coordinates_lookup(nodes)


cpdef create_point_geometries(xarray, yarray):
    return _create_point_geometries(xarray, yarray)


cpdef create_way_geometries(node_coordinates, way_elements):
    return _create_way_geometries(node_coordinates, way_elements)


cpdef create_polygon_geometries(node_coordinates, way_elements):
    return _create_polygon_geometries(node_coordinates, way_elements, )
