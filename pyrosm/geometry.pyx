import numpy as np
from pygeos import linestrings
import shapely

cdef create_node_lookup_dict(nodes):
    cdef int i
    ids = np.concatenate([group['id'] for group in nodes])
    lats = np.concatenate([group['lat'] for group in nodes])
    lons = np.concatenate([group['lon'] for group in nodes])
    coords = np.stack((lons, lats), axis=-1)
    return {ids[i]: coords[i] for i in range(0, len(ids))}

cdef pygeos_to_shapely(geom):
    if geom is None:
        return None
    geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
    return shapely.geometry.base.geom_factory(geom)

cdef to_shapely(pygeos_array):
    out = np.empty(len(pygeos_array), dtype=object)
    out[:] = [pygeos_to_shapely(geom) for geom in pygeos_array]
    return out

cdef _create_way_geometries(nodes, ways):
    cdef dict way
    cdef long long node
    cdef list coords, way_nodes
    cdef int i, ii, nn, n=len(ways['id'])

    # Lookup for all nodes that are available for given way
    lookup_dict = create_node_lookup_dict(nodes)

    geometries = []
    for i in range(0, n):
        way_nodes = ways['nodes'][i]
        coords = []
        nn = len(way_nodes)
        for ii in range(0, nn):
            node = way_nodes[ii]
            try:
                coords.append((lookup_dict[node][0], lookup_dict[node][1]))
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

cpdef create_way_geometries(nodes, ways):
    return _create_way_geometries(nodes, ways)

