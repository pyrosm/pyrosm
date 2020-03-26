import numpy as np
from pygeos import linestrings
import shapely

cdef create_node_lookup_dict(nodes):
    ids = np.concatenate([group['id'] for group in nodes])
    lats = np.concatenate([group['lat'] for group in nodes])
    lons = np.concatenate([group['lon'] for group in nodes])
    coords = np.stack((lons, lats), axis=-1)
    return {ids[i]: coords[i] for i in range(len(ids))}

cdef pygeos_to_shapely(geom):
    if geom is None:
        return None
    geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
    return shapely.geometry.base.geom_factory(geom)

cdef to_shapely(pygeos_array):
    out = np.empty(len(pygeos_array), dtype=object)
    out[:] = [pygeos_to_shapely(geom) for geom in pygeos_array]
    return out

cdef create_way_geometries(nodes, ways):
    cdef dict way, lookup_dict
    cdef long long node
    cdef list geometries, coords
    lookup_dict = create_node_lookup_dict(nodes)

    geometries = []
    for way in ways:
        coords = []
        for node in way['nodes']:
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
