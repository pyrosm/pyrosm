# cython: language_level=3

import numpy as np
cimport numpy as np

import shapely.geometry
import pygeos
#from pygeos._geos import GEOSGeometry
#cimport pygeos._geos

np.import_array()


#cdef np.ndarray in_bounding_box(
#        lats: np.ndarray,
#        lons: np.ndarray,
#        bounding_box: float[:],
#        extent: pygeos.Geometry = pygeos.Geometry("POLYGON EMPTY")
#):
cdef np.ndarray in_bounding_box(lats, lons, bounding_box):
    cdef:
        np.ndarray in_bounding_box
        np.ndarray points
        double xmin
        double ymin
        double xmax
        double ymax
        bint bounding_box_is_polygon
        #pygeos._geos.GEOSGeometry *bounding_box_polygon

    bounding_box_is_polygon = isinstance(bounding_box, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))

    if bounding_box_is_polygon:
        xmin, ymin, xmax, ymax = bounding_box.bounds
    else:
        xmin, ymin, xmax, ymax = bounding_box

    print(xmin, ymin, xmax, ymax)

    # first, test whether the coordinates are within the bounds
    in_bounding_box = (
        (xmin <= lons)
        & (lons <= xmax)
        & (ymin <= lats)
        & (lats <= ymax)
    )

    if bounding_box_is_polygon:
        # if we got a polygon, let’s now _also_ test actual
        # geometry predicates, only on those rows that fell
        # within the bounds

        # ‘deep-ish’ copy:
        bounding_box_polygon = pygeos.from_shapely(bounding_box)
        pygeos.prepare(bounding_box_polygon)

        points = pygeos.empty(len(in_bounding_box))
        points[in_bounding_box] = pygeos.points(lons[in_bounding_box], lats[in_bounding_box])

        in_bounding_box[in_bounding_box] = pygeos.contains(bounding_box_polygon, points[in_bounding_box])

    return in_bounding_box
