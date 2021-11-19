# cython: language_level=3

import numpy as np
cimport numpy as np

import pygeos

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

    bounding_box_is_polygon = isinstance(bounding_box, pygeos.Geometry)

    if bounding_box_is_polygon:
        xmin, ymin, xmax, ymax = pygeos.bounds(bounding_box).tolist()
    else:
        xmin, ymin, xmax, ymax = bounding_box

    in_bounding_box = (
        (xmin <= lons)
        & (lons <= xmax)
        & (ymin <= lats)
        & (lats <= ymax)
    )

    if bounding_box_is_polygon:
        points = pygeos.empty(len(in_bounding_box))
        points[in_bounding_box] = pygeos.points(lats[in_bounding_box], lons[in_bounding_box])
        bounding_box = pygeos.prepare(bounding_box)
        in_bounding_box[in_bounding_box] = pygeos.contains(bounding_box, points[in_bounding_box])

    return in_bounding_box
