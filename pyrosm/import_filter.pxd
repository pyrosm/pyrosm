# cython: language_level=3

cimport numpy as np
np.import_array()

cdef np.ndarray in_bounding_box(lats, lons, bounding_box)
