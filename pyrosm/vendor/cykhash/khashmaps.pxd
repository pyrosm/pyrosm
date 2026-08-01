from libc.stdint cimport  uint64_t, uint32_t,  int64_t,  int32_t
from cpython.object cimport PyObject

from .floatdef cimport float64_t, float32_t

### Common definitions:
ctypedef PyObject* pyobject_t

include "common.pxi"
include "memory.pxi"
include "khash.pxi"
include "murmurhash.pxi"


#utilities for int<->float
include "hash_functions.pxi"

# different implementation
include "maps/map_header.pxi"

