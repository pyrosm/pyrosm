from cykhash.khashmaps cimport Int64toInt64Map


cdef class NodeLocations:
    cdef Int64toInt64Map _id2idx
    cdef const double[::1] _lon
    cdef const double[::1] _lat
    cdef object _ids
    cdef dict _columns
    cdef list _column_order
    cdef bint contains(self, long long node)
    cdef long long index(self, long long node)
    cdef double lon_at(self, long long idx)
    cdef double lat_at(self, long long idx)
    cpdef tuple gather(self, node_ids)
    cdef dict _base_record(self, long long idx)
    cdef dict record(self, long long idx, long long node)
