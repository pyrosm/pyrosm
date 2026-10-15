"""zlib decompression of PBF blobs with libdeflate (vendored in ``vendor/libdeflate``),
which needs the decompressed size up front: a blob declares it as ``raw_size``."""

from libc.stddef cimport size_t
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

cdef extern from "libdeflate.h" nogil:
    struct libdeflate_decompressor
    enum libdeflate_result:
        LIBDEFLATE_SUCCESS
    libdeflate_decompressor* libdeflate_alloc_decompressor()
    void libdeflate_free_decompressor(libdeflate_decompressor* decompressor)
    libdeflate_result libdeflate_zlib_decompress(
        libdeflate_decompressor* decompressor, const void* in_data, size_t in_nbytes,
        void* out, size_t out_nbytes_avail, size_t* actual_out_nbytes_ret)

# The largest decompressed blob the PBF format allows.
MAX_BLOB_SIZE = 32 * 1024 * 1024


def zlib_decompress(data, Py_ssize_t size):
    """The zlib stream ``data`` decompressed, when it holds exactly ``size`` bytes (at
    most ``MAX_BLOB_SIZE``); None otherwise: corrupt, truncated or of another size."""
    cdef const unsigned char[::1] view = data
    cdef size_t actual = 0
    cdef libdeflate_result result
    cdef libdeflate_decompressor* decompressor
    if view.shape[0] == 0 or size < 0 or size > MAX_BLOB_SIZE:
        return None
    out = PyBytes_FromStringAndSize(NULL, size)
    cdef char* buffer = PyBytes_AS_STRING(out)
    decompressor = libdeflate_alloc_decompressor()
    if decompressor == NULL:
        raise MemoryError()
    with nogil:
        result = libdeflate_zlib_decompress(
            decompressor, &view[0], view.shape[0], buffer, size, &actual
        )
    libdeflate_free_decompressor(decompressor)
    if result != LIBDEFLATE_SUCCESS or <Py_ssize_t>actual != size:
        return None
    return out
