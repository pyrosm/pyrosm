"""zlib compression and decompression of PBF blobs with libdeflate (vendored in
``vendor/libdeflate``). Decompression needs the decompressed size up front: a blob
declares it as ``raw_size``."""

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
    struct libdeflate_compressor
    libdeflate_compressor* libdeflate_alloc_compressor(int compression_level)
    void libdeflate_free_compressor(libdeflate_compressor* compressor)
    size_t libdeflate_zlib_compress_bound(libdeflate_compressor* compressor,
                                          size_t in_nbytes)
    size_t libdeflate_zlib_compress(
        libdeflate_compressor* compressor, const void* in_data, size_t in_nbytes,
        void* out, size_t out_nbytes_avail)

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


def zlib_compress(bytes data not None, level=6):
    """``data`` as a zlib stream, compressed at ``level`` (0, no compression, to 12; 6,
    the default, is also zlib's default level)."""
    cdef const unsigned char[::1] view = data
    cdef size_t n = view.shape[0], bound, size
    cdef const unsigned char* source = &view[0] if n > 0 else NULL
    cdef char* buffer
    cdef libdeflate_compressor* compressor
    if not 0 <= level <= 12:
        raise ValueError("level must be 0 to 12; got %d." % level)
    compressor = libdeflate_alloc_compressor(level)
    if compressor == NULL:
        raise MemoryError()
    try:
        bound = libdeflate_zlib_compress_bound(compressor, n)
        out = PyBytes_FromStringAndSize(NULL, bound)
        buffer = PyBytes_AS_STRING(out)
        with nogil:
            size = libdeflate_zlib_compress(compressor, source, n, buffer, bound)
    finally:
        libdeflate_free_compressor(compressor)
    if size == 0:
        raise RuntimeError("libdeflate could not compress %d bytes." % n)
    return out[:size]
