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
cdef enum:
    _MAX_BLOB_SIZE = 32 * 1024 * 1024
MAX_BLOB_SIZE = _MAX_BLOB_SIZE


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


# --- blobs read and decompressed with the GIL released ---------------------------------

from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport int64_t, uint64_t
import numpy as np

from pyrosm.primitive_block_decoder cimport WIRE_LENGTH, WIRE_VARINT, _read_varint

# The fields of fileformat.proto's Blob message read here.
cdef enum BlobField:
    BLOB_RAW_SIZE = 2
    BLOB_ZLIB_DATA = 3


cdef int _blob_fields(const unsigned char* buf, Py_ssize_t end, Py_ssize_t* raw_size,
                      Py_ssize_t* data_start, Py_ssize_t* data_end) noexcept nogil:
    """The raw size and the zlib data's span of the ``Blob`` message ``buf[:end]``: 0, or
    -1 unless it holds exactly a raw size (at most ``MAX_BLOB_SIZE``) and zlib data, once
    each and well encoded."""
    cdef Py_ssize_t pos = 0
    cdef uint64_t tag, length
    raw_size[0] = -1
    data_start[0] = -1
    # A field's tag is its number << 3 | its wire type. A failed read moves ``pos`` past
    # ``end``.
    while pos < end:
        tag = _read_varint(buf, &pos, end)
        if tag == (BLOB_RAW_SIZE << 3 | WIRE_VARINT) and raw_size[0] < 0:
            length = _read_varint(buf, &pos, end)
            if pos > end or length > _MAX_BLOB_SIZE:
                return -1
            raw_size[0] = <Py_ssize_t>length
        elif tag == (BLOB_ZLIB_DATA << 3 | WIRE_LENGTH) and data_start[0] < 0:
            length = _read_varint(buf, &pos, end)
            if pos > end or length > <uint64_t>(end - pos):
                return -1
            data_start[0] = pos
            data_end[0] = pos + <Py_ssize_t>length
            pos += <Py_ssize_t>length
        else:
            return -1
    if pos != end or raw_size[0] < 0 or data_start[0] < 0 or data_end[0] == data_start[0]:
        return -1
    return 0


def inflate_blobs_nogil(const unsigned char[::1] data, const int64_t[::1] offsets,
                        const int64_t[::1] sizes, Py_ssize_t max_bytes=-1):
    """Decompress the zlib ``Blob`` messages at ``offsets``/``sizes`` of ``data`` (bytes
    of a PBF) into one buffer: ``(buffer, starts)``, block ``i`` being
    ``buffer[starts[i]:starts[i + 1]]``. The GIL is held only to allocate the buffer.
    Raises ``ValueError`` when a blob is not zlib data of its declared ``raw_size`` (the
    caller reads such a batch blob by blob). With ``max_bytes`` only the first blobs,
    whose raw sizes add up to at most ``max_bytes`` (the first one in any case), are
    decompressed, up to a blob it does not handle, which raises only when it is the
    first; ``starts`` then holds fewer blocks. ``data`` must not change during the
    call."""
    cdef Py_ssize_t n = offsets.shape[0], m = 0, i, raw_size = 0, d_start = 0, d_end = 0
    cdef Py_ssize_t length = data.shape[0]
    cdef int64_t offset, size
    if sizes.shape[0] != n:
        raise ValueError("offsets and sizes must have the same length")
    if n and length == 0:
        raise ValueError("blob 0 is not zlib data with a raw size")
    cdef const unsigned char* base = &data[0] if length > 0 else NULL
    starts = np.zeros(n + 1, dtype=np.int64)
    cdef int64_t[::1] at = starts
    # Blob ``i``'s zlib data is ``data[span[2 * i]:span[2 * i + 1]]``.
    spans = np.empty(2 * n + 1, dtype=np.int64)
    cdef int64_t[::1] span = spans
    cdef int bad = 0
    # Two passes: the first checks each blob and keeps its raw size and zlib data's span, so
    # the output is allocated once and block ``i`` goes to ``buffer[at[i]:at[i + 1]]``; the
    # second decompresses those spans. With ``max_bytes`` a blob that is not handled ends
    # the part before it; it then comes first in the caller's next call, which raises, and
    # the caller reads it on its own.
    with nogil:
        for i in range(n):
            # Each offset and size is read once, so the checked values are the used ones.
            offset = offsets[i]
            size = sizes[i]
            if (offset < 0 or size < 0 or offset > length - size
                    or _blob_fields(base + offset, size, &raw_size, &d_start,
                                    &d_end) < 0):
                bad = max_bytes < 0 or i == 0
                break
            if raw_size > PY_SSIZE_T_MAX - 1 - at[i]:
                bad = 2
                break
            if max_bytes >= 0 and i > 0 and at[i] + raw_size > max_bytes:
                break
            span[2 * i] = offset + d_start
            span[2 * i + 1] = offset + d_end
            at[i + 1] = at[i] + raw_size
            m = i + 1
    if bad == 2:
        raise ValueError("the blobs' raw sizes add up to more than a buffer can hold")
    if bad:
        raise ValueError("blob %d is not zlib data with a raw size" % i)
    # One spare byte, so that the output pointer is never NULL.
    buffer = np.empty(at[m] + 1, dtype=np.uint8)
    cdef unsigned char[::1] out = buffer
    cdef unsigned char* target = &out[0]
    cdef size_t actual = 0
    cdef libdeflate_decompressor* decompressor = libdeflate_alloc_decompressor()
    cdef libdeflate_result result = LIBDEFLATE_SUCCESS
    if decompressor == NULL:
        raise MemoryError()
    with nogil:
        for i in range(m):
            raw_size = at[i + 1] - at[i]
            result = libdeflate_zlib_decompress(
                decompressor, base + span[2 * i], span[2 * i + 1] - span[2 * i],
                target + at[i], raw_size, &actual)
            if result != LIBDEFLATE_SUCCESS or <Py_ssize_t>actual != raw_size:
                bad = 1
                break
    libdeflate_free_decompressor(decompressor)
    if bad:
        if max_bytes < 0 or i == 0:
            raise ValueError("blob %d did not decompress to its raw size" % i)
        m = i
    # The returned blocks view ``buffer``: nothing after them stays uninitialised.
    buffer[at[m]:] = 0
    return buffer[: at[m]], starts[: m + 1]
