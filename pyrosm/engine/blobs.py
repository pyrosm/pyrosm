"""Blob framing: index a PBF file's blobs and read/decompress one ``PrimitiveBlock``."""

import os
import struct
import zlib
from operator import itemgetter

import numpy as np

from pyrosm.exceptions import InvalidOSMFileError
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.zlib_blob import MAX_BLOB_SIZE, inflate_blobs_nogil, zlib_decompress

# A batch of blobs read in one go spans at most this many bytes of the file, and is
# decompressed in parts of at most `_BATCH_RAW_BYTES` (one blob holds up to 32 MiB).
_BATCH_BYTES = 32 * 1024 * 1024
_BATCH_RAW_BYTES = 64 * 1024 * 1024


def _read_batches(items, max_count, offset=itemgetter(0), size=itemgetter(1)):
    """``items`` (blobs in file order, each with an ``offset`` and ``size`` in the file) in
    lists of at most ``max_count`` that span at most `_BATCH_BYTES` from the first one's
    offset to the end of the last; a blob larger than that comes alone."""
    batch = []
    first = 0
    for item in items:
        end = offset(item) + size(item)
        if batch and (len(batch) == max_count or end - first > _BATCH_BYTES):
            yield batch
            batch = []
        if not batch:
            first = offset(item)
        batch.append(item)
    if batch:
        yield batch


def _read_span(f, start, end):
    """The bytes ``start:end`` of the file ``f`` read in one go, or None when they span
    more than `_BATCH_BYTES` (a single blob that large) or the read comes short."""
    if end - start > _BATCH_BYTES:
        return None
    raw = np.empty(end - start, dtype=np.uint8)
    f.seek(start)
    return raw if f.readinto(raw) == len(raw) else None


def _inflated_parts(raw, offsets, sizes, read_one):
    """The blobs at ``offsets``/``sizes`` of ``raw`` (None when they could not be read)
    decompressed without the GIL in parts of at most `_BATCH_RAW_BYTES`: yields
    ``(first, end, buffer, starts)`` for the blobs ``first:end``. A blob that
    ``inflate_blobs_nogil`` does not handle comes alone, as ``read_one(i)`` reads it. A
    part is released before the next one is decompressed, so the consumer releases its
    references to it too."""
    i = 0
    while i < len(offsets):
        if raw is not None:
            try:
                buffer, starts = inflate_blobs_nogil(
                    raw, offsets[i:], sizes[i:], _BATCH_RAW_BYTES
                )
            except ValueError:
                pass
            else:
                end = i + len(starts) - 1
                yield i, end, buffer, starts
                del buffer, starts
                i = end
                continue
        block = read_one(i)
        starts = np.array([0, len(block)], dtype=np.int64)
        yield i, i + 1, np.frombuffer(block, dtype=np.uint8), starts
        del block, starts
        i += 1


def _inflated_batch(raw, items, offset, size, read_one):
    """The blobs ``items`` of a batch (each at ``offset(item)`` with ``size(item)`` bytes in
    the file, in file order), whose bytes ``raw`` start at the first one's offset (None when
    they could not be read), decompressed in parts by :func:`_inflated_parts`: yields
    ``(part, buffer, starts)``, ``part`` the items of the part. A blob it does not
    decompress is read by ``read_one(item)``."""
    first = offset(items[0])
    offsets = np.array([offset(item) - first for item in items], dtype=np.int64)
    sizes = np.array([size(item) for item in items], dtype=np.int64)
    parts = _inflated_parts(raw, offsets, sizes, lambda i: read_one(items[i]))
    for start, end, buffer, starts in parts:
        yield items[start:end], buffer, starts
        # Dropped here and by the caller, so no part is alive while the next one is
        # decompressed.
        del buffer, starts


# The OSM PBF format's limit for a blob, compressed and decompressed.
_MAX_BLOB_SIZE = MAX_BLOB_SIZE


def _invalid_pbf(filepath, reason):
    """The error for ``filepath``, which does not follow the OSM PBF format: ``reason``."""
    return InvalidOSMFileError(
        "'%s' is not a valid OSM PBF file. Pyrosm reads OpenStreetMap data in the "
        "OSM PBF format (https://wiki.openstreetmap.org/wiki/PBF_Format); this file "
        "does not follow the OSM PBF schema (%s)." % (filepath, reason)
    )


def _decompress_blob(data, raw_size, filepath, codec="zlib", exact_size=True):
    """The ``zlib`` (or ``lzma``) payload ``data`` of a blob of ``filepath`` decompressed:
    by libdeflate when it is zlib data of the declared ``raw_size``, otherwise by Python's
    codec, to at most 32 MiB. A blob larger than that, cut off or corrupt, or with
    ``exact_size`` one that is not ``raw_size`` bytes, raises the error naming the
    file."""
    if codec == "zlib" and raw_size is not None:
        out = zlib_decompress(data, raw_size)
        if out is not None:
            return out
    if codec == "zlib":
        decompressor, codec_error = zlib.decompressobj(), zlib.error
    else:
        import lzma

        decompressor, codec_error = lzma.LZMADecompressor(), lzma.LZMAError
    try:
        out = decompressor.decompress(data, _MAX_BLOB_SIZE)
    except codec_error as err:
        raise _invalid_pbf(filepath, err) from None
    if not decompressor.eof:
        if codec == "zlib":
            larger = bool(decompressor.unconsumed_tail)
        else:
            larger = not decompressor.needs_input
        if larger:
            raise _invalid_pbf(
                filepath, "a blob is larger than 32 MiB when decompressed"
            )
        raise _invalid_pbf(filepath, "a %s blob is truncated" % codec)
    if exact_size and raw_size is not None and raw_size != len(out):
        raise _invalid_pbf(
            filepath,
            "a blob decompresses to %d bytes but declares raw_size %d"
            % (len(out), raw_size),
        )
    return out


def _index_blobs(filepath):
    """One cheap sequential pass: ``(type, data_offset, data_size)`` per blob, reading
    only the ``BlobHeader``s and skipping the payloads (no decompression)."""
    blobs = []
    with open(filepath, "rb") as f:
        while True:
            header_len_bytes = f.read(4)
            if len(header_len_bytes) < 4:
                break
            (header_len,) = struct.unpack("!L", header_len_bytes)
            header = BlobHeader()
            header.ParseFromString(f.read(header_len))
            offset = f.tell()
            blobs.append((header.type, offset, header.datasize))
            f.seek(header.datasize, os.SEEK_CUR)
    return blobs


def _data_blob_spans(blobs):
    """The data blobs of ``blobs`` (from :func:`_index_blobs`) as ``(offset, size, span)``,
    with ``span`` the blob's length in the file (its length prefix, header and payload), and
    the summed span of the other blobs. All the spans together cover the indexed bytes.
    """
    data, other, end = [], 0, 0
    for blob_type, offset, size in blobs:
        span = offset + size - end
        end = offset + size
        if blob_type == "OSMData":
            data.append((offset, size, span))
        else:
            other += span
    return data, other


def _read_block(f, offset, size):
    """Read and decompress one ``Blob`` payload into the raw ``PrimitiveBlock`` bytes, of
    at most 32 MiB; a declared ``raw_size`` that does not match is not an error."""
    f.seek(offset)
    blob = Blob()
    blob.ParseFromString(f.read(size))
    raw_size = blob.raw_size if blob.HasField("raw_size") else None
    if blob.HasField("zlib_data"):
        return _decompress_blob(blob.zlib_data, raw_size, f.name, exact_size=False)
    if blob.HasField("raw"):
        return blob.raw
    if blob.HasField("lzma_data"):
        return _decompress_blob(blob.lzma_data, raw_size, f.name, "lzma", False)
    raise ValueError("Unsupported Blob compression in '%s'." % f.name)
