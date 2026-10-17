"""Blob framing: index a PBF file's blobs and read/decompress one ``PrimitiveBlock``."""

import os
import struct
import zlib

from pyrosm.exceptions import InvalidOSMFileError
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.zlib_blob import zlib_decompress

# The OSM PBF format's limit for a blob, compressed and decompressed.
_MAX_BLOB_SIZE = 32 * 1024 * 1024


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
