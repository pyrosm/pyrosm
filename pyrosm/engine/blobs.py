"""Blob framing: index a PBF file's blobs and read/decompress one ``PrimitiveBlock``."""

import os
import struct
import zlib

from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob


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
    """Read and decompress one ``Blob`` payload into the raw ``PrimitiveBlock`` bytes."""
    f.seek(offset)
    blob = Blob()
    blob.ParseFromString(f.read(size))
    if blob.HasField("zlib_data"):
        return zlib.decompress(blob.zlib_data)
    if blob.HasField("raw"):
        return blob.raw
    if blob.HasField("lzma_data"):
        import lzma

        return lzma.decompress(blob.lzma_data)
    raise ValueError("Unsupported Blob compression in '%s'." % f.name)
