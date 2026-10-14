import zlib

import pytest


DATA = b"pyrosm reads OpenStreetMap data " * 64


@pytest.mark.parametrize(
    "case, stream, size, expected",
    [
        ("declared size", zlib.compress(DATA), len(DATA), DATA),
        ("size too small", zlib.compress(DATA), len(DATA) - 1, None),
        ("size too large", zlib.compress(DATA), len(DATA) + 1, None),
        ("truncated stream", zlib.compress(DATA)[:-8], len(DATA), None),
        ("not a zlib stream", b"not zlib", len(DATA), None),
        ("empty stream", b"", 0, None),
        ("negative size", zlib.compress(DATA), -1, None),
        ("size above the PBF limit", zlib.compress(DATA), 32 * 1024 * 1024 + 1, None),
    ],
)
def test_zlib_decompress(case, stream, size, expected):
    """libdeflate decompresses a zlib stream of exactly its declared size, and gives None
    for anything else, for the callers to fall back to zlib."""
    from pyrosm.zlib_blob import zlib_decompress

    assert zlib_decompress(stream, size) == expected


def test_read_block_falls_back_to_zlib(tmp_path):
    """A reader still decompresses a blob that declares no size, or a wrong one."""
    from pyrosm.engine.blobs import _read_block
    from pyrosm.proto.fileformat_pb2 import Blob

    stream = zlib.compress(DATA)
    blobs = [
        Blob(zlib_data=stream, raw_size=len(DATA)),
        Blob(zlib_data=stream),
        Blob(zlib_data=stream, raw_size=7),
    ]
    path = tmp_path / "blobs.bin"
    path.write_bytes(b"".join(blob.SerializeToString() for blob in blobs))
    with open(path, "rb") as f:
        offset = 0
        for blob in blobs:
            size = len(blob.SerializeToString())
            assert _read_block(f, offset, size) == DATA
            offset += size
