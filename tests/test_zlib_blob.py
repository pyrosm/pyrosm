import lzma
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


@pytest.mark.parametrize("data", [DATA, b""])
def test_zlib_compress(data):
    """libdeflate compresses to a zlib stream that zlib and libdeflate both read."""
    from pyrosm.zlib_blob import zlib_compress, zlib_decompress

    stream = zlib_compress(data)
    assert zlib.decompress(stream) == data
    assert zlib_decompress(stream, len(data)) == data
    for level in (-1, 13, 2**70):
        with pytest.raises(ValueError, match="level must be 0 to 12"):
            zlib_compress(data, level)
    with pytest.raises(TypeError):
        zlib_compress(bytearray(data))


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


def _blob(**fields):
    from pyrosm.proto.fileformat_pb2 import Blob

    return Blob(**fields).SerializeToString()


_GOOD = _blob(zlib_data=zlib.compress(DATA), raw_size=len(DATA))


@pytest.mark.parametrize(
    "case, blob, extra, expected",
    [
        ("zlib data", _GOOD, 0, DATA),
        ("empty zlib data", _blob(zlib_data=zlib.compress(b""), raw_size=0), 0, b""),
        ("no raw size", _blob(zlib_data=zlib.compress(DATA)), 0, None),
        ("raw data", _blob(raw=DATA, raw_size=len(DATA)), 0, None),
        ("lzma data", _blob(lzma_data=lzma.compress(DATA), raw_size=len(DATA)), 0, None),
        ("raw and zlib data", _blob(raw=DATA) + _GOOD, 0, None),
        ("field number 0", b"\x00" + _GOOD, 0, None),
        (
            "varint above 64 bits",
            b"\x10" + b"\xff" * 9 + b"\x02" + _blob(zlib_data=zlib.compress(DATA)),
            0,
            None,
        ),
        ("wrong raw size", _blob(zlib_data=zlib.compress(DATA), raw_size=7), 0, None),
        (
            "raw size above the PBF limit",
            _blob(zlib_data=zlib.compress(DATA), raw_size=2**25 + 1),
            0,
            None,
        ),
        ("cut off", _GOOD[:-5], 0, None),
        ("past the file end", _GOOD, 9, None),
    ],
)
def test_inflate_blobs_nogil(case, blob, extra, expected):
    """A batch of zlib blobs with their raw sizes decompresses into one buffer; a batch
    holding any other blob raises ValueError, for the caller to read it blob by blob."""
    import numpy as np

    from pyrosm.zlib_blob import inflate_blobs_nogil

    view = np.frombuffer(_GOOD + blob, dtype=np.uint8)
    offsets = np.array([0, len(_GOOD)])
    sizes = np.array([len(_GOOD), len(blob) + extra])
    if expected is None:
        with pytest.raises(ValueError, match="blob 1 "):
            inflate_blobs_nogil(view, offsets, sizes)
        return
    buffer, starts = inflate_blobs_nogil(view, offsets, sizes)
    assert starts.tolist() == [0, len(DATA), len(DATA) + len(expected)]
    assert buffer.tobytes() == DATA + expected


_RAW = _blob(raw=DATA, raw_size=len(DATA))
_LZMA = _blob(lzma_data=lzma.compress(DATA), raw_size=len(DATA))
_NOT_ZLIB = _blob(zlib_data=b"not zlib", raw_size=len(DATA))


@pytest.mark.parametrize(
    "case, blobs, max_bytes, blocks",
    [
        ("all fit", [_GOOD, _GOOD, _GOOD], 3 * len(DATA), 3),
        ("two fit", [_GOOD, _GOOD, _GOOD], 3 * len(DATA) - 1, 2),
        ("the first in any case", [_GOOD, _GOOD], 1, 1),
        ("up to a blob it does not handle", [_GOOD, _RAW, _GOOD], 3 * len(DATA), 1),
        ("up to an lzma blob", [_GOOD, _LZMA, _GOOD], 3 * len(DATA), 1),
        ("up to a blob that does not decompress", [_GOOD, _NOT_ZLIB], 3 * len(DATA), 1),
        ("a first blob it does not handle", [_RAW, _GOOD], 3 * len(DATA), None),
        ("a first lzma blob", [_LZMA, _GOOD], 3 * len(DATA), None),
        ("a first blob that does not decompress", [_NOT_ZLIB], 3 * len(DATA), None),
    ],
)
def test_inflate_blobs_nogil_with_max_bytes(case, blobs, max_bytes, blocks):
    """With max_bytes only the first blobs, whose raw sizes add up to at most it (the
    first one in any case), are decompressed, up to a blob it does not handle, which
    raises only when it is the first."""
    import numpy as np

    from pyrosm.zlib_blob import inflate_blobs_nogil

    view = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    sizes = np.array([len(b) for b in blobs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    if blocks is None:
        with pytest.raises(ValueError, match="blob 0 "):
            inflate_blobs_nogil(view, offsets, sizes, max_bytes)
        return
    buffer, starts = inflate_blobs_nogil(view, offsets, sizes, max_bytes)
    assert starts.tolist() == [len(DATA) * i for i in range(blocks + 1)]
    assert buffer.tobytes() == DATA * blocks
    # The rest of the array the blocks view holds no leftover bytes.
    assert not buffer.base[len(buffer) :].any()


@pytest.mark.parametrize(
    "data, offsets, sizes, error",
    [
        (_GOOD, [0], [], "same length"),
        (_GOOD, [], [len(_GOOD)], "same length"),
        (b"", [0], [0], "blob 0 "),
    ],
)
def test_inflate_blobs_nogil_checks_arguments(data, offsets, sizes, error):
    import numpy as np

    from pyrosm.zlib_blob import inflate_blobs_nogil

    view = np.frombuffer(data, dtype=np.uint8)
    with pytest.raises(ValueError, match=error):
        inflate_blobs_nogil(
            view, np.array(offsets, dtype=np.int64), np.array(sizes, dtype=np.int64)
        )


def test_inflate_blobs_nogil_corrupt_batches(corrupted_copies):
    """A corrupted batch of blobs decompresses or raises ValueError; it never reads
    outside its data or returns blocks that do not fit the buffer."""
    import numpy as np

    from pyrosm.zlib_blob import inflate_blobs_nogil

    batch = _GOOD * 3
    offsets = np.array([0, len(_GOOD), 2 * len(_GOOD)], dtype=np.int64)
    sizes = np.full(3, len(_GOOD), dtype=np.int64)
    for data in corrupted_copies(batch, 300, seed=1):
        view = np.frombuffer(data, dtype=np.uint8)
        for max_bytes in (-1, len(DATA)):
            try:
                buffer, starts = inflate_blobs_nogil(view, offsets, sizes, max_bytes)
            except ValueError:
                continue
            assert np.all(np.diff(starts) >= 0)
            assert starts[-1] == len(buffer)
