"""Out-of-core engine buildings reader: parity vs the in-memory OSM(fp).get_buildings()
way and relation rows, plus the output= GeoParquet path and the worker-count policy."""

import zlib
from struct import pack, unpack

import geopandas as gpd
import numpy as np
import pytest
import shapely

from pyrosm import OSM, get_data
from pyrosm.engine import get_buildings
from pyrosm.engine import pool, geoparquet
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _rewrite_pbf_raw(src, dst):
    """Re-encode every blob of ``src`` with an uncompressed ``raw`` payload. Bundled
    extracts are all zlib, so this synthesises a file that exercises the engine's ``raw``
    decompression branch."""
    with open(src, "rb") as f, open(dst, "wb") as out:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = unpack("!L", head)
            bh = BlobHeader()
            bh.ParseFromString(f.read(n))
            blob = Blob()
            blob.ParseFromString(f.read(bh.datasize))
            raw = (
                zlib.decompress(blob.zlib_data)
                if blob.HasField("zlib_data")
                else blob.raw
            )
            new_blob = Blob()
            new_blob.raw = raw
            blob_bytes = new_blob.SerializeToString()
            bh.datasize = len(blob_bytes)
            header_bytes = bh.SerializeToString()
            out.write(pack("!L", len(header_bytes)))
            out.write(header_bytes)
            out.write(blob_bytes)


def _in_memory_buildings(fp):
    # Way and relation building rows from the in-memory reader (the out-of-core engine
    # must reproduce both). osm_type makes the comparison key unambiguous since a way and
    # a relation can share an id number.
    gdf = OSM(fp).get_buildings()
    return gdf.sort_values(["osm_type", "id"]).reset_index(drop=True)


def _assert_matches(mine, ref):
    a = mine.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = ref.sort_values(["osm_type", "id"]).reset_index(drop=True)
    # Same element kinds and ids.
    np.testing.assert_array_equal(a["osm_type"].to_numpy(), b["osm_type"].to_numpy())
    np.testing.assert_array_equal(a["id"].to_numpy(), b["id"].to_numpy())
    # Same geometries, exact coordinates (order-canonical via normalize).
    na = gpd.GeoSeries(shapely.normalize(a.geometry.values))
    nb = gpd.GeoSeries(shapely.normalize(b.geometry.values))
    assert na.geom_equals_exact(nb, tolerance=0).all()


def _assert_full_parity(mine, ref):
    import pandas as pd

    a = mine.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = ref.sort_values(["osm_type", "id"]).reset_index(drop=True)
    assert set(a.columns) == set(b.columns), set(a.columns).symmetric_difference(
        b.columns
    )
    a = a[b.columns]
    na = gpd.GeoSeries(shapely.normalize(a.geometry.values))
    nb = gpd.GeoSeries(shapely.normalize(b.geometry.values))
    assert na.geom_equals_exact(nb, tolerance=0).all()
    for col in b.columns:
        if col == "geometry":
            continue
        pd.testing.assert_series_equal(
            a[col], b[col], check_dtype=False, check_names=False
        )


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_buildings_match_in_memory(fixture, request):
    fp = request.getfixturevalue(fixture)
    mine = get_buildings(fp)
    ref = _in_memory_buildings(fp)
    assert mine is not None and len(mine) == len(ref) > 0
    _assert_matches(mine, ref)


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_buildings_full_column_parity(fixture, request):
    # Every tag column, the JSON 'tags' column and the metadata columns must match
    # OSM().get_buildings() column-for-column and value-for-value.
    fp = request.getfixturevalue(fixture)
    mine = get_buildings(fp)
    ref = OSM(fp).get_buildings()
    _assert_full_parity(mine, ref)


def test_engine_buildings_keep_metadata_false_parity(helsinki_pbf):
    # keep_metadata=False must drop the element-metadata columns exactly as the in-memory
    # reader does (whatever its handling) -- full column + value parity either way.
    mine = get_buildings(helsinki_pbf, keep_metadata=False)
    ref = OSM(helsinki_pbf, keep_metadata=False).get_buildings()
    assert "changeset" not in mine.columns
    _assert_full_parity(mine, ref)


def test_engine_building_relations_match(helsinki_pbf):
    # The bundled Helsinki extract has building relations (multipolygons); they must
    # match the in-memory reader's relation rows exactly.
    mine = get_buildings(helsinki_pbf)
    ref = OSM(helsinki_pbf).get_buildings()
    n_rel = int((ref["osm_type"] == "relation").sum())
    assert n_rel > 0
    assert int((mine["osm_type"] == "relation").sum()) == n_rel
    _assert_matches(
        mine[mine["osm_type"] == "relation"], ref[ref["osm_type"] == "relation"]
    )


def test_engine_buildings_parallel_matches_single(helsinki_pbf):
    # The multiprocessing path must produce the same result as the in-process path.
    single = get_buildings(helsinki_pbf, workers=1)
    parallel = get_buildings(helsinki_pbf, workers=3)
    _assert_matches(parallel, single.sort_values("id").reset_index(drop=True))


def test_engine_buildings_raw_blob_matches(helsinki_pbf, tmp_path):
    # Uncompressed `raw` blobs must decode to the same result as the zlib originals.
    raw_fp = str(tmp_path / "helsinki_raw.osm.pbf")
    _rewrite_pbf_raw(helsinki_pbf, raw_fp)
    zlib_buildings = get_buildings(helsinki_pbf)
    raw_buildings = get_buildings(raw_fp)
    _assert_matches(
        raw_buildings, zlib_buildings.sort_values("id").reset_index(drop=True)
    )


def test_engine_buildings_output_parquet_matches(helsinki_pbf, tmp_path):
    # The streamed GeoParquet must reload equal to the in-memory frame.
    pytest.importorskip("pyarrow")
    out = str(tmp_path / "buildings.parquet")
    in_memory = get_buildings(helsinki_pbf)
    returned = get_buildings(helsinki_pbf, output=out)
    assert returned == out
    reloaded = gpd.read_parquet(out)
    assert isinstance(reloaded, gpd.GeoDataFrame)
    assert reloaded.crs == in_memory.crs
    _assert_matches(reloaded, in_memory.sort_values("id").reset_index(drop=True))


def test_engine_buildings_output_is_chunked(helsinki_pbf, tmp_path, monkeypatch):
    # Tiny chunks must stream multiple row groups AND still carry every column: a tag
    # column that first occurs only in a later chunk must not be dropped from the output.
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    in_memory = get_buildings(helsinki_pbf)
    full_out = str(tmp_path / "buildings_full.parquet")
    get_buildings(helsinki_pbf, output=full_out)
    full_reload = gpd.read_parquet(full_out)

    monkeypatch.setattr(geoparquet, "_OUTPUT_CHUNK_SIZE", 20)
    chunked_out = str(tmp_path / "buildings_chunked.parquet")
    get_buildings(helsinki_pbf, output=chunked_out)
    assert pq.ParquetFile(chunked_out).metadata.num_row_groups > 1
    chunked_reload = gpd.read_parquet(chunked_out)

    # No column dropped: the chunked output carries exactly the in-memory reader's columns,
    # and is identical value-for-value to the single-chunk output.
    assert set(chunked_reload.columns) == set(in_memory.columns)
    _assert_full_parity(chunked_reload, full_reload)


def test_engine_output_requires_pyarrow(helsinki_pbf, tmp_path, monkeypatch):
    # Without pyarrow, output= must fail fast with an actionable error (and not decode).
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    with pytest.raises(ImportError, match="pyarrow"):
        get_buildings(helsinki_pbf, output=str(tmp_path / "x.parquet"))


def test_auto_workers_decides_on_file_size_not_blob_count(tmp_path):
    # The default worker count is chosen by file size: parallelising only pays off above
    # ~70 MB, and blob count must not force a pool for a small file (sparse files give a
    # size without occupying disk).
    import os

    small = tmp_path / "small.osm.pbf"
    small.touch()
    os.truncate(small, pool._PARALLEL_MIN_FILE_BYTES - 1)
    assert pool._auto_workers(str(small), 10_000) == 1

    big = tmp_path / "big.osm.pbf"
    big.touch()
    os.truncate(big, pool._PARALLEL_MIN_FILE_BYTES + 1)
    cpus = os.cpu_count() or 1
    assert pool._auto_workers(str(big), 10_000) == cpus
    assert pool._auto_workers(str(big), 2) == min(cpus, 2)


def test_read_block_decodes_lzma_and_rejects_unsupported(tmp_path):
    # _read_block must decode an lzma-compressed Blob and raise on a Blob that carries no
    # recognised compression field.
    import lzma
    from pyrosm.engine.blobs import _read_block

    payload = b"raw primitive block bytes"

    blob = Blob()
    blob.lzma_data = lzma.compress(payload)
    encoded = blob.SerializeToString()
    lzma_fp = tmp_path / "lzma_blob.bin"
    lzma_fp.write_bytes(encoded)
    with open(lzma_fp, "rb") as f:
        assert _read_block(f, 0, len(encoded)) == payload

    unsupported = Blob()
    unsupported.raw_size = len(payload)  # a field is set, but no payload field is
    bad = unsupported.SerializeToString()
    bad_fp = tmp_path / "bad_blob.bin"
    bad_fp.write_bytes(bad)
    with open(bad_fp, "rb") as f:
        with pytest.raises(ValueError, match="Unsupported"):
            _read_block(f, 0, len(bad))
