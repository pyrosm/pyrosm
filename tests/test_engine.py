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


def _write_shard(path, **overrides):
    """Write a per-worker shard in the layout decode._decode_batch produces; every array
    is empty by default so a test can fill in only the ones it needs."""
    arrays = dict(
        node_id=np.empty(0, np.int64),
        node_lon=np.empty(0),
        node_lat=np.empty(0),
        way_id=np.empty(0, np.int64),
        refs=np.empty(0, np.int64),
        refs_off=np.array([0], np.int64),
        way_tags=np.empty(0, object),
        way_version=np.empty(0, np.int64),
        way_timestamp=np.empty(0, np.int64),
        way_visible=np.empty(0, np.int64),
        all_id=np.empty(0, np.int64),
        all_refs=np.empty(0, np.int64),
        all_refs_off=np.array([0], np.int64),
        rel_id=np.empty(0, np.int64),
        rel_memid=np.empty(0, np.int64),
        rel_memoff=np.array([0], np.int64),
        rel_memtype=np.empty(0, np.int64),
        rel_memrole=np.empty(0, object),
        rel_tags=np.empty(0, object),
        rel_meta=np.empty((0, 3), np.int64),
    )
    arrays.update(overrides)
    np.savez(path, **arrays)


def _write_pbf_no_buildings(path):
    """Write a minimal one-blob PBF whose only way is untagged, so the engine reads it
    with ways present but no buildings found."""
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    pb = PrimitiveBlock()
    pb.stringtable.s.append(b"")
    way = pb.primitivegroup.add().ways.add()
    way.id = 100
    way.refs.extend([1, 1])
    blob = Blob()
    blob.raw = pb.SerializeToString()
    blob_bytes = blob.SerializeToString()
    bh = BlobHeader()
    bh.type = "OSMData"
    bh.datasize = len(blob_bytes)
    bh_bytes = bh.SerializeToString()
    with open(path, "wb") as f:
        f.write(pack("!L", len(bh_bytes)))
        f.write(bh_bytes)
        f.write(blob_bytes)


def test_engine_get_buildings_no_buildings_returns_none(tmp_path):
    # A PBF with ways but no buildings yields None from both the in-memory and output paths.
    fp = str(tmp_path / "no_buildings.osm.pbf")
    _write_pbf_no_buildings(fp)
    assert get_buildings(fp) is None
    pytest.importorskip("pyarrow")
    assert get_buildings(fp, output=str(tmp_path / "x.parquet")) is None


def test_engine_building_ways_early_returns():
    from pyrosm.engine.decode import _building_ways

    ways = {
        "id": np.array([7], np.int64),
        "keys": np.array([0], np.int64),
        "vals": np.array([0], np.int64),
        "tags_off": np.array([0, 1], np.int64),
        "refs": np.array([1, 2], np.int64),
        "refs_off": np.array([0, 2], np.int64),
        "version": np.array([1], np.int64),
        "timestamp": np.array([1], np.int64),
        "visible": np.array([1], np.int64),
    }
    assert _building_ways([b"highway"], None) is None  # ways is None guard
    assert _building_ways([b"highway"], ways) is None  # 'building' not in string table
    # 'building' is in the table but no way carries it as a key.
    assert _building_ways([b"highway", b"building"], ways) is None


def test_engine_collect_empty_and_edge_shards(tmp_path):
    from pyrosm.engine.collect import (
        _collect_building_ways,
        _collect_kept_ways,
        _node_lookup,
        _collect_relation_ways,
        _needed_node_ids,
        _collect_buildings,
    )

    empty = str(tmp_path / "empty.npz")
    _write_shard(empty)
    assert _collect_building_ways([empty]) == []
    assert _collect_kept_ways([empty], np.empty(0, np.int64)) is None
    assert _node_lookup([empty], np.array([1, 2], np.int64)) is not None
    assert _collect_buildings([empty]) is None
    assert len(_needed_node_ids(None, None)) == 0

    bld = str(tmp_path / "bld.npz")
    _write_shard(
        bld,
        way_id=np.array([1], np.int64),
        refs=np.array([10, 11], np.int64),
        refs_off=np.array([0, 2], np.int64),
        way_tags=np.array([{"building": "yes"}], dtype=object),
        way_version=np.array([1], np.int64),
        way_timestamp=np.array([100], np.int64),
        way_visible=np.array([1], np.int64),
    )
    # Excluding the only building way (it is a relation member) leaves nothing standalone.
    assert _collect_kept_ways([bld], np.array([1], np.int64)) is None

    ways_only = str(tmp_path / "ways.npz")
    _write_shard(
        ways_only,
        all_id=np.array([5], np.int64),
        all_refs=np.array([10, 11], np.int64),
        all_refs_off=np.array([0, 2], np.int64),
    )
    # Member ids that match no stored way -> no relation ways.
    assert _collect_relation_ways([ways_only], np.array([999], np.int64)) is None


def test_engine_geoparquet_schema_helpers():
    pa = pytest.importorskip("pyarrow")
    from pyrosm.engine.geoparquet import _unify_schemas, _align_table

    s1 = pa.schema([pa.field("a", pa.int64())])
    s2 = pa.schema([pa.field("b", pa.string())])
    unified = _unify_schemas([s1, s2])  # neither carries GeoParquet 'geo' metadata
    assert set(unified.names) == {"a", "b"}
    # A table missing column 'b' is widened with typed nulls rather than dropping columns.
    aligned = _align_table(pa.table({"a": [1, 2]}), unified)
    assert aligned.schema.names == ["a", "b"]
    assert aligned.column("b").null_count == 2


def test_engine_stream_parquet_chunk_table_branches(
    helsinki_pbf, tmp_path, monkeypatch
):
    # Drive the chunked writer through its relations-only, ways-only and empty-chunk
    # branches by feeding it crafted variants of the real collected building data.
    pytest.importorskip("pyarrow")
    import tempfile
    import shutil
    from pyrosm.engine import geoparquet
    from pyrosm.engine.blobs import _index_blobs
    from pyrosm.engine.pool import _decode_all
    from pyrosm.engine.collect import _collect_buildings

    data_blobs = [(o, s) for (t, o, s) in _index_blobs(helsinki_pbf) if t == "OSMData"]
    shard_dir = tempfile.mkdtemp()
    try:
        shards = _decode_all(helsinki_pbf, data_blobs, 1, shard_dir)
        kept, relations, relation_ways, nc = _collect_buildings(shards)
        assert kept is not None and relations is not None  # helsinki has both

        def write(collected):
            monkeypatch.setattr(
                geoparquet, "_collect_buildings", lambda paths: collected
            )
            return geoparquet._stream_buildings_to_parquet(
                shards, str(tmp_path / "o.parquet"), 250_000, True
            )

        # relations only (no standalone ways): way loop skipped, relation chunk written.
        assert write((None, relations, relation_ways, nc)) is not None
        # ways only (no relations): relation branch skipped.
        assert write((kept, None, None, nc)) is not None
        # a way referencing an absent node assembles empty -> no parts -> None.
        absent = [
            {
                "id": -1,
                "version": 1,
                "timestamp": 1,
                "visible": True,
                "nodes": [10**18],
                "tags": {"building": "yes"},
            }
        ]
        assert write((absent, None, None, nc)) is None
        # a relation whose member ways are all absent assembles empty -> chunk skipped.
        empty_ways = {"id": np.empty(0, np.int64), "nodes": np.empty(0, dtype=object)}
        assert write((None, relations, empty_ways, nc)) is None
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


def test_compat_has_pyarrow_false_when_unavailable(monkeypatch):
    # The optional-dependency flag falls back to False (and require_pyarrow then raises)
    # when pyarrow cannot be imported.
    import builtins
    import importlib
    from pyrosm.utils import _compat

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        importlib.reload(_compat)
        assert _compat.HAS_PYARROW is False
        with pytest.raises(ImportError, match="pyarrow"):
            _compat.require_pyarrow()
    finally:
        monkeypatch.undo()
        importlib.reload(_compat)  # restore the real, pyarrow-present state
    assert _compat.HAS_PYARROW is True
