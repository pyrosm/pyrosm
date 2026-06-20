"""Out-of-core engine buildings reader: parity vs the in-memory OSM(fp).get_buildings()
way and relation rows, plus the output= GeoParquet path and the worker-count policy."""

import glob
import os
import zlib
from struct import pack, unpack

import geopandas as gpd
import numpy as np
import pytest
import shapely

from pyrosm import OSM, get_data
from pyrosm.engine import (
    get_buildings,
    get_landuse,
    get_natural,
    get_pois,
    get_boundaries,
    get_data_by_custom_criteria,
    get_network,
)
from pyrosm.engine import pool, geoparquet, cache
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob

# Captured before the autouse fixture below stubs the module attribute, so a test can exercise
# the real cache_dir() implementation.
_real_cache_dir = cache.cache_dir


@pytest.fixture(autouse=True, scope="session")
def _shared_cache_dir(tmp_path_factory):
    # Point the per-layer result cache at one temp dir for the whole session: parity tests that
    # read the same layer reuse the cached parquet, nothing leaks into the user's real cache dir.
    # Tests that must observe a (re)build use the function-scoped ``fresh_cache`` fixture.
    shared = tmp_path_factory.mktemp("result_cache")
    original = cache.cache_dir
    cache.cache_dir = lambda: str(shared)
    yield
    cache.cache_dir = original


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    # A private, empty cache dir for tests that need the layer to actually be (re)built.
    cache_dir = tmp_path / "fresh_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "cache_dir", lambda: str(cache_dir))
    return cache_dir


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


@pytest.fixture
def helsinki_history_pbf():
    return get_data("helsinki_test_history_pbf")


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


def _fake_executor(raise_init=None, raise_map=None):
    """A ProcessPoolExecutor stand-in that runs map() in-process, optionally raising at
    construction or at map() to exercise the parallel branch and its fallback without real
    worker processes."""

    class _F:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            if raise_init is not None:
                raise raise_init("simulated")
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, tasks):
            if raise_map is not None:
                raise raise_map("simulated")
            return [fn(t) for t in tasks]

    return _F


def test_engine_parallel_decode_runs_and_matches_serial(
    helsinki_pbf, monkeypatch, fresh_cache
):
    # The parallel branch (map over per-worker tasks, then flatten the shard paths) yields
    # the same result as the single-process path.
    monkeypatch.setattr(pool, "ProcessPoolExecutor", _fake_executor())
    mine = get_buildings(helsinki_pbf, workers=3)
    _assert_full_parity(mine, OSM(helsinki_pbf).get_buildings())


@pytest.mark.parametrize("where", ["construction", "run"])
def test_engine_parallel_decode_falls_back_to_serial(
    where, helsinki_pbf, monkeypatch, fresh_cache
):
    # A process pool that cannot start (OSError, e.g. a restricted environment) or whose
    # workers die (BrokenProcessPool, e.g. an unguarded __main__) must fall back to a single
    # process with a warning, not hang or error, and still produce the correct result.
    from concurrent.futures.process import BrokenProcessPool

    if where == "construction":
        fake = _fake_executor(raise_init=OSError)
    else:
        fake = _fake_executor(raise_map=BrokenProcessPool)
    monkeypatch.setattr(pool, "ProcessPoolExecutor", fake)
    with pytest.warns(RuntimeWarning, match="single process"):
        mine = get_buildings(helsinki_pbf, workers=3)
    _assert_full_parity(mine, OSM(helsinki_pbf).get_buildings())


def test_engine_unguarded_module_read_does_not_hang(test_pbf, tmp_path):
    # A read at module level (no `if __name__ == "__main__":` guard) must not hang: the
    # spawned workers cannot re-import the entry point, so the decode falls back to a single
    # process. Run it as a subprocess with a timeout so a regression surfaces as a failure.
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent("""
            import warnings
            from pyrosm import engine
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = engine.get_buildings(%r, workers=3)
            print("ROWS", 0 if gdf is None else len(gdf))
            """ % test_pbf))
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "ROWS" in r.stdout, r.stderr[-800:]


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
    """Write a shard in the layout decode._decode_one_block produces; every array is empty
    by default so a test can fill in only the ones it needs."""
    arrays = dict(
        node_id=np.empty(0, np.int64),
        node_lon=np.empty(0),
        node_lat=np.empty(0),
        in_box_id=np.empty(0, np.int64),
        nfeat_id=np.empty(0, np.int64),
        nfeat_lon=np.empty(0),
        nfeat_lat=np.empty(0),
        nfeat_tags=np.empty(0, object),
        nfeat_meta=np.empty((0, 4), np.int64),
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


def test_engine_matching_ways_early_returns():
    from pyrosm.engine.decode import _matching_ways

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
    assert _matching_ways([b"highway"], None, [b"building"]) is None  # ways is None
    assert _matching_ways([b"highway"], ways, [b"building"]) is None  # key not in table
    # the key is in the table but no way carries it.
    assert _matching_ways([b"highway", b"building"], ways, [b"building"]) is None


def test_engine_collect_empty_and_edge_shards(tmp_path):
    from pyrosm.engine.collect import (
        _collect_matching_ways,
        _collect_kept_ways,
        _collect_node_features,
        _node_lookup,
        _collect_relation_ways,
        _needed_node_ids,
        _collect_layer,
    )
    from pyrosm.data_manager import parse_custom_filter

    def keep_all(tag):
        return True

    data_filter, osm_keys = parse_custom_filter({"building": [True]})
    filter_spec = (osm_keys, data_filter, "keep")

    empty = str(tmp_path / "empty.npz")
    _write_shard(empty)
    assert _collect_matching_ways([empty]) == []
    assert _collect_kept_ways([empty], np.empty(0, np.int64), keep_all) is None
    assert _collect_node_features([empty], [], True, keep_all) is None
    assert _node_lookup([empty], np.array([1, 2], np.int64)) is not None
    # Node-only result: no coordinates needed -> empty NodeLocations, not a crash.
    assert _node_lookup([empty], np.empty(0, np.int64)) is not None
    assert _collect_layer([empty], [], True, filter_spec, True, True) is None
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
    # Excluding the only way (it is a relation member) leaves nothing standalone.
    assert _collect_kept_ways([bld], np.array([1], np.int64), keep_all) is None

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
    from pyrosm.config import Conf
    from pyrosm.data_manager import parse_custom_filter
    from pyrosm.engine import geoparquet
    from pyrosm.engine.blobs import _index_blobs
    from pyrosm.engine.pool import _decode_all
    from pyrosm.engine.collect import _collect_layer

    data_filter, osm_keys = parse_custom_filter({"building": [True]})
    filter_spec = (osm_keys, data_filter, "keep")
    tac = Conf.tags.building
    data_blobs = [(o, s) for (t, o, s) in _index_blobs(helsinki_pbf) if t == "OSMData"]
    shard_dir = tempfile.mkdtemp()
    try:
        shards = _decode_all(
            helsinki_pbf, data_blobs, 1, shard_dir, [b"building"], False
        )
        node_features, kept, relations, relation_ways, nc = _collect_layer(
            shards, tac, True, filter_spec, True, True
        )
        assert kept is not None and relations is not None  # helsinki has both

        def write(collected):
            monkeypatch.setattr(geoparquet, "_collect_layer", lambda *a, **k: collected)
            return geoparquet._stream_layer_to_parquet(
                shards,
                str(tmp_path / "o.parquet"),
                250_000,
                tac,
                True,
                filter_spec,
                True,
                True,
            )

        # relations only (no standalone ways): way loop skipped, relation chunk written.
        assert write((None, None, relations, relation_ways, nc)) is not None
        # ways only (no relations): relation branch skipped.
        assert write((None, kept, None, None, nc)) is not None
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
        assert write((None, absent, None, None, nc)) is None
        # a relation whose member ways are all absent assembles empty -> chunk skipped.
        empty_ways = {"id": np.empty(0, np.int64), "nodes": np.empty(0, dtype=object)}
        assert write((None, None, relations, empty_ways, nc)) is None
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


@pytest.fixture
def helsinki_region_pbf():
    return get_data("helsinki_region_pbf")


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_landuse_full_column_parity(fixture, request):
    # A different layer (different filter key and tag columns) must match
    # OSM().get_landuse() column-for-column and value-for-value.
    fp = request.getfixturevalue(fixture)
    mine = get_landuse(fp)
    ref = OSM(fp).get_landuse()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_natural_full_column_parity(fixture, request):
    # natural includes NODE point features, so this exercises the node path too.
    fp = request.getfixturevalue(fixture)
    mine = get_natural(fp)
    ref = OSM(fp).get_natural()
    if ref is None:
        assert mine is None
        return
    n_node = int((ref["osm_type"] == "node").sum())
    assert int((mine["osm_type"] == "node").sum()) == n_node
    _assert_full_parity(mine, ref)


def test_engine_natural_has_node_rows(helsinki_pbf):
    # The bundled Helsinki extract has natural node features; assert they are produced.
    mine = get_natural(helsinki_pbf)
    assert int((mine["osm_type"] == "node").sum()) > 0


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_pois_default_parity(fixture, request):
    # The default POI filter over nodes + ways + relations.
    fp = request.getfixturevalue(fixture)
    mine = get_pois(fp)
    ref = OSM(fp).get_pois()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


def test_engine_pois_value_filter_parity(helsinki_pbf):
    # A value-level custom_filter must refine to the exact values (not just key presence).
    flt = {"amenity": ["restaurant", "cafe", "bar"]}
    mine = get_pois(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf).get_pois(custom_filter=flt)
    assert mine is not None and len(mine) > 0
    assert mine["amenity"].isin(flt["amenity"]).all()
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_value_filter_parity(helsinki_pbf):
    flt = {"amenity": ["restaurant", "cafe", "pub"]}
    mine = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=flt)
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize(
    "flags",
    [
        {"keep_nodes": False},
        {"keep_ways": False},
        {"keep_relations": False},
        {"keep_nodes": False, "keep_relations": False},
    ],
)
def test_engine_custom_criteria_keep_flags_parity(helsinki_pbf, flags):
    # keep_nodes / keep_ways / keep_relations must select element kinds exactly as the
    # in-memory reader.
    flt = {"amenity": True}
    mine = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt, **flags)
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=flt, **flags)
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


def _count_decodes(monkeypatch):
    # Count how many times the engine decodes the PBF, to assert a cached read does not re-decode.
    import pyrosm.engine.readers as readers_mod

    calls = []
    real = readers_mod._decode_and_run

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(readers_mod, "_decode_and_run", spy)
    return calls


def test_engine_cache_reuse_skips_decode(helsinki_pbf, monkeypatch, fresh_cache):
    # The first read assembles + caches the layer; an identical second read reads the cached
    # GeoParquet back without decoding the PBF again, and returns the same data.
    decodes = _count_decodes(monkeypatch)
    first = get_buildings(helsinki_pbf)
    second = get_buildings(helsinki_pbf)
    assert sum(decodes) == 1
    _assert_full_parity(
        second, first.sort_values(["osm_type", "id"]).reset_index(drop=True)
    )


def test_engine_cache_distinct_params_distinct_file(helsinki_pbf):
    # A different read parameter keys a different cache file.
    base = cache.result_path(helsinki_pbf, {"keep_metadata": True})
    assert base != cache.result_path(helsinki_pbf, {"keep_metadata": False})
    assert base != cache.result_path(helsinki_pbf, {"keep_metadata": True, "x": 1})


def test_engine_cache_empty_result_is_marked(test_pbf, monkeypatch, fresh_cache):
    # A filter that matches nothing returns None and records an empty marker, so an identical
    # later read returns None without decoding the file again.
    flt = {"amenity": ["definitely_not_a_real_value_xyz"]}
    assert get_data_by_custom_criteria(test_pbf, custom_filter=flt) is None
    assert glob.glob(os.path.join(cache.cache_dir(), "*.empty"))
    decodes = _count_decodes(monkeypatch)
    assert get_data_by_custom_criteria(test_pbf, custom_filter=flt) is None
    assert sum(decodes) == 0


def test_engine_cache_pyarrow_absent_falls_back(helsinki_pbf, monkeypatch, fresh_cache):
    # With pyarrow unavailable the engine returns the in-memory frame and writes no cache file.
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    mine = get_buildings(helsinki_pbf)
    _assert_full_parity(mine, OSM(helsinki_pbf).get_buildings())
    assert glob.glob(os.path.join(cache.cache_dir(), "*.parquet")) == []


def test_cache_dir_builds_and_creates_tempdir_path(monkeypatch, tmp_path):
    # cache_dir() roots the result cache at <tempdir>/pyrosm/cache and creates it on demand. The
    # fixtures stub the module attribute, so exercise the real implementation captured at import.
    monkeypatch.setattr(cache.tempfile, "gettempdir", lambda: str(tmp_path))
    result = _real_cache_dir()
    assert result == os.path.join(str(tmp_path), "pyrosm", "cache")
    assert os.path.isdir(result)


def test_engine_boundaries_parity(helsinki_region_pbf):
    # The Helsinki region extract has administrative boundaries (relations + ways).
    mine = get_boundaries(helsinki_region_pbf)
    ref = OSM(helsinki_region_pbf).get_boundaries()
    assert ref is not None and (ref["osm_type"] == "relation").any()
    _assert_full_parity(mine, ref)


def test_engine_natural_output_parquet_matches(helsinki_pbf, tmp_path):
    # A point layer (nodes + ways) streamed to GeoParquet must reload equal to the frame.
    pytest.importorskip("pyarrow")
    out = str(tmp_path / "natural.parquet")
    in_memory = get_natural(helsinki_pbf)
    returned = get_natural(helsinki_pbf, output=out)
    assert returned == out
    reloaded = gpd.read_parquet(out)
    _assert_matches(reloaded, in_memory)


def test_engine_boundaries_name_filter_parity(helsinki_region_pbf):
    # The name= substring post-filter must match OSM().get_boundaries(name=...).
    mine = get_boundaries(helsinki_region_pbf, name="Vantaa")
    ref = OSM(helsinki_region_pbf).get_boundaries(name="Vantaa")
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_osm_keys_and_explicit_columns(helsinki_pbf):
    # osm_keys_to_keep as a string (-> list) and an explicit tags_as_columns.
    kwargs = dict(
        custom_filter={"building": True},
        osm_keys_to_keep="building",
        tags_as_columns=["building"],
    )
    mine = get_data_by_custom_criteria(helsinki_pbf, **kwargs)
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(**kwargs)
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_non_conf_key_parity(helsinki_pbf):
    # A filter key without a dedicated Conf.tags column set becomes its own column.
    flt = {"source": True}
    mine = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=flt)
    _assert_full_parity(mine, ref)


def test_engine_node_feature_edge_cases(tmp_path):
    from pyrosm.engine.decode import _matching_nodes
    from pyrosm.engine.collect import _collect_node_features

    # _matching_nodes returns None when there are no dense nodes in the block.
    assert (
        _matching_nodes([b"natural"], None, [b"natural"], np.empty(0), np.empty(0))
        is None
    )
    # A node feature that fails the value filter is dropped (no node passes).
    nfeat = str(tmp_path / "nfeat.npz")
    _write_shard(
        nfeat,
        nfeat_id=np.array([1], np.int64),
        nfeat_lon=np.array([24.9]),
        nfeat_lat=np.array([60.1]),
        nfeat_tags=np.array([{"natural": "tree"}], dtype=object),
        nfeat_meta=np.zeros((1, 4), np.int64),
    )
    assert _collect_node_features([nfeat], ["natural"], True, lambda t: False) is None


def test_engine_natural_keep_metadata_false_parity(helsinki_pbf):
    # keep_metadata=False over a point layer (nodes + ways) must match the in-memory reader.
    mine = get_natural(helsinki_pbf, keep_metadata=False)
    ref = OSM(helsinki_pbf, keep_metadata=False).get_natural()
    _assert_full_parity(mine, ref)


def test_engine_boundaries_custom_filter_adds_boundary_key(helsinki_region_pbf):
    # A custom_filter without 'boundary' gets it added, matching OSM().get_boundaries().
    flt = {"admin_level": True}
    mine = get_boundaries(helsinki_region_pbf, custom_filter=dict(flt))
    ref = OSM(helsinki_region_pbf).get_boundaries(custom_filter=dict(flt))
    if ref is None or len(ref) == 0:
        assert mine is None or len(mine) == 0
        return
    _assert_full_parity(mine, ref)


def test_engine_boundaries_name_without_name_column_raises(helsinki_pbf, monkeypatch):
    # name= on a result that has no 'name' column raises, as the in-memory reader does.
    from shapely.geometry import Point
    from pyrosm.engine import readers

    fake = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
    monkeypatch.setattr(readers, "_get_layer", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="Could not filter by name"):
        readers.get_boundaries(helsinki_pbf, name="x")


def test_engine_boundaries_name_with_output_raises(helsinki_pbf, tmp_path):
    # name= filtering cannot be combined with output= (the streamed GeoParquet would be
    # written unfiltered); it must fail fast instead of silently writing all boundaries.
    with pytest.raises(ValueError, match="cannot be combined with output"):
        get_boundaries(helsinki_pbf, name="x", output=str(tmp_path / "b.parquet"))


def test_engine_stream_skips_empty_node_chunk(tmp_path, monkeypatch):
    # A node chunk that assembles to nothing is skipped (no row group); with only that
    # chunk present, the writer produces no parts and returns None.
    pytest.importorskip("pyarrow")
    from pyrosm.config import Conf
    from pyrosm.data_manager import parse_custom_filter
    from pyrosm.engine import geoparquet

    data_filter, osm_keys = parse_custom_filter({"natural": [True]})
    filter_spec = (osm_keys, data_filter, "keep")
    node_features = {
        "id": np.array([1], np.int64)
    }  # presence is enough; assemble stubbed
    monkeypatch.setattr(geoparquet, "_assemble_chunk", lambda *a, **k: None)
    monkeypatch.setattr(
        geoparquet,
        "_collect_layer",
        lambda *a, **k: (node_features, None, None, None, None),
    )
    out = str(tmp_path / "x.parquet")
    assert (
        geoparquet._stream_layer_to_parquet(
            [], out, 250_000, Conf.tags.natural, True, filter_spec, True, True
        )
        is None
    )


@pytest.mark.parametrize(
    "network_type", ["walking", "driving", "driving+service", "cycling", "all"]
)
def test_engine_network_parity(network_type, helsinki_pbf):
    # Street network: highway ways as LineString edges + a 'length' column. 'all' keeps
    # every highway (a None data_filter); the others apply a predefined exclude filter.
    mine = get_network(helsinki_pbf, network_type=network_type)
    ref = OSM(helsinki_pbf).get_network(network_type=network_type)
    assert ref is not None and len(ref) > 0
    _assert_full_parity(mine, ref)


def test_engine_network_custom_filter_parity(helsinki_pbf):
    # The 'crossing' key (not among the default highway columns) is added to the column
    # allow-list; no kept way carries it here, so -- matching the in-memory reader -- it
    # yields no column.
    flt = {"highway": ["footway", "path", "pedestrian"], "crossing": True}
    mine = get_network(helsinki_pbf, custom_filter=flt, filter_type="keep")
    ref = OSM(helsinki_pbf).get_network(custom_filter=flt, filter_type="keep")
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("network_type", ["walking", "driving", "all"])
def test_engine_network_nodes_parity(network_type, helsinki_pbf):
    # nodes=True returns (nodes, edges): the ways are sliced into per-segment edges and the
    # graph-export node frame (node coordinates + tags + metadata) is built, both matching
    # the in-memory reader.
    mine_nodes, mine_edges = get_network(
        helsinki_pbf, network_type=network_type, nodes=True
    )
    ref_nodes, ref_edges = OSM(helsinki_pbf).get_network(
        network_type=network_type, nodes=True
    )
    assert ref_edges is not None and len(ref_edges) > 0
    _assert_full_parity(mine_edges, ref_edges)
    # The node frame has no osm_type/relation rows; key it on id alone.
    a = mine_nodes.sort_values("id").reset_index(drop=True)
    b = ref_nodes.sort_values("id").reset_index(drop=True)
    _assert_full_parity(a.assign(osm_type="node"), b.assign(osm_type="node"))


def test_engine_network_nodes_keep_metadata_false_parity(helsinki_pbf):
    mine_nodes, mine_edges = get_network(
        helsinki_pbf, network_type="driving", nodes=True, keep_metadata=False
    )
    ref_nodes, ref_edges = OSM(helsinki_pbf, keep_metadata=False).get_network(
        network_type="driving", nodes=True
    )
    assert "changeset" not in mine_nodes.columns and "visible" in mine_nodes.columns
    _assert_full_parity(mine_edges, ref_edges)
    a = mine_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    b = ref_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    _assert_full_parity(a, b)


def test_engine_network_cache_reuse_skips_decode(
    helsinki_pbf, monkeypatch, fresh_cache
):
    # The first network read assembles + caches the edges; an identical second read reads the
    # cached GeoParquet back without decoding the PBF again, returning the same data.
    decodes = _count_decodes(monkeypatch)
    first = get_network(helsinki_pbf, network_type="driving")
    second = get_network(helsinki_pbf, network_type="driving")
    assert sum(decodes) == 1
    _assert_full_parity(first, second)


def test_engine_network_nodes_cache_reuse_skips_decode(
    helsinki_pbf, monkeypatch, fresh_cache
):
    # nodes=True caches the (nodes, edges) tuple as two files; an identical second read returns
    # the tuple from those files without decoding the PBF again.
    decodes = _count_decodes(monkeypatch)
    n1, e1 = get_network(helsinki_pbf, network_type="driving", nodes=True)
    n2, e2 = get_network(helsinki_pbf, network_type="driving", nodes=True)
    assert sum(decodes) == 1
    assert len(glob.glob(os.path.join(cache.cache_dir(), "*.parquet"))) == 2
    _assert_full_parity(e1, e2)
    _assert_full_parity(n1.assign(osm_type="node"), n2.assign(osm_type="node"))


def test_engine_network_output_path_writes_edges(helsinki_pbf, tmp_path):
    # output= writes the edges to that GeoParquet and returns the path; the file matches the
    # in-memory network edges.
    out = str(tmp_path / "network.parquet")
    ret = get_network(helsinki_pbf, network_type="driving", output=out)
    assert ret == out and os.path.exists(out)
    written = cache.read_result(out)
    ref = OSM(helsinki_pbf).get_network(network_type="driving")
    _assert_full_parity(written, ref)


def test_engine_network_output_dir_with_nodes(helsinki_pbf, tmp_path):
    # output= + nodes=True writes edges.parquet + nodes.parquet into the directory and returns
    # it; the two files reload equal to the in-memory (nodes, edges) tuple.
    from pyrosm.engine import readers

    out = str(tmp_path / "net_dir")
    ret = get_network(helsinki_pbf, network_type="driving", nodes=True, output=out)
    assert ret == out
    assert os.path.exists(os.path.join(out, "edges.parquet"))
    assert os.path.exists(os.path.join(out, "nodes.parquet"))
    ref_nodes, ref_edges = OSM(helsinki_pbf).get_network(
        network_type="driving", nodes=True
    )
    _assert_full_parity(
        cache.read_result(os.path.join(out, "edges.parquet")), ref_edges
    )
    written_nodes = readers._read_nodes_parquet(os.path.join(out, "nodes.parquet"))
    a = written_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    b = ref_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    _assert_full_parity(a, b)


def test_engine_network_output_dir_empty_returns_none(test_pbf, tmp_path):
    # An empty nodes=True read with output= writes nothing and returns None.
    out = str(tmp_path / "empty_dir")
    flt = {"highway": ["definitely_not_a_real_highway_xyz"]}
    ret = get_network(
        test_pbf, custom_filter=flt, filter_type="keep", nodes=True, output=out
    )
    assert ret is None
    assert not os.path.exists(out)


def test_engine_network_pyarrow_absent_falls_back(
    helsinki_pbf, monkeypatch, fresh_cache
):
    # With pyarrow unavailable the network read returns the in-memory edges and writes no cache.
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    mine = get_network(helsinki_pbf, network_type="driving")
    _assert_full_parity(mine, OSM(helsinki_pbf).get_network(network_type="driving"))
    assert glob.glob(os.path.join(cache.cache_dir(), "*.parquet")) == []


def test_engine_network_empty_result_is_marked(test_pbf, monkeypatch, fresh_cache):
    # A network filter that matches no highway returns None, records an empty marker, and an
    # identical later read returns None without decoding the file again.
    flt = {"highway": ["definitely_not_a_real_highway_xyz"]}
    assert get_network(test_pbf, custom_filter=flt, filter_type="keep") is None
    assert glob.glob(os.path.join(cache.cache_dir(), "*.empty"))
    decodes = _count_decodes(monkeypatch)
    assert get_network(test_pbf, custom_filter=flt, filter_type="keep") is None
    assert sum(decodes) == 0


def test_engine_network_nodes_empty_result_is_marked(
    test_pbf, monkeypatch, fresh_cache
):
    # nodes=True with a no-match filter returns (None, None), records an empty marker, and an
    # identical later read returns (None, None) without decoding again.
    flt = {"highway": ["definitely_not_a_real_highway_xyz"]}
    nodes, edges = get_network(
        test_pbf, custom_filter=flt, filter_type="keep", nodes=True
    )
    assert nodes is None and edges is None
    assert glob.glob(os.path.join(cache.cache_dir(), "*.empty"))
    decodes = _count_decodes(monkeypatch)
    again = get_network(test_pbf, custom_filter=flt, filter_type="keep", nodes=True)
    assert again == (None, None)
    assert sum(decodes) == 0


def test_engine_node_records_by_id_edge_cases():
    from pyrosm.engine.decode import _node_records_by_id

    header = {"granularity": 100, "lat_offset": 0, "lon_offset": 0}
    st = [b"", b"amenity", b"cafe"]
    # A block with no dense nodes (a way/relation block) yields nothing.
    assert _node_records_by_id(st, header, None, {1}, True) is None
    nodes = {
        "id": np.array([1, 2], np.int64),
        "lat": np.array([600000000, 600000010], np.int64),
        "lon": np.array([240000000, 240000010], np.int64),
        # node 1 untagged; node 2 tagged amenity=cafe (key/value string-table indices 1,2).
        "keys_vals": np.array([0, 1, 2, 0], np.int64),
        "version": np.array([3, 3], np.int64),
        "timestamp": np.array([7, 7], np.int64),
        "changeset": np.array([9, 9], np.int64),
        "visible": np.array([1, 1], np.int64),
    }
    # No requested id present in the block -> nothing gathered.
    assert _node_records_by_id(st, header, nodes, {999}, True) is None
    rec = _node_records_by_id(st, header, nodes, {1, 2}, True)
    assert rec["tags"] == [None, {"amenity": "cafe"}]
    assert rec["version"].tolist() == [3, 3] and rec["visible"].dtype == bool
    # keep_metadata=False drops version/timestamp/changeset but keeps visible.
    lean = _node_records_by_id(st, header, nodes, {1}, False)
    assert "version" not in lean and "visible" in lean


def test_engine_gather_node_records_empty(helsinki_pbf):
    from pyrosm.engine.collect import _gather_node_records

    # No node ids requested: every block yields no record, so the rich coordinate store is
    # built from the empty-array fallback.
    nc = _gather_node_records(helsinki_pbf, np.empty(0, np.int64), True)
    assert nc is not None


def test_engine_network_invalid_args(helsinki_pbf):
    with pytest.raises(ValueError):
        get_network(helsinki_pbf, network_type="not-a-network")
    with pytest.raises(ValueError):
        get_network(helsinki_pbf, network_type=123)
    with pytest.raises(ValueError):
        get_network(helsinki_pbf, custom_filter={"highway": True}, filter_type="bogus")


def _central_bbox(gdf, frac=0.4):
    # The central `frac` of a layer's extent -- a box that still contains way and relation
    # features, to exercise the bounding_box read path.
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = (1 - frac) / 2
    return [
        minx + (maxx - minx) * pad,
        miny + (maxy - miny) * pad,
        minx + (maxx - minx) * (1 - pad),
        miny + (maxy - miny) * (1 - pad),
    ]


@pytest.mark.parametrize("complete_relations", [False, True])
def test_engine_buildings_bounding_box_parity(helsinki_pbf, complete_relations):
    # A bounding box restricts buildings (ways + relations) to that area; relations are
    # partial by default and complete with complete_relations=True -- matching the in-memory
    # reader either way.
    bbox = _central_bbox(OSM(helsinki_pbf).get_buildings())
    mine = get_buildings(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    )
    ref = OSM(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    ).get_buildings()
    assert ref is not None and (ref["osm_type"] == "relation").any()
    _assert_full_parity(mine, ref)


def test_engine_pois_bounding_box_parity(helsinki_pbf):
    # bbox over a layer with node features: nodes + ways restricted to the box.
    bbox = _central_bbox(OSM(helsinki_pbf).get_pois())
    mine = get_pois(helsinki_pbf, bounding_box=bbox)
    ref = OSM(helsinki_pbf, bounding_box=bbox).get_pois()
    assert ref is not None and (ref["osm_type"] == "node").any()
    _assert_full_parity(mine, ref)


def test_engine_network_bounding_box_parity(helsinki_pbf):
    # A bounding box keeps ways with >=1 node inside it (kept whole), then the final spatial
    # filter clips to the box -- matching the in-memory reader.
    full = OSM(helsinki_pbf).get_network(network_type="driving")
    bbox = _central_bbox(full, frac=0.5)
    ref = OSM(helsinki_pbf, bounding_box=bbox).get_network(network_type="driving")
    mine = get_network(helsinki_pbf, network_type="driving", bounding_box=bbox)
    assert ref is not None and 0 < len(ref) < len(full)
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_bounding_box_parity(helsinki_pbf):
    flt = {"building": True}
    bbox = _central_bbox(OSM(helsinki_pbf).get_buildings())
    mine = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, filter_type="keep", bounding_box=bbox
    )
    ref = OSM(helsinki_pbf, bounding_box=bbox).get_data_by_custom_criteria(
        custom_filter=flt, filter_type="keep"
    )
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


def test_engine_bounding_box_polygon_parity(helsinki_pbf):
    # A shapely-polygon bounding box is validated/normalised through the same path as the
    # in-memory reader and yields the same result as the equivalent coordinate list.
    from shapely.geometry import box

    poly = box(*_central_bbox(OSM(helsinki_pbf).get_buildings()))
    mine = get_buildings(helsinki_pbf, bounding_box=poly)
    ref = OSM(helsinki_pbf, bounding_box=poly).get_buildings()
    assert ref is not None and len(ref) > 0
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("bad", [[1, 2, 3], [25, 61, 24, 60], "not-a-bbox"])
def test_engine_bounding_box_validation_matches_in_memory(helsinki_pbf, bad):
    # Malformed boxes (wrong length, inverted coordinates, wrong type) raise the same
    # ValueError the in-memory reader raises.
    with pytest.raises(ValueError):
        OSM(helsinki_pbf, bounding_box=bad)
    with pytest.raises(ValueError):
        get_buildings(helsinki_pbf, bounding_box=bad)


def test_engine_bbox_helpers():
    from shapely.geometry import box
    from pyrosm.engine.bounding_box import (
        _bbox_bounds,
        _in_box_mask,
        _filter_features_to_box,
    )

    assert _bbox_bounds(None) is None
    assert _bbox_bounds([0, 1, 2, 3]) == (0, 1, 2, 3)
    assert _bbox_bounds(box(0, 1, 2, 3)) == (0.0, 1.0, 2.0, 3.0)

    mask = _in_box_mask(np.array([0.0, 5.0]), np.array([0.0, 5.0]), (-1, -1, 1, 1))
    assert mask.tolist() == [True, False]

    found = {
        "id": np.array([1, 2], np.int64),
        "lon": np.array([0.0, 5.0]),
        "lat": np.array([0.0, 5.0]),
        "tags": [{"a": "1"}, {"b": "2"}],
    }
    kept = _filter_features_to_box(found, (-1, -1, 1, 1))
    assert kept["id"].tolist() == [1] and kept["tags"] == [{"a": "1"}]
    assert _filter_features_to_box(found, (10, 10, 11, 11)) is None  # nothing inside


def test_engine_in_box_nodes(tmp_path):
    from pyrosm.engine.bounding_box import _in_box_nodes

    empty = str(tmp_path / "e.npz")
    _write_shard(empty)
    assert len(_in_box_nodes([empty])) == 0  # no in-box ids anywhere
    a = str(tmp_path / "a.npz")
    _write_shard(a, in_box_id=np.array([3, 1, 3], np.int64))
    b = str(tmp_path / "b.npz")
    _write_shard(b, in_box_id=np.array([2], np.int64))
    assert _in_box_nodes([a, b]).tolist() == [1, 2, 3]  # unique, sorted


def test_engine_assemble_network_empty_returns_none(tmp_path):
    from pyrosm.config import Conf
    from pyrosm.engine.assemble import _assemble_network

    empty = str(tmp_path / "e.npz")
    _write_shard(empty)
    # A None data_filter keeps every highway; with no ways at all the result is empty.
    edges, nodes = _assemble_network(
        [empty],
        list(Conf.tags.highway),
        True,
        (["highway"], None, "exclude"),
        False,
        None,
    )
    assert edges is None and nodes is None


def test_engine_assemble_network_edges_without_nodes_column(tmp_path):
    from pyrosm.config import Conf
    from pyrosm.engine.assemble import _assemble_network

    shard = str(tmp_path / "s.npz")
    _write_shard(
        shard,
        node_id=np.array([5], np.int64),
        node_lon=np.array([24.9]),
        node_lat=np.array([60.1]),
        way_id=np.array([1], np.int64),
        refs=np.array([5, 5], np.int64),
        refs_off=np.array([0, 2], np.int64),
        way_tags=np.array([{"highway": "footway"}], dtype=object),
        way_version=np.array([1], np.int64),
        way_timestamp=np.array([1], np.int64),
        way_visible=np.array([1], np.int64),
    )
    # A degenerate way (both refs the same node) assembles edges that carry no 'nodes'
    # column, so the default node-info drop is correctly skipped.
    edges, _ = _assemble_network(
        [shard],
        list(Conf.tags.highway),
        True,
        (["highway"], None, "exclude"),
        False,
        None,
    )
    assert edges is not None and "nodes" not in edges.columns


def test_engine_assemble_network_none_edges(tmp_path, monkeypatch):
    from pyrosm import frames
    from pyrosm.config import Conf
    from pyrosm.engine import assemble

    shard = str(tmp_path / "s.npz")
    _write_shard(
        shard,
        node_id=np.array([5, 6], np.int64),
        node_lon=np.array([24.90, 24.91]),
        node_lat=np.array([60.10, 60.11]),
        way_id=np.array([1], np.int64),
        refs=np.array([5, 6], np.int64),
        refs_off=np.array([0, 2], np.int64),
        way_tags=np.array([{"highway": "footway"}], dtype=object),
        way_version=np.array([1], np.int64),
        way_timestamp=np.array([1], np.int64),
        way_visible=np.array([1], np.int64),
    )
    # When the geometry pipeline yields no edges, the node-info drop is skipped.
    monkeypatch.setattr(frames, "prepare_geodataframe", lambda *a, **k: (None, None))
    edges, nodes = assemble._assemble_network(
        [shard],
        list(Conf.tags.highway),
        True,
        (["highway"], None, "exclude"),
        False,
        None,
    )
    assert edges is None and nodes is None


def test_engine_collect_bbox_relation_branches(tmp_path):
    from pyrosm.engine.collect import _collect_relation_ways, _restrict_relations_to_box

    empty = str(tmp_path / "e.npz")
    _write_shard(empty)
    # No member ids at all -> nothing to look up.
    assert _collect_relation_ways([empty], np.empty(0, np.int64)) is None
    # A relation whose only member way is outside the box is dropped entirely.
    relations = {
        "id": np.array([1], np.int64),
        "members": [
            {"member_id": np.array([5], np.int64), "member_type": np.array([b"way"])}
        ],
    }
    kept, ids = _restrict_relations_to_box(relations, set())
    assert kept is None and len(ids) == 0


# ---------------------------------------------------------------------------
# Public OSM API wired to the out-of-core engine (engine="out_of_core").
# ---------------------------------------------------------------------------


def _osm_parity(fp, method, **kwargs):
    """The out-of-core engine, reached through the public OSM API, matches the in-memory
    reader for the given feature method and keyword arguments. A fresh copy of the keyword
    arguments is passed to each reader since the in-memory reader mutates the custom_filter
    dict in place (it ensures the layer key), which would otherwise mask the engine's own
    key-ensuring branch."""
    import copy

    ref = getattr(OSM(fp), method)(**copy.deepcopy(kwargs))
    mine = getattr(OSM(fp, engine="out_of_core"), method)(**copy.deepcopy(kwargs))
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize(
    "method", ["get_buildings", "get_landuse", "get_natural", "get_pois"]
)
def test_osm_out_of_core_layer_parity(method, helsinki_pbf):
    _osm_parity(helsinki_pbf, method)


def test_osm_out_of_core_network_parity(helsinki_pbf):
    _osm_parity(helsinki_pbf, "get_network", network_type="driving")


def test_osm_out_of_core_boundaries_parity(helsinki_region_pbf):
    _osm_parity(helsinki_region_pbf, "get_boundaries")
    _osm_parity(helsinki_region_pbf, "get_boundaries", extra_attributes=["wikidata"])


def test_osm_out_of_core_custom_criteria_parity(helsinki_pbf):
    _osm_parity(
        helsinki_pbf,
        "get_data_by_custom_criteria",
        custom_filter={"amenity": ["restaurant", "cafe"]},
    )
    # An explicit tags_as_columns (the validated branch) + extra_attributes + a string
    # osm_keys_to_keep (the str-wrap branch).
    _osm_parity(
        helsinki_pbf,
        "get_data_by_custom_criteria",
        custom_filter={"amenity": ["restaurant"]},
        tags_as_columns=["amenity", "name"],
        extra_attributes=["cuisine"],
        osm_keys_to_keep="amenity",
    )


def test_osm_out_of_core_custom_filter_parity(helsinki_pbf):
    # custom_filter with the layer key present, and without it (the engine ensures the key).
    _osm_parity(helsinki_pbf, "get_buildings", custom_filter={"building": ["retail"]})
    _osm_parity(helsinki_pbf, "get_buildings", custom_filter={"amenity": ["school"]})
    _osm_parity(helsinki_pbf, "get_landuse", custom_filter={"landuse": ["residential"]})
    _osm_parity(
        helsinki_pbf, "get_natural", custom_filter={"natural": ["wood", "water"]}
    )


def test_osm_out_of_core_extra_attributes_and_tags_to_keep_parity(helsinki_pbf):
    _osm_parity(helsinki_pbf, "get_buildings", extra_attributes=["wikidata"])
    _osm_parity(helsinki_pbf, "get_buildings", tags_to_keep=["building", "name"])
    _osm_parity(
        helsinki_pbf,
        "get_pois",
        custom_filter={"amenity": ["cafe"]},
        extra_attributes=["operator"],
    )
    _osm_parity(
        helsinki_pbf,
        "get_network",
        network_type="cycling",
        extra_attributes=["maxspeed"],
    )
    _osm_parity(helsinki_pbf, "get_network", tags_to_keep=["highway", "name"])


@pytest.mark.parametrize("complete_relations", [False, True])
def test_osm_out_of_core_bbox_complete_relations_parity(
    helsinki_pbf, complete_relations
):
    bbox = _central_bbox(OSM(helsinki_pbf).get_buildings())
    ref = OSM(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    ).get_buildings()
    mine = OSM(
        helsinki_pbf,
        bounding_box=bbox,
        complete_relations=complete_relations,
        engine="out_of_core",
    ).get_buildings()
    _assert_full_parity(mine, ref)


def test_osm_engine_validation_and_unsupported_combos(helsinki_pbf):
    with pytest.raises(ValueError):
        OSM(helsinki_pbf, engine="bogus")
    ooc = OSM(helsinki_pbf, engine="out_of_core")
    # A timestamp on a non-history file routes to the in-memory path, which rejects it
    # (history timestamps require an .osh.pbf) -- the engine does not silently ignore it.
    with pytest.raises(ValueError):
        ooc.get_buildings(timestamp="2021-01-01 00:00:00")
    with pytest.raises(NotImplementedError):
        ooc.get_data_by_custom_criteria(custom_filter=None)


def test_osm_out_of_core_network_nodes_parity(helsinki_pbf):
    # The public OSM API returns the (nodes, edges) tuple from the out-of-core engine, equal
    # to the in-memory reader's.
    mine_nodes, mine_edges = OSM(helsinki_pbf, engine="out_of_core").get_network(
        network_type="cycling", nodes=True
    )
    ref_nodes, ref_edges = OSM(helsinki_pbf).get_network(
        network_type="cycling", nodes=True
    )
    _assert_full_parity(mine_edges, ref_edges)
    a = mine_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    b = ref_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    _assert_full_parity(a, b)


# History (.osh) reads route to the in-memory selection even under engine="out_of_core"
# (the latest-version-per-element merge is global, so streaming gives no benefit); the
# result must match the in-memory reader exactly.


def _osm_history_parity(fp, method, timestamp, **kwargs):
    import copy

    ref = getattr(OSM(fp, engine="in_memory"), method)(
        timestamp=timestamp, **copy.deepcopy(kwargs)
    )
    mine = getattr(OSM(fp, engine="out_of_core"), method)(
        timestamp=timestamp, **copy.deepcopy(kwargs)
    )
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("timestamp", ["2010-01-01", "2015-01-01"])
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_network", {}),
        ("get_buildings", {}),
        ("get_landuse", {}),
        ("get_natural", {}),
        ("get_pois", {}),
        ("get_boundaries", {}),
        ("get_data_by_custom_criteria", {"custom_filter": {"highway": True}}),
    ],
)
def test_osm_out_of_core_history_routes_to_in_memory(
    helsinki_history_pbf, method, kwargs, timestamp
):
    _osm_history_parity(helsinki_history_pbf, method, timestamp, **kwargs)


def test_osm_out_of_core_history_without_timestamp_routes_to_in_memory(
    helsinki_history_pbf,
):
    # An .osh.pbf with no timestamp uses the current UTC time (with a warning) in both
    # engines; the out-of-core engine routes to the in-memory selection, matching it.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = OSM(helsinki_history_pbf, engine="in_memory").get_network()
        mine = OSM(helsinki_history_pbf, engine="out_of_core").get_network()
    _assert_full_parity(mine, ref)


def test_use_engine_routing(helsinki_pbf, helsinki_history_pbf):
    # The out-of-core engine handles only non-history reads; an .osh file or an explicit
    # timestamp routes to the in-memory path.
    assert OSM(helsinki_pbf, engine="out_of_core")._use_engine(None) is True
    assert OSM(helsinki_pbf, engine="out_of_core")._use_engine("2015-01-01") is False
    assert OSM(helsinki_pbf, engine="in_memory")._use_engine(None) is False
    osh = OSM(helsinki_history_pbf, engine="out_of_core")
    assert osh._use_engine(None) is False
    assert osh._use_engine("2015-01-01") is False
