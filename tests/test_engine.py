"""Out-of-core engine buildings reader: parity vs the in-memory
OSM(fp, engine="in_memory").get_buildings() way and relation rows, plus the output=
GeoParquet path and the worker-count policy."""

import re
import shutil
import threading
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

# Captured at import, before the autouse cache fixture of conftest.py stubs the module
# attribute, so a test can exercise the real cache_dir() implementation.
_real_cache_dir = cache.cache_dir


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
    gdf = OSM(fp, engine="in_memory").get_buildings()
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
    assert list(a.columns) == list(b.columns)
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
    ref = OSM(fp, engine="in_memory").get_buildings()
    _assert_full_parity(mine, ref)


def test_engine_buildings_keep_metadata_false_parity(helsinki_pbf):
    # keep_metadata=False must drop the element-metadata columns exactly as the in-memory
    # reader does (whatever its handling) -- full column + value parity either way.
    mine = get_buildings(helsinki_pbf, keep_metadata=False)
    ref = OSM(helsinki_pbf, keep_metadata=False, engine="in_memory").get_buildings()
    assert "changeset" not in mine.columns
    _assert_full_parity(mine, ref)


def test_engine_building_relations_match(helsinki_pbf):
    # The bundled Helsinki extract has building relations (multipolygons); they must
    # match the in-memory reader's relation rows exactly.
    mine = get_buildings(helsinki_pbf)
    ref = OSM(helsinki_pbf, engine="in_memory").get_buildings()
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
    # The in-memory reader reads them too.
    _assert_full_parity(
        OSM(raw_fp, engine="in_memory").get_buildings(),
        OSM(helsinki_pbf, engine="in_memory").get_buildings(),
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

    in_memory = OSM(helsinki_pbf, engine="in_memory").get_buildings()
    full_out = str(tmp_path / "buildings_full.parquet")
    get_buildings(helsinki_pbf, output=full_out)
    full_reload = gpd.read_parquet(full_out)

    monkeypatch.setattr(geoparquet, "_OUTPUT_CHUNK_SIZE", 20)
    chunked_out = str(tmp_path / "buildings_chunked.parquet")
    get_buildings(helsinki_pbf, output=chunked_out)
    assert pq.ParquetFile(chunked_out).metadata.num_row_groups > 1
    chunked_reload = gpd.read_parquet(chunked_out)

    # The chunked output carries the in-memory reader's columns in its order (a tag column
    # that first occurs in a later chunk keeps its place), and is identical value-for-value
    # to the single-chunk output.
    assert list(chunked_reload.columns) == list(in_memory.columns)
    _assert_full_parity(chunked_reload, full_reload)


def test_engine_output_requires_pyarrow(helsinki_pbf, tmp_path, monkeypatch):
    # Without pyarrow, output= must fail fast with an actionable error (and not decode).
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    with pytest.raises(ImportError, match="pyarrow"):
        get_buildings(helsinki_pbf, output=str(tmp_path / "x.parquet"))


def _fake_executor(raise_init=None, raise_map=None):
    """A ProcessPoolExecutor stand-in that runs map() and submit() in-process, optionally
    raising at construction or at map()/submit() to exercise the parallel branch and its
    fallback without real worker processes."""

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

        def submit(self, fn, *args):
            from concurrent.futures import Future

            if raise_map is not None:
                raise raise_map("simulated")
            future = Future()
            future.set_result(fn(*args))
            return future

    return _F


@pytest.mark.parametrize("where", ["construction", "run"])
def test_engine_parallel_decode_falls_back_to_serial(
    where, helsinki_pbf, monkeypatch, fresh_cache
):
    # A process pool that cannot start (OSError, e.g. a restricted environment) or whose
    # workers die (BrokenProcessPool, e.g. an unguarded __main__) must fall back to a single
    # process with a warning, not hang or error, and still produce the correct result.
    from concurrent.futures.process import BrokenProcessPool

    monkeypatch.setenv("PYROSM_DECODE_BACKEND", "processes")
    if where == "construction":
        fake = _fake_executor(raise_init=OSError)
    else:
        fake = _fake_executor(raise_map=BrokenProcessPool)
    monkeypatch.setattr(pool, "ProcessPoolExecutor", fake)
    with pytest.warns(RuntimeWarning, match="single process"):
        mine = get_buildings(helsinki_pbf, workers=3)
    _assert_full_parity(mine, OSM(helsinki_pbf, engine="in_memory").get_buildings())


@pytest.mark.parametrize("read", ["buildings", "pois", "network"])
def test_engine_parallel_decode_matches_serial(read, helsinki_pbf, monkeypatch):
    # A pool decodes the file in many small tasks and returns their shards in file order, so
    # the read yields exactly the serial result -- same rows in the same order -- and matches
    # the in-memory reader. HAS_PYARROW=False bypasses the result cache so both reads actually
    # run; the fake executor runs the pool in-process, without progress reports.
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    monkeypatch.setattr(pool, "_available_cores", lambda: 8)
    monkeypatch.setattr(pool, "ProcessPoolExecutor", _fake_executor())

    fn = {"buildings": get_buildings, "pois": get_pois, "network": get_network}[read]
    ref = {
        "buildings": lambda: OSM(helsinki_pbf, engine="in_memory").get_buildings(),
        "pois": lambda: OSM(helsinki_pbf, engine="in_memory").get_pois(),
        "network": lambda: OSM(helsinki_pbf, engine="in_memory").get_network(),
    }[read]()

    parallel = fn(helsinki_pbf, workers=3, progress=False)
    serial = fn(helsinki_pbf, workers=1, progress=False)
    # Identical rows in identical order (the ordering contract), not just the same set.
    assert list(parallel["id"]) == list(serial["id"])
    assert parallel.geometry.geom_equals_exact(serial.geometry, tolerance=0).all()
    _assert_full_parity(parallel, ref)


class _ThreadValue:
    """A ``multiprocessing.Value`` stand-in shared by thread workers."""

    def __init__(self, typecode, value):
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


@pytest.mark.parametrize("case", ["runs", "start fails", "breaks", "callback fails"])
def test_engine_run_pool_progress(monkeypatch, case):
    """While a pool runs, the bytes its workers add are reported from this process, and the
    results keep task order. A pool that cannot start, or breaks midway, falls back to one
    process that reports from 0 again and after each addition. An error of the callback
    itself, even an OSError, propagates as is instead of falling back."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    seen, in_this_thread = [], []
    release = threading.Event()
    main = threading.current_thread()

    def report(done, total):
        seen.append(done)
        if 0 < done < total:
            release.set()
            if case == "callback fails":
                raise OSError("stderr closed")

    def task(n, advance):
        advance(n)
        if threading.current_thread() is main:
            in_this_thread.append(n)
            return n * 2
        release.wait(5)
        if case == "breaks":
            raise BrokenProcessPool("simulated")
        return n * 2

    class Threads(ThreadPoolExecutor):
        def __init__(self, max_workers, initializer, initargs):
            if case == "start fails":
                raise OSError("simulated")
            super().__init__(max_workers, initializer=initializer, initargs=initargs)

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Threads)
    monkeypatch.setattr(pool.multiprocessing, "Value", _ThreadValue)
    monkeypatch.setattr(pool, "_COUNTER", None)
    monkeypatch.setattr(pool, "_PROGRESS_INTERVAL", 0.01)
    if case == "callback fails":
        with pytest.raises(OSError, match="stderr closed"):
            pool._run_pool(task, [1, 2, 3], 3, progress=(report, 4, 20))
        assert in_this_thread == []  # no fallback to one process
        return
    results, pool_ok = pool._run_pool(task, [1, 2, 3], 3, progress=(report, 4, 20))
    assert results == [2, 4, 6]
    assert pool_ok is (case == "runs")
    assert seen[0] == 0 and seen[-1] == 20
    if case == "start fails":
        assert seen == [0, 5, 7, 10, 20]
    else:
        restart = seen.index(0, 1) if case == "breaks" else len(seen) - 1
        assert any(4 < done < 20 for done in seen[1:restart])
        if case == "breaks":
            assert seen[restart:] == [0, 5, 7, 10, 20]


def test_engine_read_progress(test_pbf, fresh_cache):
    """The engine reports the percentage of the whole read done, as (done, 100), from 0
    to 100 and never back, also through the second pass of get_network(nodes=True). The
    header and the data blob spans cover the file exactly; progress values other than
    True, False or a callable are refused before reading, by OSM and by the engine
    readers."""
    from pathlib import Path

    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs

    size = Path(test_pbf).stat().st_size
    data, header = _data_blob_spans(_index_blobs(test_pbf))
    assert header + sum(span for _, _, span in data) == size
    reads = [get_buildings, lambda fp, **kw: get_network(fp, nodes=True, **kw)]
    frames = []
    for read in reads:
        seen = []
        frames.append(read(test_pbf, workers=1, progress=lambda *r: seen.append(r)))
        done = [d for d, _ in seen]
        assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100)
        assert all(total == 100 for _, total in seen) and done == sorted(done)
    # The frame equals one decoded again without progress (not served from the cache).
    shutil.rmtree(fresh_cache)
    fresh_cache.mkdir()
    _assert_full_parity(frames[0], get_buildings(test_pbf, workers=1, progress=False))
    for refused in (
        lambda: OSM(test_pbf, progress="yes", engine="in_memory"),
        lambda: get_buildings(test_pbf, progress="yes"),
    ):
        with pytest.raises(ValueError, match="progress must be"):
            refused()


@pytest.mark.parametrize("backend", ["processes", "threads"])
def test_engine_read_progress_in_parallel(test_pbf, monkeypatch, fresh_cache, backend):
    """A read in worker processes or on threads reports from (0, 100) to (100, 100)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    if (os.cpu_count() or 1) < 2:
        pytest.skip("workers=2 runs on one core on a single-core host")
    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    if backend == "processes":
        try:
            with ProcessPoolExecutor(1) as executor:
                executor.submit(int).result()
        except (OSError, BrokenProcessPool) as e:
            pytest.skip("process pools cannot start here: %s" % e)
    ran = []
    real = pool._decode_all

    def spy(*args, **kwargs):
        shards, pool_ok = real(*args, **kwargs)
        ran.append(pool_ok)
        return shards, pool_ok

    monkeypatch.setattr(pool, "_decode_all", spy)
    seen = []
    get_buildings(test_pbf, workers=2, progress=lambda *report: seen.append(report))
    assert ran == [True]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_engine_run_pool_silent_fallback_without_warning(monkeypatch):
    # _run_pool without a fallback_warning falls back to a single process silently when the
    # pool cannot start; it still runs every task and reports pool_ok=False.
    import warnings as _warnings

    monkeypatch.setattr(pool, "ProcessPoolExecutor", _fake_executor(raise_init=OSError))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        results, pool_ok = pool._run_pool(lambda t, advance: t * 2, [1, 2, 3], 3)
    assert results == [2, 4, 6]
    assert pool_ok is False
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


@pytest.mark.parametrize(
    "backend, target",
    [("processes", 1), ("threads", 1), ("threads", 8_000_000)],
)
def test_engine_decode_shard_target_parity(
    helsinki_pbf, monkeypatch, fresh_cache, backend, target
):
    # The decode spill accumulates blocks into a shard until it reaches the byte target. A tiny
    # target spills one shard per block (and exercises the empty final-flush guard); the normal
    # target batches many blocks per shard. Either way the read matches the in-memory reader.
    # fresh_cache forces a real decode rather than a cached result.
    from pyrosm.engine import decode

    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    monkeypatch.setattr(decode, "_SHARD_TARGET_BYTES", target)
    _assert_full_parity(get_buildings(helsinki_pbf), OSM(helsinki_pbf, engine="in_memory").get_buildings())


@pytest.mark.parametrize("backend", ["processes", "threads"])
def test_engine_unguarded_module_read_does_not_hang(test_pbf, tmp_path, backend):
    # A read at module level (no `if __name__ == "__main__":` guard) must not hang. With
    # processes the spawned workers cannot re-import the entry point, so the decode may fall
    # back to a single process; threads decode in parallel without that fallback. Run it as a
    # subprocess with a timeout so a regression surfaces as a failure.
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent("""
            import warnings
            from pyrosm import engine
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gdf = engine.get_buildings(%r, workers=3)
            fell_back = any("single process" in str(w.message) for w in caught)
            print("ROWS", 0 if gdf is None else len(gdf), "FELL_BACK", fell_back)
            """ % test_pbf))
    # The child process has its own temporary folder, so it neither reads nor writes a
    # result cached outside this test.
    env = dict(
        os.environ,
        TMPDIR=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        PYROSM_DECODE_BACKEND=backend,
    )
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert "ROWS" in r.stdout, r.stderr[-800:]
    if backend == "threads":
        assert "FELL_BACK False" in r.stdout


_SHARD_KEYS = (
    "node_id node_lon node_lat in_box_id nfeat_id nfeat_lon nfeat_lat nfeat_tags "
    "nfeat_meta way_id refs refs_off way_tags way_version way_timestamp way_visible "
    "all_id all_refs all_refs_off rel_id rel_memid rel_memoff rel_memtype rel_memrole "
    "rel_tags rel_meta"
).split()
_OBJECT_SHARD_KEYS = ("nfeat_tags", "way_tags", "rel_memrole", "rel_tags")


@pytest.fixture(scope="module")
def repacked_pbf(tmp_path_factory):
    """The bundled Helsinki extract with its ways in blocks of their own (the bundled file
    keeps them in one block with the relations)."""
    from pyrosm.pbf_export import crop_pbf

    path = tmp_path_factory.mktemp("repacked") / "helsinki.osm.pbf"
    crop_pbf(
        get_data("helsinki_pbf"), str(path), [24.0, 60.0, 26.0, 61.0], repack=True, workers=1
    )
    return str(path)


def _decoded_shards(filepath, backend, workers, shard_dir, monkeypatch, **kwargs):
    """The shards ``backend`` decodes ``filepath`` into, merged into one shard dict, with
    the object arrays (tag dicts, member roles) as lists."""
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
    from pyrosm.engine.decode import _merge_shards, _open_shard, _shard_tags

    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    blobs, _ = _data_blob_spans(_index_blobs(filepath))
    shard_dir.mkdir()
    shards, _ = pool._decode_all(filepath, blobs, workers, str(shard_dir), **kwargs)
    keys = [key for key in _SHARD_KEYS if key not in _OBJECT_SHARD_KEYS]
    merged = _merge_shards([{key: _open_shard(s)[key] for key in keys} for s in shards])
    for key in _OBJECT_SHARD_KEYS:
        if key in ("way_tags", "nfeat_tags"):
            parts = [_shard_tags(_open_shard(s), key[: -len("_tags")]) for s in shards]
        else:
            parts = [_open_shard(s)[key] for s in shards]
        merged[key] = [item for part in parts for item in part]
    return merged


@pytest.mark.parametrize(
    "workers, kwargs",
    [
        (1, dict(osm_keys=[b"building"], include_nodes=True)),
        (3, dict(osm_keys=None, include_nodes=True)),
        (
            3,
            dict(
                osm_keys=[b"highway"],
                include_nodes=False,
                requested_tag_keys=[b"highway", b"name"],
                spill_all_ways=False,
            ),
        ),
        (
            1,
            dict(
                osm_keys=[b"natural", b"water"],
                include_nodes=True,
                bbox_bounds=(24.940, 60.168, 24.948, 60.174),
            ),
        ),
        (
            1,
            dict(
                osm_keys=[b"k%d" % i for i in range(70)] + [b"amenity"],
                include_nodes=True,
            ),
        ),
        (2, dict(osm_keys=[b"building"], include_nodes=True, limits=("raw", 1))),
        (2, dict(osm_keys=[b"building"], include_nodes=True, limits=("span", 1))),
    ],
    ids=[
        "key",
        "every tag",
        "minimal tags",
        "bounding box",
        "many keys",
        "blob per part",
        "blob read alone",
    ],
)
def test_engine_threaded_decode_matches_processes(
    repacked_pbf, tmp_path, monkeypatch, workers, kwargs
):
    # The GIL-free threaded decode writes the same shard arrays, in the same order and
    # with the same dtypes, as the decode of the worker processes, and elements with the
    # same tags, also when each blob of a batch is decompressed in a part of its own, or
    # read alone as a batch spanning more than _BATCH_BYTES is.
    from pyrosm.engine import blobs

    kwargs = dict(kwargs)
    limit, value = kwargs.pop("limits", (None, None))
    if limit is not None:
        name = {"raw": "_BATCH_RAW_BYTES", "span": "_BATCH_BYTES"}[limit]
        monkeypatch.setattr(blobs, name, value)
    ref = _decoded_shards(
        repacked_pbf, "processes", 1, tmp_path / "processes", monkeypatch, **kwargs
    )
    mine = _decoded_shards(
        repacked_pbf, "threads", workers, tmp_path / "threads", monkeypatch, **kwargs
    )
    assert len(ref["way_id"]) and len(ref["node_id"])
    for key in _SHARD_KEYS:
        if key in _OBJECT_SHARD_KEYS:
            assert [np.asarray(v).tolist() for v in mine[key]] == [
                np.asarray(v).tolist() for v in ref[key]
            ], key
        else:
            assert mine[key].dtype == ref[key].dtype, key
            assert mine[key].tolist() == ref[key].tolist(), key


def test_engine_batch_parts_are_released_before_the_next(tmp_path, monkeypatch):
    # A thread's batch is decompressed in parts of at most _BATCH_RAW_BYTES, each
    # released before the next one is decompressed, so the traced memory peaks at about
    # one part.
    import tracemalloc

    from pyrosm.engine import blobs, decode

    part = 2 * 2**20
    monkeypatch.setattr(blobs, "_BATCH_RAW_BYTES", part)
    zeros = b"\0" * 2**20
    fp = tmp_path / "zeros.osm.pbf"
    with open(fp, "wb") as f:
        for _ in range(8):
            blob = Blob(zlib_data=zlib.compress(zeros), raw_size=len(zeros))
            data = blob.SerializeToString()
            header = BlobHeader(type="OSMData", datasize=len(data)).SerializeToString()
            f.write(pack("!L", len(header)) + header + data)
    sizes = []
    tracemalloc.start()
    try:
        spans, _ = blobs._data_blob_spans(blobs._index_blobs(str(fp)))
        with open(fp, "rb") as f:
            for _, buffer, _ in decode._decoded_parts(f, spans):
                sizes.append(len(buffer))
                del buffer
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sizes == [part] * 4
    assert peak < 1.5 * part


def test_engine_threaded_decode_releases_each_part(tmp_path, monkeypatch):
    # The shards pending at the end of a part are copied out of its decoded arrays, so
    # a part is released before the next one is decoded: with parts of 4 blocks and a
    # shard flushed after 3 of them, the traced memory stays below one part's arrays (4
    # blocks) and the decoder's scratch, one flushed shard (3 blocks) and the pending
    # tail block (it reached about 12 blocks when the tail kept the whole part).
    import tracemalloc

    from pyrosm.engine import blobs, decode
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    n = 50_000
    block = PrimitiveBlock(granularity=100)
    block.stringtable.s.append(b"")
    dense = block.primitivegroup.add().dense
    dense.id.extend([1] * n)
    dense.lat.extend([600_000_000] + [0] * (n - 1))
    dense.lon.extend([250_000_000] + [0] * (n - 1))
    data = block.SerializeToString()
    fp = tmp_path / "nodes.osm.pbf"
    with open(fp, "wb") as f:
        for _ in range(8):
            blob = Blob(zlib_data=zlib.compress(data), raw_size=len(data))
            blob = blob.SerializeToString()
            header = BlobHeader(type="OSMData", datasize=len(blob)).SerializeToString()
            f.write(pack("!L", len(header)) + header + blob)
    per_block = 24 * n  # a block's node ids, longitudes and latitudes
    monkeypatch.setenv("PYROSM_DECODE_BACKEND", "threads")
    monkeypatch.setattr(blobs, "_BATCH_RAW_BYTES", 4 * len(data))
    monkeypatch.setattr(decode, "_SHARD_TARGET_BYTES", int(2.5 * per_block))
    spans, _ = blobs._data_blob_spans(blobs._index_blobs(str(fp)))
    tracemalloc.start()
    try:
        pool._decode_all(str(fp), spans, 1, str(tmp_path), [b"building"], False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 10 * per_block


@pytest.mark.parametrize(
    "osm_keys, data_filter, bbox, exclude",
    [
        ([b"building"], {"building": ["yes"]}, None, False),
        ([b"highway"], {"highway": ["footway", "path"]}, (24.940, 60.168, 24.948, 60.174), False),
        ([b"natural", b"water"], {"natural": ["water"]}, None, True),
    ],
    ids=["values", "values in a box", "excluded members"],
)
def test_engine_element_store_selects_as_the_python_path(
    repacked_pbf, tmp_path, monkeypatch, osm_keys, data_filter, bbox, exclude
):
    # The ways a threaded decode's element store keeps (value filter, nodes in the box,
    # relation members excluded) are the ways the Python path keeps from the same shards,
    # with the same columns.
    from pyrosm.engine import collect
    from pyrosm.engine.bounding_box import _in_box_nodes
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
    from pyrosm.engine.ways import ElementStore, kept_ways

    monkeypatch.setenv("PYROSM_DECODE_BACKEND", "threads")
    blobs, _ = _data_blob_spans(_index_blobs(repacked_pbf))
    shards, _ = pool._decode_all(
        repacked_pbf, blobs, 2, str(tmp_path), osm_keys, True, bbox_bounds=bbox
    )
    spec = ([k.decode() for k in osm_keys], data_filter, "keep")
    keep = collect._keep_fn(spec)
    in_box = _in_box_nodes(shards) if bbox is not None else None
    store = ElementStore.from_shards(shards)
    exclude_ids = store.id[::3] if exclude else np.empty(0, np.int64)
    mine = kept_ways(store, exclude_ids, spec, in_box).way_columns()
    ref = collect._filter_way_columns(
        collect._read_way_columns(shards), exclude_ids, keep, in_box
    )
    assert 0 < len(ref["id"]) < len(store)
    assert list(mine) == list(ref)
    for key in ("id", "version", "timestamp", "visible"):
        assert mine[key].tolist() == ref[key].tolist(), key
    assert list(mine["tags"]) == list(ref["tags"])
    assert [n.tolist() for n in mine["nodes"]] == [n.tolist() for n in ref["nodes"]]


def test_way_columns_take_and_membership():
    # Rows of a ragged column, ids in a set, and ways with a node in a set; offsets that do
    # not fit the values and rows outside the column raise.
    from pyrosm import way_columns as wc
    from pyrosm.int64_hash import Int64Set

    off, values = np.array([0, 2, 2, 5]), np.array([1, 2, 3, 4, 5])
    new_off, new_values = wc.take(off, values, np.array([2, 0, 1]))
    assert new_off.tolist() == [0, 3, 5, 5] and new_values.tolist() == [3, 4, 5, 1, 2]
    assert wc.in_set(np.array([1, 7, 3]), Int64Set(np.array([3, 1]))).tolist() == [
        True,
        False,
        True,
    ]
    nodes = Int64Set(np.array([4]))
    assert wc.with_node_in(off, values, nodes).tolist() == [False, False, True]
    for bad_off, rows in (([0, 2, 9], [0]), ([1, 5], [0]), ([0, 3, 2, 5], [0])):
        with pytest.raises(ValueError, match="offsets"):
            wc.take(np.array(bad_off), values, np.array(rows))
    for rows in ([3], [-1]):
        with pytest.raises(ValueError, match="rows"):
            wc.take(off, values, np.array(rows))
    with pytest.raises(ValueError, match="offsets"):
        wc.with_node_in(np.array([0, 9]), values, nodes)
    flat = np.empty((1, 0), dtype=np.int64)
    for call in (
        lambda: wc.take(np.array([0, 1]), flat, np.array([0])),
        lambda: wc.in_set(flat, Int64Set(np.array([1]))),
        lambda: wc.with_node_in(np.array([0, 1]), flat, nodes),
    ):
        with pytest.raises(ValueError, match="one-dimensional"):
            call()


def _way_store(ways):
    """An :class:`~pyrosm.engine.ways.ElementStore` of ``ways``: ``(id, refs, tags)``
    with the tags as ``(key, value)`` bytes, version and timestamp from the id,
    visible."""
    from pyrosm.engine.ways import ElementStore

    strings, pairs, refs, refs_off, pairs_off = [], [], [], [0], [0]
    for _, way_refs, tags in ways:
        refs.extend(way_refs)
        refs_off.append(len(refs))
        for key, value in tags:
            pairs.append((len(strings), len(strings) + 1))
            strings += [key, value]
        pairs_off.append(len(pairs))
    ids = np.array([w[0] for w in ways], dtype=np.int64)
    str_off = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in strings], out=str_off[1:])
    return ElementStore(
        id=ids,
        refs_off=np.array(refs_off, dtype=np.int64),
        refs=np.array(refs, dtype=np.int64),
        version=ids + 1,
        timestamp=ids * 1000,
        visible=ids % 2,
        pairs_off=np.array(pairs_off, dtype=np.int64),
        pair_key=np.array([k for k, _ in pairs], dtype=np.int64),
        pair_val=np.array([v for _, v in pairs], dtype=np.int64),
        str_off=str_off,
        str_data=np.frombuffer(b"".join(strings), dtype=np.uint8),
    )


_FILTER_WAYS = [
    (1, [1, 2], [(b"highway", b"residential")]),
    (2, [1, 2], [(b"highway", b"footway")]),
    (3, [1, 2], [(b"highway", b"residential"), (b"access", b"no"), (b"motorcar", b"yes")]),
    (4, [1, 2], [(b"highway", b"residential"), (b"access", b"private")]),
    (5, [1, 2], [(b"highway", b"service"), (b"service", b"driveway")]),
    (6, [1, 2], [(b"highway", b"residential"), (b"area", b"yes")]),
    (7, [1, 2], [(b"highway", b"residential"), (b"access", "\u00a0private;x".encode())]),
    (8, [1, 2], [(b"highway", b"residential"), (b"access", b"yes ; private")]),
    (9, [1, 2], [(b"highway", b"footway"), (b"highway", b"residential")]),
    (10, [1, 2], [(b"building", b"yes"), (b"name", b"x")]),
    (11, [1, 2], [(b"building", b"roof")]),
    (12, [1, 2], []),
    (13, [1, 2], [(b"building", b"\xff")]),
]


@pytest.mark.parametrize("filter_type", ["keep", "exclude"])
@pytest.mark.parametrize(
    "custom_filter",
    [
        {"building": [True]},
        {"building": ["yes", "garage"]},
        {"highway": ["footway"]},
        "driving",
        "walking",
        {"building": "yes"},
        {"building": ["\ufffd"]},
        {"building": ["\ud800"]},
        '["building"~"^ro"]',
        None,
    ],
)
def test_engine_way_value_filter_matches_python(custom_filter, filter_type):
    # The value filter of a store's ways decides as the Python predicate does on their tag
    # dicts (a filter it does not cover is left to that predicate).
    from pyrosm.config.osm_filters import get_osm_filter
    from pyrosm.data_filter import element_should_be_kept
    from pyrosm.engine.ways import keep_mask
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        spec = (None, None, filter_type)
    elif custom_filter in ("driving", "walking"):
        spec = (["highway"], get_osm_filter(custom_filter), filter_type)
    elif isinstance(custom_filter, str):
        compiled = validate_custom_filter(custom_filter)
        spec = (list(compiled.keys()), compiled, filter_type)
    else:
        spec = (list(custom_filter), custom_filter, filter_type)
    store = _way_store(_FILTER_WAYS)
    expected = [
        spec[1] is None
        or element_should_be_kept(store.tags(w), spec[0], spec[1], filter_type)
        for w in range(len(store))
    ]
    assert keep_mask(store, spec).tolist() == expected


def test_way_columns_string_columns_and_value_filter_check_their_input():
    # Each string's validity and column; tag columns that do not fit together raise.
    from pyrosm import way_columns as wc

    strings = [b"building", b"yes", b"id", b"\xff"]
    str_off = np.cumsum([0] + [len(x) for x in strings])
    str_data = np.frombuffer(b"".join(strings), dtype=np.uint8)
    valid, column = wc.string_columns(str_off, str_data, [b"yes"])
    assert valid.tolist() == [True, True, True, False]
    assert column.tolist() == [-1, 0, -2, -1]
    good = dict(
        pairs_off=np.array([0, 1]),
        pair_key=np.array([0]),
        pair_val=np.array([1]),
        str_off=str_off,
        str_data=str_data,
        valid=valid.view(np.uint8),
        rows=np.array([0]),
    )
    args = dict(osm_keys=[b"building"], entries=[], keep=True)
    assert wc.value_filter(**good, **args).tolist() == [wc.KEEP]
    for entries in ([(9, [b"building"], [])], [(wc.FILTER_VALUES, [], [])]):
        with pytest.raises(ValueError, match="entry"):
            wc.value_filter(**good, **dict(args, entries=entries))
    for name, value, match in (
        ("pair_val", np.array([9]), "strings of the store"),
        ("pair_key", np.array([0, 1]), "offsets"),
        ("rows", np.array([1]), "ways of the store"),
        ("valid", np.zeros(2, dtype=np.uint8), "one flag per string"),
    ):
        with pytest.raises(ValueError, match=match):
            wc.value_filter(**dict(good, **{name: value}), **args)


# A polygon inside the bundled Helsinki extract, for reads with a polygon bounding box.
_TRIANGLE = shapely.Polygon([(24.936, 60.165), (24.952, 60.166), (24.944, 60.178)])


def _assert_identical(mine, ref):
    """``mine`` and ``ref`` are the same frame: columns, dtypes, values and geometries."""
    import pandas as pd

    assert list(mine.columns) == list(ref.columns)
    for col in ref.columns:
        if col == "geometry":
            a, b = mine[col].to_numpy(), ref[col].to_numpy()
            assert shapely.to_wkb(a).tolist() == shapely.to_wkb(b).tolist()
        else:
            pd.testing.assert_series_equal(
                mine[col], ref[col], check_exact=True, check_dtype=True
            )


@pytest.mark.parametrize(
    "read, kwargs",
    [
        (get_network, {"network_type": "driving", "nodes": True}),
        (
            get_network,
            {"network_type": "driving", "nodes": True, "bounding_box": _TRIANGLE},
        ),
        (
            get_network,
            {"network_type": "walking", "nodes": True, "keep_metadata": False},
        ),
        (get_landuse, {"bounding_box": [24.940, 60.168, 24.948, 60.174]}),
        (get_buildings, {}),
    ],
    ids=[
        "driving nodes",
        "driving nodes polygon",
        "walking nodes",
        "landuse box",
        "buildings",
    ],
)
def test_engine_threaded_node_records_and_relation_ways_match_processes(
    repacked_pbf, tmp_path, monkeypatch, read, kwargs
):
    # A threaded read gathers the network's node records (nodes=True) and the relations'
    # member ways on threads, here in batches of 2 blocks and over shards of 50 kB; it
    # gives the frames of the worker processes' read.
    from pyrosm.engine import cache, decode

    monkeypatch.setattr(decode, "_THREAD_BATCH", 2)
    monkeypatch.setattr(decode, "_SHARD_TARGET_BYTES", 50_000)
    frames = {}
    for backend, workers in (("processes", 1), ("threads", 3)):
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
        folder = tmp_path / backend
        folder.mkdir()
        monkeypatch.setattr(cache, "cache_dir", lambda: folder)
        result = read(repacked_pbf, workers=workers, **kwargs)
        frames[backend] = result if isinstance(result, tuple) else (result,)
    for mine, ref in zip(frames["threads"], frames["processes"]):
        assert len(ref) > 0
        _assert_identical(mine, ref)


def test_engine_node_coordinates_match_node_lookup(tmp_path):
    # The coordinates gathered from threaded shards are those _node_lookup gathers from
    # spilled shards: ids ascending, the last shard holding an id winning, absent ids left
    # out; shards whose ids do not ascend are sorted.
    from pyrosm.engine.collect import _node_lookup
    from pyrosm.engine.decode import _Shard
    from pyrosm.engine.ways import node_coordinates

    parts = [
        {"node_id": [5, 1, 9], "node_lon": [5.0, 1.0, 9.0], "node_lat": [50.0, 10.0, 90.0]},
        {"node_id": [], "node_lon": [], "node_lat": []},
        {"node_id": [1, 7], "node_lon": [1.5, 7.0], "node_lat": [15.0, 70.0]},
    ]
    shards, paths = [], []
    for i, part in enumerate(parts):
        arrays = {
            "node_id": np.array(part["node_id"], dtype=np.int64),
            "node_lon": np.array(part["node_lon"]),
            "node_lat": np.array(part["node_lat"]),
        }
        shards.append(_Shard.write(tmp_path / ("s%d.bin" % i), arrays, {}))
        paths.append(tmp_path / ("s%d.npz" % i))
        np.savez(paths[-1], **arrays)
    refs = [np.array([9, 7, 1, 42, 1], dtype=np.int64)]
    query = np.array([1, 5, 7, 9, 42], dtype=np.int64)
    mine = node_coordinates(shards, refs).gather(query)
    ref = _node_lookup(paths, np.unique(np.concatenate(refs))).gather(query)
    for a, b in zip(mine, ref):
        np.testing.assert_array_equal(a, b)
    assert len(node_coordinates([shards[1]], refs)) == 0
    assert len(node_coordinates(shards, [])) == 0


def test_engine_thread_map_bounds_the_items_in_flight():
    # thread_map yields in order and submits at most two items per thread ahead of the one
    # it waits for: while item 0 is held, no more than four items have started.
    import time

    release = threading.Event()
    started, out = [], []

    def work(i):
        started.append(i)
        if i == 0:
            release.wait(10)
        return i

    def consume():
        token = pool._READ_THREADS.set(2)
        try:
            out.extend(pool.thread_map(work, range(20)))
        finally:
            pool._READ_THREADS.reset(token)

    consumer = threading.Thread(target=consume)
    consumer.start()
    time.sleep(0.2)
    assert len(started) <= 4
    release.set()
    consumer.join(10)
    assert out == list(range(20))


# Ways for the tag columns and the chunks of a store: ring orientation (clockwise, flat,
# and collinear points that double precision cannot orient), the area and linear tags of
# closed ways, missing and invalid nodes, the ring closing rule, JSON escaping, special
# ways (an ``id`` key, a key twice, a value that is not UTF-8, a metadata name as a key,
# many keys with one twice), and negative metadata.
_NODES = {
    1: (0.0, 0.0),
    2: (0.0, 1.0),
    3: (1.0, 1.0),
    4: (1.0, 0.0),
    5: (0.0, 2.0),
    6: (2.0, 2.0),
    50: (10.0, 95.0),
}
_BUILDING = [(b"building", b"yes")]
_METADATA = ("version", "timestamp", "visible")
_ESCAPED = 'Tie "A"\n\t\x01\x7f\\/\r\b\x0cä€\U0001f600'.encode()
_CRAFTED_WAYS = [
    (1, [1, 2, 3, 4, 1], _BUILDING),
    (2, [1, 4, 3, 2, 1], _BUILDING + [(b"source", b"survey")]),
    (3, [1, 2, 3], [(b"highway", b"footway"), (b"name", _ESCAPED), (b"note", _ESCAPED)]),
    (4, [1, 2, 3, 1], [(b"highway", b"footway")]),
    (5, [1, 2, 3, 1], [(b"highway", b"pedestrian"), (b"area", b"yes")]),
    (6, [1, 2, 3, 4, 1], [(b"area", b"no"), (b"building", b"roof")]),
    (12, [1, 2, 3, 1], [(b"id", b"5"), (b"building", b"yes")]),
    (13, [1, 2, 3, 1], [(b"building", b"yes"), (b"building", b"no")]),
    (14, [1, 2, 3, 1], [(b"name", b"\xff\xfe"), (b"building", b"yes")]),
    (15, [1, 2, 3, 1], [(b"version", b"7"), (b"building", b"yes")]),
    (18, [1, 2, 3, 4, 1], []),
    (19, [1, 2, 3, 1], [(b"k%d" % i, b"v") for i in range(20)]),
    (20, [1, 2, 3, 1], [(b"k%d" % i, b"v") for i in range(20)] + [(b"k3", b"x")]),
    (-7, [1, 2, 3], [(b"highway", b"path"), (b"surface", b"gravel")]),
    (7, [1, 2, 99, 3, 1], _BUILDING),
    (8, [1, 2, 98, 99, 1], _BUILDING),
    (9, [1, 50, 2], [(b"barrier", b"fence")]),
    (10, [1, 99], _BUILDING),
    (11, [], _BUILDING),
    (16, [1, 2, 5, 1], _BUILDING),
    (17, [1, 3, 6, 1], _BUILDING),
]


@pytest.mark.parametrize(
    "ways, tags_as_columns, keep_metadata, keep_other_tags",
    [
        (slice(None), ["building", "highway", "name", "area", "id", "id_tag"], False, True),
        (slice(None), ["building", "source", "id_tag"], True, False),
        (slice(0, 6), ["building", "highway", "name", "area", "source", "note"], True, True),
        (slice(10, 11), ["building"], True, True),
    ],
    ids=["all ways", "requested tags with metadata", "no leftovers", "untagged"],
)
def test_engine_store_tag_columns_match_the_python_path(
    ways, tags_as_columns, keep_metadata, keep_other_tags, monkeypatch
):
    # The tag columns a store's ways give, found in C (special ways by their tag dicts),
    # are those the Python path finds in the ways' tag dicts, the tag columns and JSON
    # values built in C are those it routes the tag dicts to, and so are the way arrays
    # built from the store (tag dicts without pyarrow).
    from pyrosm.engine import collect
    from pyrosm.engine.ways import result_tag_columns, tag_lists
    from pyrosm.utils import _compat

    store = _way_store(_CRAFTED_WAYS[ways])
    keys = None if keep_other_tags else set(tags_as_columns)
    ref = collect._result_tag_columns(
        store.way_columns(keys), tags_as_columns, keep_metadata, keep_other_tags
    )
    mine = result_tag_columns(store, tags_as_columns, keep_metadata, keep_other_tags)
    assert mine == ref
    assert collect._result_tag_columns(
        store, tags_as_columns, keep_metadata, keep_other_tags
    ) == ref
    column_keys, tag_cols = collect._way_column_keys(tags_as_columns, keep_metadata)
    # Without other tags the process decode's tag dicts also hold the geometry tag keys.
    from pyrosm.engine.readers import _GEOMETRY_TAG_KEYS

    by_dicts = store.way_columns(None if keys is None else keys | set(_GEOMETRY_TAG_KEYS))
    col_lists, leftover, _ = collect._route_tags(
        by_dicts["tags"],
        store.version,
        store.timestamp,
        store.visible,
        tag_cols,
        keep_other_tags,
        *(keep_other_tags and k not in column_keys for k in _METADATA),
    )
    columns, json = tag_lists(store, tags_as_columns, keep_metadata, keep_other_tags)
    assert {k: v.tolist() for k, v in columns.items()} == col_lists
    assert json.tolist() == leftover
    for columns, has_pyarrow in ((None, True), (ref, True), (ref, False)):
        monkeypatch.setattr(_compat, "HAS_PYARROW", has_pyarrow)
        expected, got = (
            collect._ways_arrays(
                cols, tags_as_columns, keep_metadata, keep_other_tags, columns=columns
            )
            for cols in (by_dicts, store)
        )
        assert list(got) == list(expected)
        for key in expected:
            assert got[key].dtype == expected[key].dtype, key
            assert [np.asarray(v).tolist() for v in got[key]] == [
                np.asarray(v).tolist() for v in expected[key]
            ], key


def test_way_columns_special_ways_and_column_presence():
    # The special ways of a store and the columns its ways' keys route to; arrays that do
    # not fit together raise.
    from pyrosm import way_columns as wc

    store = _way_store(_CRAFTED_WAYS)
    valid, column = wc.string_columns(store.str_off, store.str_data, [b"building"])
    special = wc.special_ways(
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid,
        column,
    )
    assert store.id[special].tolist() == [12, 13, 14, 15, 20]
    present, leftover = wc.column_presence(
        store.pairs_off, store.pair_key, column, np.array([0, 3]), 1
    )
    assert present.tolist() == [True] and leftover
    present, leftover = wc.column_presence(
        store.pairs_off, store.pair_key, column, np.array([0]), 1
    )
    assert present.tolist() == [True] and not leftover
    with pytest.raises(ValueError, match="one entry per string"):
        wc.special_ways(
            store.pairs_off,
            store.pair_key,
            store.pair_val,
            store.str_off,
            store.str_data,
            valid[:-1],
            column,
        )
    for rows, n_columns, match in (([99], 1, "ways of the store"), ([0], 0, "below")):
        with pytest.raises(ValueError, match=match):
            wc.column_presence(
                store.pairs_off, store.pair_key, column, np.array(rows), n_columns
            )


def _crafted_nodes():
    """The coordinate store of ``_NODES``."""
    from pyrosm.node_lookup import NodeLocations

    ids = np.array(sorted(_NODES), dtype=np.int64)
    return NodeLocations(
        {
            "id": ids,
            "lon": np.array([_NODES[i][0] for i in ids]),
            "lat": np.array([_NODES[i][1] for i in ids]),
        }
    )


@pytest.mark.parametrize("keep_other_tags", [True, False])
def test_engine_way_geometries_match_python_path(keep_other_tags):
    # The geometries of a chunk of a store's ways built without the GIL (a ring that
    # double precision cannot orient oriented by GEOS) are those the Python path builds
    # for the same ways' tag dicts, and the chunk's tag columns and JSON are those of
    # tag_lists for its ways; a chunk needs a node store.
    from pyrosm.engine import ways
    from pyrosm.engine.assemble import _assemble_chunk
    from pyrosm.engine.collect import _way_column_keys

    nodes = _crafted_nodes()
    store = _way_store(_CRAFTED_WAYS)
    tags_as_columns = ["building", "highway", "area", "name", "id_tag", "barrier"]
    _, tag_cols = _way_column_keys(tags_as_columns, True)
    ctx = ways.ChunkContext(
        store, tags_as_columns, True, keep_other_tags, set(tag_cols) | {"tags"}, nodes
    )
    chunk, rows = ways._build_chunk(ctx, np.arange(len(store)))
    gdf = _assemble_chunk(
        nodes,
        store.way_columns(None if keep_other_tags else set(tags_as_columns)),
        None,
        None,
        tags_as_columns,
        True,
        keep_other_tags=keep_other_tags,
    )
    assert store.id[rows].tolist() == gdf["id"].tolist()
    assert store.id[rows[chunk["check"]]].tolist() == [17]
    wkb, off = chunk["wkb"], chunk["wkb_off"].tolist()
    geometries = shapely.from_wkb([wkb[a:b].tobytes() for a, b in zip(off, off[1:])])
    assert shapely.equals_exact(geometries, gdf.geometry.values, tolerance=0).all()
    # The first two ways alone (no special way, no ring for GEOS) give the same WKB.
    head, head_rows = ways._build_chunk(ctx, np.arange(2))
    assert head_rows.tolist() == [0, 1] and len(head["check"]) == 0
    assert head["wkb"].tobytes() == wkb[: off[2]].tobytes()
    ctx.node_coordinates = None
    with pytest.raises(TypeError):
        ways._build_chunk(ctx, np.arange(2))
    columns, json = ways.tag_lists(store, tags_as_columns, True, keep_other_tags)
    for name, array in zip(ctx.out_names, chunk["tag_arrays"]):
        assert array.to_pylist() == columns[name][rows].tolist(), name
    if keep_other_tags:
        assert chunk["json_array"].to_pylist() == json[rows].tolist()
    else:
        assert chunk["json_array"] is None


@pytest.mark.parametrize(
    "keep_metadata, keep_other_tags, keep_node_info, bounding_box",
    [
        (True, True, False, None),
        (False, True, True, None),
        (True, False, False, [0.5, 0.5, 3.0, 3.0]),
    ],
)
def test_engine_way_chunk_matches_python_path(
    keep_metadata, keep_other_tags, keep_node_info, bounding_box
):
    # A chunk of a store's ways built without the GIL holds the table the Python path
    # builds for the same ways' tag dicts, also when the chunks are built on threads; ways
    # without a geometry, or none in the box, give no chunk.
    import pyarrow as pa
    from geopandas.io.arrow import _geopandas_to_arrow
    from pyrosm.engine import geoparquet, pool, ways
    from pyrosm.engine.assemble import _assemble_chunk
    from pyrosm.engine.collect import _result_tag_columns

    nodes = _crafted_nodes()
    store = _way_store(_CRAFTED_WAYS)
    columns = ["building", "highway", "area", "name", "id_tag", "barrier", "source"]
    tags_as_columns = columns + ["visible", "roof"] if keep_other_tags else columns[:3]
    way_columns = _result_tag_columns(
        store, tags_as_columns, keep_metadata, keep_other_tags
    )
    gdf = _assemble_chunk(
        nodes,
        store.way_columns(None if keep_other_tags else set(tags_as_columns)),
        None,
        None,
        tags_as_columns,
        keep_metadata,
        bounding_box=bounding_box,
        keep_other_tags=keep_other_tags,
        keep_node_info=keep_node_info,
        way_columns=way_columns,
    )
    ref = _geopandas_to_arrow(gdf, index=False, geometry_encoding="WKB")
    ctx = ways.ChunkContext(
        store,
        tags_as_columns,
        keep_metadata,
        keep_other_tags,
        way_columns,
        nodes,
        keep_node_info,
        bounding_box,
    )
    mine = ways._chunk_parts(ctx, 0, len(store))[0]
    assert len(mine) < len(store)
    assert geoparquet._align_table(mine, ref.schema).equals(ref)
    token = pool._READ_THREADS.set(2)
    try:
        parts = list(ways.chunk_tables(ctx, range(0, len(store), 4), 4))
    finally:
        pool._READ_THREADS.reset(token)
    assert pa.concat_tables([t for t in parts if t is not None]).equals(mine)
    no_geometry = [i for i, way in enumerate(_CRAFTED_WAYS) if way[0] in (10, 11)]
    assert ways._chunk_parts(ctx, no_geometry[0], no_geometry[-1] + 1)[0] is None
    no_lines = ways.ChunkContext(
        _way_store([_CRAFTED_WAYS[i] for i in no_geometry]),
        tags_as_columns,
        keep_metadata,
        True,
        set(),
        nodes,
        network=True,
    )
    assert ways.network_frame(no_lines, 1) is None
    ctx.area = shapely.box(50.0, 50.0, 51.0, 51.0)
    assert ways._chunk_parts(ctx, 0, len(store))[0] is None


def test_engine_network_frame_matches_python_path(monkeypatch):
    # The edges a threaded read builds in chunks are the frame the Python path builds for
    # the same ways (a closed way is a line, a tag ``length`` holds the lengths, a tag only
    # a way without a geometry has is a column of nulls), written with the schema and
    # metadata geopandas gives it (the null type pandas 2 infers for such a column too).
    import pyarrow as pa
    from geopandas.io.arrow import _geopandas_to_arrow
    from pyrosm.engine import ways
    from pyrosm.engine.collect import _result_tag_columns, _ways_arrays
    from pyrosm.frames import prepare_geodataframe

    nodes = _crafted_nodes()
    store = _way_store(
        [
            (1, [1, 2, 3], [(b"highway", b"residential"), (b"length", b"5")]),
            (2, [3, 6], [(b"highway", b"footway")]),
            (3, [1, 2, 3, 4, 1], [(b"highway", b"pedestrian"), (b"area", b"yes")]),
            (4, [1, 99], [(b"highway", b"service"), (b"name", b"lost")]),
        ]
    )
    tags_as_columns = ["highway", "length", "area", "name"]
    edges, _ = prepare_geodataframe(
        None,
        nodes,
        _ways_arrays(store.way_columns(), tags_as_columns, True),
        None,
        None,
        tags_as_columns,
        None,
        parse_network=True,
        keep_metadata=True,
    )
    ctx = ways.ChunkContext(
        store,
        tags_as_columns,
        True,
        True,
        _result_tag_columns(store, tags_as_columns, True),
        nodes,
        network=True,
    )
    frame = ways.network_frame(ctx, 2)
    assert frame.to_arrow().equals(
        _geopandas_to_arrow(edges.drop(columns=["nodes"])), check_metadata=True
    )
    schema = frame.schema()
    nulls = schema.set(schema.get_field_index("name"), pa.field("name", pa.null()))
    monkeypatch.setattr(frame, "schema", lambda: nulls)
    assert frame.to_arrow().column("name").type == pa.null()


@pytest.mark.parametrize("fallback", [False, True])
def test_engine_batched_relations_match_per_relation(helsinki_pbf, tmp_path, fallback):
    # The relation records built by vectorized calls on two threads equal those built
    # relation by relation, also when a vectorized overlay or ring build raises and is
    # repeated relation by relation.
    import types

    import shapely as real_shapely
    from shapely import GEOSException
    from pyrosm import relations
    from pyrosm.engine import pool, relation_geometry
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
    from pyrosm.engine.collect import _collect_layer
    from pyrosm.engine.pool import _decode_all

    blobs, _ = _data_blob_spans(_index_blobs(helsinki_pbf))
    shards, _ = _decode_all(helsinki_pbf, blobs, 1, str(tmp_path), None, False)
    collected = _collect_layer(shards, [], True, (None, None, "keep"))
    _, _, rels, relation_ways, nodes = collected
    ref = relations.relation_records(rels, relation_ways, nodes, True)
    if fallback:

        def raising(function):
            def call(*args, **kwargs):
                if isinstance(args[0], np.ndarray) and len(args[0]) > 1:
                    raise GEOSException("vectorized")
                return function(*args, **kwargs)

            return call

        proxy = types.SimpleNamespace(**vars(real_shapely))
        proxy.symmetric_difference = raising(real_shapely.symmetric_difference)
        proxy.linearrings = raising(real_shapely.linearrings)
        relation_geometry.shapely = proxy
    token = pool._READ_THREADS.set(2)
    try:
        mine = relation_geometry.relation_records(rels, relation_ways, nodes, True)
    finally:
        pool._READ_THREADS.reset(token)
        relation_geometry.shapely = real_shapely
    assert len(ref) > 100 and len(mine) == len(ref)
    for a, b in zip(mine, ref):
        assert list(a) == list(b)
        for key in b:
            if key == "geometry":
                assert shapely.to_wkb(a[key]) == shapely.to_wkb(b[key]), b["id"]
            else:
                assert type(a[key]) is type(b[key]) and a[key] == b[key], (b["id"], key)


def _crafted_relations(members):
    """Relations of ``members`` (lists of way ids), tagged as multipolygons."""
    relations = {
        "id": np.arange(1, len(members) + 1, dtype=np.int64),
        "members": np.empty(len(members), dtype=object),
        "tags": np.empty(len(members), dtype=object),
        "version": np.ones(len(members), dtype=np.int64),
        "timestamp": np.ones(len(members), dtype=np.int64),
        "changeset": np.ones(len(members), dtype=np.int64),
        "visible": np.ones(len(members), dtype=bool),
    }
    for i, ids in enumerate(members):
        relations["members"][i] = {
            "member_id": np.array(ids, dtype=np.int64),
            "member_type": np.array([b"way"] * len(ids), dtype=object),
            "member_role": np.array([b"outer"] * len(ids), dtype=object),
        }
        relations["tags"][i] = {"type": "multipolygon", "natural": "water"}
    return relations


def test_engine_batched_relations_edge_cases(monkeypatch):
    # Relations without members, without member lines, without closed rings and whose
    # overlay raises give the records of the per-relation code; merged member lines
    # without a LineString part give no rings.
    from shapely import GEOSException
    from pyrosm import geometry, relations
    from pyrosm.engine import relation_geometry

    nodes = _crafted_nodes()
    way_nodes = {
        10: [1, 2, 3, 4, 1],
        11: [1, 99],
        12: [1, 2, 3],
        13: [1, 2, 3, 1],
        14: [1, 2, 6, 4, 1],
        15: [],
        16: [1, 3, 2, 4, 1],
    }
    ways = {"id": np.array(sorted(way_nodes), dtype=np.int64)}
    ways["nodes"] = np.empty(len(way_nodes), dtype=object)
    ways["nodes"][:] = [np.array(way_nodes[w], dtype=np.int64) for w in ways["id"]]
    rels = _crafted_relations([[], [11], [12], [10, 13], [14, 13], [10]])
    real = shapely.symmetric_difference
    raising = shapely.polygons([_NODES[n] for n in way_nodes[14]])

    def failing(a, b, **kwargs):
        # An overlay with way 14's ring raises.
        if isinstance(a, np.ndarray):
            return np.array([failing(x, y) for x, y in zip(a, b)], dtype=object)
        if shapely.equals(a, raising) or shapely.equals(b, raising):
            raise GEOSException("overlay")
        return real(a, b, **kwargs)

    monkeypatch.setattr(geometry, "symmetric_difference", failing, raising=False)
    monkeypatch.setattr(relation_geometry.shapely, "symmetric_difference", failing)
    ref = relations.relation_records(rels, ways, nodes, True)
    mine = relation_geometry.relation_records(rels, ways, nodes, True)
    # Only the relations with a closed ring are built, and the fifth's overlay raised.
    assert [int(r["id"]) for r in ref] == [4, 6]
    assert [r["id"] for r in mine] == [r["id"] for r in ref]
    for a, b in zip(mine, ref):
        assert shapely.to_wkb(a["geometry"]) == shapely.to_wkb(b["geometry"])
    # Each kind on its own: no members, no member nodes, no member lines, no closed ring,
    # a self-crossing ring.
    for members in ([[]], [[15]], [[11]], [[12]], [[16]]):
        one = _crafted_relations(members)
        ref = relations.relation_records(one, ways, nodes, True)
        mine = relation_geometry.relation_records(one, ways, nodes, True)
        assert [shapely.to_wkb(r["geometry"]) for r in mine] == [
            shapely.to_wkb(r["geometry"]) for r in ref
        ]
    owner, rings = relation_geometry._rings(
        np.array([0, 1]), np.array([shapely.GeometryCollection(), shapely.Point(0, 0)])
    )
    assert len(owner) == len(rings) == 0
    # An area GEOS reports invalid (here: the overlay of ways 10 and 13) is fixed as the
    # per-relation code fixes it.
    target = shapely.symmetric_difference(
        *(shapely.Polygon([_NODES[n] for n in way_nodes[w]]) for w in (10, 13))
    )
    real_valid = shapely.is_valid

    def valid(g, **kwargs):
        if isinstance(g, np.ndarray):
            return np.array([valid(x) for x in g], dtype=bool)
        return bool(real_valid(g)) and not shapely.equals(g, target)

    monkeypatch.setattr(shapely, "is_valid", valid)
    monkeypatch.setattr(geometry, "is_valid", valid, raising=False)
    monkeypatch.setattr(relations, "is_valid", valid, raising=False)
    one = _crafted_relations([[10, 13]])
    ref = relations.relation_records(one, ways, nodes, True)
    mine = relation_geometry.relation_records(one, ways, nodes, True)
    assert len(ref) == 1
    assert [shapely.to_wkb(r["geometry"]) for r in mine] == [
        shapely.to_wkb(r["geometry"]) for r in ref
    ]


def test_engine_node_tag_columns_match_explode():
    # The node features' tag columns and JSON built from string ids are those
    # explode_node_tag_array builds from their tag dicts, special nodes included.
    from pyrosm.engine.ways import _node_tag_columns, python_columns
    from pyrosm.tagparser import explode_node_tag_array

    store = _way_store(_CRAFTED_WAYS)
    columns = ["building", "name", "highway", "area", "lon", "k3", "absent"]
    tags = [store.tags(w) for w in range(len(store))]
    mine = python_columns(_node_tag_columns(store, columns))
    assert mine == explode_node_tag_array(tags, columns)


def test_engine_node_table_left_to_python_path():
    # The node table is left to the Python path when a tag column took a non-tag column's
    # name or the tags are not an Arrow array (dicts, or a None for each node when no node
    # has other tags), and is None when no point is in the box.
    import pyarrow as pa

    from pyrosm.engine.ways import node_table

    nodes = {
        "id": np.array([1, 2], dtype=np.int64),
        "lon": np.array([24.9, 25.0]),
        "lat": np.array([60.1, 60.2]),
        "tags": pa.array(['{"a":"b"}', None], type=pa.large_string()),
        "visible": np.array([True, False]),
        "name": ["x", None],
    }
    table = node_table(nodes, None)
    assert table.column_names == list(nodes) + ["geometry", "osm_type"]
    for name in ("lon", "geometry", "osm_type"):
        assert node_table({**nodes, name: ["1", "2"]}, None) is False
    dicts = np.empty(2, dtype=object)
    dicts[:] = [{"a": "b"}, {}]
    for tags in (dicts, [None, None]):
        assert node_table({**nodes, "tags": tags}, None) is False
    assert node_table(nodes, [0.0, 0.0, 1.0, 1.0]) is None


def test_engine_binary_array_splits_past_32_bit_offsets(monkeypatch):
    # The geometries of a chunk go into several Arrow arrays when their bytes exceed what
    # 32-bit offsets reach.
    import pyarrow as pa
    from pyrosm.engine import ways

    values = [b"a" * 4, b"b" * 5, b"c" * 6, b"d" * 7]
    offsets = np.concatenate([[0], np.cumsum([len(v) for v in values])])
    data = np.frombuffer(b"".join(values), dtype=np.uint8)
    monkeypatch.setattr(ways, "_BINARY_LIMIT", 10)
    array = ways._binary_array(offsets, data)
    assert array.num_chunks == 3
    assert array.to_pylist() == values
    monkeypatch.setattr(ways, "_BINARY_LIMIT", 2**31 - 1)
    assert isinstance(ways._binary_array(offsets, data), pa.BinaryArray)


def test_way_columns_tag_columns_check_their_arrays():
    # tag_columns refuses arrays that do not fit together, and a tag that is not valid
    # UTF-8 in a row that is not special; bytes that are not valid UTF-8 but marked valid
    # (cut off at the end of the strings) fill the JSON value to the length it counts.
    from pyrosm import way_columns as wc

    store = _way_store(_CRAFTED_WAYS)
    valid, column = wc.string_columns(store.str_off, store.str_data, [b"building"])
    n = len(store)
    arrays = dict(
        rows=np.arange(n),
        pairs_off=store.pairs_off,
        pair_key=store.pair_key,
        pair_val=store.pair_val,
        str_off=store.str_off,
        str_data=store.str_data,
        valid=valid,
        column=column,
        out_column=np.array([0], dtype=np.int32),
        n_out=1,
        special=np.ones(n, dtype=np.uint8),
        version=store.version,
        timestamp=store.timestamp,
        visible=store.visible,
        json=True,
        json_version=True,
        json_timestamp=True,
        json_visible=True,
    )
    for change, match in (
        ({"valid": valid[:-1]}, "one entry per string"),
        ({"out_column": np.empty(0, dtype=np.int32)}, "position of out_column"),
        ({"n_out": 0}, "below n_out"),
        ({"special": np.ones(n - 1, dtype=np.uint8)}, "one entry per row"),
        ({"visible": store.visible[:-1]}, "one entry per way"),
        ({"special": np.zeros(n, dtype=np.uint8)}, "valid UTF-8"),
    ):
        with pytest.raises(ValueError, match=match):
            wc.tag_columns(**{**arrays, **change})
    cut = _way_store([(1, [1, 2], [(b"a", b"\xe2\x82"), (b"b", b"\xf0\x9f")])])
    built = wc.tag_columns(
        **{
            **arrays,
            "rows": [0],
            "pairs_off": cut.pairs_off,
            "pair_key": cut.pair_key,
            "pair_val": cut.pair_val,
            "str_off": cut.str_off,
            "str_data": cut.str_data,
            "valid": np.ones(4, dtype=np.uint8),
            "column": np.full(4, -1, dtype=np.int32),
            "special": np.zeros(1, dtype=np.uint8),
            "version": cut.version,
            "timestamp": cut.timestamp,
            "visible": cut.visible,
            "json_version": False,
            "json_timestamp": False,
            "json_visible": False,
        }
    )
    offsets, data, _ = built["json"]
    assert offsets.tolist() == [0, 33]
    assert re.fullmatch(rb'{"a":"\\u[0-9A-F]{4}","b":"(\\u[0-9A-F]{4}){2}"}', data)


def test_route_tags_drops_the_tags_of_no_column_without_other_tags():
    # Without other tags the process decode's tag dicts hold the requested keys and the
    # geometry tag keys; those that are not output columns are dropped, so the columns and
    # JSON are those of the requested keys alone.
    from pyrosm.engine import collect
    from pyrosm.engine.readers import _GEOMETRY_TAG_KEYS

    store = _way_store(_CRAFTED_WAYS)
    requested = ["building", "source", "id_tag"]
    _, tag_cols = collect._way_column_keys(requested, True)
    metadata = (store.version, store.timestamp, store.visible)
    routed = [
        collect._route_tags(
            store.way_columns(keys)["tags"], *metadata, tag_cols, False, False, False, False
        )[:2]
        for keys in (set(requested), set(requested) | set(_GEOMETRY_TAG_KEYS))
    ]
    assert routed[1] == routed[0]


def test_engine_cut_off_file_raises_the_same_error(test_pbf, tmp_path, monkeypatch):
    # A file cut off in its last blob raises the same error with either backend.
    from pathlib import Path

    cut = tmp_path / "cut.osm.pbf"
    cut.write_bytes(Path(test_pbf).read_bytes()[:-100])
    errors = []
    for backend in ("processes", "threads"):
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
        with pytest.raises(Exception) as err:
            get_buildings(str(cut))
        errors.append((type(err.value), str(err.value)))
    assert errors[0] == errors[1]


def test_engine_threaded_decode_of_no_blobs(tmp_path):
    # Without data blobs the threaded decode reads no batch and writes no shard.
    from pyrosm.engine import blobs
    from pyrosm.engine.decode import _DecodeSettings

    assert list(blobs._read_batches([], 4)) == []
    settings = _DecodeSettings(str(tmp_path / "none.osm.pbf"), str(tmp_path))
    assert pool._decode_threads([], 2, settings, None) == []


def test_engine_decode_backend_must_be_known(test_pbf, monkeypatch, fresh_cache):
    monkeypatch.setenv("PYROSM_DECODE_BACKEND", "fork")
    with pytest.raises(ValueError, match="PYROSM_DECODE_BACKEND"):
        get_buildings(test_pbf)


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
        rel_meta=np.empty((0, 4), np.int64),
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
        _read_way_columns,
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
    assert _read_way_columns([empty]) is None
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

    s1 = pa.schema([pa.field("a", pa.int64()), pa.field("c", pa.null())])
    s2 = pa.schema([pa.field("b", pa.string())])
    unified = _unify_schemas([s1, s2])  # neither carries GeoParquet 'geo' metadata
    # The fields in order of appearance; a column of Nones (arrow's null type) as a string.
    assert unified.names == ["a", "c", "b"]
    assert unified.field("c").type == pa.large_string()
    # A table missing column 'b' is widened with typed nulls rather than dropping columns,
    # and its column of Nones is cast to the schema's type.
    aligned = _align_table(pa.table({"c": pa.nulls(2), "a": [1, 2]}), unified)
    assert aligned.schema == unified
    assert aligned.column("b").null_count == 2


def test_engine_stream_parquet_chunk_table_branches(
    helsinki_pbf, tmp_path, monkeypatch
):
    # Drive the chunked writer through its relations-only, ways-only, empty-chunk and
    # failed-chunk branches by feeding it crafted variants of the real collected building
    # data.
    pytest.importorskip("pyarrow")
    import tempfile
    import shutil
    import pyarrow.parquet as pq
    from pyrosm.config import Conf
    from pyrosm.data_manager import parse_custom_filter
    from pyrosm.engine import geoparquet
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
    from pyrosm.engine.pool import _decode_all
    from pyrosm.engine.collect import _collect_layer
    from pyrosm.engine.ways import ElementStore

    data_filter, osm_keys = parse_custom_filter({"building": [True]})
    filter_spec = (osm_keys, data_filter, "keep")
    tac = Conf.tags.building
    data_blobs, _ = _data_blob_spans(_index_blobs(helsinki_pbf))
    shard_dir = tempfile.mkdtemp()
    try:
        shards, _ = _decode_all(
            helsinki_pbf, data_blobs, 1, shard_dir, [b"building"], False
        )
        node_features, kept, relations, relation_ways, nc = _collect_layer(
            shards, tac, True, filter_spec, True, True
        )
        assert kept is not None and relations is not None  # helsinki has both

        def write(collected, output="o.parquet", chunk_size=250_000):
            monkeypatch.setattr(geoparquet, "_collect_layer", lambda *a, **k: collected)
            return geoparquet._stream_layer_to_parquet(
                shards,
                str(tmp_path / output),
                chunk_size,
                tac,
                True,
                filter_spec,
                True,
                True,
            )

        # relations only (no standalone ways): way loop skipped, relation chunk written.
        assert write((None, None, relations, relation_ways, nc)) is not None
        # ways only (no relations): relation branch skipped.
        whole = write((None, kept, None, None, nc))
        assert whole is not None
        # the way columns of a process decode, in two chunks, write the rows of one chunk.
        columns = kept.way_columns() if isinstance(kept, ElementStore) else kept
        half = len(columns["id"]) // 2 + 1
        in_two = write((None, columns, None, None, nc), "in_two.parquet", half)
        assert pq.ParquetFile(whole).metadata.num_row_groups == 1
        assert pq.ParquetFile(in_two).metadata.num_row_groups == 2
        _assert_full_parity(gpd.read_parquet(in_two), gpd.read_parquet(whole))
        # a way referencing an absent node assembles empty -> no parts -> None.
        absent_nodes = np.empty(1, dtype=object)
        absent_nodes[0] = np.array([10**18], np.int64)
        absent = {
            "id": np.array([-1], np.int64),
            "nodes": absent_nodes,
            "tags": np.array([{"building": "yes"}], dtype=object),
            "version": np.array([1], np.int64),
            "timestamp": np.array([1], np.int64),
            "visible": np.array([True]),
        }
        assert write((None, absent, None, None, nc)) is None
        # a relation whose member ways are all absent assembles empty -> chunk skipped.
        empty_ways = {"id": np.empty(0, np.int64), "nodes": np.empty(0, dtype=object)}
        assert write((None, None, relations, empty_ways, nc)) is None
        # a chunk that fails after the first was written leaves no file behind.
        align = geoparquet._align_table
        written = []

        def align_once(table, schema):
            if written:
                raise MemoryError
            written.append(table.num_rows)
            return align(table, schema)

        monkeypatch.setattr(geoparquet, "_align_table", align_once)
        with pytest.raises(MemoryError):
            write((None, kept, None, None, nc), "failed.parquet", chunk_size=10)
        assert written and not (tmp_path / "failed.parquet").exists()
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
    ref = OSM(fp, engine="in_memory").get_landuse()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_engine_natural_full_column_parity(fixture, request):
    # natural includes NODE point features, so this exercises the node path too.
    fp = request.getfixturevalue(fixture)
    mine = get_natural(fp)
    ref = OSM(fp, engine="in_memory").get_natural()
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
    ref = OSM(fp, engine="in_memory").get_pois()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


def test_engine_pois_value_filter_parity(helsinki_pbf):
    # A value-level custom_filter must refine to the exact values (not just key presence).
    flt = {"amenity": ["restaurant", "cafe", "bar"]}
    mine = get_pois(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf, engine="in_memory").get_pois(custom_filter=flt)
    assert mine is not None and len(mine) > 0
    assert mine["amenity"].isin(flt["amenity"]).all()
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_value_filter_parity(helsinki_pbf):
    flt = {"amenity": ["restaurant", "cafe", "pub"]}
    mine = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf, engine="in_memory").get_data_by_custom_criteria(custom_filter=flt)
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
    ref = OSM(helsinki_pbf, engine="in_memory").get_data_by_custom_criteria(custom_filter=flt, **flags)
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_keep_other_tags_false_minimal(helsinki_pbf):
    # keep_other_tags=False returns the same elements but only the requested column(s) -- the
    # JSON 'tags' column of leftover tags is dropped, and the rest is identical.
    flt = {"building": True}
    full = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, tags_as_columns=["building"], keep_other_tags=True
    )
    minimal = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, tags_as_columns=["building"], keep_other_tags=False
    )
    assert full is not None and minimal is not None
    assert "tags" in full.columns and "tags" not in minimal.columns
    # Same columns apart from the dropped 'tags' column, and the same elements in order.
    assert set(full.columns) - {"tags"} == set(minimal.columns)
    assert minimal["id"].tolist() == full["id"].tolist()
    assert minimal["building"].fillna("").tolist() == full["building"].fillna("").tolist()
    assert minimal.geometry.geom_type.tolist() == full.geometry.geom_type.tolist()


def test_engine_custom_criteria_keep_other_tags_false_value_filter(helsinki_pbf):
    # A value filter whose key is NOT a requested column still filters correctly, and the
    # filter key is a column too, so the rows show what they matched (no 'tags' column).
    flt = {"amenity": ["restaurant", "cafe", "pub"]}
    minimal = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, tags_as_columns=["name"], keep_other_tags=False
    )
    ref = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    assert minimal is not None and len(minimal) > 0
    assert set(minimal["id"]) == set(ref["id"])  # the value filter still selects the same rows
    assert "tags" not in minimal.columns
    assert set(minimal["amenity"]) == {"restaurant", "cafe", "pub"}
    assert "name" in minimal.columns


def test_engine_custom_criteria_keep_other_tags_default_unchanged(helsinki_pbf):
    # The default (keep_other_tags=True) is identical to not passing it at all.
    flt = {"building": True}
    default = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    explicit = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, keep_other_tags=True
    )
    _assert_full_parity(default, explicit)


def test_osm_custom_criteria_keep_other_tags_in_memory_rejected(helsinki_pbf):
    # The in-memory reader caches all tags for reuse across layers, so it cannot honour the
    # minimal-tags mode and rejects it.
    osm = OSM(helsinki_pbf, engine="in_memory")
    with pytest.raises(ValueError, match="only supported by the out-of-core engine"):
        osm.get_data_by_custom_criteria({"building": True}, keep_other_tags=False)
    # The default value leaves the in-memory path working.
    assert (
        osm.get_data_by_custom_criteria({"building": True}, keep_other_tags=True) is not None
    )


def test_engine_custom_criteria_keep_other_tags_cache_isolation(helsinki_pbf, fresh_cache):
    # keep_other_tags is part of the result-cache key, so a minimal read does not serve (or get
    # served by) a full-tags cached result.
    flt = {"building": True}
    full = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, tags_as_columns=["building"], keep_other_tags=True
    )
    minimal = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, tags_as_columns=["building"], keep_other_tags=False
    )
    assert "tags" in full.columns
    assert "tags" not in minimal.columns


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


def test_osm_list_and_clear_cache_per_file_and_all(test_pbf, helsinki_pbf, fresh_cache):
    # OSM.list_cache reflects what each out-of-core read cached; OSM.clear_cache(filepath)
    # removes only that source file's entries and OSM.clear_cache() removes everything.
    assert OSM.list_cache() == []
    OSM(test_pbf, engine="out_of_core").get_buildings()
    OSM(helsinki_pbf, engine="out_of_core").get_buildings()

    assert len(OSM.list_cache()) == 2
    assert len(OSM.list_cache(test_pbf)) == 1
    assert OSM.list_cache(test_pbf)[0].endswith(".parquet")

    assert OSM.clear_cache(test_pbf) == 1
    assert len(OSM.list_cache()) == 1  # helsinki's entry survives

    assert OSM.clear_cache() == 1
    assert OSM.list_cache() == []
    assert OSM.clear_cache() == 0


def test_osm_clear_cache_forces_redecode(helsinki_pbf, monkeypatch, fresh_cache):
    # After clearing, an identical read decodes the PBF again instead of reusing the cache.
    decodes = _count_decodes(monkeypatch)
    OSM(helsinki_pbf, engine="out_of_core").get_buildings()
    OSM(helsinki_pbf, engine="out_of_core").get_buildings()
    assert sum(decodes) == 1
    OSM.clear_cache(helsinki_pbf)
    OSM(helsinki_pbf, engine="out_of_core").get_buildings()
    assert sum(decodes) == 2


def test_osm_list_and_clear_downloads_specific_and_all(tmp_path, monkeypatch):
    # OSM.list_downloads lists downloaded *.pbf; OSM.clear_downloads removes them (a filepath
    # targets one, None removes all). The out-of-core cache subdirectory is left untouched.
    import pyrosm.utils.download as dl

    ddir = tmp_path / "pyrosm"
    ddir.mkdir()
    (ddir / "cache").mkdir()
    monkeypatch.setattr(dl, "download_dir", lambda: ddir)
    names = ("finland-latest.osm.pbf", "estonia-latest.osm.pbf")
    for name in names:
        (ddir / name).write_bytes(b"x")

    assert OSM.list_downloads() == sorted(str(ddir / n) for n in names)

    assert OSM.clear_downloads("finland-latest.osm.pbf") == 1
    assert OSM.list_downloads() == [str(ddir / "estonia-latest.osm.pbf")]

    assert OSM.clear_downloads() == 1
    assert OSM.list_downloads() == []
    assert (ddir / "cache").is_dir()
    assert OSM.clear_downloads() == 0


def test_clear_cache_skips_non_file_entries(fresh_cache):
    # clear() ignores non-file entries (e.g. a stray subdirectory) in the cache directory.
    cdir = cache.cache_dir()
    (cdir / "subdir").mkdir()
    (cdir / "result_x_y.parquet").write_bytes(b"")
    assert OSM.clear_cache() == 1  # only the file is counted and removed
    assert (cdir / "subdir").is_dir()


def test_list_and_clear_downloads_missing_dir(tmp_path, monkeypatch):
    # With no download directory, listing returns [] and clearing returns 0.
    import pyrosm.utils.download as dl

    monkeypatch.setattr(dl, "download_dir", lambda: tmp_path / "absent")
    assert OSM.list_downloads() == []
    assert OSM.clear_downloads() == 0


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
    assert list(cache.cache_dir().glob("*.empty"))
    decodes = _count_decodes(monkeypatch)
    assert get_data_by_custom_criteria(test_pbf, custom_filter=flt) is None
    assert sum(decodes) == 0


def test_engine_cache_pyarrow_absent_falls_back(helsinki_pbf, monkeypatch, fresh_cache):
    # With pyarrow unavailable the engine returns the in-memory frame and writes no cache file.
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    mine = get_buildings(helsinki_pbf)
    _assert_full_parity(mine, OSM(helsinki_pbf, engine="in_memory").get_buildings())
    assert list(cache.cache_dir().glob("*.parquet")) == []


def test_cache_dir_builds_and_creates_tempdir_path(monkeypatch, tmp_path):
    # cache_dir() roots the result cache at <tempdir>/pyrosm/cache and creates it on demand. The
    # fixtures stub the module attribute, so exercise the real implementation captured at import.
    monkeypatch.setattr(cache.tempfile, "gettempdir", lambda: str(tmp_path))
    result = _real_cache_dir()
    assert result == tmp_path / "pyrosm" / "cache"
    assert result.is_dir()


def test_engine_boundaries_parity(helsinki_region_pbf):
    # The Helsinki region extract has administrative boundaries (relations + ways).
    mine = get_boundaries(helsinki_region_pbf)
    ref = OSM(helsinki_region_pbf, engine="in_memory").get_boundaries()
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
    ref = OSM(helsinki_region_pbf, engine="in_memory").get_boundaries(name="Vantaa")
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
    ref = OSM(helsinki_pbf, engine="in_memory").get_data_by_custom_criteria(**kwargs)
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_non_conf_key_parity(helsinki_pbf):
    # A filter key without a dedicated Conf.tags column set becomes its own column.
    flt = {"source": True}
    mine = get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf, engine="in_memory").get_data_by_custom_criteria(custom_filter=flt)
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
    ref = OSM(helsinki_pbf, keep_metadata=False, engine="in_memory").get_natural()
    _assert_full_parity(mine, ref)


def test_engine_boundaries_custom_filter_adds_boundary_key(helsinki_region_pbf):
    # A custom_filter without 'boundary' gets it added, matching OSM().get_boundaries().
    flt = {"admin_level": True}
    mine = get_boundaries(helsinki_region_pbf, custom_filter=dict(flt))
    ref = OSM(helsinki_region_pbf, engine="in_memory").get_boundaries(custom_filter=dict(flt))
    if ref is None or len(ref) == 0:
        assert mine is None or len(mine) == 0
        return
    _assert_full_parity(mine, ref)


def test_engine_boundaries_name_without_name_column_is_none(helsinki_pbf, monkeypatch):
    # name= on a result that has no 'name' column finds nothing, as the in-memory
    # reader does.
    from shapely.geometry import Point
    from pyrosm.engine import readers

    fake = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
    monkeypatch.setattr(readers, "_get_layer", lambda *a, **k: fake)
    assert readers.get_boundaries(helsinki_pbf, name="x") is None


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
    ref = OSM(helsinki_pbf, engine="in_memory").get_network(network_type=network_type)
    assert ref is not None and len(ref) > 0
    _assert_full_parity(mine, ref)


def test_engine_network_custom_filter_parity(helsinki_pbf):
    # The 'crossing' key (not among the default highway columns) is added to the column
    # allow-list; no kept way carries it here, so -- matching the in-memory reader -- it
    # yields no column.
    flt = {"highway": ["footway", "path", "pedestrian"], "crossing": True}
    mine = get_network(helsinki_pbf, custom_filter=flt, filter_type="keep")
    ref = OSM(helsinki_pbf, engine="in_memory").get_network(custom_filter=flt, filter_type="keep")
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
    ref_nodes, ref_edges = OSM(helsinki_pbf, engine="in_memory").get_network(
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
    ref_nodes, ref_edges = OSM(helsinki_pbf, keep_metadata=False, engine="in_memory").get_network(
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
    assert len(list(cache.cache_dir().glob("*.parquet"))) == 2
    _assert_full_parity(e1, e2)
    _assert_full_parity(n1.assign(osm_type="node"), n2.assign(osm_type="node"))


@pytest.mark.parametrize("backend", ["threads", "processes"])
def test_engine_network_output_path_writes_edges(
    helsinki_pbf, tmp_path, monkeypatch, backend
):
    # output= writes the edges to that GeoParquet and returns the path; the file matches the
    # in-memory network edges, from the Arrow tables of a threaded decode and from the
    # frame of a process decode.
    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    out = tmp_path / "network.parquet"
    ret = get_network(helsinki_pbf, network_type="driving", output=out)
    assert ret == out and out.exists()
    written = cache.read_result(out)
    ref = OSM(helsinki_pbf, engine="in_memory").get_network(network_type="driving")
    _assert_full_parity(written, ref)


def test_engine_network_output_dir_with_nodes(helsinki_pbf, tmp_path):
    # output= + nodes=True writes edges.parquet + nodes.parquet into the directory and returns
    # it; the two files reload equal to the in-memory (nodes, edges) tuple.
    from pyrosm.engine import readers

    out = tmp_path / "net_dir"
    ret = get_network(helsinki_pbf, network_type="driving", nodes=True, output=out)
    assert ret == out
    assert (out / "edges.parquet").exists()
    assert (out / "nodes.parquet").exists()
    ref_nodes, ref_edges = OSM(helsinki_pbf, engine="in_memory").get_network(
        network_type="driving", nodes=True
    )
    _assert_full_parity(cache.read_result(out / "edges.parquet"), ref_edges)
    written_nodes = readers._read_nodes_parquet(out / "nodes.parquet")
    a = written_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    b = ref_nodes.sort_values("id").reset_index(drop=True).assign(osm_type="node")
    _assert_full_parity(a, b)


def test_engine_network_output_dir_empty_returns_none(test_pbf, tmp_path):
    # An empty nodes=True read with output= writes nothing and returns None.
    out = tmp_path / "empty_dir"
    flt = {"highway": ["definitely_not_a_real_highway_xyz"]}
    ret = get_network(
        test_pbf, custom_filter=flt, filter_type="keep", nodes=True, output=out
    )
    assert ret is None
    assert not out.exists()


def test_engine_network_pyarrow_absent_falls_back(
    helsinki_pbf, monkeypatch, fresh_cache
):
    # With pyarrow unavailable the network read returns the in-memory edges and writes no cache.
    from pyrosm.utils import _compat

    monkeypatch.setattr(_compat, "HAS_PYARROW", False)
    mine = get_network(helsinki_pbf, network_type="driving")
    _assert_full_parity(mine, OSM(helsinki_pbf, engine="in_memory").get_network(network_type="driving"))
    assert list(cache.cache_dir().glob("*.parquet")) == []


def test_engine_network_empty_result_is_marked(test_pbf, monkeypatch, fresh_cache):
    # A network filter that matches no highway returns None, records an empty marker, and an
    # identical later read returns None without decoding the file again.
    flt = {"highway": ["definitely_not_a_real_highway_xyz"]}
    assert get_network(test_pbf, custom_filter=flt, filter_type="keep") is None
    assert list(cache.cache_dir().glob("*.empty"))
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
    assert list(cache.cache_dir().glob("*.empty"))
    decodes = _count_decodes(monkeypatch)
    again = get_network(test_pbf, custom_filter=flt, filter_type="keep", nodes=True)
    assert again == (None, None)
    assert sum(decodes) == 0


def test_engine_node_records_by_id_edge_cases():
    from pyrosm.engine.decode import _node_records_by_id
    from pyrosm.int64_hash import Int64Set
    from pyrosm.way_store import NodeTagsBuilder

    header = {"granularity": 100, "lat_offset": 0, "lon_offset": 0}
    st = [b"", b"amenity", b"cafe"]
    tags = NodeTagsBuilder()
    # A block with no dense nodes (a way/relation block) yields nothing.
    assert _node_records_by_id(st, header, None, Int64Set([1]), True, tags) is None
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
    assert _node_records_by_id(st, header, nodes, Int64Set([999]), True, tags) is None
    rec = _node_records_by_id(st, header, nodes, Int64Set([1, 2]), True, tags)
    assert list(tags.build()[np.arange(2)]) == [None, {"amenity": "cafe"}]
    assert rec["lat"].tolist() == [600000000 * 100 / 1e9, 600000010 * 100 / 1e9]
    assert rec["version"].tolist() == [3, 3] and rec["visible"].dtype == bool
    # keep_metadata=False drops version/timestamp/changeset but keeps visible.
    lean = _node_records_by_id(st, header, nodes, Int64Set([1]), False, NodeTagsBuilder())
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
    bbox = _central_bbox(OSM(helsinki_pbf, engine="in_memory").get_buildings())
    mine = get_buildings(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    )
    ref = OSM(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations,
        engine="in_memory",
    ).get_buildings()
    assert ref is not None and (ref["osm_type"] == "relation").any()
    _assert_full_parity(mine, ref)


def test_engine_pois_bounding_box_parity(helsinki_pbf):
    # bbox over a layer with node features: nodes + ways restricted to the box.
    bbox = _central_bbox(OSM(helsinki_pbf, engine="in_memory").get_pois())
    mine = get_pois(helsinki_pbf, bounding_box=bbox)
    ref = OSM(helsinki_pbf, bounding_box=bbox, engine="in_memory").get_pois()
    assert ref is not None and (ref["osm_type"] == "node").any()
    _assert_full_parity(mine, ref)


def test_engine_network_bounding_box_parity(helsinki_pbf):
    # A bounding box keeps ways with >=1 node inside it (kept whole), then the final spatial
    # filter clips to the box -- matching the in-memory reader.
    full = OSM(helsinki_pbf, engine="in_memory").get_network(network_type="driving")
    bbox = _central_bbox(full, frac=0.5)
    ref = OSM(helsinki_pbf, bounding_box=bbox, engine="in_memory").get_network(network_type="driving")
    mine = get_network(helsinki_pbf, network_type="driving", bounding_box=bbox)
    assert ref is not None and 0 < len(ref) < len(full)
    _assert_full_parity(mine, ref)


def test_engine_custom_criteria_bounding_box_parity(helsinki_pbf):
    flt = {"building": True}
    bbox = _central_bbox(OSM(helsinki_pbf, engine="in_memory").get_buildings())
    mine = get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, filter_type="keep", bounding_box=bbox
    )
    ref = OSM(helsinki_pbf, bounding_box=bbox, engine="in_memory").get_data_by_custom_criteria(
        custom_filter=flt, filter_type="keep"
    )
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


def test_engine_bounding_box_polygon_parity(helsinki_pbf):
    # A shapely-polygon bounding box is validated/normalised through the same path as the
    # in-memory reader and yields the same result as the equivalent coordinate list.
    from shapely.geometry import box

    poly = box(*_central_bbox(OSM(helsinki_pbf, engine="in_memory").get_buildings()))
    mine = get_buildings(helsinki_pbf, bounding_box=poly)
    ref = OSM(helsinki_pbf, bounding_box=poly, engine="in_memory").get_buildings()
    assert ref is not None and len(ref) > 0
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("bad", [[1, 2, 3], [25, 61, 24, 60], "not-a-bbox"])
def test_engine_bounding_box_validation_matches_in_memory(helsinki_pbf, bad):
    # Malformed boxes (wrong length, inverted coordinates, wrong type) raise the same
    # ValueError the in-memory reader raises.
    with pytest.raises(ValueError):
        OSM(helsinki_pbf, bounding_box=bad, engine="in_memory")
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

    ref = getattr(OSM(fp, engine="in_memory"), method)(**copy.deepcopy(kwargs))
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
    bbox = _central_bbox(OSM(helsinki_pbf, engine="in_memory").get_buildings())
    ref = OSM(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations,
        engine="in_memory",
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


def test_osm_workers_validation(test_pbf):
    # The constructor-level worker count must be a positive integer, "auto", or None.
    for bad in (0, -1, 1.5, True, "2", "bogus", np.int64(0)):
        with pytest.raises(ValueError, match="workers"):
            OSM(test_pbf, workers=bad, engine="in_memory")
    assert OSM(test_pbf, engine="in_memory").workers is None
    assert OSM(test_pbf, workers=2, engine="in_memory").workers == 2
    assert type(OSM(test_pbf, workers=np.int64(2), engine="in_memory").workers) is int
    assert OSM(test_pbf, workers="auto", engine="in_memory").workers == "auto"
    assert OSM(test_pbf, workers="AUTO", engine="in_memory").workers == "auto"  # normalized


@pytest.mark.parametrize(
    "workers, passed", [(None, "auto"), (1, 1), (2, 2), ("auto", "auto")]
)
def test_osm_workers_reach_the_reader_without_a_warning(
    test_pbf, monkeypatch, workers, passed
):
    """The engine reader gets "auto" when no count is given (one with the processes
    backend) and a given count or "auto" unchanged, and no read warns about the CPU
    cores. Worker processes run in this process."""
    import warnings as _warnings
    import pyrosm.engine.readers as readers_mod

    if workers is None and pool._decode_backend() == "processes":
        passed = 1
    monkeypatch.setattr(pool, "ProcessPoolExecutor", _fake_executor())
    monkeypatch.setattr(pool, "_available_cores", lambda: 4)
    captured = []
    real = readers_mod._decode_and_run

    def spy(filepath, osm_key_bytes, include_nodes, workers, run, **kwargs):
        captured.append(workers)
        return real(filepath, osm_key_bytes, include_nodes, workers, run, **kwargs)

    monkeypatch.setattr(readers_mod, "_decode_and_run", spy)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        OSM(test_pbf, workers=workers).get_buildings()
    assert captured == [passed]
    assert not [w for w in caught if "CPU core" in str(w.message)]


@pytest.mark.parametrize(
    "size, cores, blobs, backend, expected",
    [
        ("small", 10, 10_000, "threads", 1),
        ("medium", 10, 10_000, "threads", 4),
        ("medium", 4, 10_000, "threads", 2),
        ("big", 10, 10_000, "threads", 8),
        ("big", 16, 10_000, "processes", 8),
        ("big", 16, 10_000, "threads", 14),
        ("big", 4, 10_000, "threads", 2),
        ("big", 2, 10_000, "threads", 1),
        ("big", 10, 3, "threads", 3),
    ],
)
def test_auto_workers(tmp_path, monkeypatch, size, cores, blobs, backend, expected):
    # "auto" reads small files on a single core; above the size threshold it uses the CPU
    # cores less two, at most 4 for medium files, at least 1, at most 8 worker processes and
    # at most the data-blob count.
    import os

    monkeypatch.setattr(pool, "_available_cores", lambda: cores)
    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    fp = tmp_path / "file.osm.pbf"
    fp.touch()
    sizes = {
        "small": pool._PARALLEL_MIN_FILE_BYTES - 1,
        "medium": pool._PARALLEL_MIN_FILE_BYTES,
        "big": pool._AUTO_ALL_CORES_BYTES,
    }
    os.truncate(fp, sizes[size])
    assert pool._auto_workers(str(fp), blobs) == expected


@pytest.mark.parametrize(
    "spans, workers, per_worker, expected",
    [
        ([5, 5, 5], 1, 8, [[5, 5, 5]]),
        ([1] * 32, 2, 8, [[1, 1]] * 16),
        # Four shares of 6.5 bytes: the 10-byte blobs start in the first and third share.
        ([1, 1, 1, 1, 10, 10, 1, 1], 2, 2, [[1, 1, 1, 1, 10], [10], [1, 1]]),
        ([], 2, 8, []),
    ],
    ids=["one_worker", "equal_blobs", "unequal_blobs", "no_blobs"],
)
def test_decode_tasks_split_the_file_by_bytes(
    monkeypatch, spans, workers, per_worker, expected
):
    # The decode tasks hold consecutive blobs in file order with unique ids, about
    # _TASKS_PER_WORKER per worker of similar bytes (one task for a single worker); a blob goes
    # to the task whose share of the bytes holds its start.
    monkeypatch.setattr(pool, "_TASKS_PER_WORKER", per_worker)
    blobs = [(i, span, span) for i, span in enumerate(spans)]
    tasks = pool._decode_tasks(blobs, workers)
    assert [task_id for task_id, _ in tasks] == list(range(len(tasks)))
    assert [blob for _, task in tasks for blob in task] == blobs
    assert [[span for _, _, span in task] for _, task in tasks] == expected


def test_cap_workers_reduces_above_cpu_count(monkeypatch):
    # More workers than CPU cores is reduced to the core count with a warning; counts at or
    # below the core count pass through unchanged and silently.
    import warnings as _warnings

    monkeypatch.setattr(pool, "_available_cores", lambda: 4)
    with pytest.warns(UserWarning, match="exceeds the 4 CPU cores"):
        assert pool._cap_workers(10) == 4
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert pool._cap_workers(4) == 4
        assert pool._cap_workers(1) == 1


def test_osm_out_of_core_excess_workers_capped(helsinki_pbf, monkeypatch, fresh_cache):
    # An explicit worker count above the CPU-core count is reduced to the core count, with a
    # warning; the read still matches the in-memory reader.
    monkeypatch.setattr(pool, "_available_cores", lambda: 2)
    monkeypatch.setattr(pool, "ProcessPoolExecutor", _fake_executor())
    captured = []
    real = pool._decode_all

    def spy(filepath, blobs, workers, *args, **kwargs):
        captured.append(workers)
        return real(filepath, blobs, workers, *args, **kwargs)

    monkeypatch.setattr(pool, "_decode_all", spy)
    with pytest.warns(UserWarning, match="exceeds the 2 CPU cores"):
        mine = OSM(helsinki_pbf, engine="out_of_core", workers=50).get_buildings()
    assert captured == [2]
    _assert_full_parity(mine, OSM(helsinki_pbf, engine="in_memory").get_buildings())


def test_osm_out_of_core_network_nodes_parity(helsinki_pbf):
    # The public OSM API returns the (nodes, edges) tuple from the out-of-core engine, equal
    # to the in-memory reader's.
    mine_nodes, mine_edges = OSM(helsinki_pbf, engine="out_of_core").get_network(
        network_type="cycling", nodes=True
    )
    ref_nodes, ref_edges = OSM(helsinki_pbf, engine="in_memory").get_network(
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


def _history_blocks(path):
    """``decode_primitive_block``'s ``(string_table, header, nodes, ways, relations)`` of
    each block of the PBF ``path``, in file order."""
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs, _read_block
    from pyrosm.primitive_block_decoder import decode_primitive_block

    with open(path, "rb") as f:
        return [
            decode_primitive_block(_read_block(f, offset, size))
            for offset, size, _ in _data_blob_spans(_index_blobs(path))[0]
        ]


def _element_contents(elements):
    """Each decoded element's coordinates and tag ids (nodes), or tag ids and refs or
    members (ways, relations), in order."""
    n = len(elements["id"])
    if "keys_vals" in elements:
        kv, p, out = elements["keys_vals"], 0, []
        for i in range(n):
            pairs = []
            while p < len(kv) and kv[p] != 0:
                pairs.append((kv[p], kv[p + 1]))
                p += 2
            p += 1
            out.append((elements["lat"][i], elements["lon"][i], tuple(pairs)))
        return out
    t, out = elements["tags_off"], []
    for i in range(n):
        tags = tuple(zip(*(elements[k][t[i] : t[i + 1]] for k in ("keys", "vals"))))
        if "refs_off" in elements:
            r = elements["refs_off"]
            part = tuple(elements["refs"][r[i] : r[i + 1]])
        else:
            m = elements["members_off"]
            part = tuple(
                zip(*(elements[k][m[i] : m[i + 1]] for k in ("memids", "types", "roles")))
            )
        out.append((tags, part))
    return out


@pytest.mark.parametrize("date", ["2010-01-01", "2016-06-01", "2025-01-01"])
def test_versions_at_keep_each_elements_version_at_the_time(helsinki_history_pbf, date):
    """Each element of a history block is cut to its last version dated at or before the
    time, whole (coordinates, tags, refs, members); an element whose version then is a
    deletion, or that has none yet, is left out."""
    from pyrosm.engine import decode
    from pyrosm.utils import get_unix_time

    unix_time = get_unix_time(date, True)
    for _, _, *kinds in _history_blocks(helsinki_history_pbf):
        for elements in kinds:
            if elements is None:
                continue
            last = {}
            for i, (eid, ts) in enumerate(zip(elements["id"], elements["timestamp"])):
                if ts <= unix_time:
                    last[eid] = i
            rows = [i for i in last.values() if elements["visible"][i]]
            kept = decode._versions_at(elements, unix_time)
            if not rows:
                assert kept is None
                continue
            assert kept["id"].tolist() == elements["id"][rows].tolist()
            assert kept["version"].tolist() == elements["version"][rows].tolist()
            contents = _element_contents(elements)
            assert _element_contents(kept) == [contents[i] for i in rows]


def test_versions_at_date_the_versions_by_the_block_granularity():
    """A block's timestamps count its ``date_granularity`` milliseconds: with 500, the
    stored 20 and 40 are 10 s and 20 s, so at 15 s the first version is the one in time."""
    from pyrosm.engine import decode

    one = np.array([1, 1])
    ways = {
        "id": np.array([7, 7]),
        "keys": np.empty(0, np.int64),
        "vals": np.empty(0, np.int64),
        "tags_off": np.array([0, 0, 0]),
        "refs": np.array([1, 2, 1, 2, 3]),
        "refs_off": np.array([0, 2, 5]),
        "version": np.array([1, 2]),
        "timestamp": np.array([20, 40]),
        "changeset": one,
        "visible": one,
    }
    kept = decode._versions_at(ways, 15, date_granularity=500)
    assert kept["version"].tolist() == [1]
    assert kept["refs"].tolist() == [1, 2]


@pytest.mark.parametrize("threaded", [False, True], ids=["processes", "threads"])
def test_history_decode_spills_the_versions_at_the_time(
    helsinki_history_pbf, tmp_path, threaded
):
    """Both decode paths of a history read spill each block's elements at the read's
    time: the node coordinates, the matched ways and every way's refs."""
    from pyrosm.engine import decode
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
    from pyrosm.utils import get_unix_time

    unix_time = get_unix_time("2016-06-01", True)
    settings = decode._DecodeSettings(
        helsinki_history_pbf, str(tmp_path), [b"building"], unix_time=unix_time
    )
    blobs = _data_blob_spans(_index_blobs(helsinki_history_pbf))[0]
    worker = decode._decode_batch_threads if threaded else decode._decode_batch
    shards = [decode._open_shard(s) for s in worker(settings, (0, blobs))]
    nodes, ways = [], []
    for _, _, block_nodes, block_ways, _ in _history_blocks(helsinki_history_pbf):
        for elements, parts in ((block_nodes, nodes), (block_ways, ways)):
            kept = decode._versions_at(elements, unix_time)
            if kept is not None:
                parts.append(kept)

    def spilled(key):
        return np.concatenate([s[key] for s in shards]).tolist()

    assert spilled("node_id") == np.concatenate([n["id"] for n in nodes]).tolist()
    assert spilled("all_id") == np.concatenate([w["id"] for w in ways]).tolist()
    versions = {(i, v) for w in ways for i, v in zip(w["id"], w["version"])}
    matched = list(zip(spilled("way_id"), spilled("way_version")))
    assert matched and set(matched) <= versions


def _gather_every_shard_id(shard_paths, needed):
    """Reference for ``_gather_node_shards``: search every id of every shard in
    ``needed``."""
    pos_parts, lon_parts, lat_parts = [], [], []
    for path in shard_paths:
        z = np.load(path)
        nid = z["node_id"]
        if len(nid) == 0:
            continue
        pos = np.clip(np.searchsorted(needed, nid), 0, len(needed) - 1)
        hit = needed[pos] == nid
        pos_parts.append(pos[hit])
        lon_parts.append(z["node_lon"][hit])
        lat_parts.append(z["node_lat"][hit])
    return [np.concatenate(parts) for parts in (pos_parts, lon_parts, lat_parts)]


@pytest.mark.parametrize(
    "needed", [[3, 4, 5, 7, 10, 12, 50], [50]], ids=["some_found", "none_found"]
)
def test_gather_node_shards_matches_searching_every_id(tmp_path, needed):
    """The coordinate gather returns what searching every shard id returns, element for
    element: a sorted shard is searched within its id range, a shard that is not
    strictly increasing (unsorted, or with a repeated id) id by id."""
    from pyrosm.engine.collect import _gather_node_shards, _node_lookup

    paths = []
    for name, ids in [
        ("sorted", [1, 3, 5, 7]),
        ("outside_the_needed_range", [100, 200]),
        ("unsorted", [12, 10, 11]),
        ("repeated_id", [5, 5, 6]),
        ("empty", []),
    ]:
        ids = np.array(ids, dtype=np.int64)
        # Distinct coordinates per row, so rows of a repeated id are told apart.
        rows = np.arange(len(ids)) + 100 * len(paths)
        paths.append(tmp_path / ("%s.npz" % name))
        np.savez(paths[-1], node_id=ids, node_lon=rows * 1.0, node_lat=rows * -1.0)

    needed = np.array(needed, dtype=np.int64)
    result = _gather_node_shards(paths, needed)
    expected = _gather_every_shard_id(paths, needed)
    for got, want in zip(result, expected):
        np.testing.assert_array_equal(got, want)

    # The lookup holds each found id once, with one of its gathered coordinates.
    idx, lon, _ = _node_lookup(paths, needed).gather(needed)
    gathered = {}
    for pos, x in zip(*expected[:2]):
        gathered.setdefault(int(pos), set()).add(float(x))
    assert np.flatnonzero(idx >= 0).tolist() == sorted(gathered)
    assert all(lon[pos] in gathered[pos] for pos in gathered)


@pytest.mark.parametrize("requested", [None, [b"highway"]], ids=["all_tags", "minimal"])
def test_way_tags_replace_invalid_utf8(requested):
    """Way tags resolve through the block's string table with invalid UTF-8 bytes
    replaced, resolving every tag or (minimal-tags mode) only the requested keys."""
    from pyrosm.engine import decode

    string_table = [b"", b"highway", b"bad\xffvalue", b"name"]
    ways = {
        "id": np.array([1, 2]),
        "keys": np.array([1, 3, 1]),
        "vals": np.array([2, 2, 2]),
        "tags_off": np.array([0, 2, 3]),
        "refs": np.array([10, 11, 12]),
        "refs_off": np.array([0, 2, 3]),
        "version": np.array([1, 1]),
        "timestamp": np.array([0, 0]),
        "visible": np.array([1, 1]),
    }
    found = decode._matching_ways(
        string_table, ways, [b"highway"], requested_tag_keys=requested
    )
    value = "bad\ufffdvalue"
    first = {"highway": value} if requested else {"highway": value, "name": value}
    assert found["tags"] == [first, {"highway": value}]


def test_block_tags_share_one_decoded_string_table():
    """A block's way and relation tags resolve through one decode of its string table:
    their strings are the same objects, with invalid UTF-8 bytes replaced."""
    from pyrosm.engine import decode

    settings = decode._DecodeSettings(
        None, None, [b"building"], include_nodes=False, spill_all_ways=False
    )
    string_table = [b"", b"building", b"ye\xffs", b"type", b"multipolygon", b"outer"]
    meta = {"version": np.array([1]), "timestamp": np.array([0])}
    ways = {
        "id": np.array([1]),
        "keys": np.array([1]),
        "vals": np.array([2]),
        "tags_off": np.array([0, 1]),
        "refs": np.array([10, 11, 10]),
        "refs_off": np.array([0, 3]),
        "visible": np.array([1]),
        **meta,
    }
    relations = {
        "id": np.array([9]),
        "keys": np.array([1, 3]),
        "vals": np.array([2, 4]),
        "tags_off": np.array([0, 2]),
        "memids": np.array([1]),
        "members_off": np.array([0, 1]),
        "types": np.array([1]),
        "roles": np.array([5]),
        "changeset": np.array([0]),
        "visible": np.array([1]),
        **meta,
    }
    header = {"granularity": 100, "lat_offset": 0, "lon_offset": 0}
    shard = decode._decode_one_block(
        settings, string_table, header, None, ways, relations
    )
    way_tags, relation_tags = shard["way_tags"][0], shard["rel_tags"][0]
    assert way_tags == {"building": "ye\ufffds"}
    assert relation_tags == {"building": "ye\ufffds", "type": "multipolygon"}
    assert way_tags["building"] is relation_tags["building"]
    assert shard["rel_memrole"].tolist() == ["outer"]


@pytest.mark.parametrize(
    "read",
    [
        lambda osm: osm.get_network(),
        lambda osm: osm.get_network(nodes=True)[1],
        lambda osm: osm.get_buildings(),
        lambda osm: osm.get_pois(),
        lambda osm: osm.get_data_by_custom_criteria({"leisure": True}),
    ],
    ids=["network", "network_segments", "buildings", "pois", "custom"],
)
def test_keep_node_info_matches_in_memory(helsinki_pbf, fresh_cache, read):
    """With keep_node_info=True the out-of-core engine keeps each way's nodes as the
    in-memory reader does: the list of node ids of a way, missing for nodes and
    relations."""
    mine = read(OSM(helsinki_pbf, engine="out_of_core", workers=1, keep_node_info=True))
    ref = read(OSM(helsinki_pbf, keep_node_info=True, engine="in_memory"))
    assert "nodes" in mine.columns
    _assert_full_parity(mine, ref)


def test_keep_node_info_is_part_of_a_cached_read(test_pbf, fresh_cache, monkeypatch):
    """A cached read without the node column does not serve a read that keeps it, and a
    read that keeps it, served from the cache, returns lists of node ids again."""
    from pyrosm.engine import readers

    def read(**options):
        return OSM(test_pbf, engine="out_of_core", workers=1, **options).get_buildings()

    without = read()
    kept = read(keep_node_info=True)
    monkeypatch.setattr(readers, "_decode_and_run", None)
    cached = read(keep_node_info=True)
    ref = OSM(test_pbf, keep_node_info=True, engine="in_memory").get_buildings()
    assert "nodes" not in without.columns
    for frame in (kept, cached):
        _assert_full_parity(frame, ref)
        assert all(isinstance(v, list) and type(v[0]) is int for v in frame["nodes"])


def _untagged_relation_member_ways(fp):
    """How many ways of ``fp`` are untagged members of a tagged relation."""
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.primitive_block_decoder import decode_primitive_block

    untagged, members = set(), set()
    with open(fp, "rb") as f:
        for kind, offset, size in _index_blobs(fp):
            if kind != "OSMData":
                continue
            block = decode_primitive_block(_read_block(f, offset, size))
            ways, relations = block[3], block[4]
            if ways is not None:
                untagged |= set(ways["id"][np.diff(ways["tags_off"]) == 0].tolist())
            if relations is not None:
                counts = np.diff(relations["members_off"])
                owner = np.repeat(np.arange(len(counts)), counts)
                tagged = np.diff(relations["tags_off"]) > 0
                way_member = (relations["types"] == 1) & tagged[owner]
                members |= set(relations["memids"][way_member].tolist())
    return len(untagged & members)


@pytest.mark.parametrize(
    "fixture, options, constructor",
    [
        ("helsinki_pbf", {}, {}),
        ("test_pbf", {}, {}),
        ("test_pbf", {"keep_nodes": False}, {}),
        ("test_pbf", {"keep_ways": False}, {}),
        ("test_pbf", {"keep_relations": False}, {}),
        ("test_pbf", {}, {"keep_metadata": False}),
        ("test_pbf", {"filter_type": "exclude"}, {}),
    ],
    ids=[
        "helsinki",
        "all",
        "no_nodes",
        "no_ways",
        "no_relations",
        "no_metadata",
        "exclude",
    ],
)
def test_osm_custom_criteria_everything_matches_in_memory(
    fixture, options, constructor, request
):
    """custom_filter=None reads every tagged element with the out-of-core engine as the
    in-memory reader does: tagged nodes, ways and relations -- on Helsinki also relations
    whose member ways are untagged -- with each element kind left out, without metadata,
    and with a filter_type, which it does not use."""
    fp = request.getfixturevalue(fixture)
    mine = OSM(
        fp, engine="out_of_core", workers=1, **constructor
    ).get_data_by_custom_criteria(None, **options)
    ref = OSM(fp, **constructor, engine="in_memory").get_data_by_custom_criteria(None, **options)
    if not options and not constructor:
        assert {"node", "way", "relation"} <= set(ref["osm_type"])
    if fixture == "helsinki_pbf":
        assert _untagged_relation_member_ways(fp) > 0
    _assert_full_parity(mine, ref)
