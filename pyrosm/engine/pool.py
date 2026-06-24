"""Worker-count resolution and parallel-decode orchestration for decoding blobs."""

import os
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from pyrosm.engine.blobs import _index_blobs
from pyrosm.engine.decode import _init_worker, _decode_batch

# ``workers="auto"`` decodes in parallel only for files at or above this size.
_PARALLEL_MIN_FILE_BYTES = 70_000_000  # ~70 MB


def _auto_workers(filepath, n_blobs):
    """Worker count for ``workers="auto"``: a single core for files below ~70 MB, otherwise
    one worker per available CPU core, capped at the number of data blobs."""
    if Path(filepath).stat().st_size < _PARALLEL_MIN_FILE_BYTES:
        return 1
    return max(1, min(os.cpu_count() or 1, n_blobs))


def _cap_workers(workers):
    """Cap an explicit worker count at the host's CPU-core count, warning when it exceeds
    them."""
    n_cores = os.cpu_count() or 1
    if workers > n_cores:
        warnings.warn(
            f"workers={workers} exceeds the {n_cores} CPU cores available on this "
            f"machine; reading with {n_cores} workers instead.",
            UserWarning,
            stacklevel=2,
        )
        return n_cores
    return workers


def _run_pool(func, tasks, workers, initializer, initargs, fallback_warning=None):
    """Map ``func`` over ``tasks`` across a process pool of ``workers`` (each worker process
    initialised with ``initializer(*initargs)``); return ``(results, pool_ok)`` -- the
    per-task results in task order, and whether the pool actually ran.

    ``workers == 1`` runs every task in this process (no pool, ``pool_ok`` False). A pool that
    cannot start (``OSError`` in an environment that forbids pools) or whose workers die on
    start (``BrokenProcessPool`` -- e.g. a read not guarded by ``if __name__ == "__main__":``,
    so each spawned worker re-imports and re-runs the entry point) falls back to a single
    process and reports ``pool_ok`` False, so a later phase can stay serial instead of
    re-attempting a pool that cannot start. ``fallback_warning`` (when given) is emitted on
    that fallback; passing ``None`` keeps a downstream phase from warning a second time after
    the decode already did."""
    if workers == 1:
        initializer(*initargs)
        return [func(task) for task in tasks], False
    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initializer, initargs=initargs
        ) as pool:
            return list(pool.map(func, tasks)), True
    except (BrokenProcessPool, OSError):
        if fallback_warning is not None:
            warnings.warn(fallback_warning, RuntimeWarning, stacklevel=2)
        initializer(*initargs)
        return [func(task) for task in tasks], False


_DECODE_FALLBACK_WARNING = (
    "Parallel decoding could not start and fell back to a single process. This happens when "
    'the read is not inside an `if __name__ == "__main__":` block (the worker processes '
    "cannot re-import the entry point), or in environments that forbid process pools. Guard "
    "the entry point, or pass workers=1 to silence this."
)


def _decode_all(
    filepath,
    blobs,
    workers,
    shard_dir,
    osm_keys,
    include_nodes,
    bbox_bounds=None,
    requested_tag_keys=None,
):
    """Decode every data blob into per-block shards (each worker spills one shard per block as
    it is decoded). Returns ``(shard_paths, pool_ok)`` -- the flat list of shard paths and
    whether the decode pool ran (so the collect phase can mirror it instead of re-attempting a
    pool that could not start)."""
    n = len(blobs)
    per = (n + workers - 1) // workers
    tasks = [
        (i, blobs[i * per : (i + 1) * per])
        for i in range(workers)
        if blobs[i * per : (i + 1) * per]
    ]
    init_args = (
        filepath,
        shard_dir,
        osm_keys,
        include_nodes,
        bbox_bounds,
        requested_tag_keys,
    )
    results, pool_ok = _run_pool(
        _decode_batch, tasks, workers, _init_worker, init_args, _DECODE_FALLBACK_WARNING
    )
    return [path for paths in results for path in paths], pool_ok


def _decode_and_run(
    filepath,
    osm_key_bytes,
    include_nodes,
    workers,
    run,
    bbox_bounds=None,
    requested_tag_keys=None,
):
    """Index + parallel-decode ``filepath`` into a temp shard dir, call
    ``run(shard_paths, collect_workers)`` and clean up. The shared front half of every public
    read. ``collect_workers`` is the worker count the collect phase should use: the resolved
    decode worker count when the decode pool ran, otherwise 1 -- so after a decode fallback
    (unguarded entry point or a pool-forbidden environment) the collect phase stays serial
    instead of re-attempting a pool that cannot start (and warning a second time)."""
    data_blobs = [
        (offset, size)
        for (blob_type, offset, size) in _index_blobs(filepath)
        if blob_type == "OSMData"
    ]
    if workers is None:
        workers = 1
    elif isinstance(workers, str) and workers.lower() == "auto":
        workers = _auto_workers(filepath, len(data_blobs))
    else:
        workers = _cap_workers(workers)
    shard_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_")
    try:
        shard_paths, pool_ok = _decode_all(
            filepath,
            data_blobs,
            workers,
            shard_dir,
            osm_key_bytes,
            include_nodes,
            bbox_bounds,
            requested_tag_keys,
        )
        collect_workers = workers if pool_ok else 1
        return run(shard_paths, collect_workers)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
