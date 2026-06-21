"""Worker-count policy and parallel-decode orchestration for decoding blobs."""

import os
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from pyrosm.engine.blobs import _index_blobs
from pyrosm.engine.decode import _init_worker, _decode_batch

# Parallelising the decode only pays off above this file size; smaller files decode in a
# single process, where the process-spawn overhead would otherwise dominate.
_PARALLEL_MIN_FILE_BYTES = 70_000_000  # ~70 MB


def _auto_workers(filepath, n_blobs):
    """Pick a worker count for ``filepath``: single-core below the size threshold (where
    the process-spawn overhead dominates), otherwise a worker per CPU, capped at the blob
    count."""
    if Path(filepath).stat().st_size < _PARALLEL_MIN_FILE_BYTES:
        return 1
    return max(1, min(os.cpu_count() or 1, n_blobs))


def _decode_serial(tasks, init_args):
    """Decode every task in this process (the single-process path and the parallel
    fallback). Returns the flat list of shard paths."""
    _init_worker(*init_args)
    return [path for task in tasks for path in _decode_batch(task)]


def _decode_all(
    filepath, blobs, workers, shard_dir, osm_keys, include_nodes, bbox_bounds=None
):
    """Decode every data blob into per-block shards (each worker spills one shard per block
    as it is decoded); return the flat list of shard paths.

    Parallel decoding uses a process pool. ``ProcessPoolExecutor`` reports a broken pool
    instead of endlessly respawning dead workers (which would hang), so two failure modes
    fall back to a single process with a warning rather than hanging or erroring: the read
    running in a module not guarded by ``if __name__ == "__main__":`` (each spawned worker
    re-imports and re-runs it, so the workers die on start), and environments that forbid
    creating a process pool at all."""
    n = len(blobs)
    per = (n + workers - 1) // workers
    tasks = [
        (i, blobs[i * per : (i + 1) * per])
        for i in range(workers)
        if blobs[i * per : (i + 1) * per]
    ]
    init_args = (filepath, shard_dir, osm_keys, include_nodes, bbox_bounds)
    if workers == 1:
        return _decode_serial(tasks, init_args)
    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=init_args
        ) as pool:
            return [path for paths in pool.map(_decode_batch, tasks) for path in paths]
    except (BrokenProcessPool, OSError):
        warnings.warn(
            "Parallel decoding could not start and fell back to a single process. This "
            'happens when the read is not inside an `if __name__ == "__main__":` block (the '
            "worker processes cannot re-import the entry point), or in environments that "
            "forbid process pools. Guard the entry point, or pass workers=1 to silence this.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _decode_serial(tasks, init_args)


def _decode_and_run(
    filepath, osm_key_bytes, include_nodes, workers, run, bbox_bounds=None
):
    """Index + parallel-decode ``filepath`` into a temp shard dir, call ``run(shard_paths)``
    and clean up. The shared front half of every public read."""
    data_blobs = [
        (offset, size)
        for (blob_type, offset, size) in _index_blobs(filepath)
        if blob_type == "OSMData"
    ]
    if workers is None:
        workers = _auto_workers(filepath, len(data_blobs))
    shard_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_")
    try:
        shard_paths = _decode_all(
            filepath,
            data_blobs,
            workers,
            shard_dir,
            osm_key_bytes,
            include_nodes,
            bbox_bounds,
        )
        return run(shard_paths)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
