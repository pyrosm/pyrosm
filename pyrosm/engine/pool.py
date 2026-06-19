"""Worker-count policy and ``multiprocessing.Pool`` orchestration for decoding blobs."""

import os
import shutil
import tempfile
from multiprocessing import Pool

from pyrosm.engine.blobs import _index_blobs
from pyrosm.engine.decode import _init_worker, _decode_batch

# Parallelising the decode only pays off above this file size; smaller files decode in a
# single process, where the process-spawn overhead would otherwise dominate.
_PARALLEL_MIN_FILE_BYTES = 70_000_000  # ~70 MB


def _auto_workers(filepath, n_blobs):
    """Pick a worker count for ``filepath``: single-core below the size threshold (where
    the process-spawn overhead dominates), otherwise a worker per CPU, capped at the blob
    count."""
    if os.path.getsize(filepath) < _PARALLEL_MIN_FILE_BYTES:
        return 1
    return max(1, min(os.cpu_count() or 1, n_blobs))


def _decode_all(
    filepath, blobs, workers, shard_dir, osm_keys, include_nodes, bbox_bounds=None
):
    """Decode every data blob into per-block shards (each worker spills one shard per block
    as it is decoded); return the flat list of shard paths."""
    n = len(blobs)
    per = (n + workers - 1) // workers
    tasks = [
        (i, blobs[i * per : (i + 1) * per])
        for i in range(workers)
        if blobs[i * per : (i + 1) * per]
    ]
    init_args = (filepath, shard_dir, osm_keys, include_nodes, bbox_bounds)
    if workers == 1:
        _init_worker(*init_args)
        return _decode_batch(tasks[0]) if tasks else []
    with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        return [path for paths in pool.map(_decode_batch, tasks) for path in paths]


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
