"""Worker-count policy and ``multiprocessing.Pool`` orchestration for decoding blobs."""

import os
from multiprocessing import Pool

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


def _decode_all(filepath, blobs, workers, shard_dir, osm_keys, include_nodes):
    """Decode every data blob into per-worker shards; return the shard paths."""
    n = len(blobs)
    per = (n + workers - 1) // workers
    tasks = [
        (i, blobs[i * per : (i + 1) * per])
        for i in range(workers)
        if blobs[i * per : (i + 1) * per]
    ]
    if workers == 1:
        _init_worker(filepath, shard_dir, osm_keys, include_nodes)
        return [_decode_batch(tasks[0])] if tasks else []
    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(filepath, shard_dir, osm_keys, include_nodes),
    ) as pool:
        return pool.map(_decode_batch, tasks)
