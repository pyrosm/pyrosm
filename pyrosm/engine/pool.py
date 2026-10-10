"""Worker-count resolution and parallel-decode orchestration for decoding blobs."""

import logging
import multiprocessing
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from pyrosm._log import timed
from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
from pyrosm.engine.decode import _init_worker, _decode_batch
from pyrosm.utils.progress import reporting

logger = logging.getLogger(__name__)

# ``workers="auto"`` decodes in parallel only for files at or above this size.
_PARALLEL_MIN_FILE_BYTES = 70_000_000  # ~70 MB
# Seconds between the progress reports of a decode running in a process pool.
_PROGRESS_INTERVAL = 0.2


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


class _ReportError(Exception):
    """Carries an exception of the progress callback past the pool fallback, so an ``OSError``
    it raises is not taken for a pool that cannot run."""


def _tracked(progress):
    """``progress`` with its callback's exceptions raised as :class:`_ReportError`."""
    if progress is None:
        return None
    report, start, total = progress

    def call(done, total):
        try:
            report(done, total)
        except Exception as e:
            raise _ReportError() from e

    return call, start, total


def _run_serial(func, tasks, initializer, initargs, progress):
    """Run every task in this process (see :func:`_run_pool` for ``progress``)."""
    if progress is None:
        initializer(*initargs)
        return [func(task) for task in tasks]
    report, start, total = progress
    added = 0

    def add(span):
        nonlocal added
        added += span
        report(start + added, total)

    report(0, total)
    initializer(*initargs, add)
    results = [func(task) for task in tasks]
    report(total, total)
    return results


def _map_tracked(pool, func, tasks, counter, progress):
    """``func`` over ``tasks`` in ``pool``, results in task order, reporting the bytes the
    workers add to ``counter`` every ``_PROGRESS_INTERVAL`` seconds while they run."""
    report, start, total = progress
    report(0, total)
    futures = [pool.submit(func, task) for task in tasks]
    pending = set(futures)
    while pending:
        _, pending = wait(pending, timeout=_PROGRESS_INTERVAL)
        with counter.get_lock():
            added = counter.value
        report(start + added, total)
    results = [future.result() for future in futures]
    report(total, total)
    return results


def _run_pool(
    func, tasks, workers, initializer, initargs, fallback_warning=None, progress=None
):
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
    the decode already did.

    ``progress``, when given, is ``(report, start, total)``: ``initializer`` then gets one more
    argument, the target each task adds the bytes it has decoded to (a callable in this
    process, a shared counter in a pool), and this process calls ``report(start + added,
    total)``: ``(0, total)`` first, then after each addition in one process or every
    ``_PROGRESS_INTERVAL`` seconds while a pool runs, and ``(total, total)`` at the end. A
    fallback to one process reports from ``(0, total)`` again."""
    progress = _tracked(progress)
    try:
        if workers == 1:
            return _run_serial(func, tasks, initializer, initargs, progress), False
        try:
            counter = None
            pool_initargs = initargs
            if progress is not None:
                # Its lock needs a semaphore, which an environment forbidding pools may refuse.
                counter = multiprocessing.Value("q", 0)
                pool_initargs = initargs + (counter,)
            with ProcessPoolExecutor(
                max_workers=workers, initializer=initializer, initargs=pool_initargs
            ) as pool:
                if counter is None:
                    return list(pool.map(func, tasks)), True
                return _map_tracked(pool, func, tasks, counter, progress), True
        except (BrokenProcessPool, OSError):
            if fallback_warning is not None:
                warnings.warn(fallback_warning, RuntimeWarning, stacklevel=2)
            return _run_serial(func, tasks, initializer, initargs, progress), False
    except _ReportError as e:
        raise e.__cause__


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
    progress=None,
):
    """Decode every data blob (``(offset, size, span)``, see
    :func:`~pyrosm.engine.blobs._data_blob_spans`) into per-block shards (each worker spills one
    shard per block as it is decoded). Returns ``(shard_paths, pool_ok)`` -- the flat list of
    shard paths and whether the decode pool ran (so the collect phase can mirror it instead of
    re-attempting a pool that could not start). ``progress`` is as for :func:`_run_pool`.
    """
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
        _decode_batch,
        tasks,
        workers,
        _init_worker,
        init_args,
        _DECODE_FALLBACK_WARNING,
        progress,
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
    progress=False,
):
    """Index + parallel-decode ``filepath`` into a temp shard dir, call
    ``run(shard_paths, collect_workers, report)`` and clean up. The shared front half of every
    public read. ``collect_workers`` is the worker count the collect phase should use: the
    resolved decode worker count when the decode pool ran, otherwise 1 -- so after a decode
    fallback (unguarded entry point or a pool-forbidden environment) the collect phase stays
    serial instead of re-attempting a pool that cannot start (and warning a second time).

    ``progress`` (``True``, ``False`` or a callable, validated by the caller) reports the bytes
    of the file decoded, out of the bytes indexed (the file size), for the whole read: a bar
    shown after 2 s and cleared at the end, or the caller's ``progress(done, total)``.
    ``report`` is that callback (``None`` without progress), for a ``run`` that makes another
    pass over the file."""
    name = Path(filepath).name
    with reporting(
        progress, "Reading %s" % name, delay=2.0, leave=False, timed=True
    ) as report:
        with timed(logger, "index", file=name) as fields:
            data_blobs, header_bytes = _data_blob_spans(_index_blobs(filepath))
            fields["blobs"] = len(data_blobs)
        total = header_bytes + sum(span for _, _, span in data_blobs)
        if workers is None:
            workers = 1
        elif isinstance(workers, str) and workers.lower() == "auto":
            workers = _auto_workers(filepath, len(data_blobs))
        else:
            workers = _cap_workers(workers)
        shard_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_")
        try:
            with timed(
                logger, "decode", workers=workers, blobs=len(data_blobs)
            ) as fields:
                shard_paths, pool_ok = _decode_all(
                    filepath,
                    data_blobs,
                    workers,
                    shard_dir,
                    osm_key_bytes,
                    include_nodes,
                    bbox_bounds,
                    requested_tag_keys,
                    None if report is None else (report, header_bytes, total),
                )
                fields["pool"] = pool_ok
            collect_workers = workers if pool_ok else 1
            with timed(logger, "collect", workers=collect_workers):
                return run(shard_paths, collect_workers, report)
        finally:
            shutil.rmtree(shard_dir, ignore_errors=True)
