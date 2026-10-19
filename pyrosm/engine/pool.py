"""Worker-count resolution and parallel-decode orchestration for decoding blobs."""

import contextvars
import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

from pyrosm._log import timed
from pyrosm.engine.blobs import _data_blob_spans, _index_blobs
from pyrosm.engine.decode import (
    _DecodeSettings,
    _decode_batch,
    _decode_batch_threads,
)
from pyrosm.utils import default_workers, warn_future_once
from pyrosm.utils.progress import reporting

logger = logging.getLogger(__name__)

# ``workers="auto"`` decodes in parallel only for files at or above this size, on at most
# ``_AUTO_FEW_WORKERS`` threads below ``_AUTO_ALL_CORES_BYTES``.
_PARALLEL_MIN_FILE_BYTES = 20_000_000  # ~20 MB
_AUTO_ALL_CORES_BYTES = 70_000_000  # ~70 MB
_AUTO_FEW_WORKERS = 4
# ``workers="auto"`` leaves this many CPU cores free, and uses at most ``_AUTO_MAX_WORKERS``
# worker processes.
_AUTO_SPARE_CORES = 2
_AUTO_MAX_WORKERS = 8
# A parallel decode splits the file into about this many tasks per worker.
_TASKS_PER_WORKER = 8
# Seconds between the progress reports of a decode running in a process pool.
_PROGRESS_INTERVAL = 0.2
# In a worker process of a pool: the counter its tasks add their decoded bytes to, shared
# with the process that runs the pool (None without progress). Set by `_init_counter`.
_COUNTER = None

# The threads the threaded read running in this thread uses for its steps after the
# decode, 0 outside a threaded read (set for the read by :func:`_decode_and_run`). Each
# thread has its own value.
_READ_THREADS = contextvars.ContextVar("pyrosm_read_threads", default=0)


def read_threads():
    """The threads the threaded read running in this thread uses for its steps after the
    decode (1 outside a threaded read)."""
    return max(1, _READ_THREADS.get())


def in_threaded_read():
    """Whether a threaded read is running in this thread."""
    return _READ_THREADS.get() > 0


class _PoolStartError(Exception):
    """The pool could not start: an ``OSError`` from creating the executor, or an ``OSError``
    or ``RuntimeError`` from submitting a task (which starts a worker process or thread).
    """


def _ordered_map(executor, func, items, window):
    """Yield ``func(item)`` for each of ``items`` in order, run by ``executor`` with at most
    ``window`` items in flight."""
    pending = deque()
    for item in items:
        if len(pending) >= window:
            yield pending.popleft().result()
        try:
            pending.append(executor.submit(func, item))
        except (OSError, RuntimeError) as err:
            raise _PoolStartError(err) from err
    while pending:
        yield pending.popleft().result()


def thread_map(function, items, window=None):
    """Yield ``function(item)`` for each of ``items`` in order, run on the threads of the
    threaded read running in this thread (see :func:`read_threads`), with at most
    ``window`` items in flight (twice the threads by default)."""
    items = list(items)
    threads = read_threads()
    if threads <= 1 or len(items) <= 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(threads) as pool:
        yield from _ordered_map(pool, function, items, window or 2 * threads)


def _available_cores():
    """The CPU cores this process may run on: ``os.process_cpu_count()`` on Python 3.13+,
    else the cores of its CPU affinity where the platform reports one (Linux), else
    ``os.cpu_count()``; at least 1. A job or container limited to some of a machine's cores
    (a Slurm job, ``taskset``, ``docker --cpuset-cpus``) counts only those."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def auto_is_parallel(nbytes):
    """Whether ``workers="auto"`` decodes ``nbytes`` bytes of source data in parallel: from
    about 20 MB up."""
    return nbytes >= _PARALLEL_MIN_FILE_BYTES


def auto_worker_count(nbytes, n_blobs):
    """Worker count for ``workers="auto"`` on ``nbytes`` bytes of source data in ``n_blobs``
    data blobs: a single core below ~20 MB, otherwise the CPU cores this process may use
    (:func:`_available_cores`) less two, at most 4 below ~70 MB, at least 1, at most 8
    worker processes (threads are not capped from ~70 MB) and at most the number of data
    blobs.
    """
    if not auto_is_parallel(nbytes):
        return 1
    n_cores = _available_cores() - _AUTO_SPARE_CORES
    if nbytes < _AUTO_ALL_CORES_BYTES:
        n_cores = min(n_cores, _AUTO_FEW_WORKERS)
    if _decode_backend() == "processes":
        n_cores = min(n_cores, _AUTO_MAX_WORKERS)
    return max(1, min(n_cores, n_blobs))


def _auto_workers(filepath, n_blobs):
    """Worker count for ``workers="auto"`` on the file ``filepath`` (see
    :func:`auto_worker_count`)."""
    return auto_worker_count(Path(filepath).stat().st_size, n_blobs)


def _cap_workers(workers):
    """Cap an explicit worker count at the CPU cores this process may use
    (:func:`_available_cores`), warning when it exceeds them."""
    n_cores = _available_cores()
    if workers > n_cores:
        warnings.warn(
            f"workers={workers} exceeds the {n_cores} CPU cores available to this "
            f"process; reading with {n_cores} workers instead.",
            UserWarning,
            stacklevel=2,
        )
        return n_cores
    return workers


class _ReportError(Exception):
    """Carries an exception of the progress callback past the pool fallback, so an ``OSError``
    it raises is not taken for a pool that cannot run."""


def _raising_progress(progress):
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


def _run_serial(func, tasks, progress):
    """Run every task in this process (see :func:`_run_pool`)."""
    if progress is None:
        return [func(task, None) for task in tasks]
    report, start, total = progress
    added = 0

    def add(span):
        nonlocal added
        added += span
        report(start + added, total)

    report(0, total)
    results = [func(task, add) for task in tasks]
    report(total, total)
    return results


def _init_counter(counter):
    """Initialise a worker process of a pool with its shared counter (see `_COUNTER`)."""
    global _COUNTER
    _COUNTER = counter


def _add_to_counter(span):
    """Add ``span`` decoded bytes to the worker process's shared counter."""
    with _COUNTER.get_lock():
        _COUNTER.value += span


def _call_in_worker(func, task):
    """``func(task, advance)`` in a worker process of a pool, ``advance`` adding to its
    shared counter (None without progress)."""
    return func(task, None if _COUNTER is None else _add_to_counter)


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


def _run_pool(func, tasks, workers, fallback_warning=None, progress=None):
    """Map ``func(task, advance)`` over ``tasks`` across a process pool of ``workers``;
    return ``(results, pool_ok)`` -- the per-task results in task order, and whether the pool
    actually ran. ``func`` must pickle (a module-level function, or a ``functools.partial``
    of one), and everything a task needs comes with it, so reads running at the same time
    in this process do not share any state.

    ``workers == 1`` runs every task in this process (no pool, ``pool_ok`` False). A pool that
    cannot start (``OSError`` in an environment that forbids pools) or whose workers die on
    start (``BrokenProcessPool`` -- e.g. a read not guarded by ``if __name__ == "__main__":``,
    so each spawned worker re-imports and re-runs the entry point) falls back to a single
    process and reports ``pool_ok`` False. ``fallback_warning`` (when given) is emitted on
    that fallback.

    ``progress``, when given, is ``(report, start, total)``: each task then calls its
    ``advance(span)`` with the bytes it has decoded (``advance`` is None without
    progress), which adds them up in this process or, in a pool, in a shared counter. This
    process calls ``report(start + added, total)``: ``(0, total)`` first, then after each
    addition in one process or every ``_PROGRESS_INTERVAL`` seconds while a pool runs, and
    ``(total, total)`` at the end. A fallback to one process reports from ``(0, total)``
    again."""
    progress = _raising_progress(progress)
    try:
        if workers == 1:
            return _run_serial(func, tasks, progress), False
        try:
            counter = None
            if progress is not None:
                # Its lock needs a semaphore, which an environment forbidding pools may refuse.
                counter = multiprocessing.Value("q", 0)
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_counter, initargs=(counter,)
            ) as pool:
                in_pool = partial(_call_in_worker, func)
                if counter is None:
                    return list(pool.map(in_pool, tasks)), True
                return _map_tracked(pool, in_pool, tasks, counter, progress), True
        except (BrokenProcessPool, OSError):
            if fallback_warning is not None:
                warnings.warn(fallback_warning, RuntimeWarning, stacklevel=2)
            return _run_serial(func, tasks, progress), False
    except _ReportError as e:
        raise e.__cause__


_DECODE_FALLBACK_WARNING = (
    "Parallel decoding could not start and fell back to a single process. This happens when "
    'the read is not inside an `if __name__ == "__main__":` block (the worker processes '
    "cannot re-import the entry point), or in environments that forbid process pools. Guard "
    "the entry point, or pass workers=1 to silence this."
)


def _decode_tasks(blobs, workers):
    """Split ``blobs`` into ``(task_id, blobs)`` tasks of consecutive blobs in file order: one
    task for a single worker, otherwise about ``_TASKS_PER_WORKER`` per worker, each covering
    a similar share of the file's bytes. Each blob goes to the task whose share holds the
    blob's start."""
    n_tasks = 1 if workers == 1 else workers * _TASKS_PER_WORKER
    total = sum(span for _, _, span in blobs)
    tasks = {}
    before = 0
    for offset, size, span in blobs:
        tasks.setdefault(before * n_tasks // total, []).append((offset, size, span))
        before += span
    return list(enumerate(tasks.values()))


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
    spill_all_ways=True,
):
    """Decode every data blob (``(offset, size, span)``, see
    :func:`~pyrosm.engine.blobs._data_blob_spans`) into shards, on threads or in worker
    processes (see :func:`_decode_backend`). A pool hands the tasks of
    :func:`_decode_tasks` to its workers as they become free. Returns ``(shards,
    pool_ok)``: the shards in file order (paths, or :class:`~pyrosm.engine.decode._Shard`
    objects on threads) and whether the decode ran in parallel. ``progress`` is as for
    :func:`_run_pool`. ``spill_all_ways=False`` leaves out every way's id and node refs,
    which only the lookup of relation members reads.
    """
    tasks = _decode_tasks(blobs, workers)
    settings = _DecodeSettings(
        filepath,
        shard_dir,
        osm_keys,
        include_nodes,
        bbox_bounds,
        requested_tag_keys,
        spill_all_ways,
    )
    if _decode_backend() == "threads":
        return _decode_threads(tasks, workers, settings, progress), workers > 1
    results, pool_ok = _run_pool(
        partial(_decode_batch, settings),
        tasks,
        workers,
        _DECODE_FALLBACK_WARNING,
        progress,
    )
    return [path for paths in results for path in paths], pool_ok


_PROCESSES_BACKEND_WARNING = (
    "PYROSM_DECODE_BACKEND=processes is deprecated and will be removed in a future release "
    "of pyrosm; reads, crops and merges run on threads by default. If you need the worker "
    "processes, please tell us why in an issue: https://github.com/pyrosm/pyrosm/issues"
)


def _decode_backend():
    """How a decode runs: ``"threads"`` (GIL-free, no ``__main__`` guard needed; the
    default) or ``"processes"`` (deprecated: warns once per process), from the
    ``PYROSM_DECODE_BACKEND`` environment variable.
    """
    backend = os.environ.get("PYROSM_DECODE_BACKEND", "threads").lower()
    if backend not in ("threads", "processes"):
        raise ValueError(
            "PYROSM_DECODE_BACKEND should be 'threads' or 'processes', got %r" % backend
        )
    if backend == "processes":
        warn_future_once(_PROCESSES_BACKEND_WARNING)
    return backend


class _ThreadCounter:
    """The decoded bytes the decode threads of one read add up (``add``), read as the
    pool's ``multiprocessing.Value`` is (see :func:`_map_tracked`)."""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock

    def add(self, span):
        with self._lock:
            self.value += span


def _decode_threads(tasks, workers, settings, progress):
    """Run the decode ``tasks`` of the read with the
    :class:`~pyrosm.engine.decode._DecodeSettings` ``settings`` with
    :func:`~pyrosm.engine.decode._decode_batch_threads` on ``workers`` threads (the calling
    thread for one). Returns the shards in file order. ``progress`` is as for
    :func:`_run_pool`; it is called from the calling thread only.
    """
    if not tasks:
        return []
    decode_batch = partial(_decode_batch_threads, settings)
    if workers == 1:
        results = _run_serial(decode_batch, tasks, progress)
    else:
        with ThreadPoolExecutor(workers) as pool:
            if progress is None:
                results = list(pool.map(partial(decode_batch, advance=None), tasks))
            else:
                counter = _ThreadCounter()
                results = _map_tracked(
                    pool,
                    partial(decode_batch, advance=counter.add),
                    tasks,
                    counter,
                    progress,
                )
    return [shard for shards in results for shard in shards]


def _decode_and_run(
    filepath,
    osm_key_bytes,
    include_nodes,
    workers,
    run,
    bbox_bounds=None,
    requested_tag_keys=None,
    progress=False,
    spill_all_ways=True,
):
    """Index + parallel-decode ``filepath`` into a temp shard dir, call
    ``run(shard_paths, report)`` in this process and clean up. The shared front half of every
    public read.

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
        workers = default_workers(workers)
        if isinstance(workers, str) and workers.lower() == "auto":
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
                    spill_all_ways,
                )
                fields["pool"] = pool_ok
            token = None
            if _decode_backend() == "threads":
                token = _READ_THREADS.set(workers)
            try:
                with timed(logger, "collect"):
                    return run(shard_paths, report)
            finally:
                if token is not None:
                    _READ_THREADS.reset(token)
        finally:
            shutil.rmtree(shard_dir, ignore_errors=True)
