"""Per-layer result cache for the out-of-core engine.

With ``engine="out_of_core"`` each feature read (``get_buildings``, ``get_landuse``, ...) is
assembled by the bounded per-call path and then its **result** is written once to a persistent
GeoParquet under a temp directory, keyed by the source file + the read's parameters. An identical
later read -- in the same session or a different one -- reads that GeoParquet back instead of
re-decoding the PBF, the same spirit as how ``get_data`` keeps a downloaded ``*.osm.pbf``. Each
layer is cached separately, so peak memory stays bounded by a single layer (never the whole file's
features). ``pyarrow`` is optional: without it the read just returns the in-memory frame and writes
no cache.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from pyrosm._log import log_event
from pyrosm.engine.pool import LOADING

logger = logging.getLogger(__name__)


# The version of the cached results' content, part of every key: bumped whenever a read
# returns different output, so a file cached by an older pyrosm is not served.
_RESULT_FORMAT = 10


def cache_dir():
    """The persistent result-cache directory (created on demand): ``<tempdir>/pyrosm/cache``."""
    path = Path(tempfile.gettempdir()) / "pyrosm" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stable(obj):
    """A JSON-serialisable, order-independent view of a cache-key input: dict keys are sorted,
    lists/tuples keep their order, and a shapely ``bounding_box`` becomes its WKT."""
    if isinstance(obj, dict):
        return {str(k): _stable(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_stable(x) for x in obj]
    if hasattr(obj, "wkt"):
        return {"__wkt__": obj.wkt}
    return obj


def _source_digest(filepath):
    """Digest of a source PBF's resolved path. It depends only on the path, so every cache file
    for one source file shares this prefix and they can be cleared together."""
    return hashlib.sha1(str(Path(filepath).resolve()).encode("utf-8")).hexdigest()[:16]


def _source_digests(filepath):
    """The source digests that ``filepath`` names: that of its own path, and for a bare file
    name also that of the file of that name in ``get_data``'s download directory, which
    :meth:`pyrosm.OSM.clear_downloads` reads a bare name as."""
    path = Path(filepath)
    paths = [path]
    if path.parent == Path("."):
        from pyrosm.utils.download import download_dir

        paths.append(download_dir() / path.name)
    return sorted({_source_digest(p) for p in paths})


def result_path(filepath, key_params):
    """Deterministic per-layer result-cache path, keyed on the source file (path + modification
    time + size) and the read's parameters (the filter, tag columns, metadata/bbox/element-kind
    options). Identical reads share one cache file; any difference keys a new one. The filename is
    ``result_<source>_<read>.parquet``, where ``<source>`` depends only on the source path so all
    of one file's cache files can be matched by that prefix (see :func:`clear`)."""
    fp = Path(filepath)
    st = fp.stat()
    key = {
        "format": _RESULT_FORMAT,
        "filepath": str(fp.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "params": _stable(key_params),
    }
    digest = hashlib.sha1(
        json.dumps(key, sort_keys=True, default=str).encode("ascii")
    ).hexdigest()[:16]
    return cache_dir() / ("result_%s_%s.parquet" % (_source_digest(filepath), digest))


def list_files(filepath=None):
    """List the cached layer GeoParquet files. With no ``filepath`` every cached file is listed;
    with a ``filepath`` only the cached layers for that source PBF, where a bare file name also
    names the downloaded file of that name (see :func:`_source_digests`). Returns a sorted list
    of string paths."""
    directory = cache_dir()
    if filepath is None:
        patterns = ["*.parquet"]
    else:
        patterns = ["result_%s_*.parquet" % d for d in _source_digests(filepath)]
    return sorted(
        str(p) for pattern in patterns for p in directory.glob(pattern) if p.is_file()
    )


def clear(filepath=None):
    """Remove out-of-core result-cache files. With no ``filepath`` the whole cache directory is
    emptied; with a ``filepath`` only the cached layers for that source PBF are removed, where a
    bare file name also names the downloaded file of that name (see :func:`_source_digests`).
    Returns the number of files removed."""
    directory = cache_dir()
    if filepath is None:
        patterns = ["*"]
    else:
        patterns = ["result_%s_*" % d for d in _source_digests(filepath)]
    removed = 0
    for pattern in patterns:
        for entry in directory.glob(pattern):
            if entry.is_file():
                entry.unlink()
                removed += 1
    return removed


def read_result(cache_path):
    """Read a cached layer GeoParquet back into the GeoDataFrame the reader would have returned.
    Object columns that GeoParquet round-trips as ``None`` are normalised to ``NaN`` so the cached
    frame matches the in-memory reader's missing-value representation."""
    import geopandas as gpd

    return missing_as_nan(gpd.read_parquet(cache_path))


def missing_as_nan(frame):
    """``frame`` with the missing values of its object columns other than ``geometry`` as
    ``NaN``, as :func:`read_result` reads a cached layer back."""
    import numpy as np

    for col in frame.columns:
        if col == "geometry":
            continue
        if frame[col].dtype == object:
            frame[col] = frame[col].where(frame[col].notna(), np.nan)
    return frame


def _temp_in(cache_path):
    """A closed, unique temp-file path in ``cache_path``'s directory, for a build-then-atomic-
    replace so concurrent identical first-reads never share a temp file or observe a half-written
    cache."""
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=cache_path.name + ".",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(tmp_path)


def materialize(cache_path, build, steps):
    """Populate ``cache_path`` by running ``build(tmp_path)`` -- which writes the result
    to a unique temp file and returns whether it wrote a non-empty result -- then
    atomically move it into place. An empty result is recorded with a ``.empty`` marker
    so an identical later read skips the rebuild. Returns the cached frame read back, or
    ``None`` for an empty (or already-marked-empty) result; reading it back is the step
    ``LOADING`` of ``steps`` (a :class:`~pyrosm.utils.progress.Steps`), the only step
    of a read served from the cache."""
    empty_marker = cache_path.with_name(cache_path.name + ".empty")
    if empty_marker.exists():
        log_event(logger, logging.DEBUG, "cache", status="empty", path=empty_marker)
        return None
    if cache_path.exists():
        log_event(logger, logging.DEBUG, "cache", status="hit", path=cache_path)
        steps.plan([LOADING])
    else:
        log_event(logger, logging.DEBUG, "cache", status="miss", path=cache_path)
        tmp_path = _temp_in(cache_path)
        try:
            if not build(tmp_path):
                empty_marker.touch()
                return None
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    steps.start_step(LOADING)
    return read_result(cache_path)


def materialize_pair(edges_path, nodes_path, build, steps, read_nodes=read_result):
    """Two-file variant of :func:`materialize` for ``get_network(nodes=True)``'s ``(nodes, edges)``
    tuple: ``build(edges_tmp, nodes_tmp)`` writes both temp files and returns whether the read was
    non-empty; on success both are atomically moved into place, and an empty read records a
    ``.empty`` marker beside ``edges_path``. The node frame is read back with ``read_nodes`` (the
    edge frame with :func:`read_result`). Returns ``(nodes, edges)``, or ``(None, None)`` for an
    empty (or already-marked-empty) result; reading them back is the step ``LOADING`` of
    ``steps``."""
    empty_marker = edges_path.with_name(edges_path.name + ".empty")
    if empty_marker.exists():
        log_event(logger, logging.DEBUG, "cache", status="empty", path=empty_marker)
        return None, None
    if edges_path.exists() and nodes_path.exists():
        log_event(logger, logging.DEBUG, "cache", status="hit", path=edges_path)
        steps.plan([LOADING])
    else:
        log_event(logger, logging.DEBUG, "cache", status="miss", path=edges_path)
        edges_tmp = _temp_in(edges_path)
        nodes_tmp = _temp_in(nodes_path)
        try:
            if not build(edges_tmp, nodes_tmp):
                empty_marker.touch()
                return None, None
            edges_tmp.replace(edges_path)
            nodes_tmp.replace(nodes_path)
        finally:
            for tmp in (edges_tmp, nodes_tmp):
                if tmp.exists():
                    tmp.unlink()
    steps.start_step(LOADING)
    return read_nodes(nodes_path), read_result(edges_path)
