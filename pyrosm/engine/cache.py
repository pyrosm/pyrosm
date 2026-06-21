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
import os
import tempfile

from rapidjson import dumps


def cache_dir():
    """The persistent result-cache directory (created on demand): ``<tempdir>/pyrosm/cache``."""
    path = os.path.join(tempfile.gettempdir(), "pyrosm", "cache")
    os.makedirs(path, exist_ok=True)
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


def result_path(filepath, key_params):
    """Deterministic per-layer result-cache path, keyed on the source file (path + modification
    time + size) and the read's parameters (the filter, tag columns, metadata/bbox/element-kind
    options). Identical reads share one cache file; any difference keys a new one."""
    st = os.stat(filepath)
    key = {
        "filepath": os.path.abspath(filepath),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "params": _stable(key_params),
    }
    digest = hashlib.sha1(
        dumps(key, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(cache_dir(), "result_%s.parquet" % digest)


def read_result(cache_path):
    """Read a cached layer GeoParquet back into the GeoDataFrame the reader would have returned.
    Object columns that GeoParquet round-trips as ``None`` are normalised to ``NaN`` so the cached
    frame matches the in-memory reader's missing-value representation."""
    import numpy as np
    import geopandas as gpd

    gdf = gpd.read_parquet(cache_path)
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype == object:
            gdf[col] = gdf[col].where(gdf[col].notna(), np.nan)
    return gdf


def _temp_in(cache_path):
    """A closed, unique temp-file path in ``cache_path``'s directory, for a build-then-atomic-
    replace so concurrent identical first-reads never share a temp file or observe a half-written
    cache."""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(cache_path),
        prefix=os.path.basename(cache_path) + ".",
        suffix=".tmp",
    )
    os.close(fd)
    return tmp_path


def materialize(cache_path, build):
    """Populate ``cache_path`` by running ``build(tmp_path)`` -- which writes the result to a
    unique temp file and returns whether it wrote a non-empty result -- then atomically move it
    into place. An empty result is recorded with a ``.empty`` marker so an identical later read
    skips the rebuild. Returns the cached frame read back, or ``None`` for an empty (or
    already-marked-empty) result."""
    empty_marker = cache_path + ".empty"
    if os.path.exists(empty_marker):
        return None
    if not os.path.exists(cache_path):
        tmp_path = _temp_in(cache_path)
        try:
            if not build(tmp_path):
                with open(empty_marker, "w"):
                    pass
                return None
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return read_result(cache_path)


def materialize_pair(edges_path, nodes_path, build, read_nodes=read_result):
    """Two-file variant of :func:`materialize` for ``get_network(nodes=True)``'s ``(nodes, edges)``
    tuple: ``build(edges_tmp, nodes_tmp)`` writes both temp files and returns whether the read was
    non-empty; on success both are atomically moved into place, and an empty read records a
    ``.empty`` marker beside ``edges_path``. The node frame is read back with ``read_nodes`` (the
    edge frame with :func:`read_result`). Returns ``(nodes, edges)``, or ``(None, None)`` for an
    empty (or already-marked-empty) result."""
    empty_marker = edges_path + ".empty"
    if os.path.exists(empty_marker):
        return None, None
    if not (os.path.exists(edges_path) and os.path.exists(nodes_path)):
        edges_tmp = _temp_in(edges_path)
        nodes_tmp = _temp_in(nodes_path)
        try:
            if not build(edges_tmp, nodes_tmp):
                with open(empty_marker, "w"):
                    pass
                return None, None
            os.replace(edges_tmp, edges_path)
            os.replace(nodes_tmp, nodes_path)
        finally:
            for tmp in (edges_tmp, nodes_tmp):
                if os.path.exists(tmp):
                    os.remove(tmp)
    return read_nodes(nodes_path), read_result(edges_path)
