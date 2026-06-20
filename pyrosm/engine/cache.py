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
import importlib.util
import json
import os
import tempfile


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


def pyarrow_available():
    """Whether ``pyarrow`` is importable (the GeoParquet dependency). When it is not, the engine
    returns the in-memory frame and writes no cache instead of erroring."""
    return importlib.util.find_spec("pyarrow") is not None


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
        json.dumps(key, sort_keys=True, default=str).encode("utf-8")
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
