"""Find the smallest single OSM extract that contains an area.

Candidates come from three providers:

- Geofabrik, from the vendored ``geofabrik_index.geojson.gz``;
- BBBike city extracts, from the vendored ``bbbike_index.geojson.gz``;
- Movisda administrative areas and 1°/10° grid tiles, from the index files published at
  https://osm.download.movisda.io (fetched and cached next to the downloads).

Extracts are never merged, so the answer is always one file. Candidates are ranked by download
size: Movisda's index lists sizes; for the others a HEAD request asks the server, and the answer
is cached for a week. Refresh the vendored snapshots with ``scripts/update_extract_indexes.py``.
"""

import gzip
import hashlib
import http.client
import io
import json
import time
import warnings
from pathlib import Path
from urllib.error import HTTPError

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from pyrosm.data.geofabrik_index import _EQUAL_AREA_CRS
from pyrosm.data.geofabrik_index import _load_index as _load_geofabrik_index
from pyrosm.utils.download import download_dir, open_url, write_atomic

_BBBIKE_INDEX_PATH = Path(__file__).parent / "bbbike_index.geojson.gz"
_MOVISDA_URL = "https://osm.download.movisda.io"
_MOVISDA_INDEXES = {
    "admin": "admin/Admin-latest.geojson",
    "grid": "grid/grid-latest.geojson",
}
_INDEX_MAX_AGE = 24 * 3600
_SIZE_MAX_AGE = 7 * 24 * 3600
_COLUMNS = ["provider", "id", "name", "url", "bytes", "geometry"]
_FETCH_ERRORS = (OSError, http.client.HTTPException)
_TIMEOUT = 60

_bbbike_cache = None
_movisda_cache = {}


def _frame(provider, ids, names, urls, sizes, geometry):
    """One provider's candidates; invalid extent polygons (a few providers publish
    self-intersecting ones) are repaired so containment tests stay reliable."""
    return gpd.GeoDataFrame(
        {
            "provider": provider,
            "id": list(ids),
            "name": list(names),
            "url": list(urls),
            "bytes": list(sizes),
        },
        geometry=gpd.GeoSeries(list(geometry)).make_valid().values,
        crs="EPSG:4326",
    )


def _geofabrik_extracts():
    gdf = _load_geofabrik_index()
    names = gdf["name"].fillna(gdf["id"])
    return _frame(
        "Geofabrik", gdf["id"], names, gdf["pbf"], [None] * len(gdf), gdf.geometry
    )


def _bbbike_extracts():
    global _bbbike_cache
    if _bbbike_cache is None:
        with gzip.open(_BBBIKE_INDEX_PATH, "rt", encoding="utf-8") as f:
            gdf = gpd.GeoDataFrame.from_features(json.load(f)["features"])
        _bbbike_cache = _frame(
            "BBBike",
            gdf["id"],
            gdf["name"],
            gdf["pbf"],
            [None] * len(gdf),
            gdf.geometry,
        )
    return _bbbike_cache


def _tile_id(bounds):
    """Name a grid tile by its south-west corner, e.g. ``N60E024`` or ``S40W080-10``.

    Movisda's own tile names have the east/west letter swapped, so the id is built from the
    tile's geometry; the download URL keeps Movisda's name.
    """
    minx, miny, maxx, _ = (round(v) for v in bounds)
    size = maxx - minx
    name = "%s%02d%s%03d" % (
        "N" if miny >= 0 else "S",
        abs(miny),
        "E" if minx >= 0 else "W",
        abs(minx),
    )
    return name if size == 1 else "%s-%d" % (name, size)


def _check_movisda_index(data, kind):
    """Raise ``ValueError`` unless ``data`` is a Movisda index pyrosm can read.

    Every feature needs a non-empty (Multi)Polygon geometry within lon/lat bounds, a non-empty
    ``prefix``, an integer ``bytes`` between 0 and 2**63 and, for administrative areas, a
    ``name``.
    """

    def readable(feature):
        props = feature["properties"]
        geometry = shape(feature["geometry"])
        minx, miny, maxx, maxy = geometry.bounds
        return (
            feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
            and not geometry.is_empty
            and -180 <= minx <= maxx <= 180
            and -90 <= miny <= maxy <= 90
            and isinstance(props["prefix"], str)
            and props["prefix"] != ""
            and type(props["bytes"]) is int
            and 0 <= props["bytes"] < 2**63
            and (kind != "admin" or isinstance(props["name"], str))
        )

    try:
        features = json.loads(data)["features"]
        valid = bool(features) and all(readable(f) for f in features)
    except Exception:
        valid = False
    if not valid:
        raise ValueError("the response is not a Movisda %s index" % kind)


def _cached_index(url, path, update=False, check=None):
    """Return ``path`` holding the file at ``url``, fetching it when needed.

    ``<path>.etag`` records the ETag, size and SHA-256 of the copy, and its time stamp the last
    check. A copy checked less than a day ago is used as is unless ``update`` is set. Otherwise it
    is revalidated with its ETag (sent only when the copy's hash matches), so an unchanged file is
    not downloaded again. A new file replaces
    the copy only after ``check(data)`` accepts it; when the server cannot be reached or sends
    something ``check`` rejects, the copy is kept with a warning.
    """
    path = Path(path)
    meta_path = path.with_name(path.name + ".etag")
    have_copy = path.exists()
    try:
        meta = json.loads(meta_path.read_text())
        bound = meta["bytes"] == path.stat().st_size
        checked = meta_path.stat().st_mtime if bound else 0
    except (OSError, ValueError, TypeError, KeyError):
        meta, checked = {}, 0
    if have_copy and not update and 0 <= time.time() - checked < _INDEX_MAX_AGE:
        return path
    etag = ""
    if checked and meta.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
        etag = meta.get("etag") or ""
    headers = {"If-None-Match": etag} if etag else None
    try:
        with open_url(url, headers=headers, timeout=_TIMEOUT) as response:
            data = response.read()
            new_etag = response.headers.get("ETag") or ""
        if check is not None:
            check(data)
        write_atomic(path, lambda out_file: out_file.write(data))
        digest = hashlib.sha256(data).hexdigest()
        meta = json.dumps({"etag": new_etag, "bytes": len(data), "sha256": digest})
        meta = meta.encode()
        write_atomic(meta_path, lambda out_file: out_file.write(meta))
    except HTTPError as e:
        if e.code == 304 and have_copy:
            if hashlib.sha256(path.read_bytes()).hexdigest() == meta.get("sha256"):
                record = json.dumps(meta).encode()
                write_atomic(meta_path, lambda out_file: out_file.write(record))
        elif have_copy:
            warnings.warn(
                "Could not refresh %s (%s); using the cached copy." % (url, e)
            )
        else:
            raise
    except (*_FETCH_ERRORS, ValueError) as e:
        if not have_copy:
            raise
        warnings.warn("Could not refresh %s (%s); using the cached copy." % (url, e))
    return path


def _movisda_frame(source, kind):
    """Read a Movisda index (a path or file object) into candidate rows.

    Raises ``ValueError`` when it cannot be read.
    """
    try:
        gdf = gpd.read_file(source)
        urls = [
            "%s/%s/%slatest.osm.pbf" % (_MOVISDA_URL, kind, p) for p in gdf["prefix"]
        ]
        if kind == "admin":
            ids = gdf["prefix"].str.rstrip("-")
            names = gdf["name"]
            if "name_en" in gdf:
                names = gdf["name_en"].fillna(names)
        else:
            ids = [_tile_id(g.bounds) for g in gdf.geometry]
            names = ["%s tile %s" % (kind, i) for i in ids]
        return _frame("Movisda", ids, names, urls, gdf["bytes"], gdf.geometry)
    except Exception as e:
        raise ValueError("could not read the Movisda %s index (%s)" % (kind, e)) from e


def _movisda_extracts(directory, update=False):
    frames = []
    for kind, rel in _MOVISDA_INDEXES.items():
        path = _cached_index(
            "%s/%s" % (_MOVISDA_URL, rel),
            Path(directory) / "movisda" / Path(rel).name,
            update,
            check=lambda data, kind=kind: _check_movisda_index(data, kind)
            or _movisda_frame(io.BytesIO(data), kind),
        )
        key = (str(path), path.stat().st_mtime_ns)
        cached = _movisda_cache.get(kind)
        if cached is not None and cached[0] == key:
            frame = cached[1]
        else:
            frame = _movisda_frame(path, kind)
            _movisda_cache[kind] = (key, frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _size_record(directory, url):
    """The file caching the size of ``url``: one small JSON ``[bytes, checked]`` per URL."""
    name = hashlib.sha1(url.encode()).hexdigest() + ".json"
    return Path(directory) / "extract_sizes" / name


def _cached_size(record, now):
    """Return the size stored in ``record`` if it is valid and younger than a week."""
    try:
        size, checked = json.loads(record.read_text())
        age = now - float(checked)
        if type(size) is int and 0 <= size < 2**63 and 0 <= age < _SIZE_MAX_AGE:
            return size
    except (OSError, ValueError, TypeError, OverflowError):
        pass
    return None


def _download_sizes(urls, directory, update=False):
    """Return ``{url: bytes}`` for ``urls``, from a week-long cache or a HEAD request.

    URLs whose size cannot be read (the server is down or sends no valid ``Content-Length``) are
    left out of the result, with a warning.
    """
    sizes = {}
    for url in urls:
        record = _size_record(directory, url)
        cached = None if update else _cached_size(record, time.time())
        if cached is not None:
            sizes[url] = cached
            continue
        try:
            with open_url(url, method="HEAD", timeout=_TIMEOUT) as response:
                length = response.headers.get("Content-Length")
        except _FETCH_ERRORS as e:
            warnings.warn(
                "Could not get the size of %s (%s); it is ranked last." % (url, e)
            )
            continue
        try:
            digits = length and length.isascii() and length.isdigit()
            size = int(length) if digits else -1
        except ValueError:
            size = -1
        if not 0 <= size < 2**63:
            warnings.warn("%s sends no valid size; it is ranked last." % url)
            continue
        sizes[url] = size
        payload = json.dumps([sizes[url], time.time()]).encode()
        write_atomic(record, lambda out_file: out_file.write(payload))
    return sizes


def _covering_extracts(area, directory=None, update=False):
    """Return the extracts that fully contain ``area``, smallest download first.

    ``area`` is a Shapely geometry in lon/lat. The result is a GeoDataFrame with the columns
    ``provider``, ``id``, ``name``, ``url``, ``bytes`` and ``geometry``. Extracts whose size cannot
    be read have no ``bytes`` and come last, smallest area first. Movisda's extracts are left out
    when its index cannot be fetched.
    """
    directory = Path(directory) if directory is not None else download_dir()
    frames = [_geofabrik_extracts(), _bbbike_extracts()]
    try:
        frames.append(_movisda_extracts(directory, update))
    except (*_FETCH_ERRORS, ValueError) as e:
        warnings.warn("Movisda's extract index is unavailable (%s); skipping it." % e)
    candidates = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    covering = candidates[candidates.covers(area)].copy()
    unknown = covering["bytes"].isna()
    sizes = _download_sizes(covering.loc[unknown, "url"], directory, update)
    covering["bytes"] = pd.array(
        [
            sizes.get(url) if pd.isna(size) else int(size)
            for size, url in zip(covering["bytes"], covering["url"])
        ],
        dtype="Int64",
    )
    covering["_area"] = covering.geometry.to_crs(_EQUAL_AREA_CRS).area
    ranked = covering.sort_values(
        ["bytes", "_area", "provider", "id"], na_position="last", kind="stable"
    )
    return ranked[_COLUMNS].reset_index(drop=True)
