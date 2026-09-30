"""Find and download the smallest single OSM extract that contains an area.

Public entry points: :func:`find_extracts` lists the candidates, :func:`get_data_by_area`
downloads the best one. Candidates come from three providers:

- Geofabrik, from the vendored ``geofabrik_index.geojson.gz``;
- BBBike city extracts, from the vendored ``bbbike_index.geojson.gz``;
- Movisda administrative areas and 1°/10° grid tiles, from the index files published at
  https://osm.download.movisda.io (fetched and cached next to the downloads; when the grid
  index cannot be fetched and nothing is cached, the copy vendored as
  ``movisda_grid_index.geojson.gz`` is used).

``get_data_by_area`` never merges extracts, so its answer is always one file. Candidates are
ranked by download size: Movisda's index lists sizes; for the others a HEAD request asks the
server, and the answer is cached for a week. Refresh the vendored snapshots with
``scripts/update_extract_indexes.py``.
"""

import gzip
import hashlib
import io
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from pyrosm.data.geofabrik_index import (
    _EQUAL_AREA_CRS,
    _bbox_filename,
    _bbox_to_polygon,
    _crop,
)
from pyrosm.data.geofabrik_index import _load_index as _load_geofabrik_index
from pyrosm.exceptions import (
    DownloadError,
    ExtractDownloadError,
    ExtractNotFoundError,
)
from pyrosm.utils.download import (
    _FETCH_ERRORS,
    _TIMEOUT,
    _Net,
    _content_length,
    _retry,
    download_dir,
    write_atomic,
)

_BBBIKE_INDEX_PATH = Path(__file__).parent / "bbbike_index.geojson.gz"
_MOVISDA_GRID_PATH = Path(__file__).parent / "movisda_grid_index.geojson.gz"
_MOVISDA_URL = "https://osm.download.movisda.io"
_MOVISDA_INDEXES = {
    "admin": "admin/Admin-latest.geojson",
    "grid": "grid/grid-latest.geojson",
}
_INDEX_MAX_AGE = 24 * 3600
_SIZE_MAX_AGE = 7 * 24 * 3600
_COLUMNS = ["provider", "id", "name", "url", "bytes", "contains", "geometry"]

logger = logging.getLogger(__name__)

_bbbike_cache = None
_vendored_sizes_cache = None
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


def _geofabrik_extracts(update=False):
    gdf = _load_geofabrik_index(update)
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


def _cached_index(url, path, update=False, check=None, net=None):
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
    net = net or _Net()

    def fetch():
        with net.open(url, headers=headers) as response:
            return response.read(), response.headers.get("ETag") or ""

    try:
        data, new_etag = _retry(fetch)
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


def _movisda_index_frame(kind, path):
    """The candidate rows of the Movisda index at ``path``, parsed once while the file is
    unchanged. A ``.gz`` file is pyrosm's vendored copy; its date is in ``attrs``."""
    key = (str(path), path.stat().st_mtime_ns)
    cached = _movisda_cache.get(kind)
    if cached is not None and cached[0] == key:
        return cached[1]
    if path.suffix == ".gz":
        data = gzip.decompress(path.read_bytes())
        frame = _movisda_frame(io.BytesIO(data), kind)
        frame.attrs["snapshot_date"] = json.loads(data).get("movisda_snapshot_date")
    else:
        frame = _movisda_frame(path, kind)
    _movisda_cache[kind] = (key, frame)
    return frame


def _movisda_extracts(directory, update=False, net=None):
    """Movisda's administrative areas and grid tiles as candidate rows.

    When an index cannot be fetched and no copy is cached, the administrative areas are left
    out and the grid tiles come from pyrosm's vendored copy, each with a warning.
    """
    frames = []
    for kind, rel in _MOVISDA_INDEXES.items():
        try:
            path = _cached_index(
                "%s/%s" % (_MOVISDA_URL, rel),
                Path(directory) / "movisda" / Path(rel).name,
                update,
                check=lambda data, kind=kind: _check_movisda_index(data, kind)
                or _movisda_frame(io.BytesIO(data), kind),
                net=net,
            )
            frames.append(_movisda_index_frame(kind, path))
        except (*_FETCH_ERRORS, ValueError) as e:
            if kind == "admin":
                warnings.warn(
                    "Movisda's administrative index is unavailable (%s); skipping its "
                    "administrative areas." % e
                )
                continue
            frame = _movisda_index_frame(kind, _MOVISDA_GRID_PATH)
            warnings.warn(
                "Movisda's grid index is unavailable (%s); using pyrosm's copy of it "
                "from %s." % (e, frame.attrs["snapshot_date"])
            )
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


def _vendored_geofabrik_sizes():
    """``{pbf URL: (bytes, day read)}`` recorded in pyrosm's Geofabrik index snapshot (read
    once)."""
    global _vendored_sizes_cache
    if _vendored_sizes_cache is not None:
        return _vendored_sizes_cache
    gdf = _load_geofabrik_index(False).reindex(columns=["pbf", "bytes", "bytes_date"])
    _vendored_sizes_cache = {
        url: (int(size), day)
        for url, size, day in zip(gdf["pbf"], gdf["bytes"], gdf["bytes_date"])
        if isinstance(url, str) and pd.notna(size) and 0 <= size < 2**63
    }
    return _vendored_sizes_cache


def _download_sizes(urls, directory, update=False, net=None, fallback=None):
    """Return ``{url: bytes}`` for ``urls``, from a week-long cache or a HEAD request.

    When a URL's size cannot be read (the server is down, loops, or sends no valid
    ``Content-Length``), its ``(bytes, day)`` in ``fallback`` is used, with a warning; without
    one the URL is left out of the result, with a warning. Only sizes read from the server are
    cached.
    """
    sizes = {}
    net = net or _Net()
    fallback = fallback or {}

    def unreadable(url, reason):
        recorded = fallback.get(url)
        if recorded is None:
            warnings.warn(
                "Could not get the size of %s (%s); it is ranked last." % (url, reason)
            )
            return
        sizes[url] = recorded[0]
        warnings.warn(
            "Could not get the size of %s (%s); using the size recorded in pyrosm's "
            "Geofabrik index on %s." % (url, reason, recorded[1])
        )

    for url in urls:
        record = _size_record(directory, url)
        cached = None if update else _cached_size(record, time.time())
        if cached is not None:
            sizes[url] = cached
            continue

        def head(url=url):
            with net.open(url, method="HEAD") as response:
                return response.headers.get("Content-Length")

        try:
            length = _retry(head)
        except _FETCH_ERRORS as e:
            unreadable(url, e)
            continue
        size = _content_length(length)
        if size is None:
            unreadable(url, "no valid Content-Length")
            continue
        sizes[url] = size
        payload = json.dumps([sizes[url], time.time()]).encode()
        write_atomic(record, lambda out_file: out_file.write(payload))
    return sizes


def find_extracts(
    area,
    contains_only=False,
    update=False,
    directory=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
):
    """List the OSM extracts that overlap ``area``, best download first, without downloading.

    Compares Geofabrik extracts, BBBike city extracts and Movisda administrative areas and
    1°/10° grid tiles. Extracts that contain the whole area come first, then those that only
    overlap it; within each group the smallest download comes first. Extracts whose size cannot
    be read come last in their group, smallest extent first. :func:`get_data_by_area` downloads
    the first extract that contains the area.

    When Movisda's indexes cannot be fetched and no copy is cached, its administrative areas are
    left out and its grid tiles come from the copy vendored with pyrosm, each with a warning.

    Parameters
    ----------
    area : shapely geometry | GeoDataFrame | GeoSeries | list | tuple | numpy.ndarray
        The area of interest in lon/lat: a (Multi)Polygon, a GeoDataFrame/GeoSeries (its
        geometries are combined) or ``[minx, miny, maxx, maxy]``.

    contains_only : bool
        When ``True``, list only the extracts that contain the whole area.

    update : bool
        When ``True``, refresh the provider indexes and download sizes.

    directory : str, optional
        Directory for the cached provider indexes and download sizes. ``None`` (default) uses
        the pyrosm temp directory, as :func:`get_data_by_area` does.

    headers : dict, optional
        Extra HTTP headers for the Movisda index and size requests, e.g. a ``User-Agent``
        (not for the refresh of the vendored Geofabrik index with ``update=True``).

    timeout : float | tuple
        Seconds to wait for a server, for the connect and each read, or a ``(connect, read)``
        pair. Default 60.

    opener : object, optional
        An object with ``open(request, timeout=...)``, e.g. from
        ``urllib.request.build_opener()``, that makes the requests instead of pyrosm.

    Returns
    -------
    GeoDataFrame
        One row per extract with the columns ``provider`` (``"Geofabrik"``, ``"BBBike"`` or
        ``"Movisda"``), ``id``, ``name``, ``url``, ``bytes`` (the download size, ``<NA>`` when
        it cannot be read), ``contains`` (whether the extract contains the whole area) and
        ``geometry`` (the extract's extent, EPSG:4326).

    Raises
    ------
    ValueError
        If the area is empty, or has no width or no height.
    """
    area = _area_geometry(area)
    directory = Path(directory) if directory is not None else download_dir()
    net = _Net(headers, timeout, opener)
    frames = [_geofabrik_extracts(update), _bbbike_extracts()]
    try:
        frames.append(_movisda_extracts(directory, update, net))
    except (*_FETCH_ERRORS, ValueError) as e:
        warnings.warn("Movisda's extract index is unavailable (%s); skipping it." % e)
    candidates = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    candidates["contains"] = candidates.covers(area)
    if contains_only:
        found = candidates[candidates["contains"]].copy()
    else:
        overlaps = candidates.intersects(area) & ~candidates.touches(area)
        found = candidates[overlaps].copy()
    unknown = found["bytes"].isna()
    sizes = _download_sizes(
        found.loc[unknown, "url"], directory, update, net, _vendored_geofabrik_sizes()
    )
    found["bytes"] = pd.array(
        [
            sizes.get(url) if pd.isna(size) else int(size)
            for size, url in zip(found["bytes"], found["url"])
        ],
        dtype="Int64",
    )
    found["_area"] = found.geometry.to_crs(_EQUAL_AREA_CRS).area
    ranked = found.sort_values(
        ["contains", "bytes", "_area", "provider", "id"],
        ascending=[False, True, True, True, True],
        na_position="last",
        kind="stable",
    )
    return ranked[_COLUMNS].reset_index(drop=True)


@dataclass
class AreaExtract:
    """The file :func:`get_data_by_area` wrote and the extract it came from.

    It can be used as a path, e.g. ``OSM(get_data_by_area(area))``.

    Attributes
    ----------
    path : str
        The cropped file (or the full extract with ``crop=False``).
    provider : str
        ``"Geofabrik"``, ``"BBBike"`` or ``"Movisda"``.
    extract : str
        The extract's id at the provider, e.g. ``"finland"``, ``"Basel"``, ``"NL-NB"`` or a grid
        tile such as ``"N60E024"`` (south-west corner; ``"-10"`` marks a 10° tile).
    url : str
        The extract's download URL.
    bytes : int or None
        The extract's download size, if the provider reported it.
    download_seconds, crop_seconds : float
        Time spent downloading (near zero when the extract was already downloaded) and cropping.
    failed : list of (str, str)
        ``(url, error message)`` for smaller extracts whose download failed before this one.
    """

    path: str
    provider: str
    extract: str
    url: str
    bytes: object
    download_seconds: float
    crop_seconds: float
    failed: list = field(default_factory=list)

    def __fspath__(self):
        return self.path


def _area_geometry(area):
    """Return ``area`` as one Shapely geometry in lon/lat.

    Accepts a Shapely geometry, a GeoDataFrame/GeoSeries (its geometries are unioned, after
    reprojecting to EPSG:4326 when it has another CRS) or ``[minx, miny, maxx, maxy]``.
    """
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if area.crs is not None:
            area = area.to_crs("EPSG:4326")
        geom = shapely.union_all(area.geometry.values)
    elif isinstance(area, BaseGeometry):
        geom = area
    else:
        geom = _bbox_to_polygon(area)
    if geom.is_empty:
        raise ValueError("The area is empty.")
    minx, miny, maxx, maxy = _bbox_to_polygon(geom).bounds
    if minx == maxx or miny == maxy:
        raise ValueError(
            "The area must have a width and a height; got %s." % geom.geom_type
        )
    return geom


def get_data_by_area(
    area,
    crop=True,
    update=False,
    directory=None,
    output_path=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
):
    """Download the smallest single OSM extract that contains ``area``.

    Compares Geofabrik extracts, BBBike city extracts and Movisda administrative areas and 1°/10°
    grid tiles, keeps those that contain the whole area, and downloads the one with the smallest
    file (the first row of ``find_extracts(area, contains_only=True)``). Extracts are never
    merged. A download that fails with a network error or HTTP status 408, 425, 429 or 5xx is
    tried up to three times, waiting 1 s, 2 s or the server's ``Retry-After`` in between; when it
    still fails, the next smallest extract is tried. By default the extract is then cropped to
    the area's bounding box.

    Movisda cuts its extracts exactly at their edges, so features crossing the edge of a Movisda
    extract are clipped or missing there. The extract contains the whole area, so this only
    affects the part of the bounding box outside the area.

    Parameters
    ----------
    area : shapely geometry | GeoDataFrame | GeoSeries | list | tuple | numpy.ndarray
        The area of interest in lon/lat: a (Multi)Polygon, a GeoDataFrame/GeoSeries (its
        geometries are combined) or ``[minx, miny, maxx, maxy]``.

    crop : bool
        When ``True`` (default), crop the extract to the area's bounding box and return the
        cropped file, named ``bbox_<minx>_<miny>_<maxx>_<maxy>.osm.pbf``. When ``False``, return
        the full extract.

    update : bool
        When ``True``, re-download the extract and refresh the provider indexes and sizes.

    directory : str, optional
        Directory for the downloads, the provider indexes and the cropped file. ``None``
        (default) uses a pyrosm temp directory.

    output_path : str, optional
        Path for the cropped file when ``crop=True`` (overrides the automatic name).

    headers, timeout, opener
        Network options for the Movisda index, size and download requests, as for
        :func:`find_extracts`. The refresh of the vendored Geofabrik index (``update=True``)
        does not use them.

    Returns
    -------
    AreaExtract
        The file path and the extract it came from; usable as a path.

    Raises
    ------
    ValueError
        If the area is empty, or has no width or no height.
    pyrosm.exceptions.ExtractNotFoundError
        If no extract contains the whole area (a ``ValueError`` subclass).
    pyrosm.exceptions.ExtractDownloadError
        If every extract that contains the area failed to download; its ``errors`` hold the
        :class:`~pyrosm.exceptions.DownloadError` of each extract tried.
    """
    from pyrosm.utils.download import download as _download_file

    geom = _area_geometry(area)
    net = dict(headers=headers, timeout=timeout, opener=opener)
    candidates = find_extracts(
        geom, contains_only=True, update=update, directory=directory, **net
    )
    if candidates.empty:
        raise ExtractNotFoundError(
            "No Geofabrik, BBBike or Movisda extract contains the whole area."
        )
    failed, errors = [], []
    for extract in candidates.itertuples():
        size = (
            "unknown size"
            if pd.isna(extract.bytes)
            else "%.1f MB" % (extract.bytes / 1e6)
        )
        logger.info(
            "Smallest extract containing the area: %s '%s' (%s)",
            extract.provider,
            extract.name,
            size,
        )
        filename = "%s_%s" % (extract.provider.lower(), Path(extract.url).name)
        start = time.perf_counter()
        try:
            full_path = _download_file(extract.url, filename, update, directory, **net)
        except DownloadError as e:
            failed.append((extract.url, str(e)))
            errors.append(e)
            warnings.warn(
                "Could not download %s (%s); trying the next smallest extract."
                % (extract.url, e)
            )
            continue
        download_seconds = time.perf_counter() - start
        start = time.perf_counter()
        path = full_path
        if crop:
            envelope = box(*geom.bounds)
            path = _crop(
                full_path,
                envelope,
                _bbox_filename(envelope.bounds),
                output_path,
                directory,
            )
        return AreaExtract(
            path=path,
            provider=extract.provider,
            extract=extract.id,
            url=extract.url,
            bytes=None if pd.isna(extract.bytes) else int(extract.bytes),
            download_seconds=download_seconds,
            crop_seconds=time.perf_counter() - start,
            failed=failed,
        )
    raise ExtractDownloadError(
        "Could not download any of the %d extracts that contain the area: %s"
        % (len(failed), "; ".join("%s (%s)" % f for f in failed)),
        failed,
        errors=errors,
    )
