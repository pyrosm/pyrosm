"""Find and download the OSM extract(s) for an area: the smallest single extract that
contains it, or a smaller set of extracts merged into one file.

Public entry points: :func:`find_extracts` lists the candidates, :func:`get_data_by_area`
downloads the best one. Candidates come from three providers:

- Geofabrik, from the vendored ``geofabrik_index.geojson.gz``;
- BBBike city extracts, from the vendored ``bbbike_index.geojson.gz``;
- Movisda administrative areas and 1°/10° grid tiles, from the index files published at
  https://osm.download.movisda.io (fetched and cached next to the downloads; when the grid
  index cannot be fetched and nothing is cached, the copy vendored as
  ``movisda_grid_index.geojson.gz`` is used).

``get_data_by_area`` returns one extract, or with ``strategy="smallest_total"`` a smaller set of
extracts merged into one file. Candidates are ranked by download size: Movisda's index lists sizes; for the others a HEAD request asks the
server, and the answer is cached for a week. Refresh the vendored snapshots with
``scripts/update_extract_indexes.py``.
"""

import gzip
import hashlib
import io
import json
import logging
import numbers
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    _default_target_dir,
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
_STRATEGIES = ("single", "smallest_total")
# Uncovered area (m²) below which a set of extracts counts as covering the area; provider
# outlines are simplified, so their edges leave slivers.
_SLIVER_M2 = 1.0
# Radius (m) that gives the points and lines of ``must_cover`` an area for the greedy steps.
_POINT_RADIUS_M = 1.0

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
    must_cover=None,
):
    """List the OSM extracts that overlap ``area``, best download first, without downloading.

    Compares Geofabrik extracts, BBBike city extracts and Movisda administrative areas and
    1°/10° grid tiles. Extracts that contain the whole area (or ``must_cover`` when given) come
    first, then those that only overlap it; within each group the smallest download comes
    first. Extracts whose size cannot be read come last in their group, smallest extent first.
    With its default ``strategy="single"``, :func:`get_data_by_area` downloads the first
    extract that contains the area (or ``must_cover``).

    When Movisda's indexes cannot be fetched and no copy is cached, its administrative areas are
    left out and its grid tiles come from the copy vendored with pyrosm, each with a warning.

    Parameters
    ----------
    area : shapely geometry | GeoDataFrame | GeoSeries | list | tuple | numpy.ndarray
        The area of interest in lon/lat: a (Multi)Polygon, a GeoDataFrame/GeoSeries (its
        geometries are combined) or ``[minx, miny, maxx, maxy]``.

    contains_only : bool
        When ``True``, list only the extracts that contain the whole area (or ``must_cover``).

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

    must_cover : shapely geometry | GeoDataFrame | GeoSeries, optional
        What an extract must contain, in place of the whole area, e.g. the transit stops that
        routing needs. Any geometry type; points and lines must be contained exactly. Extracts
        that only reach it (not the area) are listed too. To also require the streets around
        each stop, pass the stops buffered in a local metric CRS, e.g.
        ``stops.to_crs(stops.estimate_utm_crs()).buffer(300)``, or a hull of them to require
        the streets between stops as well. A GeoDataFrame or GeoSeries in another CRS is
        reprojected to lon/lat.

    Returns
    -------
    GeoDataFrame
        One row per extract with the columns ``provider`` (``"Geofabrik"``, ``"BBBike"`` or
        ``"Movisda"``), ``id``, ``name``, ``url``, ``bytes`` (the download size, ``<NA>`` when
        it cannot be read), ``contains`` (whether the extract contains the whole area, or
        ``must_cover`` when given) and ``geometry`` (the extract's extent, EPSG:4326).

    Raises
    ------
    ValueError
        If the area is empty, or has no width or no height, or ``must_cover`` is empty.
    """
    area = _area_geometry(area)
    parts = None if must_cover is None else _must_cover_parts(must_cover)
    directory = Path(directory) if directory is not None else download_dir()
    net = _Net(headers, timeout, opener)
    frames = [_geofabrik_extracts(update), _bbbike_extracts()]
    try:
        frames.append(_movisda_extracts(directory, update, net))
    except (*_FETCH_ERRORS, ValueError) as e:
        warnings.warn("Movisda's extract index is unavailable (%s); skipping it." % e)
    candidates = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if parts is None:
        candidates["contains"] = candidates.covers(area)
    else:
        candidates["contains"] = True
        for part in parts:
            if part is not None:
                candidates["contains"] &= candidates.covers(part)
    if contains_only:
        found = candidates[candidates["contains"]].copy()
    else:
        overlaps = candidates.intersects(area) & ~candidates.touches(area)
        for part in parts or ():
            if part is not None:
                overlaps |= candidates.intersects(part)
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


def _covers(chosen, shapes, outlines, polygons, exact):
    """Whether the extracts ``chosen`` cover the polygon part (all but a sliver, measured on
    the projected ``shapes``) and the point and line part (exactly, on the lon/lat
    ``outlines``) of a coverage target."""
    if not chosen:
        return False
    if polygons is not None:
        left = polygons.difference(shapely.union_all(shapes.loc[chosen]))
        if left.area >= _SLIVER_M2:
            return False
    return exact is None or shapely.union_all(outlines.loc[chosen]).covers(exact)


def _prune(chosen, sizes, covered):
    """Drop extracts from ``chosen``, largest first, while ``covered(rest)`` holds."""
    for i in sorted(chosen, key=lambda i: -sizes[i]):
        rest = [j for j in chosen if j != i]
        if covered(rest):
            chosen = rest
    return chosen


def _greedy_cover(target, shapes, sizes, chosen, pool):
    """Extend ``chosen`` with extracts from ``pool`` while they leave a sliver or more of
    ``target`` uncovered, each time taking the lowest ``bytes`` per area newly covered; stop
    when no extract adds area. ``None`` when nothing at all was chosen."""
    chosen, pool = list(chosen), list(pool)
    remaining = target.difference(shapely.union_all(shapes.loc[chosen]))
    # At least one extract, even for an area smaller than a sliver.
    while not chosen or remaining.area >= _SLIVER_M2:
        gains = {i: shapes[i].intersection(remaining).area for i in pool}
        useful = [i for i in pool if gains[i] > 0]
        if not useful:
            break
        pick = min(useful, key=lambda i: (sizes[i] / gains[i], sizes[i], i))
        chosen.append(pick)
        pool.remove(pick)
        remaining = remaining.difference(shapes[pick])
    return chosen or None


def _repair(chosen, pool, sizes, outlines, exact):
    """Add the cheapest extracts from ``pool`` until ``chosen`` covers the points and lines
    ``exact`` exactly (lon/lat ``outlines`` and ``exact``, so points on an edge keep their
    ``covers`` meaning); ``None`` when no extract holds what is left."""
    chosen, pool = list(chosen), list(pool)
    while exact is not None and not shapely.union_all(outlines.loc[chosen]).covers(
        exact
    ):
        left = exact.difference(shapely.union_all(outlines.loc[chosen]))
        options = [i for i in pool if i not in chosen and outlines[i].intersects(left)]
        if not options:
            return None
        chosen.append(min(options, key=lambda i: (sizes[i], i)))
    return chosen


def _smallest_cover(target, candidates):
    """The extracts that together cover ``target`` with a small total download, in merge
    order, or ``None`` when they cannot cover it.

    ``target`` is ``(polygons, points_and_lines)`` in lon/lat, either part ``None``: polygons
    count as covered when less than 1 m² of them is left; points and lines must be covered
    exactly (for the greedy steps they are buffered by 1 m, which only guides the choice).
    Candidates are rows of :func:`find_extracts`, in its ranking order. Geofabrik and BBBike
    extracts and Movisda administrative extracts with a known size may be used, at most one of
    them from Movisda; Movisda grid tiles never are (they drop closed ways at tile edges). The
    set is found greedily (lowest bytes per area newly covered), once without Movisda and once
    starting from each Movisda extract, then pruned; the cheapest result wins. It is a good
    set, not necessarily the smallest possible. Movisda comes last in the merge order, so a
    complete copy of a way that crosses its border wins :func:`pyrosm.merge_pbf`'s ties.
    """
    known = candidates[candidates["bytes"].notna()]
    movisda = known["provider"] == "Movisda"
    admin = movisda & known["url"].str.contains("/admin/", regex=False)
    known = known[~movisda | admin]

    def project(geom):
        if geom is None:
            return None
        return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(_EQUAL_AREA_CRS).iloc[0]

    polygons, exact = project(target[0]), target[1]
    guide = [polygons] if polygons is not None else []
    if exact is not None:
        guide.append(project(exact).buffer(_POINT_RADIUS_M))
    guide = shapely.union_all(guide)
    # Projected and clipped to the target once, so the greedy steps work on small shapes;
    # the lon/lat outlines decide exact coverage; sizes stay exact Python integers.
    outlines = known.geometry
    shapes = outlines.to_crs(_EQUAL_AREA_CRS).intersection(guide)
    sizes = {i: int(size) for i, size in known["bytes"].items()}
    others = list(known.index[known["provider"] != "Movisda"])
    best = None
    for seed in [None] + list(known.index[known["provider"] == "Movisda"]):
        start = [] if seed is None else [seed]
        chosen = _greedy_cover(guide, shapes, sizes, start, others)
        if chosen is not None:
            chosen = _repair(chosen, others, sizes, outlines, exact)
        if chosen is None or not _covers(chosen, shapes, outlines, polygons, exact):
            continue
        chosen = _prune(
            chosen,
            sizes,
            lambda rest: _covers(rest, shapes, outlines, polygons, exact),
        )
        key = (sum(sizes[i] for i in chosen), len(chosen))
        if best is None or key < best[0]:
            best = (key, chosen)
    if best is None:
        return None
    order = sorted(best[1], key=lambda i: (known.at[i, "provider"] == "Movisda", i))
    return known.loc[order]


def _choose(target, candidates, strategy):
    """The extract(s) to download, or ``None``: the first extract that contains ``target``
    (see :func:`_smallest_cover`); with ``strategy="smallest_total"`` a set of extracts when
    its total is smaller than that extract, or when that extract's size is unknown."""
    single = candidates[candidates["contains"]].iloc[:1]
    if strategy == "single":
        return single if len(single) else None
    cover = _smallest_cover(target, candidates)
    if cover is None:
        return single if len(single) else None
    size = single["bytes"].iloc[0] if len(single) else pd.NA
    if pd.notna(size) and int(size) <= sum(int(b) for b in cover["bytes"]):
        return single
    return cover


def _identity(path):
    """A file's device, inode, size and modification time, to notice it changing."""
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _provenance(path):
    """``(sha256, snapshot)`` of a file: its SHA-256 hex digest and the PBF header's
    ``osmosis_replication_timestamp`` as a UTC datetime (``None`` when it has none).

    Both are read within one check of the file's identity (device, inode, size, mtime): when
    it changed while they were read, they are read again, up to three times, then ``OSError``.
    This is best effort against another process changing the file; pyrosm itself replaces its
    downloads and merges atomically.
    """
    from pyrosm.pbf_export import read_header_block

    for _ in range(3):
        before = _identity(path)
        stamp = read_header_block(str(path)).osmosis_replication_timestamp
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        if _identity(path) == before:
            snapshot = datetime.fromtimestamp(stamp, timezone.utc) if stamp else None
            return digest.hexdigest(), snapshot
    raise OSError("'%s' kept changing while it was read." % path)


@dataclass
class ExtractSource:
    """One downloaded extract that went into an :class:`AreaExtract`: where it came from, the
    downloaded file, its SHA-256 and its snapshot time (``None`` when the file has none).
    """

    provider: str
    extract: str
    url: str
    bytes: object
    path: str
    sha256: str = None
    snapshot: object = None


@dataclass
class AreaExtract:
    """The file :func:`get_data_by_area` wrote and the extract(s) it came from.

    It can be used as a path, e.g. ``OSM(get_data_by_area(area))``.

    Attributes
    ----------
    path : str
        The cropped file; with ``crop=False`` the full extract, or the merged extracts.
    provider : str
        ``"Geofabrik"``, ``"BBBike"`` or ``"Movisda"``; for merged extracts their providers
        joined with ``"+"`` in merge order.
    extract : str
        The extract's id at the provider, e.g. ``"finland"``, ``"Basel"``, ``"NL-NB"`` or a grid
        tile such as ``"N60E024"`` (south-west corner; ``"-10"`` marks a 10° tile); for merged
        extracts their ids joined with ``"+"``.
    url : str or None
        The extract's download URL; ``None`` for merged extracts (see ``sources``).
    bytes : int or None
        The extract's download size, if the provider reported it.
    download_seconds, crop_seconds : float
        Time spent downloading (near zero when the extract was already downloaded) and cropping.
    failed : list of (str, str)
        ``(url, error message)`` for smaller extracts whose download failed before this one.
    sources : list of ExtractSource
        The downloaded extracts, in merge order: one for a single extract, several for a merged
        one. For a merged file ``provider`` and ``extract`` join theirs with ``"+"``, ``url``
        is ``None`` and ``bytes`` is their total; ``crop_seconds`` then covers the merge.
    sha256 : str
        The SHA-256 hex digest of ``path``.
    snapshot : datetime or None
        The time the extract's data was taken, from its PBF header
        (``osmosis_replication_timestamp``, UTC); ``None`` when it has none. For merged
        extracts the oldest of the sources' times (each source's own is in ``sources``).
    """

    path: str
    provider: str
    extract: str
    url: str
    bytes: object
    download_seconds: float
    crop_seconds: float
    failed: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    sha256: str = None
    snapshot: object = None

    def __fspath__(self):
        return self.path


def _must_cover_parts(must_cover):
    """``must_cover`` as ``(polygons, points_and_lines)`` in lon/lat, either part ``None``.

    Accepts a Shapely geometry of any type or a GeoDataFrame/GeoSeries (reprojected to
    EPSG:4326 when it has another CRS). The two kinds are united separately, so a point inside
    a polygon keeps its own, exact coverage requirement. Raises ``ValueError`` when it is empty.
    """
    if isinstance(must_cover, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if must_cover.crs is not None:
            must_cover = must_cover.to_crs("EPSG:4326")
        geoms = list(must_cover.geometry.values)
    else:
        geoms = [must_cover]
    parts = list(shapely.get_parts(geoms))
    # Collections can hold multi-part geometries and further collections.
    while any(
        shapely.get_num_geometries(g) > 1 or g.geom_type.startswith(("Multi", "Geo"))
        for g in parts
    ):
        parts = list(shapely.get_parts(parts))
    parts = [g for g in parts if g is not None and not g.is_empty]
    polygons = [g for g in parts if g.geom_type == "Polygon"]
    others = [g for g in parts if g.geom_type != "Polygon"]
    if not parts:
        raise ValueError("must_cover is empty.")
    return (
        shapely.union_all(polygons) if polygons else None,
        shapely.union_all(others) if others else None,
    )


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


def _write_area_file(area, sources, crop, output_path, directory, workers=1):
    """The file for :class:`AreaExtract`: the one extract, cropped to the area's bounding box
    when ``crop`` (to the area itself when ``crop="polygon"``); several extracts merged (and
    cropped) with :func:`pyrosm.merge_pbf`.
    """
    from pyrosm.pbf_export import crop_pbf, merge_pbf

    paths = [s.path for s in sources]
    if not crop and len(paths) == 1:
        return paths[0]
    region = {}
    if crop == "polygon":
        region["polygon"] = area
        name = "area_%s.osm.pbf" % hashlib.sha1(area.wkb).hexdigest()[:12]
    elif crop:
        region["bounding_box"] = box(*area.bounds)
        name = _bbox_filename(area.bounds)
    else:
        urls = "\n".join(s.url for s in sources).encode()
        name = "merged_%s.osm.pbf" % hashlib.sha1(urls).hexdigest()[:12]
    target = output_path or str(_default_target_dir(directory) / name)
    Path(target).resolve().parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        return crop_pbf(paths[0], target, workers=workers, **region)
    return merge_pbf(paths, target, workers=workers, **region)


def _write_with_provenance(area, sources, crop, output_path, directory, workers=1):
    """Write the area's file (:func:`_write_area_file`) and read the provenance of it and of
    its sources afterwards, so they describe the bytes that were used. When a source changed
    meanwhile, or the written file changed before its provenance was read, both are done
    again, up to three times, then ``OSError``. Fills in each source's
    ``sha256`` and ``snapshot``; returns the file's path and ``{path: (sha256, snapshot)}``.
    """
    for _ in range(3):
        before = [_identity(s.path) for s in sources]
        path = _write_area_file(area, sources, crop, output_path, directory, workers)
        written = _identity(path)
        provenance = {p: _provenance(p) for p in {path, *(s.path for s in sources)}}
        unchanged = [_identity(s.path) for s in sources] == before
        if unchanged and _identity(path) == written:
            for s in sources:
                s.sha256, s.snapshot = provenance[s.path]
            return path, provenance
    raise OSError(
        "The downloaded extracts kept changing while the area's file was written."
    )


def get_data_by_area(
    area,
    crop=True,
    update=False,
    directory=None,
    output_path=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
    strategy="single",
    must_cover=None,
    workers=1,
):
    """Download the OSM data for ``area``: the smallest single extract that contains it, or a
    smaller set of extracts merged into one file.

    Compares Geofabrik extracts, BBBike city extracts and Movisda administrative areas and 1°/10°
    grid tiles, keeps those that contain the whole area (or ``must_cover`` when given), and
    downloads the one with the smallest
    file (the first row of ``find_extracts(area, contains_only=True, must_cover=must_cover)``,
    which shows the choice without downloading). With
    ``strategy="smallest_total"`` it may instead download several extracts that together cover
    the area and merge them into one file (see ``strategy``). A download that fails with a network error or HTTP status 408, 425, 429 or 5xx is
    tried up to three times, waiting 1 s, 2 s or the server's ``Retry-After`` in between; when it
    still fails, the choice is made again without it. By default the file is then cropped to
    the area's bounding box, or with ``crop="polygon"`` to the area itself.

    Movisda cuts its extracts exactly at their edges, so features crossing the edge of a Movisda
    extract are clipped or missing there: open ways such as streets are cut at the edge and
    kept. A grid tile can leave out a whole closed way that crosses its edge, such as a
    building or a land-use area, including its part inside the area. To keep a grid tile's
    edges at least N metres outside the area, pass the area buffered by N metres (in a local
    metric CRS) as ``must_cover``. In a merged set the Movisda extract goes last, so where
    another extract holds a complete copy of such a feature, that copy is kept.

    Parameters
    ----------
    area : shapely geometry | GeoDataFrame | GeoSeries | list | tuple | numpy.ndarray
        The area of interest in lon/lat: a (Multi)Polygon, a GeoDataFrame/GeoSeries (its
        geometries are combined) or ``[minx, miny, maxx, maxy]``.

    crop : bool or "polygon"
        When ``True`` (default), crop the extract to the area's bounding box and return the
        cropped file, named ``bbox_<minx>_<miny>_<maxx>_<maxy>.osm.pbf``. With ``"polygon"``,
        crop it to the area itself, which must be a (Multi)Polygon: a node is kept when it
        lies inside the area or on its boundary, and a way that has a kept node is kept
        whole. The file is named ``area_<hash of the area>.osm.pbf``. When ``False``, return
        the full extract.

    update : bool
        When ``True``, re-download the extract and refresh the provider indexes and sizes.

    directory : str, optional
        Directory for the downloads, the provider indexes and the cropped file. ``None``
        (default) uses a pyrosm temp directory.

    output_path : str, optional
        Path for the file pyrosm writes (overrides the automatic name): the cropped file, or
        with ``crop=False`` the merged file of a ``"smallest_total"`` set.

    headers, timeout, opener
        Network options for the Movisda index, size and download requests, as for
        :func:`find_extracts`. The refresh of the vendored Geofabrik index (``update=True``)
        does not use them.

    strategy : str
        ``"single"`` (default): the smallest extract that contains the area.
        ``"smallest_total"``: when extracts that together cover the area are smaller in total
        than that extract (or its size is unknown), download those and merge them with
        :func:`pyrosm.merge_pbf`. The set is found greedily (lowest size per area newly
        covered), so it is small but not guaranteed to be the smallest possible. Geofabrik,
        BBBike and Movisda administrative extracts may be combined, at most one of them from
        Movisda, which goes last in the merge; Movisda grid tiles are never combined. With
        ``crop=False`` the merged file is named ``merged_<hash of the URLs>.osm.pbf``.

    must_cover : shapely geometry | GeoDataFrame | GeoSeries, optional
        What the download must cover, in place of the whole area, e.g. the transit stops that
        routing needs, so sea or unserved edges of the area do not force a larger extract.
        Points and lines are covered exactly, polygons except for less than 1 m². To also
        require the streets around each stop, pass the stops buffered in a local metric CRS,
        e.g. ``stops.to_crs(stops.estimate_utm_crs()).buffer(300)``, or a hull of them to
        require the streets between stops as well. A GeoDataFrame or GeoSeries in another CRS
        is reprojected to lon/lat. The crop still follows ``area``, so parts of
        ``must_cover`` outside it are cropped away.

    workers : int
        Number of worker processes that crop (and merge) the downloaded file. ``1`` (default)
        runs in one process. More workers are used only for a file with at least
        ``2 * workers`` data blocks, and the written file is the same for every value. With
        ``crop=False`` and a single extract nothing is cropped, so it has no effect. On macOS
        and Windows each worker process starts by importing the script, so a script that
        passes ``workers > 1`` needs the ``if __name__ == "__main__":`` guard; without it, or
        for a script read from stdin, the work runs in one process with a warning.

    Returns
    -------
    AreaExtract
        The file path and the extract it came from; usable as a path.

    Raises
    ------
    ValueError
        If the area is empty, or has no width or no height, or ``must_cover`` is empty, or
        ``crop`` is a string other than ``"polygon"``, or ``crop="polygon"`` and the area is
        not a (Multi)Polygon, or ``workers`` is not an integer of at least 1.
    pyrosm.exceptions.ExtractNotFoundError
        If no extract (or set of extracts) contains the whole area, or ``must_cover`` when
        given (a ``ValueError`` subclass).
    pyrosm.exceptions.ExtractDownloadError
        If every extract that contains the area failed to download; its ``errors`` hold the
        :class:`~pyrosm.exceptions.DownloadError` of each extract tried.
    """
    from pyrosm.utils.download import download as _download_file

    if strategy not in _STRATEGIES:
        raise ValueError(
            "strategy must be one of %s; got %r." % (", ".join(_STRATEGIES), strategy)
        )
    if isinstance(crop, str) and crop != "polygon":
        raise ValueError('crop must be True, False or "polygon"; got %r.' % (crop,))
    if (
        isinstance(workers, bool)
        or not isinstance(workers, numbers.Integral)
        or workers < 1
    ):
        raise ValueError(
            "workers must be an integer of at least 1; got %r." % (workers,)
        )
    geom = _area_geometry(area)
    if crop == "polygon" and geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            'crop="polygon" needs a Polygon or MultiPolygon area; got %s.'
            % geom.geom_type
        )
    target = (geom, None) if must_cover is None else _must_cover_parts(must_cover)
    net = dict(headers=headers, timeout=timeout, opener=opener)
    candidates = find_extracts(
        geom,
        contains_only=strategy == "single",
        update=update,
        directory=directory,
        must_cover=must_cover,
        **net,
    )
    failed, errors = [], []
    while (chosen := _choose(target, candidates, strategy)) is not None:
        sources = []
        start = time.perf_counter()
        for extract in chosen.itertuples():
            size = "unknown size"
            if pd.notna(extract.bytes):
                size = "%.1f MB" % (extract.bytes / 1e6)
            logger.info(
                "Downloading %s '%s' (%s)", extract.provider, extract.name, size
            )
            filename = "%s_%s" % (extract.provider.lower(), Path(extract.url).name)
            try:
                path = _download_file(extract.url, filename, update, directory, **net)
            except DownloadError as e:
                failed.append((extract.url, str(e)))
                errors.append(e)
                warnings.warn(
                    "Could not download %s (%s); choosing again without it."
                    % (extract.url, e)
                )
                candidates = candidates[candidates["url"] != extract.url]
                break
            bytes_ = None if pd.isna(extract.bytes) else int(extract.bytes)
            sources.append(
                ExtractSource(extract.provider, extract.id, extract.url, bytes_, path)
            )
        else:
            download_seconds = time.perf_counter() - start
            start = time.perf_counter()
            path, provenance = _write_with_provenance(
                geom, sources, crop, output_path, directory, workers
            )
            merged = len(sources) > 1
            total = [s.bytes for s in sources]
            sha256 = provenance[path][0]
            snapshots = [s.snapshot for s in sources if s.snapshot is not None]
            return AreaExtract(
                path=path,
                provider="+".join(s.provider for s in sources),
                extract="+".join(s.extract for s in sources),
                url=None if merged else sources[0].url,
                bytes=sum(total) if None not in total else None,
                download_seconds=download_seconds,
                crop_seconds=time.perf_counter() - start,
                failed=failed,
                sources=sources,
                sha256=sha256,
                snapshot=min(snapshots) if snapshots else None,
            )
    if not failed:
        what = "the whole area" if must_cover is None else "must_cover"
        raise ExtractNotFoundError(
            "No Geofabrik, BBBike or Movisda extract contains %s." % what
        )
    raise ExtractDownloadError(
        "Could not download any of the %d extracts tried for the area: %s"
        % (len(failed), "; ".join("%s (%s)" % f for f in failed)),
        failed,
        errors=errors,
    )
