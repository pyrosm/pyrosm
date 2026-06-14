"""Download (and optionally crop) the Geofabrik extract covering a bounding box.

Public entry point: :func:`get_data_by_bbox`. It is backed by a vendored snapshot
of Geofabrik's ``index-v1.json`` (``geofabrik_index.geojson.gz``), a GeoJSON
``FeatureCollection`` of every extract's extent polygon and PBF URL. Refresh the
snapshot with ``scripts/update_geofabrik_index.py``.
"""

import gzip
import json
import os
import ssl
import tempfile
import urllib.request
import warnings

import certifi
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "geofabrik_index.geojson.gz")
_INDEX_URL = "https://download.geofabrik.de/index-v1.json"

# Rank covering extents by true size in an equal-area projection; lon/lat degree
# area over-states size near the poles and can mis-rank nested extracts.
_EQUAL_AREA_CRS = "EPSG:6933"

_index_cache = None


def _features_to_gdf(features):
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    # The live index nests download links under "urls"; the vendored snapshot
    # pre-flattens to a "pbf" column. Normalise both to a "pbf" column.
    if "pbf" not in gdf.columns and "urls" in gdf.columns:
        gdf["pbf"] = gdf["urls"].apply(
            lambda u: u.get("pbf") if isinstance(u, dict) else None
        )
    return gdf


def _load_index(update=False):
    global _index_cache
    if update:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(_INDEX_URL, context=context) as response:
            collection = json.loads(response.read())
        return _features_to_gdf(collection["features"])
    if _index_cache is None:
        with gzip.open(_INDEX_PATH, "rt", encoding="utf-8") as f:
            collection = json.load(f)
        _index_cache = _features_to_gdf(collection["features"])
    return _index_cache


def _bbox_to_polygon(bbox):
    if isinstance(bbox, (gpd.GeoDataFrame, gpd.GeoSeries)):
        bounds = bbox.total_bounds
    elif isinstance(bbox, BaseGeometry):
        # Use the geometry's envelope, i.e. its true bounding box.
        bounds = bbox.bounds
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        bounds = list(bbox)
        if len(bounds) != 4:
            raise ValueError(
                "A bounding box given as a list/tuple/array must have 4 values "
                "[minx, miny, maxx, maxy]. Got %d." % len(bounds)
            )
    else:
        raise ValueError(
            "'bbox' should be a list/tuple/array [minx, miny, maxx, maxy], a "
            "Shapely geometry, or a GeoDataFrame/GeoSeries. Got %r." % type(bbox)
        )

    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if not np.isfinite((minx, miny, maxx, maxy)).all():
        raise ValueError(
            "Bounding box coordinates must be finite; got "
            "(minx, miny, maxx, maxy)=%s." % ((minx, miny, maxx, maxy),)
        )
    if not (minx <= maxx and miny <= maxy):
        raise ValueError(
            "Invalid bounding box (minx, miny, maxx, maxy)=%s; expected "
            "minx <= maxx and miny <= maxy." % ((minx, miny, maxx, maxy),)
        )
    if not (-180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90):
        warnings.warn(
            "Bounding box %s lies outside the WGS84 lon/lat range; coordinates "
            "should be in EPSG:4326 (lon/lat)." % ((minx, miny, maxx, maxy),)
        )
    return box(minx, miny, maxx, maxy)


def _covering_extract_url(geom, update=False):
    """Return the PBF URL of the smallest Geofabrik extract that covers ``geom``."""
    # The crop downstream filters by the geometry's bounding-box envelope, so the
    # extract must cover that envelope, not just an irregular polygon.
    geom = geom.envelope
    gdf = _load_index(update)

    covering = gdf[gdf.covers(geom)]
    if len(covering) == 0:
        intersecting = sorted(gdf[gdf.intersects(geom)]["id"].tolist())
        if intersecting:
            preview = ", ".join(intersecting[:5])
            more = "" if len(intersecting) <= 5 else ", ..."
            raise ValueError(
                "No single Geofabrik extract fully covers the area; it extends "
                "beyond the extent(s) it overlaps (%s%s). Use a smaller area, or "
                "download a covering parent extract." % (preview, more)
            )
        raise ValueError("The area lies outside Geofabrik's available extracts.")

    ranked = covering.assign(
        _area=covering.geometry.to_crs(_EQUAL_AREA_CRS).area
    ).sort_values(["_area", "id"], kind="stable")
    best = ranked.iloc[0]

    name = best["name"] or best["id"]
    if name == best["id"]:
        label = "'%s'" % name
    else:
        label = "'%s' (id: %s)" % (name, best["id"])
    print("Geofabrik extract covering the area: %s" % label)
    return best["pbf"]


def _fmt_coord(value):
    return ("%.5f" % value).rstrip("0").rstrip(".")


def _bbox_filename(bounds):
    return "bbox_%s_%s_%s_%s.osm.pbf" % tuple(_fmt_coord(v) for v in bounds)


def _default_target_dir(directory):
    if directory is not None:
        return directory
    return os.path.join(tempfile.gettempdir(), "pyrosm")


def _download_optionally_crop(
    geom, crop, download, cropped_name, output_path, update, directory
):
    """Look up the covering extract, then optionally download and crop it.

    Shared by :func:`get_data_by_bbox` and
    :func:`pyrosm.data.geocoding.get_data_by_geocoding`.
    """
    url = _covering_extract_url(geom, update)
    if not download:
        return url

    # Aliased so the ``download`` flag does not shadow the download function.
    from pyrosm.utils.download import download as _download_file

    full_path = _download_file(url, os.path.basename(url), update, directory)
    if not crop:
        return full_path

    from pyrosm import OSM

    target = output_path or os.path.join(_default_target_dir(directory), cropped_name)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    return OSM(full_path, bounding_box=geom).to_pbf(output_path=target)


def get_data_by_bbox(
    bbox, crop=True, download=True, update=False, directory=None, output_path=None
):
    """Download (and by default crop) the OSM data covering a bounding box.

    Finds the smallest Geofabrik extract whose extent fully covers ``bbox``,
    downloads it, and -- by default -- crops it to ``bbox`` before returning the
    cropped file path.

    Parameters
    ----------
    bbox : list | tuple | numpy.ndarray | shapely geometry | GeoDataFrame | GeoSeries
        The area of interest as ``[minx, miny, maxx, maxy]`` in lon/lat, a Shapely
        geometry (its bounding box is used), or a GeoDataFrame/GeoSeries (its total
        bounds are used).

    crop : bool
        When ``True`` (default), crop the downloaded extract to ``bbox`` and return
        the cropped file, named ``bbox_<minx>_<miny>_<maxx>_<maxy>.osm.pbf``. When
        ``False``, return the full downloaded extract.

    download : bool
        When ``True`` (default), download the covering extract. When ``False``,
        skip the download and return the covering extract's PBF URL instead.

    update : bool
        When ``True``, re-download the extract even if it already exists locally.

    directory : str, optional
        Directory to download into / write the cropped file to. ``None`` (default)
        uses a pyrosm temp directory.

    output_path : str, optional
        Path for the cropped file when ``crop=True`` (overrides the automatic
        name). Ignored when ``crop=False`` or ``download=False``.

    Returns
    -------
    str
        The cropped file path (default), the full extract path (``crop=False``), or
        the covering extract's PBF URL (``download=False``).

    Raises
    ------
    ValueError
        If no single extract fully covers ``bbox``.
    """
    geom = _bbox_to_polygon(bbox)
    cropped_name = _bbox_filename(geom.bounds)
    return _download_optionally_crop(
        geom, crop, download, cropped_name, output_path, update, directory
    )
