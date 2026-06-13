"""Suggest a Geofabrik PBF extract for a bounding box.

Public entry point: :func:`get_data_by_bbox`. It is backed by a vendored
snapshot of Geofabrik's ``index-v1.json`` (``geofabrik_index.geojson.gz``), a
GeoJSON ``FeatureCollection`` of every extract's extent polygon and PBF URL.
Refresh the snapshot with ``scripts/update_geofabrik_index.py``.
"""

import gzip
import json
import os
import ssl
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


def get_data_by_bbox(bbox, url=True, update=False):
    """Suggest the Geofabrik extract to download for a bounding box.

    Returns the smallest Geofabrik extract whose extent fully covers ``bbox``
    (i.e. the most specific, smallest download). The matched extract's
    human-readable name is printed.

    Parameters
    ----------
    bbox : list | tuple | numpy.ndarray | shapely geometry | GeoDataFrame | GeoSeries
        The area of interest as ``[minx, miny, maxx, maxy]`` in lon/lat, a Shapely
        geometry (its bounding box is used), or a GeoDataFrame/GeoSeries (its
        total bounds are used).

    url : bool
        If ``True`` (default) return the PBF download URL; if ``False`` return the
        Geofabrik id. Note that ids can be path-like (e.g. ``"us/illinois"``) and
        are not always resolvable by :func:`pyrosm.get_data`.

    update : bool
        If ``True``, fetch Geofabrik's live ``index-v1.json`` instead of using the
        vendored snapshot.

    Returns
    -------
    str
        The PBF URL (default) or the Geofabrik id of the smallest extract that
        fully covers ``bbox``.

    Raises
    ------
    ValueError
        If no single extract fully covers ``bbox`` (it spans several extracts or
        lies outside Geofabrik's coverage).
    """
    poly = _bbox_to_polygon(bbox)
    gdf = _load_index(update)

    covering = gdf[gdf.covers(poly)]
    if len(covering) == 0:
        intersecting = sorted(gdf[gdf.intersects(poly)]["id"].tolist())
        if intersecting:
            preview = ", ".join(intersecting[:5])
            more = "" if len(intersecting) <= 5 else ", ..."
            raise ValueError(
                "No single Geofabrik extract fully covers the bounding box; it "
                "extends beyond the extent(s) it overlaps (%s%s). Use a smaller "
                "bounding box, or download a covering parent extract." % (preview, more)
            )
        raise ValueError(
            "The bounding box lies outside Geofabrik's available extracts."
        )

    ranked = covering.assign(
        _area=covering.geometry.to_crs(_EQUAL_AREA_CRS).area
    ).sort_values(["_area", "id"], kind="stable")
    best = ranked.iloc[0]

    name = best["name"] or best["id"]
    if name == best["id"]:
        label = "'%s'" % name
    else:
        label = "'%s' (id: %s)" % (name, best["id"])
    print("Suggested Geofabrik extract for the bounding box: %s" % label)

    return best["pbf"] if url else best["id"]
