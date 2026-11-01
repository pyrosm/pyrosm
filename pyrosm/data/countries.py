"""The country of a location on land, from the vendored borders of country-coder, and
the countries' access defaults for shared paths, from the vendored table of Valhalla with
the countries added from their traffic law."""

import functools
import gzip
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

_BORDERS_PATH = Path(__file__).with_name("country_borders.geojson.gz")
_ACCESS_PATH = Path(__file__).with_name("country_access.json")


@functools.lru_cache(maxsize=None)
def _borders():
    """The ISO codes of the areas' countries and an STRtree of the areas."""
    with gzip.open(_BORDERS_PATH, "rt", encoding="utf-8") as f:
        features = json.load(f)["features"]
    codes = np.array([f["properties"]["iso"] for f in features], dtype=object)
    return codes, shapely.STRtree([shape(f["geometry"]) for f in features])


def country_codes(lon, lat):
    """The ISO 3166-1 alpha-2 code of the country of each point (``lon``, ``lat`` in
    EPSG:4326), as an object array. A territory counts as its country (the Åland Islands as
    ``FI``). A point outside every country, such as far out at sea, and a point on a border
    between two countries get ``None``."""
    codes, tree = _borders()
    points = shapely.points(
        np.atleast_1d(np.asarray(lon, dtype=float)),
        np.atleast_1d(np.asarray(lat, dtype=float)),
    )
    result = np.full(len(points), None, dtype=object)
    point, area = tree.query(points, predicate="intersects")
    if len(point) == 0:
        return result
    order = np.lexsort((codes[area].astype(str), point))
    point, found = point[order], codes[area][order]
    first = np.ones(len(point), dtype=bool)
    first[1:] = point[1:] != point[:-1]
    last = np.ones(len(point), dtype=bool)
    last[:-1] = first[1:]
    # The points whose areas all belong to one country: its first and last match agree.
    single = found[first] == found[last]
    result[point[first][single]] = found[first][single]
    return result


@functools.lru_cache(maxsize=None)
def countries_allowing(mode_key, highway):
    """The ISO codes of the countries whose default lets the mode of ``mode_key`` use a way
    tagged ``highway`` that has no ``mode_key`` tag: ``("foot", "cycleway")`` gives the
    countries where pedestrians may use a cycle way, ``("bicycle", "footway")`` those where
    cyclists may use a foot way. Empty for any other pair."""
    allowed = json.loads(_ACCESS_PATH.read_text(encoding="utf-8"))["allowed"]
    return frozenset(allowed.get(mode_key, {}).get(highway, ()))


# The ways a read keeps whole across the edge of its area reach this far beyond it
# (degrees).
_EXTENT_MARGIN = 0.05


def countries_in(geometry):
    """The sorted ISO codes of the countries whose areas ``geometry`` (EPSG:4326)
    touches."""
    codes, tree = _borders()
    return sorted(set(codes[tree.query(geometry, predicate="intersects")]))


def read_extent(filepath, bounding_box=None):
    """The area a read of ``filepath`` covers: the box in the file's header, within
    ``bounding_box`` (a ``[minx, miny, maxx, maxy]`` list or a shapely geometry) when given,
    widened by 0.05° for the ways the read keeps whole across its edge. ``None`` for a file
    without a box in its header."""
    from pyrosm.pbf_export import read_header_block

    header = read_header_block(filepath)
    if not header.HasField("bbox"):
        return None
    b = header.bbox
    extent = shapely.box(b.left * 1e-9, b.bottom * 1e-9, b.right * 1e-9, b.top * 1e-9)
    if bounding_box is not None:
        if isinstance(bounding_box, (list, tuple)):
            bounding_box = shapely.box(*bounding_box)
        extent = extent.intersection(bounding_box)
    return extent.buffer(_EXTENT_MARGIN)


def way_points(refs, lengths, store):
    """A point of each way, for finding its country: the coordinates ``(lon, lat)`` of its
    first node that the coordinate store ``store`` has, NaN for a way without one. The
    ways' node ids follow one another in ``refs``, ``lengths`` of them per way."""
    lengths = np.asarray(lengths, dtype=np.int64)
    lon = np.full(len(lengths), np.nan)
    lat = np.full(len(lengths), np.nan)
    if len(refs) == 0:
        return lon, lat
    row, node_lon, node_lat = store.gather(np.asarray(refs, dtype=np.int64))
    # The position of each located node, or past the end; the first per way is the least.
    position = np.where(row >= 0, np.arange(len(refs)), len(refs))
    nonempty = lengths > 0
    starts = (np.cumsum(lengths) - lengths)[nonempty]
    first = np.minimum.reduceat(position, starts)
    located = first < len(refs)
    ways = np.flatnonzero(nonempty)[located]
    lon[ways], lat[ways] = node_lon[first[located]], node_lat[first[located]]
    return lon, lat
