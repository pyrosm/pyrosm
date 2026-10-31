"""The country of a location on land, from the vendored borders of country-coder, and
the countries' access defaults for shared paths, from the vendored table of Valhalla."""

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
