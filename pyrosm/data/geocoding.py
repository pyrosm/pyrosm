"""Geocode a place name and fetch the OSM data that covers it.

Public entry points:

- :func:`geocode` -- a place name -> a Shapely polygon for the place, via
  OpenStreetMap's Nominatim service.
- :func:`get_data_by_geocoding` -- geocode a place, find the Geofabrik extract
  that covers it (:func:`pyrosm.get_data_by_bbox`), download it, and optionally
  crop it to the place.

No extra dependencies: geocoding uses the stdlib ``urllib`` + ``json`` with the
bundled ``certifi``, and ``shapely`` (already required) turns the response into a
geometry. Geocoding uses Nominatim (https://nominatim.openstreetmap.org); its
data is OpenStreetMap, licensed ODbL. The public server allows about one request
per second and asks heavy users to run their own instance -- ``base_url`` lets you
point at one.
"""

import json
import os
import ssl
import urllib.parse
import urllib.request
from urllib.error import URLError

import certifi
from shapely.geometry import box, shape

from pyrosm import __version__

_NOMINATIM_URL = "https://nominatim.openstreetmap.org"
_DEFAULT_USER_AGENT = "pyrosm/%s (+https://github.com/pyrosm/pyrosm)" % __version__


def geocode(query, polygon=True, base_url=_NOMINATIM_URL, user_agent=None):
    """Geocode a place name to a Shapely polygon via Nominatim.

    Parameters
    ----------
    query : str
        The place name to look up, e.g. ``"Brighton and Hove, UK"``.

    polygon : bool
        When ``True`` (default), return the place's boundary polygon if Nominatim
        provides one; otherwise (or for point-/line-like results such as POIs and
        addresses) the place's bounding-box rectangle is returned. The result is
        always a ``Polygon``/``MultiPolygon``.

    base_url : str
        The Nominatim base URL. Defaults to the public server; point it at your
        own instance for heavy use.

    user_agent : str, optional
        The ``User-Agent`` header sent to Nominatim. Defaults to a pyrosm string.
        Nominatim rejects requests without a descriptive agent.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The place's boundary polygon, or its bounding-box rectangle.

    Raises
    ------
    ValueError
        If ``query`` is empty, the service cannot be reached, or no match is found.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' should be a non-empty place name.")

    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "polygon_geojson": 1 if polygon else 0,
        }
    )
    url = "%s/search?%s" % (base_url.rstrip("/"), params)
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent or _DEFAULT_USER_AGENT}
    )
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, context=context) as response:
            results = json.loads(response.read())
    except URLError as e:
        raise ValueError(
            "Could not reach the geocoding service at %s: %s" % (base_url, e)
        )

    if not results:
        raise ValueError("Could not geocode '%s'." % query)

    result = results[0]
    print("Geocoded '%s' to: %s" % (query, result.get("display_name", query)))

    geojson = result.get("geojson") if polygon else None
    if geojson and geojson.get("type") in ("Polygon", "MultiPolygon"):
        return shape(geojson)
    south, north, west, east = (float(v) for v in result["boundingbox"])
    return box(west, south, east, north)


def get_data_by_geocoding(
    query,
    crop=False,
    output_path=None,
    update=False,
    directory=None,
    base_url=_NOMINATIM_URL,
    user_agent=None,
):
    """Download the Geofabrik extract that covers a geocoded place name.

    Geocodes ``query`` (:func:`geocode`), finds the smallest Geofabrik extract
    that covers it (:func:`pyrosm.get_data_by_bbox`), downloads that extract, and
    returns the local file path.

    Parameters
    ----------
    query : str
        The place name to look up, e.g. ``"Brighton and Hove, UK"``.

    crop : bool
        When ``True``, crop the downloaded extract to the geocoded place (a
        smaller PBF) before returning, and return the cropped file instead of the
        full extract. Defaults to ``False``.

    output_path : str, optional
        Where to write the cropped PBF when ``crop=True``. ``None`` (default)
        writes to a temporary file. Ignored when ``crop=False``.

    update : bool
        When ``True``, re-download the extract even if it already exists locally.

    directory : str, optional
        Directory to download the extract into. ``None`` (default) uses a pyrosm
        temp directory.

    base_url : str
        The Nominatim base URL (see :func:`geocode`).

    user_agent : str, optional
        The ``User-Agent`` header sent to Nominatim (see :func:`geocode`).

    Returns
    -------
    str
        Path to the downloaded extract, or to the cropped PBF when ``crop=True``.
    """
    from pyrosm.data.geofabrik_index import get_data_by_bbox
    from pyrosm.utils.download import download

    geom = geocode(query, base_url=base_url, user_agent=user_agent)
    url = get_data_by_bbox(geom, url=True)
    path = download(url, os.path.basename(url), update, directory)
    if not crop:
        return path

    from pyrosm import OSM

    return OSM(path, bounding_box=geom).to_pbf(output_path=output_path)
