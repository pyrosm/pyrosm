"""Geocode a place name and fetch the OSM data that covers it.

Public entry points:

- :func:`geocode` -- a place name -> a Shapely polygon for the place, via
  OpenStreetMap's Nominatim service.
- :func:`get_data_by_geocoding` -- geocode a place, then download (and by default
  crop) the Geofabrik extract that covers it.

No extra dependencies: geocoding uses the stdlib ``urllib`` + ``json`` with the
bundled ``certifi``, and ``shapely`` (already required) turns the response into a
geometry. Geocoding uses Nominatim (https://nominatim.openstreetmap.org); its
data is OpenStreetMap, licensed ODbL. The public server allows about one request
per second and asks heavy users to run their own instance -- ``base_url`` lets you
point at one.
"""

import json
import os
import re
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


def _slug_filename(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return "%s.osm.pbf" % (slug or "place")


def get_data_by_geocoding(
    query,
    crop=True,
    download=True,
    update=False,
    directory=None,
    output_path=None,
    base_url=_NOMINATIM_URL,
    user_agent=None,
):
    """Download (and by default crop) the OSM data for a geocoded place name.

    Geocodes ``query`` (:func:`geocode`), finds the smallest Geofabrik extract
    that covers it, downloads it, and -- by default -- crops it to the place
    before returning the cropped file path.

    Parameters
    ----------
    query : str
        The place name to look up, e.g. ``"Brighton and Hove, UK"``.

    crop : bool
        When ``True`` (default), crop the downloaded extract to the place and
        return the cropped file, named after the query (e.g.
        ``brighton-and-hove-uk.osm.pbf``). When ``False``, return the full extract.

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

    base_url : str
        The Nominatim base URL (see :func:`geocode`).

    user_agent : str, optional
        The ``User-Agent`` header sent to Nominatim (see :func:`geocode`).

    Returns
    -------
    str
        The cropped file path (default), the full extract path (``crop=False``), or
        the covering extract's PBF URL (``download=False``).
    """
    from pyrosm.data.geofabrik_index import _download_optionally_crop

    geom = geocode(query, base_url=base_url, user_agent=user_agent)
    cropped_name = _slug_filename(query)
    return _download_optionally_crop(
        geom, crop, download, cropped_name, output_path, update, directory
    )
