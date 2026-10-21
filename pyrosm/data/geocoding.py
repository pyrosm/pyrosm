"""Geocode a place name and fetch the OSM data that covers it.

Public entry points:

- :func:`geocode` -- a place name -> a Shapely polygon for the place, via
  OpenStreetMap's Nominatim service.
- :func:`get_data_by_geocoding` -- geocode a place, then download (and by default
  crop) the data for it as :func:`pyrosm.get_data_by_area` does.

No extra dependencies: geocoding uses the stdlib ``urllib`` + ``json`` with the
bundled ``certifi``, and ``shapely`` (already required) turns the response into a
geometry. Geocoding uses Nominatim (https://nominatim.openstreetmap.org); its
data is OpenStreetMap, licensed ODbL. The public server allows about one request
per second and asks heavy users to run their own instance -- ``base_url`` lets you
point at one.
"""

import json
import logging
import os
import re
import urllib.parse

from shapely.geometry import box, shape

from pyrosm._log import log_event
from pyrosm.utils.download import _FETCH_ERRORS, _TIMEOUT, _retry, open_url

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org"


def geocode(
    query,
    polygon=True,
    base_url=_NOMINATIM_URL,
    user_agent=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
):
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

    headers : dict, optional
        Extra HTTP headers for the request; ``user_agent`` replaces a ``User-Agent`` in
        them.

    timeout : float | tuple
        Seconds to wait for the server to accept the connection and then for each read
        (60 by default), or a ``(connect, read)`` pair, as for downloads.

    opener : object, optional
        An object with ``open(request, timeout=...)``, e.g. from
        ``urllib.request.build_opener()``, that makes the request instead of pyrosm.

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
    headers = dict(headers or {})
    if user_agent:
        headers = {k: v for k, v in headers.items() if k.lower() != "user-agent"}
        headers["User-Agent"] = user_agent

    def fetch():
        with open_url(url, headers=headers, timeout=timeout, opener=opener) as response:
            return response.read()

    try:
        results = json.loads(_retry(fetch))
    except _FETCH_ERRORS as e:
        raise ValueError(
            "Could not reach the geocoding service at %s: %s" % (base_url, e)
        )

    if not results:
        raise ValueError("Could not geocode '%s'." % query)

    result = results[0]
    log_event(
        logger,
        logging.INFO,
        "geocode",
        query=query,
        result=result.get("display_name", query),
    )

    geojson = result.get("geojson") if polygon else None
    if geojson and geojson.get("type") in ("Polygon", "MultiPolygon"):
        return shape(geojson)
    south, north, west, east = (float(v) for v in result["boundingbox"])
    return box(west, south, east, north)


def _slug_filename(text):
    """The name of the file cropped to the place ``text``: ``area_`` and the place in
    lower case, so it never replaces a downloaded extract named after the place."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return "area_%s.osm.pbf" % (slug or "place")


def get_data_by_geocoding(
    query,
    crop=True,
    download=True,
    update=False,
    directory=None,
    output_path=None,
    base_url=_NOMINATIM_URL,
    user_agent=None,
    progress=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
    strategy="single",
    must_cover=None,
    workers=None,
):
    """Download (and by default crop) the OSM data for a geocoded place name.

    Geocodes ``query`` (:func:`geocode`) and gets the data for the place as
    :func:`pyrosm.get_data_by_area` does: the smallest Geofabrik, BBBike or Movisda
    extract (or set of extracts) that covers it, downloaded and -- by default -- cropped
    to the place.

    Parameters
    ----------
    query : str
        The place name to look up, e.g. ``"Brighton and Hove, UK"``.

    crop : bool
        When ``True`` (default), crop the downloaded extract to the place (its boundary
        polygon, or its box when the geocoder gives none) and return the cropped file,
        named after the query (e.g. ``area_brighton-and-hove-uk.osm.pbf``) and used again
        while it and its extract are unchanged. When ``False``, return the full extract.

    download : bool
        When ``True`` (default), download the covering extract. When ``False``,
        skip the download and return the PBF URL of the extract it would download
        instead (a list of URLs for a set of extracts).

    update, directory
        As for :func:`pyrosm.get_data_by_area`.

    output_path : str, optional
        Path for the file pyrosm writes, as for :func:`pyrosm.get_data_by_area`.
        Ignored when ``download=False``.

    base_url : str
        The Nominatim base URL (see :func:`geocode`).

    user_agent : str, optional
        The ``User-Agent`` header sent to Nominatim (see :func:`geocode`).

    progress : bool | callable | None
        ``None`` (default) follows :func:`pyrosm.enable_logging`, which is ``True`` unless
        set. ``True`` shows the download as a bar on stderr, a widget in Jupyter when
        ipywidgets is installed (``pip install "pyrosm[notebook]"``); without a terminal,
        only a line naming the file is printed. ``False`` shows nothing. A callable is called
        as ``progress(written, total)`` after each chunk, with the bytes written so far and
        the file size, ``None`` when the server does not give it.
        The crop shows its progress the same way; a callable is then called with the
        percentage of the crop done (see :func:`pyrosm.pbf_export.crop_pbf`).

    headers, timeout, opener
        Network options for the geocoding request and for the requests of
        :func:`pyrosm.get_data_by_area`.

    strategy, must_cover, workers
        As for :func:`pyrosm.get_data_by_area`.

    Returns
    -------
    str | list
        The cropped file path (default), the full extract path (``crop=False``), or
        the covering extract's PBF URL (``download=False``; a list of URLs for a set of
        extracts).
    """
    from pyrosm.data.extract_index import _get_area, _url_result

    net = dict(headers=headers, timeout=timeout, opener=opener)
    result = _get_area(
        lambda: geocode(query, base_url=base_url, user_agent=user_agent, **net),
        "polygon" if crop and not isinstance(crop, str) else crop,
        update,
        directory,
        output_path,
        headers,
        timeout,
        opener,
        strategy,
        must_cover,
        workers,
        progress,
        name=_slug_filename(query),
        download=download,
    )
    return _url_result(result) if not download else result.path
