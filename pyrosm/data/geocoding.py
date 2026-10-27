"""Geocode a place name and fetch the OSM data that covers it.

Public entry points:

- :func:`geocode` -- a place name -> a Shapely polygon for the place, via
  OpenStreetMap's Nominatim service.
- :func:`get_data_by_geocoding` -- geocode a place, then download (and by default
  crop) the data for it as :func:`pyrosm.get_data_by_area` does.

No extra dependencies: geocoding uses the stdlib ``urllib`` + ``json`` with the
bundled ``certifi``, and ``shapely`` (already required) turns the response into a
geometry. Geocoding uses Nominatim (https://nominatim.openstreetmap.org); its
data is OpenStreetMap, licensed ODbL. Its usage policy
(https://operations.osmfoundation.org/policies/nominatim/) allows at most one request
per second from an application that identifies itself, asks for results to be cached
and asks heavy users to run their own instance -- ``base_url`` lets you point at one.
pyrosm keeps to it: it caches each answer for 7 days, sends the public server at most
one request a second from a process, and starts no geocoding with pyrosm's own
User-Agent, which every pyrosm user shares, once a process has sent 50 requests with it.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.parse

from shapely.geometry import box, shape

from pyrosm._log import log_event
from pyrosm.exceptions import RequestRefusedError
from pyrosm.utils import download as _downloads
from pyrosm.utils.download import _FETCH_ERRORS, _TIMEOUT, _retry, open_url

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org"
_POLICY_URL = "https://operations.osmfoundation.org/policies/nominatim/"
# Seconds between two requests to the public server, how long an answer is kept, and how
# many requests a process may send the public server with pyrosm's own User-Agent.
_INTERVAL = 1.0
_CACHE_SECONDS = 7 * 24 * 3600
_SHARED_AGENT_LIMIT = 50


class _PublicServer:
    """The requests this process sends the public Nominatim server: one at a time, at least
    :data:`_INTERVAL` apart, and no geocoding started after :data:`_SHARED_AGENT_LIMIT`
    requests with pyrosm's own User-Agent."""

    def __init__(self):
        # The turn lock is held through each request; the count lock only to count, so a
        # refused geocoding never waits for a request in flight.
        self.lock = threading.Lock()
        self.count_lock = threading.Lock()
        self.last = None
        self.shared_agent_requests = 0

    def reserve(self, own_agent):
        """Count the first request of a geocoding with pyrosm's own User-Agent (unless
        ``own_agent``, the caller's own), or raise
        :class:`~pyrosm.exceptions.RequestRefusedError` when the process has sent the
        public server :data:`_SHARED_AGENT_LIMIT` requests with it."""
        if own_agent:
            return
        with self.count_lock:
            sent = self.shared_agent_requests
            if sent < _SHARED_AGENT_LIMIT:
                self.shared_agent_requests += 1
                return
        raise RequestRefusedError(
            "pyrosm has sent %d geocoding requests to the public Nominatim server from "
            "this process with its own User-Agent, which every pyrosm user shares, and "
            "sends no more. Nominatim's usage policy (%s) allows at most one request per "
            "second from an application that identifies itself, and bans heavy users. "
            "Pass your own user_agent (your application's name and a contact address), "
            "or set base_url to your own Nominatim server for bulk geocoding."
            % (sent, _POLICY_URL),
            url=_NOMINATIM_URL,
            attempts=0,
        )

    @contextlib.contextmanager
    def turn(self, own_agent, retry):
        """Hold the server while a request is sent and answered, once :data:`_INTERVAL`
        has passed since the last one started; a ``retry`` with pyrosm's own User-Agent
        (unless ``own_agent``) is counted, the first request being counted by
        :meth:`reserve`."""
        with self.lock:
            if self.last is not None:
                wait = self.last + _INTERVAL - time.monotonic()
                if wait > 0:
                    _downloads._sleep(wait)
            if retry and not own_agent:
                with self.count_lock:
                    self.shared_agent_requests += 1
            self.last = time.monotonic()
            yield

    def forget_lock(self):
        """Give a forked child process locks of its own, as another thread of the parent
        may have held these."""
        self.lock = threading.Lock()
        self.count_lock = threading.Lock()


_PUBLIC = _PublicServer()
with contextlib.suppress(AttributeError):  # os.register_at_fork is not on Windows
    os.register_at_fork(after_in_child=lambda: _PUBLIC.forget_lock())


def _cache_dir():
    """The folder geocoding answers are cached in: ``geocoding`` in pyrosm's default
    folder (see :func:`~pyrosm.utils.download.download_dir`)."""
    return _downloads.download_dir() / "geocoding"


def _cache_path(url, headers):
    """The file that caches the answer to the request ``url`` with ``headers`` (pairs), all
    but its User-Agent, which does not change the answer."""
    sent = sorted((k.lower(), v) for k, v in headers if k.lower() != "user-agent")
    key = json.dumps([url, sent]).encode("utf-8")
    return _cache_dir() / ("%s.json" % hashlib.sha256(key).hexdigest())


def _cached(url, headers):
    """The cached answer to the request ``url`` with ``headers``, or None when there is
    none younger than :data:`_CACHE_SECONDS` or it cannot be read."""
    path = _cache_path(url, headers)
    try:
        if time.time() - path.stat().st_mtime < _CACHE_SECONDS:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def _cache(url, headers, results):
    """Cache ``results``, the answer to the request ``url`` with ``headers``; a folder that
    cannot be written leaves the answer uncached."""
    data = json.dumps(results).encode("utf-8")
    try:
        _downloads.write_atomic(_cache_path(url, headers), lambda out: out.write(data))
    except OSError:
        pass


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
        The ``User-Agent`` header sent to Nominatim. Defaults to a pyrosm string, which
        every pyrosm user shares: once a process has sent the public server 50 requests
        with it, geocoding with it raises :class:`~pyrosm.exceptions.RequestRefusedError`.
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
    RequestRefusedError
        If the process has sent the public server 50 requests with pyrosm's own User-Agent.

    Notes
    -----
    Each answer is cached for 7 days in ``geocoding`` in pyrosm's default folder, so the
    same query with the same headers (but for the User-Agent) is answered from there. The public server gets at most one request a second
    from a process, as Nominatim's usage policy asks
    (https://operations.osmfoundation.org/policies/nominatim/).
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

    public = base_url.rstrip("/") == _NOMINATIM_URL
    own_agent = any(k.lower() == "user-agent" for k in headers)
    # A urllib opener's own headers, such as Accept-Language, may change the answer too;
    # it sends them where the request has none of that name.
    sent = {}
    for k, v in getattr(opener, "addheaders", ()):
        sent.setdefault(k.lower(), v)
    sent.update((k.lower(), v) for k, v in headers.items())
    sent = sent.items()
    attempts = []

    def fetch():
        attempts.append(None)
        with contextlib.ExitStack() as held:
            if public:
                held.enter_context(_PUBLIC.turn(own_agent, len(attempts) > 1))
            with open_url(
                url, headers=headers, timeout=timeout, opener=opener
            ) as response:
                return response.read()

    results = _cached(url, sent)
    if results is None:
        if public:
            _PUBLIC.reserve(own_agent)
        try:
            results = json.loads(_retry(fetch))
        except _FETCH_ERRORS as e:
            raise ValueError(
                "Could not reach the geocoding service at %s: %s" % (base_url, e)
            )
        _cache(url, sent, results)

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
        percentage of the crop done.

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
