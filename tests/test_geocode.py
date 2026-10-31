import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import pyrosm
import pyrosm.data.extract_index as ei
import pyrosm.data.geocoding as gc

# The cache folder function itself; each test's fixture replaces it with its own folder.
CACHE_DIR = gc._cache_dir

BRIGHTON = [
    {
        "display_name": "Brighton and Hove, England, United Kingdom",
        "name": "Brighton and Hove",
        "boundingbox": ["50.7982097", "50.8923741", "-0.2450771", "-0.0160307"],
        "geojson": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-0.2450771, 50.7982097],
                    [-0.0160307, 50.7982097],
                    [-0.0160307, 50.8923741],
                    [-0.2450771, 50.8923741],
                    [-0.2450771, 50.7982097],
                ]
            ],
        },
    }
]

POINT_RESULT = [
    {
        "display_name": "A point of interest",
        "boundingbox": ["50.79", "50.81", "-0.15", "-0.13"],
        "geojson": {"type": "Point", "coordinates": [-0.14, 50.80]},
    }
]

NO_GEOJSON = [
    {
        "display_name": "Somewhere",
        "boundingbox": ["50.79", "50.81", "-0.15", "-0.13"],
    }
]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mock_urlopen(payload, captured=None):
    def fake(request, context=None, timeout=None):
        if captured is not None:
            captured.append(request)
        return _FakeResponse(payload)

    return fake


def test_geocode_returns_boundary_polygon(monkeypatch, caplog, capsys):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON))
    caplog.set_level(logging.INFO, logger="pyrosm")
    geom = pyrosm.geocode("Brighton and Hove, UK")
    assert geom.geom_type == "Polygon"
    assert geom.bounds == pytest.approx(
        (-0.2450771, 50.7982097, -0.0160307, 50.8923741)
    )
    assert "result='Brighton and Hove, England, United Kingdom'" in caplog.text
    assert capsys.readouterr().out == ""


def test_geocode_bbox_fallback_without_geojson(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(NO_GEOJSON))
    geom = pyrosm.geocode("Somewhere")
    assert geom.geom_type == "Polygon"
    assert geom.bounds == pytest.approx((-0.15, 50.79, -0.13, 50.81))


def test_geocode_point_geojson_falls_back_to_bbox(monkeypatch):
    # Nominatim returns point-like GeoJSON for POIs/addresses; geocode must still
    # return a polygon (the bounding box), never a Point.
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(POINT_RESULT))
    geom = pyrosm.geocode("A point of interest")
    assert geom.geom_type == "Polygon"
    assert geom.bounds == pytest.approx((-0.15, 50.79, -0.13, 50.81))


def test_geocode_sends_descriptive_user_agent(monkeypatch):
    captured = []
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON, captured))
    pyrosm.geocode("Brighton and Hove, UK")
    ua = captured[0].get_header("User-agent")
    assert ua and "pyrosm" in ua and "urllib" not in ua


def test_geocode_no_match_raises(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen([]))
    with pytest.raises(ValueError, match="Could not geocode"):
        pyrosm.geocode("Nowhereville XYZ 99999")


@pytest.mark.parametrize("query", ["   ", 3, [], ["Helsinki", " "], ["Helsinki", 3]])
def test_geocode_rejects_a_bad_query(monkeypatch, query):
    """An empty place name, an empty list, or a list holding an empty name or one that is
    not text is refused before any request."""
    monkeypatch.setattr(gc, "open_url", lambda *a, **k: pytest.fail("a request was sent"))
    for function in (pyrosm.geocode, pyrosm.get_data_by_geocoding):
        with pytest.raises(ValueError, match="non-empty place name"):
            function(query)


def test_geocode_combines_a_list_of_places(monkeypatch):
    """Each name of a list is geocoded in turn, a second apart, and the places' polygons (or
    boxes, for a place without one) are combined into one; a name without a match raises an
    error that names it."""
    from shapely.geometry import mapping

    from pyrosm.utils import download

    places = {
        "West": [{"geojson": mapping(box(0, 0, 1, 1))}],
        "East": [{"boundingbox": ["0", "1", "1", "2"]}],
    }
    asked, waits = [], []

    def fake(request, context=None, timeout=None):
        query = urllib.parse.urlsplit(request.full_url).query
        name = urllib.parse.parse_qs(query)["q"][0]
        asked.append(name)
        return _FakeResponse(places.get(name, []))

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr(download, "_sleep", waits.append)
    assert pyrosm.geocode(["West", "East"]).equals(box(0, 0, 2, 1))
    assert len(waits) == 1
    assert pyrosm.geocode(["West"]).equals(pyrosm.geocode("West"))
    with pytest.raises(ValueError, match="Could not geocode 'Nowhere'"):
        pyrosm.geocode(["West", "Nowhere"])
    assert asked == ["West", "East", "Nowhere"]


def test_geocode_service_error_raises(monkeypatch):
    from urllib.error import URLError

    def fake(request, context=None, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="Could not reach .* 'Brighton and Hove, UK'"):
        pyrosm.geocode("Brighton and Hove, UK")


def test_geocode_caches_answers_for_seven_days(monkeypatch):
    import os
    import time

    from pyrosm.utils.download import download_dir

    assert CACHE_DIR() == download_dir() / "geocoding"
    captured = []
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON, captured))
    first = pyrosm.geocode("Brighton and Hove, UK")
    assert pyrosm.geocode("Brighton and Hove, UK", user_agent="other").equals(first)
    assert len(captured) == 1
    pyrosm.geocode("Brighton and Hove, UK", headers={"Accept-Language": "fr"})
    assert len(captured) == 2
    # The request's header wins over an opener's, as urllib sends it.
    english = urllib.request.build_opener()
    english.addheaders = [("Accept-Language", "en")]
    french = {"Accept-Language": "fr"}
    pyrosm.geocode("Brighton and Hove, UK", headers=french, opener=english)
    assert len(captured) == 2
    cached = gc._cache_path(captured[0].full_url, [])
    assert cached.exists()
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(cached, (eight_days_ago, eight_days_ago))
    pyrosm.geocode("Brighton and Hove, UK")
    assert len(captured) == 3
    # An answer that cannot be read is asked again, and one that cannot be written is
    # still returned.
    cached.write_text("{", encoding="utf-8")
    pyrosm.geocode("Brighton and Hove, UK")
    assert len(captured) == 4
    not_a_folder = cached.parent / "file"
    not_a_folder.write_text("", encoding="utf-8")
    monkeypatch.setattr(gc, "_cache_dir", lambda: not_a_folder / "geocoding")
    assert pyrosm.geocode("Brighton and Hove, UK").equals(first)
    assert len(captured) == 5


def test_geocode_paces_and_limits_the_public_server(monkeypatch):
    import time
    from types import SimpleNamespace

    from pyrosm.exceptions import RequestRefusedError
    from pyrosm.utils import download

    # The geocoding module's clock moves only when a wait or the test moves it.
    captured, waits, now = [], [], [0.0]

    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds

    clock = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    monkeypatch.setattr(gc, "time", clock)
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON, captured))
    monkeypatch.setattr(download, "_sleep", wait)
    monkeypatch.setattr(gc, "_SHARED_AGENT_LIMIT", 2)
    pyrosm.geocode("one")
    pyrosm.geocode("two")
    with pytest.raises(RequestRefusedError, match="user_agent") as refused:
        pyrosm.geocode("three")
    assert isinstance(refused.value, ValueError) and refused.value.attempts == 0
    pyrosm.geocode("one")  # cached, so not counted
    now[0] += 0.25  # a quarter second after "two" started, so wait the rest
    pyrosm.geocode("three", user_agent="my-study (me@example.org)")
    now[0] += 2  # a request two seconds later goes out at once
    pyrosm.geocode("four", user_agent="my-study (me@example.org)")
    pyrosm.geocode("three", base_url="https://nominatim.example.org")
    queries = [urllib.parse.urlsplit(r.full_url) for r in captured]
    assert [(q.netloc, urllib.parse.parse_qs(q.query)["q"]) for q in queries] == [
        ("nominatim.openstreetmap.org", ["one"]),
        ("nominatim.openstreetmap.org", ["two"]),
        ("nominatim.openstreetmap.org", ["three"]),
        ("nominatim.openstreetmap.org", ["four"]),
        ("nominatim.example.org", ["three"]),
    ]
    assert waits == [1.0, 0.75]

    # A retried request counts too: the next geocoding is refused.
    from urllib.error import URLError

    monkeypatch.setattr(gc, "_PUBLIC", gc._PublicServer())
    replies = [URLError("reset"), _FakeResponse(BRIGHTON)]

    def flaky(request, context=None, timeout=None):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    pyrosm.geocode("five")
    with pytest.raises(RequestRefusedError):
        pyrosm.geocode("six")

    # A refused geocoding does not wait for a request in flight, and a forked child gets
    # locks of its own, whoever held the parent's.
    server = gc._PublicServer()
    server.shared_agent_requests = gc._SHARED_AGENT_LIMIT
    server.lock.acquire()
    with pytest.raises(RequestRefusedError):
        server.reserve(False)
    server.count_lock.acquire()
    server.forget_lock()
    assert server.lock.acquire(blocking=False)
    assert server.count_lock.acquire(blocking=False)


@pytest.mark.parametrize(
    "place, name",
    [
        ("Brighton and Hove, UK", "area_brighton-and-hove-uk.osm.pbf"),
        (["Helsinki, Finland", "Espoo, Finland"], "area_helsinki-finland_espoo-finland.osm.pbf"),
    ],
)
def test_get_data_by_geocoding(monkeypatch, tmp_path, place, name):
    # The place is a Helsinki sub-box, and its extract the bundled Helsinki extract.
    helsinki = pyrosm.get_data("helsinki_pbf")
    url = "https://download.bbbike.org/osm/bbbike/Helsinki/Helsinki.osm.pbf"
    found = gpd.GeoDataFrame(
        {"provider": ["BBBike"], "id": ["Helsinki"], "name": ["Helsinki"], "url": [url]},
        geometry=[box(24.9, 60.1, 25.0, 60.2)],
        crs="EPSG:4326",
    ).assign(bytes=pd.array([50], dtype="Int64"), contains=True)
    tolerances = []
    monkeypatch.setattr(
        ei, "find_extracts", lambda *a, **k: tolerances.append(k["tolerance"]) or found
    )
    asked = []
    monkeypatch.setattr(
        gc, "geocode", lambda q, **k: asked.append(q) or box(24.93, 60.16, 24.96, 60.18)
    )
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: helsinki)
    out = pyrosm.get_data_by_geocoding(place, directory=tmp_path, progress=False)
    assert asked == [place]
    assert Path(out) == tmp_path / name
    assert Path(out).stat().st_size < Path(helsinki).stat().st_size
    assert len(pyrosm.OSM(out).get_buildings()) > 0
    assert pyrosm.get_data_by_geocoding(place, crop=False, progress=False) == helsinki
    assert pyrosm.get_data_by_geocoding(place, download=False, tolerance=0.01) == url
    assert tolerances == [0.001, 0.001, 0.01]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"strategy": "fast"}, "strategy must be"),
        ({"workers": 0}, "'workers' should be"),
        ({"must_cover": box(0, 0, 1, 1).buffer(-1)}, "must_cover is empty"),
        ({"progress": "yes"}, "progress must be"),
        ({"crop": "bbox"}, "crop must be"),
    ],
)
def test_get_data_by_geocoding_checks_options_first(monkeypatch, options, message):
    monkeypatch.setattr(gc, "open_url", lambda *a, **k: pytest.fail("a request was sent"))
    with pytest.raises(ValueError, match=message):
        pyrosm.get_data_by_geocoding("Helsinki", **options)


def test_slug_filename():
    assert gc._slug_filename("Brighton and Hove, UK") == "area_brighton-and-hove-uk.osm.pbf"
    assert gc._slug_filename("  Some, Place!!  ") == "area_some-place.osm.pbf"
    assert gc._slug_filename("???") == "area_place.osm.pbf"
    places = ["Helsinki, Finland", "Espoo, Finland"]
    assert gc._slug_filename(places) == "area_helsinki-finland_espoo-finland.osm.pbf"
    assert gc._slug_filename(places[:1]) == "area_helsinki-finland.osm.pbf"
    # A long list is cut at 80 characters and told apart by a hash of its names.
    many = ["Place %d" % i for i in range(20)]
    cut = "area_" + "_".join("place-%d" % i for i in range(10))
    assert gc._slug_filename(many) == cut + "_2a2de63a.osm.pbf"
    assert gc._slug_filename(many[:-1]) == cut + "_66bc633d.osm.pbf"


@pytest.mark.live_download
def test_geocode_live():
    geom = pyrosm.geocode("Brighton and Hove, UK")
    assert geom.geom_type in ("Polygon", "MultiPolygon")
    assert pyrosm.get_data_by_bbox(geom, download=False).endswith(".osm.pbf")
