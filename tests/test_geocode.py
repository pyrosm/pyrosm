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


def test_geocode_empty_query_raises():
    with pytest.raises(ValueError):
        pyrosm.geocode("   ")


def test_geocode_service_error_raises(monkeypatch):
    from urllib.error import URLError

    def fake(request, context=None, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="Could not reach"):
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
    from pyrosm.exceptions import RequestRefusedError
    from pyrosm.utils import download

    captured, waits = [], []
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON, captured))
    monkeypatch.setattr(download, "_sleep", waits.append)
    monkeypatch.setattr(gc, "_SHARED_AGENT_LIMIT", 2)
    pyrosm.geocode("one")
    pyrosm.geocode("two")
    with pytest.raises(RequestRefusedError, match="user_agent") as refused:
        pyrosm.geocode("three")
    assert isinstance(refused.value, ValueError) and refused.value.attempts == 0
    pyrosm.geocode("one")  # cached, so not counted
    pyrosm.geocode("three", user_agent="my-study (me@example.org)")
    gc._PUBLIC.last -= 2  # a request two seconds later goes out at once
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
    assert len(waits) == 2 and all(0.5 < w <= 1 for w in waits)

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


def test_get_data_by_geocoding(monkeypatch, tmp_path):
    # The place is a Helsinki sub-box, and its extract the bundled Helsinki extract.
    helsinki = pyrosm.get_data("helsinki_pbf")
    url = "https://download.bbbike.org/osm/bbbike/Helsinki/Helsinki.osm.pbf"
    found = gpd.GeoDataFrame(
        {"provider": ["BBBike"], "id": ["Helsinki"], "name": ["Helsinki"], "url": [url]},
        geometry=[box(24.9, 60.1, 25.0, 60.2)],
        crs="EPSG:4326",
    ).assign(bytes=pd.array([50], dtype="Int64"), contains=True)
    monkeypatch.setattr(ei, "find_extracts", lambda *a, **k: found)
    monkeypatch.setattr(gc, "geocode", lambda q, **k: box(24.93, 60.16, 24.96, 60.18))
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: helsinki)
    place = "Brighton and Hove, UK"
    out = pyrosm.get_data_by_geocoding(place, directory=tmp_path, progress=False)
    assert Path(out) == tmp_path / "area_brighton-and-hove-uk.osm.pbf"
    assert Path(out).stat().st_size < Path(helsinki).stat().st_size
    assert len(pyrosm.OSM(out).get_buildings()) > 0
    assert pyrosm.get_data_by_geocoding(place, crop=False, progress=False) == helsinki
    assert pyrosm.get_data_by_geocoding(place, download=False) == url


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


@pytest.mark.live_download
def test_geocode_live():
    geom = pyrosm.geocode("Brighton and Hove, UK")
    assert geom.geom_type in ("Polygon", "MultiPolygon")
    assert pyrosm.get_data_by_bbox(geom, download=False).endswith(".osm.pbf")
