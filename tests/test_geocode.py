import json
import os
from pathlib import Path

import pytest
from shapely.geometry import box

import pyrosm
import pyrosm.data.geocoding as gc

# The live geocoding test reaches Nominatim over the network; gate it like the
# other download tests so the CI matrix doesn't hammer the service.
run_downloads_only_once = pytest.mark.skipif(
    os.environ.get("RUN_DOWNLOAD_TESTS") != "true",
    reason="Live download tests run on a single CI runner "
    "(windows-latest + Python 3.14); set RUN_DOWNLOAD_TESTS=true to run locally.",
)

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
    def fake(request, context=None):
        if captured is not None:
            captured.append(request)
        return _FakeResponse(payload)

    return fake


def test_geocode_returns_boundary_polygon(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON))
    geom = pyrosm.geocode("Brighton and Hove, UK")
    assert geom.geom_type == "Polygon"
    assert geom.bounds == pytest.approx(
        (-0.2450771, 50.7982097, -0.0160307, 50.8923741)
    )


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

    def fake(request, context=None):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="Could not reach"):
        pyrosm.geocode("Brighton and Hove, UK")


def test_get_data_by_geocoding_download_false_returns_url(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON))
    out = pyrosm.get_data_by_geocoding("Brighton and Hove, UK", download=False)
    assert "england-latest.osm.pbf" in out


def test_get_data_by_geocoding_crop_false_returns_full_extract(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen(BRIGHTON))
    captured = {}

    def fake_download(url, filename, update, directory):
        captured["url"] = url
        return "/tmp/fake-england-latest.osm.pbf"

    monkeypatch.setattr("pyrosm.utils.download.download", fake_download)
    out = pyrosm.get_data_by_geocoding("Brighton and Hove, UK", crop=False)
    assert out == "/tmp/fake-england-latest.osm.pbf"
    assert "england-latest.osm.pbf" in captured["url"]


def test_get_data_by_geocoding_crop_default_names_by_place(monkeypatch):
    # crop=True is the default; the cropped file is named after the query. Mock
    # geocode to a Helsinki sub-box and download to the bundled Helsinki extract.
    helsinki = pyrosm.get_data("helsinki_pbf")
    monkeypatch.setattr(
        gc, "geocode", lambda q, **kwargs: box(24.93, 60.16, 24.96, 60.18)
    )
    monkeypatch.setattr(
        "pyrosm.utils.download.download",
        lambda url, filename, update, directory: helsinki,
    )
    out = pyrosm.get_data_by_geocoding("Brighton and Hove, UK")
    assert Path(out).name == "brighton-and-hove-uk.osm.pbf"
    assert Path(out).stat().st_size < Path(helsinki).stat().st_size
    assert pyrosm.OSM(out).get_buildings() is not None


def test_slug_filename():
    assert gc._slug_filename("Brighton and Hove, UK") == "brighton-and-hove-uk.osm.pbf"
    assert gc._slug_filename("  Some, Place!!  ") == "some-place.osm.pbf"
    assert gc._slug_filename("???") == "place.osm.pbf"


@run_downloads_only_once
def test_geocode_live():
    geom = pyrosm.geocode("Brighton and Hove, UK")
    assert geom.geom_type in ("Polygon", "MultiPolygon")
    assert "england-latest.osm.pbf" in pyrosm.get_data_by_bbox(geom, download=False)
