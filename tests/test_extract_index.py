import contextlib
import hashlib
import http.client
import io
import json
import os
import time
from urllib.error import HTTPError, URLError

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from pyrosm.data import extract_index as ei


class _Response(io.BytesIO):
    def __init__(self, data=b"", headers=None):
        super().__init__(data)
        self.headers = headers or {}


class _Broken(_Response):
    def read(self, *args):
        raise OSError("connection reset")


def _fake_open_url(responses, calls):
    """An ``open_url`` stand-in that returns or raises the next queued response."""

    def fake(url, method="GET", headers=None, timeout=None):
        assert timeout == ei._TIMEOUT
        calls.append((url, method, headers or {}))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return fake


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


@pytest.mark.parametrize(
    "loader, point, expected",
    [
        (ei._bbbike_extracts, Point(7.59, 47.56), ["Basel"]),
        (ei._geofabrik_extracts, Point(24.94, 60.17), ["europe", "finland"]),
    ],
)
def test_vendored_indexes(loader, point, expected):
    gdf = loader()
    assert gdf["id"].is_unique and gdf.geometry.is_valid.all()
    assert gdf["url"].str.match(r"https://download\.(bbbike\.org|geofabrik\.de)/").all()
    assert sorted(gdf[gdf.covers(point)]["id"]) == expected


NOT_MODIFIED = HTTPError("u", 304, "Not Modified", {}, None)
SERVER_ERROR = HTTPError("u", 500, "Server Error", {}, None)
DOWN = URLError("down")


def _new_index():
    return _Response(b"new", {"ETag": '"2"'})


def _bad_index():
    return _Response(b"bad")


STALE = 2 * 86400


def _check(data):
    if data == b"bad":
        raise ValueError("not an index")


@pytest.mark.parametrize(
    "copy_age, meta, update, response, content, requested, warns",
    [
        (None, None, False, _new_index, b"new", True, False),
        (60, "ok", False, None, b"old", False, False),
        (-86400, "ok", False, NOT_MODIFIED, b"old", True, False),
        (60, "ok", True, NOT_MODIFIED, b"old", True, False),
        (STALE, "ok", False, NOT_MODIFIED, b"old", True, False),
        (STALE, "ok", False, _new_index, b"new", True, False),
        (STALE, "ok", False, DOWN, b"old", True, True),
        (STALE, "ok", False, SERVER_ERROR, b"old", True, True),
        (STALE, "ok", False, _Broken, b"old", True, True),
        (STALE, "ok", False, http.client.IncompleteRead(b""), b"old", True, True),
        (STALE, "ok", False, _bad_index, b"old", True, True),
        (60, None, False, DOWN, b"old", True, True),
        (60, "wrong size", False, _new_index, b"new", True, False),
        (STALE, "wrong hash", False, _new_index, b"new", True, False),
        (None, None, False, DOWN, None, True, False),
        (None, None, False, SERVER_ERROR, None, True, False),
        (None, None, False, _bad_index, None, True, False),
    ],
)
def test_cached_index(
    tmp_path, monkeypatch, copy_age, meta, update, response, content, requested, warns
):
    path = tmp_path / "index.geojson"
    meta_path = tmp_path / "index.geojson.etag"
    if copy_age is not None:
        path.write_bytes(b"old")
    if meta is not None:
        size = 99 if meta == "wrong size" else 3
        digest = hashlib.sha256(b"old" if meta == "ok" else b"other").hexdigest()
        record = {"etag": '"1"', "bytes": size, "sha256": digest}
        meta_path.write_text(json.dumps(record))
        _age(meta_path, copy_age)
    calls = []
    if callable(response):
        response = response()
    monkeypatch.setattr(ei, "open_url", _fake_open_url([response], calls))

    def cached_index():
        return ei._cached_index("https://x/index.geojson", path, update, _check)

    if content is None:
        with pytest.raises((OSError, ValueError)):
            cached_index()
        return
    expect = pytest.warns(UserWarning, match="cached copy")
    with expect if warns else contextlib.nullcontext():
        assert cached_index() == path
    assert path.read_bytes() == content
    assert bool(calls) == requested and not list(tmp_path.glob("*.part"))
    if calls:
        sent = {"If-None-Match": '"1"'} if meta == "ok" else {}
        assert calls[0][2] == sent
    if requested and not warns:
        assert time.time() - meta_path.stat().st_mtime < 60


def _index(**props):
    base = {"prefix": "N60W024-", "bytes": 1, "name": "x"}
    square = [[[24, 60], [25, 60], [25, 61], [24, 61], [24, 60]]]
    geometry = props.pop("geometry", {"type": "Polygon", "coordinates": square})
    feature = {"geometry": geometry, "properties": {**base, **props}}
    return json.dumps({"features": [feature]}).encode()


@pytest.mark.parametrize(
    "data, kind, valid",
    [
        (_index(), "admin", True),
        (_index(name=None), "grid", True),
        (_index(name=None), "admin", False),
        (_index(bytes="12"), "grid", False),
        (_index(bytes=True), "grid", False),
        (_index(bytes=-1), "grid", False),
        (_index(prefix=""), "grid", False),
        (_index(geometry=None), "grid", False),
        (_index(geometry={"type": "Point"}), "grid", False),
        (
            _index(
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1e400, 0], [0, 1], [0, 0]]],
                }
            ),
            "grid",
            False,
        ),
        (
            _index(
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [200, 0], [0, 1], [0, 0]]],
                }
            ),
            "grid",
            False,
        ),
        (
            _index(geometry={"type": "Polygon", "coordinates": [[[0, 0]]]}),
            "grid",
            False,
        ),
        (b'{"features": []}', "grid", False),
        (b"<html>Bad gateway</html>", "grid", False),
        (b"[]", "grid", False),
    ],
)
def test_check_movisda_index(data, kind, valid):
    if valid:
        ei._check_movisda_index(data, kind)
    else:
        with pytest.raises(ValueError, match="%s index" % kind):
            ei._check_movisda_index(data, kind)


def test_download_sizes(tmp_path, monkeypatch):
    def size(n):
        return _Response(headers={"Content-Length": str(n)})

    calls = []
    bad = [DOWN, _Response(), size("-5"), size("\u00b2"), size("9" * 19)]
    responses = [size("0100")] + bad
    urls = ["https://%s" % c for c in "abcdef"]
    monkeypatch.setattr(ei, "open_url", _fake_open_url(responses, calls))
    with pytest.warns(UserWarning) as record:
        sizes = ei._download_sizes(urls, tmp_path)
    assert sizes == {"https://a": 100} and len(record) == len(bad)
    assert [c[1] for c in calls] == ["HEAD"] * 6

    # A size is reused for a week, then asked again; update=True always asks, and a
    # malformed record counts as missing.
    calls.clear()
    assert ei._download_sizes(["https://a"], tmp_path) == {"https://a": 100}
    record = ei._size_record(tmp_path, "https://a")
    record.write_text(json.dumps([100, time.time() - 8 * 86400]))
    responses.extend([size(n) for n in (200, 300, 400, 500, 600, 700, 800, 900)])
    assert ei._download_sizes(["https://a"], tmp_path) == {"https://a": 200}
    assert ei._download_sizes(["https://a"], tmp_path, True) == {"https://a": 300}
    for malformed in (
        "[]",
        "[1]",
        "[true, %d]" % time.time(),
        "[%d, %d]" % (2**63, time.time()),
        "[1, 1e400]",
        "[1, NaN]",
    ):
        record.write_text(malformed)
        assert ei._download_sizes(["https://a"], tmp_path)["https://a"] > 300
    assert len(calls) == 8


def test_movisda_extracts(tmp_path, monkeypatch):
    admin = gpd.GeoDataFrame(
        {
            "prefix": ["NL-NB-"],
            "name_en": ["North Brabant"],
            "name": ["Noord-Brabant"],
            "bytes": [188],
        },
        geometry=[box(4.0, 51.0, 6.0, 52.0)],
        crs="EPSG:4326",
    )
    # Movisda's grid prefixes have the east/west letter swapped.
    grid = gpd.GeoDataFrame(
        {"prefix": ["N60W024-", "S40E080-10-"], "bytes": [66, 203]},
        geometry=[box(24, 60, 25, 61), box(-80, -40, -70, -30)],
        crs="EPSG:4326",
    )
    folder = tmp_path / "movisda"
    folder.mkdir()
    for gdf, name in ((admin, "Admin-latest.geojson"), (grid, "grid-latest.geojson")):
        gdf.to_file(folder / name, driver="GeoJSON")
        meta = {"etag": '"1"', "bytes": (folder / name).stat().st_size}
        (folder / (name + ".etag")).write_text(json.dumps(meta))
    monkeypatch.setattr(ei, "_movisda_cache", {})

    got = ei._movisda_extracts(tmp_path)
    assert got["id"].tolist() == ["NL-NB", "N60E024", "S40W080-10"]
    assert got["name"].tolist()[0] == "North Brabant"
    assert got["url"].tolist() == [
        "https://osm.download.movisda.io/admin/NL-NB-latest.osm.pbf",
        "https://osm.download.movisda.io/grid/N60W024-latest.osm.pbf",
        "https://osm.download.movisda.io/grid/S40E080-10-latest.osm.pbf",
    ]
    assert got["bytes"].tolist() == [188, 66, 203]
    # An unchanged index file is parsed once; an unreadable one raises ValueError.
    monkeypatch.setattr(ei.gpd, "read_file", lambda *a, **k: pytest.fail("re-read"))
    assert ei._movisda_extracts(tmp_path).equals(got)
    monkeypatch.setattr(ei, "_movisda_cache", {})
    monkeypatch.setattr(ei.gpd, "read_file", lambda *a, **k: 1 / 0)
    with pytest.raises(ValueError, match="could not read the Movisda admin index"):
        ei._movisda_extracts(tmp_path)


def _candidates(provider, rows):
    return ei._frame(
        provider,
        [r[0] for r in rows],
        [r[0] for r in rows],
        ["https://%s/%s" % (provider, r[0]) for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
    )


@pytest.mark.parametrize("movisda_up", [True, False])
def test_covering_extracts_ranked_by_size(tmp_path, monkeypatch, movisda_up):
    area = box(24.9, 60.1, 25.1, 60.3)
    geofabrik = _candidates(
        "Geofabrik",
        [("finland", None, box(19, 59, 32, 71)), ("sweden", None, box(10, 55, 24, 69))],
    )
    bbbike = _candidates(
        "BBBike",
        [
            ("Helsinki", None, box(24.5, 60.0, 25.5, 60.5)),
            ("Espoo", None, box(24.5, 60.0, 25.0, 60.5)),
        ],
    )
    movisda = _candidates(
        "Movisda",
        [
            ("N60E024", 66, box(24, 60, 25, 61)),
            ("N60E020-10", 900, box(20, 60, 30, 70)),
            ("FI", 700, box(19, 59, 32, 71)),
        ],
    )
    monkeypatch.setattr(ei, "_geofabrik_extracts", lambda: geofabrik)
    monkeypatch.setattr(ei, "_bbbike_extracts", lambda: bbbike)

    def movisda_extracts(directory, update=False):
        if not movisda_up:
            raise URLError("down")
        return movisda

    monkeypatch.setattr(ei, "_movisda_extracts", movisda_extracts)
    # finland has no readable size, so it comes last.
    sizes = {"https://BBBike/Helsinki": 50}
    monkeypatch.setattr(
        ei,
        "_download_sizes",
        lambda urls, directory, update=False: {u: sizes[u] for u in urls if u in sizes},
    )

    if movisda_up:
        got = ei._covering_extracts(area, tmp_path)
        assert got["id"].tolist() == ["Helsinki", "FI", "N60E020-10", "finland"]
    else:
        with pytest.warns(UserWarning, match="Movisda"):
            got = ei._covering_extracts(area, tmp_path)
        assert got["id"].tolist() == ["Helsinki", "finland"]
    assert got["bytes"].isna().tolist()[-1]
