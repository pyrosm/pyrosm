import contextlib
import hashlib
import http.client
import io
import json
import logging
import os
import time
import warnings
from urllib.error import HTTPError, URLError

import geopandas as gpd
import pandas as pd
import pytest
from pathlib import Path
from shapely.geometry import Point, box

import pyrosm
from pyrosm.data import extract_index as ei
from pyrosm.utils import download as dl
from pyrosm.exceptions import (
    DownloadError,
    ExtractDownloadError,
    ExtractNotFoundError,
)


class _Response(io.BytesIO):
    def __init__(self, data=b"", headers=None):
        super().__init__(data)
        self.headers = headers or {}


class _Broken(_Response):
    def read(self, *args):
        raise OSError("connection reset")


def _fake_open_url(responses, calls):
    """An ``open_url`` stand-in that returns or raises the next queued response; the last
    one is repeated once the queue is down to it."""

    def fake(url, method="GET", headers=None, timeout=None, opener=None):
        assert timeout == ei._TIMEOUT
        calls.append((url, method, headers or {}))
        response = responses.pop(0) if len(responses) > 1 else responses[0]
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
    assert ei._bbbike_extracts() is ei._bbbike_extracts()


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
    check = _check if response is _bad_index else None
    if callable(response):
        response = response()
    monkeypatch.setattr(dl, "open_url", _fake_open_url([response], calls))

    def cached_index():
        return ei._cached_index("https://x/index.geojson", path, update, check)

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


def test_cached_index_304_after_the_copy_changed(tmp_path, monkeypatch):
    path = tmp_path / "index.geojson"
    meta_path = tmp_path / "index.geojson.etag"
    path.write_bytes(b"old")
    digest = hashlib.sha256(b"old").hexdigest()
    meta_path.write_text(json.dumps({"etag": '"1"', "bytes": 3, "sha256": digest}))
    _age(meta_path, STALE)

    def replaced_meanwhile(url, method="GET", headers=None, timeout=None, opener=None):
        path.write_bytes(b"new")
        raise NOT_MODIFIED

    monkeypatch.setattr(dl, "open_url", replaced_meanwhile)
    assert ei._cached_index("https://x/index.geojson", path) == path
    # The 304 vouches for the old bytes only, so the check time is not refreshed.
    assert time.time() - meta_path.stat().st_mtime > 86400


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
    # A network error is retried; after three failures the size stays unknown.
    bad = [DOWN] * 3 + [
        _Response(),
        size("-5"),
        size("\u00b2"),
        size("9" * 19),
        size("9" * 5000),
    ]
    responses = [size("0100")] + bad
    urls = ["https://%s" % c for c in "abcdefg"]
    monkeypatch.setattr(dl, "open_url", _fake_open_url(responses, calls))
    with pytest.warns(UserWarning) as record:
        sizes = ei._download_sizes(urls, tmp_path)
    assert sizes == {"https://a": 100} and len(record) == 6
    assert [c[1] for c in calls] == ["HEAD"] * 9

    # A size is reused for a week, then asked again; update=True always asks, and a
    # malformed record counts as missing.
    calls.clear()
    assert ei._download_sizes(["https://a"], tmp_path) == {"https://a": 100}
    record = ei._size_record(tmp_path, "https://a")
    record.write_text(json.dumps([100, time.time() - 8 * 86400]))
    responses[:] = [size(n) for n in (200, 300, 400, 500, 600, 700, 800, 900)]
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


LOOP = HTTPError(
    "u",
    301,
    "The HTTP server returned a redirect error that would lead to an infinite loop.",
    {},
    None,
)


def test_download_sizes_fall_back_to_recorded_sizes(tmp_path, monkeypatch):
    """A size that cannot be read (a redirect loop, a server that is down, a bad
    Content-Length) comes from the recorded sizes when there is one; it is not cached.
    """
    answers = {
        "https://loop": [LOOP],
        "https://down": [DOWN] * 3,
        "https://bad": [_Response(headers={"Content-Length": "x"})],
    }
    calls = []

    def fake(url, method="GET", headers=None, timeout=None, opener=None):
        calls.append(url)
        answer = answers[url].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(dl, "open_url", fake)
    recorded = {"https://loop": (10, "2026-10-01"), "https://bad": (30, "2026-10-01")}
    with pytest.warns(UserWarning) as record:
        sizes = ei._download_sizes(list(answers), tmp_path, fallback=recorded)
    assert sizes == {"https://loop": 10, "https://bad": 30}
    # The loop is not retried; the server that is down is tried three times.
    assert calls == ["https://loop"] + ["https://down"] * 3 + ["https://bad"]
    messages = sorted(str(w.message) for w in record)
    assert (
        sum("recorded in pyrosm's Geofabrik index on 2026-10-01" in m for m in messages)
        == 2
    )
    assert sum("ranked last" in m for m in messages) == 1
    assert not (tmp_path / "extract_sizes").exists()


def test_vendored_geofabrik_sizes():
    """The vendored Geofabrik index records a size and the day it was read for its
    extracts, and the date of the refresh."""
    import gzip

    path = Path(ei.__file__).parent / "geofabrik_index.geojson.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        collection = json.load(f)
    assert len(collection["geofabrik_sizes_date"]) == 10
    extracts = [
        f["properties"] for f in collection["features"] if f["properties"].get("pbf")
    ]
    recorded = [props for props in extracts if props["bytes"] is not None]
    assert len(recorded) > 400
    for props in extracts:
        if props["bytes"] is None:
            assert props["bytes_date"] is None
        else:
            assert type(props["bytes"]) is int and 0 <= props["bytes"] < 2**63
            assert len(props["bytes_date"]) == 10
    sizes = ei._vendored_geofabrik_sizes()
    size, day = sizes["https://download.geofabrik.de/europe/finland-latest.osm.pbf"]
    assert 100 * 2**20 < size < 10 * 2**30 and len(day) == 10
    assert ei._vendored_geofabrik_sizes() is sizes


def test_find_extracts_ranks_by_recorded_size(tmp_path, monkeypatch):
    """When Geofabrik gives no size, its recorded size ranks the extract."""
    area = box(24.9, 60.1, 25.1, 60.3)
    geofabrik = _candidates("Geofabrik", [("finland", None, box(19, 59, 32, 71))])
    movisda = _candidates("Movisda", [("FI", 700, box(19, 59, 32, 71))])
    monkeypatch.setattr(ei, "_geofabrik_extracts", lambda update: geofabrik)
    monkeypatch.setattr(ei, "_bbbike_extracts", lambda: _candidates("BBBike", []))
    monkeypatch.setattr(ei, "_movisda_extracts", lambda *a: movisda)
    monkeypatch.setattr(
        ei,
        "_vendored_geofabrik_sizes",
        lambda: {"https://Geofabrik/finland": (600, "2026-10-01")},
    )
    monkeypatch.setattr(dl, "open_url", _fake_open_url([LOOP], []))
    with pytest.warns(UserWarning, match="recorded in pyrosm's Geofabrik index"):
        got = pyrosm.find_extracts(area, contains_only=True, directory=tmp_path)
    assert list(zip(got["id"], got["bytes"])) == [("finland", 600), ("FI", 700)]


def test_page_size_parser():
    """The refresh script reads an extract page's "File size" in binary units."""
    import importlib.util

    script = Path(__file__).parents[1] / "scripts" / "update_extract_indexes.py"
    spec = importlib.util.spec_from_file_location("update_extract_indexes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.page_size("up to 2026. File size: 521&nbsp;MB. ") == 521 * 2**20
    assert module.page_size("File size: 1.3 GB") == round(1.3 * 2**30)
    assert module.page_size("no size here") is None
    assert module.page_size("File size: 9999999999 GB") is None
    assert module.page_size("File size: 1" + "0" * 400 + " GB") is None
    assert module.snapshot_day("Wed, 30 Sep 2026 10:00:00 GMT") == "2026-09-30"
    assert len(module.snapshot_day(None)) == 10
    # Only Geofabrik's own latest-extract URLs are asked for a size.
    assert module.read_size("https://example.invalid/x.osm.pbf") == (None, None)


@pytest.fixture
def movisda_indexes(tmp_path, monkeypatch):
    """Small admin and grid indexes, cached under ``tmp_path/movisda`` as if downloaded."""
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
    return admin


def test_movisda_extracts(tmp_path, monkeypatch, movisda_indexes):
    got = ei._movisda_extracts(tmp_path)
    assert got["id"].tolist() == ["NL-NB", "N60E024", "S40W080-10"]
    assert got["name"].tolist()[0] == "North Brabant"
    assert got["url"].tolist() == [
        "https://osm.download.movisda.io/admin/NL-NB-latest.osm.pbf",
        "https://osm.download.movisda.io/grid/N60W024-latest.osm.pbf",
        "https://osm.download.movisda.io/grid/S40E080-10-latest.osm.pbf",
    ]
    assert got["bytes"].tolist() == [188, 66, 203]
    without_en = movisda_indexes.drop(columns="name_en").to_json().encode()
    frame = ei._movisda_frame(io.BytesIO(without_en), "admin")
    assert frame["name"].tolist() == ["Noord-Brabant"]
    # An unchanged index file is parsed once; an unreadable one raises ValueError.
    monkeypatch.setattr(ei.gpd, "read_file", lambda *a, **k: pytest.fail("re-read"))
    assert ei._movisda_extracts(tmp_path).equals(got)
    monkeypatch.setattr(ei.gpd, "read_file", lambda *a, **k: 1 / 0)
    with pytest.raises(ValueError, match="could not read the Movisda admin index"):
        ei._movisda_frame(io.BytesIO(b"{}"), "admin")


@pytest.mark.parametrize(
    "admin, grid", [("up", "down"), ("down", "up"), ("down", "down"), ("up", "bad")]
)
def test_movisda_extracts_fall_back(
    tmp_path, monkeypatch, movisda_indexes, admin, grid
):
    """Without a fetched or cached index, the administrative areas are left out and the grid
    tiles come from pyrosm's vendored copy, each with a warning."""
    real = ei._cached_index
    state = {"admin": admin, "grid": grid}

    def cached_index(url, path, update=False, check=None, net=None):
        kind = "admin" if "/admin/" in url else "grid"
        if state[kind] == "down":
            raise DOWN
        if state[kind] == "bad":
            path.write_text("not an index")
            path.with_name(path.name + ".etag").unlink()
            return path
        return real(url, path, update, check, net)

    monkeypatch.setattr(ei, "_cached_index", cached_index)
    with pytest.warns(UserWarning) as record:
        got = ei._movisda_extracts(tmp_path)
    messages = [str(w.message) for w in record]
    assert ("NL-NB" in got["id"].tolist()) == (admin == "up")
    assert any("administrative index is unavailable" in m for m in messages) == (
        admin == "down"
    )
    if grid == "up":
        assert got["id"].tolist()[-2:] == ["N60E024", "S40W080-10"]
    else:
        # The vendored copy has every tile, e.g. the one holding Helsinki.
        assert len(got) > 1000 and "N60E024" in got["id"].tolist()
        date = ei._movisda_index_frame("grid", ei._MOVISDA_GRID_PATH).attrs
        assert any(
            "using pyrosm's copy of it from %s" % date["snapshot_date"] in m
            for m in messages
        )


def test_vendored_movisda_grid_index():
    import gzip

    data = gzip.decompress(ei._MOVISDA_GRID_PATH.read_bytes())
    ei._check_movisda_index(data, "grid")
    frame = ei._movisda_index_frame("grid", ei._MOVISDA_GRID_PATH)
    assert len(frame) > 1000 and len(frame.attrs["snapshot_date"]) == 10
    assert frame["id"].str.fullmatch(r"[NS]\d{2}[EW]\d{3}(-10)?").all()
    assert (frame["bytes"] >= 0).all()
    helsinki = frame[frame.covers(Point(24.94, 60.17))]["id"]
    assert set(helsinki) == {"N60E024", "N60E020-10"}


def _candidates(provider, rows):
    return ei._frame(
        provider,
        [r[0] for r in rows],
        [r[0] for r in rows],
        ["https://%s/%s" % (provider, r[0]) for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
    )


@pytest.mark.parametrize("contains_only", [False, True])
@pytest.mark.parametrize("movisda_up", [True, False])
def test_find_extracts_ranked_by_size(tmp_path, monkeypatch, movisda_up, contains_only):
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
            ("Vantaa", None, box(25.1, 60.0, 25.5, 60.5)),
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
    refreshed = []
    monkeypatch.setattr(
        ei, "_geofabrik_extracts", lambda update: refreshed.append(update) or geofabrik
    )
    monkeypatch.setattr(ei, "_bbbike_extracts", lambda: bbbike)

    nets, fallbacks = [], []

    def movisda_extracts(directory, update=False, net=None):
        refreshed.append(update)
        nets.append(net)
        if not movisda_up:
            raise URLError("down")
        return movisda

    monkeypatch.setattr(ei, "_movisda_extracts", movisda_extracts)
    # finland has no readable size, so it comes last among the extracts containing the area.
    sizes = {"https://BBBike/Helsinki": 50, "https://BBBike/Espoo": 40}
    asked = []

    def download_sizes(urls, directory, update=False, net=None, fallback=None):
        refreshed.append(update)
        fallbacks.append(fallback)
        nets.append(net)
        asked.extend(urls)
        return {u: sizes[u] for u in urls if u in sizes}

    monkeypatch.setattr(ei, "_download_sizes", download_sizes)

    warns = contextlib.nullcontext() if movisda_up else pytest.warns(UserWarning)
    with warns:
        got = pyrosm.find_extracts(
            area,
            contains_only=contains_only,
            update=True,
            directory=tmp_path,
            headers={"User-Agent": "transitio/1"},
            timeout=30,
            opener=OPENER,
        )
    containing = ["Helsinki", "FI", "N60E020-10", "finland"]
    # Espoo and N60E024 only overlap the area; Vantaa and sweden only touch or miss it.
    overlapping = [] if contains_only else ["Espoo", "N60E024"]
    if not movisda_up:
        containing = ["Helsinki", "finland"]
        overlapping = overlapping[:1]
    assert got["id"].tolist() == containing + overlapping
    assert got["contains"].tolist() == [True] * len(containing) + [False] * len(
        overlapping
    )
    assert list(got.columns) == ei._COLUMNS and got.crs == "EPSG:4326"
    assert got["bytes"].isna().tolist()[len(containing) - 1]
    assert sorted(asked) == sorted(
        "https://%s/%s" % (p, i)
        for p, i in zip(got["provider"], got["id"])
        if p != "Movisda"
    )
    # Geofabrik's index, Movisda's indexes and the download sizes are all refreshed.
    assert refreshed == [True, True, True]
    # The sizes recorded in pyrosm's Geofabrik index back up the size requests.
    assert fallbacks == [ei._vendored_geofabrik_sizes()]
    # The caller's network options reach the Movisda index and the size requests.
    assert [(n.headers, n.timeout, n.opener) for n in nets] == [
        ({"User-Agent": "transitio/1"}, 30, OPENER)
    ] * 2


HELSINKI = [24.93, 60.16, 24.96, 60.18]
OPENER = object()
BBBIKE = ("BBBike", "Helsinki", "https://b/Helsinki/Helsinki.osm.pbf", 50)
MOVISDA = ("Movisda", "N60E024", "https://m/grid/N60W024-latest.osm.pbf", 66)


def _ranked(*rows):
    """Candidates in the shape ``find_extracts(..., contains_only=True)`` returns."""
    return gpd.GeoDataFrame(
        {
            "provider": [r[0] for r in rows],
            "id": [r[1] for r in rows],
            "name": [r[1] for r in rows],
            "url": [r[2] for r in rows],
            "bytes": pd.array([r[3] for r in rows], dtype="Int64"),
            "contains": [True] * len(rows),
        },
        geometry=[box(*HELSINKI)] * len(rows),
        crs="EPSG:4326",
    )


@pytest.mark.parametrize("crop", [True, False])
def test_get_data_by_area_falls_back_to_next_extract(
    tmp_path, monkeypatch, caplog, capsys, crop
):
    helsinki = pyrosm.get_data("helsinki_pbf")
    found = []
    monkeypatch.setattr(
        ei, "find_extracts", lambda *a, **k: found.append(k) or _ranked(BBBIKE, MOVISDA)
    )
    tried = []

    def download(url, filename, update, directory, **net):
        tried.append((filename, net))
        if "Helsinki" in url:
            raise DownloadError("unavailable", url=url, status=503, attempts=3)
        return helsinki

    monkeypatch.setattr("pyrosm.utils.download.download", download)
    caplog.set_level(logging.INFO, logger="pyrosm")
    with pytest.warns(UserWarning, match="next smallest"):
        got = pyrosm.get_data_by_area(
            box(*HELSINKI),
            crop=crop,
            directory=tmp_path,
            headers={"User-Agent": "t"},
            timeout=30,
            opener=OPENER,
        )
    net = {"headers": {"User-Agent": "t"}, "timeout": 30, "opener": OPENER}
    assert {k: found[0][k] for k in net} == net
    assert tried == [
        ("bbbike_Helsinki.osm.pbf", net),
        ("movisda_N60W024-latest.osm.pbf", net),
    ]
    assert "Movisda 'N60E024'" in caplog.text and capsys.readouterr().out == ""
    assert (got.provider, got.extract, got.bytes) == ("Movisda", "N60E024", 66)
    assert got.failed == [(BBBIKE[2], "unavailable")]
    if crop:
        assert Path(got).name == "bbox_24.93_60.16_24.96_60.18.osm.pbf"
        assert Path(got).stat().st_size < Path(helsinki).stat().st_size
        assert pyrosm.OSM(got).filepath == got.path
    else:
        assert got.path == helsinki


@pytest.mark.parametrize(
    "candidates, error",
    [((), ExtractNotFoundError), ((BBBIKE,), ExtractDownloadError)],
)
def test_get_data_by_area_errors(monkeypatch, candidates, error):
    monkeypatch.setattr(ei, "find_extracts", lambda *a, **k: _ranked(*candidates))

    def download(url, filename, update, directory, **net):
        raise DownloadError("down", url=url, attempts=3)

    monkeypatch.setattr("pyrosm.utils.download.download", download)
    warns = pytest.warns(UserWarning) if candidates else contextlib.nullcontext()
    with pytest.raises(error) as info, warns:
        pyrosm.get_data_by_area(box(*HELSINKI))
    if error is ExtractDownloadError:
        assert info.value.failed == [(BBBIKE[2], "down")]
        assert [(e.url, e.attempts) for e in info.value.errors] == [(BBBIKE[2], 3)]
        assert not isinstance(info.value, ValueError)
    else:
        assert isinstance(info.value, ValueError)


@pytest.mark.parametrize(
    "area",
    [
        box(*HELSINKI),
        HELSINKI,
        gpd.GeoDataFrame(geometry=[box(*HELSINKI)], crs="EPSG:4326").to_crs(3067),
        gpd.GeoDataFrame(geometry=[box(*HELSINKI)]),
    ],
)
def test_get_data_by_area_accepts_area_forms(monkeypatch, area):
    seen = []
    monkeypatch.setattr(
        ei, "find_extracts", lambda geom, **k: seen.append((geom, k)) or _ranked()
    )
    with pytest.raises(ValueError, match="No Geofabrik"):
        pyrosm.get_data_by_area(area)
    geom, options = seen[0]
    assert geom.bounds == pytest.approx(HELSINKI, abs=1e-6)
    assert options == {
        "contains_only": True,
        "update": False,
        "directory": None,
        "headers": None,
        "timeout": 60,
        "opener": None,
    }


@pytest.mark.parametrize(
    "area, message",
    [
        (gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), "empty"),
        (Point(24.94, 60.17), "width and a height"),
        ([24.93, 60.16, 24.93, 60.18], "width and a height"),
    ],
)
def test_get_data_by_area_rejects_flat_or_empty_area(area, message):
    with pytest.raises(ValueError, match=message) as info:
        pyrosm.get_data_by_area(area)
    # An invalid area is not reported as "no extract contains the area".
    assert type(info.value) is ValueError


@pytest.mark.live_download
def test_get_data_by_area_downloads_smallest_extract(tmp_path):
    area = box(7.420, 43.733, 7.424, 43.736)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = pyrosm.get_data_by_area(area, directory=tmp_path)
    if any("Movisda" in str(w.message) for w in caught):
        pytest.skip("Movisda's index is unavailable")
    # Movisda's Monaco extract (~0.5 MB) is smaller than Geofabrik's; no BBBike city covers it.
    assert (got.provider, got.extract) == ("Movisda", "MC") and got.bytes < 5_000_000
    assert len(pyrosm.OSM(got).get_buildings()) > 0
