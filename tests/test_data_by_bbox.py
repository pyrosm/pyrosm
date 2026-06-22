import json
import os
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

import pyrosm
import pyrosm.data.geofabrik_index as gi
from pyrosm import get_data_by_bbox

# The live update=True path fetches Geofabrik's index over the network; gate it
# like the other download tests so the CI matrix doesn't hammer the service.
run_downloads_only_once = pytest.mark.skipif(
    os.environ.get("RUN_DOWNLOAD_TESTS") != "true",
    reason="Live download tests run on a single CI runner "
    "(windows-latest + Python 3.14); set RUN_DOWNLOAD_TESTS=true to run locally.",
)

HELSINKI = [24.93, 60.16, 24.96, 60.18]
LONDON = [-0.20, 51.45, 0.0, 51.55]
CHICAGO = [-87.70, 41.80, -87.60, 41.95]
FINLAND_URL = "https://download.geofabrik.de/europe/finland-latest.osm.pbf"


@pytest.fixture
def mock_download(monkeypatch):
    """Return the bundled Helsinki extract instead of downloading a large file."""
    helsinki = pyrosm.get_data("helsinki_pbf")
    monkeypatch.setattr(
        "pyrosm.utils.download.download",
        lambda url, filename, update, directory: helsinki,
    )
    return helsinki


# -- download=False: the no-download lookup --


def test_download_false_returns_covering_extract_url():
    assert get_data_by_bbox(HELSINKI, download=False) == FINLAND_URL


def test_download_false_picks_smallest_covering_extract():
    # London and Chicago each sit inside several nested extents; the smallest
    # covering one must win over its parents.
    assert "greater-london" in get_data_by_bbox(LONDON, download=False)
    assert "illinois" in get_data_by_bbox(CHICAGO, download=False)


def test_lookup_prints_matched_extract(capsys):
    get_data_by_bbox(HELSINKI, download=False)
    out = capsys.readouterr().out
    assert "Finland" in out and "finland" in out


def test_lookup_prints_name_without_redundant_id(capsys):
    get_data_by_bbox(CHICAGO, download=False)
    out = capsys.readouterr().out
    assert "us/illinois" in out
    assert "(id:" not in out


@pytest.mark.parametrize(
    "bbox",
    [
        list(HELSINKI),
        tuple(HELSINKI),
        box(*HELSINKI),
        box(*HELSINKI).buffer(0.01),  # non-box polygon -> uses its envelope
        gpd.GeoSeries([box(*HELSINKI)], crs=4326),
        gpd.GeoDataFrame(geometry=[box(*HELSINKI)], crs=4326),
    ],
)
def test_accepts_bbox_input_forms(bbox):
    assert get_data_by_bbox(bbox, download=False) == FINLAND_URL


# -- download + crop --


def test_crop_default_writes_bbox_named_file(mock_download):
    out = get_data_by_bbox(HELSINKI)  # crop=True, download=True (defaults)
    assert Path(out).name == "bbox_24.93_60.16_24.96_60.18.osm.pbf"
    assert Path(out).stat().st_size < Path(mock_download).stat().st_size
    assert pyrosm.OSM(out).get_buildings() is not None


def test_crop_false_returns_full_extract(mock_download):
    assert get_data_by_bbox(HELSINKI, crop=False) == mock_download


def test_output_path_overrides_name(mock_download, tmp_path):
    target = str(tmp_path / "myclip.osm.pbf")
    out = get_data_by_bbox(HELSINKI, output_path=target)
    assert out == target
    assert Path(target).exists()


def test_directory_controls_output_location(mock_download, tmp_path):
    out = get_data_by_bbox(HELSINKI, directory=str(tmp_path))
    assert str(Path(out).parent) == str(tmp_path)
    assert Path(out).name == "bbox_24.93_60.16_24.96_60.18.osm.pbf"


def test_bbox_filename_format():
    assert (
        gi._bbox_filename((24.93, 60.16, 24.96, 60.18))
        == "bbox_24.93_60.16_24.96_60.18.osm.pbf"
    )
    assert (
        gi._bbox_filename((-0.245, 50.798, -0.016, 50.892))
        == "bbox_-0.245_50.798_-0.016_50.892.osm.pbf"
    )


def test_lookup_uses_envelope_not_raw_polygon(monkeypatch):
    # An extract covering an irregular polygon but not its bounding-box envelope
    # must be rejected: the crop filters by the envelope, so the chosen extract
    # must cover it. (Without the envelope fix the smaller "L" extract would win.)
    from shapely.geometry import Polygon

    el = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])
    square = box(0, 0, 10, 10)
    synthetic = gpd.GeoDataFrame(
        {
            "id": ["l-shape", "square"],
            "name": ["l-shape", "square"],
            "pbf": ["l-shape.pbf", "square.pbf"],
            "geometry": [el, square],
        },
        crs="EPSG:4326",
    )
    monkeypatch.setattr(gi, "_load_index", lambda update=False: synthetic)
    assert gi._covering_extract_url(el) == "square.pbf"


# -- errors / edge cases (no download needed) --


def test_raises_when_no_extract_covers():
    # A mid-Pacific box extends beyond any single extent.
    with pytest.raises(ValueError, match="fully covers"):
        get_data_by_bbox([-150, 0, -140, 10], download=False)


def test_rejects_malformed_bbox():
    with pytest.raises(ValueError):
        get_data_by_bbox([1, 2, 3], download=False)  # not four values
    with pytest.raises(ValueError):
        get_data_by_bbox([1, 2, 0, 3], download=False)  # minx > maxx
    with pytest.raises(ValueError):
        get_data_by_bbox([float("-inf"), 0, float("inf"), 1], download=False)


def test_rejects_unsupported_bbox_type():
    with pytest.raises(ValueError, match="should be"):
        get_data_by_bbox("not-a-bbox", download=False)


def test_out_of_range_bbox_warns_and_has_no_coverage():
    # Longitude 200 is off the map: it warns about the range and finds no extract.
    with pytest.warns(UserWarning, match="lon/lat range"):
        with pytest.raises(ValueError, match="outside Geofabrik"):
            get_data_by_bbox([200.0, 0.0, 210.0, 5.0], download=False)


def test_spanning_bbox_reports_overlapping_extents():
    # A box spanning North America and Europe is covered by no single extract.
    with pytest.raises(ValueError, match="fully covers"):
        get_data_by_bbox([-50.0, 45.0, 10.0, 55.0], download=False)


# -- snapshot / index --


def test_vendored_snapshot_integrity():
    gdf = gi._load_index()
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) > 500
    for col in ("id", "name", "pbf"):
        assert col in gdf.columns


def test_update_reads_live_index_schema(monkeypatch):
    # Simulate Geofabrik's live index (download links nested under "urls", as the
    # vendored snapshot is not) without hitting the network.
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(box(20, 59, 31, 71)),  # covers Helsinki
                "properties": {
                    "id": "finland",
                    "parent": "europe",
                    "name": "Finland",
                    "urls": {"pbf": FINLAND_URL},
                },
            }
        ],
    }

    class _FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        gi.urllib.request,
        "urlopen",
        lambda url, context=None: _FakeResponse(json.dumps(fc).encode()),
    )
    # update=True refreshes the index; download=False returns the looked-up URL.
    assert get_data_by_bbox(HELSINKI, update=True, download=False) == FINLAND_URL


def test_data_module_getattr_rejects_unknown():
    import pyrosm.data

    with pytest.raises(AttributeError):
        pyrosm.data.this_attribute_does_not_exist


@run_downloads_only_once
def test_update_fetches_live_index():
    assert get_data_by_bbox(HELSINKI, update=True, download=False) == FINLAND_URL
