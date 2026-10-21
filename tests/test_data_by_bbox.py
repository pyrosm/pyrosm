from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import pyrosm
import pyrosm.data.extract_index as ei
import pyrosm.data.geofabrik_index as gi
from pyrosm import get_data_by_bbox

HELSINKI = [24.93, 60.16, 24.96, 60.18]
BBBIKE_URL = "https://download.bbbike.org/osm/bbbike/Helsinki/Helsinki.osm.pbf"


def _found(*urls):
    """Extracts in the shape ``find_extracts(..., contains_only=True)`` returns, each
    containing the Helsinki box."""
    return gpd.GeoDataFrame(
        {
            "provider": ["BBBike"] * len(urls),
            "id": list(urls),
            "name": list(urls),
            "url": list(urls),
            "bytes": pd.array([50] * len(urls), dtype="Int64"),
            "contains": [True] * len(urls),
        },
        geometry=[box(*HELSINKI)] * len(urls),
        crs="EPSG:4326",
    )


@pytest.fixture
def searches(monkeypatch):
    """The extract search finds the BBBike Helsinki extract for every area, and downloads
    return the bundled Helsinki extract; gives the areas searched."""
    helsinki = pyrosm.get_data("helsinki_pbf")
    areas = []
    monkeypatch.setattr(
        ei, "find_extracts", lambda geom, **k: areas.append(geom) or _found(BBBIKE_URL)
    )
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: helsinki)
    return areas


@pytest.mark.parametrize(
    "bbox, bounds",
    [
        (list(HELSINKI), HELSINKI),
        (tuple(HELSINKI), HELSINKI),
        (box(*HELSINKI), HELSINKI),
        # A polygon other than a box: its envelope is used.
        (box(*HELSINKI).buffer(0.01), box(*HELSINKI).buffer(0.01).bounds),
        (gpd.GeoSeries([box(*HELSINKI)], crs=4326), HELSINKI),
        (gpd.GeoDataFrame(geometry=[box(*HELSINKI)], crs=4326), HELSINKI),
    ],
)
def test_accepts_bbox_input_forms(searches, bbox, bounds):
    assert get_data_by_bbox(bbox, download=False) == BBBIKE_URL
    assert searches[0].equals(box(*bounds))


def test_download_false_lists_the_urls_of_a_set(monkeypatch):
    found = _found("https://b/a.osm.pbf", "https://b/b.osm.pbf")
    monkeypatch.setattr(ei, "find_extracts", lambda *a, **k: found)
    monkeypatch.setattr(ei, "_choose", lambda target, candidates, strategy: found)
    urls = get_data_by_bbox(HELSINKI, download=False, strategy="smallest_total")
    assert urls == ["https://b/a.osm.pbf", "https://b/b.osm.pbf"]


def test_crops_to_the_box_or_returns_the_extract(searches, tmp_path):
    helsinki = pyrosm.get_data("helsinki_pbf")
    out = get_data_by_bbox(HELSINKI, directory=tmp_path, progress=False)
    assert Path(out) == tmp_path / "bbox_24.93_60.16_24.96_60.18.osm.pbf"
    assert Path(out).stat().st_size < Path(helsinki).stat().st_size
    assert len(pyrosm.OSM(out).get_buildings()) > 0
    assert get_data_by_bbox(HELSINKI, crop=False, progress=False) == helsinki


def test_bbox_filename_format():
    assert (
        gi._bbox_filename((24.93, 60.16, 24.96, 60.18))
        == "bbox_24.93_60.16_24.96_60.18.osm.pbf"
    )
    assert (
        gi._bbox_filename((-0.245, 50.798, -0.016, 50.892))
        == "bbox_-0.245_50.798_-0.016_50.892.osm.pbf"
    )


# -- errors / edge cases (no download needed) --


@pytest.mark.parametrize(
    "bbox, error, match",
    [
        ([1, 2, 3], ValueError, "should be four numbers"),
        ([1, 2, 0, 3], ValueError, "minx < maxx"),
        ([float("-inf"), 0, float("inf"), 1], ValueError, "finite"),
        ("not-a-bbox", ValueError, "should be"),
        ([200.0, 0.0, 210.0, 5.0], ValueError, "lon/lat range"),
    ],
)
def test_errors(monkeypatch, bbox, error, match):
    monkeypatch.setattr(ei, "find_extracts", lambda *a, **k: _found())
    with pytest.raises(error, match=match):
        get_data_by_bbox(bbox, download=False)


# -- snapshot / index --


def test_vendored_snapshot_integrity():
    gdf = gi._load_index()
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) > 500
    for col in ("id", "name", "pbf"):
        assert col in gdf.columns


def test_data_module_getattr_rejects_unknown():
    import pyrosm.data

    with pytest.raises(AttributeError):
        pyrosm.data.this_attribute_does_not_exist
