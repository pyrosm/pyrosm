import os

import geopandas as gpd
import pytest
from shapely.geometry import box

from pyrosm import get_data_by_bbox

# The update=True path fetches Geofabrik's live index over the network; gate it
# like the other download tests so the CI matrix doesn't hammer the service.
run_downloads_only_once = pytest.mark.skipif(
    os.environ.get("RUN_DOWNLOAD_TESTS") != "true",
    reason="Live download tests run on a single CI runner "
    "(windows-latest + Python 3.14); set RUN_DOWNLOAD_TESTS=true to run locally.",
)

HELSINKI = [24.8, 60.1, 25.1, 60.3]
LONDON = [-0.20, 51.45, 0.0, 51.55]
CHICAGO = [-87.70, 41.80, -87.60, 41.95]
FINLAND_URL = "https://download.geofabrik.de/europe/finland-latest.osm.pbf"


def test_returns_pbf_url_by_default():
    assert get_data_by_bbox(HELSINKI) == FINLAND_URL


def test_returns_id_when_url_false():
    assert get_data_by_bbox(HELSINKI, url=False) == "finland"


def test_picks_smallest_covering_extract():
    # London and Chicago each sit inside several nested extents; the smallest
    # covering one must win over its parents (england/uk/europe, us/north-america).
    assert get_data_by_bbox(LONDON, url=False) == "greater-london"
    assert get_data_by_bbox(CHICAGO, url=False) == "us/illinois"


def test_prints_readable_name(capsys):
    get_data_by_bbox(HELSINKI)
    out = capsys.readouterr().out
    assert "Finland" in out
    assert "finland" in out


def test_prints_name_without_redundant_id(capsys):
    # Where Geofabrik's name equals the id, the printed line shows it only once.
    get_data_by_bbox(CHICAGO)
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
    assert get_data_by_bbox(bbox, url=False) == "finland"


def test_raises_when_no_extract_covers():
    # A mid-Pacific box extends beyond any single extent.
    with pytest.raises(ValueError, match="fully covers"):
        get_data_by_bbox([-150, 0, -140, 10])


def test_rejects_malformed_bbox():
    with pytest.raises(ValueError):
        get_data_by_bbox([1, 2, 3])  # not four values
    with pytest.raises(ValueError):
        get_data_by_bbox([1, 2, 0, 3])  # minx > maxx
    with pytest.raises(ValueError):
        get_data_by_bbox([float("-inf"), 0, float("inf"), 1])  # non-finite


def test_deterministic():
    assert get_data_by_bbox(HELSINKI) == get_data_by_bbox(HELSINKI)


def test_vendored_snapshot_integrity():
    from pyrosm.data.geofabrik_index import _load_index

    gdf = _load_index()
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) > 500
    for col in ("id", "name", "pbf"):
        assert col in gdf.columns


@run_downloads_only_once
def test_update_fetches_live_index():
    assert get_data_by_bbox(HELSINKI, update=True, url=False) == "finland"
