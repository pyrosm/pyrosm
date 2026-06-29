"""Parity tests comparing pyrosm against OSMnx on the same data, area and time.

The fixtures in ``tests/data/osmnx_reference/`` were fetched with OSMnx from the Overpass
API at the same historical moment (an Overpass attic ``[date:]`` query) and the same
central-Helsinki bounding box as a subset of ``helsinki_region_pbf`` -- see
``scripts/generate_osmnx_reference.py``. Each test reads the same area, time and filter
from the PBF with pyrosm and checks that the selected OSM ids agree with the OSMnx
fixture to a Jaccard similarity of at least ``JACCARD_MIN``.

The small residual differences are not filter-logic differences: OSMnx keeps ways/areas
clipped or edge-truncated at the bounding box while pyrosm keeps complete ways, and the
PBF snapshot is a little later than the attic anchor (so a few elements re-mapped in that
window appear on one side only). The tests read only the committed fixtures and the PBF,
so they need neither OSMnx nor network access.
"""

from pathlib import Path

import geopandas as gpd
import pytest

from pyrosm import OSM, get_data

# Must match scripts/generate_osmnx_reference.py.
CENTRAL_HELSINKI_BBOX = [24.92, 60.16, 24.97, 60.18]
JACCARD_MIN = 0.995
FIXTURE_DIR = Path(__file__).parent / "data" / "osmnx_reference"

# Each case: fixture name and the pyrosm query that mirrors the OSMnx fixture's filter.
CASES = [
    (
        "network_paths",
        lambda o: o.get_network(
            custom_filter='["highway"~"cycleway|footway|path"]', filter_type="keep"
        ),
    ),
    (
        "network_path_bicycle_designated",
        lambda o: o.get_network(
            custom_filter='["highway"~"path"]["bicycle"~"designated"]',
            filter_type="keep",
        ),
    ),
    (
        "buildings_all",
        lambda o: o.get_buildings(),
    ),
    (
        "buildings_residential",
        lambda o: o.get_buildings(custom_filter={"building": ["residential"]}),
    ),
]


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _osmnx_ids(name):
    """OSM ids in a fixture: networks store ``osmid`` as a column, features as an index."""
    gdf = gpd.read_parquet(FIXTURE_DIR / f"{name}.parquet")
    if "osmid" in gdf.columns:
        return {int(x) for x in gdf["osmid"].unique()}
    gdf = gdf.reset_index()
    return {int(x) for x in gdf["id"].unique()}


def _ids(gdf):
    return {int(x) for x in gdf["id"].unique()}


@pytest.fixture(scope="module")
def osm():
    return OSM(get_data("helsinki_region_pbf"), bounding_box=CENTRAL_HELSINKI_BBOX)


@pytest.mark.parametrize("name,pyrosm_query", CASES, ids=[c[0] for c in CASES])
def test_osmnx_parity(osm, name, pyrosm_query):
    fixture = FIXTURE_DIR / f"{name}.parquet"
    if not fixture.exists():
        pytest.skip(f"OSMnx reference fixture missing: {fixture}")

    osmnx_ids = _osmnx_ids(name)
    pyrosm_ids = _ids(pyrosm_query(osm))
    similarity = _jaccard(osmnx_ids, pyrosm_ids)

    assert similarity >= JACCARD_MIN, (
        f"{name}: Jaccard {similarity:.4f} < {JACCARD_MIN} "
        f"(OSMnx {len(osmnx_ids)}, pyrosm {len(pyrosm_ids)}, "
        f"shared {len(osmnx_ids & pyrosm_ids)}, "
        f"only-OSMnx {len(osmnx_ids - pyrosm_ids)}, "
        f"only-pyrosm {len(pyrosm_ids - osmnx_ids)})"
    )
