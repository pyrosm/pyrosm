import pytest
from pyrosm import get_path


@pytest.fixture
def test_pbf():
    pbf_path = get_path("test_pbf")
    return pbf_path


def test_parse_network_geodataframe(test_pbf):
    from pyrosm import parse_osm
    from geopandas import GeoDataFrame
    gdf = parse_osm(test_pbf)

    # Test type
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (2636, 30)
