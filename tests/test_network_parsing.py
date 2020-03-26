import pytest
from pyrosm import get_path


@pytest.fixture
def test_pbf():
    pbf_path = get_path("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_path("helsinki_pbf")
    return pbf_path


def test_parse_network_geodataframe(test_pbf):
    from pyrosm import Osm
    from geopandas import GeoDataFrame
    osm = Osm(filepath=test_pbf)
    gdf = osm.parse_osm()

    # Test type
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (2636, 16)


# def test_parse_network_helsinki(helsinki_pbf):
#     from pyrosm import Osm
#     from geopandas import GeoDataFrame
#     osm = Osm(filepath=helsinki_pbf)
#     gdf = osm.parse_osm()

