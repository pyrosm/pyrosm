import pytest
from pyrosm import get_data


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


def test_network(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_network()
    assert isinstance(gdf, GeoDataFrame)


def test_buildings(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_buildings()
    assert isinstance(gdf, GeoDataFrame)


def test_landuse(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_landuse()
    assert isinstance(gdf, GeoDataFrame)


def test_pois(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_pois()
    assert isinstance(gdf, GeoDataFrame)


def test_natural(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_natural()
    assert isinstance(gdf, GeoDataFrame)


def test_custom(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(test_pbf)
    gdf = osm.get_data_by_custom_criteria({"highway": ["secondary"]})
    assert isinstance(gdf, GeoDataFrame)


def test_boundaries(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(helsinki_pbf)
    gdf = osm.get_boundaries()
    assert isinstance(gdf, GeoDataFrame)


def test_passing_incorrect_filepath():
    from pyrosm import OSM
    try:
        OSM(11)
    except ValueError:
        pass
    except Exception as e:
        raise e


def test_passing_wrong_file_format():
    from pyrosm import OSM
    try:
        OSM("test.osm")
    except ValueError:
        pass
    except Exception as e:
        raise e

