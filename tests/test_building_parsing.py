import pytest
from pyrosm import get_path


@pytest.fixture
def test_pbf():
    pbf_path = get_path("test_pbf")
    return pbf_path


@pytest.fixture
def test_output_dir():
    import os, tempfile
    return os.path.join(tempfile.gettempdir(), "pyrosm_test_results")


def test_parsing_building_elements(test_pbf):
    from pyrosm import OSM
    from pyrosm.buildings import get_building_data
    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    buildings = get_building_data(osm._way_records,
                                  osm.conf.tag_filters.buildings,
                                  None)
    assert isinstance(buildings, dict)

    # Required keys
    required = ['id', 'nodes']
    for col in required:
        assert col in buildings.keys()

    # Test shape
    assert len(buildings["id"]) == 2219


def test_creating_building_geometries(test_pbf):
    from pyrosm import OSM
    from pyrosm.buildings import get_building_data
    from pyrosm.geometry import create_polygon_geometries
    from shapely.geometry import Polygon
    from numpy import ndarray
    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    gdf = get_building_data(osm._way_records,
                                  osm.conf.tag_filters.buildings,
                                  None)
    geometries = create_polygon_geometries(osm._nodes, gdf)
    assert isinstance(geometries, ndarray)
    assert isinstance(geometries[0], Polygon)
    assert len(geometries) == len(gdf["id"])


def test_reading_buildings_with_defaults(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings()

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.shape == (2193, 18)

    required_cols = ['building', 'addr:city', 'addr:street', 'addr:country',
                     'addr:postcode', 'addr:housenumber', 'source', 'opening_hours',
                     'building:levels', 'id',
                     'timestamp', 'version', 'geometry']

    for col in required_cols:
        assert col in gdf.columns


def test_parse_buildings_with_bbox(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import Polygon

    bounds = [26.94, 60.525, 26.96, 60.535]
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)
    gdf = osm.get_buildings()

    assert isinstance(gdf.loc[0, 'geometry'], Polygon)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (569, 14)

    required_cols = ['building', 'addr:street',
                     'addr:postcode', 'addr:housenumber',
                     'opening_hours', 'id',
                     'timestamp', 'version', 'geometry', 'tags']

    for col in required_cols:
        assert col in gdf.columns

    # The total bounds of the result should not be larger than the filter
    # (allow some rounding error)
    result_bounds = gdf.total_bounds
    for coord1, coord2 in zip(bounds, result_bounds):
        assert round(coord2, 3) >= round(coord1, 3)


def test_saving_buildings_to_geopackage(test_pbf, test_output_dir):
    import os
    from pyrosm import OSM
    import geopandas as gpd
    import shutil
    from pandas.testing import assert_frame_equal

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    temp_path = os.path.join(test_output_dir, "pyrosm_test.gpkg")
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings()
    gdf.to_file(temp_path, driver="GPKG")

    # Ensure it can be read and matches with original one
    gdf2 = gpd.read_file(temp_path)

    # When reading integers they
    # might be imported as strings instead of ints which is
    # normal, however, the values should be identical
    convert_to_ints = ["id", "timestamp", "version"]
    for col in convert_to_ints:
        gdf[col] = gdf[col].astype(int)
        gdf2[col] = gdf2[col].astype(int)

    assert_frame_equal(gdf, gdf2)

    # Clean up
    shutil.rmtree(test_output_dir)


def test_reading_buildings_with_filter(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame
    # Filter for 'industrial' buildings
    custom_filter = {'building': ['industrial']}
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings(tag_filters=custom_filter)

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.shape == (28, 6)

    required_cols = ['building', 'id', 'timestamp', 'version', 'tags', 'geometry']

    for col in required_cols:
        assert col in gdf.columns
