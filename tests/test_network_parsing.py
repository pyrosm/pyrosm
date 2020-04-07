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


@pytest.fixture
def test_output_dir():
    import os, tempfile
    return os.path.join(tempfile.gettempdir(), "pyrosm_test_results")


def test_filter_network_by_walking(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="walking")

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (238, 14)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' ways by default
    assert "motorway" not in gdf["highway"].unique()


def test_filter_network_by_driving(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="driving")

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (200, 14)

    required_cols = ['access', 'bridge', 'highway', 'int_ref', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id', 'geometry']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'footway' or 'path' ways by default
    assert "footway" not in gdf["highway"].unique()
    assert "path" not in gdf["highway"].unique()


def test_filter_network_by_cycling(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="cycling")

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (290, 16)

    required_cols = ['access', 'bicycle', 'bridge', 'foot', 'highway', 'lanes', 'lit',
                     'maxspeed', 'name', 'oneway', 'ref', 'service', 'surface', 'tunnel',
                     'id', 'geometry']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' or 'motorway_link' ways by default
    assert "motorway" not in gdf["highway"].unique()
    assert "motorway_link" not in gdf["highway"].unique()


def test_saving_network_to_shapefile(test_pbf, test_output_dir):
    import os
    from pyrosm import OSM
    import geopandas as gpd
    import shutil
    from pandas.testing import assert_frame_equal

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    temp_path = os.path.join(test_output_dir, "pyrosm_test.shp")
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="cycling")
    gdf.to_file(temp_path)

    # Ensure it can be read and matches with original one
    gdf2 = gpd.read_file(temp_path)

    # When reading large OSM id integers (long) they
    # might be imported as strings instead of ints which is normal,
    # however, the values should be identical
    gdf["id"] = gdf["id"].astype(int)
    gdf2["id"] = gdf2["id"].astype(int)

    assert_frame_equal(gdf, gdf2)

    # Clean up
    shutil.rmtree(test_output_dir)


def test_parse_network_with_bbox(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString

    bounds = [26.94, 60.525, 26.96, 60.535]
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)
    gdf = osm.get_network()

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (65, 14)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' ways by default
    assert "motorway" not in gdf["highway"].unique()

    # The total bounds of the result should not be larger than the filter
    # (allow some rounding error)
    result_bounds = gdf.total_bounds
    for coord1, coord2 in zip(bounds, result_bounds):
        assert round(coord2, 3) >= round(coord1, 3)

def test_parse_network_with_shapely_bbox(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString, box

    bounds = box(*[26.94, 60.525, 26.96, 60.535])
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)
    gdf = osm.get_network()

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (65, 14)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' ways by default
    assert "motorway" not in gdf["highway"].unique()

    # The total bounds of the result should not be larger than the filter
    # (allow some rounding error)
    result_bounds = gdf.total_bounds
    for coord1, coord2 in zip(bounds.bounds, result_bounds):
        assert round(coord2, 3) >= round(coord1, 3)


def test_passing_incorrect_bounding_box(test_pbf):
    from pyrosm import OSM

    wrong_format = "[26.94, 60.525, 26.96, 60.535]"
    try:
        osm = OSM(filepath=test_pbf, bounding_box=wrong_format)
    except ValueError as e:
        if "bounding_box should be" in str(e):
            pass
        else:
            raise(e)
    except Exception as e:
        raise e



