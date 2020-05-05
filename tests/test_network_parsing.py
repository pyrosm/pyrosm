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
    assert gdf.shape == (265, 20)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry', 'tags', 'osm_type']
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
    assert gdf.shape == (207, 18)

    required_cols = ['access', 'bridge', 'highway', 'int_ref', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id', 'geometry', 'tags',
                     'osm_type']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'footway' or 'path' ways by default
    assert "footway" not in gdf["highway"].unique()
    assert "path" not in gdf["highway"].unique()


def test_filter_network_by_driving_with_service_roads(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="driving+service")

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (207, 18)

    required_cols = ['access', 'bridge', 'highway', 'int_ref', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id', 'geometry', 'tags',
                     'osm_type']
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
    assert gdf.shape == (290, 20)

    required_cols = ['access', 'bicycle', 'bridge', 'foot', 'highway', 'lanes', 'lit',
                     'maxspeed', 'name', 'oneway', 'ref', 'service', 'surface', 'tunnel',
                     'id', 'geometry', 'tags', 'osm_type']
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' or 'motorway_link' ways by default
    assert "motorway" not in gdf["highway"].unique()
    assert "motorway_link" not in gdf["highway"].unique()


def test_filter_network_by_all(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="all")

    assert isinstance(gdf.loc[0, 'geometry'], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (331, 21)

    required_cols = ['access', 'bicycle', 'bridge', 'foot', 'highway', 'lanes', 'lit',
                     'maxspeed', 'name', 'oneway', 'ref', 'service', 'surface', 'tunnel',
                     'id', 'geometry', 'tags', 'osm_type']
    for col in required_cols:
        assert col in gdf.columns


def test_saving_network_to_shapefile(test_pbf, test_output_dir):
    import os
    from pyrosm import OSM
    import geopandas as gpd
    import shutil

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    temp_path = os.path.join(test_output_dir, "pyrosm_test.shp")
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="cycling")
    gdf.to_file(temp_path)

    # Ensure it can be read and matches with original one
    gdf2 = gpd.read_file(temp_path)

    cols = gdf.columns
    for col in cols:
        assert gdf[col].tolist() == gdf2[col].tolist()

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
    assert gdf.shape == (74, 20)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry', 'tags', 'osm_type']
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
    assert gdf.shape == (74, 20)

    required_cols = ['access', 'bridge', 'foot', 'highway', 'lanes', 'lit', 'maxspeed',
                     'name', 'oneway', 'ref', 'service', 'surface', 'id',
                     'geometry', 'tags', 'osm_type']
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


def test_passing_incorrect_net_type(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    try:
        osm.get_network("wrong_network")
    except ValueError as e:
        if "'network_type' should be one of the following" in str(e):
            pass
        else:
            raise(e)
    except Exception as e:
        raise e

    try:
        osm.get_network(42)
    except ValueError as e:
        if "'network_type' should be one of the following" in str(e):
            pass
        else:
            raise(e)
    except Exception as e:
        raise e


def test_reading_network_from_area_without_data(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Bounding box for area that does not have any data
    bbox = [24.940514, 60.173849, 24.942, 60.175892]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)

    # The tool should warn if no buildings were found
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_network()
        # Check the warning text
        if "could not find any network data" in str(w):
            pass

    # Result should be None
    assert gdf is None


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_network()
    extra_col = "wikidata"
    extra = osm.get_network(extra_attributes=[extra_col])

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1]+1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)
