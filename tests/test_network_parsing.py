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
def helsinki_history_pbf():
    pbf_path = get_data("helsinki_test_history_pbf")
    return pbf_path


@pytest.fixture
def test_output_dir():
    import os, tempfile

    return os.path.join(tempfile.gettempdir(), "pyrosm_test_results")


def test_filter_network_by_walking(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="walking")

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (265, 21)

    required_cols = [
        "access",
        "bridge",
        "foot",
        "highway",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' ways by default
    assert "motorway" not in gdf["highway"].unique()


def test_filter_network_by_driving(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="driving")

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (207, 19)

    required_cols = [
        "access",
        "bridge",
        "highway",
        "int_ref",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'footway' or 'path' ways by default
    assert "footway" not in gdf["highway"].unique()
    assert "path" not in gdf["highway"].unique()


def test_filter_network_by_driving_with_service_roads(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="driving+service")

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (207, 19)

    required_cols = [
        "access",
        "bridge",
        "highway",
        "int_ref",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'footway' or 'path' ways by default
    assert "footway" not in gdf["highway"].unique()
    assert "path" not in gdf["highway"].unique()


def test_filter_network_by_cycling(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="cycling")

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (290, 21)

    required_cols = [
        "access",
        "bicycle",
        "bridge",
        "foot",
        "highway",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "tunnel",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
    for col in required_cols:
        assert col in gdf.columns

    # Should not include 'motorway' or 'motorway_link' ways by default
    assert "motorway" not in gdf["highway"].unique()
    assert "motorway_link" not in gdf["highway"].unique()


def test_filter_network_by_all(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="all")

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (331, 22)

    required_cols = [
        "access",
        "bicycle",
        "bridge",
        "foot",
        "highway",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "tunnel",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
    for col in required_cols:
        assert col in gdf.columns


def test_saving_network_to_shapefile(test_pbf, test_output_dir):
    import os
    from pyrosm import OSM
    import geopandas as gpd
    import shutil
    import numpy as np

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
        # Geometry col might contain different types of geoms
        # (due to saving MultiLineGeometries which might be read as a "single")
        if col == "geometry":
            continue

        try:
            assert gdf[col].tolist() == gdf2[col].tolist()
        except AssertionError:
            # Skip if the column contains only None values (to avoid conflict between None and np.nan)
            if gdf[col].unique().tolist() == [None]:
                continue

    # Clean up
    shutil.rmtree(test_output_dir)


def test_parse_network_with_bbox(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    bounds = [26.94, 60.525, 26.96, 60.535]
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)
    gdf = osm.get_network()

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (74, 21)

    required_cols = [
        "access",
        "bridge",
        "foot",
        "highway",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
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
    from shapely.geometry import MultiLineString, box

    bounds = box(*[26.94, 60.525, 26.96, 60.535])
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)
    gdf = osm.get_network()

    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (74, 21)

    required_cols = [
        "access",
        "bridge",
        "foot",
        "highway",
        "lanes",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "ref",
        "service",
        "surface",
        "id",
        "geometry",
        "tags",
        "osm_type",
        "length",
    ]
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
            raise (e)
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
            raise (e)
    except Exception as e:
        raise e

    try:
        osm.get_network(42)
    except ValueError as e:
        if "'network_type' should be one of the following" in str(e):
            pass
        else:
            raise (e)
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
    assert extra.shape[1] == gdf.shape[1] + 1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)


def test_getting_nodes_and_edges(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import Point, LineString

    osm = OSM(filepath=test_pbf)

    nodes, edges = osm.get_network(nodes=True)
    nodes = nodes.reset_index(drop=True)

    assert isinstance(edges, GeoDataFrame)
    assert isinstance(edges.loc[0, "geometry"], LineString)

    assert isinstance(nodes, GeoDataFrame)
    assert isinstance(nodes.loc[0, "geometry"], Point)

    # Test shape
    assert edges.shape == (1215, 23)
    assert nodes.shape == (1147, 9)

    # Edges should have "u" and "v" columns
    required = ["u", "v", "length"]
    ecols = edges.columns
    for col in required:
        assert col in ecols

    # Nodes should have (at least) "id", "lat", and "lon" columns
    required = ["id", "lat", "lon"]
    ncols = nodes.columns
    for col in required:
        assert col in ncols


def test_getting_nodes_and_edges_with_bbox(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import Point, LineString, box

    bounds = [26.94, 60.525, 26.96, 60.535]
    # Init with bounding box
    osm = OSM(filepath=test_pbf, bounding_box=bounds)

    nodes, edges = osm.get_network(nodes=True)
    nodes = nodes.reset_index(drop=True)

    assert isinstance(edges, GeoDataFrame)
    assert isinstance(edges.loc[0, "geometry"], LineString)

    assert isinstance(nodes, GeoDataFrame)
    assert isinstance(nodes.loc[0, "geometry"], Point)

    # Test shape
    assert edges.shape == (321, 23)
    assert nodes.shape == (317, 9)

    # Edges should have "u" and "v" columns
    required = ["u", "v", "length"]
    ecols = edges.columns
    for col in required:
        assert col in ecols

    # Nodes should have (at least) "id", "lat", and "lon" columns
    required = ["id", "lat", "lon"]
    ncols = nodes.columns
    for col in required:
        assert col in ncols


def test_reading_network_from_osh(helsinki_history_pbf):
    from pyrosm import OSM
    from pyrosm.utils import datetime_to_unix_time
    import pandas as pd
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString

    timestamp = "2010-01-01"
    dt = pd.to_datetime(timestamp, utc=True)
    unix_time = datetime_to_unix_time(dt)

    osm = OSM(filepath=helsinki_history_pbf)
    gdf = osm.get_network(timestamp=timestamp)

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], MultiLineString)
    assert gdf.shape == (210, 25)

    required_cols = ["highway", "id", "timestamp", "version", "geometry"]

    for col in required_cols:
        assert col in gdf.columns

    # None of the features should be newer than the given timestamp
    assert gdf["timestamp"].max() <= unix_time

    # There should be only a single version per id
    cnt = len(gdf)
    assert cnt == len(gdf.drop_duplicates(subset=["id"]))

    # Changing the timestamp should parse a different set of elements
    new_timestamp = "2015-01-01"
    dt2 = pd.to_datetime(new_timestamp, utc=True)
    unix_time2 = datetime_to_unix_time(dt2)

    gdf2 = osm.get_network(timestamp=new_timestamp)

    # Now the "current_timestamp" should have been updated
    assert osm._current_timestamp == unix_time2

    # Number of elements should (likely) be higher
    assert len(gdf2) > len(gdf)

    # There should be newer timestamps in the data
    assert gdf2["timestamp"].max() > unix_time

    # But is shouldn't be higher than the given timestamp
    assert gdf2["timestamp"].max() <= unix_time2

    # Test reading network with nodes
    n, e = osm.get_network(timestamp=new_timestamp, nodes=True)
    assert isinstance(n, GeoDataFrame)
    assert isinstance(e, GeoDataFrame)
    # Timestamp shouldn't be higher than the given timestamp
    assert e["timestamp"].max() <= unix_time2
    assert n["timestamp"].max() <= unix_time2
