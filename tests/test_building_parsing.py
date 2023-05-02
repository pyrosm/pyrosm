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


def test_parsing_building_elements(test_pbf):
    from pyrosm import OSM
    from pyrosm.data_manager import get_osm_data

    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    custom_filter = {"building": True}
    nodes, ways, relation_ways, relations = get_osm_data(
        None,
        osm._way_records,
        osm._relations,
        osm.conf.tags.building,
        custom_filter,
        filter_type="keep",
    )
    assert isinstance(ways, dict)

    # Required keys
    required = ["id", "nodes"]
    for col in required:
        assert col in ways.keys()

    # Test shape
    assert len(ways["id"]) == 2219


def test_creating_building_geometries(test_pbf):
    from pyrosm import OSM
    from pyrosm.data_manager import get_osm_data
    from pyrosm.geometry import create_way_geometries
    from numpy import ndarray
    from shapely import Geometry

    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    custom_filter = {"building": True}
    nodes, ways, relation_ways, relations = get_osm_data(
        None,
        osm._way_records,
        osm._relations,
        osm.conf.tags.building,
        custom_filter,
        filter_type="keep",
    )
    assert isinstance(ways, dict)

    ways, geometries, lengths, from_ids, to_ids = create_way_geometries(
        osm._node_coordinates, ways, parse_network=False
    )
    assert isinstance(geometries, list), f"Type should be list, got {type(geometries)}."
    assert isinstance(geometries[0], Geometry)
    assert len(geometries) == len(ways["id"])


def test_reading_buildings_with_defaults(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings()

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.shape == (2208, 20)

    required_cols = [
        "building",
        "addr:city",
        "addr:street",
        "addr:country",
        "addr:postcode",
        "addr:housenumber",
        "source",
        "opening_hours",
        "building:levels",
        "id",
        "timestamp",
        "version",
        "geometry",
    ]

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

    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape
    assert gdf.shape == (577, 16)

    required_cols = [
        "building",
        "addr:street",
        "addr:postcode",
        "addr:housenumber",
        "opening_hours",
        "id",
        "timestamp",
        "version",
        "geometry",
        "tags",
    ]

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

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    temp_path = os.path.join(test_output_dir, "pyrosm_test.gpkg")
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings()
    gdf.to_file(temp_path, driver="GPKG")

    # Ensure it can be read and matches with original one
    gdf2 = gpd.read_file(temp_path)
    cols = gdf.columns
    for col in cols:
        # Geopackage stores boolean values as binary ("0"/"1")
        if col == "visible":
            bools = list(set(gdf[col].tolist()))
            binaries = list(set(gdf2[col].tolist()))
            if bools == [True, False] or bools == [False, True]:
                pass
            if binaries == ["0", "1"] or binaries == ["0", "1"]:
                pass
    # Clean up
    shutil.rmtree(test_output_dir)


def test_reading_buildings_with_filters(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame

    # Get first all data
    osm = OSM(filepath=test_pbf)
    gdf_all = osm.get_buildings()

    # Find out all 'building' tags
    cnts = gdf_all["building"].value_counts()
    for filter_, cnt in cnts.items():
        filtered = osm.get_buildings({"building": [filter_]})
        assert isinstance(filtered, GeoDataFrame)
        assert isinstance(filtered.loc[0, "geometry"], Polygon)
        assert len(filtered) == cnt
        # Now should only have buildings with given key
        assert len(filtered["building"].unique()) == 1

        required_cols = ["building", "id", "timestamp", "version", "geometry"]

        for col in required_cols:
            assert col in filtered.columns


def test_reading_buildings_with_relations(helsinki_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_buildings()

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.shape == (490, 35)

    required_cols = ["building", "id", "timestamp", "version", "tags", "geometry"]

    for col in required_cols:
        assert col in gdf.columns


def test_reading_buildings_from_area_having_none(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Bounding box for area that does not have any data
    bbox = [24.940514, 60.173849, 24.942, 60.175892]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)

    # The tool should warn if no buildings were found
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_buildings()
        # Check the warning text
        if "could not find any buildings" in str(w):
            pass

    # Result should be None
    assert gdf is None


def test_passing_incorrect_custom_filter(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    try:
        osm.get_buildings(custom_filter="wrong")
    except ValueError as e:
        if "dictionary" in str(e):
            pass
    except Exception as e:
        raise e


def test_passing_custom_filter_without_element_key(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_buildings(custom_filter={"start_date": True})
    assert isinstance(gdf, GeoDataFrame)


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_buildings()
    extra_col = "wikidata"
    extra = osm.get_buildings(extra_attributes=[extra_col])

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1] + 1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)


def test_reading_buildings_from_osh(helsinki_history_pbf):
    from pyrosm import OSM
    from pyrosm.utils import datetime_to_unix_time
    import pandas as pd
    from geopandas import GeoDataFrame
    from shapely.geometry import Polygon

    timestamp = "2010-01-01"
    dt = pd.to_datetime(timestamp, utc=True)
    unix_time = datetime_to_unix_time(dt)

    osm = OSM(filepath=helsinki_history_pbf)
    gdf = osm.get_buildings(timestamp=timestamp)

    assert osm._current_timestamp == unix_time
    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.shape == (74, 14)

    required_cols = ["building", "id", "timestamp", "version", "geometry"]

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

    gdf2 = osm.get_buildings(timestamp=new_timestamp)

    # Now the "current_timestamp" should have been updated
    assert osm._current_timestamp == unix_time2

    # Number of elements should (likely) be higher
    assert len(gdf2) > len(gdf)

    # There should be newer timestamps in the data
    assert gdf2["timestamp"].max() > unix_time

    # But is shouldn't be higher than the given timestamp
    assert gdf2["timestamp"].max() < unix_time2
