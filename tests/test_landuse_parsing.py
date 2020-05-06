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


def test_parsing_landuse_with_defaults(test_pbf):
    from pyrosm import OSM
    from pyrosm.landuse import get_landuse_data
    from geopandas import GeoDataFrame
    import pyproj
    from pyrosm._arrays import concatenate_dicts_of_arrays
    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    tags_as_columns = osm.conf.tags.landuse

    nodes = concatenate_dicts_of_arrays(osm._nodes)
    gdf = get_landuse_data(nodes,
                           osm._node_coordinates,
                           osm._way_records,
                           osm._relations,
                           tags_as_columns,
                           None,
                           None)

    assert isinstance(gdf, GeoDataFrame)

    # Required keys
    required = ['id', 'geometry']
    for col in required:
        assert col in gdf.columns

    # Test shape
    assert len(gdf) == 50
    assert gdf.crs == pyproj.CRS.from_epsg(4326)


def test_reading_landuse_from_area_having_none(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Bounding box for area that does not have any data
    bbox = [24.947241, 60.174997, 24.948240, 60.175716]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)

    # The tool should warn if no buildings were found
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_landuse()
        # Check the warning text
        if "could not find any buildings" in str(w):
            pass

    # Result should be None
    assert gdf is None


def test_passing_incorrect_custom_filter(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    try:
        osm.get_landuse(custom_filter="wrong")
    except ValueError as e:
        if "dictionary" in str(e):
            pass
    except Exception as e:
        raise e


def test_passing_custom_filter_without_element_key(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_landuse(custom_filter={"leisure": True})
    assert isinstance(gdf, GeoDataFrame)


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_landuse()
    extra_col = "wikidata"
    extra = osm.get_landuse(extra_attributes=[extra_col])

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1]+1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)