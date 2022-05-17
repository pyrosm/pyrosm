import pytest
from pyrosm import get_data


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def helsinki_history_pbf():
    pbf_path = get_data("helsinki_test_history_pbf")
    return pbf_path


def test_reading_points_of_interest_with_defaults(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    import pyproj

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_pois()

    assert isinstance(gdf, GeoDataFrame)
    assert len(gdf) == 1712
    assert gdf.crs == pyproj.CRS.from_epsg(4326)

    gdf_cols = gdf.columns.to_list()

    required = ["id", "geometry"]
    for col in required:
        assert col in gdf_cols


def test_reading_points_of_interest_from_area_having_none(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Bounding box for area that does not have any data
    bbox = [24.940514, 60.173849, 24.942, 60.175892]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_pois()
        if "could not find any buildings" in str(w):
            pass

    # Result should be none
    assert gdf is None


def test_passing_incorrect_custom_filter(helsinki_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=helsinki_pbf)
    try:
        osm.get_pois(custom_filter="wrong")
    except ValueError as e:
        if "dictionary" in str(e):
            pass
    except Exception as e:
        raise e


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    import pyproj

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_pois()

    extra_col = "wikidata"
    extra = osm.get_pois(extra_attributes=[extra_col])

    assert extra.shape[1] == gdf.shape[1] + 1
    assert extra_col in extra.columns.to_list()
    assert len(gdf) == len(extra)


def test_using_rare_tag(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_pois({"park_ride": ["yes"]})

    # Result should be none
    assert gdf is None


def test_using_multiple_filters(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_pois({"shop": ["alcohol"], "amenity": ["pub"]})
    shop = gdf["shop"].unique()
    shop = [item for item in shop if isinstance(item, str)]
    amenity = gdf["amenity"].unique().tolist()
    amenity = [item for item in amenity if isinstance(item, str)]

    assert isinstance(gdf, GeoDataFrame)
    assert shop == ["alcohol"]
    assert amenity == ["pub"]
    assert gdf.shape == (59, 33)
