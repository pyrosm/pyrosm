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
def default_filter():
    return {"amenity": True,
            "craft": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True
            }


@pytest.fixture
def test_output_dir():
    import os, tempfile
    return os.path.join(tempfile.gettempdir(), "pyrosm_test_results")


def test_parsing_pois_with_defaults(helsinki_pbf, default_filter):
    from pyrosm import OSM
    from pyrosm.pois import get_poi_data
    from geopandas import GeoDataFrame
    import pyproj
    from pyrosm._arrays import concatenate_dicts_of_arrays
    osm = OSM(filepath=helsinki_pbf)
    osm._read_pbf()
    tags_as_columns = []
    for k in default_filter.keys():
        tags_as_columns += getattr(osm.conf.tags, k)

    nodes = concatenate_dicts_of_arrays(osm._nodes)
    gdf = get_poi_data(nodes,
                       osm._node_coordinates,
                       osm._way_records,
                       osm._relations,
                       tags_as_columns,
                       default_filter)

    assert isinstance(gdf, GeoDataFrame)

    # Required keys
    required = ['id', 'geometry']
    for col in required:
        assert col in gdf.columns

    # Test shape
    assert len(gdf) == 1784
    assert gdf.crs == pyproj.CRS.from_epsg(4326)
