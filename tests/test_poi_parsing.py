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


def test_parsing_poi_elements(helsinki_pbf, default_filter):
    from pyrosm import OSM
    from pyrosm.pois import get_poi_data
    osm = OSM(filepath=helsinki_pbf)
    osm._read_pbf()
    tags_as_columns = []
    for k in default_filter.keys():
        tags_as_columns += getattr(osm.conf.tags, k)

    ways, relation_ways, relations = get_poi_data(osm._way_records,
                                                  osm._relations,
                                                  tags_as_columns,
                                                  default_filter
                                                  )
    assert isinstance(ways, dict)
    assert isinstance(relation_ways, dict)
    assert isinstance(relations, dict)

    # Required keys
    required = ['id', 'nodes']
    for col in required:
        assert col in ways.keys()

    # Test shape
    assert len(ways["id"]) == 123
    assert len(relation_ways["id"]) == 12
    assert len(relations["id"]) == 5




