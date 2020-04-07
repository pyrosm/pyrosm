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


def test_invalid_filter_type(test_pbf):
    from pyrosm import OSM
    osm = OSM(filepath=test_pbf)
    try:
        osm.get_network("MyNetwork")
    except ValueError:
        pass
    except Exception as e:
        raise e



