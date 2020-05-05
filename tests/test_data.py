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


def test_available():
    import pyrosm
    assert isinstance(pyrosm.data.available, dict)


def test_not_available():
    try:
        get_data("file_not_existing")
    except ValueError as e:
        if "is not available" in str(e):
            pass
        else:
            raise e
    except Exception as e:
        raise e
