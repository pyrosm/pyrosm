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


def test_available():
    import pyrosm
    assert isinstance(pyrosm.data.available, list)


def test_not_available():
    try:
        get_path("file_not_existing")
    except ValueError as e:
        if "is not available" in str(e):
            pass
        else:
            raise e
    except Exception as e:
        raise e


def test_temp_dir():
    import pyrosm
    import os
    assert os.path.isdir(os.path.dirname(
        pyrosm.data._temp_path))
