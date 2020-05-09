import pytest
from pyrosm import get_data
import sys


def get_source_url(name):
    name = name.lower()
    from pyrosm.data import sources
    for source, available in sources.available.items():
        if source == "cities":
            available = [src.lower() for src in available]
        if name in available:
            return sources.__dict__[source].__dict__[name]["url"]


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def geofabrik_urls():
    from pyrosm.data import sources
    geofabrik_sources = []
    for k, v in sources.available.items():
        if k == "cities":
            continue
        geofabrik_sources += v
    return [get_source_url(name) for name in geofabrik_sources]


@pytest.fixture
def bbbike_urls():
    from pyrosm.data import sources
    cities = sources.available["cities"]
    return [get_source_url(name) for name in cities]


@pytest.fixture
def directory():
    import tempfile
    import os
    import shutil
    temp_dir = tempfile.gettempdir()
    target_dir = os.path.join(temp_dir, 'pyrosm_dir')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    yield target_dir
    # Remove after testing
    shutil.rmtree(target_dir)


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


def test_test_data():
    import os
    fp1 = get_data("test_pbf")
    fp2 = get_data("helsinki_pbf")
    fp3 = get_data("helsinki_region_pbf")
    assert os.path.exists(fp1)
    assert os.path.exists(fp2)
    assert os.path.exists(fp3)


def test_geofabrik_download_to_temp():
    from pyrosm import get_data
    import os
    fp = get_data("monaco", update=True)
    assert os.path.exists(fp)


def test_bbbike_download_to_temp():
    from pyrosm import get_data
    import os
    fp = get_data("UlanBator", update=True)
    assert os.path.exists(fp)


def test_geofabrik_download_to_directory():
    from pyrosm import get_data
    import os
    fp = get_data("monaco", update=True)
    assert os.path.exists(fp)


def test_geofabrik_download_to_directory(directory):
    from pyrosm import get_data
    import os
    fp = get_data("monaco", update=True, directory=directory)
    assert os.path.exists(fp)


def test_bbbike_download_to_directory(directory):
    from pyrosm import get_data
    import os
    fp = get_data("UlanBator", update=True, directory=directory)
    assert os.path.exists(fp)


@pytest.mark.skipif("sys.version_info > (3,6)")
def test_geofabrik_sources(geofabrik_urls):
    import requests
    # There might be some sources that are not available
    not_successful = []
    for url in geofabrik_urls:
        conn = requests.head(url)
        if not conn.ok:
            not_successful.append(url)

    if len(not_successful) > 20:
        msg = "There were significant number of PBF sources unavailable: \n" + \
              "\n".join(not_successful)
        raise ValueError(msg)


@pytest.mark.skipif("sys.version_info > (3,6)")
def test_bbbike_sources(bbbike_urls):
    import requests
    # There might be some sources that are not available
    not_successful = []
    for url in bbbike_urls:
        conn = requests.head(url)
        if not conn.ok:
            not_successful.append(url)

    if len(not_successful) > 20:
        msg = "There were significant number of PBF sources unavailable: \n" + \
              "\n".join(not_successful)
        raise ValueError(msg)

