from pathlib import Path

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
def directory():
    import tempfile
    import shutil

    temp_dir = tempfile.gettempdir()
    target_dir = str(Path(temp_dir) / "pyrosm_dir")
    if not Path(target_dir).exists():
        Path(target_dir).mkdir(parents=True)
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
    fp1 = get_data("test_pbf")
    fp2 = get_data("helsinki_pbf")
    fp3 = get_data("helsinki_region_pbf")
    fp4 = get_data("helsinki_history_pbf")
    fp5 = get_data("helsinki_test_history_pbf")
    assert Path(fp1).exists()
    assert Path(fp2).exists()
    assert Path(fp3).exists()
    assert Path(fp4).exists()
    assert Path(fp5).exists()


@pytest.mark.live_download
def test_geofabrik_download_to_temp():
    from pyrosm import get_data

    fp = get_data("monaco", update=True)
    assert Path(fp).exists()


@pytest.mark.live_download
def test_geofabrik_download_to_directory(directory):
    from pyrosm import get_data

    fp = get_data("monaco", update=True, directory=directory)
    assert Path(fp).exists()


def test_get_data_passes_network_options(monkeypatch):
    """get_data hands the caller's headers, timeout and opener to the download."""
    import pyrosm.data as data

    seen = []
    monkeypatch.setattr(
        data, "download", lambda **kwargs: seen.append(kwargs) or "/fake/Helsinki"
    )
    opener = object()
    get_data("helsinki", headers={"User-Agent": "t/1"}, timeout=(5, 30), opener=opener)
    assert seen[0]["filename"] == "Helsinki.osm.pbf"
    net = {k: seen[0][k] for k in ("headers", "timeout", "opener")}
    assert net == {
        "headers": {"User-Agent": "t/1"},
        "timeout": (5, 30),
        "opener": opener,
    }
