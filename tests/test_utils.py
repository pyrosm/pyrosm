import pytest
from pyrosm import get_data


@pytest.fixture
def helsinki_history_pbf():
    pbf_path = get_data("helsinki_test_history_pbf")
    return pbf_path


def test_timestamp_string():
    from pyrosm.utils import get_unix_time

    # Test that passing date string works
    t = "2021-10-15 07:45"
    unix_time = get_unix_time(t, osh_file=True)
    assert isinstance(unix_time, int)
    assert unix_time == 1634283900


def test_timestamp_integer():
    from pyrosm.utils import get_unix_time

    # Test that passing integer value works
    t = 1634283900
    unix_time = get_unix_time(t, osh_file=True)
    assert isinstance(unix_time, int)
    assert unix_time == 1634283900


def test_timestamp_datetime():
    from pyrosm.utils import get_unix_time
    from datetime import datetime

    # Test that passing date as datetime works
    t = datetime(2021, 10, 15, 7, 45)
    unix_time = get_unix_time(t, osh_file=True)
    assert isinstance(unix_time, int)
    assert unix_time == 1634283900


def test_future_timestamp():
    from pyrosm.utils import get_unix_time

    # Test that future time cannot be passed
    t = "2100-01-01 12:00"
    try:
        unix_time = get_unix_time(t, osh_file=True)
    except ValueError:
        pass
    except Exception as e:
        raise e


def test_timestamp_older_than_OSM_history():
    from pyrosm.utils import get_unix_time

    # Test that older time than OSM history cannot be passed
    t = "2000-01-01 12:00"
    try:
        unix_time = get_unix_time(t, osh_file=True)
    except ValueError:
        pass
    except Exception as e:
        raise e


def test_API_with_timestamp(helsinki_history_pbf):
    from pyrosm import OSM

    osm = OSM(helsinki_history_pbf)
    osm._set_current_time("2021-10-15 07:45")

    # The current timestamp should be unix time as integer
    assert osm._current_timestamp == 1634283900


def test_OSH_file_without_timestamp(helsinki_history_pbf):
    from pyrosm import OSM

    osm = OSM(helsinki_history_pbf)
    # Should give a warning
    with pytest.warns(UserWarning):
        osm._set_current_time(None)

    # Should give warning and update the current_timestamp
    assert osm._current_timestamp > 0
