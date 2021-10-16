import pytest


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
