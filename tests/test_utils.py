import email.utils
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

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


@pytest.mark.parametrize("timeout, expected", [(None, {}), (5, {"timeout": 5})])
def test_open_url_timeout(monkeypatch, timeout, expected):
    from pyrosm.utils import download as dl

    seen = {}

    def urlopen(request, context=None, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    dl.open_url("https://example.invalid", timeout=timeout)
    assert seen == expected


def test_write_atomic_keeps_target_when_write_fails(tmp_path):
    from pyrosm.utils.download import write_atomic

    target = tmp_path / "x.json"
    target.write_text("old")

    def fail(out_file):
        out_file.write(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_atomic(target, fail)
    assert target.read_text() == "old" and sorted(tmp_path.iterdir()) == [target]


def _http_error(code, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return HTTPError("https://example.invalid", code, "status", headers, None)


@pytest.mark.parametrize(
    "outcomes, waits, raised",
    [
        ([_http_error(503), "ok"], [1], None),
        ([_http_error(429, "5"), "ok"], [5], None),
        ([_http_error(503, "in 10 s"), "ok"], [10], None),
        ([_http_error(503, "soon"), "ok"], [1], None),
        ([_http_error(429, "3600")], [], 429),
        ([_http_error(429, "9" * 5000)], [], 429),
        ([_http_error(404)], [], 404),
        ([_http_error(408), _http_error(500), "ok"], [1, 2], None),
        ([OSError("reset")] * 3, [1, 2], "reset"),
    ],
)
def test_retry(monkeypatch, outcomes, waits, raised):
    """Network errors and HTTP 408/425/429/5xx are retried with backoff or Retry-After;
    other statuses and a Retry-After above a minute are raised at once."""
    from pyrosm.utils import download as dl

    waited, calls = [], []
    monkeypatch.setattr(dl, "_sleep", waited.append)
    for outcome in outcomes:
        if getattr(outcome, "headers", {}).get("Retry-After") == "in 10 s":
            when = datetime.now(timezone.utc) + timedelta(seconds=10)
            outcome.headers["Retry-After"] = email.utils.format_datetime(when, True)

    def fetch():
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    if raised is None:
        assert dl._retry(fetch) == "ok"
    else:
        with pytest.raises(OSError) as info:
            dl._retry(fetch)
        assert getattr(info.value, "code", str(info.value)) == raised
    assert len(calls) == len(outcomes)
    # An HTTP-date is read against the clock, so its wait is a little under 10 s.
    assert waited == pytest.approx(waits, abs=1)
