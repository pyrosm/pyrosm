import email.utils
import urllib.request
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def _http_error(code, retry_after=None, body=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    fp = None if body is None else io.BytesIO(body)
    return HTTPError("https://example.invalid", code, "status", headers, fp)


@pytest.mark.parametrize(
    "outcomes, waits, raised",
    [
        ([_http_error(503), "ok"], [1], None),
        # An error with a response body is closed before the next attempt.
        ([_http_error(502, body=b"Bad gateway"), "ok"], [1], None),
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
    assert all(e.fp.closed for e in outcomes if isinstance(e, HTTPError))
    # An HTTP-date is read against the clock, so its wait is a little under 10 s.
    assert waited == pytest.approx(waits, abs=1)


_BODY = bytes(range(100))
_MODIFIED = "Wed, 30 Sep 2026 10:00:00 GMT"
_STRONG = {"ETag": '"v1"'}


class _Served(io.BytesIO):
    """A response body that drops the connection once ``drop`` bytes of it were read."""

    def __init__(self, data, status, headers, drop=None):
        super().__init__(data)
        self.status, self.headers, self.drop, self.sent = status, headers, drop, 0

    def read(self, size=-1):
        if self.drop is not None and self.sent >= self.drop:
            raise OSError("connection reset")
        if self.drop is not None:
            size = (
                self.drop - self.sent if size < 0 else min(size, self.drop - self.sent)
            )
        chunk = super().read(size)
        self.sent += len(chunk)
        return chunk


@pytest.mark.parametrize(
    "headers, ranges, drops, requests, attempts",
    [
        # A strong ETag resumes at the byte where the connection dropped.
        (_STRONG, "honour", [40, None], [(None, None), ("bytes=40-", '"v1"')], None),
        # A Last-Modified a second older than Date is strong; a weak ETag is not used.
        (
            {
                "ETag": 'W/"v1"',
                "Last-Modified": _MODIFIED,
                "Date": "Wed, 30 Sep 2026 10:00:02 GMT",
            },
            "honour",
            [40, None],
            [(None, None), ("bytes=40-", _MODIFIED)],
            None,
        ),
        # A Last-Modified as new as Date is weak, so the copy restarts.
        (
            {"Last-Modified": _MODIFIED, "Date": _MODIFIED},
            "honour",
            [40, None],
            [(None, None), (None, None)],
            None,
        ),
        ({}, "honour", [40, None], [(None, None), (None, None)], None),
        # An ETag outside the entity-tag grammar, or a folded one, is not replayed.
        ({"ETag": "v1"}, "honour", [40, None], [(None, None), (None, None)], None),
        (
            {"ETag": '"v1"\r\nX-Injected: 1'},
            "honour",
            [40, None],
            [(None, None), (None, None)],
            None,
        ),
        # A Last-Modified with a numeric zone is sent back as a GMT HTTP-date.
        (
            {
                "Last-Modified": "Wed, 30 Sep 2026 12:00:00 +0200",
                "Date": "Wed, 30 Sep 2026 10:00:02 GMT",
            },
            "honour",
            [40, None],
            [(None, None), ("bytes=40-", _MODIFIED)],
            None,
        ),
        # A 200 answer to the range request rewrites the file from the start.
        (_STRONG, "ignore", [40, None], [(None, None), ("bytes=40-", '"v1"')], None),
        # A 206 at the wrong byte, going backwards, longer than its range or not asked for
        # switches resuming off for the rest of the call.
        (
            _STRONG,
            "wrong",
            [40, None, None],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            None,
        ),
        (
            _STRONG,
            "wrong",
            [40, None, 40],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            3,
        ),
        (
            _STRONG,
            "backwards",
            [40, None, None],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            None,
        ),
        (
            _STRONG,
            "long",
            [40, None, None],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            None,
        ),
        (_STRONG, "unsolicited", [None, None], [(None, None), (None, None)], None),
        # A 206 whose Content-Range cannot be read is not appended either.
        (
            _STRONG,
            "garbled",
            [40, None, None],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            None,
        ),
        # A full answer with a content coding drops the validator of the earlier copy.
        (
            _STRONG,
            "recode",
            [40, 60, None],
            [(None, None), ("bytes=40-", '"v1"'), (None, None)],
            None,
        ),
        (
            {**_STRONG, "Content-Encoding": "gzip"},
            "honour",
            [40, None],
            [(None, None), (None, None)],
            None,
        ),
        # Rounds that keep more bytes earn more attempts than the three of one round.
        (
            _STRONG,
            "honour",
            [10, 20, 30, 40, 50, None],
            [(None, None)] + [("bytes=%d-" % n, '"v1"') for n in (10, 20, 30, 40, 50)],
            None,
        ),
        (_STRONG, "honour", [0, 0, 0], [(None, None)] * 3, 3),
        ({}, "honour", [40] * 6, [(None, None)] * 3, 3),
    ],
)
def test_download_resumes(
    tmp_path, monkeypatch, headers, ranges, drops, requests, attempts
):
    """After a dropped connection the download continues with Range and If-Range when the
    server gave a strong validator, and starts over otherwise."""
    from pyrosm.exceptions import DownloadError
    from pyrosm.utils import download as dl

    seen = []

    def urlopen(request, context=None, timeout=None):
        wanted = (request.get_header("Range"), request.get_header("If-range"))
        drop = drops[len(seen)]
        seen.append(wanted)
        start, status, extra = 0, 200, {"Content-Length": "100"}
        if ranges == "unsolicited" and len(seen) == 1:
            partial = {"Content-Range": "bytes 0-39/100", "Content-Length": "40"}
            return _Served(_BODY[:40], 206, {**headers, **partial})
        if wanted[0] and ranges == "recode":
            extra = {**extra, "Content-Encoding": "gzip"}
        elif wanted[0] and ranges != "ignore":
            start = int(wanted[0][6:-1]) - (10 if ranges == "wrong" else 0)
            status = 206
            last = {"backwards": start - 10, "long": start + 9}.get(ranges, 99)
            extra = {"Content-Range": "bytes %d-%d/100" % (start, last)}
            if ranges == "garbled":
                extra = {"Content-Range": "bytes */100"}
            extra["Content-Length"] = str(100 - start)
        cut = None if drop is None else max(0, drop - start)
        return _Served(_BODY[start:], status, {**headers, **extra}, cut)

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    url = "https://example.invalid/x.osm.pbf"
    if attempts is None:
        path = dl.download(url, "x.osm.pbf", True, tmp_path)
        assert Path(path).read_bytes() == _BODY
    else:
        with pytest.raises(DownloadError) as info:
            dl.download(url, "x.osm.pbf", True, tmp_path)
        assert info.value.attempts == attempts
        assert list(tmp_path.iterdir()) == []
    assert seen == requests


class _Opener:
    """An ``OpenerDirector`` stand-in that records requests and returns ``responses``."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        return self.responses[len(self.requests) - 1]


def test_open_url_opener_and_timeouts(monkeypatch):
    """An injected opener gets the request and timeout; a (connect, read) pair builds
    pyrosm's own opener; an opener with a pair is refused."""
    from pyrosm.utils import download as dl

    url = "https://example.invalid/x"
    opener = _Opener("response", "response")
    # The caller's opener needs no certificates from pyrosm and always gets the timeout.
    monkeypatch.setattr(dl.ssl, "create_default_context", lambda **k: 1 / 0)
    assert dl.open_url(url, headers={"User-Agent": "t/1"}, timeout=7, opener=opener)
    assert dl.open_url(url, opener=opener)
    sent = [(r.get_header("User-agent"), t) for r, t in opener.requests]
    assert sent == [("t/1", 7), (dl.USER_AGENT, None)]
    monkeypatch.undo()

    built, own = [], _Opener("response")
    monkeypatch.setattr(
        dl.urllib.request,
        "build_opener",
        lambda *handlers: built.append(handlers) or own,
    )
    assert dl.open_url(url, timeout=(3, 30)) == "response"
    assert [type(h) for h in built[0]] == [dl._HTTPSHandler, dl._HTTPHandler]
    assert own.requests[0][1] == 3

    with pytest.raises(ValueError, match="connect, read"):
        dl.open_url(url, timeout=(3, 30), opener=opener)


def test_timed_connection_sets_read_timeout(monkeypatch):
    """pyrosm's handlers open their requests with timed connection classes, which connect
    with the connect timeout and then set the read timeout on the socket."""
    import http.client

    from pyrosm.utils import download as dl

    class Socket:
        timeout = None

        def settimeout(self, value):
            self.timeout = value

    def connect(self):
        self.sock = Socket()

    monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", connect)
    opened = []
    monkeypatch.setattr(
        urllib.request.AbstractHTTPHandler,
        "do_open",
        lambda self, connection, req, **kw: opened.append((connection, kw)) or "ok",
    )
    context = object()
    assert dl._HTTPSHandler(context, 30).https_open("request") == "ok"
    assert dl._HTTPHandler(30).http_open("request") == "ok"
    assert [kw for _, kw in opened] == [{"context": context}, {}]
    for (connection, _), base in zip(
        opened, (http.client.HTTPSConnection, http.client.HTTPConnection)
    ):
        assert connection is not base and issubclass(connection, base)
        opened_connection = connection("example.invalid", timeout=3)
        opened_connection.connect()
        assert (opened_connection.timeout, opened_connection.sock.timeout) == (3, 30)


def test_download_with_caller_options(tmp_path):
    """download() creates a missing directory, sends the caller's User-Agent through the
    caller's opener, and keeps its own Range when it resumes."""
    from pyrosm.utils import download as dl

    etag = {"ETag": '"v1"', "Content-Length": "100"}
    ranged = {"ETag": '"v1"', "Content-Range": "bytes 40-99/100"}
    opener = _Opener(_Served(_BODY, 200, etag, 40), _Served(_BODY[40:], 206, ranged))
    target = tmp_path / "new" / "dir"
    path = dl.download(
        "https://example.invalid/x.osm.pbf",
        "x.osm.pbf",
        True,
        target,
        headers={
            "User-Agent": "t/1",
            "range": "bytes=0-",
            "Accept-Encoding": "gzip",
            "accept-encoding": "br",
        },
        timeout=9,
        opener=opener,
    )
    assert Path(path).read_bytes() == _BODY and Path(path).parent == target.resolve()
    sent = [
        (r.get_header("User-agent"), r.get_header("Range"), t)
        for r, t in opener.requests
    ]
    assert sent == [("t/1", "bytes=0-", 9), ("t/1", "bytes=40-", 9)]
    # The file is always asked for unencoded, whatever the caller accepts.
    codings = {r.get_header("Accept-encoding") for r, _ in opener.requests}
    assert codings == {"identity"}


@pytest.mark.parametrize(
    "served, calls",
    [
        (
            [_Served(_BODY, 200, {"Content-Length": "100"})],
            [(30, 100), (60, 100), (90, 100), (100, 100)],
        ),
        ([_Served(_BODY, 200, {})], [(30, None), (60, None), (90, None), (100, None)]),
        # A resumed download counts on from the bytes it kept, a restarted one from 0.
        (
            [
                _Served(_BODY, 200, {**_STRONG, "Content-Length": "100"}, 40),
                _Served(_BODY[40:], 206, {"Content-Range": "bytes 40-99/100"}),
            ],
            [(30, 100), (40, 100), (70, 100), (100, 100)],
        ),
        (
            [
                _Served(_BODY, 200, {"Content-Length": "100"}, 40),
                _Served(_BODY, 200, {"Content-Length": "100"}),
            ],
            [(30, 100), (40, 100), (30, 100), (60, 100), (90, 100), (100, 100)],
        ),
    ],
)
def test_download_reports_progress(tmp_path, monkeypatch, capsys, served, calls):
    """A progress callback gets the bytes written and the file size after each chunk, and
    replaces pyrosm's own bar."""
    from pyrosm.utils import download as dl

    monkeypatch.setattr(dl, "_CHUNK", 30)
    monkeypatch.setattr(dl, "_sleep", lambda seconds: None)
    seen = []
    path = dl.download(
        "https://example.invalid/x.osm.pbf",
        "x.osm.pbf",
        True,
        tmp_path,
        opener=_Opener(*served),
        progress=lambda *args: seen.append(args),
    )
    assert Path(path).read_bytes() == _BODY
    assert seen == calls
    assert capsys.readouterr() == ("", "")


def test_download_progress_without_terminal(tmp_path, capsys):
    """Without a terminal, progress=True prints the bar's description once as a line, its
    control characters as spaces; progress=False prints nothing; other values are refused.
    """
    from pyrosm.utils import download as dl
    from pyrosm.utils.progress import Bar

    url = "https://example.invalid/x.osm.pbf"
    for progress, err in ((True, "Downloading x.osm.pbf\n"), (False, "")):
        opener = _Opener(_Served(_BODY, 200, {"Content-Length": "100"}))
        dl.download(url, "x.osm.pbf", True, tmp_path, opener=opener, progress=progress)
        assert capsys.readouterr() == ("", err)
    bar = Bar("a\r\nb\x1b[2J\x85\x7f")
    bar(1, None)
    bar.close()
    assert capsys.readouterr().err == "a  b [2J  \n"
    # Refused before the existing file is returned.
    with pytest.raises(ValueError, match="progress must be"):
        dl.download(url, "x.osm.pbf", False, tmp_path, progress="yes")


def test_bar_follows_a_restart(monkeypatch, capsys):
    """pyrosm's bar goes back with a restarted download and takes its new size, also an
    unknown one."""
    from tqdm import std

    from pyrosm.utils import progress

    monkeypatch.setattr(progress, "_bar_class", lambda: (std.tqdm, False))
    bar = progress.Bar("x.osm.pbf")
    shown = []
    for written, total in ((40, 100), (30, 100), (60, None), (90, None)):
        bar(written, total)
        shown.append((bar.bar.n, bar.bar.total))
    bar.close()
    assert shown == [(40, 100), (30, 100), (60, None), (90, None)]
    assert "x.osm.pbf: " in capsys.readouterr().err


@pytest.mark.parametrize(
    "shell, widgets, expected",
    [
        (None, True, ("std", None)),
        ("TerminalInteractiveShell", True, ("std", None)),
        ("ZMQInteractiveShell", True, ("notebook", False)),
        ("ZMQInteractiveShell", False, ("std", False)),
    ],
)
def test_bar_class(monkeypatch, shell, widgets, expected):
    """In a Jupyter kernel the bar is the widget when ipywidgets is installed, else the text
    bar, shown either way; elsewhere it is the text bar, hidden without a terminal."""
    import sys
    import types

    from tqdm import notebook, std

    from pyrosm.utils import progress

    if shell is None:
        monkeypatch.delitem(sys.modules, "IPython", raising=False)
    else:
        kernel = type(shell, (), {})()
        ipython = types.SimpleNamespace(get_ipython=lambda: kernel)
        monkeypatch.setitem(sys.modules, "IPython", ipython)
    found = object() if widgets else None
    monkeypatch.setattr(progress.importlib.util, "find_spec", lambda name: found)
    classes = {"std": std.tqdm, "notebook": notebook.tqdm}
    assert progress._bar_class() == (classes[expected[0]], expected[1])


def test_open_url_keeps_credentials_on_their_origin():
    """Credentials and cookies from the caller are not sent on after a redirect."""
    import email.message

    from pyrosm.utils import download as dl

    opener = _Opener("response")
    caller = {"User-Agent": "t/1", "Authorization": "Bearer x", "Cookie": "c=1"}
    dl.open_url("https://example.invalid/x", headers=caller, opener=opener)
    request = opener.requests[0][0]
    assert request.get_header("Authorization") == "Bearer x"
    moved = urllib.request.HTTPRedirectHandler().redirect_request(
        request, None, 302, "Found", email.message.Message(), "https://other.invalid/x"
    )
    assert moved.get_header("User-agent") == "t/1"
    assert moved.get_header("Authorization") is None
    assert moved.get_header("Cookie") is None
