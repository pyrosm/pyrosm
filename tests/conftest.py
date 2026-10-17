"""Shared pytest configuration for the pyrosm test suite."""

import http.client
import os
from urllib.error import URLError

import pytest

pytest_plugins = ["pytester"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_download: downloads from an outside service (Geofabrik, Nominatim); "
        "runs only when RUN_DOWNLOAD_TESTS=true",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the live_download tests unless RUN_DOWNLOAD_TESTS=true.

    CI sets it on the Ubuntu runners of the oldest and newest supported Python only,
    so the outside services are not downloaded from by every job of the matrix.
    """
    if os.environ.get("RUN_DOWNLOAD_TESTS") == "true":
        return
    skip = pytest.mark.skip(
        reason="Live download tests run only on the Ubuntu CI runners of the oldest "
        "and newest supported Python; set RUN_DOWNLOAD_TESTS=true to run locally."
    )
    for item in items:
        if "live_download" in item.keywords:
            item.add_marker(skip)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Skip a live_download test that fails because the outside service failed.

    A download that failed after its retries, a network error or a timeout skips the test
    with the error as the reason; any other error still fails it. A wrong URL built by
    pyrosm is skipped here too; the unit tests cover the URLs.
    """
    if "live_download" not in item.keywords:
        return (yield)
    from pyrosm.exceptions import ExtractDownloadError

    try:
        return (yield)
    except (
        ExtractDownloadError,
        URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ) as err:
        pytest.skip("The outside service failed: %s" % err)


@pytest.fixture(autouse=True)
def _result_cache(tmp_path_factory, monkeypatch):
    """Each test reads and writes the out-of-core result cache in its own empty
    directory (outside its ``tmp_path``), so no test is served a result another test
    cached, and nothing is written to the user's cache."""
    from pyrosm.engine import cache

    cache_dir = tmp_path_factory.mktemp("result_cache")
    monkeypatch.setattr(cache, "cache_dir", lambda: cache_dir)
    return cache_dir


@pytest.fixture
def fresh_cache(_result_cache):
    """The test's own, initially empty, out-of-core result cache directory."""
    return _result_cache


@pytest.fixture(autouse=True)
def _default_workers_warned(monkeypatch):
    """The warning that the default of ``workers`` changes in pyrosm 1.0, shown once per
    process, counts as shown, so no test sees it unless it resets this."""
    import pyrosm.utils

    monkeypatch.setattr(pyrosm.utils, "_default_workers_warned", True)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Retried downloads do not wait between attempts during the tests."""
    from pyrosm.utils import download

    monkeypatch.setattr(download, "_sleep", lambda seconds: None)


def _corrupted_copies(data, count, seed):
    """``count`` corrupted copies of ``data``, the same ones for the same ``seed``: one
    byte changed, the data cut short, bytes inserted, or a run of 0xff bytes (a varint
    that never ends) written over it."""
    import random

    rng = random.Random(seed)
    for _ in range(count):
        copy = bytearray(data)
        kind = rng.randrange(4)
        at = rng.randrange(len(copy))
        if kind == 0:
            copy[at] ^= rng.randrange(1, 256)
        elif kind == 1:
            del copy[at:]
        elif kind == 2:
            copy[at:at] = rng.randbytes(rng.randrange(1, 9))
        else:
            copy[at : at + 11] = b"\xff" * 11
        yield bytes(copy)


@pytest.fixture
def corrupted_copies():
    """A function that gives seeded corrupted copies of some bytes (see
    :func:`_corrupted_copies`), for checking that a parser raises on them instead of
    crashing."""
    return _corrupted_copies
