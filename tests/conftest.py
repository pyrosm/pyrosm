"""Shared pytest configuration for the pyrosm test suite."""

import os

import pytest


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


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Retried downloads do not wait between attempts during the tests."""
    from pyrosm.utils import download

    monkeypatch.setattr(download, "_sleep", lambda seconds: None)
