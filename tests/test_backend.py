"""Guard the protobuf backend pyrosm parses PBF data with.

pyrosm relies on Google's protobuf. The pure-Python implementation is an order
of magnitude slower, so a fallback to it is treated as an unsupported
configuration and must fail loudly rather than degrade silently.
"""
import pytest


def test_protobuf_fast_backend_is_active():
    from google.protobuf.internal import api_implementation

    assert api_implementation.Type() != "python", (
        "protobuf fell back to its pure-Python backend; expected the C/upb "
        "backend. Parsing would be much slower."
    )


def test_pbfreader_uses_google_protobuf_messages():
    import pyrosm.pbfreader as pbfreader
    from google.protobuf.message import Message

    assert issubclass(pbfreader.Blob, Message)
    assert issubclass(pbfreader.PrimitiveBlock, Message)


def test_pure_python_backend_emits_warning(monkeypatch):
    import pyrosm.pbfreader as pbfreader
    from google.protobuf.internal import api_implementation
    from pyrosm import get_data

    # Force the slow-backend branch and reset the once-per-process guard.
    monkeypatch.setattr(api_implementation, "Type", lambda: "python")
    monkeypatch.setattr(pbfreader, "_warned_slow_backend", False)

    fp = get_data("test_pbf")
    with pytest.warns(RuntimeWarning, match="pure-Python backend"):
        pbfreader.parse_osm_data(fp, None, False, None)
