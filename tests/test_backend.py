"""Checks on the protobuf backend pyrosm reads and writes PBF data with.

pyrosm relies on Google's protobuf. The pure-Python implementation is an order
of magnitude slower. This test verifies that the fast (C/upb) backend is in use.
"""


def test_protobuf_fast_backend_is_active():
    from google.protobuf.internal import api_implementation

    assert api_implementation.Type() != "python", (
        "protobuf fell back to its pure-Python backend; expected the C/upb "
        "backend. Writing PBF files would be much slower."
    )
