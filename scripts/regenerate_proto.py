#!/usr/bin/env python
"""Regenerate the vendored protobuf modules in ``pyrosm/proto``.

Run this after editing ``proto/fileformat.proto`` or ``proto/osmformat.proto``::

    python scripts/regenerate_proto.py

Requires ``grpcio-tools`` (bundles ``protoc``). The generated ``*_pb2.py`` files
embed a runtime-version guard tied to the generating protoc version, which sets
the ``protobuf`` lower bound in ``setup.py``. Generating with an older
``grpcio-tools``/``protoc`` lowers that bound if broader compatibility is needed;
keep ``install_requires`` and the ``ci/*-conda.yaml`` files in sync with it.
"""
from __future__ import absolute_import
import os
import sys
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
PROTO_DIR = join(REPO_ROOT, "proto")
OUT_DIR = join(REPO_ROOT, "pyrosm", "proto")
PROTO_FILES = ["fileformat.proto", "osmformat.proto"]


def main():
    from grpc_tools import protoc

    args = [
        "grpc_tools.protoc",
        "-I" + PROTO_DIR,
        "--python_out=" + OUT_DIR,
    ] + [join(PROTO_DIR, name) for name in PROTO_FILES]
    rc = protoc.main(args)
    if rc != 0:
        sys.exit(rc)
    print("Generated %s into %s" % (", ".join(PROTO_FILES), OUT_DIR))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()
