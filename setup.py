#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import io
from os.path import dirname
from os.path import join
from os import path
from setuptools import find_packages
from setuptools import setup
import os
from Cython.Build import cythonize


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


def read_long_description():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


requirements = [
    "python-rapidjson",
    "setuptools>=18.0",
    "geopandas>=0.12.0",
    "shapely>=2.1",
    "cykhash",
    "protobuf>=6.33.5",
]

# Optional line-trace build for measuring Cython (.pyx) test coverage. Enabled
# only when PYROSM_LINETRACE=1 so normal/production builds keep full speed
# (linetrace adds significant per-line overhead). Requires both the Cython
# 'linetrace' directive AND the CYTHON_TRACE / CYTHON_TRACE_NOGIL C macros; the Cython.Coverage
# plugin (see .coveragerc) then reports per-line .pyx coverage. force=_linetrace
# forces regeneration of the C sources whenever tracing is toggled, so a stale
# .c compiled without the macro can never silently leave the .pyx untraced.
_linetrace = os.environ.get("PYROSM_LINETRACE") == "1"
_directives = {"language_level": "3"}
if _linetrace:
    _directives["linetrace"] = True
    _directives["profile"] = True
_ext_modules = cythonize(
    os.path.join("pyrosm", "*.pyx"),
    annotate=False,
    compiler_directives=_directives,
    force=_linetrace,
)
if _linetrace:
    for _ext in _ext_modules:
        # CYTHON_TRACE enables per-line trace hooks; CYTHON_TRACE_NOGIL extends
        # them into nogil sections. Set both explicitly (the latter implies the
        # former in recent Cython, but being explicit is version-robust).
        _ext.define_macros.append(("CYTHON_TRACE", "1"))
        _ext.define_macros.append(("CYTHON_TRACE_NOGIL", "1"))

setup(
    name="pyrosm",
    version="0.8.0",
    license="MIT",
    description="A Python tool to parse OSM data from Protobuf format into GeoDataFrame.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Henrikki Tenkanen",
    author_email="henrikki.tenkanen@aalto.fi",
    url="https://pyrosm.readthedocs.io/",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://pyrosm.readthedocs.org/",
        "Issue Tracker": "https://github.com/pyrosm/pyrosm/issues",
    },
    keywords=[
        "OpenStreetMap",
        "Geopandas",
        "GeoDataFrame",
        "parser",
        "protobuf",
        "PBF",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    ext_modules=_ext_modules,
)
