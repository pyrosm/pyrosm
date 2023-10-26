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
    "shapely>=2.0.1",
    "cykhash",
    "pyrobuf",
]

setup(
    name="pyrosm",
    version="0.6.2",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://pyrosm.github.io/",
        "Issue Tracker": "https://github.com/htenkanen/pyrosm/issues",
    },
    keywords=[
        "OpenStreetMap",
        "Geopandas",
        "GeoDataFrame",
        "parser",
        "protobuf",
        "PBF",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    pyrobuf_modules="proto",
    ext_modules=cythonize(
        os.path.join("pyrosm", "*.pyx"),
        annotate=False,
        compiler_directives={
            "language_level": "3",
            # 'linetrace': True
        },
    ),
)
