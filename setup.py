#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import io
import re
from os.path import dirname
from os.path import join
from os import path
from setuptools import find_packages
from setuptools import setup
import os

# TODO: The installation of Cython, Cykhash and Pyrobuf this way is a hack.
#  Integrate cykhash function directly to pyrosm to avoid these and publish in conda-forge.

# Cython needs to be installed before running setup
# https://luminousmen.com/post/resolve-cython-and-numpy-dependencies
try:
    from Cython.Build import cythonize
except ImportError:
    os.system('pip install Cython')
    from Cython.Build import cythonize

# Cykhash needs to be installed before running setup
try:
    import cykhash
except ImportError:
    os.system('pip install https://github.com/HTenkanen/cykhash/archive/master.zip')

# Pyrobuf needs to be installed before running setup
try:
    import pyrobuf_list
except ImportError:
    os.system('pip install pyrobuf')


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


def read_long_description():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


requirements = [
    'python-rapidjson',
    'setuptools>=18.0',
    'geopandas',
    'pygeos',
]

setup(
    name='pyrosm',
    version='0.5.2',
    license='MIT',
    description='A Python tool to parse OSM data from Protobuf format into GeoDataFrame.',
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author='Henrikki Tenkanen',
    author_email='h.tenkanen@ucl.ac.uk',
    url='https://pyrosm.readthedocs.io/',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    project_urls={
        # 'Documentation': 'https://pyrosm.github.io/',
        'Issue Tracker': 'https://github.com/htenkanen/pyrosm/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],

    python_requires='>=3, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <3.9',
    install_requires=requirements,
    setup_requires=requirements,
    pyrobuf_modules="proto",
    ext_modules=cythonize(os.path.join("pyrosm", "*.pyx"),
                          annotate=False,
                          compiler_directives={'language_level': "3",
                                               #'linetrace': True
                                               }
                          )
)
