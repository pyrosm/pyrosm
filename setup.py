#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import io
import re
from os.path import dirname
from os.path import join
from setuptools import find_packages
from setuptools import setup
import os
from Cython.Build import cythonize

def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


requirements = ['geopandas',
                'pyrobuf',
                'pygeos',
                'cykhash',
                'cython',
                ]

# extensions = [Extension("pyrosm.khash.khashsets", ["pyrosm/khash/khashsets.pyx"]),
#               Extension("pyrosm.khash.khashsets", ["pyrosm/khash/khashmaps.pyx"]),
#               ]

setup(
    name='pyrosm',
    version='0.0.1',
    license='MIT',
    description='A Python tool to parse OSM data from Protobuf format into GeoDataFrame.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.md')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.md'))
    ),
    author='Henrikki Tenkanen',
    author_email='h.tenkanen@ucl.ac.uk',
    url='https://github.com/htenkanen/PyrOSM',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
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
        'Documentation': 'https://pyrosm.github.io/',
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
                          #compiler_directives={'linetrace': True}
                          )
)

