Pyrosm
======

Pyrosm is a Python library for reading OpenStreetMap from `Protocolbuffer Binary Format <https://wiki.openstreetmap.org/wiki/PBF_Format>`__ -files (*.osm.pbf)
into `Geopandas <https://geopandas.org/>`__ GeoDataFrames.
Pyrosm makes it easy to extract various datasets from OpenStreetMap pbf-dumps including e.g. road networks, buildings,
Points of Interest (POI), landuse, natural elements, administrative boundaries and much more.
Fully customized queries are supported which makes it possible to parse any kind of data from OSM,
even with more specific filters.

Pyrosm is easy to use and it provides a somewhat similar user interface as `OSMnx <https://github.com/gboeing/osmnx>`__.
The main difference between pyrosm and OSMnx is that OSMnx reads the data using an OverPass API, whereas pyrosm reads
the data from local OSM data dumps that are downloaded from the PBF data providers (Geofabrik, BBBike).
This makes it possible to parse OSM data faster and make it more feasible to extract data covering large regions.

For instance, parsing all roads from the state of New York (USA) with a "basic" work laptop (16GB memory, SSD, and Intel Core i5 CPU),
takes less than **3 minutes** and parsing the buildings from the same region takes less than
**4 minutes** (see `benchmarks <https://pyrosm.readthedocs.io/en/latest/benchmarking.html>`__ for details):

.. figure:: img/NY_roads_and_buildings.PNG


Current features
----------------

- download PBF data easily from hundreds of locations across the world
- read street networks (separately for driving, cycling, walking and all-combined)
- read buildings from PBF
- read Points of Interest (POI) from PBF
- read landuse from PBF
- read "natural" from PBF
- read boundaries from PBF (such as administrative borders)
- read any other data from PBF by using a custom user-defined filter
- filter data based on bounding box
- export networks as a directed graph to `igraph`, `networkx` and `pandana`

When should I use Pyrosm?
-------------------------

Pyrosm can of course be used whenever you need to parse data from OSM into geopandas GeoDataFrames.
However, `pyrosm` is better suited for situations where you want to fetch data for whole city or larger regions (even whole country).

If you are interested to fetch OSM data for smaller areas such as neighborhoods, or search data around a specific location/address,
we recommend using `OSMnx <https://github.com/gboeing/osmnx>`__ which is more flexible in terms of specifying the area of interest.
That being said, it is also possible to extract neighborhood level information with pyrosm and filter data based on a bounding box
(see `docs <https://pyrosm.readthedocs.io/en/latest/basics.html#filtering-data-based-on-bounding-box>`__).

License
-------

Pyrosm is licensed under MIT (see `license <https://github.com/HTenkanen/pyrosm/blob/master/LICENSE>`__).

Data © `Geofabrik GmbH <http://www.geofabrik.de/>`__, `BBBike <https://download.bbbike.org>`__ and `OpenStreetMap Contributors <http://www.openstreetmap.org>`__.
All data from the `OpenStreetMap <https://www.openstreetmap.org>`__ is licensed under the `OpenStreetMap License <https://www.openstreetmap.org/copyright>`__.

Getting started
---------------


.. toctree::
    :caption: Contents

    installation.ipynb
    basics.ipynb
    custom_filter.ipynb
    graphs.ipynb
    benchmarking.ipynb
    contributions.rst

.. toctree::
    :caption: Reference Guide

    Reference to All Attributes and Methods <reference>
    Changelog <changelog>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
