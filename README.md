# Pyrosm 
[![PyPI version](https://badge.fury.io/py/pyrosm.svg)](https://badge.fury.io/py/pyrosm)[![build status](https://api.travis-ci.org/HTenkanen/pyrosm.svg?branch=master)](https://travis-ci.org/HTenkanen/pyrosm)[![Coverage Status](https://codecov.io/gh/HTenkanen/pyrosm/branch/master/graph/badge.svg)](https://codecov.io/gh/HTenkanen/pyrosm)

**Pyrosm** is a Python library for reading OpenStreetMap from `protobuf` files (`*.osm.pbf`) into Geopandas GeoDataFrames. 
Pyrosm makes it easy to extract various datasets from OpenStreetMap pbf-dumps including e.g. road networks and buildings (points of interest in progress).


The library has been developed by keeping performance in mind, hence, it is mainly written in Cython (*Python with C-like performance*) 
which makes it probably faster than any other Python alternatives for parsing OpenStreetMap data.
Pyrosm is built on top of another Cython library called [Pyrobuf](https://github.com/appnexus/pyrobuf) which is a faster Cython alternative 
to Google's Protobuf library: It provides 2-4x boost in performance for deserializing the protocol buffer messages compared to 
Google's version with C++ backend. Google's Protocol Buffers is a commonly used and efficient method to serialize and compress structured data 
which is also used by OpenStreetMap contributors to distribute the OSM data in PBF format (Protocolbuffer Binary Format). 

 
**Pyrosm** is easy to use and it provides a somewhat similar user interface as another popular Python library [OSMnx](https://github.com/gboeing/osmnx)
for parsing different datasets from the OpenStreetMap pbf-dump including road networks and buildings (later also points of interest and landuse). The main difference between 
pyrosm and OSMnx is that OSMnx reads the data over internet using OverPass API, whereas pyrosm reads the data from local OSM data dumps
that can be downloaded e.g. from [GeoFabrik's website](http://download.geofabrik.de/). This makes it possible to read data much faster thus 
allowing e.g. parsing street networks for whole country in a matter of minutes instead of hours (however, see [caveats](#caveats)).

## Current features

 - read street networks (separately for driving, cycling, walking and all-combined)
 - read buildings from PBF
 - filter data based on bounding box
 - apply custom filter to filter data 
    - e.g. with buildings you can filter specific types of buildings with `{'building': ['residential', 'retail']}` 
 
 
## Roadmap

 - add parsing of places of interests (POIs)
 - add parsing of landuse
 - add more tests

## Install

Pyrosm is distributed via PyPi and it can be installed with pip:

`$ pip install pyrosm`

### Troubleshooting

Notice that `pyrosm` requires geopandas to work. 
On Linux and Mac installing geopandas with `pip` should work without a problem, which is handled automatically when installing pyrosm. 

However, on Windows installing geopandas with pip is likely to cause issues, hence, it is recommended to install Geopandas before installing
`pyrosm`. See instructions from [Geopandas website](https://geopandas.org/install.html#installation).

## How to use?

Using `pyrosm` is straightforward. To read drivable street networks from OpenStreetMap protobuf file (package includes a small test protobuf file), simply:

```python
from pyrosm import OSM
from pyrosm import get_path
fp = get_path("test_pbf")
# Initialize the OSM parser object
osm = OSM(fp)

# Read all drivable roads
# =======================
drive_net = osm.get_network(network_type="driving")
drive_net.head()
...
  access bridge  ...        id                                           geometry
0   None   None  ...   4732994  LINESTRING (26.94310 60.52580, 26.94295 60.525...
1   None   None  ...   5184588  LINESTRING (26.94778 60.52231, 26.94717 60.522...
2   None    yes  ...   5184589  LINESTRING (26.94891 60.52181, 26.94778 60.52231)
3   None   None  ...   5184590  LINESTRING (26.94310 60.52580, 26.94452 60.525...
4   None   None  ...  22731285  LINESTRING (26.93072 60.52252, 26.93094 60.522...

[5 rows x 14 columns]

# Read all residential and retail buildings
# =========================================
custom_filter = {'building': ['residential', 'retail']}
buildings = osm.get_buildings(tag_filters=custom_filter)
buildings.head()
...
      building  ...                                           geometry
0       retail  ...  POLYGON ((26.94511 60.52322, 26.94487 60.52314...
1       retail  ...  POLYGON ((26.95093 60.53644, 26.95083 60.53642...
2  residential  ...  POLYGON ((26.96536 60.52540, 26.96528 60.52539...
3  residential  ...  POLYGON ((26.93920 60.53257, 26.93940 60.53254...
4  residential  ...  POLYGON ((26.96578 60.52129, 26.96569 60.52137...
```   

To get further information how to use the tool, you can use good old `help`:

```python

help(osm.get_network)

...

Help on method get_network in module pyrosm.pyrosm:

get_network(network_type='walking') method of pyrosm.pyrosm.OSM instance
    Reads data from OSM file and parses street networks
    for walking, driving, and cycling.
    
    Parameters
    ----------
    
    network_type : str
        What kind of network to parse. Possible values are: 'walking' | 'cycling' | 'driving' | 'all'.

```

## Examples

For further usage examples (for now), take a look at the tests, such as:
  - [test_network_parsing.py](tests/test_network_parsing.py)
  - [test_building_parsing.py](tests/test_building_parsing.py)

## Performance

Proper benchmarking results are on their way, but to give some idea, reading all drivable roads in Helsinki Region (approx. 85,000 roads) 
takes approximately **12 seconds** (laptop with 16GB memory, SSD drive, and Intel Core i5-8250U CPU 1.6 GHZ). And the result looks something like:

![Helsinki_driving_net](resources/img/Helsinki_driving_net.PNG)

Parsing all buildings from the same area (approx. 180,000) takes approximately **17 seconds**. And the result looks something like:

![Helsinki_building_footprints](resources/img/Helsinki_building_footprints.png)

## Get in touch

If you find a bug from the tool, have question, 
or would like to suggest a new feature to it, you can [make a new issue here](https://github.com/HTenkanen/pyrosm/issues).

## Development

You can install a local development version of the tool by 1) installing necessary packages with conda and 2) building pyrosm from source:

 1. install conda-environment for Python 3.7 or 3.8 by:
 
    - Python 3.7 (you might want to modify the env-name which is `test` by default): `$ conda env create -f ci/37-conda.yaml`
    - Python 3.8: `$ conda env create -f ci/38-conda.yaml`
    
 2. build pyrosm development version from master (activate the environment first):
 
    - `pip install -e .`

## Caveats

### Filtering large files by bounding box 

Although `pyrosm` provides possibility to filter even larger data files based on bounding box, 
this process can slow down the reading process significantly (1.5-3x longer) due to necessary lookups when parsing the data. 
This might not be an issue with smaller files (up to ~100MB) but with larger data dumps this can take longer than necessary.

Hence, a recommended approach with large data files is to **first** filter the protobuf file based on bounding box into a 
smaller subset by using a dedicated open source Java tool called [Osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) 
which is available for all operating systems. Detailed installation instructions are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Installation), 
and instructions how to filter data based on bounding box are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Examples#Extract_administrative_Boundaries_from_a_PBF_Extract).


