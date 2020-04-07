# Pyrosm 
[![build status](https://api.travis-ci.org/HTenkanen/pyrosm.svg?branch=master)](https://travis-ci.org/HTenkanen/pyrosm)[![Coverage Status](https://codecov.io/gh/HTenkanen/pyrosm/branch/master/graph/badge.svg)](https://codecov.io/gh/HTenkanen/pyrosm)

**Pyrosm** is a Python library for reading OpenStreetMap from `protobuf` files (`*.osm.pbf`) into Geopandas GeoDataFrames. 
Pyrosm makes it easy to extract various datasets from OpenStreetMap pbf-dumps including e.g. road networks (buildings and points of interest in progress).


The library has been developed by keeping performance in mind, hence, it is mainly written in Cython (*Python with C-like performance*) 
which makes it probably faster than any other Python alternatives for parsing OpenStreetMap data.
Pyrosm is built on top of another Cython library called [Pyrobuf](https://github.com/appnexus/pyrobuf) which is a faster Cython alternative 
to Google's Protobuf library: It provides 2-4x boost in performance for deserializing the protocol buffer messages compared to 
Google's version with C++ backend. Google's Protocol Buffers is a commonly used and efficient method to serialize and compress structured data 
which is also used by OpenStreetMap contributors do distribute the OSM data in PBF format (Protocolbuffer Binary Format). 

 
**Pyrosm** is easy to use and it provides a somewhat similar user interface as another popular Python library [OSMnx](https://github.com/gboeing/osmnx)
for parsing different datasets from the OpenStreetMap pbf-dump including road networks, buildings and points of interest. The main difference between 
pyrosm and OSMnx is that OSMnx reads the data over internet using OverPass API, whereas pyrosm reads the data from local OSM data dumps
that can be downloaded e.g. from [GeoFabrik's website](http://download.geofabrik.de/). This makes it possible to read data much faster thus 
allowing e.g. parsing street networks for whole country in a matter of minutes instead of hours (however, see [caveats](#caveats)).

## Current features

 - read street networks (separately for driving, cycling, walking and all-combined)
 - filter data based on bounding box 
 
## Road map

 - add parsing of building information
 - add parsing of places of interests (POIs)

## Install

Pyrosm is distributed via PyPi and it can be installed with pip:

`$ pip install pyrosm`  

## How to use?

Using `pyrosm` is easy and it can be imported into Python as any other library. 

To read drivable street networks from OpenStreetMap protobuf file, simply:

```ipython
In  [1]: from pyrosm import OSM
In  [2]: from pyrosm import get_path
In  [3]: fp = get_path("test_pbf")

# Initialize the OSM parser object
In  [4]: osm = OSM(fp)

# Read all drivable roads
In  [5]: drive_net = osm.get_network(net_type="driving")
In  [6]: drive_net.head()
Out [6]:
  access bridge  ...        id                                           geometry
0   None   None  ...   4732994  LINESTRING (26.94310 60.52580, 26.94295 60.525...
1   None   None  ...   5184588  LINESTRING (26.94778 60.52231, 26.94717 60.522...
2   None    yes  ...   5184589  LINESTRING (26.94891 60.52181, 26.94778 60.52231)
3   None   None  ...   5184590  LINESTRING (26.94310 60.52580, 26.94452 60.525...
4   None   None  ...  22731285  LINESTRING (26.93072 60.52252, 26.93094 60.522...

[5 rows x 14 columns]
```   

## Performance

Proper benchmarking results are on their way, but to give some idea, reading all drivable roads in Helsinki Region (approx. 85,000 roads) 
takes approximately **10 seconds** (laptop with 16GB memory, SSD drive, and Intel Core i5-8250U CPU 1.6 GHZ). And the result looks something like:

![Helsinki_driving_net](resources/img/Helsinki_driving_net.PNG)

## Caveats

### Filtering large files by bounding box 

Although `pyrosm` provides possibility to filter even larger data files based on bounding box, 
this process can slow down the reading process significantly (1.5-3x longer) due to necessary lookups when parsing the data. 
This might not be an issue with smaller files (up to ~100MB) but with larger data dumps this can take longer than necessary.

Hence, a recommended approach with large data files is to **first** filter the protobuf file based on bounding box into a 
smaller subset by using a dedicated open source Java tool called [Osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) 
which is available for all operating systems. Detailed installation instructions are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Installation), 
and instructions how to filter data based on bounding box are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Examples#Extract_administrative_Boundaries_from_a_PBF_Extract).


