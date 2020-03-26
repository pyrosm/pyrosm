# Pyrosm 
[![build status](https://api.travis-ci.com/HTenkanen/pyrosm.svg?branch=master)](https://travis-ci.com/HTenkanen/pyrosm)

**Pyrosm** is a Python library for reading OpenStreetMap from `protobuf` files (`*.osm.pbf`) into Geopandas GeoDataFrames. 
Pyrosm makes it easy to extract various datasets from OpenStreetMap pbf-dumps including road networks, buildings and points of interest. 

The library has been developed by keeping performance in mind, hence, it is mainly written in Cython (*Python with C-like performance*) 
which makes it much faster than any Python alternatives for parsing OpenStreetMap data.
Pyrosm is built on top of another Cython library called [Pyrobuf](https://github.com/appnexus/pyrobuf) which is a faster Cython alternative 
to Google's Protobuf library: It provides 2-4x boost in performance for deserializing the protocol buffer messages compared to 
Google's own Protobuf library with C++ backend. 
 
**Pyrosm** is easy to use and it provides a somewhat similar user interface as another popular Python library [OSMnx](https://github.com/gboeing/osmnx)
for parsing different datasets from the OpenStreetMap pbf-dump including road networks, buildings and points of interest. The main difference between 
pyrosm and OSMnx is that OSMnx reads the data over internet using OverPass API, whereas pyrosm reads the data from local OSM data dumps
that can be downloaded e.g. from [GeoFabrik's website](http://download.geofabrik.de/). This makes it possible to read data much faster thus 
allowing e.g. parsing street networks for whole country in a matter of minutes instead of hours (however, see [caveats](#caveats)).

## Features

 - read street networks (separately for driving, cycling, walking and all-combined)
 - filter data based on bounding box 

## Install

Pyrosm is distributed via PyPi and it can be installed with pip:

`$ pip install pyrosm`  

## How to use?

Using `pyrosm` is easy and it can be imported into Python as any other library. 

To read drivable street networks from OpenStreetMap protobuf file, simply:

```python
from pyrosm import OSM
fp = "mydata.osm.pbf"
# Initialize the OSM parser object
osm = OSM(fp)
osm.get_driving_network(fp)
```   

## Performance

## Caveats

### Filtering large files by bounding box 

Although `pyrosm` provides possibility to filter even larger data files based on bounding box while reading (also lowering memory consumption), 
this process can slow down the reading process significantly (1.5-3x longer) due to necessary lookups when parsing the data. 
This might not be an issue with smaller files (up to ~100MB) but with larger data dumps this can start consuming a lot of 
processing time.

Hence, a recommended approach is to **first** filter the protobuf file based on bounding box into a smaller subset by using a dedicated 
open source Java tool called [Osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) which is available for all operating systems. 
Detailed installation instructions are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Installation), and instructions how to filter
data based on bounding box are [here](https://wiki.openstreetmap.org/wiki/Osmosis/Examples#Extract_administrative_Boundaries_from_a_PBF_Extract).


