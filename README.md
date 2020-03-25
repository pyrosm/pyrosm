# PyrOSM

**PyrOSM** is a `fast` Python library for reading OpenStreetMap from `protobuf` files (`*.osm.pbf`) into Geopandas GeoDataFrames. 
The library has been developed by keeping performance in mind, hence, it is mainly written in Cython (*Python with C-like performance*).
PyrOSM is built on top of another Cython library called [Pyrobuf](https://github.com/appnexus/pyrobuf) which is a faster Cython alternative 
to Google's Protobuf library: It provides 2-4x boost in performance for deserializing the protocol buffer messages compared to 
Google's own Pyrobuf library with C++ backend. 
 
**PyrOSM** is also easy to use and it provides a somewhat similar user interface as another popular Python library [OSMnx](https://github.com/gboeing/osmnx)
for parsing different datasets from the OpenStreetMap pbf-dump including road networks, buildings and points of interest. The main difference between 
PyrOSM and OSMnx is that OSMnx reads the data over internet using OverPass API, whereas PyrOSM reads the data from local OSM data dumps
that can be downloaded e.g. from [GeoFabrik's website](http://download.geofabrik.de/). This makes it possible to read data much faster thus 
allowing e.g. parsing street networks for whole country in a matter of minutes instead of hours (however, see [caveats](#caveats)).  

## Caveats


