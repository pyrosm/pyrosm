Changelog
=========

v0.6.0
------

- NEW: Adds possibility to export street networks to igraph, networkx and pandana (#57, #58, #70)
  - Add functionality to parse/return the nodes of the network when requested (#52)
  - Calculate length of the edge for networks in meters (#56, #70)
  - Filter out weakly connected component by default when exporting to graph (#59)
  - Add (vectorized) functionality to create directed edges according `oneway` rules (#68)

- Fix installation issue with pip on Windows (#61)
- Fix numpy deprecation warning (#50)
- Update the documentation to use new theme (#74)
- Add possibility to test the tool using JupyterLab in browser (#75)
- Fix issue when parsing POIs using rare tags as a custom filter (#47)
- Fix issue when filtering with bounding box polygon (#54)
- Add documentation about exporting the networks to graphs (#69)
- Improve documentation overall
 

v0.5.3
------

Changes:

- Ensures that geometry construction works with new Pygeos release v0.8.0 (#46)


v0.5.2
------

- Fix data source for New York City 

v0.5.1
------

- Fix multi-level filtering 
- Add support for using "exclude" also with nodes and relations

v0.5.0
------

- Adds a function to download PBF data from Geofabrik and BBBike easily from hundreds of locations across the world
- Improved geometry parsing for relations
- Parse boundary geometries as Polygons instead of LinearRings (following OSM definition) 
- Fix invalid geometries automatically (self-intersection and "bowties")
- Add better documentation about custom filters
- Make parsing more robust for incorrectly tagged OSM entries.
- Bug fixes
- Update website to a new theme.

v0.4.3
------

- Fixes a bug related to filtering with custom filters (see details [here](https://github.com/HTenkanen/pyrosm/issues/22#issuecomment-620005087).)

v0.4.2
------

- Add functionality to parse boundaries from PBF (+ integrate name search for finding e.g. specific administrative boundary)
- Support using Shapely Polygon / MultiPolygon to filter the data spatially
- add possibility to add "extra attributes" (i.e. OSM keys) that will be parsed as columns.
- improve documentation
 
v0.4.1
------

- add documentation 
- create website: https://pyrosm.readthedocs.io

v0.4.0
------

- read PBF using custom queries (allows anything to be fetched)
- read landuse from PBF
- read natural from PBF
- improve geometry parsing so that geometry type is read automatically according OSM rules
- modularize code-base 
- improve test coverage


v0.3.1
------

- generalize code base
- read Points of Interest (POI) from PBF

v0.2.0
------

- read buildings from PBF into GeoDataFrame
- enable applying custom filter to filter data: e.g. with buildings you can filter specific 
types of buildings with `{'building': ['residential', 'retail']}`
- handle Relations as well
- handle cases where data is not available (warn user and return empty GeoDataFrame) 

v0.1.8
------

- read street networks from PBF into GeoDataFrame (separately for driving, cycling, walking and all-combined)
- filter data based on bounding box