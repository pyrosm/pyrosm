Changelog
=========

Unreleased
----------

- NEW: Add a ``tags_to_keep`` parameter to the feature methods (``get_network``, ``get_buildings``, ``get_pois``, ``get_landuse``, ``get_natural``, ``get_boundaries``). When given, only those OSM tag keys are kept as columns (replacing the default tag-column set), reducing memory; structural columns, filtering and ``extra_attributes`` are unaffected and the default behaviour is unchanged (#87)
- NEW: Add a ``keep_metadata`` parameter to ``OSM(...)`` (default ``True``). Set ``keep_metadata=False`` to drop the element metadata columns (``timestamp``, ``version``, ``changeset``) from the returned GeoDataFrames and skip decoding the per-node metadata while parsing, lowering memory use and parse time on node-heavy files; the default behaviour is unchanged and history (``.osh.pbf``) files keep the metadata they require (#87, #150)
- CHANGED: Replace the per-node coordinate lookup (previously a dict-of-dicts) with a compact ``cykhash`` id->index map plus contiguous column arrays, cutting the read path's peak memory (~12% on a 138 MB extract) and making coordinate lookups during geometry building a little faster, with no change to the returned data (#53)
- CHANGED: Build network way geometries with a single batched ``shapely.linestrings`` call across ways (and skip the from/to ids and node-attribute records that a plain ``get_network`` discards), cutting network geometry construction ~37% and the multi-layer wall-clock ~17% on a 138 MB extract, with no change to the returned data (#53)
- CHANGED: Stream PBF blocks through a generator (each block is parsed then discarded) instead of holding the whole decompressed file in a list, cutting the read path's peak memory -- ~20% for a whole-file read and ~50-70% for bounding-box reads, which no longer pay the whole-file cost. Output is unchanged; bounding-box reads re-stream the file once to complete boundary geometries, which adds roughly 10% time to those reads (#53)
- CHANGED: When splitting node tags into columns, build only the tag-columns that actually occur in the data instead of materialising every candidate column and dropping the all-empty ones, cutting ``get_pois`` (and other node-feature) assembly ~37% and the multi-layer wall-clock ~14% on a 138 MB extract, with no change to the returned data (#53)
- CHANGED: Apply the same build-only-occurring-columns approach when splitting way and relation tags into columns (``get_network``, ``get_buildings``, ``get_landuse``, ``get_natural`` and custom-criteria reads), so candidate tag-columns absent from the data are no longer materialised and dropped, cutting the way/relation tag-assembly step ~12-16% on a 138 MB extract (the full feature call improves less, as geometry construction dominates these features), with no change to the returned data (#53)
- NEW: Add ``OSM.write_pbf(data, output_path)`` to write the OSM data back to a valid, re-readable PBF after modifying attributes/tags in pandas (e.g. fill ``maxspeed``, add a ``travel_time`` tag). The whole dataset that was read is written; each row of ``data`` (a GeoDataFrame or list of them) updates the tags of the matching element by ``osm_type``+``id``, and rows whose ``id`` is not in the source are added as new elements synthesized from their geometry (Point->node, LineString->way, hole-less Polygon->closed way; negative ids). Topology and coordinates come from the parsed data, so the output round-trips coordinates exactly and is read by pyrosm, osmium, GDAL and r5py/R5 (a modified pedestrian/car network exported from pyrosm builds a routable R5 network). v1 applies edits and additions, not deletions (#286, #285)
- NEW: Add ``OSM.to_pbf(output_path=None, keep_relations=True, workers=1)`` to crop the source ``.osm.pbf`` by the object's ``bounding_box`` and write a valid, re-readable PBF to disk (a temp file when no path is given). Cropping is memory-efficient (streams the file blob-by-blob, holds only id sets) and "complete-ways" (a way is kept when >=1 node is inside the box and keeps its full node list); coordinates round-trip exactly. ``workers>1`` parallelizes the per-block work over a single process pool and produces byte-identical output (with a sequential fallback for files too small to amortize pool startup) (#284, #6)
- CHANGED: Import ``pyrosm`` lazily so ``import pyrosm`` no longer eagerly loads geopandas/shapely (~2 s); ``OSM``, ``get_data`` and ``get_path`` are still importable as before and load on first use (#284, #6)

v0.8.0 (Jun 10, 2026)
---------------------

This is a major release that changes the PBF parsing backend.

- CHANGED: Replace the `Pyrobuf <https://github.com/appnexus/pyrobuf>`_ PBF backend with `Google's Protobuf <https://protobuf.dev/>`_ (its fast C ``upb`` backend) for parsing the protocol-buffer messages. Pyrobuf is unmaintained and its source build fails with modern ``setuptools`` (breaking ``pip install pyrosm``); Google's Protobuf is actively maintained and ships wheels and conda-forge packages for Python 3.10–3.14. Parsing speed is unchanged — see the `backend benchmark <https://github.com/pyrosm/pyrosm/blob/master/benchmarks/README.md>`_. v0.8.0 is the first release built on Google's Protobuf; v0.7.0 was the last to use Pyrobuf. (#276)
- NEW: Automate PyPI releases — a GitHub Actions ``release`` workflow builds binary wheels (cibuildwheel; Linux/macOS/Windows × CPython 3.10–3.14) and an sdist, then publishes them to PyPI via Trusted Publishing and creates a GitHub release when a ``vX.Y.Z`` tag is pushed (#288, #287)
- NEW: Expose relation members under the ``members`` key of the JSON ``tags`` column (each ``{member_id, member_type, member_role}``), so relations carry their members in the returned GeoDataFrame (#281, #216)
- NEW: Raise a clear ``InvalidOSMFileError`` when the input ``.pbf`` is not a valid OSM PBF file, instead of a cryptic zlib/protobuf error (#280, #160)
- NEW: Accept ``pathlib.Path`` (and any ``os.PathLike``) filepaths in the ``OSM`` constructor, not just strings (#279, #145)
- FIXED: Decode node coordinates at full float64 precision (exact OSM 7-decimal values, matching GDAL/osmium); they were truncated to float32, introducing a ~0.1 m error, false extra precision, and visible distortion of straight geometry edges (#283, #245, #225)
- FIXED: Normalize polygon/multipolygon ring orientation to the OGC/GeoJSON right-hand rule (exterior counter-clockwise, holes clockwise), matching osmium and QGIS; previously rings inherited the OSM way node order and were inconsistently wound (#282, #230)
- FIXED: ``get_bounding_box`` now reads the header bounding box correctly; it returned ``None`` for every file after the protobuf backend migration (#280, #160)
- FIXED: Download data over HTTPS using certifi's CA bundle instead of the OS trust store, so fetching datasets no longer fails on Windows with ``ssl.SSLError [ASN1: NOT_ENOUGH_DATA]`` (a CPython bug triggered by a malformed entry in the Windows certificate store) (#294)
- Refresh the README badges (#278)

v0.7.0 (Jun 7, 2026)
--------------------

- NEW: Add ``pandarm`` graph-export backend (the maintained, NumPy 2-compatible fork of pandana); deprecate ``graph_type="pandana"`` (#271)
- NEW: Make cycling networks directed and honour ``oneway:bicycle`` (#255)
- NEW: Add ``custom_filter`` to ``get_network`` so custom-filtered networks also return graph nodes (#264)
- NEW: Add ``street_count`` node attribute to the NetworkX export (compatible with OSMnx ``basic_stats``) (#265)
- NEW: Support combining ``custom_filter`` ``True`` with explicit tag values (#251)
- Support Python 3.10–3.14 (drop 3.9) and fix OSH parsing under pandas 3.0 (#248)

- Return complete (uncut) geometries for ways/edges that straddle a bounding-box edge (#268)
- Keep bounding-box network ``nodes`` consistent with the kept ``edges`` so graph export works without manual cleanup (#269)
- Fix non-dense PBF node parsing (``parse_nodes``) (#275)
- Handle bounding boxes that select no nodes instead of raising ``KeyError`` (#267)
- Fix ``custom_filter`` with ``highway`` turning closed-way polygons into lines (#266)
- Fix network exclude/keep filters leaking on multi-key filters (#263)
- Fix duplicate "phantom" nodes in the NetworkX export (#259)
- Correct relation ids and surface a colliding ``id`` tag as ``id_tag`` (#234, #249)
- Stop ``get_*`` methods from mutating the shared default-tag config (#252)
- Fix spurious pandas chained-assignment warnings from the Cython frame builders (#256)
- Fix Geofabrik UK sub-region downloads (moved under ``united-kingdom``) (#258)
- Fix reading PBF produced by ``osmconvert`` (#238)
- Fix documentation URL (#223)

- Measure Cython (``.pyx``) coverage and raise overall test coverage (#273)
- Document the ``pandarm`` graph backend and the ``pandana`` deprecation, and reading OSM history files (``.osh.pbf``) (#257)
- Fix the Read the Docs build; run live download tests on a single CI runner; bump GitHub Actions to Node 24 (#250, #254, #260)

Thanks for all the contributors who helped to improve the library either via PRs or reporting bugs:

- AnBowell (#233, #234)
- eracle (#238)
- meeuw (#174, #178)
- mattijsdp (#224, #226)
- Jontata (#237)
- anatrk (#112)
- Eph97 (#117)
- gregoriiv (#144)
- chourmo (#170)
- arredond (#176)
- lenkahas (#181)
- rohanaras (#199)
- AdrianKriger (#236)
- my4ng (#239)
- skull3r7 (#241)
- wood-chris (#243)
- llebocq (#247)


v0.6.2 (Oct 26, 2023)
---------------------

- Fix installation issues and support only Python >= 3.9 (#221)
- Fix GA actions and use micromamba to install environments (#221)
- Use Shapely 2.0 instead of pygeos (#214)

Thanks for the following contributors:

- knthis (#214)
- hbruch (#215)


v0.6.1 (Oct 11, 2021)
---------------------

- Support Python 3.9 (#122, #106)
- Use Github Actions for CI (#95)
- Drop Travis CI (#95)
- Add contribution guidelines (#90)
- Follow PEP8 style guide and add linting test for CI using Black

v0.6.0 (Nov 18, 2020)
---------------------

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


v0.5.3 (Sep 13, 2020)
---------------------

Changes:

- Ensures that geometry construction works with new Pygeos release v0.8.0 (#46)

v0.5.1/2 (May 11, 2020)
-----------------------

- Fix multi-level filtering
- Add support for using "exclude" also with nodes and relations
- Fix data source for New York City


v0.5.0 (May 7, 2020)
--------------------

- Adds a function to download PBF data from Geofabrik and BBBike easily from hundreds of locations across the world
- Improved geometry parsing for relations
- Parse boundary geometries as Polygons instead of LinearRings (following OSM definition)
- Fix invalid geometries automatically (self-intersection and "bowties")
- Add better documentation about custom filters
- Make parsing more robust for incorrectly tagged OSM entries.
- Bug fixes
- Update website to a new theme.


v0.4.3 (April 27, 2020)
-----------------------

- Fixes a bug related to filtering with custom filters (see details `here <https://github.com/HTenkanen/pyrosm/issues/22#issuecomment-620005087>`__.)

v0.4.2 (April 23, 2020)
-----------------------

- Add functionality to parse boundaries from PBF (+ integrate name search for finding e.g. specific administrative boundary)
- Support using Shapely Polygon / MultiPolygon to filter the data spatially
- add possibility to add "extra attributes" (i.e. OSM keys) that will be parsed as columns.
- improve documentation

v0.4.1 (April 17, 2020)
-----------------------

- add documentation
- create website: https://pyrosm.readthedocs.io

v0.4.0 (April 16, 2020)
-----------------------

- read PBF using custom queries (allows anything to be fetched)
- read landuse from PBF
- read natural from PBF
- improve geometry parsing so that geometry type is read automatically according OSM rules
- modularize code-base
- improve test coverage

v0.3.1 (April 15, 2020)
-----------------------

- generalize code base
- read Points of Interest (POI) from PBF

v0.2.0 (April 13, 2020)
-----------------------

- read buildings from PBF into GeoDataFrame
- enable applying custom filter to filter data: e.g. with buildings you can filter specific
types of buildings with `{'building': ['residential', 'retail']}`
- handle Relations as well
- handle cases where data is not available (warn user and return empty GeoDataFrame)

v0.1.8 (April 8, 2020)
----------------------

- read street networks from PBF into GeoDataFrame (separately for driving, cycling, walking and all-combined)
- filter data based on bounding box


v0.1.0 (April 7, 2020)
----------------------

- first release on PyPi