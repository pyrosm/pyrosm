Changelog
=========

v0.10.0
-------

This release adds an opt-in out-of-core ("streaming") reading engine that decodes large PBF files in a single streaming pass with bounded memory — the decode, the node-coordinate gather and the standalone-way read all run in parallel across a worker pool, and each layer's result is cached automatically — making whole-country and even whole-continent extracts (e.g. South America) readable quickly on modest machines without running out of memory (tested on Macbook Air with 24 GB of RAM). `get_data` can now also download whole-continent extracts. The default in-memory reader is unchanged.

- NEW: Add an opt-in out-of-core reading engine, selected with `OSM(filepath, engine="out_of_core")` (the default `engine="in_memory"` is unchanged). It decodes the PBF in a single streaming pass — each block's node coordinates and matching features are spilled to files on disk, then only the coordinates the kept features reference are gathered to assemble geometries — so peak memory is bounded by the working set rather than the whole file, making whole-country extracts readable on modest machines, resolving the out-of-memory errors and kernel crashes previously reported when reading large extracts ([#111](https://github.com/pyrosm/pyrosm/issues/111), [#147](https://github.com/pyrosm/pyrosm/issues/147), [#166](https://github.com/pyrosm/pyrosm/issues/166), [#205](https://github.com/pyrosm/pyrosm/issues/205)). Every feature method (`get_network`, `get_buildings`, `get_pois`, `get_landuse`, `get_natural`, `get_boundaries`, `get_data_by_custom_criteria`) and the options `custom_filter`, `bounding_box`, `complete_relations`, `extra_attributes`/`tags_to_keep` and `keep_metadata` behave as on the in-memory reader and return column-for-column identical GeoDataFrames; `get_network(nodes=True)` returns the graph-export nodes and edges. History (`.osh.pbf`) files and timestamped reads are served by the in-memory reader ([#321](https://github.com/pyrosm/pyrosm/pull/321), [#322](https://github.com/pyrosm/pyrosm/pull/322), [#323](https://github.com/pyrosm/pyrosm/pull/323), [#324](https://github.com/pyrosm/pyrosm/pull/324), [#325](https://github.com/pyrosm/pyrosm/pull/325), [#327](https://github.com/pyrosm/pyrosm/pull/327), [#329](https://github.com/pyrosm/pyrosm/pull/329))
- NEW: The out-of-core engine caches each layer's result to a GeoParquet file under a temporary directory, keyed by the source file and the read parameters, and reuses it on identical later reads — even in a later Python session — instead of re-decoding the PBF. Caching uses the optional `pyarrow` package; without it the reader still works but returns an in-memory GeoDataFrame without caching. Passing `output="path"` streams a layer straight to a GeoParquet file instead ([#330](https://github.com/pyrosm/pyrosm/pull/330), [#331](https://github.com/pyrosm/pyrosm/pull/331))
- NEW: Control out-of-core parallelism with a `workers` parameter on `OSM(...)`. By default the engine reads on a single core and the first out-of-core read reports how many CPU cores are available; pass `workers="auto"` to let pyrosm pick the count by file size (one worker per CPU core above ~70 MB, capped at the data-blob count), or `workers=N` for an explicit count (reduced to the CPU-core count, with a warning, if it exceeds it). On macOS and Windows a parallel read launched from a standalone script must run under an `if __name__ == "__main__":` guard; without it the read falls back to a single process with a warning ([#326](https://github.com/pyrosm/pyrosm/pull/326), [#328](https://github.com/pyrosm/pyrosm/pull/328), [#334](https://github.com/pyrosm/pyrosm/pull/334))
- NEW: Add `OSM` helper methods for managing the out-of-core engine's temporary files. `OSM.list_cache()` and `OSM.clear_cache()` list and remove the cached layer GeoParquet files, and `OSM.list_downloads()` and `OSM.clear_downloads()` list and remove the `*.osm.pbf` extracts `get_data` downloads to the temp directory. `list_cache`/`clear_cache` and `clear_downloads` take an optional source file to scope the operation to a single file; the bundled package datasets are never touched ([#336](https://github.com/pyrosm/pyrosm/pull/336))
- NEW: Add a `keep_other_tags` parameter (default `True`) to the out-of-core `get_data_by_custom_criteria`. With `keep_other_tags=False` the engine resolves only the requested tags (`tags_as_columns` plus the filter keys) and returns no JSON `tags` column of leftover tags, so a geometry-and-key read skips the long-tail tag parsing (and the leftover column that would have been dropped is no longer built at all). Only the out-of-core engine supports it — the in-memory reader raises ([#337](https://github.com/pyrosm/pyrosm/pull/337), [#340](https://github.com/pyrosm/pyrosm/pull/340))
- NEW: `get_data` now downloads whole-continent extracts — `get_data("south_america")` (and the other continents) resolves to the continent-wide Geofabrik `*-latest.osm.pbf` instead of raising, so an entire continent can be fetched in one call and then read with the out-of-core engine's bounded memory ([#339](https://github.com/pyrosm/pyrosm/pull/339))
- NEW: Add Spain's autonomous communities as Geofabrik sub-region sources, so each can be downloaded individually with `get_data` ([#203](https://github.com/pyrosm/pyrosm/pull/203))
- CHANGED: Parallelise the out-of-core engine's collect phase. After the parallel decode, the node-coordinate gather and the standalone-way read are split across the same worker pool (the way records are carried as columnar arrays so they cross process boundaries cheaply) instead of running single-process. On country-scale reads this collect phase is the dominant post-decode cost, so parallelising it markedly speeds up large reads, with byte-identical output ([#338](https://github.com/pyrosm/pyrosm/pull/338))
- CHANGED: Spill the out-of-core engine's intermediate shards coarser — decode now batches several PrimitiveBlocks into one shard (up to a byte target) rather than writing one shard per ~8k-element block, so a country file produces a few hundred shards instead of tens of thousands of tiny files. This cuts the per-file open/read overhead the collect phase pays re-reading them (roughly halving the collect time on a Finland buildings read), with no change to the output, and keeps peak memory bounded by about one shard per worker ([#342](https://github.com/pyrosm/pyrosm/pull/342))

Thanks for all the contributors who helped to improve the library either via PRs, or by raising or participating in an issue:

- marcbosch-idencity ([#111](https://github.com/pyrosm/pyrosm/issues/111))
- MarkKotwicz ([#111](https://github.com/pyrosm/pyrosm/issues/111))
- mmz15 ([#111](https://github.com/pyrosm/pyrosm/issues/111))
- tjnangosha ([#111](https://github.com/pyrosm/pyrosm/issues/111))
- gegen07 ([#147](https://github.com/pyrosm/pyrosm/issues/147))
- webcoderz ([#166](https://github.com/pyrosm/pyrosm/issues/166))
- ricsatjr ([#205](https://github.com/pyrosm/pyrosm/issues/205))
- rgreinho ([#203](https://github.com/pyrosm/pyrosm/pull/203))

v0.9.1
------

This release adds reading all tagged data without a filter and correct relation geometries for bounding-box reads, speeds up area-feature (Polygon) geometry building, and fixes incomplete-boundary handling and ambiguous region-name lookups.

- NEW: Add a `complete_relations` parameter to `OSM(...)` (default `False`). When reading with a `bounding_box`, a relation (e.g. a multipolygon or boundary) is assembled from only the member ways inside the box, so a relation straddling the box edge gets a partial geometry. With `complete_relations=True` the reader fetches each such relation's full member set (member ways and their nodes, even outside the box) so the geometry is correct, applying to every relation-returning layer (`get_buildings`, `get_landuse`, `get_natural`, `get_boundaries`, `get_pois`, `get_data_by_custom_criteria`). It is opt-in: the default reproduces the existing output, and completion adds two streaming passes over the file only when a relation actually has missing members. The fetched member ways are kept out of the normal way features, so other layers (e.g. `get_network`) are unaffected. Only member ways are completed; relations whose members are themselves relations (super-relations) are not. A whole-file read (no `bounding_box`) already holds every member, so the option is a no-op there. When a bounding-box read returns relations the box cut and `complete_relations` was not enabled, a `UserWarning` reports how many relations were returned with incomplete geometry and points to the option ([#236](https://github.com/pyrosm/pyrosm/issues/236))
- NEW: `get_data_by_custom_criteria` now accepts `custom_filter=None` (the new default) to read every **tagged** element without enumerating tag keys — tagged nodes as Points, ways as Lines/Polygons and relations as (Multi)Polygons/Lines; standalone untagged ways are dropped (matching GDAL's OSM driver). When `tags_as_columns` is not given it defaults to the union of the per-feature default tag columns, so common keys become columns and the rest land in the JSON `tags` column, and `keep_metadata`/`keep_nodes`/`keep_ways`/`keep_relations` still apply ([#113](https://github.com/pyrosm/pyrosm/issues/113))
- FIXED: `get_boundaries()` no longer force-closes incomplete administrative boundaries into polygons. A boundary relation whose member ways run off the PBF extent cannot form a closed ring, and was previously closed with a spurious straight edge bridging the gap (the stray lines crossing boundary plots); such incomplete boundaries are now dropped, matching how osmium and GDAL skip areas they cannot assemble ([#154](https://github.com/pyrosm/pyrosm/issues/154))
- FIXED: `get_data()` no longer silently returns the wrong extract when a dataset name is shared by multiple regions — e.g. `get_data("georgia")` (the US state vs. the country) now raises a `ValueError` listing the region-qualified alternatives instead of returning the first match. Region-qualified names are accepted (`get_data("usa/georgia")` vs. `get_data("europe/georgia")`), and names that resolve to the same file (e.g. a UK county reachable via both `great_britain` and `united_kingdom`) still resolve as before ([#162](https://github.com/pyrosm/pyrosm/issues/162))
- CHANGED: Vectorise closed-area (Polygon) way geometry construction — build the geometries for closed area ways (e.g. `get_buildings`, `get_landuse`, `get_natural`) with a single batched `shapely.polygons`/`shapely.linearrings` call instead of a per-way Python loop, falling back to the exact per-way builder for the rest (open ways, closed ways tagged as linear features, ways with dropped nodes), so the output is identical. Cuts `get_buildings` wall-clock ~28% on an Estonia extract (52.0 s -> 37.4 s), complementing the network-geometry vectorisation added in v0.9.0 ([#315](https://github.com/pyrosm/pyrosm/pull/315))

Thanks for all the contributors who helped to improve the library either via PRs or reporting bugs:

- Eph97 ([#113](https://github.com/pyrosm/pyrosm/issues/113))
- Zigur ([#154](https://github.com/pyrosm/pyrosm/issues/154))
- jspalink ([#162](https://github.com/pyrosm/pyrosm/issues/162))
- AdrianKriger ([#236](https://github.com/pyrosm/pyrosm/issues/236))

v0.9.0
------

This release lowers the read path's memory use and speeds it up, adds fetching data by bounding box and by place name, and writes OSM data back to PBF (editing attributes/tags and cropping).

- NEW: Add a `tags_to_keep` parameter to the feature methods (`get_network`, `get_buildings`, `get_pois`, `get_landuse`, `get_natural`, `get_boundaries`). When given, only those OSM tag keys are kept as columns (replacing the default tag-column set), reducing memory; structural columns, filtering and `extra_attributes` are unaffected and the default behaviour is unchanged ([#87](https://github.com/pyrosm/pyrosm/issues/87))
- NEW: Add a `keep_metadata` parameter to `OSM(...)` (default `True`). Set `keep_metadata=False` to drop the element metadata columns (`timestamp`, `version`, `changeset`) from the returned GeoDataFrames and skip decoding the per-node metadata while parsing, lowering memory use and parse time on node-heavy files; the default behaviour is unchanged and history (`.osh.pbf`) files keep the metadata they require ([#87](https://github.com/pyrosm/pyrosm/issues/87), [#150](https://github.com/pyrosm/pyrosm/issues/150))
- CHANGED: Replace the per-node coordinate lookup (previously a dict-of-dicts) with a compact `cykhash` id→index map plus contiguous column arrays, cutting the read path's peak memory (~12% on a 138 MB extract) and making coordinate lookups during geometry building a little faster, with no change to the returned data ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- CHANGED: Build network way geometries with a single batched `shapely.linestrings` call across ways (and skip the from/to ids and node-attribute records that a plain `get_network` discards), cutting network geometry construction ~37% and the multi-layer wall-clock ~17% on a 138 MB extract, with no change to the returned data ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- CHANGED: Stream PBF blocks through a generator (each block is parsed then discarded) instead of holding the whole decompressed file in a list, cutting the read path's peak memory — ~20% for a whole-file read and ~50–70% for bounding-box reads, which no longer pay the whole-file cost. Output is unchanged; bounding-box reads re-stream the file once to complete boundary geometries, which adds roughly 10% time to those reads ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- CHANGED: When splitting node tags into columns, build only the tag-columns that actually occur in the data instead of materialising every candidate column and dropping the all-empty ones, cutting `get_pois` (and other node-feature) assembly ~37% and the multi-layer wall-clock ~14% on a 138 MB extract, with no change to the returned data ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- CHANGED: Apply the same build-only-occurring-columns approach when splitting way and relation tags into columns (`get_network`, `get_buildings`, `get_landuse`, `get_natural` and custom-criteria reads), so candidate tag-columns absent from the data are no longer materialised and dropped, cutting the way/relation tag-assembly step ~12–16% on a 138 MB extract (the full feature call improves less, as geometry construction dominates these features), with no change to the returned data ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- NEW: Add `OSM.write_pbf(data, output_path)` to write the OSM data back to a valid, re-readable PBF after modifying attributes/tags in pandas (e.g. fill `maxspeed`, add a `travel_time` tag). The whole dataset that was read is written; each row of `data` (a GeoDataFrame or list of them) updates the tags of the matching element by `osm_type`+`id`, and rows whose `id` is not in the source are added as new elements synthesized from their geometry (Point->node, LineString->way, hole-less Polygon->closed way; negative ids). Topology and coordinates come from the parsed data, so the output round-trips coordinates exactly and is read by pyrosm, osmium, GDAL and r5py/R5 (a modified pedestrian/car network exported from pyrosm builds a routable R5 network). v1 applies edits and additions, not deletions ([#286](https://github.com/pyrosm/pyrosm/issues/286), [#285](https://github.com/pyrosm/pyrosm/issues/285))
- NEW: Add `OSM.to_pbf(output_path=None, keep_relations=True, workers=1, compact=False, repack=False)` to crop the source `.osm.pbf` by the object's `bounding_box` and write a valid, re-readable PBF to disk (a temp file when no path is given). Cropping is memory-efficient (streams the file blob-by-blob, holds only id sets) and "complete-ways" (a way is kept when >=1 node is inside the box and keeps its full node list); coordinates round-trip exactly. `workers>1` parallelizes the per-block work over a single process pool and produces byte-identical output (with a sequential fallback for files too small to amortize pool startup). `compact=True` prunes each output block's string table to only the strings its kept elements reference, producing a smaller file (~18% smaller on a Helsinki-region crop) at the cost of some extra per-block work. `repack=True` goes further and re-chunks the kept elements into canonical, densely packed blocks (as `osmium`/Osmosis produce) for the smallest output, at the cost of speed (the re-pack write is sequential, though `workers` still parallelizes the selection); it supersedes `compact`. The defaults `compact=False`/`repack=False` keep the faster current behaviour, and the written OSM data — coordinates, tags and element metadata — is identical for every combination ([#284](https://github.com/pyrosm/pyrosm/issues/284), [#6](https://github.com/pyrosm/pyrosm/issues/6))
- CHANGED: Import `pyrosm` lazily so `import pyrosm` no longer eagerly loads geopandas/shapely (~2 s); `OSM`, `get_data` and `get_path` are still importable as before and load on first use ([#284](https://github.com/pyrosm/pyrosm/issues/284), [#6](https://github.com/pyrosm/pyrosm/issues/6))
- NEW: Add `pyrosm.get_data_by_bbox(bbox, crop=True, download=True, update=False, directory=None, output_path=None)` to download — and by default crop — the OSM data covering a bounding box. It finds the smallest Geofabrik extract whose extent fully covers the box, downloads it, and (with `crop=True`, the default) crops it to the box, returning the cropped file named `bbox_<minx>_<miny>_<maxx>_<maxy>.osm.pbf`; `crop=False` returns the full extract, and `download=False` returns the covering extract's PBF URL without downloading. The bounding box may be a `[minx, miny, maxx, maxy]` list/tuple/array in lon/lat, a Shapely geometry, or a GeoDataFrame/GeoSeries. It is backed by a vendored snapshot of Geofabrik's `index-v1.json`, so the extract lookup works offline (refresh it with `scripts/update_geofabrik_index.py`); `update=True` refreshes the index and re-downloads ([#165](https://github.com/pyrosm/pyrosm/issues/165), [#197](https://github.com/pyrosm/pyrosm/issues/197))
- NEW: Add `pyrosm.geocode(query)` and `pyrosm.get_data_by_geocoding(query, crop=True, download=True, update=False, directory=None, output_path=None)` to fetch data by place name. `geocode` returns a Shapely polygon for a place (e.g. `"Brighton and Hove, UK"`) via OpenStreetMap's Nominatim service — its boundary polygon when available, otherwise its bounding-box rectangle. `get_data_by_geocoding` geocodes the place, then downloads — and by default crops — the smallest Geofabrik extract that covers it, returning the cropped file named after the place (e.g. `brighton-and-hove-uk.osm.pbf`); `crop=False` returns the full extract and `download=False` returns the covering extract's PBF URL. No new dependencies (stdlib `urllib`/`json` with the bundled `certifi`, and `shapely`) ([#165](https://github.com/pyrosm/pyrosm/issues/165))

Thanks for all the contributors who helped to improve the library either via PRs or reporting bugs:

- Padarn ([#53](https://github.com/pyrosm/pyrosm/issues/53))
- majkshkurti ([#150](https://github.com/pyrosm/pyrosm/issues/150))
- emiliovfx ([#165](https://github.com/pyrosm/pyrosm/issues/165))
- bstrdsmkr ([#197](https://github.com/pyrosm/pyrosm/issues/197))

v0.8.0
------

This is a major release that changes the PBF parsing backend.

- CHANGED: Replace the [Pyrobuf](https://github.com/appnexus/pyrobuf) PBF backend with [Google's Protobuf](https://protobuf.dev/) (its fast C `upb` backend) for parsing the protocol-buffer messages. Pyrobuf is unmaintained and its source build fails with modern `setuptools` (breaking `pip install pyrosm`); Google's Protobuf is actively maintained and ships wheels and conda-forge packages for Python 3.10–3.14. Parsing speed is unchanged — see `benchmarks/README.md`. v0.8.0 is the first release built on Google's Protobuf; v0.7.0 was the last to use Pyrobuf. ([#276](https://github.com/pyrosm/pyrosm/issues/276))
- NEW: Automate PyPI releases — a GitHub Actions `release` workflow builds binary wheels (cibuildwheel; Linux/macOS/Windows × CPython 3.10–3.14) and an sdist, then publishes them to PyPI via Trusted Publishing and creates a GitHub release when a `vX.Y.Z` tag is pushed ([#288](https://github.com/pyrosm/pyrosm/issues/288), [#287](https://github.com/pyrosm/pyrosm/issues/287))
- NEW: Expose relation members under the `members` key of the JSON `tags` column (each `{member_id, member_type, member_role}`), so relations carry their members in the returned GeoDataFrame ([#281](https://github.com/pyrosm/pyrosm/issues/281), [#216](https://github.com/pyrosm/pyrosm/issues/216))
- NEW: Raise a clear `InvalidOSMFileError` when the input `.pbf` is not a valid OSM PBF file, instead of a cryptic zlib/protobuf error ([#280](https://github.com/pyrosm/pyrosm/issues/280), [#160](https://github.com/pyrosm/pyrosm/issues/160))
- NEW: Accept `pathlib.Path` (and any `os.PathLike`) filepaths in the `OSM` constructor, not just strings ([#279](https://github.com/pyrosm/pyrosm/issues/279), [#145](https://github.com/pyrosm/pyrosm/issues/145))
- FIXED: Decode node coordinates at full float64 precision (exact OSM 7-decimal values, matching GDAL/osmium); they were truncated to float32, introducing a ~0.1 m error, false extra precision, and visible distortion of straight geometry edges ([#283](https://github.com/pyrosm/pyrosm/issues/283), [#245](https://github.com/pyrosm/pyrosm/issues/245), [#225](https://github.com/pyrosm/pyrosm/issues/225))
- FIXED: Normalize polygon/multipolygon ring orientation to the OGC/GeoJSON right-hand rule (exterior counter-clockwise, holes clockwise), matching osmium and QGIS; previously rings inherited the OSM way node order and were inconsistently wound ([#282](https://github.com/pyrosm/pyrosm/issues/282), [#230](https://github.com/pyrosm/pyrosm/issues/230))
- FIXED: `get_bounding_box` now reads the header bounding box correctly; it returned `None` for every file after the protobuf backend migration ([#280](https://github.com/pyrosm/pyrosm/issues/280), [#160](https://github.com/pyrosm/pyrosm/issues/160))
- FIXED: Download data over HTTPS using certifi's CA bundle instead of the OS trust store, so fetching datasets no longer fails on Windows with `ssl.SSLError [ASN1: NOT_ENOUGH_DATA]` (a CPython bug triggered by a malformed entry in the Windows certificate store) ([#294](https://github.com/pyrosm/pyrosm/issues/294))
- Refresh the README badges ([#278](https://github.com/pyrosm/pyrosm/issues/278))

Thanks for all the contributors who helped to improve the library either via PRs or reporting bugs:

- chrstnbwnkl ([#145](https://github.com/pyrosm/pyrosm/issues/145))
- leonefamily ([#216](https://github.com/pyrosm/pyrosm/issues/216))
- dp12024 ([#225](https://github.com/pyrosm/pyrosm/issues/225))
- 3dfirelab ([#230](https://github.com/pyrosm/pyrosm/issues/230))
- tpwrules ([#245](https://github.com/pyrosm/pyrosm/issues/245))

v0.7.0
------

- NEW: Add `pandarm` graph-export backend (the maintained, NumPy 2-compatible fork of pandana); deprecate `graph_type="pandana"` ([#271](https://github.com/pyrosm/pyrosm/issues/271))
- NEW: Make cycling networks directed and honour `oneway:bicycle` ([#255](https://github.com/pyrosm/pyrosm/issues/255))
- NEW: Add `custom_filter` to `get_network` so custom-filtered networks also return graph nodes ([#264](https://github.com/pyrosm/pyrosm/issues/264))
- NEW: Add `street_count` node attribute to the NetworkX export (compatible with OSMnx `basic_stats`) ([#265](https://github.com/pyrosm/pyrosm/issues/265))
- NEW: Support combining `custom_filter` `True` with explicit tag values ([#251](https://github.com/pyrosm/pyrosm/issues/251))
- Support Python 3.10–3.14 (drop 3.9) and fix OSH parsing under pandas 3.0 ([#248](https://github.com/pyrosm/pyrosm/issues/248))

- Return complete (uncut) geometries for ways/edges that straddle a bounding-box edge ([#268](https://github.com/pyrosm/pyrosm/issues/268))
- Keep bounding-box network `nodes` consistent with the kept `edges` so graph export works without manual cleanup ([#269](https://github.com/pyrosm/pyrosm/issues/269))
- Fix non-dense PBF node parsing (`parse_nodes`) ([#275](https://github.com/pyrosm/pyrosm/issues/275))
- Handle bounding boxes that select no nodes instead of raising `KeyError` ([#267](https://github.com/pyrosm/pyrosm/issues/267))
- Fix `custom_filter` with `highway` turning closed-way polygons into lines ([#266](https://github.com/pyrosm/pyrosm/issues/266))
- Fix network exclude/keep filters leaking on multi-key filters ([#263](https://github.com/pyrosm/pyrosm/issues/263))
- Fix duplicate "phantom" nodes in the NetworkX export ([#259](https://github.com/pyrosm/pyrosm/issues/259))
- Correct relation ids and surface a colliding `id` tag as `id_tag` ([#234](https://github.com/pyrosm/pyrosm/issues/234), [#249](https://github.com/pyrosm/pyrosm/issues/249))
- Stop `get_*` methods from mutating the shared default-tag config ([#252](https://github.com/pyrosm/pyrosm/issues/252))
- Fix spurious pandas chained-assignment warnings from the Cython frame builders ([#256](https://github.com/pyrosm/pyrosm/issues/256))
- Fix Geofabrik UK sub-region downloads (moved under `united-kingdom`) ([#258](https://github.com/pyrosm/pyrosm/issues/258))
- Fix reading PBF produced by `osmconvert` ([#238](https://github.com/pyrosm/pyrosm/issues/238))
- Fix documentation URL ([#223](https://github.com/pyrosm/pyrosm/issues/223))

- Measure Cython (`.pyx`) coverage and raise overall test coverage ([#273](https://github.com/pyrosm/pyrosm/issues/273))
- Document the `pandarm` graph backend and the `pandana` deprecation, and reading OSM history files (`.osh.pbf`) ([#257](https://github.com/pyrosm/pyrosm/issues/257))
- Fix the Read the Docs build; run live download tests on a single CI runner; bump GitHub Actions to Node 24 ([#250](https://github.com/pyrosm/pyrosm/issues/250), [#254](https://github.com/pyrosm/pyrosm/issues/254), [#260](https://github.com/pyrosm/pyrosm/issues/260))

Thanks for all the contributors who helped to improve the library either via PRs or reporting bugs:

- AnBowell ([#233](https://github.com/pyrosm/pyrosm/issues/233), [#234](https://github.com/pyrosm/pyrosm/issues/234))
- eracle ([#238](https://github.com/pyrosm/pyrosm/issues/238))
- meeuw ([#174](https://github.com/pyrosm/pyrosm/issues/174), [#178](https://github.com/pyrosm/pyrosm/issues/178))
- mattijsdp ([#224](https://github.com/pyrosm/pyrosm/issues/224), [#226](https://github.com/pyrosm/pyrosm/issues/226))
- Jontata ([#237](https://github.com/pyrosm/pyrosm/issues/237))
- anatrk ([#112](https://github.com/pyrosm/pyrosm/issues/112))
- Eph97 ([#117](https://github.com/pyrosm/pyrosm/issues/117))
- gregoriiv ([#144](https://github.com/pyrosm/pyrosm/issues/144))
- chourmo ([#170](https://github.com/pyrosm/pyrosm/issues/170))
- arredond ([#176](https://github.com/pyrosm/pyrosm/issues/176))
- lenkahas ([#181](https://github.com/pyrosm/pyrosm/issues/181))
- rohanaras ([#199](https://github.com/pyrosm/pyrosm/issues/199))
- AdrianKriger ([#236](https://github.com/pyrosm/pyrosm/issues/236))
- my4ng ([#239](https://github.com/pyrosm/pyrosm/issues/239))
- skull3r7 ([#241](https://github.com/pyrosm/pyrosm/issues/241))
- wood-chris ([#243](https://github.com/pyrosm/pyrosm/issues/243))
- llebocq ([#247](https://github.com/pyrosm/pyrosm/issues/247))


v0.6.2
------

- Fix installation issues and support only Python >= 3.9 ([#221](https://github.com/pyrosm/pyrosm/issues/221))
- Fix GA actions and use micromamba to install environments ([#221](https://github.com/pyrosm/pyrosm/issues/221))
- Use Shapely 2.0 instead of pygeos ([#214](https://github.com/pyrosm/pyrosm/issues/214))

Thanks for the following contributors:

- knthis ([#214](https://github.com/pyrosm/pyrosm/issues/214))
- hbruch ([#215](https://github.com/pyrosm/pyrosm/issues/215))


v0.6.1
------

- Support Python 3.9 ([#122](https://github.com/pyrosm/pyrosm/issues/122), [#106](https://github.com/pyrosm/pyrosm/issues/106))
- Use Github Actions for CI ([#95](https://github.com/pyrosm/pyrosm/issues/95))
- Drop Travis CI ([#95](https://github.com/pyrosm/pyrosm/issues/95))
- Add contribution guidelines ([#90](https://github.com/pyrosm/pyrosm/issues/90))
- Follow PEP8 style guide and add linting test for CI using Black 

v0.6.0
------

- NEW: Adds possibility to export street networks to igraph, networkx and pandana ([#57](https://github.com/pyrosm/pyrosm/issues/57), [#58](https://github.com/pyrosm/pyrosm/issues/58), [#70](https://github.com/pyrosm/pyrosm/issues/70))
  - Add functionality to parse/return the nodes of the network when requested ([#52](https://github.com/pyrosm/pyrosm/issues/52))
  - Calculate length of the edge for networks in meters ([#56](https://github.com/pyrosm/pyrosm/issues/56), [#70](https://github.com/pyrosm/pyrosm/issues/70))
  - Filter out weakly connected component by default when exporting to graph ([#59](https://github.com/pyrosm/pyrosm/issues/59))
  - Add (vectorized) functionality to create directed edges according `oneway` rules ([#68](https://github.com/pyrosm/pyrosm/issues/68))

- Fix installation issue with pip on Windows ([#61](https://github.com/pyrosm/pyrosm/issues/61))
- Fix numpy deprecation warning ([#50](https://github.com/pyrosm/pyrosm/issues/50))
- Update the documentation to use new theme ([#74](https://github.com/pyrosm/pyrosm/issues/74))
- Add possibility to test the tool using JupyterLab in browser ([#75](https://github.com/pyrosm/pyrosm/issues/75))
- Fix issue when parsing POIs using rare tags as a custom filter ([#47](https://github.com/pyrosm/pyrosm/issues/47))
- Fix issue when filtering with bounding box polygon ([#54](https://github.com/pyrosm/pyrosm/issues/54))
- Add documentation about exporting the networks to graphs ([#69](https://github.com/pyrosm/pyrosm/issues/69))
- Improve documentation overall
 

v0.5.3
------

Changes:

- Ensures that geometry construction works with new Pygeos release v0.8.0 ([#46](https://github.com/pyrosm/pyrosm/issues/46))


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