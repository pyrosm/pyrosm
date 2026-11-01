# Upgrading

This page lists the changes since pyrosm 0.6.2 (2023) that change what a read, a download, a
written file or a graph export gives, or that need a change in your code, with what to do about
each. Find the release you upgrade from and read the sections of the releases after it, newest
first; the releases not listed here changed nothing of this kind. Changes in speed and memory use,
new options that leave the defaults as they were, and fixes of calls that failed are left out. The
[changelog](changelog.rst) lists every change.

## Unreleased

- **Shared foot and cycle paths in the walking and cycling networks.** A `highway=cycleway` with a
  `foot` tag can now stay in the `walking` network, and a `highway=footway` with a `bicycle` tag
  in the `cycling` network. The tag counts as on any other way: `no` or `private` leaves the way
  out, and other values, such as `designated`, `yes`, `use_sidepath` or `dismount`, let it stay
  unless another rule of the filter leaves it out. A cycle way without a `foot` tag, or a foot way
  without a `bicycle` tag, follows the default of the country it lies in. Pedestrians may use
  cycle ways by default in 23 countries, including Estonia, Finland, Sweden, the United Kingdom
  and the United States, and cyclists may use foot ways in Australia, Belarus, Iceland and
  Norway. Estonia, Lithuania, New Zealand, Portugal and Ukraine are among them because their
  traffic laws let pedestrians walk on cycle paths, in Lithuania, Portugal and Ukraine where
  there is no sidewalk or footpath.
  Elsewhere, and where pyrosm finds no country for the way, such a way is left out. Earlier
  versions followed OSMnx and left every cycle way out of the walking network and every foot way
  out of the cycling network. In the Helsinki region, the walking network grows from 14,662 km to
  17,996 km and the cycling network from 14,288 km to 14,803 km. To get the ways that an earlier
  version would have selected, read a walking network with this version and then drop every row
  whose `highway` tag is `cycleway`. Also drop the rows where `cycleway` is one of several
  `highway` values, as in `cycleway;footway`. For a cycling network, drop the rows whose `highway`
  tag is or contains `footway` in the same way. In a filter built from `get_osm_filter("walking")`
  or `Conf.network_filters.walking`, `cycleway` is no longer in the `highway` list. In the walking
  filter, the `access` entry now leaves out the cycle ways without a `foot` tag, and `get_network`
  and `get_graph` keep those of them that their country allows. This entry is an `AccessRule` with
  `other_mode_highway=["cycleway"]`. The cycling filter's `access` entry does the same for foot
  ways.

## pyrosm 0.28.0

- **Which extract pyrosm downloads.** `find_extracts`, `get_data_by_area`, `get_data_by_bbox`
  and `get_data_by_geocoding` can now choose an extract even when a small part of the area lies
  outside the extract. By default, the part left out must be smaller than 0.1 % of the area. If
  0.1 % of the area is less than 1 m², the 1 m² limit applies instead. Earlier versions allowed
  only parts smaller than 1 m². As a result, pyrosm may choose a smaller extract than before.
  The downloaded data then leaves out the part of the area that falls outside the extract. When
  that part is 1 m² or more, a warning gives its size and bounds. Pass `tolerance=0` to choose
  extracts as earlier versions did. If the extract must contain a particular part of the area,
  pass that part as `must_cover`. In a recipe, set `tolerance: 0` in the `extract` section.
- **Reusing an existing download.** Before downloading an extract, the downloading functions
  look in the download folder for a file that pyrosm 0.27.0 or later downloaded from the
  extract's URL. If the file has not changed since that download, the function reuses it. It
  returns the file's actual path, even when the file has a different name from the one the
  function would normally use. Use the path a function returns instead of building it from the
  extract's name.
- **The `download_extract` log event.** When `get_data_by_area` chooses an extract, it now logs
  a `choose_extract` event. If your code checks pyrosm's log for `download_extract`, change the
  code to check for `choose_extract` instead.

## pyrosm 0.27.0

- **Repeated downloads.** A 4th full download of one URL within 24 hours raises
  `pyrosm.exceptions.RequestRefusedError`, also in `get_data_by_area`, which then stops
  instead of choosing another extract. Keep a downloaded file and reuse it; the message names
  the log file that resets the count when deleted.
- **Geocoding with pyrosm's User-Agent.** After 50 requests to the public Nominatim server
  from one process with pyrosm's own User-Agent, `geocode` and `get_data_by_geocoding` raise
  `RequestRefusedError`, and requests go out at least a second apart. Pass `user_agent` with
  your application's name and a contact, or `base_url` of your own server.
- **The `pyrosm` command in Python.** `pyrosm.recipes.main` moved to `pyrosm.cli.main`; the
  `pyrosm` command itself works as before.
- **pyarrow is installed with pyrosm.** pyrosm requires pyarrow 19.0.1 or later, whose wheels
  need macOS 12 or later. On Linux with glibc older than 2.28, pyarrow has wheels up to Python 3.13;
  Python 3.14 and later need glibc 2.28. On older systems, pip has to build pyarrow from
  source.
- **`pyrosm.recipes.validate` and `run`.** Their first parameter is now called `recipe`,
  as it takes a file or a mapping; a call that names it `path=` needs `recipe=`.

## pyrosm 0.26.0

- **Node metadata of history reads.** With `keep_metadata=False`, an in-memory history read
  leaves out the nodes' `version`, `timestamp` and `changeset`, as other reads do. Keep the
  default `keep_metadata=True` to read them.
- **`tags` of nodes without other tags.** With pandas 3, when none of the nodes a read returns
  has tags outside the tag columns, an in-memory read, or an out-of-core read without `pyarrow`,
  gives their `tags` as NaN in a `str` column, as a cached read does. They were `None` in an
  object column; `isna()` finds both.
- **Timestamps of blocks that do not count seconds.** For a file whose blocks have a
  `date_granularity` other than 1000, ways and relations, and with the out-of-core engine
  every element, have their `timestamp` in seconds, where it was the stored count; an in-memory
  history read leaves out the ways and relations that did not exist yet at its time.

## pyrosm 0.24.0

- **Polygons and lines.** A closed way is a Polygon when one of its tags is in the OSM wiki's list
  of [polygon features](https://wiki.openstreetmap.org/wiki/Overpass_turbo/Polygon_Features), as
  in OSMnx, and a LineString otherwise; `area=no` makes it a LineString. Closed coastlines,
  cliffs, tree rows, streams, railway loops and closed ways without an area tag are LineStrings,
  where they were Polygons; closed hedges, city walls, retaining walls and motorway service areas
  are Polygons, where they were LineStrings. In 0.23.0 a closed way with a `barrier`, `highway` or
  `route` tag was a LineString even when another tag made it an area, such as a fenced sports
  pitch or school; such areas are Polygons again.
- **Columns and dtypes.** Both engines give the same columns with the same dtypes. Every read has
  a bool `visible` column; network, landuse, natural and custom reads had the flag in the JSON
  `tags` column, so read it from the column instead. Out-of-core node rows have int32 `version`,
  uint32 `timestamp` and int32 `changeset`, as in-memory ones do. In-memory reads with a bounding
  box and `complete_relations=True` no longer keep an empty column of a feature outside the box,
  and the keys in the JSON `tags` column of a history read follow the order of the tags in the
  file.
- **Filter keys as columns.** Every reader keeps the keys of its filter as columns, also with
  `tags_to_keep` or `keep_other_tags=False`: `get_buildings(tags_to_keep=["name"])` has a
  `building` column.
- **`custom_filter` of `get_network`.** A plain dict keeps the ways it matches:
  `get_network(custom_filter={"highway": ["footway"]})` returns the footways, where it returned
  every way but the footways. Add `filter_type="exclude"` to leave them out.
- **`get_boundaries(name=...)`** finds the name as plain text, ignoring case, and returns `None`
  with a warning when no name contains it. It was a case-sensitive regular expression; to match a
  pattern, filter the result of `get_boundaries()` with
  `boundaries[boundaries["name"].str.contains(pattern, na=False)]`.
- **`get_data_by_custom_criteria(osm_keys_to_keep=...)`** without a `custom_filter` returns only
  the elements that have one of the keys. It returned every element.
- **NetworkX graphs.** With `osmnx_compatible=True`, the default, the edge attributes `oneway` and
  `reversed` are booleans, as in OSMnx, or on a simplified edge whose ways differ the list of
  their values; the OSM text of the `oneway` tag is in `oneway_tag`. Default graphs keep their
  largest connected part before they are simplified, so they are fully simplified and have fewer
  nodes and edges than before.
- **Graphs with `tags_to_keep`.** `get_graph(tags_to_keep=...)` also reads the tags the graph is
  built from: `oneway`, `junction` and, for cycling, `oneway:bicycle`, and when simplifying the
  `edge_attrs_differ` tags. A driving graph was two-way everywhere, and the simplification merged
  edges across changes of speed limit and road class.
- **Graph exports.** Every graph type leaves out the edges whose end nodes are not in the node
  frame and identifies the nodes by `node_id_col`. The node `x`/`y` of NetworkX, pandarm and
  pandana graphs come from the node geometry, so a projected network gives a graph with projected
  coordinates. With `simplify=False`, the line of an edge travelled against its way's node order
  runs from `u` to `v`. `simplify_kwargs={"node_attrs_include": [...]}` finds node tags as well as
  node columns.
- **Progress functions.** A progress function given to `OSM`, `merge_pbf`, `OSM.write_pbf`,
  `OSM.to_graph`, the graph exporters, `find_extracts` or `pyrosm.enable_logging` is called as
  `progress(done, 100)`, with the percentage of the whole operation done. It got the bytes read in
  each pass over the file of a read, crop or merge, and the steps done of a graph export.
  Downloads still call `progress(written, total)` with bytes.
- **Extracts by box or place.** `get_data_by_bbox` and `get_data_by_geocoding` choose among
  Geofabrik, BBBike and Movisda extracts, so the file can come from another provider than
  Geofabrik. With `download=False` they return a URL, or a list of URLs when the area needs a set
  of extracts. `get_data_by_geocoding` crops to the place's polygon instead of its bounding
  rectangle and names its crops `area_<place>.osm.pbf`. `get_data_by_area(crop=False)` with one
  extract copies the extract to `output_path`; it ignored `output_path`.
- **File names of the two Georgias.** `get_data("usa/georgia")` and `get_data("europe/georgia")`
  save `us-georgia-latest.osm.pbf` and `europe-georgia-latest.osm.pbf`; both were saved as
  `georgia-latest.osm.pbf`, so `get_data("europe/georgia")` could return the US state.
- **`OSM.clear_downloads(path)`** with a path removes exactly the file at that path, and only a
  `.osm.pbf` or `.osh.pbf` file; it removed the file of that name in the download folder. A bare
  file name, such as `"finland-latest.osm.pbf"`, still removes the file of that name in the
  download folder.
- **Bounding boxes.** Every function that takes a bounding box takes the same forms, and a
  projected GeoDataFrame or GeoSeries is reprojected to lon/lat; `get_data_by_bbox` read its
  metres as degrees. Crops, merges and `get_data_by_bbox` refuse coordinates outside lon/lat,
  where crops and merges wrote an empty file. `OSM(bounding_box=polygon).to_pbf()` crops to the
  polygon, as a read with it does, instead of the polygon's bounding rectangle.
- **Written files.** `OSM.write_pbf` merges the rows of an element: a change made in one row of a
  way read with `get_network(nodes=True)`, or in one of two frames that hold the same element, is
  written for the whole element, with a `UserWarning` when other rows of the same frame keep the
  old value, and rows that give a tag different new values raise a `ValueError`; the last row was
  written. It writes its elements sorted by id, writes the metadata that was read and none where
  none was read (new elements get version 1 and the time of writing), no longer writes the network
  columns `u` and `v` as tags while it keeps real tags such as `length`, and writes the `id_tag`
  column back as the tag `id`. After a read with a bounding box it writes only the relations with
  a member among the nodes and ways read; it wrote every relation of the file. With
  `complete_relations=True`, `OSM.to_pbf` and `OSM.write_pbf` also write the member ways outside
  the box of the relations that reach into it, and their nodes.

## pyrosm 0.23.0

- **Closed ways** are Polygons or LineStrings by all their tags, also those that are not columns
  of the read: `get_buildings()` returns a building tagged `area=no` as a LineString. 0.24.0
  replaced this rule with the polygon features described in its section.
- **`visible` values.** A missing `visible` flag reads as `True`, as the PBF format reads it.
  Every node and way of a file that is not a history file had `visible=False`. Relations have a
  `visible` value too.
- **Tags keyed `id`.** A node's or relation's tag keyed `id` comes out as `id_tag`, as a way's
  does, instead of replacing the element's OSM id. Filters read it as `id_tag` in every reader
  (`custom_filter={"id_tag": [...]}`); the filter key `id` raises a `ValueError`.
- **`custom_filter` of the layer readers.** `get_boundaries`, `get_buildings`, `get_landuse` and
  `get_natural` narrow their result by a `custom_filter`:
  `get_boundaries(custom_filter={"admin_level": ["10"]})` returns the boundaries with
  `admin_level=10`, where it returned every boundary and everything else with `admin_level=10`.
- **Driving networks.** `driving` and `driving+service` leave out `highway=busway`, `ladder` and
  `trail` ways, which cars cannot use.
- **Simplified graphs** sum a `travel_time` column along a merged edge, as OSMnx does;
  `simplify_kwargs={"edge_attr_aggs": {...}}` aggregates other columns.
- **pandarm graphs** get the CRS of the nodes; pandarm assumed geographic coordinates.
- **Parallel reads.** Out-of-core reads (the default engine), crops and merges run on several
  threads from about 20 MB of data (`workers="auto"`, the default). Results and written files are
  the same for every worker count; `workers=1` keeps a read on one core. The worker-process
  backend (`PYROSM_DECODE_BACKEND=processes`) is deprecated.

## pyrosm 0.22.0

- **Python versions.** pyrosm supports Python 3.11–3.15. On Python 3.10, `pip` installs the last
  release that supports it.
- **NetworkX attributes.** NetworkX graphs store only the attributes an edge or node has a value
  for, as OSMnx's graphs do: test whether an attribute is present (`"maxspeed" in data`) instead
  of whether it is NaN or None.
- **Node `tags`.** Nodes whose tags all became columns have no `tags` value.
- **Reads at the same time.** Out-of-core reads that run at the same time in several threads of
  one process keep their own settings. A read that started while another one decoded in the
  calling process could change the file, filter keys or bounding box of the decoding read, which
  then returned too few rows without an error or read the wrong file: run such reads again.
- **Multipolygons** have the same geometries, but their rings and vertices can come in another
  order, so a comparison of coordinates or of written bytes can differ.
- **Deprecations.** `get_path()` and the pandana graph backend warn with a `FutureWarning` that
  they will be removed in pyrosm 1.0. Use `get_data()` and `graph_type="pandarm"`
  (`pyrosm.graphs.to_pandarm`).

## pyrosm 0.21.0

- **Written files.** Crops, `merge_pbf` and `write_pbf` compress their blocks with libdeflate. The
  files hold the same data in slightly fewer bytes, so they no longer match files written by
  earlier versions byte for byte, and the SHA-256 of a cropped or merged extract changes: a recipe
  whose `extract.sha256` was pinned with an earlier version needs the new value.

## pyrosm 0.19.1

- **Access rules of the networks.** The predefined networks leave out a way that `access=no` or
  `access=private` closes to the public unless a tag of the network's mode of transport allows the
  use, following OSM's access hierarchy (walking: `foot`, `access`; cycling: `bicycle`, `vehicle`,
  `access`; driving: `motorcar`, `motor_vehicle`, `vehicle`, `access`). A way tagged
  `access=private` and `foot=yes` is now in the walking network.

## pyrosm 0.19.0

- **Out-of-core engine by default.** `OSM(...)` reads with the out-of-core engine, which reads
  country-sized files with bounded memory and keeps each read's result in the temporary folder, so
  an identical read returns it without decoding the file again. `OSM.clear_cache()` removes the
  kept results, and `OSM(..., engine="in_memory")` reads with the earlier reader. Both return the
  same elements and values, but the order of the rows can differ, so sort both results by the
  same columns before you compare them.
- **Column order of the out-of-core engine.** The out-of-core engine gives the columns in the
  order of the in-memory reader: `timestamp` before `version`, `visible` and `nodes` at their
  place among the other columns, and `tags` after `lat` in the node frame of
  `get_network(nodes=True)`. Select columns by name when the order matters.
- **LineStrings.** `get_network()` returns each edge as one LineString through its way's nodes
  instead of a MultiLineString of two-point segments, and the other readers return each line way
  as one LineString too; relations keep their MultiLineStrings. The coordinates and the `length`
  column are the same. Code that went through `geometry.geoms` of a network edge can use the line
  itself.
- **Simplified graphs.** `to_graph` and the graph exporters of `pyrosm.graphs` simplify the graph
  by default; `simplify=False` keeps one edge per segment. A chain is not merged across a change
  of `maxspeed` or `highway`, so each edge has one speed limit and one road class;
  `simplify_kwargs={"edge_attrs_differ": None}` merges as OSMnx does by default. pandana and
  pandarm graphs sum their weight columns along a merged chain, as `length` is summed.
- **Network type of a graph.** `to_graph` takes the network type from the frames it is given; it
  took the type of the network read last, so driving edges exported after a walking read became a
  two-way graph.
- **History reads.** A read of a history file no longer returns elements deleted before the
  `timestamp`, and a `bounding_box` applies to each element's state at the `timestamp`.
- **Untagged nodes** of plain (non-dense) node groups have `None` tags, as dense ones do, instead
  of an empty dict.

## pyrosm 0.18.0

- **History reads** take each element's latest version whole: a tag that the latest version had
  removed no longer comes back from an older version.
- **Column order.** `get_pois`, `get_natural` and `get_data_by_custom_criteria` give their node
  columns in one order on every run; the order changed with Python's hash seed.
- **Written files.** `write_pbf` keeps the tags of the elements a frame matches when the frame has
  no tag columns, such as one that names elements by `id` and `osm_type` only; it wrote them
  without tags.

## pyrosm 0.14.0

- **Overlapping ways in simplified graphs.** Where two OSM ways run along the same nodes, the
  simplified graph keeps one edge, as OSMnx does; it counted the stretch twice in the edge count
  and the total length.
- **Dataset names.** `get_data("madrid")`, `get_data("beijing")` and `get_data("victoria")` each
  name both a BBBike city and a Geofabrik region: use `"cities/madrid"` or `"spain/madrid"`, and
  so on for `"china/beijing"` and `"australia/victoria"`. Spain's Geofabrik regions have ASCII
  names (`castilla_y_leon`, `pais_vasco`). `sources.asia.china`, and the same attribute of the
  other countries that gained regions, is the country's group of regions; the whole-country file
  stays available through `get_data("china")` and the group's `country` record.

## pyrosm 0.13.0

- **Network types as in OSMnx.** The predefined network types take the way filters of OSMnx
  (0.19.1 and 0.23.0 changed them further, as described above). `driving` is OSMnx's `drive` and
  no longer includes service roads: use `driving+service` to keep them. `all` is filtered instead
  of returning every way, and the new `all_public` leaves out privately accessible ways. Every
  mode-specific network leaves out `access=private` ways, and `walking` leaves out ways whose
  sidewalk is mapped separately. Every network leaves out the `highway` values of ways that are
  not built yet or no more (such as `proposed`, `construction` and `abandoned`) and of service
  areas (`rest_area`, `services`); read them with `get_data_by_custom_criteria`, for example
  `custom_filter={"highway": ["construction"]}`. `driving` keeps ways tagged `psv=yes`, and the
  network filters match a tag that holds several values (`access=private;delivery`).
- **NetworkX graphs** work with OSMnx: parallel edges between two nodes are all kept (only the
  last one was), `key` is no longer an edge attribute, `osmid` is the node key and no longer a
  node attribute, and node `tags` are a JSON string, as the edge tags are.
- **Unknown network types** raise a `ValueError` in every reader.

## pyrosm 0.11.0

- **Multipolygons.** Multipolygon and boundary relations are assembled by the even-odd rule, as
  osmium and GDAL assemble them, instead of by the roles of their members. Relations with islands
  in holes, touching inner rings or wrong member roles get different, correct geometries.

## pyrosm 0.9.1

- **Incomplete boundaries.** `get_boundaries()` leaves out a boundary whose member ways run off
  the extract; it closed the ring with a straight edge across the gap.
- **Ambiguous dataset names.** `get_data()` raises a `ValueError` for a name that several regions
  share, such as `"georgia"`: use a region-qualified name, `"usa/georgia"` or `"europe/georgia"`.

## pyrosm 0.8.0

- **Coordinates** are read at full float64 precision, the exact 7-decimal OSM values; they were
  float32, about 0.1 m off.
- **Ring orientation.** Polygon exteriors run counter-clockwise and holes clockwise, by the
  right-hand rule of GeoJSON and OGC, as in osmium and QGIS. Rings followed the node order of
  their ways.
- **Relation members.** Relations carry their members under the `members` key of the JSON `tags`
  column, each as `{member_id, member_type, member_role}`.

## pyrosm 0.7.0

- **Python versions.** Python 3.9 is no longer supported.
- **Relation ids** are the OSM ids of the relations; 0.6.2 decoded them wrongly.
- **Bounding-box reads** return the ways that cross the edge of the box whole instead of cut at
  the edge, so a result can reach a little beyond the box: clip it with GeoPandas
  (`gdf.clip(box)`) when you need its exact bounds. The node frame of `get_network(nodes=True)`
  keeps every end node of the kept edges, also those outside the box.
- **Tags keyed `id`.** A way's tag keyed `id` comes out as `id_tag` instead of replacing the OSM
  id of the way.
- **Cycling networks** are directed and follow `oneway:bicycle`.
- **Default columns.** A `get_*` call no longer adds columns to the default tag lists of
  `pyrosm.config.Conf.tags`: in 0.6.2 a call with `extra_attributes` added them to every later
  read of that layer in the process, so a read's columns depended on the calls before it.
- **Filters.** A `custom_filter` with a `highway` key no longer turns closed-way polygons into
  lines. Network filters check every filter key a way has: a `highway=service` way tagged
  `service=parking_aisle` is left out of the driving network, which kept it.
- **NetworkX graphs** no longer hold duplicate nodes.
