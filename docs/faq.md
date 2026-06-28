# Frequently asked questions

Answers to questions that come up often (click a question to expand it). If yours isn't
here, search the [issue tracker](https://github.com/pyrosm/pyrosm/issues) or open a new issue.

## Reading data

:::{dropdown} How do I read a large scale extract without running out of memory?

For reading large file sizes covering e.g. whole countries, use the **out-of-core engine**, which decodes the file in a single streaming pass with bounded peak memory, spilling intermediate data to disk instead of holding the whole file at once:

```python
osm = OSM("country.osm.pbf", engine="out_of_core", workers="auto")
buildings = osm.get_buildings()
```

To decode in parallel pass `workers="auto"` (or `workers=N`). On macOS and Windows a parallel
read must run under an `if __name__ == "__main__":` guard (the worker processes re-import the
entry point); without it the read still completes but falls back to a single process with a
warning:

```python
if __name__ == "__main__":
    osm = OSM("country.osm.pbf", engine="out_of_core", workers="auto")
    buildings = osm.get_buildings()
```

Other ways to lower memory use: read only the area you need with a `bounding_box` (or crop the
file first with `to_pbf()`), keep fewer tag columns with `tags_to_keep=[...]` on the feature
call, and drop the element metadata columns with `keep_metadata=False`.
:::

:::{dropdown} How do I get OSM data for a specific place or bounding box?
Download an extract by place name (geocoded) or by bounding box, then read it — and/or read
only a sub-area of a file with `bounding_box`:

```python
from pyrosm import OSM, get_data, get_data_by_bbox

fp = get_data("Helsinki")                              # an extract by place name
# fp = get_data_by_bbox(bbox=[minx, miny, maxx, maxy]) # an extract for a bounding box

osm = OSM(fp, bounding_box=[minx, miny, maxx, maxy])   # read only this area of the file
```

See [Downloading data](downloading_data.ipynb) for more.
:::

:::{dropdown} How does `bounding_box` keep features that are at the edge of the box ("intersects" or "within")?

It is **intersects**, and the features are kept **whole, not clipped**. A way, building or
other area is included as long as **at least one of its nodes is inside the box**, and it is
returned with its *complete* geometry — so a building that is half inside and half outside the
box comes back as the whole building, with vertices beyond the box edge. (Internally pyrosm
selects "complete ways", like `osmconvert --complete-ways`.)

A **relation** (a multipolygon or boundary) that straddles the box is the exception: by
default it is assembled from only the member ways that fall inside the box, so it has a
*partial* geometry. Pass `complete_relations=True` to fetch each such relation's full member
set so its geometry is complete:

```python
from pyrosm import OSM

osm = OSM("city.osm.pbf", bounding_box=bbox, complete_relations=True)
buildings = osm.get_buildings()
```

:::

:::{dropdown} What coordinate reference system are the results in, and how do I reproject?

pyrosm returns GeoDataFrames in **WGS84 (EPSG:4326)** — longitude/latitude in degrees,
matching the OSM source. Reproject with GeoPandas, e.g. to a metric CRS before measuring
distances or areas:

```python
buildings = osm.get_buildings().to_crs(epsg=3067)   # a metric CRS (here ETRS-TM35FIN)
```

A `CRSError` when importing or first using pyrosm almost always means a broken `pyproj`/PROJ
install in your environment, not a pyrosm problem — reinstall `pyproj` (conda-forge recommended).
:::

## Filtering

:::{dropdown} How does `custom_filter` combine multiple keys and values — AND or OR?

`custom_filter` is **OR**: an element is kept if it matches **any** of the given criteria.
Several values for one key match any of them, and several keys are combined with OR as well.
`True` means "any value for this key".

```python
# kept if it is a building OR carries any amenity tag:
osm.get_data_by_custom_criteria(custom_filter={"building": True, "amenity": True})

# kept if highway is residential OR primary:
osm.get_data_by_custom_criteria(custom_filter={"highway": ["residential", "primary"]})
```

Pass `filter_type="exclude"` to drop the matching elements instead of keeping them. See
[Custom filters](custom_filter.ipynb).
:::

:::{dropdown} `get_boundaries()` returns only administrative areas — how do I get other boundary types?
`get_boundaries()` returns only `boundary=administrative` areas **by default**. For other
types, pass `boundary_type`:

```python
osm.get_boundaries(boundary_type="all")          # every boundary type
osm.get_boundaries(boundary_type="postal_code")  # e.g. postal-code areas
```

Two things related to filtering that are good to understand:

- `name=` filters by the area's **name**, not by its boundary type — e.g.
  `get_boundaries(name="Helsinki")` searches names, it does not select a type.
- A neighbourhood/suburb mapped as an **area** is usually tagged `boundary=place` (e.g.
  `place=suburb`), so it is *not* an administrative boundary. Use `boundary_type="all"`, or
  query the `place` tag directly:

  ```python
  osm.get_data_by_custom_criteria(custom_filter={"place": ["suburb", "neighbourhood"]})
  ```
:::

## Networks and graphs

:::{dropdown} How do I find street intersections and their coordinates?

Build a graph with `to_graph(..., simplify=True)`. Simplification reduces the network to its
topological nodes — every node is a real intersection or a dead-end — and each node carries
`x`/`y` (longitude/latitude in EPSG:4326), a Point `geometry`, and a `street_count` (how many
streets meet there). Keep the nodes where three or more streets meet:

```python
import geopandas as gpd
from pyrosm import OSM, get_data

osm = OSM(get_data("Helsinki"))
nodes, edges = osm.get_network(network_type="driving", nodes=True)
G = osm.to_graph(nodes, edges, graph_type="networkx", simplify=True)

intersections = gpd.GeoDataFrame(
    [
        {"osmid": n, "lon": d["x"], "lat": d["y"],
         "street_count": d["street_count"], "geometry": d["geometry"]}
        for n, d in G.nodes(data=True)
        if d["street_count"] >= 3
    ],
    crs="EPSG:4326",
)
```

`street_count == 1` marks dead-ends. Use `network_type="walking"` or `"cycling"` to count
crossings for those modes instead.
:::

## Installation

:::{dropdown} How should I install pyrosm, and what if installation fails?
The recommended way is from **conda-forge** (with `mamba`/`micromamba` or `conda`):

```
mamba install -c conda-forge pyrosm
```

`pip install pyrosm` also works. On **Windows**, install GeoPandas first, because pip-installing
its native dependencies there is error-prone. See [Installation](installation.ipynb) for details.
:::
