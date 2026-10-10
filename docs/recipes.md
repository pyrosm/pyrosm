# Recipes

A recipe is a YAML file that describes how the OSM data of an analysis is made: which
extract to download, which layers to read from it and which graph to build from it. `pyrosm
run` carries out a recipe, writes its outputs and records how they were made. The recipe
file is then the method: running it again reproduces the data, and the record shows which
data and which package versions went into it.

Recipes need PyYAML and pyarrow:

```bash
pip install "pyrosm[recipes]"
```

The recipes on this page are in the
[examples/recipes](https://github.com/pyrosm/pyrosm/tree/master/examples/recipes) folder of
the repository.

## A first recipe

This recipe reads the walking network, the buildings and the restaurants and cafes of
central Helsinki from the extract that comes with pyrosm:

```yaml
recipe: pyrosm
extract:
  area: {name: helsinki_pbf}
layers:
  walking_network: {read: network, network_type: walking}
  buildings: {read: buildings}
  food: {read: pois, custom_filter: {amenity: [restaurant, cafe]}}
outputs:
  extract: helsinki.osm.pbf
  layers:
    walking_network: walking_network.parquet
    buildings: buildings.parquet
    food: food.parquet
```

Run it with the output folder `data`:

```bash
pyrosm run helsinki_layers.yaml -o data
```

The command prints the path of each output, after `written` or `reused`:

```text
written /home/me/project/data/helsinki.osm.pbf
written /home/me/project/data/walking_network.parquet
written /home/me/project/data/buildings.parquet
written /home/me/project/data/food.parquet
written /home/me/project/data/helsinki_layers.provenance.json
```

Each layer is a GeoParquet file:

```python
import geopandas as gpd

buildings = gpd.read_parquet("data/buildings.parquet")
```

## Recipe format

A recipe is a mapping with these keys:

| Key | What it holds |
|---|---|
| `recipe` | Always `pyrosm`. Required. |
| `version` | The recipe format version. Only `1` exists, the default. |
| `extract` | The extract stage: the OSM data to download. |
| `pbf` | The PBF file to read when there is no extract stage: `{file: path}`. |
| `reader` | Keywords for {class}`pyrosm.OSM`, such as `bounding_box` or `keep_metadata`. |
| `layers` | The layers stage: the layers to read. |
| `graph` | The graph stage: the graph to build. |
| `outputs` | The file name of each output. Required. |

A recipe has at least one stage. Layers and a graph are read from the extract stage's file or
from the `pbf` file, so they need one of the two, and a recipe cannot have both. The paths of
input files are relative to the recipe's folder.

The other keys of a stage are keywords of the function the stage calls, with their values as
YAML gives them. pyrosm checks each keyword's name against that function before anything
runs, so a misspelt keyword is reported by name. The function checks the values when the
stage runs.

### Extract stage

The `area` key says what to download, as exactly one of:

- `bbox: [minx, miny, maxx, maxy]`, in longitude and latitude;
- `place: "Kallio, Helsinki"`, a place name, turned into an area with {func}`pyrosm.geocode`;
- `file: area.gpkg`, a GeoPackage, GeoJSON or GeoParquet file whose geometries make up the
  area (with `layer: name` for one layer of a GeoPackage);
- `name: helsinki_pbf`, a dataset name of {func}`pyrosm.get_data`.

For a bounding box, a place or an area file, the stage calls {func}`pyrosm.get_data_by_area`,
and the other keys of `extract` are its keywords, such as `strategy`, `crop`, `must_cover`,
`workers` and `directory`. `must_cover` names a file: `must_cover: {file: stops.gpkg}`. For a
dataset name, the stage calls `get_data` and takes its keywords. The recipe sets
`output_path` itself, and `opener` cannot be given. `directory`, the folder that keeps the
downloads, is relative to the recipe's folder and must not lie inside the output folder or
contain it.

This recipe downloads the smallest extract that contains Kallio, or, when it is smaller in
total, a set of extracts that together cover it, merged (`strategy: smallest_total`), and
crops the data to the district:

```yaml
recipe: pyrosm
extract:
  area: {place: "Kallio, Helsinki"}
  strategy: smallest_total
  crop: polygon
layers:
  driving_network: {read: network, network_type: driving}
outputs:
  extract: kallio.osm.pbf
  layers: {driving_network: driving_network.parquet}
```

### Layers stage

`layers` names each layer. A layer's `read` key is one of `network`, `buildings`, `pois`,
`landuse`, `natural`, `boundaries` or `custom`, which calls the {class}`~pyrosm.OSM` method
of that name (`custom` calls `get_data_by_custom_criteria`). Its other keys are that method's
keywords. Each layer is written as GeoParquet; a read that finds nothing is written as an
empty layer.

### Graph stage

`graph.network` holds the keywords of {meth}`pyrosm.OSM.get_network`. The other keys of
`graph` are keywords of {func}`pyrosm.graphs.graph_tables`, such as `simplify` and
`retain_all`, which builds the graph as the graph exporters do. The network type sets the
direction of the edges: a walking graph runs both ways, a driving graph keeps one-way streets
one-way. `force_bidirectional: true` makes every edge run both ways.

```yaml
recipe: pyrosm
extract:
  area: {name: helsinki_pbf}
graph:
  network: {network_type: walking}
  simplify: true
outputs:
  extract: helsinki.osm.pbf
  graph: {nodes: walking_nodes.parquet, edges: walking_edges.parquet}
```

The nodes and the edges are written as two GeoParquet files. A column that holds lists or
dicts, such as the attributes of edges merged by `simplify`, is written as text, with the
lists and dicts as JSON.

### Outputs

`outputs` names every file a stage writes: `extract` for the extract stage's PBF, a file for
each layer under `layers`, and `nodes` and `edges` under `graph`. `provenance` names the
provenance record and defaults to `<recipe name>.provenance.json`. The names are relative to
the output folder, which is the folder given with `-o` or, by default, the recipe's folder.
They end with `.pbf`, `.parquet` and `.json`, and must be names that Windows accepts too.

## Provenance record

The provenance record is a JSON file with:

- `versions`: the versions of pyrosm, geopandas, shapely, pandas, numpy, pyarrow and
  protobuf;
- `recipe`: the recipe with every path resolved (the extract's `headers` are left out, as
  they can hold credentials);
- `inputs` and `outputs`: the SHA-256 of each input file and each output;
- `extract`: where the extract came from, such as its provider, extract name, URL, snapshot
  time and the extracts merged into it;
- `layers` and `graph`: the number of features of each layer, and of nodes and edges;
- `invocation` and `written_at`: how the run was started, and when it finished.

The outputs are made in a staging folder and moved into place only when every stage has
succeeded, so a stage that fails leaves the previous outputs and record as they were. While
a recipe runs, a lock file `.<record name>.lock` in the output folder keeps a second run of
it from starting. If a run was killed, delete the lock file once no run is active.

## Reruns

A second run reuses a stage when its part of the recipe, the `reader` keywords, its output
names, the data it reads and the package versions are all unchanged, and its outputs are in
place with the SHA-256 the record gives them. The command prints `reused` for the outputs of
such a stage. Changing a layer reruns that layer only; changing `reader` reruns the layers
and the graph but keeps the extract.

An extract stage is reused as long as the recipe is unchanged, even when the provider has
published newer data since. To download again, run with `--force` (in Python,
`force=True`), which reruns every stage, or set `update: true` in `extract`, which reruns the
extract stage every time.

## Pins

A recipe can pin the data it expects:

```yaml
extract:
  area: {place: "Kallio, Helsinki"}
  snapshot: 2026-10-01
  sha256: "<the 64 hexadecimal characters of the file's SHA-256>"
```

`sha256` can also be given on the `pbf`, `area` file and `must_cover` inputs. A run whose
input or extract differs from a pin stops, and the previous outputs and record stay as they
were. A `snapshot` date
matches an extract whose data was taken on that day (UTC); a date and time must match the
snapshot time exactly. Put a SHA-256 in quotes, so that YAML reads it as text.

## Command line

```text
pyrosm validate RECIPE
pyrosm run RECIPE [-o DIR] [--force]
```

`pyrosm validate` checks a recipe without running it. Both commands exit with status 0 on
success, 1 when the recipe or its data is refused (the reason goes to stderr) and 2 when the
command is used wrongly. They never prompt, so they can run in scripts and workflow tools.

## Python

```python
from pyrosm import recipes

recipes.validate("helsinki_layers.yaml")
paths = recipes.run("helsinki_layers.yaml", out_dir="data")
```

{func}`pyrosm.recipes.run` returns the paths of the outputs, the provenance record last.

## Snakemake

A recipe fits a Snakemake rule. The rule lists the outputs it needs; `pyrosm run` writes
them, and reruns only the stages that changed:

```python
rule osm_layers:
    input:
        "recipes/helsinki_layers.yaml",
    output:
        "data/walking_network.parquet",
        "data/buildings.parquet",
        "data/helsinki_layers.provenance.json",
    shell:
        "pyrosm run {input} -o data"
```
