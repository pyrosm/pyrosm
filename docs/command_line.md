# Command line

Installing pyrosm adds a `pyrosm` command. It runs recipes, finds the extracts of an area,
describes PBF files and keeps the download folder in order. It downloads only through
recipes, which reuse an extract that is already there.

```text
pyrosm run RECIPE [-o DIR] [--force]
pyrosm validate RECIPE
pyrosm find AREA [--contains-only] [--json]
pyrosm info FILE [--json]
pyrosm downloads list
pyrosm downloads clear [FILE]
```

`pyrosm run` and `pyrosm validate` are described on the [Recipes](recipes.ipynb) page.

## Find the extracts of an area

`pyrosm find` lists the extracts that overlap an area, best download first, without
downloading any (see {func}`pyrosm.find_extracts`). `AREA` is a bounding box in lon/lat
(`minx,miny,maxx,maxy`), a vector file such as a GeoPackage or GeoJSON file, or a place
name, which Nominatim geocodes. `--contains-only` keeps the extracts that contain the whole
area. `--json` prints a list of records with `provider`, `id`, `name`, `url`, `bytes`
(`null` when the size is unknown), `contains` and `bounds`.

```text
$ pyrosm find 24.93,60.16,24.96,60.18 --contains-only
 provider              name      MB contains                                                              url
   BBBike          Helsinki    65.1      yes https://download.bbbike.org/osm/bbbike/Helsinki/Helsinki.osm.pbf
Geofabrik           Finland   773.8      yes      https://download.geofabrik.de/europe/finland-latest.osm.pbf
```

## Describe a PBF file

`pyrosm info` prints a file's size in bytes, its bounding box, its snapshot time (the
replication timestamp in the file's header, UTC) and its SHA-256, the values a recipe can
pin. `--json` prints them as one record.

## Downloaded files

`pyrosm downloads list` prints the PBF files in pyrosm's default download folder
(`<tempdir>/pyrosm`). `pyrosm downloads clear` removes them, or the one named, with the
partial files of interrupted downloads and their records.

## Exit status

Every command exits with status 0 on success, 1 when the input, the data or a request is
refused (the reason goes to stderr) and 2 when the command is used wrongly. The commands
never prompt, so they can run in scripts and workflow tools.
