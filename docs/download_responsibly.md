# Download responsibly

pyrosm downloads OpenStreetMap extracts from Geofabrik, BBBike and Movisda, and geocodes
place names with Nominatim. These services are free, run by small teams or volunteers, and
shared by everyone who uses them. pyrosm is made for analysing the areas you study: it
downloads a file when you ask for it and reuses it afterwards. It is not a tool for
mirroring the providers or for downloading the same file again and again.

This page says what the providers ask, how to get the data you need with few downloads,
and what pyrosm does to stay within the providers' limits. The last section holds rules for
AI coding agents that write and run pyrosm code.

## What the providers ask

**Geofabrik** asked its users to download responsibly in a
[blog post of 10 September 2025](https://blog.geofabrik.de/index.php/2025/09/10/download-responsibly/).
Single clients had downloaded the same 20 GB file hundreds of times a day, and one had
downloaded almost 10,000 copies of the Italy extract in 24 hours. When Geofabrik blocks such
a client, other users on the same network are blocked too. Geofabrik asks you to:

- get the whole planet from [planet.openstreetmap.org](https://planet.openstreetmap.org)
  instead of in pieces;
- keep a large region current with
  [pyosmium-up-to-date](https://docs.osmcode.org/pyosmium/latest/user_manual/10-Replication-Tools/),
  which downloads only the changes, about 98 % less data than a fresh download;
- watch what an automated script does, so that it cannot download one file a thousand
  times.

**BBBike**'s [robots.txt](https://download.bbbike.org/robots.txt) disallows `.pbf` files
for all automated clients. pyrosm downloads a BBBike city extract only when you ask for it,
by its name or for an area that it covers best, one file at a time.

**Movisda**: we found no written download policy. Treat its server as Geofabrik asks.

**Nominatim**, the geocoder of the OpenStreetMap Foundation, has a
[usage policy](https://operations.osmfoundation.org/policies/nominatim/): at most one
request per second, a User-Agent that names the application, attribution to OpenStreetMap,
and results cached on your side. Bulk geocoding is discouraged; a small one-time task must
run in a single thread on one machine, and a script that runs longer than a day or on a
schedule may send at most 4 requests per minute. Systematic queries, such as geocoding
every town of a country, are forbidden and get a client banned. A language model may
suggest the service only if it points to the policy and explains its limits, and code it
writes must follow the policy.

## Get the data with few downloads

Start with one extract, download it once and reuse it. The advice below keeps the number of
downloads small as a project grows: more areas, newer data, scripts that run again.

### Look before you download

With `--contains-only`, `pyrosm find` (see [Command line](command_line.md)) lists the
extracts that contain an area, smallest first, with their sizes, without downloading any;
`pyrosm.find_extracts(area, contains_only=True)` does the same in Python:

```text
pyrosm find 24.93,60.16,24.96,60.18 --contains-only
```

[Downloading data](downloading_data.ipynb) shows how pyrosm picks the smallest extract that
contains an area, and how `must_cover` makes the download cover only the places that
matter, such as transit stops, instead of the whole area.

### Keep the files and reuse them

Download into a folder that lasts, such as one inside your project:

```python
import pyrosm

fp = pyrosm.get_data("Helsinki", directory="data/osm")
```

A later call finds the file there and does not download it again. Without `directory=`,
the files go to `<tempdir>/pyrosm` in the system's temporary directory, which the system
may empty. `pyrosm downloads list` shows the files in it, and `pyrosm downloads clear`
removes them.

### Update only when you need newer data

Pass `update=True` when you want newer data. For a file that pyrosm downloaded from that URL
and that has not changed on your disk since pyrosm downloaded it, pyrosm asks the provider
whether the file changed, and downloads it only if it did; any other file is downloaded in
full (see "What pyrosm does" below). Do not delete a file to get a fresh copy: the provider
then sends the whole file again. To keep a large region current, download it once and
update it with pyosmium-up-to-date, which downloads only the changes.

### Download several areas

Pass the areas together, as one GeoDataFrame, GeoSeries or MultiPolygon, to
`get_data_by_area` with `strategy="smallest_total"`:

```python
import geopandas as gpd
from shapely.geometry import box
from pyrosm import OSM, get_data_by_area

# Central Helsinki and central Stockholm
cities = gpd.GeoSeries(
    [box(24.90, 60.14, 25.00, 60.20), box(18.00, 59.30, 18.12, 59.36)], crs=4326
)
fp = get_data_by_area(cities, strategy="smallest_total", directory="data/osm")
helsinki = OSM(fp, bounding_box=cities[0]).get_buildings()
```

The smallest extract that contains both cities is Geofabrik's Europe extract, about 35 GB.
With `strategy="smallest_total"`, pyrosm downloads a set of smaller extracts that together
cover the areas when that is less data, or when the size of the single extract is unknown:
here the BBBike extracts of the two cities, about 120 MB together, merged into one file. If
files that you already have contain the areas, read the areas from those instead.

### Download many areas

For every country of a continent, every city of a country or a grid of tiles, download the
one extract that contains them all, once, and read each area from it:

```python
from pyrosm import OSM, get_data

fp = get_data("Finland", directory="data/osm")
city_boxes = {
    "Helsinki": [24.90, 60.14, 25.00, 60.20],
    "Tampere": [23.70, 61.47, 23.83, 61.52],
}
buildings = {
    city: OSM(fp, bounding_box=bbox).get_buildings() for city, bbox in city_boxes.items()
}
```

`OSM(fp, bounding_box=bbox).to_pbf("city.osm.pbf")` writes an area that you read often to
a smaller file of its own. For the whole world, download the planet file from
planet.openstreetmap.org. Do not download the areas one by one in a loop: a script that
works through BBBike's cities is what its robots.txt disallows, and pyrosm's check on
repeated downloads, which counts the downloads of each URL on its own, does not stop a loop
over many URLs.

### Use pyrosm in tests, continuous integration and clusters

- `pyrosm.get_data("test_pbf")` and `pyrosm.get_data("helsinki_pbf")` return small files
  that ship with pyrosm, so tests that use them download nothing.
- Without a cache, a continuous integration job that runs on a fresh machine, such as a
  hosted runner, downloads its extract again on every run, and pyrosm's checks do not see
  the earlier runs. Cache the download folder between runs, for example with GitHub Actions'
  `actions/cache`.
- For jobs on a cluster, download the file once, to storage that all the jobs can read,
  before you start them, and pass its path to the jobs. pyrosm's checks see only the
  downloads of one machine.
- For an analysis that you run again, write a [recipe](recipes.md): a rerun reuses the
  extract and every stage that did not change.

### Geocode sparingly

- When you know an area's bounding box, pass it instead of a place name: each place name
  that you geocode sends a request to Nominatim, unless pyrosm has the answer from the last
  7 days.
- Pass a `user_agent` that names your application, for example
  `user_agent="myproject/1.0 (me@example.org)"`.
- Geocoding a few place names is fine. For many, run your own Nominatim server, or use a
  provider that offers the Nominatim API, and pass its `base_url`.

## What pyrosm does

- A file that is in the download folder already is not downloaded again. With
  `update=True`, pyrosm sends the `ETag` or `Last-Modified` date of the earlier download
  with its request, so that the server sends the file only if it changed; a "not modified"
  answer keeps the file. A file that pyrosm did not download from that URL, one changed
  since, or one whose server gave neither is downloaded in full.
- pyrosm records each full download in a log per URL in `<tempdir>/pyrosm/download-log/`.
  After 3 full downloads of one URL within 24 hours, a 4th raises
  `pyrosm.exceptions.RequestRefusedError`, whose message names the log file; deleting that
  file resets the count. A check that finds the file unchanged does not count.
- Downloads from one host go one at a time on a machine, also across processes: a second
  download waits for the first and logs that it waits.
- When a download from a provider still fails with 429 (too many requests) or 503
  (unavailable) after its retries, `get_data_by_area` leaves out that provider's other
  extracts not downloaded yet for the rest of the call, instead of trying its larger ones.
  A refused download ends the call.
- A request that fails with a network error or HTTP status 408, 425, 429 or 5xx is tried at
  most three times, waiting the time the server asks for, or else 1 s and 2 s. When the
  server asks for more than 60 s, pyrosm gives up at once.
- Geocoding answers are cached for 7 days in `<tempdir>/pyrosm/geocoding/`. Requests to the
  public Nominatim server go out at most once a second from one process; several processes
  together can send more. After 50 requests from one process with pyrosm's own User-Agent,
  which names no application of yours, pyrosm raises `RequestRefusedError`; pass your own
  `user_agent`, or the `base_url` of your own server.
- Unless you pass your own User-Agent, every request names pyrosm and its version in the
  User-Agent. When an AI coding agent runs pyrosm (the environment variable `AI_AGENT` or
  `CLAUDECODE` is set), it also says `agent`, so that the providers can tell those requests
  apart.

These checks run on your machine, so anyone can get around them. They catch mistakes, such
as a script stuck in a loop, and make a repeated download a deliberate act.

## Rules for AI coding agents

If you are an AI agent that writes or runs pyrosm code for a user, follow the advice above
and these rules. An index of the documentation's pages is at
<a href="llms.txt">llms.txt</a>. pyrosm's repository has these rules as a skill for coding
agents, [skills/pyrosm/SKILL.md](https://github.com/pyrosm/pyrosm/blob/main/skills/pyrosm/SKILL.md);
to give it to Claude Code, copy the `pyrosm` folder into `~/.claude/skills/` or a project's
`.claude/skills/`.

1. Plan before you download. Run `pyrosm find AREA --contains-only` or call
   `pyrosm.find_extracts(area, contains_only=True)` to see which extracts contain the area
   and how large they are, without downloading any. Before downloading, tell the user which
   files you expect to download and roughly how much data that is.
2. Download each extract once, into a folder that lasts, and reuse the file. Do not delete a
   file to download it again, and do not call `update=True` in a loop.
3. Download only what the user's task needs. When the task names a few areas that the
   files you have do not already contain, pass them together to
   `get_data_by_area(areas, strategy="smallest_total")`, as in "Download several areas"
   above: it downloads the smallest extract that contains them all or, when that is less
   data or that extract's size is unknown, a set of smaller extracts that together cover
   them, merged into one file. To tell
   the user the likely files and sizes beforehand, run `pyrosm find AREA --contains-only`
   for each area; for a place name, call
   `pyrosm.find_extracts(pyrosm.geocode(name, user_agent=...), contains_only=True)`, so
   that the geocoding request names the user's application (rule 6). Download one file
   after another, never in parallel.
4. When the user asks for a whole set of areas, such as every country of a continent,
   every city above a size or a grid of tiles, download the smallest extract that contains
   them all, even if it is larger than separate files, and read each area from it with
   `OSM(filepath, bounding_box=...)`. A long list counts as a whole set even when the user
   names each area. Choose the extract by the region that the task gives, the country or
   continent the areas lie in; if the task gives no region, ask the user for one. Find the
   areas in the extract instead of geocoding them, for example with
   `OSM.get_boundaries(name=...)`, and check what it returns: `name` keeps every boundary
   whose name contains the text. If no extract contains them all, point the user to the
   planet (rule 7). Do not work through a list of countries, cities or tiles that the user
   did not ask for.
5. Never work around a refusal. When pyrosm raises `RequestRefusedError`, stop and show its
   message to the user. Do not delete the download log, change pyrosm, or fetch the file
   another way (curl, wget, a browser).
6. Geocode as little as possible, and pass a bounding box when you know one. When you
   geocode, tell the user that the public Nominatim server has a
   [usage policy](https://operations.osmfoundation.org/policies/nominatim/), link to it,
   say what it allows, and pass a `user_agent` that names the user's application.
   Geocoding the few places that a task names is fine; do not geocode long lists of places
   with the public server, and do not geocode from several threads or processes at once.
7. For a whole country or continent that must stay current, or for the planet, point the
   user to pyosmium-up-to-date and planet.openstreetmap.org instead of downloading again.
8. In tests and continuous integration, use the files that ship with pyrosm
   (`get_data("test_pbf")`, `get_data("helsinki_pbf")`), or cache the download folder
   between runs. Do not write a test or a job that downloads an extract on every run.
