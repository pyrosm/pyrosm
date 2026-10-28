---
name: pyrosm
description: Use when writing or running Python code that reads OpenStreetMap data with pyrosm, downloads OSM extracts with it (get_data, get_data_by_area, get_data_by_bbox, get_data_by_geocoding, find_extracts, recipes) or geocodes places with it. Says how to plan a download and reuse it, what the data providers ask, and what pyrosm's refusals mean.
---

# pyrosm

pyrosm reads OpenStreetMap PBF files into GeoPandas GeoDataFrames. It downloads the files
from Geofabrik, BBBike and Movisda and geocodes place names with Nominatim; these are free
services shared by everyone. The page
https://pyrosm.readthedocs.io/en/stable/download_responsibly.html explains the rules below,
and https://pyrosm.readthedocs.io/en/stable/llms.txt lists every page of the documentation.

## Plan, then download once

1. Find the extract before downloading. `pyrosm find "minx,miny,maxx,maxy" --contains-only`
   (lon/lat), or `pyrosm.find_extracts(area, contains_only=True)`, lists the extracts that
   contain the area, smallest download first, without downloading any. Before downloading,
   tell the user which files you expect to download and roughly how much data that is.
2. Download into a folder that lasts and reuse the file. The default folder is in the
   system's temporary directory; a file that is there already is not downloaded again.

   ```python
   import pyrosm

   extract = pyrosm.get_data_by_area([24.93, 60.16, 24.96, 60.18], directory="data/osm")
   osm = pyrosm.OSM(extract)
   buildings = osm.get_buildings()
   ```

3. For an analysis that will run again, write a recipe
   (https://pyrosm.readthedocs.io/en/stable/recipes.html) and run it with
   `pyrosm run recipe.yaml`: a rerun reuses the extract and every stage that did not change.
4. Pass `update=True` only when the user wants newer data. pyrosm then sends the server
   the `ETag` or `Last-Modified` value that it saved from the earlier download, so that the
   server sends the file only if it changed. Without a saved value (a file that pyrosm did
   not download from that URL, or a server that gave neither), or for a file changed on
   disk since, the file is downloaded in full.

## Rules

- Download only what the user's task needs. When the task names a few areas that the
  files you have do not already contain, pass them together, as one GeoDataFrame,
  GeoSeries or MultiPolygon, to `get_data_by_area(areas, strategy="smallest_total")`: it
  downloads the smallest extract that contains them all or, when a set of extracts that
  together cover them is smaller in total or that extract's size is unknown, that set,
  merged into one file. For two cities in different countries, the smallest extract that
  contains both can be a whole continent. To tell the user the likely files and sizes beforehand,
  run `pyrosm find AREA --contains-only` for each area; for a place name, call
  `pyrosm.find_extracts(pyrosm.geocode(name, user_agent=...), contains_only=True)`, so that
  the geocoding request names the user's application. Download one file after another,
  never in parallel.
- When the user asks for a whole set of areas, such as every country of a continent, every
  city above a size or a grid of tiles, download the smallest extract that contains them
  all, even if it is larger than separate files, and read each area from it with
  `OSM(filepath, bounding_box=...)`. A long list counts as a whole set even when the user
  names each area. Choose the extract by the region that the task gives, the country or
  continent the areas lie in; if the task gives no region, ask the user for one. Find the
  areas in the extract instead of geocoding them, for example with
  `OSM.get_boundaries(name=...)`, and check what it returns: `name` keeps every boundary
  whose name contains the text. If no extract contains them all, point the user to the
  planet file at https://planet.openstreetmap.org. Do not work through a list of
  countries, cities or tiles that the user did not ask for.
- Do not delete a downloaded file to download it again, and do not call `update=True` in a
  loop.
- Never work around a refusal: do not delete pyrosm's download log, change pyrosm, or fetch
  the file another way (curl, wget, a browser). Stop and show the message to the user.
- Pass a bounding box when you know one instead of geocoding a place name. When you
  geocode with the public Nominatim server, tell the user that it has a usage policy
  (https://operations.osmfoundation.org/policies/nominatim/: at most one request per second,
  a User-Agent that names the application, no systematic queries), and pass
  `user_agent="<application>/<version> (<contact>)"`. Geocoding the few places that a task
  names is fine; do not geocode long lists of places with it, nor from several threads or
  processes at once.
- For a whole country or continent that must stay current, or for the planet, point the
  user to pyosmium-up-to-date and https://planet.openstreetmap.org instead of downloading
  again.
- In tests and continuous integration, use the files that ship with pyrosm
  (`get_data("test_pbf")`, `get_data("helsinki_pbf")`), or cache the download folder
  between runs. Do not write a test or a job that downloads an extract on every run.

## What pyrosm's refusals mean

- `RequestRefusedError: pyrosm has downloaded '<url>' in full 3 times in the last 24 hours`:
  a 4th full download of one URL within 24 hours is refused, also by `get_data_by_area`.
  Reuse the file that is there. The message names the log file; deleting it resets the
  count, which is the user's decision, not yours.
- `RequestRefusedError: pyrosm has sent 50 geocoding requests to the public Nominatim
  server`: the cap per process for pyrosm's own User-Agent. Ask the user for their
  application's name and a contact for `user_agent`, or for their own server's `base_url`,
  or use a bounding box.
- `DownloadError` with `status` 429 or 503: the provider asks clients to slow down or is
  unavailable. Do not retry at once; `get_data_by_area` already left out that provider's
  other extracts.
- `ExtractDownloadError`: no extract for the area could be downloaded; its `errors` hold the
  failure of each extract tried. Report them to the user instead of retrying in a loop.
- A `download_waiting` log line: another download from the same host is running on this
  machine, and this one waits for it.
