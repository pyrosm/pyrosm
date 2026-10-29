# Rules for AI coding agents

If you are an AI agent that writes or runs pyrosm code for a user, follow the rules on this
page. The [Download responsibly](download_responsibly.ipynb) page explains why the rules are
needed and shows examples of how to follow them.

The pyrosm repository contains a more detailed version of these rules in
[skills/pyrosm/SKILL.md](https://github.com/pyrosm/pyrosm/blob/main/skills/pyrosm/SKILL.md).
Coding agents can use the file as a skill. To use it with Claude Code, copy the `pyrosm` folder
into `~/.claude/skills/` or into the `.claude/skills/` folder of your project. Language models
can use <a href="llms.txt">llms.txt</a> as an index to this documentation.

1. Plan before downloading. Use `find_extracts(area, contains_only=True)` to check which extracts
   contain the area. If the area is a bounding box or a vector file, you can make the same check
   on the command line with `pyrosm find AREA --contains-only`. For a place name, use
   `find_extracts(geocode(name, user_agent=...), contains_only=True)`, so that the geocoding
   request identifies the user's application. Tell the user which files you are going to
   download and how large they are.
2. Download each file only once and save it in a folder that you keep. Reuse the same file the
   next time. Do not delete a file in order to download it again, and do not call `update=True`
   in a loop.
3. Download only the data needed for the user's task. Download one file at a time.
4. If the task needs a few areas, first check which extracts contain each area, and then which
   extracts contain all the areas together. Do both checks before you tell the user what will be
   downloaded, because pyrosm may use either one extract that contains all the areas or a
   combination of smaller ones. Then download the areas with one call to
   `get_data_by_area(areas, strategy="smallest_total")`.
5. If the task needs many areas, download one extract that contains all of them and read each
   area from that file. This applies, for example, to all countries of a continent, all cities
   of a country, a grid of tiles, or any long list of places. Choose the extract based on the
   region that the user has given in the task. If the task does not state a region, ask the user
   which region they need. If no extract contains all the areas, point the user to the planet
   file. After you have downloaded the extract, find the areas in it instead of geocoding each of
   them, for example with `OSM.get_boundaries(name=...)`. Notice that `name` matches every
   boundary whose name contains the given text, so check the results before you use them.
6. If pyrosm raises a `RequestRefusedError`, stop and show the error message to the user. Do not
   delete the download log, modify pyrosm, or download the file in some other way (for example
   with curl, wget, or a browser).
7. Geocode as little as possible. When you need to geocode, pass a `user_agent` that identifies
   the user's application. Give the user a link to Nominatim's
   [usage policy](https://operations.osmfoundation.org/policies/nominatim/) and explain what the
   public server allows. Never use the public server to geocode a long list of places, or to
   geocode several places at the same time.
8. If the user needs to keep the data for a country or continent up to date, point them to
   [pyosmium-up-to-date](https://docs.osmcode.org/pyosmium/latest/user_manual/10-Replication-Tools/),
   which updates a downloaded file with only the changes made since the last update. If the user
   needs data for the whole world, point them to the planet file at planet.openstreetmap.org.
9. For tests and CI, store a small PBF file of the user's study area with the code, for example
   in `tests/data/`, and read this file in the tests instead of downloading an extract. Create
   the small file once with `OSM(filepath, bounding_box=...).to_pbf(...)` from an extract that
   has already been downloaded. If the file is too large to store with the code, save it once on
   GitHub, for example as a release asset, and let the pipeline fetch it from there. If a
   pipeline must download an extract, cache the download folder between runs. Never write a test
   or a job that downloads an extract on every run.
