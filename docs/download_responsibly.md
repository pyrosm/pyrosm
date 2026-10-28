# Download responsibly

pyrosm downloads OpenStreetMap extracts from Geofabrik, BBBike and Movisda, and geocodes
place names with Nominatim. These services are free, run by small teams or volunteers, and
shared by everyone who uses them. pyrosm is made for analysing the areas you study: it
downloads a file when you ask for it and reuses it afterwards. It is not a tool for
mirroring the providers or for downloading the same file again and again.

This page says what the providers ask, what pyrosm does to stay within that, and where to
get data when you need more. The last section holds rules for AI coding agents that write
and run pyrosm code.

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

## When you need more data

- Keep downloaded files in a folder that lasts (`directory=`) and reuse them. The default
  folder is in the system's temporary directory, which may be emptied.
- Use a [recipe](recipes.md): a rerun reuses the extract and every stage that did not
  change.
- For a large region that you keep current, download it once and update it with
  pyosmium-up-to-date.
- For the whole planet, use planet.openstreetmap.org.
- For many place names, run your own Nominatim server, or use a provider that offers the
  Nominatim API, and pass its `base_url`.

## Rules for AI coding agents

If you are an AI agent that writes or runs pyrosm code for a user, follow these rules. An
index of the documentation's pages is at
<a href="llms.txt">llms.txt</a>. pyrosm's repository has these rules as a skill for coding
agents, [skills/pyrosm/SKILL.md](https://github.com/pyrosm/pyrosm/blob/main/skills/pyrosm/SKILL.md);
to give it to Claude Code, copy the `pyrosm` folder into `~/.claude/skills/` or a project's
`.claude/skills/`.

1. Plan before you download. Run `pyrosm find AREA` or call `pyrosm.find_extracts` to see
   which extract covers the area and how large it is, without downloading it.
2. Download each extract once, into a folder that lasts, and reuse the file. Do not delete a
   file to download it again, and do not call `update=True` in a loop.
3. Download only what the user's task needs. When it needs several areas, use
   `pyrosm find` to compare the extract that contains them all (their combined bounding box
   with `--contains-only`) with the extracts of each area, and download whichever is less
   data: for two cities in different countries, the one extract that contains both can be
   a whole continent. Before downloading, tell the user how many files and roughly how much
   data that is. Download one file after another, never in parallel, and do not work
   through a list of countries, cities or tiles that the user did not ask for.
4. Never work around a refusal. When pyrosm raises `RequestRefusedError`, stop and show its
   message to the user. Do not delete the download log, change pyrosm, or fetch the file
   another way (curl, wget, a browser).
5. Geocode as little as possible, and pass a bounding box when you know one. When you
   geocode, tell the user that the public Nominatim server has a usage policy, link to it,
   say what it allows, and pass a `user_agent` that names the user's application. Do not
   geocode lists of places with the public server, and do not geocode from several
   processes at once.
6. For a whole country or continent that must stay current, or for the planet, point the
   user to pyosmium-up-to-date and planet.openstreetmap.org instead of downloading again.
