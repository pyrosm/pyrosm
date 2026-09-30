#!/usr/bin/env python
"""Refresh the vendored extract indexes in ``pyrosm/data``.

Run this to update the extract lists used by ``pyrosm.get_data_by_bbox`` and the area extract
lookup::

    python scripts/update_extract_indexes.py            # both
    python scripts/update_extract_indexes.py geofabrik  # or: bbbike

``geofabrik``: downloads Geofabrik's ``index-v1.json`` (one GeoJSON FeatureCollection holding
every extract's extent polygon and download URLs), trims each feature to the fields pyrosm uses
(``id``, ``parent``, ``name`` and the ``pbf`` URL) while keeping the full-resolution geometry, and
writes ``pyrosm/data/geofabrik_index.geojson.gz``. The upstream ``Last-Modified`` date is stored as
a top-level ``geofabrik_snapshot_date`` member so staleness is auditable.

``bbbike``: lists the city extracts on download.bbbike.org, reads each city's ``.poly`` boundary
and writes ``pyrosm/data/bbbike_index.geojson.gz`` with ``id``, ``name`` and the ``pbf`` URL per
city and the download date as ``bbbike_snapshot_date``.
"""

import gzip
import json
import re
import ssl
import sys
import tempfile
import time
import urllib.request
from datetime import date
from pathlib import Path

import certifi

DATA_DIR = Path(__file__).resolve().parents[1] / "pyrosm" / "data"
GEOFABRIK_URL = "https://download.geofabrik.de/index-v1.json"
BBBIKE_URL = "https://download.bbbike.org/osm/bbbike"
USER_AGENT = "pyrosm index update (+https://github.com/pyrosm/pyrosm)"


def fetch(url):
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, context=context) as response:
        return response.read(), response.headers.get("Last-Modified")


def write(name, collection):
    out_path = DATA_DIR / name
    data = json.dumps(collection, separators=(",", ":")).encode("utf-8")
    fd, partial = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=DATA_DIR)
    try:
        with open(fd, "wb") as raw, gzip.open(raw, "wb", compresslevel=9) as out_file:
            out_file.write(data)
        Path(partial).replace(out_path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    print(
        "Wrote %d features (%.1f MB raw / %.2f MB gz) to %s"
        % (
            len(collection["features"]),
            len(data) / 1024 / 1024,
            out_path.stat().st_size / 1024 / 1024,
            out_path,
        )
    )


def geofabrik():
    payload, snapshot_date = fetch(GEOFABRIK_URL)
    features = []
    for feature in json.loads(payload)["features"]:
        props = feature["properties"]
        features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "id": props["id"],
                    "parent": props.get("parent"),
                    "name": props.get("name"),
                    "pbf": props.get("urls", {}).get("pbf"),
                },
            }
        )
    write(
        "geofabrik_index.geojson.gz",
        {
            "type": "FeatureCollection",
            "geofabrik_snapshot_date": snapshot_date,
            "features": features,
        },
    )


def parse_poly(text):
    """Parse an Osmosis ``.poly`` file into GeoJSON MultiPolygon coordinates.

    Sections whose name starts with ``!`` are holes of the preceding outer ring.
    """
    polygons = []
    lines = [line.strip() for line in text.splitlines()][1:]
    i = 0
    while i < len(lines) and lines[i] != "END":
        section = lines[i]
        ring = []
        i += 1
        while lines[i] != "END":
            lon, lat = (float(v) for v in lines[i].split())
            ring.append([lon, lat])
            i += 1
        i += 1
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if section.startswith("!"):
            polygons[-1].append(ring)
        else:
            polygons.append([ring])
    return polygons


def bbbike():
    listing, _ = fetch(BBBIKE_URL + "/")
    cities = sorted(set(re.findall(r'href="([A-Za-z][^"/]*)/"', listing.decode())))
    if len(cities) < 200 or "Berlin" not in cities:
        sys.exit("The BBBike listing lists only %d cities; not writing." % len(cities))
    features = []
    for city in cities:
        time.sleep(0.2)
        poly, _ = fetch("%s/%s/%s.poly" % (BBBIKE_URL, city, city))
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": parse_poly(poly.decode()),
                },
                "properties": {
                    "id": city,
                    "name": city,
                    "pbf": "%s/%s/%s.osm.pbf" % (BBBIKE_URL, city, city),
                },
            }
        )
    write(
        "bbbike_index.geojson.gz",
        {
            "type": "FeatureCollection",
            "bbbike_snapshot_date": date.today().isoformat(),
            "features": features,
        },
    )


if __name__ == "__main__":
    targets = sys.argv[1:] or ["geofabrik", "bbbike"]
    for target in targets:
        {"geofabrik": geofabrik, "bbbike": bbbike}[target]()
