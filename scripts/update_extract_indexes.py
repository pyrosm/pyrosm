#!/usr/bin/env python
"""Refresh the vendored extract indexes in ``pyrosm/data``.

Run this to update the extract lists used by ``pyrosm.get_data_by_bbox`` and the area extract
lookup::

    python scripts/update_extract_indexes.py            # both
    python scripts/update_extract_indexes.py geofabrik  # or: bbbike, geofabrik-sizes

``geofabrik-sizes``: adds each extract's download size (``bytes``, with the day it was read as
``bytes_date``) to the existing ``pyrosm/data/geofabrik_index.geojson.gz`` without changing
anything else. A size comes from a HEAD request, else from the ``File size`` on the extract's
page, else the previous snapshot's size is kept. The ``geofabrik`` target reads sizes the same
way. pyrosm uses these sizes when the server does not answer a HEAD request.

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
import http.client
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

from pyrosm.utils.download import _content_length

DATA_DIR = Path(__file__).resolve().parents[1] / "pyrosm" / "data"
GEOFABRIK_URL = "https://download.geofabrik.de/index-v1.json"
BBBIKE_URL = "https://download.bbbike.org/osm/bbbike"
USER_AGENT = "pyrosm index update (+https://github.com/pyrosm/pyrosm)"


PAGE_SIZE = re.compile(
    r"File size:(?:\s|&nbsp;)*([0-9]{1,6}(?:\.[0-9]{1,3})?)(?:\s|&nbsp;)*([KMG]B)"
)
# The pages give binary units: "333 MB" is Belarus at 349,606,795 bytes (333.4 MiB).
PAGE_UNITS = {"KB": 2**10, "MB": 2**20, "GB": 2**30}


GEOFABRIK_DOWNLOADS = "https://download.geofabrik.de/"
LATEST_PBF = "-latest.osm.pbf"


def fetch(url, method="GET", limit=None):
    """GET (or HEAD) ``url``: ``(body, Last-Modified)``, or the headers for HEAD. ``limit``
    caps the bytes read."""
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}, method=method
    )
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        if method == "HEAD":
            return response.headers
        body = response.read() if limit is None else response.read(limit)
        return body, response.headers.get("Last-Modified")


def page_size(html):
    """The size an extract page states for its ``.osm.pbf`` ("File size: 521 MB"), or None."""
    match = PAGE_SIZE.search(html)
    if match is None:
        return None
    size = round(float(match[1]) * PAGE_UNITS[match[2]])
    return size if size < 2**63 else None


def read_size(pbf_url):
    """``(bytes, source)`` of a Geofabrik extract: its HEAD ``Content-Length``, else the size
    on its page; ``(None, None)`` when neither can be read. Only Geofabrik's own
    ``https://download.geofabrik.de/...-latest.osm.pbf`` URLs are asked."""
    if not (pbf_url.startswith(GEOFABRIK_DOWNLOADS) and pbf_url.endswith(LATEST_PBF)):
        return None, None
    try:
        size = _content_length(fetch(pbf_url, "HEAD").get("Content-Length"))
        if size is not None:
            return size, "head"
    except (OSError, http.client.HTTPException):
        pass
    try:
        page_url = pbf_url[: -len(LATEST_PBF)] + ".html"
        html, _ = fetch(page_url, limit=2**20)
        size = page_size(html.decode("utf-8", "replace"))
        if size is not None:
            return size, "page"
    except (OSError, http.client.HTTPException):
        pass
    return None, None


def add_sizes(features, previous):
    """Set ``bytes`` and ``bytes_date`` on every feature with a ``pbf`` URL; a size that
    cannot be read keeps the ``(bytes, bytes_date)`` in ``previous``, else both are None.
    """
    today = date.today().isoformat()
    counts = {"head": 0, "page": 0, "kept": 0, "none": 0}
    for feature in features:
        props = feature["properties"]
        url = props.get("pbf")
        if not url:
            continue
        size, source = read_size(url)
        time.sleep(0.2)
        props["bytes"], props["bytes_date"] = None, None
        if size is not None:
            props["bytes"], props["bytes_date"] = size, today
        elif url in previous:
            props["bytes"], props["bytes_date"] = previous[url]
            source = "kept"
        counts[source or "none"] += 1
    print("Geofabrik sizes: %s" % ", ".join("%s %d" % item for item in counts.items()))
    return today


def previous_sizes():
    """``{pbf URL: (bytes, bytes_date)}`` from the vendored Geofabrik index, if it has any."""
    path = DATA_DIR / "geofabrik_index.geojson.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        features = json.load(f)["features"]
    return {
        p["pbf"]: (p["bytes"], p.get("bytes_date"))
        for p in (feature["properties"] for feature in features)
        if p.get("pbf") and isinstance(p.get("bytes"), int)
    }


def geofabrik_sizes():
    path = DATA_DIR / "geofabrik_index.geojson.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        collection = json.load(f)
    collection["geofabrik_sizes_date"] = add_sizes(
        collection["features"], previous_sizes()
    )
    write("geofabrik_index.geojson.gz", collection)


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
    sizes_date = add_sizes(features, previous_sizes())
    write(
        "geofabrik_index.geojson.gz",
        {
            "type": "FeatureCollection",
            "geofabrik_snapshot_date": snapshot_date,
            "geofabrik_sizes_date": sizes_date,
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
        commands = {
            "geofabrik": geofabrik,
            "geofabrik-sizes": geofabrik_sizes,
            "bbbike": bbbike,
        }
        commands[target]()
