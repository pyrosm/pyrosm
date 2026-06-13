#!/usr/bin/env python
"""Refresh the vendored Geofabrik extent index in ``pyrosm/data``.

Run this to update the bounding-box -> PBF suggestion data used by
``pyrosm.get_data_by_bbox``::

    python scripts/update_geofabrik_index.py

Downloads Geofabrik's ``index-v1.json`` (one GeoJSON FeatureCollection holding
every extract's extent polygon and download URLs), trims each feature to the
fields pyrosm uses (``id``, ``parent``, ``name`` and the ``pbf`` URL) while
keeping the full-resolution geometry, and writes a gzipped snapshot to
``pyrosm/data/geofabrik_index.geojson.gz``. The upstream ``Last-Modified`` date
is stored as a top-level ``geofabrik_snapshot_date`` member so staleness is
auditable.
"""

import gzip
import json
import ssl
import urllib.request
from os.path import abspath, dirname, getsize, join

import certifi

REPO_ROOT = dirname(dirname(abspath(__file__)))
OUT_PATH = join(REPO_ROOT, "pyrosm", "data", "geofabrik_index.geojson.gz")
INDEX_URL = "https://download.geofabrik.de/index-v1.json"


def fetch(url):
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        snapshot_date = response.headers.get("Last-Modified")
        payload = response.read()
    return json.loads(payload), snapshot_date


def trim(index):
    features = []
    for feature in index["features"]:
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
    return features


def main():
    index, snapshot_date = fetch(INDEX_URL)
    features = trim(index)
    collection = {
        "type": "FeatureCollection",
        "geofabrik_snapshot_date": snapshot_date,
        "features": features,
    }
    data = json.dumps(collection, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_PATH, "wb", compresslevel=9) as out_file:
        out_file.write(data)
    print(
        "Wrote %d extents (%.1f MB raw / %.1f MB gz) to %s\nGeofabrik snapshot: %s"
        % (
            len(features),
            len(data) / 1024 / 1024,
            getsize(OUT_PATH) / 1024 / 1024,
            OUT_PATH,
            snapshot_date,
        )
    )


if __name__ == "__main__":
    main()
