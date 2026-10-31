#!/usr/bin/env python
"""Refresh the vendored country data in ``pyrosm/data``.

``borders``: reads the borders of country-coder (ISC licence), the dataset the iD editor
uses to find the country of a point on land, keeps each area's geometry and the ISO 3166-1
alpha-2 code of its country (the sovereign country for a territory or part of a country,
such as ``FI`` for the Åland Islands and Mainland Finland), and writes
``pyrosm/data/country_borders.geojson.gz``, with the release read as the top-level
``source`` member, and the licence as ``pyrosm/data/country_borders.LICENSE.md``::

    python scripts/update_country_data.py borders
    python scripts/update_country_data.py borders --source borders.json --license LICENSE.md

With ``--source`` and ``--license``, the files are read from disk instead of downloaded.
The gzip file is written without a timestamp, so the same data gives the same bytes.
"""

import argparse
import gzip
import json
import ssl
import urllib.request
from pathlib import Path

import certifi

DATA_DIR = Path(__file__).resolve().parents[1] / "pyrosm" / "data"
COUNTRY_CODER_RELEASE = "v5.6.1"
COUNTRY_CODER_URL = (
    "https://raw.githubusercontent.com/rapideditor/country-coder/"
    f"{COUNTRY_CODER_RELEASE}/"
)
USER_AGENT = "pyrosm country data update (+https://github.com/pyrosm/pyrosm)"


def _download(url):
    """The bytes at ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        return response.read()


def country_borders(country_coder):
    """The vendored borders: a FeatureCollection of the areas with a geometry, each with
    the ISO code of its country as the property ``iso``."""
    features = []
    for feature in country_coder["features"]:
        if not feature.get("geometry"):
            continue
        properties = feature["properties"]
        code = properties.get("country") or properties.get("iso1A2")
        if not code:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"iso": code},
                "geometry": feature["geometry"],
            }
        )
    return {
        "type": "FeatureCollection",
        "source": "country-coder %s, src/data/borders.json (ISC licence)"
        % COUNTRY_CODER_RELEASE,
        "features": features,
    }


def write_gzip_json(path, document):
    """Write ``document`` as compact JSON to the gzip file ``path``, without a timestamp."""
    data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            f.write(data)


def update_borders(source=None, license=None):
    """Write the borders and their licence from ``source`` and ``license``, or from the
    pinned release."""
    if source is None:
        source_bytes = _download(COUNTRY_CODER_URL + "src/data/borders.json")
    else:
        source_bytes = Path(source).read_bytes()
    if license is None:
        license_bytes = _download(COUNTRY_CODER_URL + "LICENSE.md")
    else:
        license_bytes = Path(license).read_bytes()
    borders = country_borders(json.loads(source_bytes))
    path = DATA_DIR / "country_borders.geojson.gz"
    write_gzip_json(path, borders)
    (DATA_DIR / "country_borders.LICENSE.md").write_bytes(license_bytes)
    print("wrote %d areas to %s" % (len(borders["features"]), path))


def main():
    """Refresh the target given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", choices=["borders"])
    parser.add_argument("--source", help="read the data from this file")
    parser.add_argument("--license", help="read the licence from this file")
    args = parser.parse_args()
    if args.target == "borders":
        update_borders(args.source, args.license)


if __name__ == "__main__":
    main()
