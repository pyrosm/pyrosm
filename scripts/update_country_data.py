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

``access``: reads the country access table of the router Valhalla (MIT licence,
``kCountryAccess`` in ``valhalla/mjolnir/adminconstants.h``, taken from the OSM wiki's
tables) and writes ``pyrosm/data/country_access.json``: for each shared path, the
countries whose default lets the mode use it without the mode's tag (``foot`` on a
``cycleway``, ``bicycle`` on a ``footway``), with the licence as
``pyrosm/data/country_access.LICENSE.md``. Valhalla names the countries; the entries of
parts of the United Kingdom are left out, as the United Kingdom's entry covers them::

    python scripts/update_country_data.py access
    python scripts/update_country_data.py access --source adminconstants.h --license COPYING

With ``--source`` and ``--license``, the files are read from disk instead of downloaded.
The gzip file is written without a timestamp, so the same data gives the same bytes.
"""

import argparse
import gzip
import json
import re
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
VALHALLA_RELEASE = "3.9.1"
VALHALLA_URL = (
    f"https://raw.githubusercontent.com/valhalla/valhalla/{VALHALLA_RELEASE}/"
)
USER_AGENT = "pyrosm country data update (+https://github.com/pyrosm/pyrosm)"

# The columns of Valhalla's kCountryAccess, in order.
VALHALLA_COLUMNS = [
    "trunk",
    "trunk_link",
    "track",
    "footway",
    "pedestrian",
    "bridleway",
    "cycleway",
    "path",
    "motorroad",
]

# The ISO codes of the countries Valhalla names; None leaves the entry out.
VALHALLA_COUNTRIES = {
    "Alba / Scotland": None,
    "Australia": "AU",
    "Austria": "AT",
    "Belarus": "BY",
    "Belgium": "BE",
    "Brazil": "BR",
    "China": "CN",
    "Cymru / Wales": None,
    "Denmark": "DK",
    "England": None,
    "Finland": "FI",
    "France": "FR",
    "Greece": "GR",
    "Hungary": "HU",
    "Iceland": "IS",
    "Ireland": "IE",
    "Italy": "IT",
    "Netherlands": "NL",
    "Northern Ireland": None,
    "Norway": "NO",
    "Oman": "OM",
    "Philippines": "PH",
    "Poland": "PL",
    "Romania": "RO",
    "Russia": "RU",
    "Slovakia": "SK",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Türkiye": "TR",
    "United Kingdom": "GB",
    "United States": "US",
    "ประเทศไทย": "TH",
}

# The shared paths: the mode's tag, the highway value (a Valhalla column) and the
# mode's access flag in Valhalla.
SHARED_PATHS = [
    ("foot", "cycleway", "kPedestrianAccess"),
    ("bicycle", "footway", "kBicycleAccess"),
]


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


def _top_level_items(text):
    """The comma-separated items of ``text``, not splitting inside parentheses."""
    items, depth, current = [], 0, ""
    for char in text:
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += char
    items.append(current.strip())
    return items


def country_access(adminconstants):
    """The vendored access defaults from the text of Valhalla's ``adminconstants.h``: for
    each mode's tag and highway value, the sorted ISO codes of the countries whose default
    lets the mode use such a way. Raises ``ValueError`` for a country it has no ISO code
    for."""
    start = adminconstants.index("kCountryAccess{")
    table = adminconstants[start : adminconstants.index("};", start)]
    allowed = {mode: {highway: set()} for mode, highway, _ in SHARED_PATHS}
    for name, body in re.findall(r'\{"([^"]+)",\s*\{(.*?)\}\}', table, re.S):
        if name not in VALHALLA_COUNTRIES:
            raise ValueError("Valhalla's table names a new country: %s" % name)
        code = VALHALLA_COUNTRIES[name]
        if code is None:
            continue
        values = dict(zip(VALHALLA_COLUMNS, _top_level_items(body)))
        for mode, highway, flag in SHARED_PATHS:
            if re.search(r"\b%s\b" % flag, values[highway]):
                allowed[mode][highway].add(code)
    return {
        "source": "Valhalla %s, valhalla/mjolnir/adminconstants.h, kCountryAccess "
        "(MIT licence)" % VALHALLA_RELEASE,
        "allowed": {
            mode: {highway: sorted(codes) for highway, codes in by_highway.items()}
            for mode, by_highway in allowed.items()
        },
    }


def update_access(source=None, license=None):
    """Write the access defaults and their licence from ``source`` and ``license``, or from
    the pinned release."""
    if source is None:
        source_bytes = _download(VALHALLA_URL + "valhalla/mjolnir/adminconstants.h")
    else:
        source_bytes = Path(source).read_bytes()
    if license is None:
        license_bytes = _download(VALHALLA_URL + "COPYING")
    else:
        license_bytes = Path(license).read_bytes()
    access = country_access(source_bytes.decode("utf-8"))
    path = DATA_DIR / "country_access.json"
    path.write_text(json.dumps(access, indent=1) + "\n", encoding="utf-8")
    (DATA_DIR / "country_access.LICENSE.md").write_bytes(license_bytes)
    print("wrote %s" % path)


def main():
    """Refresh the target given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", choices=["access", "borders"])
    parser.add_argument("--source", help="read the data from this file")
    parser.add_argument("--license", help="read the licence from this file")
    args = parser.parse_args()
    if args.target == "borders":
        update_borders(args.source, args.license)
    else:
        update_access(args.source, args.license)


if __name__ == "__main__":
    main()
