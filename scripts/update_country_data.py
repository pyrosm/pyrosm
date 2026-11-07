#!/usr/bin/env python
"""Refresh the vendored country data in ``pyrosm/data``.

``borders``: reads the national boundaries of OpenStreetMap through the Overpass API (the
``admin_level=2`` boundary relations, which take in the territorial sea) and writes
``pyrosm/data/country_borders.geojson.gz``: one area per ISO 3166-1 alpha-2 code, simplified
to within ``TOLERANCE``, with the data's date in the top-level ``source`` member. A
territory inside a country counts as that country (the Åland Islands as ``FI``), except a
territory that the wiki page below gives its own table (Hong Kong), which gets its own
area. Where the boundaries of two countries overlap, as in some disputed regions, the
overlap belongs to both::

    python scripts/update_country_data.py borders
    python scripts/update_country_data.py borders --source boundaries.json

``access``: reads the country tables of the OSM wiki page "OSM tags for routing/Access
restrictions" and writes ``pyrosm/data/country_access.json``: for each shared path, the
countries whose default lets the mode use it without the mode's tag (``foot`` on a
``cycleway``, ``bicycle`` on a ``footway``), with the page's revision in the ``source``
member. A cell allows the mode when it is ``yes`` or ``partial`` (allowed under a
condition, such as where no sidewalk exists), except a ``partial`` that says ``dismount``;
``no`` does not. A country whose table has no row or value for the way follows OSM's
worldwide defaults. The countries of ``TRAFFIC_LAW`` are added, with their provisions in
the ``law`` member::

    python scripts/update_country_data.py access
    python scripts/update_country_data.py access --source access_restrictions.json

With ``--source``, the data is read from a file instead of downloaded: for the borders, an
Overpass API answer in JSON with the boundary relations and their geometry, such as the
answer to ``BOUNDARIES_QUERY`` with ``out geom`` in place of ``out tags``; for the access
defaults, the wiki API's answer that the download reads (``WIKI_API`` with
``WIKI_QUERY``). The gzip file is written without a timestamp, so the same data gives the
same bytes.
"""

import argparse
import gzip
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import shapely
from shapely.geometry import mapping

DATA_DIR = Path(__file__).resolve().parents[1] / "pyrosm" / "data"
OVERPASS_API = "https://overpass-api.de/api/interpreter"
# The boundaries that carry a country code: countries at admin_level 2, territories such as
# Hong Kong at lower levels.
BOUNDARIES_QUERY = (
    "[out:json][timeout:600];"
    'relation["boundary"="administrative"]["ISO3166-1:alpha2"];out tags;'
)
GEOMETRY_QUERY = "[out:json][timeout:900];relation(id:%s);out geom;"
GEOMETRY_BATCH = 10
# The borders are simplified to within this distance, in degrees (about 1 km), and their
# coordinates rounded to this precision.
TOLERANCE = 0.01
PRECISION = 1e-5
WIKI_API = "https://wiki.openstreetmap.org/w/api.php"
WIKI_PAGE = "OSM tags for routing/Access restrictions"
# The query for the page's latest revision: its wikitext, id and date.
WIKI_QUERY = urllib.parse.urlencode(
    {
        "action": "query",
        "prop": "revisions",
        "titles": WIKI_PAGE,
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
    }
)
USER_AGENT = "pyrosm country data update (+https://github.com/pyrosm/pyrosm)"

# The ISO codes of the countries that the wiki page has a section for.
WIKI_COUNTRIES = {
    "Australia": "AU",
    "Austria": "AT",
    "Belarus": "BY",
    "Belgium": "BE",
    "Brazil": "BR",
    "China": "CN",
    "Croatia": "HR",
    "Denmark": "DK",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Hong Kong": "HK",
    "Hungary": "HU",
    "Iceland": "IS",
    "Ireland": "IE",
    "Italy": "IT",
    "Netherlands": "NL",
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
    "Thailand": "TH",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "GB",
    "United States of America": "US",
}

# The shared paths: the mode's tag and the highway value.
SHARED_PATHS = [("foot", "cycleway"), ("bicycle", "footway")]


# Countries added to the wiki's tables from their traffic law: for each shared path, the
# countries whose law lets the mode use such a way, also those that allow it only where
# no way of the mode runs beside it, with the provision.
TRAFFIC_LAW = {
    ("foot", "cycleway"): {
        "EE": "Liiklusseadus (Traffic Act), § 22(2)",
        "LT": "Kelių eismo taisyklės (Road Traffic Rules), point 48.5",
        "NZ": "Land Transport (Road User) Rule 2004, clause 1.6, cycle path",
        "PT": "Código da Estrada, articles 78(4) and 99(2)(b) and (3)",
        "UA": "Правила дорожнього руху (Traffic Rules, resolution 1306/2001), point 4.1",
    },
}


def _download(url):
    """The bytes at ``url``, sent gzip-compressed where the server can."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    request = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context, timeout=900) as response:
        data = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def _complete_answer(answer):
    """The Overpass API's ``answer``, checked: ValueError when it reports that its query
    stopped early, such as on a timeout, so that its elements may be incomplete."""
    if "remark" in answer:
        raise ValueError("The Overpass API stopped the query: %s" % answer["remark"])
    return answer


def _overpass(query):
    """The Overpass API's answer to ``query``, checked by :func:`_complete_answer`."""
    url = OVERPASS_API + "?" + urllib.parse.urlencode({"data": query})
    return _complete_answer(json.loads(_download(url)))


def border_relations(boundaries):
    """The ids of the boundary relations that the borders are built from, by ISO code: each
    country's ``admin_level=2`` boundary and, for a territory that the wiki gives its own
    table (``WIKI_COUNTRIES``), its boundary of the lowest admin level."""
    countries, territories = {}, {}
    for element in boundaries:
        tags = element.get("tags", {})
        code, level = tags.get("ISO3166-1:alpha2"), tags.get("admin_level", "")
        if code is None or tags.get("boundary") != "administrative":
            continue
        if level == "2":
            if code in countries:
                raise ValueError("Two country boundaries have the code %s" % code)
            countries[code] = element["id"]
        elif code in WIKI_COUNTRIES.values() and level.isdigit():
            territories.setdefault(code, []).append((int(level), element["id"]))
    for code, candidates in territories.items():
        countries.setdefault(code, min(candidates)[1])
    missing = sorted(set(WIKI_COUNTRIES.values()) - set(countries))
    if missing:
        raise ValueError("No boundary has the code %s of the wiki" % ", ".join(missing))
    return countries


def relation_area(relation):
    """The area of a boundary relation: the rings that its member ways form, a ring inside
    another a hole. ValueError when a way is not part of a ring or the relation has no
    area."""
    lines = [
        shapely.linestrings(
            [(point["lon"], point["lat"]) for point in member["geometry"]]
        )
        for member in relation["members"]
        if member["type"] == "way"
    ]
    if any(not part.is_empty for part in shapely.polygonize_full(lines)[1:]):
        raise ValueError("The boundary relation %d is not closed" % relation["id"])
    area = shapely.build_area(shapely.geometrycollections(lines))
    if area.is_empty:
        raise ValueError("The boundary relation %d has no area" % relation["id"])
    return area


def _coverage_pieces(areas, codes):
    """The areas cut into pieces that do not overlap, and the indices of the areas that
    each piece belongs to: where two areas overlap or one lies within the other, such as in
    a disputed region, their common part is a piece of both. ValueError when two areas are
    the same or three overlap in one place."""
    pieces, owners = list(areas), [[i] for i in range(len(areas))]
    tree = shapely.STRtree(areas)
    first, second = tree.query(areas, predicate="contains")
    contains = {(i, j) for i, j in zip(first, second) if i != j}
    same = sorted({codes[i] for i, j in contains if (j, i) in contains})
    if same:
        raise ValueError("The boundaries of %s are the same" % ", ".join(same))
    first, second = tree.query(areas, predicate="overlaps")
    pairs = {(min(i, j), max(i, j)) for i, j in [*zip(first, second), *contains]}
    for i, j in sorted(pairs):
        parts = shapely.get_parts(shapely.intersection(areas[i], areas[j]))
        overlap = shapely.multipolygons(parts[shapely.get_type_id(parts) == 3])
        if overlap.is_empty:
            continue
        pieces[i] = shapely.difference(pieces[i], overlap)
        pieces[j] = shapely.difference(pieces[j], overlap)
        pieces.append(overlap)
        owners.append([i, j])
    pieces = np.array(pieces)
    invalid = ~shapely.is_empty(shapely.coverage_invalid_edges(pieces))
    if invalid.any():
        names = sorted({codes[i] for k in np.flatnonzero(invalid) for i in owners[k]})
        raise ValueError(
            "The boundaries of %s overlap where they cannot be split" % ", ".join(names)
        )
    return pieces, owners


def country_borders(document):
    """The vendored borders from an Overpass answer with the boundary relations and their
    geometry: a FeatureCollection with one area per ISO code (the property ``iso``),
    sorted by code. A territory's area is cut out of its country's; where two countries'
    boundaries overlap, the overlap stays in both. The areas are simplified together, so
    that neighbours keep a common border."""
    relations = {e["id"]: e for e in document["elements"] if e["type"] == "relation"}
    ids = border_relations(relations.values())
    codes = sorted(ids)
    areas = np.array([relation_area(relations[ids[code]]) for code in codes])
    territory = np.array(
        [relations[ids[code]]["tags"]["admin_level"] != "2" for code in codes]
    )
    if territory.any():
        cut = shapely.union_all(areas[territory])
        areas[~territory] = shapely.difference(areas[~territory], cut)
    pieces, owners = _coverage_pieces(areas, codes)
    pieces = shapely.coverage_simplify(pieces, TOLERANCE)
    areas = [
        shapely.union_all([pieces[k] for k, owner in enumerate(owners) if i in owner])
        for i in range(len(codes))
    ]
    areas = shapely.set_precision(areas, PRECISION, mode="pointwise")
    features = [
        {"type": "Feature", "properties": {"iso": code}, "geometry": mapping(area)}
        for code, area in zip(codes, areas)
    ]
    return {
        "type": "FeatureCollection",
        "source": "OpenStreetMap contributors (ODbL), national boundaries of %s"
        % document["osm3s"]["timestamp_osm_base"][:10],
        "features": features,
    }


def write_gzip_json(path, document):
    """Write ``document`` as compact JSON to the gzip file ``path``, without a timestamp."""
    data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            f.write(data)


def update_borders(source=None):
    """Write the borders from the Overpass answer saved in ``source``, or from the
    boundaries' current geometry."""
    if source is None:
        listing = _overpass(BOUNDARIES_QUERY)
        selection = border_relations(listing["elements"])
        ids = sorted(selection.values())
        document = {"osm3s": listing["osm3s"], "elements": []}
        for start in range(0, len(ids), GEOMETRY_BATCH):
            batch = ids[start : start + GEOMETRY_BATCH]
            answer = _overpass(GEOMETRY_QUERY % ",".join(map(str, batch)))
            found = sorted(
                e["id"] for e in answer["elements"] if e["type"] == "relation"
            )
            if found != batch:
                raise ValueError(
                    "The Overpass API returned the relations %s for %s"
                    % (", ".join(map(str, found)), ", ".join(map(str, batch)))
                )
            document["elements"] += answer["elements"]
            print("read %d of %d boundaries" % (start + len(batch), len(ids)))
        if border_relations(document["elements"]) != selection:
            raise ValueError("The boundaries changed during the download; run it again")
    else:
        document = _complete_answer(json.loads(Path(source).read_bytes()))
    borders = country_borders(document)
    path = DATA_DIR / "country_borders.geojson.gz"
    write_gzip_json(path, borders)
    print("wrote %d areas to %s" % (len(borders["features"]), path))


def _without_template(text, name):
    """``text`` without the templates ``{{name|...}}``, nested templates included; the
    first letter of the name may be either case, as in MediaWiki."""
    first = name[0]
    pattern = re.compile(
        r"\{\{[%s%s]%s" % (first.upper(), first.lower(), re.escape(name[1:]))
    )
    kept, start = [], 0
    while found := pattern.search(text, start):
        begin = found.start()
        kept.append(text[start:begin])
        depth, end = 0, begin
        while end < len(text):
            if text.startswith("{{", end):
                depth, end = depth + 1, end + 2
            elif text.startswith("}}", end):
                depth, end = depth - 1, end + 2
                if depth == 0:
                    break
            else:
                end += 1
        start = end
    return "".join(kept) + text[start:]


def _without_notes(wikitext):
    """``wikitext`` without comments and footnotes, which can hold table markup."""
    wikitext = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    wikitext = re.sub(r"<ref[^>]*/>", "", wikitext)
    wikitext = re.sub(r"<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S)
    return _without_template(wikitext, "refn")


def _top_level_split(text, separator):
    """``text`` split at each ``separator`` outside templates and links."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        if text.startswith(("{{", "[["), i):
            depth, i = depth + 1, i + 2
        elif text.startswith(("}}", "]]"), i):
            depth, i = depth - 1, i + 2
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i = start = i + len(separator)
        else:
            i += 1
    parts.append(text[start:])
    return parts


def _cell_parts(cell):
    """A table cell's attributes, such as ``style="..."`` or ``rowspan="5"``, and its
    content."""
    parts = _top_level_split(cell, "|")
    attributes = r"\s*([\w-]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s\"']+)\s*)+"
    if len(parts) > 1 and re.fullmatch(attributes, parts[0]):
        return parts[0].strip(), "|".join(parts[1:]).strip()
    return "", cell.strip()


def _column_names(cell):
    """The keys a header cell names, such as ``{"foot", "ski"}`` for
    ``foot<br/>ski``."""
    cell = _cell_parts(cell)[1]
    cell = re.sub(r"\[\[[Kk]ey:[^|\]]*\|([^\]]*)\]\]", r"\1", cell)
    cell = re.sub(r"\{\{[Tt]ag(?:Key)?\|([^}|]*)\}\}", r"\1", cell)
    return {
        re.sub(r"[^a-z_]", "", part.lower().split("(")[0])
        for part in re.split(r"<br\s*/?>", cell)
    }


def _highway_values(cell):
    """The highway values the first cell of a row names, plain or as
    ``{{TagValue|highway|...}}``."""
    tagged = re.findall(r"\{\{[Tt]ag(?:Value)?\|highway\|([a-z_]+)\}\}", cell)
    return set(tagged) | set(re.findall(r"[a-z_]+", re.sub(r"\{\{.*?\}\}", " ", cell)))


def _table_rows(table):
    """The column names of a wiki table and its rows, each a list of the cells'
    attributes and contents."""
    columns, rows, row = [], [], None
    for line in table.splitlines()[1:]:
        line = line.strip()
        if line.startswith("|}"):
            break
        if line.startswith("|-"):
            if row:
                rows.append(row)
            row = []
        elif line.startswith("!"):
            columns += [
                _column_names(cell.lstrip("|"))
                for cell in _top_level_split(line[1:], "!!")
            ]
        elif line.startswith("|") and not line.startswith("|+") and row is not None:
            row += [_cell_parts(cell) for cell in _top_level_split(line[1:], "||")]
        elif row:
            attributes, content = row[-1]
            row[-1] = (attributes, (content + "\n" + line).strip())
    if row:
        rows.append(row)
    return columns, rows


def _row_spans(attributes):
    """The largest ``rowspan`` and ``colspan`` among the attributes of a row's cells, 1
    without one. Raises ``ValueError`` for a span that is not a number."""
    spans = []
    for key in ("rowspan", "colspan"):
        given = re.findall(key + r"\s*=\s*[\"']?([^\s\"']*)", attributes)
        if not all(value.isdigit() for value in given):
            raise ValueError(
                "A span in the wiki's table is not a number: %r" % attributes
            )
        spans.append(max((int(value) for value in given), default=1))
    return spans


def _cell_allows_mode(cell):
    """Whether a table cell lets the mode use the way: ``{{yes...}}`` and
    ``{{partial...}}`` do, unless the latter says dismount; ``{{no...}}`` does not.
    Raises ``ValueError`` for another value."""
    value = re.search(r"\{\{(\w+)([^}]*)\}\}", cell)
    if value is None or value.group(1).lower() not in ("yes", "no", "partial"):
        raise ValueError("unknown access value in the wiki's table: %r" % cell)
    kind = value.group(1).lower()
    return kind == "yes" or (kind == "partial" and "dismount" not in cell.lower())


def _section_defaults(name, section):
    """The shared paths, as ``(mode, highway)``, that the tables of the country ``name``
    in its section of the wiki page let the mode use: the first row that names the
    highway value decides, leaving out the rows with merged cells (``colspan``, or
    ``rowspan`` there or in a row above). Raises ``ValueError`` when the tables have no
    column for a mode and for a row that names the highway value without a cell for
    every column."""
    found = re.findall(r"^\s*\{\|.*?^\s*\|\}", section, flags=re.M | re.S)
    tables = [_table_rows(table) for table in found]
    modes = {mode for mode, _ in SHARED_PATHS}
    missing = sorted(
        modes - {mode for columns, _ in tables for names in columns for mode in names}
    )
    if missing:
        raise ValueError(
            "The wiki's table for %s has no %s column" % (name, " or ".join(missing))
        )
    decided, allowed = set(), set()
    for columns, rows in tables:
        spanned = 0  # the rows below still covered by a rowspan cell
        for row in rows:
            attributes = " ".join(cell[0] for cell in row)
            rows_down, across = _row_spans(attributes)
            merged = spanned > 0 or rows_down > 1 or across > 1
            spanned = max(spanned - 1, rows_down - 1)
            highways = _highway_values(row[0][1])
            for mode, highway in SHARED_PATHS:
                index = [i for i, names in enumerate(columns) if mode in names]
                if (mode, highway) in decided or not index or highway not in highways:
                    continue
                if merged:
                    continue
                decided.add((mode, highway))
                if len(row) != len(columns):
                    raise ValueError(
                        "A row of the wiki's table for %s has %d cells for %d "
                        "columns: %r" % (name, len(row), len(columns), row[0][1])
                    )
                if _cell_allows_mode(row[index[0]][1]):
                    allowed.add((mode, highway))
    return allowed


def wiki_country_defaults(wikitext):
    """For each shared path, the ISO codes of the countries whose table on the wiki page
    lets the mode use such a way (see ``_section_defaults``). Raises ``ValueError`` for a
    page without country sections, for a country it has no ISO code for and for a
    country with two sections."""
    text = _without_notes(wikitext)
    by_country = text[text.index("== By country ==") : text.index("== See Also ==")]
    sections = re.split(r"^===(?!=) *(.+?) *(?<!=)===\s*$", by_country, flags=re.M)[1:]
    if not sections:
        raise ValueError("The wiki page has no country sections")
    allowed, seen = {path: set() for path in SHARED_PATHS}, set()
    for name, section in zip(sections[0::2], sections[1::2]):
        if name not in WIKI_COUNTRIES:
            raise ValueError("The wiki page has a new country: %s" % name)
        if name in seen:
            raise ValueError("The wiki page has two sections for %s" % name)
        seen.add(name)
        for path in _section_defaults(name, section):
            allowed[path].add(WIKI_COUNTRIES[name])
    return allowed


def country_access(wikitext, revision):
    """The vendored access defaults from the wiki page's wikitext and ``TRAFFIC_LAW``: for
    each mode's tag and highway value, the sorted ISO codes of the countries whose default
    lets the mode use such a way, and the provisions of the countries added from their
    law. ``revision`` is the page's revision id and date, named in the source."""
    allowed = wiki_country_defaults(wikitext)
    law = {}
    for (mode, highway), provisions in TRAFFIC_LAW.items():
        allowed[mode, highway].update(provisions)
        law.setdefault(mode, {})[highway] = dict(sorted(provisions.items()))
    by_mode = {}
    for (mode, highway), codes in allowed.items():
        by_mode.setdefault(mode, {})[highway] = sorted(codes)
    return {
        "source": "OSM wiki, %s, revision %d of %s; the countries in 'law' from their "
        "traffic law" % (WIKI_PAGE, *revision),
        "allowed": by_mode,
        "law": law,
    }


def _page_revision(response):
    """The wikitext of the revision in the wiki API's ``response`` to ``WIKI_QUERY``, with
    the revision's id and date."""
    revision = response["query"]["pages"][0]["revisions"][0]
    content = revision["slots"]["main"]["content"]
    return content, (revision["revid"], revision["timestamp"][:10])


def update_access(source=None):
    """Write the access defaults from the wiki API's answer saved in ``source``, or from
    the wiki page's latest revision."""
    if source is None:
        response = _download(WIKI_API + "?" + WIKI_QUERY)
    else:
        response = Path(source).read_bytes()
    access = country_access(*_page_revision(json.loads(response)))
    path = DATA_DIR / "country_access.json"
    path.write_text(
        json.dumps(access, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote %s" % path)


def main():
    """Refresh the target given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", choices=["access", "borders"])
    parser.add_argument("--source", help="read the data from this file")
    args = parser.parse_args()
    if args.target == "borders":
        update_borders(args.source)
    else:
        update_access(args.source)


if __name__ == "__main__":
    main()
