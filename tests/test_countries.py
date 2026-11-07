def test_country_codes():
    """The country of each point by the vendored borders, which also cover islands and the
    water near the coast; a territory counts as its country unless the wiki gives it a
    table; a disputed region that two countries' boundaries share and points far out at
    sea get None, on both sides of the antimeridian too."""
    from pyrosm.data.countries import country_codes

    places = [
        (24.94, 60.17, "FI"),  # Helsinki
        (24.988, 60.146, "FI"),  # Suomenlinna, an island off Helsinki
        (21.37, 59.78, "FI"),  # Utö, an island in the outer archipelago
        (19.94, 60.10, "FI"),  # Mariehamn, on the Åland Islands
        (10.75, 59.91, "NO"),  # Oslo
        (114.17, 22.30, "HK"),  # Hong Kong, with its own table on the wiki
        (113.55, 22.19, "CN"),  # Macau
        (34.10, 44.95, None),  # Simferopol, in both Russia's and Ukraine's boundary
        (179.9, -16.6, "FJ"),  # Fiji, east of the antimeridian
        (-179.9, -16.6, "FJ"),  # Fiji, west of it
        (-40.0, 30.0, None),  # the Atlantic
    ]
    lon, lat, codes = zip(*places)
    assert country_codes(lon, lat).tolist() == list(codes)
    assert country_codes(24.94, 60.17).tolist() == ["FI"]
    assert country_codes([], []).tolist() == []


def test_country_codes_on_a_border(monkeypatch):
    """A point on the border of two countries gets None; on the border of two areas of one
    country, that country."""
    import numpy as np
    import shapely

    from pyrosm.data import countries

    areas = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), shapely.box(2, 0, 3, 1)]
    codes = np.array(["AA", "BB", "BB"], dtype=object)
    monkeypatch.setattr(countries, "_borders", lambda: (codes, np.array(areas)))
    lon, lat = [0.5, 1.0, 2.0], [0.5, 0.5, 0.5]
    assert countries.country_codes(lon, lat).tolist() == ["AA", None, "BB"]


def test_country_borders_refresh(tmp_path, monkeypatch):
    """The refresh script builds one area per country from its admin_level=2 boundary, a
    ring inside another a hole; a territory counts as its country unless the wiki gives it
    a table, and then it is cut out of its country; where two countries' boundaries
    overlap, or one lies within the other, their common part stays in both; a download in
    batches, gzip-compressed or not, writes the same bytes as the saved answer, on every
    run; a relation that is not an administrative boundary is left out. A broken boundary,
    a boundary without ways, two countries with one code, a wiki territory without a
    boundary, two countries with the same area, three countries overlapping in one place,
    an answer that reports a stopped query (downloaded or saved), a batch without exactly
    its relations and a boundary whose tags change during the download stop the
    refresh."""
    import copy
    import gzip
    import importlib.util
    import json
    from pathlib import Path
    from urllib.parse import parse_qs, urlsplit

    import pytest
    from shapely.geometry import shape

    script = Path(__file__).parents[1] / "scripts" / "update_country_data.py"
    if not script.exists():
        pytest.skip("The maintainer script %s is not in this source tree." % script.name)
    spec = importlib.util.spec_from_file_location("update_country_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def way(*coords):
        geometry = [{"lon": x, "lat": y} for x, y in coords]
        return {"type": "way", "ref": 0, "role": "outer", "geometry": geometry}

    def boundary(osm_id, code, level, *ways):
        tags = {"boundary": "administrative", "admin_level": level}
        if code is not None:
            tags["ISO3166-1:alpha2"] = code
        label = {"type": "node", "ref": 0, "role": "label"}
        return {"type": "relation", "id": osm_id, "tags": tags, "members": [*ways, label]}

    relations = [
        boundary(
            1,
            "AA",
            "2",
            way((0, 0), (2, 0), (2, 2)),
            way((2, 2), (0, 2), (0, 0)),
            way((1.5, 1.5), (1.8, 1.5), (1.8, 1.8), (1.5, 1.8), (1.5, 1.5)),
        ),
        boundary(2, "BB", "2", way((2, 0), (4, 0), (4, 2), (2, 2), (2, 0))),
        boundary(6, "CC", "2", way((3, 0), (5, 0), (5, 2), (3, 2), (3, 0))),
        boundary(3, "AX", "3", way((0, 1), (0.2, 1), (0.2, 1.2), (0, 1))),
        boundary(4, "HK", "3", way((0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1), (0.5, 0.5))),
        boundary(5, None, "4", way((3, 0), (3, 1), (3.5, 1), (3, 0))),
        boundary(7, "EE", "2", way((2.2, 0.2), (2.6, 0.2), (2.6, 0.6), (2.2, 0.2))),
        boundary(9, "QQ", "2", way((6, 0), (7, 0), (7, 1), (6, 0))),
    ]
    relations[-1]["tags"]["boundary"] = "maritime"
    answer = {"osm3s": {"timestamp_osm_base": "2026-11-07T08:00:00Z"}, "elements": relations}
    source = tmp_path / "boundaries.json"
    source.write_text(json.dumps(answer))
    out = tmp_path / "data"
    out.mkdir()
    monkeypatch.setattr(module, "DATA_DIR", out)
    monkeypatch.setattr(module, "WIKI_COUNTRIES", {"Hong Kong": "HK"})

    module.update_borders(source)
    first = (out / "country_borders.geojson.gz").read_bytes()
    with gzip.open(out / "country_borders.geojson.gz", "rt") as f:
        borders = json.load(f)
    assert "2026-11-07" in borders["source"]
    areas = {f["properties"]["iso"]: shape(f["geometry"]) for f in borders["features"]}
    assert list(areas) == ["AA", "BB", "CC", "EE", "HK"]
    assert [round(areas[code].area, 6) for code in areas] == [3.66, 4.0, 4.0, 0.08, 0.25]
    assert round(areas["BB"].intersection(areas["CC"]).area, 6) == 2.0
    assert round(areas["BB"].intersection(areas["EE"]).area, 6) == 0.08

    class Response:
        def __init__(self, document, compressed):
            self.body = json.dumps(document).encode()
            self.headers = {}
            if compressed:
                self.body = gzip.compress(self.body)
                self.headers["Content-Encoding"] = "gzip"

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    sent, served = [], {"remark": None, "skip": None, "tags": {}}

    def urlopen(request, context=None, timeout=None):
        sent.append(request.get_header("Accept-encoding"))
        query = parse_qs(urlsplit(request.full_url).query)["data"][0]
        if query == module.BOUNDARIES_QUERY:
            tags = [{"type": "relation", "id": r["id"], "tags": r["tags"]} for r in relations]
            return Response({"osm3s": answer["osm3s"], "elements": tags}, True)
        ids = query.split("relation(id:")[1].split(")")[0].split(",")
        found = copy.deepcopy(
            [r for r in relations if str(r["id"]) in ids and r["id"] != served["skip"]]
        )
        for relation in found:
            if relation["id"] == 2:
                relation["tags"].update(served["tags"])
        document = {"osm3s": answer["osm3s"], "elements": found}
        if served["remark"]:
            document["remark"] = served["remark"]
        return Response(document, False)

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(module, "GEOMETRY_BATCH", 2)
    module.update_borders()
    assert (out / "country_borders.geojson.gz").read_bytes() == first
    assert set(sent) == {"gzip"}
    served["remark"] = "runtime error: Query timed out"
    with pytest.raises(ValueError, match="stopped the query: runtime error"):
        module.update_borders()
    served.update(remark=None, skip=2)
    with pytest.raises(ValueError, match="returned the relations 1 for 1, 2"):
        module.update_borders()
    for changed in ({"admin_level": "4"}, {"ISO3166-1:alpha2": "ZZ"}, {"boundary": ""}):
        served.update(skip=None, tags=changed)
        with pytest.raises(ValueError, match="changed during the download"):
            module.update_borders()
    stopped = tmp_path / "stopped.json"
    stopped.write_text(json.dumps({**answer, "remark": "runtime error: out of memory"}))
    with pytest.raises(ValueError, match="stopped the query: runtime error"):
        module.update_borders(stopped)

    broken = copy.deepcopy(answer)
    del broken["elements"][0]["members"][1]
    wayless = copy.deepcopy(answer)
    wayless["elements"][1]["members"] = wayless["elements"][1]["members"][-1:]
    twice = copy.deepcopy(answer)
    twice["elements"][1]["tags"]["ISO3166-1:alpha2"] = "AA"
    missing = copy.deepcopy(answer)
    del missing["elements"][4]
    triple = copy.deepcopy(answer)
    triple["elements"].append(
        boundary(8, "DD", "2", way((3.5, 0), (4.5, 0), (4.5, 2), (3.5, 2), (3.5, 0)))
    )
    same = copy.deepcopy(answer)
    same["elements"][2]["members"] = same["elements"][1]["members"]
    cases = [
        (broken, "relation 1 is not closed"),
        (wayless, "relation 2 has no area"),
        (same, "The boundaries of BB, CC are the same"),
        (twice, "Two country boundaries have the code AA"),
        (missing, "No boundary has the code HK"),
        (triple, "The boundaries of BB, CC, DD overlap"),
    ]
    for document, message in cases:
        with pytest.raises(ValueError, match=message):
            module.country_borders(document)


def test_countries_allowing():
    """The countries where pedestrians may use a cycle way and cyclists a foot way without
    a tag for the mode, from the OSM wiki's tables and, for Estonia, Lithuania, New
    Zealand, Portugal and Ukraine, from their traffic law."""
    from pyrosm.data.countries import countries_allowing

    assert countries_allowing("foot", "cycleway") == {
        "BE", "BY", "CH", "CN", "DK", "EE", "FI", "GB", "GR", "HU", "IS", "LT", "NL",
        "NO", "NZ", "OM", "PH", "PL", "PT", "RU", "SE", "TH", "TR", "UA", "US",
    }  # fmt: skip
    assert countries_allowing("bicycle", "footway") == {"BY", "IS", "NO", "RU"}
    assert countries_allowing("foot", "path") == frozenset()


def test_country_access_refresh(tmp_path, monkeypatch):
    """The refresh script reads the shared paths from the country tables of the OSM wiki
    page: ``yes`` and ``partial`` allow the mode, ``no`` and a ``partial`` dismount do
    not, the first row naming the way decides, leaving out rows with merged cells, and
    a table without a value leaves the country out; comments, footnotes, cell attributes
    (spans of 1 merge nothing), header spellings, template names with a lower-case first
    letter, separators inside templates and cells over several lines are read through. It adds the countries of its traffic-law table with
    their provisions, takes the revision from the saved API answer, and stops at a page
    without country sections, at a country it has no ISO code for or with two sections, at a table without a column for a mode, at a row
    without a cell for every column, at a span that is not a number and at an unknown
    value."""
    import importlib.util
    import json
    from pathlib import Path

    import pytest

    script = Path(__file__).parents[1] / "scripts" / "update_country_data.py"
    if not script.exists():
        pytest.skip("The maintainer script %s is not in this source tree." % script.name)
    spec = importlib.util.spec_from_file_location("update_country_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wiki = """== Worldwide ==
{| class="wikitable"
! highway !! foot
|-
| cycleway || {{yes}}
|}
== By country ==
=== Australia ===
{| class="wikitable"
 ! {{Tag|highway}} !! bicycle !! foot
 |-
 | {{TagValue|highway|path}} || colspan="2" rowspan="2" | ''See [[Guidelines|the guidelines]]''
 |-
 | {{TagValue|highway|cycleway}}
 |}
=== Finland ===
{| class="wikitable"
!| {{Tag|highway}}
!| bicycle
!| foot<br/>ski
<!-- |-
 | cycleway
 | {{no}}
 | {{no}} -->
 |-
 | cycleway
 | {{yes||designated}}
 | {{partial|yes (3)}}
 |-
 | footway
 | {{no}} (6)
 | {{yes|designated}}
 |}
=== Norway ===
{| class="wikitable"
!| {{Tag|highway}}
!| [[Key:bicycle|bicycle]]<ref group="NO">A note | with a bar
over two lines</ref>
!| [[key:foot|foot]]{{Refn|group=no|name=b|A note on {{tag|foot}}}}
 |-
 | footway (in parks) || colspan="2" | ''See below''
 |-
 | * {{tagValue|highway|footway}}
 | {{yes}}{{refn|group=no|name=a|Cyclists {{Tag|bicycle|yes}}}}
 | {{yes}}
 |-
 | footway + footway=sidewalk
 | {{no}}
 | {{yes}}
 |-
 | * {{TagValue|highway|path}}
* {{TagValue|highway|cycleway}}
 | {{yes|designated}}
 | {{yes|yes}}
 |}
==== Notes ====
A subsection of Norway's section.
{| class="wikitable"
! highway !! bicycle !! foot
|-
| footway || {{no}} || {{no}}
|}
=== Switzerland ===
{| class="wikitable"
! scope="col" | {{Tag|highway}} !! scope="col" | bicycle !! scope="col" | foot
 |-
 | cycleway || {{yes|designated}} || rowspan="1" style="background:#ffd" | {{partial|No (4)}}
 |-
 | footway || {{partial|dismount}} || {{yes}}
 |}
== See Also ==
"""
    law = sorted(module.TRAFFIC_LAW["foot", "cycleway"])
    revision = (3111745, "2026-11-04")
    access = module.country_access(wiki, revision)
    assert access["allowed"] == {
        "foot": {"cycleway": sorted(["CH", "FI", "NO"] + law)},
        "bicycle": {"footway": ["NO"]},
    }
    assert "revision 3111745 of 2026-11-04" in access["source"]
    with pytest.raises(ValueError, match="Atlantis"):
        module.country_access(wiki.replace("Switzerland", "Atlantis"), revision)
    with pytest.raises(ValueError, match="two sections for Switzerland"):
        module.country_access(wiki.replace("=== Finland", "=== Switzerland"), revision)
    with pytest.raises(ValueError, match="no country sections"):
        module.country_access(wiki.replace("===", "===="), revision)
    with pytest.raises(ValueError, match="maybe"):
        module.country_access(wiki.replace("{{partial|dismount}}", "{{maybe}}"), revision)
    with pytest.raises(ValueError, match="Finland has no bicycle column"):
        module.country_access(wiki.replace("!| bicycle\n", ""), revision)
    with pytest.raises(ValueError, match="Finland has 2 cells for 3 columns"):
        module.country_access(wiki.replace(" | {{no}} (6)\n", ""), revision)
    with pytest.raises(ValueError, match="not a number"):
        module.country_access(wiki.replace('rowspan="1"', 'rowspan="all"'), revision)
    answer = {
        "query": {
            "pages": [
                {
                    "revisions": [
                        {
                            "revid": 3111745,
                            "timestamp": "2026-11-04T09:12:00Z",
                            "slots": {"main": {"content": wiki}},
                        }
                    ]
                }
            ]
        }
    }
    source = tmp_path / "access.json"
    source.write_text(json.dumps(answer), encoding="utf-8")
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    module.update_access(source)
    written = json.loads((tmp_path / "country_access.json").read_text(encoding="utf-8"))
    assert written["allowed"]["foot"]["cycleway"] == sorted(["CH", "FI", "NO"] + law)
    assert written["law"] == {"foot": {"cycleway": module.TRAFFIC_LAW["foot", "cycleway"]}}
    assert written["source"] == access["source"]


def test_filter_for_extent(tmp_path):
    """A walking or cycling filter decides the ways without a tag of the mode for the whole
    read when the countries of its area agree, and leaves them to each way's country when
    they differ or the area is unknown; the area comes from the file's header box, within
    the read's bounding box."""
    import numpy as np
    import shapely

    from pyrosm.config.osm_filters import filter_for_extent, get_osm_filter
    from pyrosm.data.countries import read_extent
    from pyrosm.pbf_export import write_pbf_from_records

    walking, cycling = get_osm_filter("walking"), get_osm_filter("cycling")
    helsinki = shapely.box(24.8, 60.1, 25.1, 60.3)
    benelux = shapely.box(6.8, 51.9, 7.7, 52.3)  # the Netherlands and Germany
    # Finland lets pedestrians use cycle ways: the filter keeps them (the rule's access
    # hierarchy alone), and leaves foot ways without a bicycle tag out of cycling.
    kept = filter_for_extent(walking, "exclude", helsinki)["access"]
    assert not kept.other_mode_highway
    dropped = filter_for_extent(cycling, "exclude", helsinki)["access"]
    assert list(dropped.other_mode_highway) == ["footway"] and not dropped.by_country
    for extent in (benelux, None):
        assert filter_for_extent(walking, "exclude", extent) is walking
    assert filter_for_extent(walking, "keep", helsinki) is walking
    driving = get_osm_filter("driving")
    assert filter_for_extent(driving, "exclude", helsinki) is driving

    nodes = {
        "id": np.array([1, 2], dtype=np.int64),
        "lon": np.array([24.9, 25.0]),
        "lat": np.array([60.2, 60.2]),
        "version": np.ones(2, dtype=np.int64),
        "timestamp": np.full(2, 1600000000, dtype=np.int64),
        "changeset": np.zeros(2, dtype=np.int64),
        "tags": [None, None],
    }
    way = {"id": 10, "refs": [1, 2], "version": 1, "timestamp": 1600000000, "tags": {}}
    boxed = write_pbf_from_records(
        nodes, [way], [], str(tmp_path / "boxed.osm.pbf"), (24.0, 60.0, 26.0, 61.0)
    )
    extent = read_extent(boxed, [24.9, 60.1, 25.0, 60.3])
    assert np.allclose(extent.bounds, (24.85, 60.05, 25.05, 60.35))
    unboxed = write_pbf_from_records(nodes, [way], [], str(tmp_path / "unboxed.osm.pbf"), None)
    assert read_extent(unboxed) is None


def test_allowed_in_by_each_ways_highway_value():
    """With several other-mode values, each way is allowed by its own highway value, also
    among several values; a way without a located point is not allowed."""
    import numpy as np

    from pyrosm.config.osm_filters import AccessRule
    from pyrosm.data.countries import way_points

    # Norway lets cyclists use foot ways, but its table has no row for pedestrian streets.
    rule = AccessRule("bicycle", other_mode_highway=["footway", "pedestrian"])
    highway = ["footway", "pedestrian", "pedestrian;footway", "footway"]
    lon, lat = [10.75, 10.75, 10.75, np.nan], [59.91, 59.91, 59.91, np.nan]
    assert rule.allowed_in(highway, lon, lat).tolist() == [True, False, True, False]
    # A part outside the rule's values does not count: Finland lets pedestrians use
    # cycle ways, but this rule is about foot ways and pedestrian streets.
    walking = AccessRule("foot", other_mode_highway=["footway", "pedestrian"])
    assert walking.allowed_in(["footway;cycleway"], [24.94], [60.17]).tolist() == [False]
    lon, lat = way_points([], [0, 0], None)
    assert np.isnan(lon).all() and np.isnan(lat).all()
