def test_country_codes():
    """The country of each point by the vendored borders, which also cover islands and the
    water near the coast; a territory counts as its country; points far out at sea get
    None, on both sides of the antimeridian too."""
    from pyrosm.data.countries import country_codes

    places = [
        (24.94, 60.17, "FI"),  # Helsinki
        (24.988, 60.146, "FI"),  # Suomenlinna, an island off Helsinki
        (21.37, 59.78, "FI"),  # Utö, an island in the outer archipelago
        (19.131, 60.3005, "FI"),  # Märket, on the Åland Islands' side
        (10.75, 59.91, "NO"),  # Oslo
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
    monkeypatch.setattr(countries, "_borders", lambda: (codes, shapely.STRtree(areas)))
    lon, lat = [0.5, 1.0, 2.0], [0.5, 0.5, 0.5]
    assert countries.country_codes(lon, lat).tolist() == ["AA", None, "BB"]


def test_country_borders_refresh(tmp_path, monkeypatch):
    """The refresh script keeps the areas with a geometry and a country, codes each by its
    sovereign country, copies the licence, and writes the same bytes on every run."""
    import gzip
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

    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    features = [
        {"properties": {"nameEn": "Mainland Finland", "country": "FI"}, "geometry": square},
        {"properties": {"iso1A2": "AX", "country": "FI"}, "geometry": square},
        {"properties": {"iso1A2": "NO"}, "geometry": square},
        {"properties": {"iso1A2": "EU"}, "geometry": None},
        {"properties": {"nameEn": "No code"}, "geometry": square},
    ]
    source = tmp_path / "borders.json"
    source.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    license = tmp_path / "LICENSE.md"
    license.write_bytes(b"ISC License\n")
    out = tmp_path / "data"
    out.mkdir()
    monkeypatch.setattr(module, "DATA_DIR", out)

    module.update_borders(source, license)
    first = (out / "country_borders.geojson.gz").read_bytes()
    module.update_borders(source, license)
    assert (out / "country_borders.geojson.gz").read_bytes() == first
    with gzip.open(out / "country_borders.geojson.gz", "rt") as f:
        borders = json.load(f)
    assert [f["properties"]["iso"] for f in borders["features"]] == ["FI", "FI", "NO"]
    assert (out / "country_borders.LICENSE.md").read_bytes() == b"ISC License\n"


def test_countries_allowing():
    """The countries where pedestrians may use a cycle way and cyclists a foot way without
    a tag for the mode, from Valhalla's table."""
    from pyrosm.data.countries import countries_allowing

    assert countries_allowing("foot", "cycleway") == {
        "BE", "BY", "CH", "DK", "FI", "GB", "GR", "HU", "IS",
        "NL", "NO", "OM", "PH", "PL", "SE", "TH", "TR", "US",
    }  # fmt: skip
    assert countries_allowing("bicycle", "footway") == {"AU", "BY", "IS", "NO"}
    assert countries_allowing("foot", "path") == frozenset()


def test_country_access_refresh(tmp_path, monkeypatch):
    """The refresh script reads the shared paths from Valhalla's table, leaves out the
    parts of the United Kingdom and stops at a country it has no ISO code for."""
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

    ped, bike = "baldr::kPedestrianAccess", "baldr::kBicycleAccess"
    table = (
        "const std::unordered_map<std::string, std::vector<int>>\n"
        '    kCountryAccess{{"Finland", {-1, -1, -1, -1, (%s), -1, (%s | %s), -1, -1}},\n'
        '                   {"Norway", {-1, -1, -1, (%s | %s), -1, -1, (%s), -1, -1}},\n'
        '                   {"England", {-1, -1, -1, -1, -1, -1, (%s), -1, -1}},\n'
        '                   {"Austria", {-1, -1, -1, -1, -1, -1, -1, (%s), -1}}};\n'
    ) % (ped, ped, bike, ped, bike, ped, ped, ped)
    assert module.country_access(table)["allowed"] == {
        "foot": {"cycleway": ["FI", "NO"]},
        "bicycle": {"footway": ["NO"]},
    }
    with pytest.raises(ValueError, match="Atlantis"):
        module.country_access(table.replace("Austria", "Atlantis"))
    source = tmp_path / "adminconstants.h"
    source.write_text(table, encoding="utf-8")
    license = tmp_path / "COPYING"
    license.write_bytes(b"MIT License\n")
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    module.update_access(source, license)
    written = json.loads((tmp_path / "country_access.json").read_text())
    assert written["allowed"]["foot"]["cycleway"] == ["FI", "NO"]
    assert (tmp_path / "country_access.LICENSE.md").read_bytes() == b"MIT License\n"


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
