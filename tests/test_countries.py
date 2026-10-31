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
