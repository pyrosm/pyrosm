import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from pyrosm import cli, get_data


def test_find_lists_the_extracts_of_an_area(tmp_path, monkeypatch, capsys):
    """``pyrosm find`` takes a bounding box (also one that starts with a minus sign), a
    vector file or a place name and prints the extracts as a table or as JSON; a place that
    cannot be geocoded or a file that cannot be read exits 1, a missing area 2."""
    from pyrosm.data import extract_index, geocoding

    found = gpd.GeoDataFrame(
        {
            "provider": ["BBBike", "Geofabrik"],
            "id": ["Helsinki", "finland"],
            "name": ["Helsinki", "Finland"],
            "url": ["https://bbbike/Helsinki.osm.pbf", "https://geofabrik/finland.osm.pbf"],
            "bytes": pd.array([65_100_000, None], dtype="Int64"),
            "contains": [True, False],
        },
        geometry=[box(24, 60, 25, 61), box(20, 59, 32, 70)],
        crs="EPSG:4326",
    )
    asked = []

    def find_extracts(area, contains_only=False):
        asked.append((area, contains_only))
        return found if len(asked) < 4 else found.iloc[:0]

    def geocode(query):
        if query != "Helsinki, Finland":
            raise ValueError("Could not geocode '%s'." % query)
        return box(24.9, 60.1, 25.0, 60.2)

    monkeypatch.setattr(extract_index, "find_extracts", find_extracts)
    monkeypatch.setattr(geocoding, "geocode", geocode)
    vector = tmp_path / "area.geojson"
    gpd.GeoDataFrame(geometry=[box(24.9, 60.1, 25.0, 60.2)], crs=4326).to_file(vector)

    assert cli.main(["find", "-74.1,40.6,-73.9,40.8"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[1].split() == ["BBBike", "Helsinki", "65.1", "yes", found.url[0]]
    assert table[2].split() == ["Geofabrik", "Finland", "unknown", "no", found.url[1]]
    assert cli.main(["find", str(vector), "--contains-only", "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert records[0] == {
        "provider": "BBBike",
        "id": "Helsinki",
        "name": "Helsinki",
        "url": found.url[0],
        "bytes": 65_100_000,
        "contains": True,
        "bounds": [24.0, 60.0, 25.0, 61.0],
    }
    assert records[1]["bytes"] is None
    assert cli.main(["find", "Helsinki, Finland"]) == 0
    assert cli.main(["find", "Helsinki, Finland"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "none"
    assert cli.main(["find", "Atlantis, a, b, c"]) == 1
    assert capsys.readouterr().err == "pyrosm: Could not geocode 'Atlantis, a, b, c'.\n"
    (tmp_path / "notes.geojson").write_text("not a vector file")
    assert cli.main(["find", str(tmp_path / "notes.geojson")]) == 1
    assert "Could not read" in capsys.readouterr().err
    for argv in (["find"], ["find", "--json"], ["find", "Helsinki", "-x"]):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 2
    assert asked[0] == ([-74.1, 40.6, -73.9, 40.8], False)
    assert isinstance(asked[1][0], gpd.GeoDataFrame) and asked[1][1] is True
    assert asked[2][0].equals(box(24.9, 60.1, 25.0, 60.2)) and len(asked) == 4


def test_info_describes_a_pbf_file(tmp_path, capsys):
    """``pyrosm info`` prints a PBF file's size, bounding box, snapshot time and SHA-256, as
    lines or as JSON; a file that is not a PBF file exits 1."""
    path = get_data("test_pbf")
    assert cli.main(["info", path, "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    sha256 = "39a274a125205531b4d1de7d0059802ffbb3f1a4cec915d0399c8b195274767b"
    assert info["sha256"] == sha256 and info["snapshot"] is None
    assert info["bounding_box"] == pytest.approx([26.93, 60.52, 26.97, 60.54])
    assert cli.main(["info", path]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == ["snapshot: none", "sha256: %s" % sha256]
    assert lines[2].startswith("bounding box: 26.9")
    text = tmp_path / "notes.osm.pbf"
    text.write_text("not a PBF file")
    assert cli.main(["info", str(text)]) == 1
    assert "is not a valid OSM PBF file" in capsys.readouterr().err


def test_downloads_lists_and_clears_the_downloaded_files(tmp_path, monkeypatch, capsys):
    """``pyrosm downloads list`` prints the downloaded PBF files; ``clear`` removes the one
    named or all of them; a name that is not a PBF file exits 1."""
    from pyrosm.utils import download as dl

    monkeypatch.setattr(dl, "download_dir", lambda: tmp_path)
    for name in ("a.osm.pbf", "b.osm.pbf", "c.osm.pbf"):
        (tmp_path / name).write_bytes(b"x")
    assert cli.main(["downloads", "list"]) == 0
    expected = [str(tmp_path / n) for n in ("a.osm.pbf", "b.osm.pbf", "c.osm.pbf")]
    assert capsys.readouterr().out.splitlines() == expected
    assert cli.main(["downloads", "clear", "a.osm.pbf"]) == 0
    assert cli.main(["downloads", "clear"]) == 0
    assert capsys.readouterr().out.splitlines() == ["removed 1 file", "removed 2 files"]
    assert cli.main(["downloads", "clear", "notes.txt"]) == 1
    assert "removes .osm.pbf and .osh.pbf files only" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["downloads"])
    assert exit_info.value.code == 2
