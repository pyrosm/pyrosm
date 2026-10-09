import dataclasses
import functools
import json
import os
import re
import shutil
import struct
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geopandas.testing import assert_geodataframe_equal
import pytest
from shapely.geometry import box

import pyrosm
from pyrosm import OSM, get_data, recipes
from pyrosm.data import extract_index
from pyrosm.exceptions import ExtractDownloadError
from pyrosm.graphs import graph_tables
from pyrosm.proto.fileformat_pb2 import BlobHeader

PBF = Path(get_data("test_pbf")).as_posix()


def _recipe(tmp_path, text, name="recipe.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_validate_resolves_a_recipe(tmp_path):
    """A valid recipe resolves: input paths against the recipe's folder, keywords as given,
    the default provenance name from the recipe's name."""
    (tmp_path / "stops.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    layers = _recipe(
        tmp_path,
        f"""
        recipe: pyrosm
        pbf: {{file: '{PBF}'}}
        reader: {{keep_metadata: false, bounding_box: [24.9, 60.1, 25.0, 60.2]}}
        layers:
          walk: {{read: network, network_type: walking, extra_attributes: [lit]}}
          shops: {{read: pois, custom_filter: {{shop: true}}}}
        graph:
          network: {{network_type: driving}}
          simplify: true
        outputs:
          layers: {{walk: out/walk.parquet, shops: shops.parquet}}
          graph: {{nodes: nodes.parquet, edges: edges.parquet}}
        """,
        name="study.yaml",
    )
    resolved = recipes.validate(layers)
    assert resolved["pbf"] == {"file": Path(PBF).resolve()}
    assert resolved["reader"]["bounding_box"] == [24.9, 60.1, 25.0, 60.2]
    assert resolved["layers"]["shops"] == {
        "read": "pois",
        "keywords": {"custom_filter": {"shop": True}},
    }
    assert resolved["graph"] == {
        "network": {"network_type": "driving"},
        "keywords": {"simplify": True},
    }
    assert resolved["outputs"]["layers"]["walk"] == "out/walk.parquet"
    assert resolved["outputs"]["provenance"] == "study.provenance.json"

    extract = _recipe(
        tmp_path,
        """
        recipe: pyrosm
        extract:
          area: {bbox: [24.9, 60.1, 25.0, 60.2]}
          strategy: smallest_total
          must_cover: {file: stops.geojson, layer: stops}
        outputs: {extract: data.osm.pbf}
        """,
    )
    resolved = recipes.validate(extract)["extract"]
    assert resolved["call"] == "get_data_by_area"
    assert resolved["area"] == {"bbox": [24.9, 60.1, 25.0, 60.2]}
    assert resolved["keywords"]["must_cover"] == {
        "file": tmp_path / "stops.geojson",
        "layer": "stops",
    }


_EXTRACT = "recipe: pyrosm\noutputs: {extract: a.pbf}\nextract: "

_LAYERS = (
    "recipe: pyrosm\nlayers: {walk: {read: network}}\n"
    "outputs: {layers: {walk: walk.parquet}}\n"
)
_GOOD = _LAYERS + f"pbf: {{file: '{PBF}'}}\n"
_TWO = _GOOD.replace("{read: network}", "{read: network}, b: {read: pois}")


def _keyword(text):
    """The good recipe with ``text`` added to its layer's keywords."""
    return _GOOD.replace("read: network", "read: network, " + text)


@pytest.mark.parametrize(
    "text, message",
    [
        ("- just\n- a list\n", "must be a YAML mapping"),
        (_TWO.replace("parquet}", "parquet, b: WALK.parquet/b.parquet}"), "inside the other"),
        (_GOOD.replace(PBF, "loop.pbf"), "does not exist|cannot be resolved"),
        (_GOOD + "version: 1.0\n", "not a known recipe version"),
        (_GOOD.replace(PBF, "cut.pbf"), "does not start with an OSM PBF header"),
        (_GOOD.replace(PBF, "headless.pbf"), "does not start with an OSM PBF header"),
        (_GOOD.replace("read: network", "read: [network]"), "layers.walk.read: must be one"),
        (_GOOD.replace("walk: {read", "1: {read"), "layer names must be text"),
        (_keyword("extra_attributes: &a [*a]"), "repeats a list or mapping"),
        (_keyword("extra_attributes: [&a [x], *a]"), "repeats a list or mapping"),
        ("? [a, b]\n: 1\n", "mapping keys must be scalars"),
        ("recipe: pyrosm\nlayers: [walk]\n", "layers: must be a mapping"),
        ("recipe: pyrosm\nlayers: {}\noutputs: {}\n", "needs at least one layer"),
        (_GOOD.replace("outputs: {layers: {walk: walk.parquet}}", ""), "needs an outputs"),
        (_keyword("extra_attributes: {1: x}"), "keys must be text"),
        (_keyword("network_type: .inf"), "finite"),
        (_GOOD.replace(PBF, "short.pbf"), "does not start with an OSM PBF header"),
        (_GOOD.replace(PBF, "garbage.pbf"), "does not start with an OSM PBF header"),
        (_GOOD.replace(PBF, ""), "pbf.file: must be a file path"),
        (_EXTRACT + "{area: {file: empty.gpkg}}\n", "'empty.gpkg' is empty"),
        (_EXTRACT + "{area: {file: a.gpkg, layer: ''}}\n", "must be a layer name"),
        (_EXTRACT + "{area: {bbox: [1, 2, 3]}}\n", "must be four numbers"),
        (_EXTRACT + "{strategy: single}\n", "extract: needs an area"),
        (_GOOD.replace("walk: walk.parquet", "walk: 5"), "must be a file path"),
        ("recipe: pyrosm\nextract: {area: {name: test_pbf}}\noutputs: {}\n", "needs 'extract'"),
        (_GOOD.replace("{layers:", "{extract: a.pbf, layers:"), "has no extract stage"),
        (_GOOD.replace("{walk: walk.parquet}", "{walk: w.parquet, b: b.parquet}"), "no layer"),
        (_GOOD.replace("parquet}", "parquet}, graph: {}"), "has no graph stage"),
        ("recipe: pyrosm\nrecipe: pyrosm\n", "duplicate key 'recipe'"),
        ("recipe: pyrosm\n: [\n", "not valid YAML"),
        (_GOOD + "colour: red\n", "unknown key 'colour'"),
        (_GOOD.replace("recipe: pyrosm", "recipe: cafein"), "must be 'pyrosm'"),
        (_GOOD + "version: 2\n", "not a known recipe version"),
        ("recipe: pyrosm\noutputs: {}\n", "needs at least one of extract"),
        (_LAYERS, "need an extract stage or a pbf input"),
        (_GOOD + "extract: {area: {name: test_pbf}}\n", "either an extract stage or a pbf input"),
        (_EXTRACT + "{area: {name: x, place: y}}\n", "needs exactly one of"),
        (_EXTRACT + "{area: {name: nowhere}}\n", "area.name: The dataset 'nowhere' is not"),
        (_EXTRACT + "{area: {file: a.parquet, layer: x}}\n", "GeoParquet file has no layers"),
        (_EXTRACT + "{area: {name: test_pbf}, directory: 5}\n", "must be a folder path"),
        (_EXTRACT + "{area: {bbox: [1, 2, 0, 3]}}\n", "needs minx < maxx"),
        (_EXTRACT + "{area: {place: ' '}}\n", "must be a non-empty text"),
        (_EXTRACT + "{area: {file: no.gpkg}}\n", "'no.gpkg' does not exist"),
        (_EXTRACT + "{area: {file: a.csv}}\n", "must end with one of"),
        (_GOOD.replace(PBF, "fake.pbf"), "does not start with an OSM PBF header"),
        (_EXTRACT + "{area: {name: test_pbf}, strategy: single}\n", "unknown keyword 'strategy'"),
        (_EXTRACT + "{area: {place: x}, output_path: x}\n", "output_path: set by the recipe"),
        (_EXTRACT + "{area: {name: test_pbf}, opener: {}}\n", "extract.opener: set by the recipe"),
        (_GOOD.replace("read: network", "read: rivers"), "layers.walk.read: must be one of"),
        (_keyword("nodes: true"), "set by the recipe"),
        (_GOOD + "graph: {colour: red}\n", "graph: unknown keyword 'colour'"),
        (_GOOD + "graph: {network_type: walking}\n", "set it as graph.network.network_type"),
        (_GOOD + "graph: {edges: e}\n", "graph.edges: set by the recipe"),
        (_keyword("timestamp: 2020-01-01"), "is not a text, number"),
        (_GOOD + "reader: {filepath: x.pbf}\n", "reader.filepath: set by the recipe"),
        (_GOOD.replace("walk: walk.parquet", "run: run.parquet"), "needs a file for layer"),
        (_GOOD.replace("walk.parquet", "/tmp/walk.parquet"), "must be a relative file path"),
        (_GOOD.replace("walk.parquet", "../walk.parquet"), "must be a relative file path"),
        (_GOOD.replace("walk.parquet", "walk.csv"), "must end with .parquet"),
        (_GOOD.replace("walk.parquet", "con.parquet"), "not a portable file name"),
        (_GOOD.replace("parquet}", "parquet}, provenance: %s.json" % ("a" * 245)), "over 249"),
        (_TWO.replace("parquet}", "parquet, b: Walk.parquet}"), "is named twice"),
        (_TWO.replace("walk.parquet}", "\u00e9.parquet, b: E\u0301.parquet}"), "named twice"),
        (
            "recipe: pyrosm\nextract: {area: {name: test_pbf}}\noutputs: {extract: a.pbf}\n"
            "reader: {engine: in_memory}\n",
            "no layers or graph to read",
        ),
    ],
)
def test_validate_refuses(tmp_path, text, message):
    """Each mistake is refused by name before anything runs."""
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "a.gpkg").write_bytes(b"SQLite format 3\x00")
    (tmp_path / "a.parquet").write_bytes(b"PAR1")
    (tmp_path / "empty.gpkg").write_bytes(b"")
    (tmp_path / "fake.pbf").write_bytes(b"not a pbf at all")
    (tmp_path / "short.pbf").write_bytes(b"ab")
    (tmp_path / "garbage.pbf").write_bytes(b"\x00\x00\x00\x05\xff\xff\xff\xff\xff")
    header = BlobHeader(type="OSMHeader", datasize=100).SerializeToString()
    # A header that claims more bytes than the file has, and one whose blob is missing.
    (tmp_path / "cut.pbf").write_bytes(struct.pack("!L", len(header) + 9) + header)
    (tmp_path / "headless.pbf").write_bytes(struct.pack("!L", len(header)) + header)
    try:  # a symbolic link loop; where links cannot be made the file is simply missing
        os.symlink(tmp_path / "loop2.pbf", tmp_path / "loop.pbf")
        os.symlink(tmp_path / "loop.pbf", tmp_path / "loop2.pbf")
    except OSError:
        pass
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        recipes.validate(_recipe(tmp_path, text))



@pytest.fixture
def offline(monkeypatch):
    """``get_data_by_area`` and ``geocode`` answered with the bundled test PBF; returns the
    calls to ``get_data_by_area``. A cropping call writes ``output_path``, a call with
    ``crop=False`` returns the downloaded file."""
    calls = []

    @functools.wraps(extract_index.get_data_by_area)
    def get_data_by_area(area, crop=True, output_path=None, **keywords):
        calls.append(dict(keywords, area=area, crop=crop))
        path = shutil.copyfile(PBF, output_path) if crop else PBF
        snapshot = datetime(2026, 10, 1, tzinfo=timezone.utc)
        source = extract_index.ExtractSource("Geofabrik", "t", "https://t/t.pbf", 9, PBF)
        source.snapshot = snapshot
        return extract_index.AreaExtract(
            str(path), "Geofabrik", "t", "https://t/t.pbf", 9, 0.0, 0.0,
            sources=[source], sha256=extract_index._sha256(path),
        )

    monkeypatch.setattr(extract_index, "get_data_by_area", get_data_by_area)
    monkeypatch.setattr(
        "pyrosm.data.geocoding.geocode", lambda place: box(24.9, 60.1, 25.0, 60.2)
    )
    return calls


@pytest.mark.parametrize(
    "extract",
    [
        "{area: {name: test_pbf}}",
        "{area: {bbox: [24.9, 60.1, 25.0, 60.2]}, must_cover: {file: stops.geojson}, "
        "headers: {Authorization: secret}}",
        "{area: {place: Kallio}, crop: false}",
        "{area: {file: area.parquet}, strategy: smallest_total}",
    ],
)
def test_run_extract(tmp_path, offline, extract):
    """The extract stage writes its PBF and a record of where it came from; nothing else is
    left in the output folder."""
    area = gpd.GeoDataFrame(geometry=[box(24.9, 60.1, 25.0, 60.2)], crs="EPSG:4326")
    area.to_parquet(tmp_path / "area.parquet")
    area.to_file(tmp_path / "stops.geojson")
    text = "recipe: pyrosm\nextract: %s\noutputs: {extract: x/a.osm.pbf}\n" % extract
    out = tmp_path / "out"
    written = recipes.run(_recipe(tmp_path, text), out)
    assert written == [out / "x/a.osm.pbf", out / "recipe.provenance.json"]
    assert sorted(p.name for p in out.iterdir()) == ["recipe.provenance.json", "x"]
    record = json.loads(written[1].read_text(encoding="utf-8"))
    sha256 = extract_index._sha256(written[0])
    assert record["outputs"] == {"x/a.osm.pbf": sha256}
    assert record["extract"]["sha256"] == sha256
    assert record["invocation"]["entry_point"] == "python"
    assert record["versions"]["pyrosm"]
    if "name" in extract:
        assert (record["extract"]["dataset"], record["extract"]["url"]) == ("test_pbf", None)
    else:
        assert record["extract"]["area"]["bounds"] == [24.9, 60.1, 25.0, 60.2]
        source = record["extract"]["sources"][0]
        assert (source["url"], source["snapshot"]) == ("https://t/t.pbf", "2026-10-01T00:00:00+00:00")
    if "must_cover" in extract:
        assert len(offline[0]["must_cover"]) == 1
        assert offline[0]["headers"] == {"Authorization": "secret"}
        assert record["recipe"]["extract"]["keywords"]["headers"] == "<not recorded>"
        assert list(record["inputs"]) == [(tmp_path / "stops.geojson").as_posix()]



def test_run_layers(tmp_path, monkeypatch):
    """Each layer is written as GeoParquet as read directly, an empty read as an empty layer;
    a run whose second layer fails leaves the previous outputs and record as they were."""
    recipe = _recipe(
        tmp_path,
        f"""
        recipe: pyrosm
        pbf: {{file: '{PBF}'}}
        layers:
          walk: {{read: network, network_type: walking}}
          buildings: {{read: buildings}}
          none: {{read: custom, custom_filter: {{amenity: [no_such_value]}}}}
        outputs:
          layers: {{walk: walk.parquet, buildings: b/buildings.parquet, none: none.parquet}}
        """,
    )
    with pytest.warns(UserWarning, match="Could not find any OSM data"):
        written = recipes.run(recipe)
    osm = OSM(PBF)
    expected = [osm.get_network(network_type="walking"), osm.get_buildings()]
    for path, frame in zip(written, expected):
        assert list(gpd.read_parquet(path).columns) == list(frame.columns)
        assert len(gpd.read_parquet(path)) == len(frame)
    empty = gpd.read_parquet(written[2])
    assert (len(empty), empty.crs.to_epsg()) == (0, 4326)
    record = json.loads(written[3].read_text(encoding="utf-8"))
    counts = {"walk": len(expected[0]), "buildings": len(expected[1]), "none": 0}
    assert record["layers"] == {name: {"features": n} for name, n in counts.items()}
    assert record["inputs"] == {PBF: extract_index._sha256(PBF)}
    assert sorted(record["outputs"]) == ["b/buildings.parquet", "none.parquet", "walk.parquet"]

    before = {path: path.read_bytes() for path in written}

    def broken(self, *args, **kwargs):
        raise RuntimeError("cannot read buildings")

    monkeypatch.setattr(OSM, "get_buildings", functools.wraps(OSM.get_buildings)(broken))
    with pytest.raises(RuntimeError, match="cannot read buildings"):
        recipes.run(recipe)
    assert {path: path.read_bytes() for path in written} == before



@pytest.mark.parametrize(
    "graph, two_way",
    [
        ("{network: {network_type: walking}, simplify: true}", True),
        ("{network: {network_type: driving}}", False),
        ("{network: {network_type: driving}, force_bidirectional: true}", True),
    ],
)
def test_run_graph(tmp_path, graph, two_way):
    """The graph stage writes the node and edge tables ``graph_tables`` builds: a walking
    graph runs both ways, a driving graph keeps one-way streets one-way unless
    ``force_bidirectional``."""
    text = f"recipe: pyrosm\npbf: {{file: '{PBF}'}}\ngraph: {graph}\n"
    text += "outputs: {graph: {nodes: n.parquet, edges: e.parquet}}\n"
    recipe = _recipe(tmp_path, text)
    nodes_path, edges_path, record_path = recipes.run(recipe)
    resolved = recipes.validate(recipe)["graph"]
    network = resolved["network"]
    read = OSM(PBF).get_network(nodes=True, **network)
    expected = graph_tables(*read, network_type=network["network_type"], **resolved["keywords"])
    nodes, edges = gpd.read_parquet(nodes_path), gpd.read_parquet(edges_path)
    for written, frame in zip((nodes, edges), expected):
        nested = frame.map(lambda value: isinstance(value, (list, dict)))
        plain = frame.columns[~nested.any()]
        assert_geodataframe_equal(written[plain], frame[plain], check_dtype=False)
        for column in frame.columns[nested.any()]:  # written as text
            for text, value in zip(written[column], frame[column]):
                if isinstance(value, list):  # a missing item is written as null
                    value = [item if item == item else None for item in value]
                if isinstance(value, (list, dict)):
                    assert json.loads(text) == value
                elif pd.isna(value):
                    assert pd.isna(text)
                else:
                    assert text == str(value)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["graph"] == {"nodes": len(nodes), "edges": len(edges)}
    pairs = set(zip(edges["u"], edges["v"]))
    assert all((v, u) in pairs for u, v in pairs) == two_way


@pytest.mark.parametrize(
    "case, message",
    [
        ("no network", "graph.network: the PBF holds no such network"),
        ("escape", "resolves outside"),
        ("alias", "'x/p.parquet' and 'link/p.parquet' resolve to one file"),
        ("input", "would overwrite the recipe or one of its inputs"),
        ("bundled", "would overwrite the recipe or one of its inputs"),
        ("changed", "an input file changed while the recipe ran"),
        ("copied", "t.osm.pbf changed while it was copied"),
        ("lock path", "would be written over the lock file"),
        ("lock", "recipe.provenance.json.lock exists (pid 1"),
        ("lock garbled", "recipe.provenance.json.lock exists (unreadable)"),
        ("cache", "must lie apart"),
        ("failure", "no extract"),
        ("record path", None),
    ],
)
def test_run_refuses(tmp_path, offline, monkeypatch, case, message):
    """A run that cannot write its outputs safely is refused, and a failed stage leaves the
    output folder as it was, without a lock or staging folder. An output folder named like
    the record's file is no obstacle."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "recipe.provenance.json").write_text("old")
    area = gpd.GeoDataFrame(geometry=[box(24.9, 60.1, 25.0, 60.2)], crs="EPSG:4326")
    area.to_file(tmp_path / "area.json", driver="GeoJSON")
    extract, outputs = "{area: {bbox: [0, 0, 1, 1]}}", "{extract: a.pbf}"
    if case == "no network":
        outputs = "{extract: a.pbf, graph: {nodes: n.parquet, edges: e.parquet}}\ngraph: {}\n"
        outputs += "reader: {bounding_box: [0, 0, 0.1, 0.1]}"
    elif case in ("escape", "alias"):
        (out / "x").mkdir()
        try:
            link = tmp_path if case == "escape" else out / "x"
            os.symlink(link, out / "link", target_is_directory=True)
        except OSError:
            pytest.skip("symbolic links cannot be made here")
        outputs = "{extract: link/a.pbf}"
        if case == "alias":
            outputs = "{extract: a.pbf, layers: {p: x/p.parquet, q: link/p.parquet}}\n"
            outputs += "layers: {p: {read: pois}, q: {read: pois}}"
    elif case == "input":
        out = tmp_path
        extract = "{area: {file: area.json}}"
        outputs = "{extract: a.pbf, provenance: area.json}"
    elif case == "bundled":
        out = Path(PBF).parent
        extract, outputs = "{area: {name: test_pbf}}", "{extract: test.osm.pbf}"
    elif case == "changed":
        extract = "{area: {bbox: [0, 0, 1, 1]}, must_cover: {file: area.json}}"
    elif case == "copied":
        extract = "{area: {bbox: [0, 0, 1, 1]}, crop: false}"
    elif case == "lock path":
        outputs = "{extract: .recipe.provenance.json.lock/a.pbf}"
    elif case == "record path":
        outputs = "{extract: r.json/a.pbf, provenance: m/r.json}"
    elif case == "lock":
        (out / ".recipe.provenance.json.lock").write_text("pid 1 on elsewhere")
    elif case == "lock garbled":
        (out / ".recipe.provenance.json.lock").write_bytes(b"\xff")
    elif case == "cache":
        extract = "{area: {bbox: [0, 0, 1, 1]}, directory: out/cache}"
    stub = extract_index.get_data_by_area

    def replaced(*args, **kwargs):
        if case == "failure":
            raise ValueError("no extract")
        if case == "changed":
            with open(tmp_path / "area.json", "a") as f:
                f.write(" ")
        result = stub(*args, **kwargs)
        return dataclasses.replace(result, sha256="0" * 64) if case == "copied" else result

    monkeypatch.setattr(extract_index, "get_data_by_area", functools.wraps(stub)(replaced))
    text = "recipe: pyrosm\nextract: %s\noutputs: %s\n" % (extract, outputs)
    if message is None:
        recipes.run(_recipe(tmp_path, text), out)
        assert (out / "m/r.json").is_file()
        return
    with pytest.raises((ValueError, OSError), match=re.escape(message)):
        recipes.run(_recipe(tmp_path, text), out)
    if case not in ("input", "bundled"):
        left = sorted(p.name for p in out.iterdir() if p.name not in ("link", "x"))
        lock = [".recipe.provenance.json.lock"] if case in ("lock", "lock garbled") else []
        assert left == lock + ["recipe.provenance.json"]
        assert (out / "recipe.provenance.json").read_text() == "old"


def test_command(tmp_path, offline, capsys, monkeypatch):
    """``pyrosm run`` prints what it wrote and ``pyrosm validate`` confirms a recipe, both
    exiting 0; a refused recipe exits 1 with the reason on stderr (also when PyYAML is
    missing), a usage error 2."""
    text = "recipe: pyrosm\nextract: {area: {name: test_pbf}}\noutputs: {extract: a.pbf}\n"
    good = _recipe(tmp_path, text, name="good.yaml")
    bad = _recipe(tmp_path, _GOOD + "colour: red\n", name="bad.yaml")
    argv = ["run", str(good), "-o", str(tmp_path / "out")]
    monkeypatch.setattr(recipes, "_PACKAGES", ("pyrosm", "not-installed"))
    assert recipes.main(argv) == 0
    written = [tmp_path / "out" / name for name in ("a.pbf", "good.provenance.json")]
    assert capsys.readouterr().out.split() == [str(path) for path in written]
    record = json.loads(written[1].read_text(encoding="utf-8"))
    assert record["invocation"]["argv"] == argv
    assert record["versions"] == {"pyrosm": pyrosm.__version__, "not-installed": None}
    assert recipes.main(["validate", str(good)]) == 0
    assert capsys.readouterr().out == "%s: valid pyrosm recipe\n" % good
    assert recipes.main(["validate", str(bad)]) == 1
    assert "unknown key 'colour'" in capsys.readouterr().err

    def unavailable(*args):
        raise ExtractDownloadError("no extract could be downloaded")

    monkeypatch.setattr(recipes, "_run", unavailable)
    assert recipes.main(argv) == 1
    assert capsys.readouterr().err == "pyrosm: no extract could be downloaded\n"
    with pytest.raises(SystemExit) as exit_info:
        recipes.main(["nonsense"])
    assert exit_info.value.code == 2
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert recipes.main(["validate", str(good)]) == 1
    assert 'pip install "pyrosm[recipes]"' in capsys.readouterr().err
