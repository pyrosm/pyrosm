import os
import struct
import sys
import textwrap
from pathlib import Path

import pytest

from pyrosm import get_data, recipes
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
        reader: {{keep_metadata: false}}
        layers:
          walk: {{read: network, network_type: walking, extra_attributes: [lit]}}
          shops: {{read: pois, custom_filter: {{shop: true}}}}
        graph:
          network: {{network_type: driving}}
        outputs:
          layers: {{walk: out/walk.parquet, shops: shops.parquet}}
          graph: {{nodes: nodes.parquet, edges: edges.parquet}}
        """,
        name="study.yaml",
    )
    resolved = recipes.validate(layers)
    assert resolved["pbf"] == {"file": Path(PBF).resolve()}
    assert resolved["reader"] == {"keep_metadata": False}
    assert resolved["layers"]["shops"] == {
        "read": "pois",
        "keywords": {"custom_filter": {"shop": True}},
    }
    assert resolved["graph"]["network"] == {"network_type": "driving"}
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
        ("recipe: pyrosm\nextract: {area: {name: x}}\noutputs: {}\n", "needs 'extract'"),
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
        (_GOOD + "extract: {area: {name: x}}\n", "either an extract stage or a pbf input"),
        (_EXTRACT + "{area: {name: x, place: y}}\n", "needs exactly one of"),
        (_EXTRACT + "{area: {bbox: [1, 2, 0, 3]}}\n", "needs minx < maxx"),
        (_EXTRACT + "{area: {place: ' '}}\n", "must be a non-empty text"),
        (_EXTRACT + "{area: {file: no.gpkg}}\n", "'no.gpkg' does not exist"),
        (_EXTRACT + "{area: {file: a.csv}}\n", "must end with one of"),
        (_GOOD.replace(PBF, "fake.pbf"), "does not start with an OSM PBF header"),
        (_EXTRACT + "{area: {name: x}, strategy: single}\n", "unknown keyword 'strategy'"),
        (_EXTRACT + "{area: {place: x}, output_path: x}\n", "output_path: set by the recipe"),
        (_EXTRACT + "{area: {name: x}, opener: {}}\n", "extract.opener: set by the recipe"),
        (_GOOD.replace("read: network", "read: rivers"), "layers.walk.read: must be one of"),
        (_keyword("nodes: true"), "set by the recipe"),
        (_GOOD + "graph: {simplify: true}\n", "graph: unknown key 'simplify'"),
        (_keyword("timestamp: 2020-01-01"), "is not a text, number"),
        (_GOOD + "reader: {filepath: x.pbf}\n", "reader.filepath: set by the recipe"),
        (_GOOD.replace("walk: walk.parquet", "run: run.parquet"), "needs a file for layer"),
        (_GOOD.replace("walk.parquet", "/tmp/walk.parquet"), "must be a relative file path"),
        (_GOOD.replace("walk.parquet", "../walk.parquet"), "must be a relative file path"),
        (_GOOD.replace("walk.parquet", "walk.csv"), "must end with .parquet"),
        (_GOOD.replace("walk.parquet", "con.parquet"), "not a portable file name"),
        (_TWO.replace("parquet}", "parquet, b: Walk.parquet}"), "is named twice"),
        (_TWO.replace("walk.parquet}", "\u00e9.parquet, b: E\u0301.parquet}"), "named twice"),
        (
            "recipe: pyrosm\nextract: {area: {name: x}}\noutputs: {extract: a.pbf}\n"
            "reader: {engine: in_memory}\n",
            "no layers or graph to read",
        ),
    ],
)
def test_validate_refuses(tmp_path, text, message):
    """Each mistake is refused by name before anything runs."""
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "a.gpkg").write_bytes(b"SQLite format 3\x00")
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


def test_validate_command(tmp_path, capsys, monkeypatch):
    """`pyrosm validate` exits 0 for a valid recipe, 1 with the reason on stderr for an invalid
    one or when PyYAML is missing, and 2 for a usage error."""
    good = _recipe(tmp_path, _GOOD, name="good.yaml")
    bad = _recipe(tmp_path, _GOOD + "colour: red\n", name="bad.yaml")
    assert recipes.main(["validate", str(good)]) == 0
    assert "valid pyrosm recipe" in capsys.readouterr().out
    assert recipes.main(["validate", str(bad)]) == 1
    assert "unknown key 'colour'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        recipes.main(["nonsense"])
    assert exit_info.value.code == 2
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert recipes.main(["validate", str(good)]) == 1
    assert 'pip install "pyrosm[recipes]"' in capsys.readouterr().err
