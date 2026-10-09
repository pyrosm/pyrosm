"""Recipes: a YAML file that says how the OSM data of an analysis is obtained (``extract``),
read into layers (``layers``) and turned into a graph (``graph``). pyrosm checks a recipe with
:func:`validate`; the file is the reproducible method."""

import inspect
import math
import os
import pathlib
import re
import struct
import unicodedata

_VERSIONS = (1,)
_TOP_KEYS = tuple("recipe version extract pbf reader layers graph outputs".split())
_AREA_KINDS = ("bbox", "file", "place", "name")
_VECTOR_SUFFIXES = (".gpkg", ".geojson", ".json", ".parquet")
# The read methods a layer can name, as OSM methods.
_READS = {
    n: "get_" + n for n in "network buildings pois landuse natural boundaries".split()
}
_READS["custom"] = "get_data_by_custom_criteria"
# Names that fail on Windows although POSIX accepts them.
_UNPORTABLE = re.compile(r'[<>:"|?*\x00-\x1f]')
_WINDOWS_RESERVED = re.compile(
    r"^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(\..*)?$", re.I
)


def _load_yaml(path):
    """The recipe file as a mapping, refusing duplicate and non-scalar keys (PyYAML's own loader
    keeps the last of two equal keys)."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            'Reading a recipe needs PyYAML: pip install "pyrosm[recipes]".'
        ) from error

    class _StrictLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise ValueError(
                    "%s: mapping keys must be scalars, not %r" % (path, key)
                )
            if key in result:
                raise ValueError("%s: duplicate key %r" % (path, key))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        document = yaml.load(
            pathlib.Path(path).read_text(encoding="utf-8"), _StrictLoader
        )
    except yaml.YAMLError as error:
        raise ValueError("%s: not valid YAML: %s" % (path, error)) from None
    if not isinstance(document, dict):
        raise ValueError("%s: a recipe must be a YAML mapping" % path)
    return document


def _resolve(path, where):
    """``path`` made absolute with symbolic links followed; a link loop (which raises
    ``RuntimeError`` on Python before 3.13) is refused by name."""
    try:
        return pathlib.Path(path).resolve()
    except (RuntimeError, OSError) as error:
        raise ValueError(
            "%s: %s cannot be resolved: %s" % (where, path, error)
        ) from None


def _mapping(value, where):
    if not isinstance(value, dict):
        raise ValueError("%s: must be a mapping" % where)
    return value


def _reject_foreign(mapping, allowed, where):
    for key in mapping:
        if key not in allowed:
            raise ValueError(
                "%s: unknown key %r (one of %s)" % (where, key, ", ".join(allowed))
            )


def _json_safe(value, where, seen=None):
    """Refuse a value a provenance record could not hold as JSON. A list or mapping that occurs
    twice (a YAML alias, possibly of itself) is refused too, so a value can neither loop nor
    multiply when written out; ``seen`` holds the ids of the lists and mappings met so far.
    """
    seen = set() if seen is None else seen
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            raise ValueError(
                "%s: repeats a list or mapping through a YAML alias; write it out"
                % where
            )
        seen.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("%s: keys must be text, not %r" % (where, key))
            _json_safe(item, "%s.%s" % (where, key), seen)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _json_safe(item, "%s[%d]" % (where, i), seen)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s: must be a finite number" % where)
    elif not (value is None or isinstance(value, (str, int, bool))):
        raise ValueError(
            "%s: %r is not a text, number, true/false, list or mapping value"
            % (where, value)
        )


def _keywords(given, function, fixed, where, extra=()):
    """The keywords in ``given`` checked against the named parameters of ``function``: a
    keyword in ``fixed`` is set by the recipe, ``extra`` are the section's own keys."""
    names = [
        name
        for name, param in inspect.signature(function).parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        and name not in ("self",) + tuple(fixed)
    ]
    keywords = {}
    for key, value in given.items():
        if key in extra:
            continue
        if key in fixed:
            raise ValueError("%s.%s: set by the recipe, not a keyword" % (where, key))
        if key not in names:
            raise ValueError(
                "%s: unknown keyword %r (one of %s)"
                % (where, key, ", ".join(sorted(names + list(extra))))
            )
        _json_safe(value, "%s.%s" % (where, key))
        keywords[key] = value
    return keywords


def _pbf_first_blob(path):
    """The type of the first blob of ``path`` (``"OSMHeader"`` for a PBF), or ``None`` when the
    file does not start with a whole blob header followed by the blob it announces."""
    from pyrosm.proto.fileformat_pb2 import BlobHeader

    with open(path, "rb") as f:
        size = f.read(4)
        if len(size) < 4:
            return None
        (length,) = struct.unpack("!L", size)
        if not 0 < length <= 64 * 1024:
            return None
        data = f.read(length)
        remaining = os.fstat(f.fileno()).st_size - 4 - len(data)
    header = BlobHeader()
    try:
        header.ParseFromString(data)
    except Exception:
        return None
    if len(data) != length or not 0 < header.datasize <= remaining:
        return None
    return header.type


def _file_source(value, where, recipe_dir, suffixes, pbf=False):
    """A file input ``{file: path, layer: name}``, its path resolved against the
    recipe's folder and checked to exist, to be non-empty and of an accepted type."""
    _mapping(value, where)
    allowed = ("file",) if pbf else ("file", "layer")
    _reject_foreign(value, allowed, where)
    name = value.get("file")
    if not isinstance(name, str) or not name:
        raise ValueError("%s.file: must be a file path" % where)
    path = _resolve(recipe_dir / name, where + ".file")
    if not path.name.lower().endswith(suffixes):
        raise ValueError(
            "%s.file: %r must end with one of %s" % (where, name, ", ".join(suffixes))
        )
    if not path.is_file():
        raise ValueError("%s.file: %r does not exist" % (where, name))
    if path.stat().st_size == 0:
        raise ValueError("%s.file: %r is empty" % (where, name))
    if pbf and _pbf_first_blob(path) != "OSMHeader":
        raise ValueError(
            "%s.file: %r does not start with an OSM PBF header" % (where, name)
        )
    source = {"file": path}
    if "layer" in value:
        if not isinstance(value["layer"], str) or not value["layer"]:
            raise ValueError("%s.layer: must be a layer name" % where)
        source["layer"] = value["layer"]
    return source


def _area(value, where, recipe_dir):
    _mapping(value, where)
    kinds = [kind for kind in _AREA_KINDS if kind in value]
    if len(kinds) != 1:
        raise ValueError(
            "%s: needs exactly one of %s" % (where, ", ".join(_AREA_KINDS))
        )
    kind = kinds[0]
    if kind == "file":
        return {"file": _file_source(value, where, recipe_dir, _VECTOR_SUFFIXES)}
    _reject_foreign(value, (kind,), where)
    item = value[kind]
    if kind == "bbox":
        if (
            not isinstance(item, list)
            or len(item) != 4
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in item
            )
        ):
            raise ValueError(
                "%s.bbox: must be four numbers [minx, miny, maxx, maxy]" % where
            )
        if not (item[0] < item[2] and item[1] < item[3]):
            raise ValueError(
                "%s.bbox: needs minx < maxx and miny < maxy, got %r" % (where, item)
            )
        return {"bbox": [float(v) for v in item]}
    if not isinstance(item, str) or not item.strip():
        raise ValueError("%s.%s: must be a non-empty text" % (where, kind))
    return {kind: item}


def _extract(section, recipe_dir):
    from pyrosm.data import get_data
    from pyrosm.data.extract_index import get_data_by_area

    _mapping(section, "extract")
    if "area" not in section:
        raise ValueError("extract: needs an area")
    area = _area(section["area"], "extract.area", recipe_dir)
    if "name" in area:
        function, fixed = get_data, ("dataset", "opener")
    else:
        function, fixed = get_data_by_area, ("area", "output_path", "opener")
    keywords = _keywords(section, function, fixed, "extract", extra=("area",))
    if "must_cover" in keywords:
        keywords["must_cover"] = _file_source(
            keywords["must_cover"], "extract.must_cover", recipe_dir, _VECTOR_SUFFIXES
        )
    return {"area": area, "call": function.__name__, "keywords": keywords}


def _layers(section):
    from pyrosm import OSM

    _mapping(section, "layers")
    if not section:
        raise ValueError("layers: needs at least one layer")
    layers = {}
    for name, layer in section.items():
        if not isinstance(name, str) or not name:
            raise ValueError("layers: layer names must be text, not %r" % (name,))
        where = "layers.%s" % name
        _mapping(layer, where)
        read = layer.get("read")
        if not isinstance(read, str) or read not in _READS:
            raise ValueError(
                "%s.read: must be one of %s, got %r" % (where, ", ".join(_READS), read)
            )
        method = getattr(OSM, _READS[read])
        fixed = ("nodes",) if read == "network" else ()
        keywords = _keywords(layer, method, fixed, where, extra=("read",))
        layers[name] = {"read": read, "keywords": keywords}
    return layers


def _graph(section):
    from pyrosm import OSM

    _mapping(section, "graph")
    _reject_foreign(section, ("network",), "graph")
    network = _mapping(section.get("network", {}), "graph.network")
    return {
        "network": _keywords(network, OSM.get_network, ("nodes",), "graph.network"),
    }


def _output_path(value, where, suffix):
    """A relative, portable output path ending with ``suffix``."""
    if not isinstance(value, str) or not value:
        raise ValueError("%s: must be a file path" % where)
    posix = pathlib.PurePosixPath(value)
    windows = pathlib.PureWindowsPath(value)
    if (
        "\\" in value
        or windows.drive
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or posix.name in ("", ".")
        or value.endswith("/")
    ):
        raise ValueError(
            "%s: %r must be a relative file path inside the output directory (no absolute "
            "paths or '..')" % (where, value)
        )
    if not posix.name.endswith(suffix):
        raise ValueError("%s: %r must end with %s" % (where, value, suffix))
    for part in posix.parts:
        if (
            _UNPORTABLE.search(part)
            or part.endswith((" ", "."))
            or _WINDOWS_RESERVED.match(part)
            or len(part.encode("utf-8")) > 255
        ):
            raise ValueError(
                "%s: %r is not a portable file name (a character or name Windows refuses, a "
                "trailing dot or space, or over 255 bytes)" % (where, part)
            )
    return posix.as_posix()


def _outputs(section, stem, extract, layers, graph):
    _mapping(section, "outputs")
    _reject_foreign(section, ("extract", "layers", "graph", "provenance"), "outputs")
    outputs = {"extract": None, "layers": {}, "graph": None}
    if extract is not None:
        if "extract" not in section:
            raise ValueError(
                "outputs: needs 'extract', the PBF the extract stage writes"
            )
        outputs["extract"] = _output_path(section["extract"], "outputs.extract", ".pbf")
    elif "extract" in section:
        raise ValueError("outputs.extract: the recipe has no extract stage")
    given = _mapping(section.get("layers", {}), "outputs.layers")
    for name in layers:
        if name not in given:
            raise ValueError("outputs.layers: needs a file for layer %r" % name)
    for name, value in given.items():
        if name not in layers:
            raise ValueError(
                "outputs.layers.%s: the recipe has no layer %r" % (name, name)
            )
        outputs["layers"][name] = _output_path(
            value, "outputs.layers.%s" % name, ".parquet"
        )
    if graph is not None:
        files = _mapping(section.get("graph"), "outputs.graph")
        _reject_foreign(files, ("nodes", "edges"), "outputs.graph")
        outputs["graph"] = {
            key: _output_path(files.get(key), "outputs.graph.%s" % key, ".parquet")
            for key in ("nodes", "edges")
        }
    elif "graph" in section:
        raise ValueError("outputs.graph: the recipe has no graph stage")
    outputs["provenance"] = _output_path(
        section.get("provenance", "%s.provenance.json" % stem),
        "outputs.provenance",
        ".json",
    )
    paths = [outputs["extract"], outputs["provenance"], *outputs["layers"].values()]
    paths += list((outputs["graph"] or {}).values())
    seen = {}
    for path in filter(None, paths):
        key = unicodedata.normalize("NFC", path.casefold())
        parts = pathlib.PurePosixPath(key).parts
        for other, other_parts in seen.items():
            if parts == other_parts:
                raise ValueError("outputs: %r is named twice" % path)
            shorter, longer = sorted((parts, other_parts), key=len)
            if longer[: len(shorter)] == shorter:
                raise ValueError(
                    "outputs: %r and %r cannot both be written, as one is inside the other"
                    % (other, path)
                )
        seen[path] = parts
    return outputs


def validate(path):
    """Check a recipe file without running it and return it resolved.

    Every section is checked eagerly: the keys, the area, the input files (they must exist; a
    PBF must start with an OSM header), every keyword against the function it is
    passed to, and the output paths, which must be relative, portable and named once. Raises
    ``ValueError`` naming the key that is wrong, or ``ImportError`` when PyYAML is missing.

    Parameters
    ----------
    path : str | pathlib.Path
        The recipe YAML file.

    Returns
    -------
    dict
        The resolved recipe: ``recipe_path``, ``recipe_dir``, ``version``, ``extract``,
        ``pbf``, ``reader``, ``layers``, ``graph`` and ``outputs``.
    """
    from pyrosm import OSM

    recipe_path = _resolve(path, "recipe")
    document = _load_yaml(recipe_path)
    _reject_foreign(document, _TOP_KEYS, "recipe")
    if document.get("recipe") != "pyrosm":
        raise ValueError("recipe: must be 'pyrosm', got %r" % document.get("recipe"))
    version = document.get("version", 1)
    if type(version) is not int or version not in _VERSIONS:
        raise ValueError("version: %r is not a known recipe version (1)" % (version,))
    recipe_dir = recipe_path.parent
    extract = pbf = graph = None
    layers = {}
    if "extract" in document:
        extract = _extract(document["extract"], recipe_dir)
    if "pbf" in document:
        if extract is not None:
            raise ValueError("pbf: a recipe has either an extract stage or a pbf input")
        pbf = _file_source(document["pbf"], "pbf", recipe_dir, (".pbf",), pbf=True)
    if "layers" in document:
        layers = _layers(document["layers"])
    if "graph" in document:
        graph = _graph(document["graph"])
    if extract is None and not layers and graph is None:
        raise ValueError("recipe: needs at least one of extract, layers, graph")
    if (layers or graph is not None) and extract is None and pbf is None:
        raise ValueError(
            "recipe: layers and graph need an extract stage or a pbf input"
        )
    reader = {}
    if "reader" in document:
        if not layers and graph is None:
            raise ValueError("reader: the recipe has no layers or graph to read")
        reader = _keywords(
            _mapping(document["reader"], "reader"),
            OSM.__init__,
            ("filepath",),
            "reader",
        )
    if "outputs" not in document:
        raise ValueError("recipe: needs an outputs section")
    outputs = _outputs(document["outputs"], recipe_path.stem, extract, layers, graph)
    return {
        "recipe_path": recipe_path,
        "recipe_dir": recipe_dir,
        "version": version,
        "extract": extract,
        "pbf": pbf,
        "reader": reader,
        "layers": layers,
        "graph": graph,
        "outputs": outputs,
    }


def main(argv=None):
    """The ``pyrosm`` command: ``pyrosm validate RECIPE`` checks a recipe.

    Exits 0 when the recipe is valid, 1 with the reason on stderr when it is not, and 2 on a
    usage error. It never prompts.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="pyrosm", description="Check a pyrosm recipe."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("validate", help="check a recipe without running it")
    checker.add_argument("recipe", help="the recipe YAML file")
    arguments = parser.parse_args(argv)
    try:
        validate(arguments.recipe)
    except (ValueError, OSError, ImportError) as error:
        print("pyrosm: %s" % error, file=sys.stderr)
        return 1
    print("%s: valid pyrosm recipe" % arguments.recipe)
    return 0
