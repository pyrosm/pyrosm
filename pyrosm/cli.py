import argparse
import json
import sys
from pathlib import Path


def _parser():
    """The parser of the ``pyrosm`` command's arguments (see :func:`main`)."""
    parser = argparse.ArgumentParser(
        prog="pyrosm",
        description="Run or check a pyrosm recipe, find the extracts of an area, describe "
        "a PBF file, or list and remove the downloaded files.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run a recipe and write its outputs")
    runner.add_argument("recipe", help="the recipe YAML file")
    runner.add_argument(
        "-o", "--out", help="the output folder (default: the recipe's folder)"
    )
    runner.add_argument(
        "--force",
        action="store_true",
        help="run every stage, changed or not, and download the extract again",
    )
    checker = commands.add_parser("validate", help="check a recipe without running it")
    checker.add_argument("recipe", help="the recipe YAML file")
    finder = commands.add_parser(
        "find",
        help="list the extracts of an area, best download first, downloading none",
    )
    # Optional here, so that a bounding box starting with "-" can come in as an unknown
    # option (see main).
    finder.add_argument(
        "area",
        nargs="?",
        help="a bounding box 'minx,miny,maxx,maxy' in lon/lat, a vector file, or a place "
        "name to geocode",
    )
    finder.add_argument(
        "--contains-only",
        action="store_true",
        help="list only the extracts that contain the whole area",
    )
    finder.add_argument("--json", action="store_true", help="print JSON")
    describer = commands.add_parser(
        "info", help="print a PBF file's bounding box, snapshot time and SHA-256"
    )
    describer.add_argument("file", help="the PBF file")
    describer.add_argument("--json", action="store_true", help="print JSON")
    downloads = commands.add_parser(
        "downloads", help="list or remove the files downloaded into pyrosm's folder"
    )
    actions = downloads.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="print the path of each downloaded PBF file")
    remover = actions.add_parser(
        "clear", help="remove the downloaded PBF files, or the one named"
    )
    remover.add_argument("file", nargs="?", help="a file name or path (default: all)")
    return parser


def _bbox(text):
    """``text`` as a bounding box, four numbers separated by commas, or None."""
    numbers = text.split(",")
    if len(numbers) == 4:
        try:
            return [float(v) for v in numbers]
        except ValueError:
            pass
    return None


def _parse_area(text):
    """The area that ``pyrosm find`` was given: a bounding box (see :func:`_bbox`), the
    geometries of a vector file, or the place a name geocodes to."""
    import geopandas as gpd

    from pyrosm.data.geocoding import geocode

    bbox = _bbox(text)
    if bbox is not None:
        return bbox
    if Path(text).is_file():
        try:
            return gpd.read_file(text)
        except Exception as e:
            raise ValueError("Could not read '%s' as a vector file: %s" % (text, e))
    return geocode(text)


def _print_extracts(area, contains_only, as_json):
    """Print the extracts of ``area`` (see :func:`pyrosm.find_extracts`), as a table or
    JSON."""
    import pandas as pd

    from pyrosm.data.extract_index import find_extracts

    found = find_extracts(_parse_area(area), contains_only=contains_only)
    if as_json:
        records = [
            {
                "provider": row.provider,
                "id": row.id,
                "name": row.name,
                "url": row.url,
                "bytes": None if pd.isna(row.bytes) else int(row.bytes),
                "contains": bool(row.contains),
                "bounds": list(row.geometry.bounds),
            }
            for row in found.itertuples()
        ]
        print(json.dumps(records, indent=1))
        return
    table = pd.DataFrame(
        {
            "provider": found["provider"],
            "name": found["name"],
            "MB": [None if pd.isna(b) else round(b / 1e6, 1) for b in found["bytes"]],
            "contains": found["contains"].map({True: "yes", False: "no"}),
            "url": found["url"],
        }
    )
    print(table.to_string(index=False, na_rep="unknown") if len(table) else "none")


def _print_file_info(file, as_json):
    """Print the bounding box, snapshot time and SHA-256 of the PBF file ``file``, as lines
    or JSON."""
    from pyrosm.data.extract_index import _provenance
    from pyrosm.utils import get_bounding_box

    bbox = get_bounding_box(file)
    sha256, snapshot = _provenance(file)
    info = {
        "file": str(Path(file).resolve()),
        "bytes": Path(file).stat().st_size,
        "bounding_box": None if bbox is None else list(bbox.bounds),
        "snapshot": None if snapshot is None else snapshot.isoformat(),
        "sha256": sha256,
    }
    if as_json:
        print(json.dumps(info, indent=1))
        return
    for key, value in info.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        print("%s: %s" % (key.replace("_", " "), "none" if value is None else value))


def _list_or_clear_downloads(action, file):
    """List the downloaded PBF files, or remove them or the one named ``file`` (see
    :func:`pyrosm.utils.download.clear_downloads`)."""
    from pyrosm.utils.download import clear_downloads, list_downloads

    if action == "list":
        for path in list_downloads():
            print(path)
    else:
        removed = clear_downloads(file)
        print("removed %d file%s" % (removed, "" if removed == 1 else "s"))


def main(argv=None):
    """The ``pyrosm`` command:

    - ``pyrosm run RECIPE [-o DIR] [--force]`` runs a recipe and prints each output path,
      after ``written`` or ``reused``; ``pyrosm validate RECIPE`` checks one.
    - ``pyrosm find AREA [--contains-only] [--json]`` lists the extracts of an area, best
      download first, without downloading (see :func:`pyrosm.find_extracts`). ``AREA`` is a
      bounding box ``minx,miny,maxx,maxy``, a vector file or a place name to geocode.
    - ``pyrosm info FILE [--json]`` prints a PBF file's bounding box, snapshot time and
      SHA-256.
    - ``pyrosm downloads list`` prints the PBF files downloaded into pyrosm's default
      folder; ``pyrosm downloads clear [FILE]`` removes them, or the one named.

    Exits 0 on success, 1 with the reason on stderr when the input, the data or a request
    is refused, and 2 on a usage error. It never prompts.
    """
    from pyrosm.exceptions import ExtractDownloadError, PBFException
    from pyrosm.recipes import _run_recipe, validate

    parser = _parser()
    arguments, unknown = parser.parse_known_args(argv)
    if arguments.command == "find" and arguments.area is None:
        # argparse takes a bounding box that starts with "-" for an option.
        if len(unknown) == 1 and _bbox(unknown[0]) is not None:
            arguments.area, unknown = unknown[0], []
        else:
            parser.error("pyrosm find needs an AREA")
    if unknown:
        parser.error("unrecognized arguments: %s" % " ".join(unknown))
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        if arguments.command == "run":
            invocation = {"entry_point": "cli", "argv": argv}
            done = _run_recipe(
                arguments.recipe, arguments.out, invocation, arguments.force
            )
            for path, status in done.items():
                print(status, path)
        elif arguments.command == "validate":
            validate(arguments.recipe)
            print("%s: valid pyrosm recipe" % arguments.recipe)
        elif arguments.command == "find":
            _print_extracts(arguments.area, arguments.contains_only, arguments.json)
        elif arguments.command == "info":
            _print_file_info(arguments.file, arguments.json)
        else:
            _list_or_clear_downloads(arguments.action, getattr(arguments, "file", None))
    except (
        ValueError,
        OSError,
        ImportError,
        ExtractDownloadError,
        PBFException,
    ) as error:
        print("pyrosm: %s" % error, file=sys.stderr)
        return 1
    return 0
