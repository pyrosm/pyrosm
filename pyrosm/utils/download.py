import urllib.request
import tempfile
import enum
import shutil
import ssl
from pathlib import Path
from urllib.error import HTTPError

import certifi


class UNIT(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4


def convert_unit(size_in_bytes, unit):
    if unit == UNIT.KB:
        return size_in_bytes / 1024
    elif unit == UNIT.MB:
        return size_in_bytes / (1024 * 1024)
    elif unit == UNIT.GB:
        return size_in_bytes / (1024 * 1024 * 1024)
    else:
        return size_in_bytes


def get_file_size(file_name, size_type=UNIT.MB):
    size = Path(file_name).stat().st_size
    return round(convert_unit(size, size_type), 2)


def download_dir():
    """The default directory ``get_data`` downloads PBF extracts into: ``<tempdir>/pyrosm``."""
    return Path(tempfile.gettempdir()) / "pyrosm"


def list_downloads():
    """List the PBF files downloaded by ``get_data`` in the default download directory
    (``<tempdir>/pyrosm``). Returns a sorted list of string paths."""
    directory = download_dir()
    if not directory.is_dir():
        return []
    return sorted(str(p) for p in directory.glob("*.pbf") if p.is_file())


def clear_downloads(filepath=None):
    """Remove PBF files downloaded by ``get_data`` from the default download directory
    (``<tempdir>/pyrosm``). With no ``filepath`` every downloaded ``*.pbf`` there is removed; with
    a ``filepath`` (a path or a bare filename) only that file is removed. The out-of-core ``cache``
    subdirectory and the bundled package data are left untouched. Returns the number of files
    removed."""
    directory = download_dir()
    if not directory.is_dir():
        return 0
    if filepath is None:
        targets = sorted(p for p in directory.glob("*.pbf") if p.is_file())
    else:
        target = directory / Path(filepath).name
        targets = [target] if target.is_file() else []
    removed = 0
    for path in targets:
        path.unlink()
        removed += 1
    return removed


def download(url, filename, update, target_dir):
    if target_dir is None:
        target_dir = download_dir()
    else:
        target_dir = Path(target_dir)
        if not target_dir.is_dir():
            raise ValueError(f"The provided directory does not exist: " f"{target_dir}")

    filepath = (target_dir / Path(filename).name).resolve()

    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    # Check if file exists
    file_exists = False
    if filepath.exists():
        file_exists = True

    if update and file_exists:
        filepath.unlink()

    # Download data to temp if it does not exist or if update is requested
    if update or file_exists is False:
        try:
            # Build the HTTPS context from certifi's CA bundle instead of the OS
            # trust store. On Windows, loading the system certificate store can
            # raise ssl.SSLError [ASN1: NOT_ENOUGH_DATA] (a CPython bug triggered
            # by a malformed entry in the store); certifi avoids it and works the
            # same across platforms.
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=context) as response, open(
                filepath, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
        except HTTPError:
            raise ValueError(
                f"PBF-file '{url}' is temporarily unavailable. " f"Try again later."
            )
        except Exception as e:
            raise e

        filesize = get_file_size(filepath)
        if filesize == 0:
            raise ValueError(
                f"PBF-file '{filename}' from the provider was empty. "
                "This is likely a temporary issue, try again later."
            )
        print(
            f"Downloaded Protobuf data '{filepath.name}' "
            f"({filesize} MB) to:\n'{filepath}'"
        )
    return str(filepath)
