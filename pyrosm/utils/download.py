import urllib.request
import tempfile
import os
import enum
import shutil
import ssl
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
    size = os.path.getsize(file_name)
    return round(convert_unit(size, size_type), 2)


def download(url, filename, update, target_dir):
    if target_dir is None:
        temp_dir = tempfile.gettempdir()
        target_dir = os.path.join(temp_dir, "pyrosm")
    else:
        if not os.path.isdir(target_dir):
            raise ValueError(f"The provided directory does not exist: " f"{target_dir}")

    filepath = os.path.abspath(os.path.join(target_dir, os.path.basename(filename)))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if file exists
    file_exists = False
    if os.path.exists(filepath):
        file_exists = True

    if update and file_exists:
        os.remove(filepath)

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
            f"Downloaded Protobuf data '{os.path.basename(filepath)}' "
            f"({filesize} MB) to:\n'{filepath}'"
        )
    return filepath
