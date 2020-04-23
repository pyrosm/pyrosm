import urllib
import tempfile
import os
import enum



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
    return round(convert_unit(size, size_type), 1)


def download(url, filename, update):
    temp_dir = tempfile.gettempdir()
    target_dir = os.path.join(temp_dir, 'pyrosm')
    filepath = os.path.join(target_dir, os.path.basename(filename))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if file exists already in temp
    file_exists = False
    if os.path.exists(filepath):
        file_exists = True

    if update and file_exists:
            os.remove(filepath)

    # Download data to temp if it does not exist or if update is requested
    if update or file_exists is False:
        filepath, msg = urllib.request.urlretrieve(url, filepath)
        filesize = get_file_size(filepath)
        print(f"Downloaded Protobuf data '{os.path.basename(filepath)}' "
              f"({filesize} MB) to TEMP:\n'{filepath}'")
    return filepath
