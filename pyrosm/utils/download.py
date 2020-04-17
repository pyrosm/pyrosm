import urllib
import tempfile
import os


def download(url, filename):
    temp_dir = tempfile.gettempdir()
    target_dir = os.path.join(temp_dir, 'pyrosm')
    target_file = os.path.join(target_dir, os.path.basename(filename))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(target_file):
        os.remove(target_file)

    # Download data to temp
    filepath, msg = urllib.request.urlretrieve(url, target_file)
    print(f"Downloaded Protobuf data '{os.path.basename(filepath)}' to TEMP:\n'{filepath}'")
    return filepath
