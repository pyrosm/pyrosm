from importlib.metadata import version, PackageNotFoundError

from pyrosm.data import get_data, get_path  # drop get_path in the future
from pyrosm.pyrosm import OSM

try:
    __version__ = version("pyrosm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "unknown"
