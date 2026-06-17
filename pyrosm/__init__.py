from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyrosm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "unknown"

# `OSM` pulls in geopandas/shapely (~2 s); import it lazily so that lightweight
# entry points (e.g. the multiprocessing workers in pyrosm.pbf_export, which only
# need protobuf + numpy) do not pay that cost when importing a pyrosm submodule.
__all__ = [
    "OSM",
    "generate_tiles",
    "geocode",
    "get_data",
    "get_data_by_bbox",
    "get_data_by_geocoding",
    "get_path",
    "read_tiled",
]


def __getattr__(name):
    if name == "OSM":
        from pyrosm.pyrosm import OSM

        return OSM
    if name in ("get_data", "get_path"):  # drop get_path in the future
        from pyrosm.data import get_data, get_path

        return get_data if name == "get_data" else get_path
    if name == "get_data_by_bbox":
        from pyrosm.data import get_data_by_bbox

        return get_data_by_bbox
    if name in ("geocode", "get_data_by_geocoding"):
        from pyrosm.data import geocode, get_data_by_geocoding

        return geocode if name == "geocode" else get_data_by_geocoding
    if name in ("read_tiled", "generate_tiles"):
        from pyrosm.tiling import read_tiled, generate_tiles

        return read_tiled if name == "read_tiled" else generate_tiles
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
