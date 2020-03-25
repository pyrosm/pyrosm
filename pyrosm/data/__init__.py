import os
import tempfile

__all__ = ["available", "get_path"]

_module_path = os.path.dirname(__file__)
_temp_path = os.path.join(tempfile.gettempdir(), "pyrosm")

_package_files = {"test_pbf": "test.osm.pbf",
                  }

_temp_files = {"helsinki_pbf": "Helsinki.osm.pbf"}

available = list(_package_files.keys()) + list(_temp_files.keys())


def get_path(dataset):
    """
    Get the path to the data file.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``pyrosm.data.available`` for
        all options.
    """
    if dataset in _package_files:
        return os.path.abspath(os.path.join(_module_path, _package_files[dataset]))
    elif dataset in _temp_files:
        return os.path.join(_temp_path, _temp_files[dataset])
    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(available))
        raise ValueError(msg)
