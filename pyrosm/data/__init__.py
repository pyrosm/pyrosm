import os
import tempfile
from pyrosm.utils.download import download

__all__ = ["available", "get_path"]
_module_path = os.path.dirname(__file__)

_package_files = {"test_pbf": "test.osm.pbf",
                  "helsinki_pbf": "Helsinki.osm.pbf"
                  }

# Larger files are fetched from remote url
# Can be used to specify e.g. Geofabrik PBF download links
_temp_files = {"helsinki_region_pbf":
                   {"name": "Helsinki_region.osm.pbf",
                    "url": "https://gist.github.com/HTenkanen/"
                           "02dcfce32d447e65024d93d39ddb1812/"
                           "raw/5fe7ffb625f091591d8c29128a9e3b37870a5012/"
                           "Helsinki_region.osm.pbf"}
               }

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
        # For large datasets, download and fetch from temp
        return download(url=_temp_files[dataset]["url"],
                        filename=_temp_files[dataset]["name"])
    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(available))
        raise ValueError(msg)
