import os
from pyrosm.utils.download import download
from pyrosm.data.geofabrik import Africa, Antarctica, Asia, AustraliaOceania, \
    Europe, NorthAmerica, SouthAmerica, CentralAmerica, Brazil, Canada, France, \
    Germany, GreatBritain, Italy, Japan, Netherlands, Poland, Russia, USA
from pyrosm.data.bbbike import Cities
import warnings


class DataSources:
    def __init__(self):
        self.africa = Africa()
        self.antarctica = Antarctica()
        self.asia = Asia()
        self.australia_oceania = AustraliaOceania()
        self.europe = Europe()
        self.north_america = NorthAmerica()
        self.south_america = SouthAmerica()
        self.central_america = CentralAmerica()

        self.brazil = Brazil()
        self.canada = Canada()
        self.france = France()
        self.germany = Germany()
        self.great_britain = GreatBritain()
        self.italy = Italy()
        self.japan = Japan()
        self.netherlands = Netherlands()
        self.poland = Poland()
        self.russia = Russia()
        self.usa = USA()

        self.cities = Cities()

        self.available = {
            "africa": self.africa.available,
            "antarctica": self.antarctica.available,
            "asia": self.asia.available,
            "australia_oceania": self.australia_oceania.available,
            "central_america": self.central_america.available,
            "europe": self.europe.available,
            "north_america": self.north_america.available,
            "south_america": self.south_america.available,

            "brazil": self.brazil.available,
            "canada": self.canada.available,
            "france": self.france.available,
            "germany": self.germany.available,
            "great_britain": self.great_britain.available,
            "italy": self.italy.available,
            "japan": self.japan.available,
            "netherlands": self.netherlands.available,
            "poland": self.poland.available,
            "russia": self.russia.available,
            "usa": self.usa.available,

            "cities": self.cities.available,
        }


# Initialize DataSources
sources = DataSources()

__all__ = ["available", "get_data", "get_path"]
_module_path = os.path.dirname(__file__)
_package_files = {"test_pbf": "test.osm.pbf", "helsinki_pbf": "Helsinki.osm.pbf"}

available = {
    "test_data": list(_package_files.keys()) + ["helsinki_region_pbf"],
    "regions": {k: v for k, v in sources.available.items() if k != "cities"},
    "cities": sources.cities.available,
}

_helsinki_region_pbf = {"name": "Helsinki_region.osm.pbf",
                        "url": "https://gist.github.com/HTenkanen/"
                               "02dcfce32d447e65024d93d39ddb1812/"
                               "raw/5fe7ffb625f091591d8c29128a9e3b37870a5012/"
                               "Helsinki_region.osm.pbf"}


def retrieve(data, update, directory):
    return download(url=data["url"],
                    filename=data["name"],
                    update=update,
                    target_dir=directory
                    )


def get_data(dataset, update=False, directory=None):
    """
    Get the path to a PBF data file, and download the data if needed.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``pyrosm.data.available`` for
        all options.

    update : bool
        Whether the PBF file should be downloaded/updated if the dataset
        with the same name exists in the temp.

    directory : str (optional)
        Path to a directory where the PBF data will be downloaded.
        (does not apply for test data sets bundled with the package).
    """
    all_sources = []
    for source, available in sources.available.items():
        all_sources += available
    all_sources = [src.lower() for src in all_sources]

    # Static test data for Helsinki Region
    # that should be able to download
    all_sources += ["helsinki_region_pbf"]

    if not isinstance(dataset, str):
        raise ValueError(f"'dataset' should be text. Got {dataset}.")
    dataset = dataset.lower().strip()

    if dataset in _package_files:
        return os.path.abspath(os.path.join(_module_path, _package_files[dataset]))

    elif dataset == "helsinki_region_pbf":
        return retrieve(_helsinki_region_pbf,
                        update, directory)

    elif dataset in all_sources:
        for source, available in sources.available.items():
            # Cities are kept as CamelCase, so need to make lower
            if source == "cities":
                available = [src.lower() for src in available]

            if dataset in available:
                return retrieve(sources.__dict__[source].__dict__[dataset],
                                update, directory)

    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(available))
        raise ValueError(msg)


# Keep temporarily for backward compatibility
def get_path(dataset, update=False, directory=None):
    warnings.warn(
        "'get_path()' is deprecated, use 'get_data()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_data(dataset, update, directory)
