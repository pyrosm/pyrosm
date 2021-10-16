import os
from pyrosm.utils.download import download
from pyrosm.data.geofabrik import (
    Africa,
    Antarctica,
    Asia,
    AustraliaOceania,
    Europe,
    NorthAmerica,
    SouthAmerica,
    CentralAmerica,
    Brazil,
    Canada,
    France,
    Germany,
    GreatBritain,
    Italy,
    Japan,
    Netherlands,
    Poland,
    Russia,
    USA,
    SubRegions,
)
from pyrosm.data.bbbike import Cities
import warnings

__all__ = ["available", "get_data", "get_path"]
_module_path = os.path.dirname(__file__)
_package_files = {"test_pbf": "test.osm.pbf", "helsinki_pbf": "Helsinki.osm.pbf"}

# Static test data
_helsinki_region_pbf = {
    "name": "Helsinki_region.osm.pbf",
    "url": "https://gist.github.com/HTenkanen/"
    "02dcfce32d447e65024d93d39ddb1812/"
    "raw/5fe7ffb625f091591d8c29128a9e3b37870a5012/"
    "Helsinki_region.osm.pbf",
}

_helsinki_history_pbf = {
    "name": "Helsinki-sample.osh.pbf",
    "url": "https://gist.github.com/HTenkanen/"
    "02dcfce32d447e65024d93d39ddb1812/"
    "raw/885154d451772bef6ac5160027589ddddc97272c/"
    "helsinki-internal.osh.pbf",
}

_helsinki_test_history_pbf = {
    "name": "Helsinki-test.osh.pbf",
    "url": "https://gist.github.com/HTenkanen/"
    "02dcfce32d447e65024d93d39ddb1812/"
    "raw/219f5655ff3ce0a80f84ce424534dfbcdae77792/"
    "helsinki-test.osh.pbf",
}


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

        self.cities = Cities()
        self.subregions = SubRegions()

        self.available = {
            "africa": self.africa.available,
            "antarctica": self.antarctica.available,
            "asia": self.asia.available,
            "australia_oceania": self.australia_oceania.available,
            "central_america": self.central_america.available,
            "europe": self.europe.available,
            "north_america": self.north_america.available,
            "south_america": self.south_america.available,
            "cities": self.cities.available,
            "subregions": self.subregions.available,
        }

        # Gather all data sources
        # Keep hidden to avoid encouraging iteration of the whole
        # world at once which most likely would end up
        # in memory error / filling the disk etc.
        self._all_sources = [
            k for k in self.available.keys() if k not in ["cities", "subregions"]
        ]

        for source, available in self.available.items():
            self._all_sources += available

        for subregion in self.subregions.available:
            self._all_sources += self.subregions.__dict__[subregion].available

        self._all_sources = [src.lower() for src in self._all_sources]

        # Static data for Helsinki Region
        # that should be able to download
        # (needed for tests)
        self._all_sources += [
            "helsinki_region_pbf",
            "helsinki_history_pbf",
            "helsinki_test_history_pbf",
        ]
        self._all_sources = list(set(self._all_sources))


# Initialize DataSources
sources = DataSources()

available = {
    "test_data": list(_package_files.keys())
    + ["helsinki_region_pbf", "helsinki_history_pbf", "helsinki_test_history_pbf"],
    "regions": {
        k: v for k, v in sources.available.items() if k not in ["cities", "subregions"]
    },
    "subregions": sources.subregions.available,
    "cities": sources.cities.available,
}


def retrieve(data, update, directory):
    return download(
        url=data["url"], filename=data["name"], update=update, target_dir=directory
    )


def search_source(name):
    for source, available in sources.available.items():
        # Cities are kept as CamelCase, so need to make lower
        if source == "cities":
            available = [src.lower() for src in available]
        if isinstance(available, list):
            if name in available:
                return sources.__dict__[source].__dict__[name]
        elif isinstance(available, dict):
            # Sub-regions should be looked one level further down
            for subregion, available2 in available.items():
                if name in available2:
                    return sources.subregions.__dict__[subregion].__dict__[name]
    raise ValueError(f"Could not retrieve url for '{name}'.")


def get_data(dataset, update=False, directory=None):
    """
    Get the path to a PBF data file, and download the data if needed.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Run ``pyrosm.data.available`` for
        all available options.

    update : bool
        Whether the PBF file should be downloaded/updated if the dataset
        with the same name exists in the temp.

    directory : str (optional)
        Path to a directory where the PBF data will be downloaded.
        (does not apply for test data sets bundled with the package).
    """

    if not isinstance(dataset, str):
        raise ValueError(f"'dataset' should be text. Got {dataset}.")
    dataset = dataset.lower().strip()

    if dataset in _package_files:
        return os.path.abspath(os.path.join(_module_path, _package_files[dataset]))

    elif dataset == "helsinki_region_pbf":
        return retrieve(_helsinki_region_pbf, update, directory)

    elif dataset == "helsinki_history_pbf":
        return retrieve(_helsinki_history_pbf, update, directory)

    elif dataset == "helsinki_test_history_pbf":
        return retrieve(_helsinki_test_history_pbf, update, directory)

    elif dataset in sources._all_sources:
        return retrieve(search_source(dataset), update, directory)

    # Users might pass city names with spaces (e.g. Rio De Janeiro)
    elif dataset.replace(" ", "") in sources._all_sources:
        return retrieve(search_source(dataset.replace(" ", "")), update, directory)

    # Users might pass country names without underscores (e.g. North America)
    elif dataset.replace(" ", "_") in sources._all_sources:
        return retrieve(search_source(dataset.replace(" ", "_")), update, directory)

    # Users might pass country names with dashes instead of underscores (e.g. canary-islands)
    elif dataset.replace("-", "_") in sources._all_sources:
        return retrieve(search_source(dataset.replace("-", "_")), update, directory)

    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(sources._all_sources))
        raise ValueError(msg)


# Keep temporarily for backward compatibility
def get_path(dataset, update=False, directory=None):
    warnings.warn(
        "'get_path()' is deprecated, use 'get_data()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_data(dataset, update, directory)
