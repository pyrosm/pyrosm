"""Post-build smoke test run by cibuildwheel against each built wheel.

Imports the installed wheel (cibuildwheel runs this from outside the source
tree, so ``import pyrosm`` resolves to the wheel, not the repo) and parses the
bundled ``test_pbf`` extract end to end, exercising the compiled Cython
pipeline. Kept dependency-light and network-free so it works in every wheel's
isolated test environment.
"""

from pyrosm import OSM, get_data
from pyrosm.data.extract_index import _MOVISDA_GRID_PATH, _movisda_index_frame


def main():
    osm = OSM(get_data("test_pbf"))
    network = osm.get_network()
    assert network is not None and len(network) > 0, "empty network from test_pbf"
    # The vendored index files ship inside the wheel.
    tiles = _movisda_index_frame("grid", _MOVISDA_GRID_PATH)
    assert len(tiles) > 1000, "the vendored Movisda grid index is missing or empty"
    print(
        f"smoke test OK: parsed {len(network)} network edges, {len(tiles)} grid tiles"
    )


if __name__ == "__main__":
    main()
