"""Post-build smoke test run by cibuildwheel against each built wheel.

Imports the installed wheel (cibuildwheel runs this from outside the source
tree, so ``import pyrosm`` resolves to the wheel, not the repo) and parses the
bundled ``test_pbf`` extract end to end, exercising the compiled Cython
pipeline. Kept dependency-light and network-free so it works in every wheel's
isolated test environment.
"""

from pyrosm import OSM, get_data


def main():
    osm = OSM(get_data("test_pbf"))
    network = osm.get_network()
    assert network is not None and len(network) > 0, "empty network from test_pbf"
    print(f"smoke test OK: parsed {len(network)} network edges")


if __name__ == "__main__":
    main()
