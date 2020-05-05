import pytest
from pyrosm import get_data


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


def test_parsing_basic_elements_from_pbf(test_pbf):
    from pyrosm import OSM
    import numpy as np
    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    nodes, ways = osm._nodes, osm._way_records

    assert isinstance(nodes, list)
    assert isinstance(ways, list)

    # Required node columns
    node_cols = ['id', 'version', 'changeset', 'timestamp', 'lon', 'lat', 'tags']
    for col in node_cols:
        for node_set in nodes:
            assert col in node_set.keys()
            # Nodes should be in numpy arrays
            assert isinstance(node_set[col], np.ndarray)

            # Check shape
            assert len(node_set[col]) in [6222, 8000]

    # Check ways shape
    assert len(ways) == 2653
    for way in ways:
        assert isinstance(way, dict)

    # Required way columns
    way_cols = ['id', 'version', 'timestamp', 'nodes']
    for way in ways:
        for col in way_cols:
            assert col in way.keys()


def test_getting_nodes(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    nodes = osm._nodes_gdf

    assert isinstance(nodes, GeoDataFrame)

    # Required node columns
    node_cols = ['id', 'version', 'changeset', 'timestamp', 'lon', 'lat', 'tags']
    for col in node_cols:
        assert col in nodes.columns

    # Check shape
    assert nodes.shape == (14222, 8)