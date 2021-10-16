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
    nodes, ways, relations, node_coordinates = (
        osm._nodes,
        osm._way_records,
        osm._relations,
        osm._node_coordinates,
    )

    # NODES
    # -----
    assert isinstance(
        nodes, dict
    ), "nodes should be a dictionary with nd.arrays as values."

    # Required node columns
    node_cols = [
        "id",
        "version",
        "changeset",
        "timestamp",
        "lon",
        "lat",
        "tags",
        "visible",
    ]
    for col in node_cols:
        assert col in nodes.keys()
        # Nodes should be in numpy arrays
        assert isinstance(nodes[col], np.ndarray), (
            f"nodes should be in a dictionary with {node_cols} as keys"
            f" and numpy arrays as values."
        )
        # Check shape
        assert len(nodes[col]) == 14222

    # WAYS
    # ----

    # Ways should be a list of dictionaries
    assert isinstance(
        ways, list
    ), f"way_records should be a list of dictionaries, got '{type(ways)}'."
    for way in ways:
        assert isinstance(way, dict)

    # Check ways shape
    assert len(ways) == 2653

    # Required way columns
    way_cols = ["id", "version", "timestamp", "nodes"]
    for way in ways:
        for col in way_cols:
            assert col in way.keys()

    # RELATIONS
    # ---------
    assert isinstance(
        relations, dict
    ), "relations should be a dictionary with nd.arrays as values."
    relation_cols = [
        "id",
        "version",
        "changeset",
        "timestamp",
        "members",
        "tags",
        "visible",
    ]
    for col in relation_cols:
        assert col in relations.keys()
        # Nodes should be in numpy arrays
        assert isinstance(relations[col], np.ndarray), (
            f"relations should be in a dictionary with {node_cols} as keys"
            f" and numpy arrays as values."
        )
        # Check shape
        assert len(relations[col]) == 5

    # NODE COORDINATES
    # ----------------
    assert isinstance(node_coordinates, dict)
    # Keys should be integers representing the ids
    for key, value in node_coordinates.items():
        assert isinstance(key, int)

        # Each value should have lat and lon information as floats
        assert "lat" in value.keys()
        assert "lon" in value.keys()
        assert isinstance(value["lat"], float)
        assert isinstance(value["lon"], float)


def test_getting_nodes(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    nodes = osm._nodes_gdf

    assert isinstance(nodes, GeoDataFrame)

    # Required node columns
    node_cols = [
        "id",
        "version",
        "changeset",
        "timestamp",
        "lon",
        "lat",
        "tags",
        "visible",
    ]
    for col in node_cols:
        assert col in nodes.columns

    # Check shape
    assert nodes.shape == (14222, 9)
