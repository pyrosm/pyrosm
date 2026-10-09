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
    # Compact store (cykhash id->index map + column arrays) replacing the former
    # dict-of-dicts; it still iterates as (id, record) pairs.
    from pyrosm.node_lookup import NodeLocations

    assert isinstance(node_coordinates, NodeLocations)
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


def test_relation_ids_are_not_delta_decoded(test_pbf):
    """Regression test for #170.

    Relation object ids in the PBF format are plain int64 values (only
    DenseNodes ids and relation member references are delta-encoded). They must
    therefore be read as-is, not cumulatively summed. The cumsum bug produced
    inflated, monotonically increasing ids; here we assert the real OSM ids.
    """
    import numpy as np
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    ids = osm._relations["id"]

    # Ids must stay int64 (consistent with the rest of the id handling)
    assert ids.dtype == np.int64

    # Real OSM relation ids in test.osm.pbf. The old cumsum bug would instead
    # have produced cumulative sums [32694, 352283, 2617378, 5307012, 8486578].
    assert ids.tolist() == [32694, 319589, 2265095, 2689634, 3179566]


def test_id_tag_does_not_overwrite_osm_id():
    """Regression test for #233.

    An OSM tag literally keyed ``id`` would otherwise overwrite the element's
    OSM id when way tags are flattened. It must instead be surfaced under the
    ``id_tag`` column, leaving the element ``id`` intact.
    """
    from pyrosm.tagparser import explode_way_tags

    ways = [
        {
            "id": 12345,
            "version": 1,
            "nodes": [1, 2, 3],
            "tags": {"building": "yes", "id": "stray-tag-value"},
        },
        {
            "id": 67890,
            "version": 1,
            "nodes": [4, 5],
            "tags": {"highway": "residential"},
        },
    ]

    exploded = explode_way_tags(ways)

    # The colliding 'id' tag is surfaced as 'id_tag', element id is preserved
    assert exploded[0]["id"] == 12345
    assert exploded[0]["id_tag"] == "stray-tag-value"
    assert exploded[0]["building"] == "yes"
    assert "tags" not in exploded[0]

    # A way without an 'id' tag is unaffected (no spurious id_tag key)
    assert exploded[1]["id"] == 67890
    assert "id_tag" not in exploded[1]


@pytest.mark.parametrize(
    "bounding_box, passes", [(None, 1), ([26.94, 60.525, 26.96, 60.535], 2)]
)
def test_read_progress(test_pbf, bounding_box, passes):
    """A progress callback gets the bytes of the file read in each pass over it, from
    (0, size) to (size, size), and does not change what is read. A bounding box that ways
    cross makes a second pass for their vertices outside the box."""
    from pathlib import Path

    from pyrosm import OSM

    size = Path(test_pbf).stat().st_size
    seen = []
    osm = OSM(
        test_pbf, bounding_box=bounding_box, progress=lambda *report: seen.append(report)
    )
    buildings = osm.get_buildings()
    assert {total for _, total in seen} == {size}
    starts = [i for i, (done, _) in enumerate(seen) if done == 0]
    assert len(starts) == passes
    for start, end in zip(starts, starts[1:] + [len(seen)]):
        done = [d for d, _ in seen[start:end]]
        assert done == sorted(done) and done[-1] == size
    expected = OSM(test_pbf, bounding_box=bounding_box, progress=False).get_buildings()
    assert buildings.equals(expected)


@pytest.mark.parametrize(
    "elapsed, err", [(1.9, ""), (2.0, "Reading test.osm.pbf took 00:02\n")]
)
def test_read_progress_without_terminal(test_pbf, monkeypatch, capsys, elapsed, err):
    """Without a terminal the default read shows no bar; a read that took 2 s or more
    prints one line with the time it took."""
    from pyrosm import OSM
    from pyrosm.utils import progress

    monkeypatch.setattr(progress, "_clock", iter([0.0, elapsed]).__next__)
    OSM(test_pbf).get_buildings()
    assert capsys.readouterr() == ("", err)
