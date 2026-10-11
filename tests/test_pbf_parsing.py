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
    from pyrosm.way_store import NodeTags
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
        # Nodes should be in numpy arrays, and their tags in a NodeTags store
        expected = NodeTags if col == "tags" else np.ndarray
        assert isinstance(nodes[col], expected), (
            f"nodes should be in a dictionary with {node_cols} as keys"
            f" and numpy arrays (tags: NodeTags) as values."
        )
        # Check shape
        assert len(nodes[col]) == 14222

    # WAYS
    # ----

    # The way store yields a dictionary per way
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


def _write_pbf(path, *blocks):
    """Write a PBF of a header block and then ``blocks`` (``PrimitiveBlock``s)."""
    import zlib
    from struct import pack

    from pyrosm.proto.fileformat_pb2 import Blob, BlobHeader
    from pyrosm.proto.osmformat_pb2 import HeaderBlock

    header = HeaderBlock()
    header.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
    with open(path, "wb") as f:
        for kind, message in [("OSMHeader", header)] + [("OSMData", b) for b in blocks]:
            data = message.SerializeToString()
            blob = Blob(raw_size=len(data), zlib_data=zlib.compress(data))
            blob_bytes = blob.SerializeToString()
            blob_header = BlobHeader(type=kind, datasize=len(blob_bytes))
            header_bytes = blob_header.SerializeToString()
            f.write(pack("!L", len(header_bytes)) + header_bytes + blob_bytes)


def test_id_tag_does_not_overwrite_osm_id(tmp_path, monkeypatch):
    """Regression test for #233.

    An OSM tag literally keyed ``id`` would otherwise overwrite the element's
    OSM id when way tags are flattened. It must instead be surfaced under the
    ``id_tag`` column, leaving the element ``id`` intact.
    """
    from pyrosm import OSM
    from pyrosm.engine import geoparquet
    from pyrosm.pbfreader import parse_osm_data
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    strings = [b"", b"building", b"yes", b"id", b"stray-tag-value", b"highway", b"residential"]
    nodes = PrimitiveBlock()
    nodes.stringtable.s.extend(strings)
    dense = nodes.primitivegroup.add().dense
    dense.id.extend([1, 1])  # delta coded: nodes 1 and 2
    dense.lat.extend([0, 1])
    dense.lon.extend([0, 1])
    ways = PrimitiveBlock()
    ways.stringtable.s.extend(strings)
    group = ways.primitivegroup.add()
    for way_id, keys, vals in [(67890, [5], [6]), (12345, [1, 3], [2, 4])]:
        way = group.ways.add(id=way_id)
        way.keys.extend(keys)
        way.vals.extend(vals)
        way.refs.extend([1, 1])
    fp = str(tmp_path / "id_tag.osm.pbf")
    _write_pbf(fp, nodes, ways)

    exploded = list(parse_osm_data(fp, None, False, None)[1])

    # The colliding 'id' tag is surfaced as 'id_tag', element id is preserved
    assert exploded[1]["id"] == 12345
    assert exploded[1]["id_tag"] == "stray-tag-value"
    assert exploded[1]["building"] == "yes"
    assert list(exploded[1]) == [
        "id", "version", "timestamp", "visible", "nodes", "building", "id_tag"
    ]

    # A way without an 'id' tag is unaffected (no spurious id_tag key)
    assert exploded[0]["id"] == 67890
    assert "id_tag" not in exploded[0]

    # Both engines return it in an id_tag column at the same place, also when the
    # out-of-core engine writes its result in chunks of one way, the first without it.
    monkeypatch.setattr(geoparquet, "_OUTPUT_CHUNK_SIZE", 1)
    frames = [
        OSM(fp, engine=engine).get_data_by_custom_criteria(
            {"building": True, "highway": True}, extra_attributes=["id_tag"]
        )
        for engine in ("in_memory", "out_of_core")
    ]
    assert list(frames[1].columns) == list(frames[0].columns)
    for frame in frames:
        assert frame.set_index("id")["id_tag"][12345] == "stray-tag-value"


@pytest.mark.parametrize(
    "osm_keys, relation_way_ids, keep_all, rows",
    [
        (["highway"], None, False, [0]),
        (["id_tag"], None, False, [1]),
        (["version"], None, False, [0, 1, 2, 3]),
        (["building"], [13], False, [2, 3]),
        ([], None, True, [0, 1, 2]),
    ],
    ids=["tag_key", "id_tag", "record_key", "relation_member", "keep_all"],
)
def test_way_store_candidate_rows(osm_keys, relation_way_ids, keep_all, rows):
    """A way store selects the ways the key check of filter_osm_records can keep: ways
    with a tag keyed in osm_keys (an id tag for id_tag), every way for a key every
    record has, ways with any tag with keep_all, and relation members."""
    import numpy as np
    from pyrosm.way_store import WayStoreBuilder

    strings = ["", "highway", "primary", "id", "x", "building", "yes"]
    tags = [[(1, 2)], [(3, 4)], [(5, 6)], []]
    keys = [k for way in tags for k, _ in way]
    vals = [v for way in tags for _, v in way]
    builder = WayStoreBuilder()
    builder.add(
        {
            "id": np.array([10, 11, 12, 13]),
            "version": np.ones(4, dtype=np.int64),
            "timestamp": np.ones(4, dtype=np.int64),
            "visible": np.ones(4, dtype=np.int64),
            "refs": np.array([1, 2, 3, 4, 5]),
            "refs_off": np.array([0, 2, 3, 4, 5]),
            "keys": np.array(keys),
            "vals": np.array(vals),
            "tags_off": np.array([0, 1, 2, 3, 3]),
        },
        strings,
    )
    store = builder.build()
    assert store.candidate_rows(osm_keys, relation_way_ids, keep_all).tolist() == rows
    assert [way["id"] for way in store] == [10, 11, 12, 13]


def _node_tags():
    """Node tags of four nodes (shop=bakery; none; amenity=cafe and name=x; none), the
    last one not kept."""
    import numpy as np
    from pyrosm.way_store import NodeTagsBuilder

    strings = ["", "shop", "bakery", "amenity", "cafe", "name", "x"]
    builder = NodeTagsBuilder()
    builder.add(
        np.array([1, 2, 0, 0, 3, 4, 5, 6, 0, 0]),
        4,
        strings,
        keep=np.array([True, True, True, False]),
    )
    return builder.build()


@pytest.mark.parametrize(
    "osm_keys, keep_all, rows",
    [
        (["amenity"], False, [2]),
        (["amenity", "shop"], False, [0, 2]),
        (["missing"], False, []),
        ([], True, [0, 2]),
        ("amenity", False, [0, 2]),
    ],
    ids=["one_key", "two_keys", "absent_key", "keep_all", "keys_not_a_list"],
)
def test_node_tags_candidate_rows(osm_keys, keep_all, rows):
    """The node rows the node filter can keep: the tagged nodes carrying one of the keys,
    or every tagged node with keep_all or when the keys are not a list."""
    assert _node_tags().candidate_rows(osm_keys, keep_all).tolist() == rows


def test_node_tags_lookup():
    """Node tags kept as arrays give each kept node's tags as a dict (None for an
    untagged node) by row, by array of rows and by iterating."""
    import numpy as np

    tags = _node_tags()
    cafe = {"amenity": "cafe", "name": "x"}
    assert len(tags) == 3
    assert list(tags) == [{"shop": "bakery"}, None, cafe]
    assert tags[-1] == cafe
    assert tags[np.array([2, 1])].tolist() == [cafe, None]
    with pytest.raises(IndexError):
        tags[3]


@pytest.mark.parametrize(
    "bounding_box, passes", [(None, 1), ([26.94, 60.525, 26.96, 60.535], 2)]
)
def test_read_progress(test_pbf, bounding_box, passes):
    """A progress callback of the in-memory reader gets the bytes of the file read in
    each pass over it, from (0, size) to (size, size), and does not change what is read.
    A bounding box that ways cross makes a second pass for their vertices outside the
    box."""
    from pathlib import Path

    from pyrosm import OSM

    size = Path(test_pbf).stat().st_size
    seen = []
    osm = OSM(
        test_pbf,
        bounding_box=bounding_box,
        progress=lambda *report: seen.append(report),
        engine="in_memory",
    )
    buildings = osm.get_buildings()
    assert {total for _, total in seen} == {size}
    starts = [i for i, (done, _) in enumerate(seen) if done == 0]
    assert len(starts) == passes
    for start, end in zip(starts, starts[1:] + [len(seen)]):
        done = [d for d, _ in seen[start:end]]
        assert done == sorted(done) and done[-1] == size
    expected = OSM(
        test_pbf, bounding_box=bounding_box, progress=False, engine="in_memory"
    ).get_buildings()
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
