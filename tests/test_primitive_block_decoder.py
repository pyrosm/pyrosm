"""Byte-exact parity of the raw ``primitive_block_decoder`` against the protobuf path.

Every node/way/relation field the raw decoder emits must equal what protobuf decodes
from the same ``PrimitiveBlock`` bytes, on the bundled extracts.
"""

import zlib
from struct import unpack

import numpy as np
import pytest

from pyrosm import get_data
from pyrosm.exceptions import InvalidOSMFileError
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.proto.osmformat_pb2 import PrimitiveBlock
from pyrosm.primitive_block_decoder import decode_primitive_block


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture(scope="module")
def nondense_pbf(tmp_path_factory):
    """The bundled test PBF rewritten by osmium with plain (non-dense) node groups."""
    osmium = pytest.importorskip("osmium")
    path = str(tmp_path_factory.mktemp("nondense") / "test_nondense.osm.pbf")
    with osmium.SimpleWriter(
        osmium.io.File(path, "pbf,pbf_dense_nodes=false")
    ) as writer:
        for obj in osmium.FileProcessor(get_data("test_pbf")):
            writer.add(obj)
    return path


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _iter_block_bytes(fp):
    """Yield the decompressed ``PrimitiveBlock`` bytes of every OSMData blob."""
    with open(fp, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = unpack("!L", head)
            bh = BlobHeader()
            bh.ParseFromString(f.read(n))
            blob = Blob()
            blob.ParseFromString(f.read(bh.datasize))
            if blob.HasField("zlib_data"):
                data = zlib.decompress(blob.zlib_data)
            elif blob.HasField("raw"):
                data = blob.raw
            else:
                continue
            if bh.type == "OSMData":
                yield data


def _offsets(counts):
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    if counts:
        np.cumsum(counts, out=off[1:])
    return off


def _i64(field):
    return np.fromiter(field, dtype=np.int64, count=len(field))


def _ref_decode(raw, keep_metadata=True):
    """Reference decode of a PrimitiveBlock via protobuf, shaped like the raw decoder."""
    pb = PrimitiveBlock()
    pb.ParseFromString(raw)
    header = {
        "granularity": pb.granularity,
        "date_granularity": pb.date_granularity,
        "lat_offset": pb.lat_offset,
        "lon_offset": pb.lon_offset,
    }

    nodes = None
    node_groups = []
    w = {
        k: []
        for k in (
            "id",
            "keys",
            "vals",
            "tags",
            "refs",
            "refn",
            "version",
            "timestamp",
            "changeset",
            "uid",
            "user_sid",
            "visible",
        )
    }
    r = {
        k: []
        for k in (
            "id",
            "keys",
            "vals",
            "tags",
            "memids",
            "memn",
            "types",
            "roles",
            "version",
            "timestamp",
            "changeset",
            "uid",
            "user_sid",
            "visible",
        )
    }

    for g in pb.primitivegroup:
        d = g.dense
        if len(d.id):
            di = d.denseinfo
            node_groups.append(
                {
                    "id": np.cumsum(_i64(d.id)),
                    "lat": np.cumsum(_i64(d.lat)),
                    "lon": np.cumsum(_i64(d.lon)),
                    "keys_vals": _i64(d.keys_vals),
                    "version": _i64(di.version),
                    "timestamp": np.cumsum(_i64(di.timestamp)),
                    "changeset": np.cumsum(_i64(di.changeset)),
                    "uid": np.cumsum(_i64(di.uid)),
                    "user_sid": np.cumsum(_i64(di.user_sid)),
                    "visible": _i64(di.visible),
                }
            )
        if len(g.nodes):
            node_groups.append(_ref_plain_nodes(g.nodes))
        for way in g.ways:
            w["id"].append(way.id)
            w["keys"].append(_i64(way.keys))
            w["vals"].append(_i64(way.vals))
            w["tags"].append(len(way.keys))
            w["refs"].append(np.cumsum(_i64(way.refs)))
            w["refn"].append(len(way.refs))
            _ref_info(w, way.info)
        for rel in g.relations:
            r["id"].append(rel.id)
            r["keys"].append(_i64(rel.keys))
            r["vals"].append(_i64(rel.vals))
            r["tags"].append(len(rel.keys))
            r["memids"].append(np.cumsum(_i64(rel.memids)))
            r["memn"].append(len(rel.memids))
            r["types"].append(_i64(rel.types))
            r["roles"].append(_i64(rel.roles_sid))
            _ref_info(r, rel.info)

    if node_groups:
        keys = ["id", "lat", "lon", "keys_vals"]
        if keep_metadata:
            keys += list(_META)
        nodes = {k: _ref_join(node_groups, k) for k in keys}

    ways = _ref_build(w, "refs", "refn", keep_metadata)
    relations = _ref_build(r, "memids", "memn", keep_metadata)
    if relations is not None:
        relations["types"] = (
            np.concatenate(r["types"]) if r["types"] else np.empty(0, np.int64)
        )
        relations["roles"] = (
            np.concatenate(r["roles"]) if r["roles"] else np.empty(0, np.int64)
        )
        relations["members_off"] = relations.pop("memids_off")
    return string_table_ref(pb), header, nodes, ways, relations


_META = ("version", "timestamp", "changeset", "uid", "user_sid", "visible")


def _ref_plain_nodes(plain):
    """One group of plain nodes in the dense form: absolute values, and each node's
    key/value ids followed by a 0 when any node of the group has tags."""
    tagged = any(len(node.keys) for node in plain)
    keys_vals = []
    meta = {k: [] for k in _META}
    for node in plain:
        if tagged:
            for k, v in zip(node.keys, node.vals):
                keys_vals += [k, v]
            keys_vals.append(0)
        _ref_info(meta, node.info)
    group = {
        "id": np.array([node.id for node in plain], dtype=np.int64),
        "lat": np.array([node.lat for node in plain], dtype=np.int64),
        "lon": np.array([node.lon for node in plain], dtype=np.int64),
        "keys_vals": np.array(keys_vals, dtype=np.int64),
    }
    group.update({k: np.array(v, dtype=np.int64) for k, v in meta.items()})
    return group


def _ref_join(groups, key):
    """Concatenate ``key`` over the node groups, a 0 per node for a group without it
    when another group has it."""
    have = [len(g[key]) > 0 for g in groups]
    if all(have) or not any(have):
        return np.concatenate([g[key] for g in groups])
    return np.concatenate(
        [
            g[key] if h else np.zeros(len(g["id"]), np.int64)
            for g, h in zip(groups, have)
        ]
    )


def string_table_ref(pb):
    return list(pb.stringtable.s)


def _ref_info(acc, info):
    acc["version"].append(info.version)
    acc["timestamp"].append(info.timestamp)
    acc["changeset"].append(info.changeset)
    acc["uid"].append(info.uid)
    acc["user_sid"].append(info.user_sid)
    acc["visible"].append(int(info.visible))


def _ref_build(acc, members, memn, keep_metadata):
    if not acc["id"]:
        return None
    out = {
        "id": np.array(acc["id"], dtype=np.int64),
        "keys": np.concatenate(acc["keys"]) if acc["keys"] else np.empty(0, np.int64),
        "vals": np.concatenate(acc["vals"]) if acc["vals"] else np.empty(0, np.int64),
        "tags_off": _offsets(acc["tags"]),
        members: (
            np.concatenate(acc[members]) if acc[members] else np.empty(0, np.int64)
        ),
        members + "_off": _offsets(acc[memn]),
    }
    if keep_metadata:
        for k in ("version", "timestamp", "changeset", "uid", "user_sid", "visible"):
            out[k] = np.array(acc[k], dtype=np.int64)
    return out


def _assert_arrays(mine, ref, label):
    assert mine is not None and ref is not None, label
    assert set(mine) >= set(ref), "%s: missing keys %s" % (label, set(ref) - set(mine))
    for key, ref_val in ref.items():
        np.testing.assert_array_equal(
            np.asarray(mine[key]), np.asarray(ref_val), err_msg="%s[%s]" % (label, key)
        )


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf", "nondense_pbf"])
def test_decoder_matches_protobuf(fixture, request):
    fp = request.getfixturevalue(fixture)
    n_blocks = n_nodes = n_ways = n_rels = 0
    for raw in _iter_block_bytes(fp):
        st, header, nodes, ways, relations = decode_primitive_block(raw)
        ref_st, ref_header, ref_nodes, ref_ways, ref_rels = _ref_decode(raw)

        assert st == ref_st
        assert header == ref_header
        n_blocks += 1

        assert (nodes is None) == (ref_nodes is None)
        if nodes is not None:
            _assert_arrays(nodes, ref_nodes, "nodes")
            n_nodes += len(nodes["id"])
        assert (ways is None) == (ref_ways is None)
        if ways is not None:
            _assert_arrays(ways, ref_ways, "ways")
            n_ways += len(ways["id"])
        assert (relations is None) == (ref_rels is None)
        if relations is not None:
            _assert_arrays(relations, ref_rels, "relations")
            n_rels += len(relations["id"])

    # The extracts must actually exercise all three element kinds.
    assert n_blocks > 0 and n_nodes > 0 and n_ways > 0 and n_rels > 0


def test_keep_metadata_false_omits_metadata(helsinki_pbf):
    meta = {"version", "timestamp", "changeset", "uid", "user_sid", "visible"}
    saw_ways = False
    for raw in _iter_block_bytes(helsinki_pbf):
        _, _, nodes, ways, relations = decode_primitive_block(raw, keep_metadata=False)
        for elem in (nodes, ways, relations):
            if elem is not None:
                assert meta.isdisjoint(elem.keys())
        if ways is not None:
            saw_ways = True
    assert saw_ways


def test_empty_block_returns_empty(helsinki_pbf):
    st, header, nodes, ways, relations = decode_primitive_block(b"")
    assert st == [] and nodes is None and ways is None and relations is None
    assert header["granularity"] == 100


def test_node_groups_keep_each_nodes_own_values():
    """In a block whose node groups differ in having tags or metadata (a tagged dense
    group with DenseInfo, an untagged dense group without, and plain nodes with and
    without Info), every node keeps its own tags and metadata."""
    pb = PrimitiveBlock()
    pb.stringtable.s.extend([b"", b"amenity", b"cafe", b"name", b"x"])
    tagged = pb.primitivegroup.add().dense
    tagged.id.extend([1, 1])
    tagged.lat.extend([10, 1])
    tagged.lon.extend([20, 1])
    tagged.keys_vals.extend([1, 2, 0, 0])
    tagged.denseinfo.version.extend([3, 4])
    tagged.denseinfo.timestamp.extend([100, 1])
    untagged = pb.primitivegroup.add().dense
    untagged.id.extend([3, 1])
    untagged.lat.extend([30, 1])
    untagged.lon.extend([40, 1])
    plain = pb.primitivegroup.add()
    node = plain.nodes.add(id=5, lat=50, lon=60)
    node.keys.append(3)
    node.vals.append(4)
    node.info.version = 2
    node.info.timestamp = 500
    plain.nodes.add(id=6, lat=51, lon=61)

    _, _, nodes, ways, relations = decode_primitive_block(pb.SerializeToString())
    assert ways is None and relations is None
    assert nodes["id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert nodes["lat"].tolist() == [10, 11, 30, 31, 50, 51]
    assert nodes["lon"].tolist() == [20, 21, 40, 41, 60, 61]
    assert nodes["keys_vals"].tolist() == [1, 2, 0, 0, 0, 0, 3, 4, 0, 0]
    assert nodes["version"].tolist() == [3, 4, 0, 0, 2, -1]
    assert nodes["timestamp"].tolist() == [100, 101, 0, 0, 500, 0]


def _all_nodes(fp):
    """Every node of ``fp`` in file order: ids, coordinates in degrees, tag dicts and
    metadata, the user id resolved to its name."""
    ids, lon, lat, tags, users = [], [], [], [], []
    meta = {k: [] for k in ("version", "timestamp", "changeset", "uid")}
    for raw in _iter_block_bytes(fp):
        st, header, nodes, _, _ = decode_primitive_block(raw)
        if nodes is None:
            continue
        gran = header["granularity"]
        ids.append(nodes["id"])
        lon.append((nodes["lon"] * gran + header["lon_offset"]) / 1e9)
        lat.append((nodes["lat"] * gran + header["lat_offset"]) / 1e9)
        stream = nodes["keys_vals"].tolist()
        p = 0
        for _ in range(len(nodes["id"])):
            node_tags = {}
            while p < len(stream) and stream[p] != 0:
                node_tags[st[stream[p]]] = st[stream[p + 1]]
                p += 2
            p += 1
            tags.append(node_tags)
        for k in meta:
            meta[k].append(nodes[k])
        users.extend(st[i] for i in nodes["user_sid"])
    out = {"id": np.concatenate(ids), "lon": np.concatenate(lon)}
    out["lat"] = np.concatenate(lat)
    out.update({k: np.concatenate(v) for k, v in meta.items()})
    return out, tags, users


def test_nondense_nodes_equal_the_dense_ones(test_pbf, nondense_pbf):
    """The non-dense copy decodes to the same nodes as the dense file: ids,
    coordinates, tags and metadata."""
    dense, dense_tags, dense_users = _all_nodes(test_pbf)
    plain, plain_tags, plain_users = _all_nodes(nondense_pbf)
    assert len(dense["id"]) > 0 and any(dense_tags)
    for key in dense:
        np.testing.assert_array_equal(plain[key], dense[key], err_msg=key)
    assert plain_tags == dense_tags
    assert plain_users == dense_users


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if not value:
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def _field(number, wire_type, payload):
    """One protobuf field: a varint value (wire type 0) or a length-delimited one (2)."""
    tag = _varint(number << 3 | wire_type)
    if wire_type == 2:
        return tag + _varint(len(payload)) + payload
    return tag + _varint(payload)


def _block_with_plain_node(*fields):
    """A PrimitiveBlock whose only group holds one plain node made of ``fields``."""
    pb = PrimitiveBlock()
    pb.stringtable.s.extend([b"", b"a", b"b", b"c", b"d"])
    group = _field(1, 2, b"".join(fields))
    return pb.SerializeToString() + _field(2, 2, group)


def test_plain_node_tags_in_any_protobuf_encoding():
    """A plain node's key and value ids decode the same packed in several runs or not
    packed at all; a node with more keys than values is refused."""
    node = [_field(1, 0, 14), _field(8, 0, 140), _field(9, 0, 159)]  # 7, 70, -80
    keys = [_field(2, 2, _varint(1)), _field(2, 2, _varint(3))]
    vals = [_field(3, 0, 2), _field(3, 0, 4)]
    _, _, nodes, _, _ = decode_primitive_block(
        _block_with_plain_node(*node, *keys, *vals)
    )
    assert nodes["id"].tolist() == [7]
    assert nodes["lat"].tolist() == [70] and nodes["lon"].tolist() == [-80]
    assert nodes["keys_vals"].tolist() == [1, 2, 3, 4, 0]
    with pytest.raises(InvalidOSMFileError, match="2 tag keys but 1 values"):
        decode_primitive_block(_block_with_plain_node(*node, *keys, vals[0]))



def _batch(blocks):
    """The blocks as one buffer with their start offsets, as decode_batch_nogil takes them."""
    buffer = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    return buffer, _offsets([len(b) for b in blocks])


_DENSE_META = ("version", "timestamp", "changeset", "visible")


def _expected_dense(block, keys, bbox):
    """What decode_batch_nogil gives for a block of dense nodes, from
    decode_primitive_block: the nodes, the in-box flags and the node features."""
    table, header, nodes, _, _ = decode_primitive_block(block)
    if nodes is None:
        nodes = {k: np.empty(0, np.int64) for k in ("id", "lon", "lat", "keys_vals")}
        nodes.update({k: np.empty(0, np.int64) for k in _DENSE_META})
    lon = (nodes["lon"] * header["granularity"] + header["lon_offset"]) / 1e9
    lat = (nodes["lat"] * header["granularity"] + header["lat_offset"]) / 1e9
    xmin, ymin, xmax, ymax = bbox
    wanted = None if keys is None else {table.index(k) for k in keys if k in table}
    meta = [
        nodes[name] if len(nodes[name]) else np.zeros(len(nodes["id"]), dtype=np.int64)
        for name in _DENSE_META
    ]
    rows, pairs, feature_meta = [], [], []
    kv = nodes["keys_vals"].tolist()
    pos = 0
    for row in range(len(nodes["id"]) if kv else 0):
        tags = []
        while kv[pos]:
            tags.append((kv[pos], kv[pos + 1]))
            pos += 2
        pos += 1
        if tags and (wanted is None or any(k in wanted for k, _ in tags)):
            rows.append(row)
            pairs.append(tags)
            feature_meta.append([column[row] for column in meta])
    return {
        "node_id": nodes["id"],
        "node_lon": lon,
        "node_lat": lat,
        "in_box": (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax),
        "rows": rows,
        "pairs": pairs,
        "feature_meta": feature_meta,
    }


@pytest.mark.parametrize("keys", [None, [b"amenity", b"shop"]])
def test_decode_batch_nogil_dense_nodes(test_pbf, helsinki_pbf, keys):
    """The GIL-free batch decoder gives the dense nodes and node features of the bundled
    extracts as decode_primitive_block decodes them; it hands back the blocks with
    relations (status 1). block_string_table gives a block's string table."""
    from pyrosm.primitive_block_decoder import (
        BLOCK_COUNTS,
        block_string_table,
        decode_batch_nogil,
    )

    blocks = list(_iter_block_bytes(test_pbf)) + list(_iter_block_bytes(helsinki_pbf))
    bbox = (24.94, 60.16, 24.96, 60.17)
    out = decode_batch_nogil(*_batch(blocks), keys=keys, features=True, bbox=bbox)
    counts = dict(zip(BLOCK_COUNTS, out["counts"].T.tolist()))
    dense = [decode_primitive_block(block)[4] is None for block in blocks]
    assert counts["status"] == [0 if d else 1 for d in dense]
    assert 0 < sum(dense) < len(blocks)

    expected = [_expected_dense(b, keys, bbox) for b, d in zip(blocks, dense) if d]
    for name in ("node_id", "node_lon", "node_lat", "in_box"):
        assert out[name].tolist() == np.concatenate([e[name] for e in expected]).tolist()
    assert out["in_box"].any()
    row0 = np.cumsum([0] + [len(e["node_id"]) for e in expected])
    rows = [r + row0[i] for i, e in enumerate(expected) for r in e["rows"]]
    assert out["feature_row"].tolist() == rows
    assert 0 < len(rows) < len(out["node_id"])
    pairs = [p for e in expected for p in e["pairs"]]
    off = out["feature_pairs_off"]
    got = [
        list(zip(out["feature_keys"][a:b].tolist(), out["feature_vals"][a:b].tolist()))
        for a, b in zip(off[:-1], off[1:])
    ]
    assert got == pairs
    meta = [m for e in expected for m in e["feature_meta"]]
    assert out["feature_meta"].tolist() == np.array(meta).tolist()
    # The tags' string ids are resolved with the block's string table.
    for block in blocks + [_two_string_tables()]:
        assert block_string_table(block) == decode_primitive_block(block)[0]
    for corrupt in (_two_string_tables() + b"\x12\x02x", _corrupt_string_table()):
        with pytest.raises(InvalidOSMFileError):
            block_string_table(corrupt)


def _dense_block(id_runs, keys_vals=None, untagged_group=False):
    """A PrimitiveBlock with a dense group of the nodes 1, 2, ... (their id deltas packed in
    ``id_runs`` runs), optionally tagged, and optionally a second, untagged dense group.
    """
    n = sum(id_runs)
    dense = b"".join(_field(1, 2, _varint(2) * run) for run in id_runs)
    dense += _field(8, 2, _varint(2) * n) + _field(9, 2, _varint(4) * n)
    if keys_vals is not None:
        dense += _field(10, 2, b"".join(_varint(v) for v in keys_vals))
    groups = _field(2, 2, _field(2, 2, dense))
    if untagged_group:
        other = (
            _field(1, 2, _varint(2))
            + _field(8, 2, _varint(2))
            + _field(9, 2, _varint(4))
        )
        groups += _field(2, 2, _field(2, 2, other))
    pb = PrimitiveBlock()
    pb.stringtable.s.extend([b"", b"amenity", b"cafe"])
    return pb.SerializeToString() + groups


def _dense_groups(*extras):
    """A PrimitiveBlock with one dense group of node 1 per item of ``extras``: the encoded
    DenseNodes fields that group has besides its ids and coordinates."""
    pb = PrimitiveBlock()
    pb.stringtable.s.extend([b"", b"amenity", b"cafe"])
    nodes = _field(1, 2, _varint(2)) + _field(8, 2, _varint(2)) + _field(9, 2, _varint(4))
    return pb.SerializeToString() + b"".join(
        _field(2, 2, _field(2, 2, nodes + extra)) for extra in extras
    )


def _plain_node_block():
    return _block_with_plain_node(
        _field(1, 0, 14), _field(8, 0, 140), _field(9, 0, 159)
    )


def _relation_block():
    pb = PrimitiveBlock()
    pb.stringtable.s.append(b"")
    pb.primitivegroup.add().relations.add(id=1)
    return pb.SerializeToString()


def _raw_block(group):
    """A PrimitiveBlock with one group made of the encoded ``group`` fields."""
    pb = PrimitiveBlock()
    pb.stringtable.s.extend([b"", b"amenity", b"cafe"])
    return pb.SerializeToString() + _field(2, 2, group)


_CUT_OFF_ID = _raw_block(
    _field(
        2,
        2,
        _field(1, 2, b"\x02\x82")
        + _field(8, 2, b"\x02\x02")
        + _field(9, 2, b"\x04\x04"),
    )
)
_CUT_OFF_WAY_INFO = _raw_block(_field(3, 2, _field(1, 0, 7) + _field(4, 2, b"\x08")))


def _two_string_tables():
    """A block whose first string table maps 1 to another key than its last one does."""
    first = PrimitiveBlock()
    first.stringtable.s.extend([b"", b"x", b"amenity"])
    return first.SerializeToString() + _dense_block([2], [1, 2, 0, 0])


def _corrupt_string_table():
    """A block whose string table holds a string cut off."""
    table = PrimitiveBlock()
    table.stringtable.s.extend([b"", b"amenity", b"cafe"])
    groups = _dense_block([2])[len(table.SerializeToString()) :]
    return _field(1, 2, _field(1, 2, b"ab")[:-1]) + groups


@pytest.mark.parametrize(
    "case, block, status",
    [
        (
            "dense nodes in runs",
            _dense_block([1, 2, 3, 1, 1, 1, 1, 1], [1, 2, 0] + [0] * 10),
            0,
        ),
        ("more packed runs than kept", _dense_block([1] * 9), 1),
        (
            "tags in only some groups",
            _dense_block([2], [1, 2, 0, 0], untagged_group=True),
            1,
        ),
        ("tag id outside the string table", _dense_block([2], [1, 3, 0, 0]), 1),
        (
            "way tag id outside the string table",
            _raw_block(_field(3, 2, _field(1, 0, 7) + _field(2, 2, b"\x01") + _field(3, 2, b"\x03"))),
            1,
        ),
        ("a way", _raw_block(_field(3, 2, _field(1, 0, 7) + _field(8, 2, b"\x02"))), 0),
        (
            "metadata in only some groups",
            _dense_groups(_field(5, 2, _field(1, 2, _varint(1))), b""),
            1,
        ),
        (
            "tags in two runs",
            _dense_groups(_field(10, 2, _varint(1) + _varint(2)) + _field(10, 2, b"\0")),
            1,
        ),
        (
            "tags that do not end once per node",
            _dense_groups(_field(10, 2, _varint(1) + _varint(2))),
            "corrupt",
        ),
        ("plain nodes", _plain_node_block(), 1),
        ("relations", _relation_block(), 1),
        ("cut off", _dense_block([2])[:-3], "corrupt"),
        ("cut-off varint in a packed field", _CUT_OFF_ID, "corrupt"),
        ("cut-off way info", _CUT_OFF_WAY_INFO, "corrupt"),
        ("two string tables", _two_string_tables(), 1),
        ("cut-off string table", _corrupt_string_table(), "corrupt"),
        (
            "cut-off first string table",
            b"\x0a\x04\x0a\x05ab" + _two_string_tables(),
            "corrupt",
        ),
    ],
)
def test_decode_batch_nogil_decodes_or_hands_back(case, block, status):
    """The GIL-free batch decoder decodes a block as decode_primitive_block does, or gives it
    status 1 (no elements) for the caller to decode with decode_primitive_block, which
    raises for a corrupt block."""
    from pyrosm.primitive_block_decoder import BLOCK_COUNTS, decode_batch_nogil

    dense = _dense_block([2], [1, 2, 0, 0])
    buffer = np.frombuffer(dense + block, dtype=np.uint8)
    starts = np.array([0, len(dense), len(dense) + len(block)], dtype=np.int64)
    out = decode_batch_nogil(
        buffer, starts, keys=[b"amenity"], features=True, all_ways=True
    )
    counts = dict(zip(BLOCK_COUNTS, out["counts"].T.tolist()))
    assert counts["status"] == [0, 0 if status == 0 else 1]
    unfiltered = decode_batch_nogil(buffer, starts, features=True, all_ways=True)
    assert unfiltered["counts"][:, 0].tolist() == counts["status"]
    if status:
        assert counts["nodes"][1] == counts["features"][1] == counts["all_ways"][1] == 0
    else:
        _, _, nodes, _, _ = decode_primitive_block(block)
        node_ids = [] if nodes is None else nodes["id"].tolist()
        assert counts["nodes"][1] == len(node_ids)
    if status == "corrupt":
        with pytest.raises(InvalidOSMFileError):
            decode_primitive_block(block)
    assert out["node_id"].tolist()[:2] == [1, 2]
    assert out["node_id"].tolist()[2:] == ([] if status else node_ids)
    # Each block's first node carries amenity=cafe.
    assert out["feature_row"].tolist() == [0] + ([2] if not status and node_ids else [])


@pytest.mark.parametrize(
    "starts", [[], [-1, 4], [0, 9], [4, 0, 8], [0, 2**63 - 1, -(2**63)]]
)
def test_decode_batch_nogil_checks_starts(starts):
    from pyrosm.primitive_block_decoder import decode_batch_nogil

    buffer = np.zeros(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="starts"):
        decode_batch_nogil(buffer, np.array(starts, dtype=np.int64))


def test_decode_batch_nogil_empty_buffer():
    """A batch of empty blocks in an empty buffer decodes to nothing."""
    from pyrosm.primitive_block_decoder import decode_batch_nogil

    out = decode_batch_nogil(np.zeros(0, dtype=np.uint8), np.array([0, 0], dtype=np.int64))
    assert out["counts"].shape[0] == 1 and not out["counts"].any()
    assert len(out["node_id"]) == 0


@pytest.mark.parametrize(
    "case, block, tags",
    [
        ("tagged, ids in two runs", _dense_block([2, 2], [0, 1, 2, 0, 0, 0]), [1, 0]),
        ("untagged", _dense_block([4]), [0, 0]),
    ],
)
def test_decode_batch_nogil_selects_nodes_by_id(case, block, tags):
    """With node_ids a dense node whose id is in the set is a node feature, tagged or
    not, with its tags (here node 2's amenity=cafe, string ids 1 and 2) and its metadata
    (0 without DenseInfo)."""
    from pyrosm.int64_hash import Int64Set
    from pyrosm.primitive_block_decoder import decode_batch_nogil

    buffer = np.frombuffer(block, dtype=np.uint8)
    starts = np.array([0, len(block)], dtype=np.int64)
    out = decode_batch_nogil(
        buffer,
        starts,
        keys=[],
        features=True,
        node_ids=Int64Set(np.array([2, 3, 9])),
    )
    assert out["counts"][0, 0] == 0
    assert out["node_id"][out["feature_row"]].tolist() == [2, 3]
    assert np.diff(out["feature_pairs_off"]).tolist() == tags
    pairs = list(zip(out["feature_keys"].tolist(), out["feature_vals"].tolist()))
    assert pairs == ([(1, 2)] if sum(tags) else [])
    assert out["feature_meta"].tolist() == [[0, 0, 0, 0]] * 2


def test_decode_batch_nogil_corrupt_blocks(test_pbf, corrupted_copies):
    """A corrupted block of dense nodes and ways is decoded as decode_primitive_block
    decodes it, or handed back (status 1); the decoder never reads outside the block."""
    from pyrosm.primitive_block_decoder import decode_batch_nogil

    block = list(_iter_block_bytes(test_pbf))[1]
    for data in corrupted_copies(block, 300, seed=2):
        out = decode_batch_nogil(*_batch([data]), features=True, all_ways=True)
        if out["counts"][0, 0] == 1:
            continue
        _, _, nodes, ways, _ = decode_primitive_block(data)
        assert out["node_id"].tolist() == ([] if nodes is None else nodes["id"].tolist())
        assert out["all_id"].tolist() == ([] if ways is None else ways["id"].tolist())


# A way's tag amenity=cafe: string ids 1 and 2 of the table of ``_raw_block``.
_WAY_TAG = _field(2, 2, b"\x01") + _field(3, 2, b"\x02")


def _expected_ways(block, keys):
    """What decode_batch_nogil gives for the ways of a block (with ``all_ways``), from
    decode_primitive_block: every way's id and refs, and the ways with a wanted key with
    all their tags and their version, timestamp and visible."""
    table, _, _, ways, _ = decode_primitive_block(block)
    if ways is None:
        return {"all": [], "matched": []}
    wanted = None if keys is None else {table.index(k) for k in keys if k in table}
    refs_off, tags_off = ways["refs_off"].tolist(), ways["tags_off"].tolist()
    every, matched = [], []
    for i, way_id in enumerate(ways["id"].tolist()):
        refs = ways["refs"][refs_off[i] : refs_off[i + 1]].tolist()
        keys_i = ways["keys"][tags_off[i] : tags_off[i + 1]].tolist()
        vals_i = ways["vals"][tags_off[i] : tags_off[i + 1]].tolist()
        every.append((way_id, refs))
        if keys_i and (wanted is None or wanted.intersection(keys_i)):
            meta = [int(ways[c][i]) for c in ("version", "timestamp", "visible")]
            matched.append((way_id, refs, list(zip(keys_i, vals_i)), meta))
    return {"all": every, "matched": matched}


def _rows(off, *columns):
    """The rows of CSR ``columns`` that share the offsets ``off``."""
    off = off.tolist()
    return [
        [column[a:b].tolist() for column in columns] for a, b in zip(off[:-1], off[1:])
    ]


@pytest.mark.parametrize("keys", [None, [b"highway", b"building"]])
def test_decode_batch_nogil_ways(test_pbf, helsinki_pbf, keys):
    """The GIL-free batch decoder gives every way's id and refs, and the ways with a wanted
    key with their tags and metadata, as decode_primitive_block decodes them."""
    from pyrosm.primitive_block_decoder import decode_batch_nogil

    blocks = list(_iter_block_bytes(test_pbf)) + list(_iter_block_bytes(helsinki_pbf))
    # A way without Info, whose metadata defaults as decode_primitive_block's does.
    blocks.append(_raw_block(_field(3, 2, _field(1, 0, 7) + _WAY_TAG + _field(8, 2, b"\x02"))))
    out = decode_batch_nogil(*_batch(blocks), keys=keys, all_ways=True)
    decoded = [b for b, s in zip(blocks, out["counts"][:, 0].tolist()) if s == 0]
    expected = [_expected_ways(b, keys) for b in decoded]
    assert any(e["matched"] for e in expected)
    every = [w for e in expected for w in e["all"]]
    got = _rows(out["all_refs_off"], out["all_refs"])
    assert list(zip(out["all_id"].tolist(), [r[0] for r in got])) == every
    matched = [w for e in expected for w in e["matched"]]
    refs = _rows(out["way_refs_off"], out["way_refs"])
    tags = _rows(out["way_tags_off"], out["way_keys"], out["way_vals"])
    got = [
        (way_id, r[0], list(zip(*t)), meta)
        for way_id, r, t, meta in zip(
            out["way_id"].tolist(), refs, tags, out["way_meta"].tolist()
        )
    ]
    assert got == matched
