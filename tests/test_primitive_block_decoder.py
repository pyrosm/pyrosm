"""Byte-exact parity of the raw ``primitive_block_decoder`` against the protobuf path.

Every node/way/relation field the raw decoder emits must equal what protobuf decodes
from the same ``PrimitiveBlock`` bytes, on the bundled extracts.
"""

import zlib
from struct import unpack

import numpy as np
import pytest

from pyrosm import get_data
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.proto.osmformat_pb2 import PrimitiveBlock
from pyrosm.primitive_block_decoder import decode_primitive_block


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


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
    n = {
        k: []
        for k in (
            "id",
            "lat",
            "lon",
            "keys_vals",
            "version",
            "timestamp",
            "changeset",
            "uid",
            "user_sid",
            "visible",
        )
    }
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
            n["id"].append(np.cumsum(_i64(d.id)))
            n["lat"].append(np.cumsum(_i64(d.lat)))
            n["lon"].append(np.cumsum(_i64(d.lon)))
            n["keys_vals"].append(_i64(d.keys_vals))
            di = d.denseinfo
            n["version"].append(_i64(di.version))
            n["timestamp"].append(np.cumsum(_i64(di.timestamp)))
            n["changeset"].append(np.cumsum(_i64(di.changeset)))
            n["uid"].append(np.cumsum(_i64(di.uid)))
            n["user_sid"].append(np.cumsum(_i64(di.user_sid)))
            n["visible"].append(_i64(di.visible))
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

    if n["id"]:
        nodes = {
            "id": np.concatenate(n["id"]),
            "lat": np.concatenate(n["lat"]),
            "lon": np.concatenate(n["lon"]),
            "keys_vals": np.concatenate(n["keys_vals"]),
        }
        if keep_metadata:
            for k in (
                "version",
                "timestamp",
                "changeset",
                "uid",
                "user_sid",
                "visible",
            ):
                nodes[k] = np.concatenate(n[k])

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


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
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
