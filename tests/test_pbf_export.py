from pathlib import Path

import numpy as np
import pytest

from pyrosm import OSM, get_data

# A sub-bbox well inside the bundled Helsinki extent (contains streets, buildings
# and at least one relation).
CROP_BBOX = [24.9424, 60.1701, 24.9461, 60.1731]


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _read_elements(path):
    """Read a PBF block-by-block into id sets + node coordinates.

    Returns (node_ids, way_ids, relation_ids, node_coords, way_refs) where
    node_coords maps node id -> (lon, lat) in degrees and way_refs maps way id ->
    list of (absolute) node ids.
    """
    from pyrosm.pbf_export import _iter_primitive_blocks

    node_ids, way_ids, rel_ids = set(), set(), set()
    coords, way_refs = {}, {}
    for pblock in _iter_primitive_blocks(path):
        g, lo, ln = pblock.granularity, pblock.lat_offset, pblock.lon_offset
        for grp in pblock.primitivegroup:
            if len(grp.dense.id) > 0:
                ids = np.cumsum(np.array(list(grp.dense.id), dtype=np.int64))
                lats = (
                    np.cumsum(np.array(list(grp.dense.lat), dtype=np.int64)) * g + lo
                ) / 1e9
                lons = (
                    np.cumsum(np.array(list(grp.dense.lon), dtype=np.int64)) * g + ln
                ) / 1e9
                for i, nid in enumerate(ids):
                    node_ids.add(int(nid))
                    coords[int(nid)] = (lons[i], lats[i])
            for node in grp.nodes:
                node_ids.add(node.id)
                coords[node.id] = ((node.lon * g + ln) / 1e9, (node.lat * g + lo) / 1e9)
            for way in grp.ways:
                way_ids.add(way.id)
                way_refs[way.id] = [
                    int(r) for r in np.cumsum(np.array(list(way.refs), dtype=np.int64))
                ]
            for rel in grp.relations:
                rel_ids.add(rel.id)
    return node_ids, way_ids, rel_ids, coords, way_refs


def _expected_selection(path, bbox):
    """Independently compute the complete-ways crop selection from the source."""
    node_ids, way_ids, rel_ids, coords, way_refs = _read_elements(path)
    xmin, ymin, xmax, ymax = bbox
    nodes_in = {
        nid for nid, (x, y) in coords.items() if xmin <= x <= xmax and ymin <= y <= ymax
    }
    expected_ways = {
        wid for wid, refs in way_refs.items() if any(n in nodes_in for n in refs)
    }
    expected_nodes = set(nodes_in)
    for wid in expected_ways:
        expected_nodes.update(way_refs[wid])
    # A way ref can point to a node not present in this (already clipped) extract;
    # the crop can only write nodes it actually has.
    expected_nodes &= node_ids
    return expected_ways, expected_nodes, nodes_in


def test_to_pbf_roundtrip_readable(helsinki_pbf):
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf()
    try:
        assert Path(out).exists()

        cropped = OSM(out)
        net = cropped.get_network()
        assert net is not None and len(net) > 0

        buildings = cropped.get_buildings()
        assert buildings is not None and len(buildings) > 0

        # Reading the cropped file gives the same network as reading the source
        # with the same bounding box.
        ref = OSM(helsinki_pbf, bounding_box=CROP_BBOX).get_network()
        assert len(net) == len(ref)
    finally:
        Path(out).unlink()


def test_to_pbf_exact_selection_contract(helsinki_pbf):
    expected_ways, expected_nodes, nodes_in = _expected_selection(
        helsinki_pbf, CROP_BBOX
    )
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf()
    try:
        node_ids, way_ids, rel_ids, _, _ = _read_elements(out)
        assert way_ids == expected_ways
        # Exact complete-ways guarantee: the cropped node set is precisely the
        # in-bbox nodes plus all refs of the kept ways (intersected with the
        # nodes actually present in the source extract).
        assert node_ids == expected_nodes

        # Every in-bbox node is retained.
        assert nodes_in <= node_ids
    finally:
        Path(out).unlink()


def test_to_pbf_relation_selection(helsinki_pbf):
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_keep = osm.to_pbf(keep_relations=True)
    out_drop = osm.to_pbf(keep_relations=False)
    try:
        _, _, rel_keep, _, _ = _read_elements(out_keep)
        _, _, rel_drop, _, _ = _read_elements(out_drop)
        assert len(rel_keep) > 0
        assert len(rel_drop) == 0
        # Dropping relations does not affect nodes/ways.
        nodes_k, ways_k, _, _, _ = _read_elements(out_keep)
        nodes_d, ways_d, _, _, _ = _read_elements(out_drop)
        assert nodes_k == nodes_d
        assert ways_k == ways_d
    finally:
        Path(out_keep).unlink()
        Path(out_drop).unlink()


def test_to_pbf_coordinate_fidelity(helsinki_pbf):
    _, _, _, src_coords, _ = _read_elements(helsinki_pbf)
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf()
    try:
        _, _, _, crop_coords, _ = _read_elements(out)
        max_err = 0.0
        for nid, (x, y) in crop_coords.items():
            ox, oy = src_coords[nid]
            max_err = max(max_err, abs(ox - x), abs(oy - y))
        # Re-encoding stays in the raw integer grid, so coordinates are exact;
        # allow ~1 cm just in case the grid offset/granularity differs per block.
        assert max_err < 1e-7
    finally:
        Path(out).unlink()


def test_to_pbf_given_path(helsinki_pbf, tmp_path):
    target = str(tmp_path / "cropped.osm.pbf")
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf(target)
    assert out == target
    assert Path(target).exists()
    assert OSM(out).get_network() is not None


def test_to_pbf_requires_bounding_box(helsinki_pbf):
    osm = OSM(helsinki_pbf)
    with pytest.raises(ValueError):
        osm.to_pbf()


def test_to_pbf_parallel_equals_sequential(helsinki_pbf):
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_seq = osm.to_pbf(workers=1)
    out_par = osm.to_pbf(workers=2)
    try:
        with open(out_seq, "rb") as f:
            seq_bytes = f.read()
        with open(out_par, "rb") as f:
            par_bytes = f.read()
        assert seq_bytes == par_bytes
    finally:
        Path(out_seq).unlink()
        Path(out_par).unlink()


def _write_multiblock_pbf(src_path, dst_path, replicas):
    """Replicate ``src_path``'s blocks (with id offsets) into one multi-block PBF.

    The bundled file has only a few OSMData blocks; replicating them gives enough
    blocks that the parallel path genuinely spreads several across the workers.
    """
    import zlib
    from struct import pack
    from pyrosm.pbf_export import _iter_primitive_blocks
    from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
    from pyrosm.proto.osmformat_pb2 import HeaderBlock

    def frame(out, btype, msg):
        data = msg.SerializeToString()
        blob = Blob()
        blob.raw_size = len(data)
        blob.zlib_data = zlib.compress(data)
        blob_bytes = blob.SerializeToString()
        bh = BlobHeader()
        bh.type = btype
        bh.datasize = len(blob_bytes)
        hb = bh.SerializeToString()
        out.write(pack(">L", len(hb)))
        out.write(hb)
        out.write(blob_bytes)

    blocks = list(_iter_primitive_blocks(src_path))
    with open(dst_path, "wb") as out:
        header = HeaderBlock()
        header.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
        frame(out, "OSMHeader", header)
        for r in range(replicas):
            # Offset well above real OSM id ranges so replicas don't collide.
            off = r * 200_000_000_000
            for pb in blocks:
                nb = type(pb)()
                nb.CopyFrom(pb)
                if off:
                    for g in nb.primitivegroup:
                        if len(g.dense.id) > 0:
                            g.dense.id[0] += off
                        for node in g.nodes:
                            node.id += off
                        for way in g.ways:
                            way.id += off
                            if len(way.refs) > 0:
                                way.refs[0] += off
                        for rel in g.relations:
                            rel.id += off
                            if len(rel.memids) > 0:
                                rel.memids[0] += off
                frame(out, "OSMData", nb)


def test_to_pbf_parallel_multiblock(helsinki_pbf, tmp_path):
    # 4 replicas of the bundled blocks -> enough OSMData blocks that workers=3
    # genuinely runs several blocks per worker through the persistent pool.
    big = str(tmp_path / "multiblock.osm.pbf")
    _write_multiblock_pbf(helsinki_pbf, big, replicas=4)
    osm = OSM(big, bounding_box=CROP_BBOX)
    out_seq = osm.to_pbf(workers=1)
    out_par = osm.to_pbf(workers=3)
    try:
        with open(out_seq, "rb") as f:
            seq_bytes = f.read()
        with open(out_par, "rb") as f:
            par_bytes = f.read()
        # Parallel output is byte-identical to sequential and re-reads in pyrosm.
        assert seq_bytes == par_bytes
        net = OSM(out_par).get_network()
        assert net is not None and len(net) > 0
    finally:
        Path(out_seq).unlink()
        Path(out_par).unlink()


def test_to_pbf_osmium_cross_check(helsinki_pbf):
    osmium = pytest.importorskip("osmium")
    expected_ways, expected_nodes, _ = _expected_selection(helsinki_pbf, CROP_BBOX)
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf()
    try:

        class Counter(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.nodes = self.ways = self.relations = 0

            def node(self, n):
                self.nodes += 1

            def way(self, w):
                self.ways += 1

            def relation(self, r):
                self.relations += 1

        counter = Counter()
        counter.apply_file(out)
        assert counter.nodes == len(expected_nodes)
        assert counter.ways == len(expected_ways)
    finally:
        Path(out).unlink()


# ---------------------------------------------------------------------------
# OSM.to_pbf(compact=...) string-table compaction
# ---------------------------------------------------------------------------
def _block_string_tables(path):
    """Per-block string tables (the list of byte entries) of a PBF."""
    from pyrosm.pbf_export import _iter_primitive_blocks

    return [list(pb.stringtable.s) for pb in _iter_primitive_blocks(path)]


def test_to_pbf_compact_defaults_to_unchanged_output(helsinki_pbf):
    # compact defaults to False; passing it explicitly must not change the bytes.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_default = osm.to_pbf()
    out_false = osm.to_pbf(compact=False)
    try:
        with open(out_default, "rb") as f:
            default_bytes = f.read()
        with open(out_false, "rb") as f:
            false_bytes = f.read()
        assert default_bytes == false_bytes
    finally:
        Path(out_default).unlink()
        Path(out_false).unlink()


def test_to_pbf_default_copies_source_string_tables(helsinki_pbf):
    # The default path copies each source block's string table verbatim. Checking
    # this on the decompressed blocks guards the default-path *encoding*, which the
    # behavioral round-trip tests (comparing only parsed data) would not catch.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf(compact=False)
    try:
        source_tables = {tuple(t) for t in _block_string_tables(helsinki_pbf)}
        for table in _block_string_tables(out):
            assert tuple(table) in source_tables
    finally:
        Path(out).unlink()


def test_to_pbf_compact_data_is_identical(helsinki_pbf):
    # compact=True changes only the encoding -> the parsed data is unchanged.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_false = osm.to_pbf(compact=False)
    out_true = osm.to_pbf(compact=True)
    try:
        for getter in ("get_network", "get_buildings"):
            a = getattr(OSM(out_false), getter)()
            b = getattr(OSM(out_true), getter)()
            assert (a is None) == (b is None)
            if a is None:
                continue
            a = a.sort_values("id").reset_index(drop=True)
            b = b.sort_values("id").reset_index(drop=True)
            assert list(a["id"]) == list(b["id"])
            assert (a.geometry.to_wkb() == b.geometry.to_wkb()).all()
            assert set(a.columns) == set(b.columns)
            for col in a.columns:
                if col == "geometry":
                    continue
                # astype(object) first so NaN -> None (NaN != NaN would break ==).
                left = a[col].astype(object).where(a[col].notna(), None).tolist()
                right = b[col].astype(object).where(b[col].notna(), None).tolist()
                assert left == right, col
    finally:
        Path(out_false).unlink()
        Path(out_true).unlink()


def test_to_pbf_compact_shrinks_string_tables(helsinki_pbf):
    # compact=True prunes each block's table to a subset-or-equal of the default's,
    # and the total entry count (and file size) strictly shrinks on this fixture.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_false = osm.to_pbf(compact=False)
    out_true = osm.to_pbf(compact=True)
    try:
        tables_false = _block_string_tables(out_false)
        tables_true = _block_string_tables(out_true)
        assert len(tables_false) == len(tables_true)  # same blocks, pruned tables
        total_false = total_true = 0
        for default_table, compact_table in zip(tables_false, tables_true):
            assert set(compact_table) <= set(default_table)
            total_false += len(default_table)
            total_true += len(compact_table)
        assert total_true < total_false
        assert Path(out_true).stat().st_size < Path(out_false).stat().st_size
    finally:
        Path(out_false).unlink()
        Path(out_true).unlink()


def test_to_pbf_compact_parallel_multiblock(helsinki_pbf, tmp_path):
    # compact=True output is byte-identical between the sequential and the parallel
    # (multi-block) write paths -> the pruned table order is deterministic.
    big = str(tmp_path / "multiblock.osm.pbf")
    _write_multiblock_pbf(helsinki_pbf, big, replicas=4)
    osm = OSM(big, bounding_box=CROP_BBOX)
    out_seq = osm.to_pbf(compact=True, workers=1)
    out_par = osm.to_pbf(compact=True, workers=3)
    try:
        with open(out_seq, "rb") as f:
            seq_bytes = f.read()
        with open(out_par, "rb") as f:
            par_bytes = f.read()
        assert seq_bytes == par_bytes
        assert OSM(out_par).get_network() is not None
    finally:
        Path(out_seq).unlink()
        Path(out_par).unlink()


def test_to_pbf_compact_preserves_user_metadata(tmp_path):
    # Editor metadata stored as a string index (the user name) must survive the
    # remap. The bundled fixtures are anonymized (no user names), so write a small
    # fixture whose out-of-box elements carry unique strings: cropping drops them,
    # so compaction actually prunes and the remap runs over the kept elements.
    osmium = pytest.importorskip("osmium")
    src = str(tmp_path / "users.osm.pbf")
    writer = osmium.SimpleWriter(src)
    inside = list(range(1, 11))
    for i in inside:
        writer.add_node(
            osmium.osm.mutable.Node(
                id=i,
                location=(24.9445 + i * 0.0001, 60.1708 + i * 0.0001),
                tags={"name": f"inside{i}", "addr:street": "Shared Street"},
                user="inside_mapper",
                uid=111,
                version=2,
                timestamp="2020-05-01T00:00:00Z",
            )
        )
    for i in range(100, 130):  # far away, unique strings -> dropped then pruned
        writer.add_node(
            osmium.osm.mutable.Node(
                id=i,
                location=(25.8 + i * 0.001, 61.5 + i * 0.001),
                tags={"name": f"outside_unique_{i}", f"junkkey_{i}": f"junkval_{i}"},
                user=f"outside_mapper_{i}",
                uid=2000 + i,
                version=1,
                timestamp="2019-01-01T00:00:00Z",
            )
        )
    writer.add_way(
        osmium.osm.mutable.Way(
            id=1,
            nodes=inside,
            tags={"highway": "residential", "name": "Kept Way"},
            user="way_mapper",
            uid=42,
            version=3,
            timestamp="2021-06-01T00:00:00Z",
        )
    )
    writer.close()

    bbox = [24.94, 60.16, 24.96, 60.18]  # covers only the inside cluster
    out_false = OSM(src, bounding_box=bbox).to_pbf(
        str(tmp_path / "f.pbf"), compact=False
    )
    out_true = OSM(src, bounding_box=bbox).to_pbf(str(tmp_path / "t.pbf"), compact=True)

    def meta(path):
        result = {}
        for entity, kind in ((osmium.osm.NODE, "n"), (osmium.osm.WAY, "w")):
            for o in osmium.FileProcessor(path).with_filter(
                osmium.filter.EntityFilter(entity)
            ):
                result[(kind, o.id)] = (o.user, o.uid, o.version)
        return result

    meta_false, meta_true = meta(out_false), meta(out_true)
    assert (
        Path(out_true).stat().st_size < Path(out_false).stat().st_size
    )  # compaction ran
    assert meta_false == meta_true  # user/uid/version identical, incl. user names
    assert meta_true[("w", 1)][0] == "way_mapper"
    assert meta_true[("n", 1)][0] == "inside_mapper"


# ---------------------------------------------------------------------------
# OSM.to_pbf(repack=...) canonical block re-packing
# ---------------------------------------------------------------------------
META_BBOX = [24.94, 60.16, 24.96, 60.18]


def _block_fill(path):
    """(number of OSMData blocks, total node count) of a PBF."""
    from pyrosm.pbf_export import _iter_primitive_blocks

    nblocks = nnodes = 0
    for pb in _iter_primitive_blocks(path):
        nblocks += 1
        for g in pb.primitivegroup:
            nnodes += len(g.dense.id) + len(g.nodes)
    return nblocks, nnodes


def _make_metadata_fixture(path, with_relation=False):
    """Write a small PBF with real editor metadata via osmium. Out-of-box elements
    carry unique strings, so a crop drops them and re-pack/pruning actually runs."""
    import osmium

    writer = osmium.SimpleWriter(path)
    inside = list(range(1, 11))
    for i in inside:
        writer.add_node(
            osmium.osm.mutable.Node(
                id=i,
                location=(24.9445 + i * 0.0001, 60.1708 + i * 0.0001),
                tags={"name": f"inside{i}", "addr:street": "Shared Street"},
                user="inside_mapper",
                uid=111,
                version=2,
                changeset=555,
                timestamp="2020-05-01T00:00:00Z",
            )
        )
    for i in range(100, 130):  # far away, unique strings -> dropped then pruned
        writer.add_node(
            osmium.osm.mutable.Node(
                id=i,
                location=(25.8 + i * 0.001, 61.5 + i * 0.001),
                tags={"name": f"outside_{i}", f"junkkey_{i}": f"junkval_{i}"},
                user=f"outside_mapper_{i}",
                uid=2000 + i,
                version=1,
                changeset=900 + i,
                timestamp="2019-01-01T00:00:00Z",
            )
        )
    writer.add_way(
        osmium.osm.mutable.Way(
            id=1,
            nodes=inside,
            tags={"highway": "residential", "name": "Kept Way"},
            user="way_mapper",
            uid=42,
            version=3,
            changeset=777,
            timestamp="2021-06-01T00:00:00Z",
        )
    )
    if with_relation:
        writer.add_relation(
            osmium.osm.mutable.Relation(
                id=1,
                members=[("w", 1, "outer"), ("n", 1, "label")],
                tags={"type": "multipolygon", "name": "Kept Rel"},
                user="rel_mapper",
                uid=7,
                version=4,
                changeset=888,
                timestamp="2022-02-02T00:00:00Z",
            )
        )
    writer.close()


def test_to_pbf_repack_defaults_to_unchanged_output(helsinki_pbf):
    # repack defaults to False; passing it explicitly must not change the bytes.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_default = osm.to_pbf()
    out_false = osm.to_pbf(repack=False)
    try:
        with open(out_default, "rb") as f:
            a = f.read()
        with open(out_false, "rb") as f:
            b = f.read()
        assert a == b
    finally:
        Path(out_default).unlink()
        Path(out_false).unlink()


def test_to_pbf_repack_data_is_identical(helsinki_pbf):
    # repack=True re-encodes into canonical blocks; the parsed data is unchanged.
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_off = osm.to_pbf(repack=False)
    out_on = osm.to_pbf(repack=True)
    try:
        for getter in ("get_network", "get_buildings"):
            a = getattr(OSM(out_off), getter)()
            b = getattr(OSM(out_on), getter)()
            assert (a is None) == (b is None)
            if a is None:
                continue
            a = a.sort_values("id").reset_index(drop=True)
            b = b.sort_values("id").reset_index(drop=True)
            assert list(a["id"]) == list(b["id"])
            assert (a.geometry.to_wkb() == b.geometry.to_wkb()).all()
            assert set(a.columns) == set(b.columns)
            for col in a.columns:
                if col == "geometry":
                    continue
                left = a[col].astype(object).where(a[col].notna(), None).tolist()
                right = b[col].astype(object).where(b[col].notna(), None).tolist()
                assert left == right, col
    finally:
        Path(out_off).unlink()
        Path(out_on).unlink()


def test_to_pbf_repack_is_smaller_and_denser(helsinki_pbf):
    osmium = pytest.importorskip("osmium")
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out_off = osm.to_pbf(repack=False)
    out_on = osm.to_pbf(repack=True)
    try:
        assert Path(out_on).stat().st_size < Path(out_off).stat().st_size
        nb_off, nn_off = _block_fill(out_off)
        nb_on, nn_on = _block_fill(out_on)
        assert nn_off == nn_on  # same node count
        assert nb_on <= nb_off  # re-pack uses no more blocks ...
        assert (nn_on / nb_on) >= (nn_off / nb_off)  # ... at >= the average fill

        counter = [0]

        class _H(osmium.SimpleHandler):
            def node(self, n):
                counter[0] += 1

        _H().apply_file(out_on)  # re-readable by osmium
        assert counter[0] > 0
    finally:
        Path(out_off).unlink()
        Path(out_on).unlink()


def test_to_pbf_repack_coordinates_exact(helsinki_pbf):
    _, _, _, src_coords, _ = _read_elements(helsinki_pbf)
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf(repack=True)
    try:
        _, _, _, crop_coords, _ = _read_elements(out)
        max_err = 0.0
        for nid, (x, y) in crop_coords.items():
            ox, oy = src_coords[nid]
            max_err = max(max_err, abs(ox - x), abs(oy - y))
        assert max_err < 1e-7
    finally:
        Path(out).unlink()


def test_to_pbf_repack_preserves_metadata(tmp_path):
    osmium = pytest.importorskip("osmium")
    src = str(tmp_path / "users.osm.pbf")
    _make_metadata_fixture(src)
    out_off = OSM(src, bounding_box=META_BBOX).to_pbf(
        str(tmp_path / "off.pbf"), repack=False
    )
    out_on = OSM(src, bounding_box=META_BBOX).to_pbf(
        str(tmp_path / "on.pbf"), repack=True
    )

    def meta(path):
        result = {}
        for entity, kind in ((osmium.osm.NODE, "n"), (osmium.osm.WAY, "w")):
            for o in osmium.FileProcessor(path).with_filter(
                osmium.filter.EntityFilter(entity)
            ):
                result[(kind, o.id)] = (
                    o.user,
                    o.uid,
                    o.version,
                    str(o.timestamp),
                    o.changeset,
                    o.visible,
                )
        return result

    m_off, m_on = meta(out_off), meta(out_on)
    assert (
        Path(out_on).stat().st_size < Path(out_off).stat().st_size
    )  # re-pack actually ran
    assert m_off == m_on
    assert m_on[("w", 1)][0] == "way_mapper"
    assert m_on[("n", 1)][0] == "inside_mapper"


def test_to_pbf_repack_preserves_relations(tmp_path):
    osmium = pytest.importorskip("osmium")
    src = str(tmp_path / "rel.osm.pbf")
    _make_metadata_fixture(src, with_relation=True)
    out_off = OSM(src, bounding_box=META_BBOX).to_pbf(
        str(tmp_path / "off.pbf"), repack=False
    )
    out_on = OSM(src, bounding_box=META_BBOX).to_pbf(
        str(tmp_path / "on.pbf"), repack=True
    )

    def rels(path):
        result = {}
        for r in osmium.FileProcessor(path).with_filter(
            osmium.filter.EntityFilter(osmium.osm.RELATION)
        ):
            result[r.id] = (
                [(m.type, m.ref, m.role) for m in r.members],
                {t.k: t.v for t in r.tags},
                (r.user, r.uid, r.version, str(r.timestamp), r.changeset, r.visible),
            )
        return result

    assert 1 in rels(out_on)
    assert rels(out_off) == rels(out_on)


def test_to_pbf_repack_deterministic_and_overrides_compact(helsinki_pbf):
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    a = osm.to_pbf(repack=True)
    b = osm.to_pbf(repack=True)
    c = osm.to_pbf(repack=True, compact=True)
    try:
        with open(a, "rb") as f:
            ab = f.read()
        with open(b, "rb") as f:
            bb = f.read()
        with open(c, "rb") as f:
            cb = f.read()
        assert ab == bb  # deterministic across runs
        assert ab == cb  # compact ignored when repack=True
    finally:
        Path(a).unlink()
        Path(b).unlink()
        Path(c).unlink()


def _frame_pbf_blob(out, btype, msg):
    import zlib as _zlib
    from struct import pack as _pack
    from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob

    data = msg.SerializeToString()
    blob = Blob()
    blob.raw_size = len(data)
    blob.zlib_data = _zlib.compress(data)
    bb = blob.SerializeToString()
    bh = BlobHeader()
    bh.type = btype
    bh.datasize = len(bb)
    hb = bh.SerializeToString()
    out.write(_pack(">L", len(hb)))
    out.write(hb)
    out.write(bb)


@pytest.mark.parametrize(
    "field, value",
    [
        ("granularity", 1000),
        ("lat_offset", 100),
        ("lon_offset", 100),
        ("date_granularity", 2000),  # guards timestamp fidelity
    ],
)
def test_to_pbf_repack_rejects_nonstandard_grid(tmp_path, field, value):
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock, HeaderBlock

    src = str(tmp_path / "nonstd.osm.pbf")
    blk = PrimitiveBlock()
    blk.granularity = 100
    blk.lat_offset = 0
    blk.lon_offset = 0
    blk.date_granularity = 1000
    setattr(blk, field, value)  # exactly one grid field non-standard
    blk.stringtable.s.append(b"")
    dense = blk.primitivegroup.add().dense
    dense.id.extend([1, 1])
    dense.lat.extend([600000, 0])
    dense.lon.extend([250000, 0])
    hdr = HeaderBlock()
    hdr.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
    with open(src, "wb") as out:
        _frame_pbf_blob(out, "OSMHeader", hdr)
        _frame_pbf_blob(out, "OSMData", blk)
    with pytest.raises(ValueError):
        OSM(src, bounding_box=[-1, -1, 2, 2]).to_pbf(
            str(tmp_path / "out.pbf"), repack=True
        )


def test_to_pbf_repack_rejects_mixed_metadata(tmp_path):
    # Two dense blocks, one carrying DenseInfo and one without: re-pack cannot merge
    # them into a canonical block faithfully, so it must raise (not crash or drop).
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock, HeaderBlock

    def dense_block(start_id, with_meta):
        blk = PrimitiveBlock()
        blk.granularity = 100
        blk.date_granularity = 1000
        blk.stringtable.s.append(b"")
        d = blk.primitivegroup.add().dense
        d.id.extend([start_id, 1])  # ids start_id, start_id+1 (delta encoded)
        d.lat.extend([600000, 100])
        d.lon.extend([250000, 100])
        if with_meta:
            d.denseinfo.version.extend([1, 1])
            d.denseinfo.timestamp.extend([0, 0])
        return blk

    src = str(tmp_path / "mixed.osm.pbf")
    hdr = HeaderBlock()
    hdr.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
    with open(src, "wb") as out:
        _frame_pbf_blob(out, "OSMHeader", hdr)
        _frame_pbf_blob(out, "OSMData", dense_block(1, with_meta=True))
        _frame_pbf_blob(out, "OSMData", dense_block(100, with_meta=False))
    with pytest.raises(ValueError):
        OSM(src, bounding_box=[-1, -1, 2, 2]).to_pbf(
            str(tmp_path / "out.pbf"), repack=True
        )


# ---------------------------------------------------------------------------
# OSM.write_pbf (issue #285)
# ---------------------------------------------------------------------------
import geopandas as gpd  # noqa: E402
from shapely.geometry import (  # noqa: E402
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)


def _way_tags(path):
    """Map way id -> {tag key: value} read straight from a written PBF."""
    from pyrosm.pbf_export import _iter_primitive_blocks

    out = {}
    for pblock in _iter_primitive_blocks(path):
        st = [s.decode("utf-8", "replace") for s in pblock.stringtable.s]
        for grp in pblock.primitivegroup:
            for way in grp.ways:
                out[way.id] = {st[k]: st[v] for k, v in zip(way.keys, way.vals)}
    return out


def test_write_pbf_roundtrip_edit(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    edges = osm.get_network("driving").copy()
    edges["maxspeed"] = edges["maxspeed"].fillna("50")
    edges["travel_time"] = (edges["length"] / 10.0).round(2)

    out = str(tmp_path / "edited.osm.pbf")
    osm.write_pbf(edges, out)

    re = OSM(out).get_network("driving")
    assert len(re) == len(edges)
    # maxspeed is now fully populated (the edit took effect)...
    assert re["maxspeed"].notna().all()
    # ...and the brand-new travel_time tag is present on every edited way.
    has_tt = re["tags"].dropna().astype(str).str.contains("travel_time")
    assert has_tt.sum() == len(re)
    # `visible` (a structural flag) must NOT be written as a tag.
    raw = _way_tags(out)
    assert all("visible" not in tags for tags in raw.values())


def test_write_pbf_whole_dataset_preserved(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    out = str(tmp_path / "whole.osm.pbf")
    # Pass only the network, but buildings (untouched) must survive the write.
    osm.write_pbf(osm.get_network(), out)
    buildings = OSM(out).get_buildings()
    assert buildings is not None and len(buildings) > 0


def test_write_pbf_untouched_poi_tags_survive(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    pois = osm.get_pois()
    node_pois = pois[pois["osm_type"] == "node"]
    # pick a POI node carrying a 'name' tag
    named = node_pois[node_pois["name"].notna()]
    poi_id = int(named.iloc[0]["id"])
    poi_name = named.iloc[0]["name"]

    out = str(tmp_path / "poi.osm.pbf")
    # Write only the network; the untouched POI node must keep its tags.
    osm.write_pbf(osm.get_network(), out)

    from pyrosm.pbf_export import _iter_primitive_blocks

    found = None
    for pblock in _iter_primitive_blocks(out):
        st = [s.decode("utf-8", "replace") for s in pblock.stringtable.s]
        for grp in pblock.primitivegroup:
            if len(grp.dense.id) == 0:
                continue
            ids = np.cumsum(np.array(list(grp.dense.id), dtype=np.int64))
            # split keys_vals into per-node (key,val) segments
            kv = list(grp.dense.keys_vals)
            segments, cur = [], {}
            it = iter(kv)
            for k in it:
                if k == 0:
                    segments.append(cur)
                    cur = {}
                else:
                    cur[st[k]] = st[next(it)]
            for nid, seg in zip(ids, segments):
                if int(nid) == poi_id:
                    found = seg
    assert found is not None
    assert found.get("name") == poi_name


def test_write_pbf_coordinate_fidelity(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    out = str(tmp_path / "coords.osm.pbf")
    osm.write_pbf(osm.get_network(), out)

    _, _, _, src_coords, _ = _read_elements(helsinki_pbf)
    _, _, _, out_coords, _ = _read_elements(out)
    common = set(src_coords) & set(out_coords)
    assert len(common) > 0
    max_err = max(
        max(
            abs(src_coords[n][0] - out_coords[n][0]),
            abs(src_coords[n][1] - out_coords[n][1]),
        )
        for n in common
    )
    assert max_err < 1e-7


def test_write_pbf_new_linestring(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    line = [(24.94, 60.17), (24.945, 60.172), (24.95, 60.17)]
    new = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [10**18], "highway": ["footway"], "tags": [None]},
        geometry=[LineString(line)],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "new_line.osm.pbf")
    osm.write_pbf([osm.get_network(), new], out)

    raw = _way_tags(out)
    synth = [
        wid for wid, tags in raw.items() if wid < 0 and tags.get("highway") == "footway"
    ]
    assert len(synth) == 1  # synthesized way carries a negative id
    assert OSM(out).get_network() is not None

    # The synthesized way's geometry matches the input LineString (exact grid).
    _, _, _, coords, way_refs = _read_elements(out)
    refs = way_refs[synth[0]]
    assert len(refs) == len(line)
    for (in_x, in_y), ref in zip(line, refs):
        out_x, out_y = coords[ref]
        assert abs(out_x - in_x) < 1e-7
        assert abs(out_y - in_y) < 1e-7


def test_write_pbf_new_point_and_polygon(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    pt = gpd.GeoDataFrame(
        {"osm_type": ["node"], "id": [10**18], "amenity": ["bench"], "tags": [None]},
        geometry=[Point(24.945, 60.171)],
        crs="EPSG:4326",
    )
    poly = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [10**18 + 1], "building": ["yes"], "tags": [None]},
        geometry=[
            Polygon([(24.94, 60.17), (24.941, 60.17), (24.941, 60.171), (24.94, 60.17)])
        ],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "new_pt_poly.osm.pbf")
    osm.write_pbf([pt, poly], out)

    # The synthesized polygon is a negative-id, closed way tagged building=yes.
    raw = _way_tags(out)
    _, _, _, _, way_refs = _read_elements(out)
    synth_poly = [
        wid for wid, tags in raw.items() if wid < 0 and tags.get("building") == "yes"
    ]
    assert len(synth_poly) == 1
    refs = way_refs[synth_poly[0]]
    assert refs[0] == refs[-1]  # closed ring

    re = OSM(out)
    pois = re.get_pois()
    assert pois is not None and (pois["amenity"] == "bench").any()
    buildings = re.get_buildings()
    assert buildings is not None and len(buildings) > 0


@pytest.mark.parametrize(
    "geom",
    [
        MultiLineString(
            [[(24.94, 60.17), (24.95, 60.17)], [(24.96, 60.17), (24.97, 60.17)]]
        ),
        Polygon(
            [(24.94, 60.17), (24.95, 60.17), (24.95, 60.18), (24.94, 60.17)],
            [[(24.943, 60.172), (24.946, 60.172), (24.946, 60.175), (24.943, 60.172)]],
        ),
        MultiPolygon(
            [
                Polygon(
                    [(24.94, 60.17), (24.95, 60.17), (24.95, 60.18), (24.94, 60.17)]
                ),
                Polygon(
                    [(24.96, 60.17), (24.97, 60.17), (24.97, 60.18), (24.96, 60.17)]
                ),
            ]
        ),
        GeometryCollection(
            [Point(24.94, 60.17), LineString([(24.95, 60.17), (24.96, 60.17)])]
        ),
    ],
)
def test_write_pbf_unsupported_geometry_raises(helsinki_pbf, tmp_path, geom):
    osm = OSM(helsinki_pbf)
    bad = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [10**18], "highway": ["path"], "tags": [None]},
        geometry=[geom],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError):
        osm.write_pbf(bad, str(tmp_path / "bad.osm.pbf"))


def test_write_pbf_api(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    edges = osm.get_network()
    single = str(tmp_path / "single.osm.pbf")
    assert osm.write_pbf(edges, single) == single
    assert Path(single).exists()

    as_list = str(tmp_path / "list.osm.pbf")
    osm.write_pbf([edges], as_list)
    assert OSM(as_list).get_network() is not None

    with pytest.raises(ValueError):
        osm.write_pbf("not a geodataframe", str(tmp_path / "z.osm.pbf"))


def test_write_pbf_osmium_cross_check(helsinki_pbf, tmp_path):
    osmium = pytest.importorskip("osmium")
    osm = OSM(helsinki_pbf)
    out = str(tmp_path / "osmium.osm.pbf")
    osm.write_pbf(osm.get_network(), out)
    expected_nodes = len(osm._node_coordinates)
    expected_ways = len(osm._way_records)
    expected_relations = len(osm._relations["id"]) if "id" in osm._relations else 0

    class Counter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.nodes = self.ways = self.relations = 0

        def node(self, n):
            self.nodes += 1

        def way(self, w):
            self.ways += 1

        def relation(self, r):
            self.relations += 1

    counter = Counter()
    counter.apply_file(out)
    assert counter.nodes == expected_nodes
    assert counter.ways == expected_ways
    assert counter.relations == expected_relations


def test_write_pbf_r5py_routable(helsinki_pbf, tmp_path):
    r5py = pytest.importorskip("r5py")

    modes = [("driving", r5py.TransportMode.CAR), ("walking", r5py.TransportMode.WALK)]
    for mode_name, transport_mode in modes:
        osm = OSM(helsinki_pbf)
        edges = osm.get_network(mode_name).copy()
        edges["maxspeed"] = edges["maxspeed"].fillna("30")
        out = str(tmp_path / ("r5_%s.osm.pbf" % mode_name))
        osm.write_pbf(edges, out)

        # R5 builds a routable street network from the exported PBF (strict,
        # independent OSM-PBF consumer); construction must not raise.
        transport_network = r5py.TransportNetwork(out)

        xmin, ymin, xmax, ymax = osm._data_bounding_box.bounds
        points = gpd.GeoDataFrame(
            {"id": [0, 1]},
            geometry=[
                Point(xmin + (xmax - xmin) * 0.4, ymin + (ymax - ymin) * 0.4),
                Point(xmin + (xmax - xmin) * 0.6, ymin + (ymax - ymin) * 0.6),
            ],
            crs="EPSG:4326",
        )
        # TravelTimeMatrix computes on construction and is a GeoDataFrame.
        travel_times = r5py.TravelTimeMatrix(
            transport_network,
            origins=points,
            destinations=points,
            snap_to_network=True,
            transport_modes=[transport_mode],
        )
        # The exported network is routable between two DISTINCT locations
        # (off-diagonal entry), not just trivially origin==destination.
        off_diagonal = travel_times[travel_times["from_id"] != travel_times["to_id"]]
        assert off_diagonal["travel_time"].notna().any()


def test_write_pbf_tag_str_normalization():
    from pyrosm.pbf_writer import _tag_str

    assert _tag_str(True) == "yes"
    assert _tag_str(False) == "no"
    assert _tag_str(50.0) == "50"  # integer-valued float (pandas NaN-widened int)
    assert _tag_str(150.4) == "150.4"
    assert _tag_str(2) == "2"
    assert _tag_str("30") == "30"


def test_write_pbf_metadata_preserved(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    osm._read_pbf()
    # a source node id and its cached changeset
    nid, rec = next(iter(osm._node_coordinates.items()))
    src_changeset = int(rec.get("changeset") or 0)

    out = str(tmp_path / "meta.osm.pbf")
    osm.write_pbf(osm.get_network(), out)

    # changeset carried through; no 'visible' tag emitted on any way.
    from pyrosm.pbf_export import _iter_primitive_blocks

    found = None
    for pblock in _iter_primitive_blocks(out):
        for grp in pblock.primitivegroup:
            if len(grp.dense.id) > 0:
                ids = np.cumsum(np.array(list(grp.dense.id), dtype=np.int64))
                changesets = np.cumsum(
                    np.array(list(grp.dense.denseinfo.changeset), dtype=np.int64)
                )
                idx = np.where(ids == nid)[0]
                if len(idx):
                    found = int(changesets[idx[0]])
    assert found == src_changeset


def test_write_pbf_row_tags_strips_members():
    import pandas as pd
    from pyrosm.pbf_writer import _row_tags

    row = pd.Series(
        {
            "osm_type": "relation",
            "id": 1,
            "route": "bicycle",
            "tags": {"members": [{"member_id": 1}], "network": "ncn"},
            "geometry": None,
        }
    )
    tags = _row_tags(row, "geometry")
    assert "members" not in tags  # relation-member metadata is not an OSM tag
    assert tags.get("network") == "ncn"
    assert tags.get("route") == "bicycle"


def test_write_pbf_new_linestring_3d(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    # 3D coordinates (lon, lat, z) must be accepted; z is dropped.
    new = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [10**18], "highway": ["footway"], "tags": [None]},
        geometry=[LineString([(24.94, 60.17, 5.0), (24.95, 60.17, 6.0)])],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "line3d.osm.pbf")
    osm.write_pbf([osm.get_network(), new], out)
    raw = _way_tags(out)
    assert any(
        wid < 0 and tags.get("highway") == "footway" for wid, tags in raw.items()
    )


def test_write_pbf_relation_members_absolute(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    osm._read_pbf()
    rels = osm._relations
    rid0 = int(rels["id"][0])
    src_members = sorted(int(m) for m in rels["members"][0]["member_id"])

    out = str(tmp_path / "rel.osm.pbf")
    osm.write_pbf(osm.get_network(), out)

    from pyrosm.pbf_export import _iter_primitive_blocks

    found = None
    for pblock in _iter_primitive_blocks(out):
        for grp in pblock.primitivegroup:
            for rel in grp.relations:
                if rel.id == rid0:
                    memids = np.cumsum(np.array(list(rel.memids), dtype=np.int64))
                    found = sorted(int(m) for m in memids)
    # Written member ids match the source's absolute member ids exactly.
    assert found == src_members


def test_write_pbf_osmium_counts_additions(helsinki_pbf, tmp_path):
    osmium = pytest.importorskip("osmium")
    osm = OSM(helsinki_pbf)
    osm._read_pbf()
    base_nodes = len(osm._node_coordinates)
    base_ways = len(osm._way_records)

    new = gpd.GeoDataFrame(
        {
            "osm_type": ["node", "way"],
            "id": [10**18, 10**18 + 1],
            "amenity": ["bench", None],
            "highway": [None, "footway"],
            "tags": [None, None],
        },
        geometry=[
            Point(24.945, 60.171),
            LineString([(24.94, 60.17), (24.95, 60.17)]),
        ],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "add.osm.pbf")
    osm.write_pbf([osm.get_network(), new], out)

    class Counter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.nodes = self.ways = 0

        def node(self, n):
            self.nodes += 1

        def way(self, w):
            self.ways += 1

    counter = Counter()
    counter.apply_file(out)
    # 1 new Point node + 2 new LineString vertices; 1 new way.
    assert counter.nodes == base_nodes + 3
    assert counter.ways == base_ways + 1


# ---------------------------------------------------------------------------
# pbf_writer unit coverage (issue #285 modularization)
# ---------------------------------------------------------------------------
def test_pbf_writer_tag_helpers():
    import pandas as pd
    from pyrosm.pbf_writer import _row_tags, _record_tags, _tag_key, _is_missing

    # JSON-string tags: members + empty key stripped, real tags kept.
    row = pd.Series(
        {
            "osm_type": "way",
            "id": 1,
            "highway": "residential",
            "tags": '{"members":[1],"surface":"asphalt","":"x"}',
            "geometry": None,
        }
    )
    tags = _row_tags(row, "geometry")
    assert tags["highway"] == "residential" and tags["surface"] == "asphalt"
    assert "members" not in tags and "" not in tags

    # Malformed JSON tags are ignored (no crash).
    row2 = pd.Series(
        {
            "osm_type": "way",
            "id": 1,
            "highway": "path",
            "tags": "nope",
            "geometry": None,
        }
    )
    assert _row_tags(row2, "geometry")["highway"] == "path"

    assert _tag_key("") is None
    assert _tag_key(5) == "5"

    # _is_missing handles arrays/dicts without raising.
    assert _is_missing(np.array([1, 2])) is False
    assert _is_missing({"a": 1}) is False
    assert _is_missing(float("nan")) is True
    assert _is_missing(None) is True

    # _record_tags: structural/members/visible/empty stripped, extra dict merged.
    rec = {
        "id": 1,
        "version": 2,
        "nodes": [1, 2],
        "highway": "path",
        "tags": {"surface": "gravel", "visible": False, "": "skip"},
    }
    assert _record_tags(rec) == {"highway": "path", "surface": "gravel"}


def test_pbf_writer_frame_helpers():
    from pyrosm.pbf_writer import _as_frames, _normalize_osm_type, _normalize_id

    with pytest.raises(ValueError):
        _as_frames("not a geodataframe")
    assert _normalize_osm_type(b"Way") == "way"
    assert _normalize_osm_type("NODE") == "node"
    assert _normalize_id(np.int64(5)) == 5
    assert _normalize_id("x") is None
    assert _normalize_id(None) is None


def test_pbf_writer_builder_geometry_errors():
    from pyrosm.pbf_writer import _RecordBuilder

    builder = _RecordBuilder()
    builder.begin_synthesis()
    with pytest.raises(ValueError):  # coordinate out of lon/lat range
        builder.add_geometry(1, Point(500.0, 0.0), None)
    with pytest.raises(ValueError):  # empty geometry
        builder.add_geometry(2, LineString(), None)
    with pytest.raises(ValueError):  # None geometry
        builder.add_geometry(3, None, None)
    with pytest.raises(ValueError):  # unsupported geometry type
        builder.add_geometry(
            4, MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]), None
        )


def test_write_pbf_reprojects_crs(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    # New geometry supplied in a projected CRS (EPSG:3857) must be reprojected.
    line = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [10**18], "highway": ["footway"], "tags": [None]},
        geometry=[LineString([(24.94, 60.17), (24.95, 60.17)])],
        crs="EPSG:4326",
    ).to_crs(3857)
    out = str(tmp_path / "crs.osm.pbf")
    osm.write_pbf([osm.get_network(), line], out)

    _, _, _, coords, way_refs = _read_elements(out)
    raw = _way_tags(out)
    synth = [w for w, t in raw.items() if w < 0 and t.get("highway") == "footway"][0]
    for ref in way_refs[synth]:
        x, y = coords[ref]
        assert 24.9 < x < 25.0 and 60.1 < y < 60.2


def test_write_pbf_matches_bytes_and_numpy_keys(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    e = osm.get_network("driving").iloc[[0]].copy()
    wid = int(e.iloc[0]["id"])
    e["osm_type"] = [b"WAY"]  # bytes + uppercase must still match the cached way
    e["maxspeed"] = ["123"]
    out = str(tmp_path / "match.osm.pbf")
    osm.write_pbf(e, out)

    raw = _way_tags(out)
    assert raw[wid].get("maxspeed") == "123"  # edit applied to the existing way
    assert all(w >= 0 for w in raw)  # no new (negative-id) way synthesized


def test_write_pbf_point_shares_line_vertex(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    shared = (24.945, 60.172)
    new = gpd.GeoDataFrame(
        {
            "osm_type": ["way", "node"],
            "id": [10**18, 10**18 + 1],
            "highway": ["footway", None],
            "amenity": [None, "bench"],
            "tags": [None, None],
        },
        geometry=[LineString([(24.94, 60.17), shared]), Point(shared)],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "share.osm.pbf")
    osm.write_pbf([osm.get_network(), new], out)

    _, _, _, coords, _ = _read_elements(out)
    # The bench Point coincides with the line's endpoint -> one shared node, so
    # only 2 new (negative-id) nodes exist, not 3.
    neg_nodes = [n for n in coords if n < 0]
    assert len(neg_nodes) == 2


def test_pbf_writer_reproject_and_normalize_helpers():
    from pyrosm.pbf_writer import _reproject_to_wgs84, _normalize_osm_type

    g_none = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(24.9, 60.1)], crs=None)
    g_proj = gpd.GeoDataFrame(
        {"a": [1]}, geometry=[Point(24.9, 60.1)], crs="EPSG:4326"
    ).to_crs(3857)
    out = _reproject_to_wgs84([g_none, g_proj])
    assert out[0].crs is None  # CRS-less frame passes through untouched
    assert out[1].crs.to_epsg() == 4326  # projected frame is reprojected
    assert _normalize_osm_type(None) is None  # non-string types pass through
    assert _normalize_osm_type(7) == 7


def test_pbf_writer_tag_edge_cases():
    import pandas as pd
    from pyrosm.pbf_writer import _row_tags, _record_tags

    # None-valued column and empty-name column are both skipped.
    row = pd.Series(
        {
            "osm_type": "way",
            "id": 1,
            "highway": "path",
            "ref": None,
            "": "x",
            "tags": None,
            "geometry": None,
        }
    )
    assert _row_tags(row, "geometry") == {"highway": "path"}

    rec = {
        "id": 1,
        "version": 2,
        "nodes": [1, 2],
        "highway": "path",
        "ref": None,
        "": "skipcol",
        "tags": {"surface": "gravel"},
    }
    assert _record_tags(rec) == {"highway": "path", "surface": "gravel"}


def test_write_pbf_edit_node_and_relation(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    osm._read_pbf()
    node_pois = osm.get_pois()
    node_pois = node_pois[node_pois["osm_type"] == "node"]
    nid = int(node_pois.iloc[0]["id"])
    rid = int(osm._relations["id"][0])

    edit = gpd.GeoDataFrame(
        {
            "osm_type": ["node", "relation"],
            "id": [nid, rid],
            "note": ["edited_node", "edited_rel"],
            "tags": [None, None],
        },
        geometry=[Point(24.94, 60.17), Point(24.95, 60.17)],
        crs="EPSG:4326",
    )
    out = str(tmp_path / "edit_nr.osm.pbf")
    osm.write_pbf([osm.get_network(), edit], out)

    from pyrosm.pbf_export import _iter_primitive_blocks

    node_note = rel_note = None
    for pblock in _iter_primitive_blocks(out):
        st = [s.decode("utf-8", "replace") for s in pblock.stringtable.s]
        for grp in pblock.primitivegroup:
            for rel in grp.relations:
                if rel.id == rid:
                    rel_note = {st[k]: st[v] for k, v in zip(rel.keys, rel.vals)}.get(
                        "note"
                    )
            if len(grp.dense.id) == 0:
                continue
            ids = np.cumsum(np.array(list(grp.dense.id), dtype=np.int64))
            kv = list(grp.dense.keys_vals)
            segments, cur, it = [], {}, iter(kv)
            for k in it:
                if k == 0:
                    segments.append(cur)
                    cur = {}
                else:
                    cur[st[k]] = st[next(it)]
            for node_id, seg in zip(ids, segments):
                if int(node_id) == nid:
                    node_note = seg.get("note")
    assert node_note == "edited_node"
    assert rel_note == "edited_rel"


def test_pbf_writer_no_relations():
    from pyrosm.pbf_writer import _RecordBuilder, _add_base_relations

    builder = _RecordBuilder()
    _add_base_relations(builder, {}, {})  # relations cache without "id" -> no-op
    assert builder.rels == []


def test_pbf_writer_node_tag_records_empty():
    from pyrosm.pbf_writer import _node_tag_records

    # A nodes cache without "id"/"tags" (or None) yields no POI tag index.
    assert _node_tag_records(None) == {}
    assert _node_tag_records({}) == {}
    assert _node_tag_records({"id": [1], "lat": [0.0]}) == {}


# ---------------------------------------------------------------------------
# OSM.write_pbf(subset_only=True) — export only selected layers (issue #348)
# ---------------------------------------------------------------------------
def _ids(gdf):
    return set() if gdf is None else set(int(i) for i in gdf["id"])


def test_write_pbf_subset_only_buildings(helsinki_pbf, tmp_path):
    """subset_only=True writes only the passed layer: buildings survive (same ids,
    same count -> relation member ways were pulled in), the unrelated road network
    is absent. Inverse of test_write_pbf_whole_dataset_preserved."""
    osm = OSM(helsinki_pbf)
    buildings = osm.get_buildings()
    out = str(tmp_path / "buildings_only.osm.pbf")
    osm.write_pbf(buildings, out, subset_only=True)

    back = OSM(out)
    b2 = back.get_buildings()
    assert b2 is not None and len(b2) == len(buildings)
    assert _ids(b2) == _ids(buildings)
    assert b2.geometry.notna().all()
    # if the source had multipolygon building relations, they round-trip too
    if "relation" in set(buildings["osm_type"]):
        assert "relation" in set(b2["osm_type"])
    net2 = back.get_network()
    assert net2 is None or len(net2) == 0


def test_write_pbf_subset_only_network_excludes_buildings(helsinki_pbf, tmp_path):
    osm = OSM(helsinki_pbf)
    network = osm.get_network()
    out = str(tmp_path / "network_only.osm.pbf")
    osm.write_pbf(network, out, subset_only=True)

    back = OSM(out)
    net2 = back.get_network()
    assert net2 is not None and len(net2) > 0
    assert _ids(network) <= _ids(net2)
    b = back.get_buildings()
    assert b is None or len(b) == 0


def test_write_pbf_subset_only_multilayer(helsinki_pbf, tmp_path):
    """A list of frames writes the union of their elements; both layers come back
    and standalone POIs (in neither layer) are not carried along."""
    osm = OSM(helsinki_pbf)
    buildings = osm.get_buildings()
    network = osm.get_network()
    src_poi_nodes = osm.get_pois()
    src_poi_nodes = _ids(src_poi_nodes[src_poi_nodes["osm_type"] == "node"])

    out = str(tmp_path / "multi.osm.pbf")
    osm.write_pbf([buildings, network], out, subset_only=True)

    back = OSM(out)
    b2, net2 = back.get_buildings(), back.get_network()
    assert b2 is not None and len(b2) > 0 and _ids(buildings) <= _ids(b2)
    assert net2 is not None and len(net2) > 0 and _ids(network) <= _ids(net2)
    # standalone POI nodes were in neither layer -> far fewer remain as tagged nodes
    out_pois = back.get_pois()
    out_poi_nodes = (
        _ids(out_pois[out_pois["osm_type"] == "node"])
        if out_pois is not None
        else set()
    )
    assert len(out_poi_nodes) < len(src_poi_nodes)


def test_write_pbf_subset_only_false_is_whole_dataset(helsinki_pbf, tmp_path):
    """subset_only=False (default) is unchanged: passing one layer still writes the
    whole cached dataset, so the network (not passed) survives."""
    osm = OSM(helsinki_pbf)
    out = str(tmp_path / "whole.osm.pbf")
    osm.write_pbf(osm.get_buildings(), out, subset_only=False)
    assert len(OSM(out).get_network()) > 0


def test_write_pbf_subset_only_empty(helsinki_pbf, tmp_path):
    """An empty subset (no matched elements, no new rows) writes a readable PBF."""
    osm = OSM(helsinki_pbf)
    empty = osm.get_buildings().iloc[:0]
    out = str(tmp_path / "empty.osm.pbf")
    osm.write_pbf(empty, out, subset_only=True)
    b = OSM(out).get_buildings()
    assert b is None or len(b) == 0


def test_write_pbf_subset_only_network_with_nodes(helsinki_pbf, tmp_path):
    """get_network(nodes=True) returns (nodes, edges); subset-exporting that tuple
    writes the road ways + their (real) nodes and does not synthesize duplicate
    negative-id nodes from the node frame (which has no osm_type column)."""
    osm = OSM(helsinki_pbf)
    nodes, edges = osm.get_network(nodes=True)
    out = str(tmp_path / "net_nodes.osm.pbf")
    osm.write_pbf([nodes, edges], out, subset_only=True)

    back = OSM(out)
    net2 = back.get_network()
    assert net2 is not None and len(net2) > 0
    # no synthesized duplicates: every written node id is a real (positive) id
    node_ids, _, _, _, _ = _read_elements(out)
    assert node_ids and all(nid > 0 for nid in node_ids)


def test_infer_osm_type():
    from pyrosm.pbf_writer import _infer_osm_type

    assert _infer_osm_type(Point(0, 0)) == "node"
    assert _infer_osm_type(LineString([(0, 0), (1, 1)])) == "way"
    assert _infer_osm_type(Polygon([(0, 0), (1, 0), (1, 1)])) == "way"
    assert _infer_osm_type(MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)])])) is None
    assert _infer_osm_type(None) is None


def test_subset_keep_sets_closure():
    """The closure resolves way/node/sub-relation members and way node-refs, and
    tolerates kept/member ids that are absent from the cache."""
    from pyrosm.pbf_writer import _subset_keep_sets

    # No relations cache: a kept way pulls in its node refs; nothing else.
    kn, kw, kr = _subset_keep_sets(
        {}, {5: {}}, {}, {5: {"id": 5, "nodes": [50, 51]}}, {}
    )
    assert kr == set() and kw == {5} and kn == {50, 51}

    # Relation 1 has way/node/sub-relation members (incl. sub-rel 999 absent from the
    # cache); relation 2 (a sub-relation) has a way. 888 is a kept relation absent
    # from the cache. way 10 -> [100, 101], way 11 -> [101, 102].
    relations = {
        "id": np.array([1, 2], dtype=np.int64),
        "members": np.array(
            [
                {
                    "member_type": [b"way", b"node", b"relation", b"relation"],
                    "member_id": np.array([10, 100, 2, 999], dtype=np.int64),
                    "member_role": [b"outer", b"", b"", b""],
                },
                {
                    # plain str member_type (not bytes) is handled too
                    "member_type": ["way"],
                    "member_id": np.array([11], dtype=np.int64),
                    "member_role": ["outer"],
                },
            ],
            dtype=object,
        ),
    }
    way_by_id = {
        10: {"id": 10, "nodes": [100, 101]},
        11: {"id": 11, "nodes": [101, 102]},
    }
    kn, kw, kr = _subset_keep_sets({}, {}, {1: {}, 888: {}}, way_by_id, relations)
    assert kr == {1, 2, 888}  # sub-relation 2 resolved; 888 kept; 999 absent
    assert kw == {10, 11}  # way members of relations 1 and 2
    assert kn == {100, 101, 102}  # node member + way node-refs
