import os

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
        assert os.path.exists(out)

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
        os.remove(out)


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
        os.remove(out)


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
        os.remove(out_keep)
        os.remove(out_drop)


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
        os.remove(out)


def test_to_pbf_given_path(helsinki_pbf, tmp_path):
    target = str(tmp_path / "cropped.osm.pbf")
    osm = OSM(helsinki_pbf, bounding_box=CROP_BBOX)
    out = osm.to_pbf(target)
    assert out == target
    assert os.path.exists(target)
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
        os.remove(out_seq)
        os.remove(out_par)


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
        os.remove(out)
