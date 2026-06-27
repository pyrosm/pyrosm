"""Tests for pyrosm.graph_simplify (issue #89): OSMnx-equivalent simplification."""

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point


def _graph(node_xy, rows):
    """Build (nodes, directed_edges) from a {node_id: (x, y)} dict and a list of
    directed rows (u, v, [coord, ...]). pyrosm reverse rows reuse forward geometry,
    so pass coords in whatever orientation pyrosm would store."""
    nodes = GeoDataFrame(
        {
            "id": list(node_xy),
            "lon": [c[0] for c in node_xy.values()],
            "lat": [c[1] for c in node_xy.values()],
            "geometry": [Point(c) for c in node_xy.values()],
        },
        crs="epsg:4326",
    )
    geoms = [LineString(coords) for _, _, coords in rows]
    lengths = [LineString(coords).length for _, _, coords in rows]
    edges = GeoDataFrame(
        {
            "u": [r[0] for r in rows],
            "v": [r[1] for r in rows],
            "length": lengths,
            "highway": ["residential"] * len(rows),
            "geometry": geoms,
        },
        crs="epsg:4326",
    )
    return nodes, edges


def test_endpoint_rules_table():
    """Each row of the plan's §5 endpoint table."""
    from pyrosm.graph_simplify import _factorize_nodes, detect_endpoints

    # two-way through X (A<->X<->B): X is a pass-through (removed), A/B endpoints
    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (2, 0)},
        [
            (1, 2, [(0, 0), (1, 0)]),
            (2, 1, [(0, 0), (1, 0)]),
            (2, 3, [(1, 0), (2, 0)]),
            (3, 2, [(1, 0), (2, 0)]),
        ],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep = detect_endpoints(u, v, len(uniq))
    ep_by_id = dict(zip(uniq, ep))
    assert ep_by_id[1] and ep_by_id[3] and not ep_by_id[2]

    # one-way through X (A->X->B): still a pass-through
    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (2, 0)},
        [(1, 2, [(0, 0), (1, 0)]), (2, 3, [(1, 0), (2, 0)])],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep_by_id = dict(zip(uniq, detect_endpoints(u, v, len(uniq))))
    assert not ep_by_id[2] and ep_by_id[1] and ep_by_id[3]

    # two-way dead-end tip D (A<->D): D kept (distinct neighbours == 1)
    nodes, edges = _graph(
        {1: (0, 0), 4: (1, 0)},
        [(1, 4, [(0, 0), (1, 0)]), (4, 1, [(0, 0), (1, 0)])],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep_by_id = dict(zip(uniq, detect_endpoints(u, v, len(uniq))))
    assert ep_by_id[4] and ep_by_id[1]

    # one-way dead-end tip D (A->D): D kept (out_degree == 0)
    nodes, edges = _graph({1: (0, 0), 4: (1, 0)}, [(1, 4, [(0, 0), (1, 0)])])
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep_by_id = dict(zip(uniq, detect_endpoints(u, v, len(uniq))))
    assert ep_by_id[4] and ep_by_id[1]

    # 3-way junction J (two-way to A, B, C): J kept (distinct neighbours == 3)
    nodes, edges = _graph(
        {10: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1)},
        [
            (10, 1, [(0, 0), (-1, 0)]),
            (1, 10, [(0, 0), (-1, 0)]),
            (10, 2, [(0, 0), (1, 0)]),
            (2, 10, [(0, 0), (1, 0)]),
            (10, 3, [(0, 0), (0, 1)]),
            (3, 10, [(0, 0), (0, 1)]),
        ],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep_by_id = dict(zip(uniq, detect_endpoints(u, v, len(uniq))))
    assert ep_by_id[10]


def test_simplify_twoway_chain_and_orientation():
    """A-X-B collapses to two reciprocal edges; reverse edge geometry is flipped."""
    from pyrosm.graph_simplify import simplify_graph

    nodes, edges = _graph(
        {100: (0, 0), 200: (1, 0), 300: (2, 0)},
        [
            (100, 200, [(0, 0), (1, 0)]),
            (200, 100, [(0, 0), (1, 0)]),
            (200, 300, [(1, 0), (2, 0)]),
            (300, 200, [(1, 0), (2, 0)]),
        ],
    )
    sn, se = simplify_graph(nodes, edges)
    assert sorted(sn["id"]) == [100, 300]
    assert len(se) == 2
    by = {(int(r.u), int(r.v)): r for _, r in se.iterrows()}
    assert list(by[(100, 300)].geometry.coords) == [(0, 0), (1, 0), (2, 0)]
    assert list(by[(300, 100)].geometry.coords) == [(2, 0), (1, 0), (0, 0)]
    assert by[(100, 300)].length == pytest.approx(2.0)
    assert se.crs == edges.crs  # CRS preserved on the simplified edges


def test_cython_matches_reference():
    from pyrosm.graph_simplify import (
        _factorize_nodes,
        detect_endpoints,
        _build_csr,
        _reference_walk,
    )
    from pyrosm._simplify_walk import walk_chains

    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)},
        [
            (1, 2, [(0, 0), (1, 0)]),
            (2, 1, [(0, 0), (1, 0)]),
            (2, 3, [(1, 0), (2, 0)]),
            (3, 2, [(1, 0), (2, 0)]),
            (3, 4, [(2, 0), (3, 0)]),
            (4, 3, [(2, 0), (3, 0)]),
        ],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep = detect_endpoints(u, v, len(uniq))
    indptr, indices, edge_id, src = _build_csr(u, v, len(uniq))
    ref = _reference_walk(indptr, indices, edge_id, ep, src, True)
    cy = walk_chains(
        indptr.astype(np.longlong),
        indices.astype(np.longlong),
        edge_id.astype(np.longlong),
        ep.astype(np.uint8),
        src.astype(np.longlong),
        True,
    )
    assert np.array_equal(ref[0], cy[0]) and np.array_equal(ref[1], cy[1])


def test_attribute_merge_mixed_vs_uniform():
    """A column that varies along a collapsed chain becomes a list in walk order;
    a uniform column (including all-missing) stays a scalar."""
    from pyrosm.graph_simplify import simplify_graph

    nodes = GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "lon": [0, 1, 2, 3],
            "lat": [0, 0, 0, 0],
            "geometry": [Point(i, 0) for i in range(4)],
        },
        crs="epsg:4326",
    )
    # one-way chain 1->2->3->4; nodes 2,3 are interstitial -> single collapsed edge
    edges = GeoDataFrame(
        {
            "u": [1, 2, 3],
            "v": [2, 3, 4],
            "length": [1.0, 1.0, 1.0],
            "highway": ["residential", "primary", "residential"],  # varies
            "name": [None, None, None],  # uniform (all missing)
            "geometry": [LineString([(i, 0), (i + 1, 0)]) for i in range(3)],
        },
        crs="epsg:4326",
    )
    _, se = simplify_graph(nodes, edges)
    assert len(se) == 1
    row = se.iloc[0]
    assert (int(row["u"]), int(row["v"])) == (1, 4)
    assert row["length"] == pytest.approx(3.0)
    # varying column -> list of the per-segment values in walk order
    assert row["highway"] == ["residential", "primary", "residential"]
    # uniform all-missing column -> scalar NaN/None, not a list
    assert not isinstance(row["name"], list)
    assert pd.isna(row["name"])


def test_parallel_edges_stay_separate():
    """Two distinct parallel edges between the same endpoints must not be merged."""
    from pyrosm.graph_simplify import simplify_graph

    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0)},
        [
            (1, 2, [(0, 0), (1, 0)]),
            (2, 1, [(0, 0), (1, 0)]),
            (1, 2, [(0, 0), (0.5, 1), (1, 0)]),
            (2, 1, [(0, 0), (0.5, 1), (1, 0)]),
        ],
    )
    sn, se = simplify_graph(nodes, edges)
    assert sorted(sn["id"]) == [1, 2]
    assert len(se) == 4  # both endpoints kept, all 4 directed rows survive


def test_self_loop_node_is_endpoint():
    from pyrosm.graph_simplify import _factorize_nodes, detect_endpoints

    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0)},
        [
            (1, 1, [(0, 0), (0.2, 0.2), (0, 0)]),
            (1, 2, [(0, 0), (1, 0)]),
            (2, 1, [(0, 0), (1, 0)]),
        ],
    )
    u, v, uniq = _factorize_nodes(edges, "u", "v")
    ep = dict(zip(uniq, detect_endpoints(u, v, len(uniq))))
    assert ep[1]  # self-loop -> endpoint


def test_ring_remove_flag():
    """A pure ring (no endpoints) is dropped with remove_rings=True, kept otherwise."""
    from pyrosm.graph_simplify import simplify_graph

    # one-way triangle ring A->B->C->A, all interstitial
    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (0.5, 1)},
        [
            (1, 2, [(0, 0), (1, 0)]),
            (2, 3, [(1, 0), (0.5, 1)]),
            (3, 1, [(0.5, 1), (0, 0)]),
        ],
    )
    sn_drop, se_drop = simplify_graph(nodes, edges, remove_rings=True)
    assert len(se_drop) == 0 and len(sn_drop) == 0
    sn_keep, se_keep = simplify_graph(nodes, edges, remove_rings=False)
    assert len(se_keep) == 1  # collapsed to a single self-loop edge
    # the ring's pseudo-endpoint node must be present in the output node table, and
    # the kept edge must be a self-loop on it (guards the remove_rings=False node bug)
    ring_edge = se_keep.iloc[0]
    assert int(ring_edge.u) == int(ring_edge.v)
    assert int(ring_edge.u) in set(sn_keep["id"].astype(int))


def test_oneway_loop_back_to_endpoint():
    """A one-way path that loops back to its origin endpoint collapses to a single
    self-loop edge whose geometry runs in walk order (E -> A -> B -> E)."""
    from pyrosm.graph_simplify import simplify_graph

    # D is a dead-end so E (node 1) has 3 distinct neighbours -> a real endpoint;
    # A (2) and B (3) are one-way interstitial nodes on the loop back to E.
    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (-1, 0)},
        [
            (4, 1, [(-1, 0), (0, 0)]),
            (1, 2, [(0, 0), (1, 0)]),
            (2, 3, [(1, 0), (1, 1)]),
            (3, 1, [(1, 1), (0, 0)]),
        ],
    )
    _, se = simplify_graph(nodes, edges)
    loops = se[se["u"] == se["v"]]
    assert len(loops) == 1
    loop = loops.iloc[0]
    assert int(loop.u) == 1
    assert list(loop.geometry.coords) == [(0, 0), (1, 0), (1, 1), (0, 0)]
    # the straight D -> E edge survives unchanged as a separate edge
    assert (4, 1) in set(zip(se["u"].astype(int), se["v"].astype(int)))


def test_relaxation_flags_accept_arraylike():
    """edge_attrs_differ / node_attrs_include accept array-likes (pd.Index, ndarray)
    without raising on an ambiguous truth value."""
    from pyrosm.graph_simplify import simplify_graph

    nodes, edges = _graph(
        {1: (0, 0), 2: (1, 0), 3: (2, 0)},
        [(1, 2, [(0, 0), (1, 0)]), (2, 3, [(1, 0), (2, 0)])],
    )
    _, se = simplify_graph(
        nodes,
        edges,
        edge_attrs_differ=pd.Index(["highway"]),
        node_attrs_include=np.array(["highway"]),
    )
    assert len(se) == 1  # uniform highway -> chain still collapses, no crash


def test_no_graph_library_imports():
    import pyrosm.graph_simplify as gs

    src = open(gs.__file__).read()
    assert "import networkx" not in src and "import igraph" not in src
    # Stronger than a source scan: importing the module in a fresh interpreter must
    # not pull networkx or igraph into sys.modules (a transitive/conditional runtime
    # import would slip past the textual check above).
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pyrosm.graph_simplify; "
            "assert 'networkx' not in sys.modules, 'networkx imported'; "
            "assert 'igraph' not in sys.modules, 'igraph imported'",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "bbox",
    [
        [24.93, 60.16, 24.96, 60.18],  # dense central core
        [24.90, 60.17, 24.94, 60.20],  # mixed central-north
        [24.95, 60.17, 24.99, 60.20],  # eastern, more one-way / arterial
    ],
)
def test_osmnx_equivalence(bbox):
    """Gold standard: match osmnx.simplification.simplify_graph on a real extract."""
    nx = pytest.importorskip("networkx")
    oxs = pytest.importorskip("osmnx.simplification")
    from pyrosm import OSM, get_data
    from pyrosm.graphs import get_directed_edges
    import pyrosm.graph_simplify as gs

    osm = OSM(get_data("helsinki", update=False), bounding_box=bbox)
    nodes, edges = osm.get_network("driving", nodes=True)
    _, directed = get_directed_edges(nodes, edges, network_type="driving")
    directed = directed.reset_index(drop=True)

    G = nx.MultiDiGraph(crs="epsg:4326")
    for _, n in nodes.iterrows():
        G.add_node(int(n["id"]), x=float(n["lon"]), y=float(n["lat"]))
    for _, e in directed.iterrows():
        if int(e["u"]) in G and int(e["v"]) in G:
            G.add_edge(
                int(e["u"]),
                int(e["v"]),
                osmid=int(e["id"]),
                length=float(e["length"]),
                geometry=e["geometry"],
            )

    Gs = oxs.simplify_graph(G, edge_attrs_differ=None)
    sn, se = gs.simplify_graph(nodes, directed)

    assert set(map(int, Gs.nodes())) == set(sn["id"].astype(int))
    assert Gs.number_of_edges() == len(se)
    ox_len = sum(d["length"] for _, _, d in Gs.edges(data=True))
    assert abs(ox_len - se["length"].sum()) < 1e-3

    # orientation invariant: each simplified edge geometry runs u -> v
    xy = {int(n["id"]): (float(n["lon"]), float(n["lat"])) for _, n in nodes.iterrows()}
    for _, r in se.iterrows():
        c = list(r.geometry.coords)
        assert c[0] == pytest.approx(xy[int(r.u)])
        assert c[-1] == pytest.approx(xy[int(r.v)])

    # EXACT per-edge geometry equality: match each pyrosm edge to a unique OSMnx edge
    # by directed endpoints (consuming matches one-to-one to disambiguate parallel
    # edges) and assert the merged geometries coincide within tolerance. This is what
    # catches a per-edge geometry/orientation divergence that aggregate totals miss.
    from collections import defaultdict
    from shapely.geometry import LineString

    def _ordered_match(g1, g2, tol=1e-6):
        # Sequence-sensitive (stronger than hausdorff, which ignores order): require
        # the same vertex count and the same coordinate *sequence*. Either direction
        # is accepted because OSMnx does not re-orient un-simplified endpoint-to-
        # endpoint edges, so its geometry for a (u, v) key may run v -> u; pyrosm's
        # absolute u -> v orientation is asserted by the orientation-invariant loop
        # above. A scrambled-order geometry with a coincidentally-close point set,
        # which hausdorff could pass, still fails here.
        a = np.asarray(g1.coords)
        b = np.asarray(g2.coords)
        if a.shape != b.shape:
            return False
        return np.allclose(a, b, atol=tol) or np.allclose(a, b[::-1], atol=tol)

    ox_by_uv = defaultdict(list)
    for a, b, d in Gs.edges(data=True):
        g = d.get("geometry") or LineString([xy[int(a)], xy[int(b)]])
        ox_by_uv[(int(a), int(b))].append(g)
    for _, r in se.iterrows():
        cands = ox_by_uv[(int(r.u), int(r.v))]
        match = next(
            (i for i, g in enumerate(cands) if _ordered_match(r.geometry, g)),
            None,
        )
        assert match is not None, f"no matching OSMnx geometry for edge ({r.u},{r.v})"
        cands.pop(match)


def test_to_graph_simplify_integration():
    from pyrosm import OSM, get_data

    osm = OSM(
        get_data("helsinki", update=False), bounding_box=[24.93, 60.16, 24.95, 60.17]
    )
    nodes, edges = osm.get_network("driving", nodes=True)
    full = osm.to_graph(
        nodes, edges, graph_type="igraph", network_type="driving", retain_all=True
    )
    simp = osm.to_graph(
        nodes,
        edges,
        graph_type="igraph",
        network_type="driving",
        retain_all=True,
        simplify=True,
    )
    assert simp.vcount() < full.vcount()
    assert simp.ecount() < full.ecount()
