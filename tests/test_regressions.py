"""Regression tests guarding against specific bugs reappearing."""

import sys

import pytest


def test_get_methods_do_not_mutate_shared_tag_config():
    """#252 — get_* must not mutate the shared Conf default-tag lists."""
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf

    osm = OSM(get_data("test_pbf"))

    building_before = list(Conf.tags.building)
    highway_before = list(Conf.tags.highway)
    natural_before = list(Conf.tags.natural)

    osm.get_buildings()
    osm.get_network()
    osm.get_natural()
    osm.get_buildings(extra_attributes=["my_extra_attr"])

    assert Conf.tags.building == building_before
    assert Conf.tags.highway == highway_before
    assert Conf.tags.natural == natural_before
    assert "my_extra_attr" not in Conf.tags.building


def test_frame_building_emits_no_chained_assignment_warning():
    """#237, PR #256 — frame builders emit no pandas chained-assignment warning."""
    import warnings

    try:
        from pandas.errors import ChainedAssignmentError
    except ImportError:  # pandas too old to have the warning
        import pytest

        pytest.skip("pandas has no ChainedAssignmentError")

    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ChainedAssignmentError)
        osm.get_buildings()
        osm.get_network()
        osm.get_network(nodes=True)
        osm.get_pois()
        osm.get_landuse()
        osm.get_natural()
        osm.get_boundaries()
        osm.get_data_by_custom_criteria(custom_filter={"building": True})


def test_uk_subregions_use_united_kingdom_path():
    """#239 — Geofabrik moved the UK sub-regions (England, Scotland, Wales and
    the English counties) under the 'united-kingdom' path. 'great-britain' and
    'united-kingdom' remain distinct whole-region files (GB without vs. with
    Northern Ireland)."""
    from pyrosm.data import search_source

    # Sub-regions now live under europe/united-kingdom/...
    for name in ["england", "scotland", "wales", "greater_london", "merseyside"]:
        url = search_source(name)["url"]
        assert "united-kingdom" in url
        assert "great-britain" not in url

    # The two whole-region country files stay distinct and both valid.
    assert search_source("united_kingdom")["url"].endswith(
        "europe/united-kingdom-latest.osm.pbf"
    )
    assert search_source("great_britain")["url"].endswith(
        "europe/great-britain-latest.osm.pbf"
    )

    # Sub-region navigation via the great_britain group still works and resolves
    # to the united-kingdom path.
    from pyrosm.data import sources

    gb = sources.subregions.great_britain
    assert "europe/united-kingdom/" in gb.scotland["url"]
    assert gb() == gb.available


def test_nxgraph_keys_nodes_by_id_not_dataframe_index():
    """#247 — _create_nxgraph must key nodes by node_id_col, not the DataFrame
    index, so it does not create duplicate 'phantom' nodes when the input nodes
    have a non-id index (e.g. a default RangeIndex)."""
    import pytest

    pytest.importorskip("networkx")
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from pyrosm.graph_export import _create_nxgraph

    # Nodes with non-sequential ids and a default RangeIndex (0, 1, 2).
    nodes = gpd.GeoDataFrame(
        {
            "id": [1000, 2000, 3000],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "u": [1000, 2000],
            "v": [2000, 3000],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ],
        },
        crs="EPSG:4326",
    )

    graph = _create_nxgraph(nodes, edges, "u", "v", "id")

    # Exactly the three real node ids, no index-keyed phantoms (0, 1, 2).
    assert graph.number_of_nodes() == 3
    assert sorted(graph.nodes()) == [1000, 2000, 3000]
    # Every node carries its attributes.
    assert all("geometry" in data for _, data in graph.nodes(data=True))


_EXCLUDED_SERVICE = {"parking", "parking_aisle", "private", "emergency_access"}


def test_exclude_filter_does_not_leak_secondary_keys():
    """#112 — an exclude custom_filter that lists `service` values must drop those
    ways. The old ways filter broke on the first filter key in the record, so a
    `highway=service` way leaked because `highway` was checked before `service`."""
    from pyrosm import OSM, get_data

    drive_filter = dict(
        area=["yes"],
        service=["parking", "parking_aisle", "private", "emergency_access"],
        highway=[
            "cycleway",
            "footway",
            "path",
            "pedestrian",
            "steps",
            "track",
            "corridor",
            "elevator",
            "escalator",
            "proposed",
            "construction",
            "bridleway",
            "abandoned",
            "platform",
            "raceway",
        ],
        motor_vehicle=["no"],
        motorcar=["no"],
    )
    osm = OSM(get_data("test_pbf"))
    gdf = osm.get_data_by_custom_criteria(
        custom_filter=drive_filter,
        osm_keys_to_keep="highway",
        filter_type="exclude",
    )
    present = set(gdf["service"].dropna().unique())
    assert not (present & _EXCLUDED_SERVICE), f"leaked: {present & _EXCLUDED_SERVICE}"


def test_driving_network_excludes_service_roads():
    """#108 — get_network('driving') must not leak roads excluded by a secondary
    driving-filter key (e.g. service=parking_aisle on a highway=service road),
    which the old early-break ways filter kept."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    edges = osm.get_network(network_type="driving")

    excluded_highway = {
        "cycleway",
        "footway",
        "path",
        "pedestrian",
        "steps",
        "track",
        "corridor",
        "elevator",
        "escalator",
        "proposed",
        "construction",
        "bridleway",
        "abandoned",
        "platform",
        "raceway",
    }
    hw = set(edges["highway"].dropna().unique())
    assert not (
        hw & excluded_highway
    ), f"non-drivable highway leaked: {hw & excluded_highway}"

    if "service" in edges.columns:
        sv = set(edges["service"].dropna().unique())
        assert not (
            sv & _EXCLUDED_SERVICE
        ), f"excluded service leaked: {sv & _EXCLUDED_SERVICE}"


def test_keep_filter_matches_any_key_or_semantics():
    """#108/#112 follow-on — keep filters with multiple keys must match if ANY key
    matches (OR), not only when the first-visited filter key matches. Pre-fix a
    way with highway=service + service=driveway was dropped by
    keep={'highway':['path'], 'service':['driveway']}."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": ["path"], "service": ["driveway"]},
        osm_keys_to_keep="highway",
        filter_type="keep",
    )
    assert gdf is not None and len(gdf) > 0
    assert "driveway" in set(gdf["service"].dropna().unique())


def test_get_network_custom_filter_returns_nodes():
    """#118/#181 — get_network must accept a custom_filter and, with nodes=True,
    return graph-ready (nodes, edges) so a custom-filtered network can be turned
    into a routable graph."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    result = osm.get_network(
        custom_filter={"highway": ["footway", "residential"]},
        filter_type="keep",
        nodes=True,
    )

    assert isinstance(result, tuple) and len(result) == 2
    nodes, edges = result

    assert edges is not None and len(edges) > 0
    for col in ["u", "v", "length"]:
        assert col in edges.columns
    assert set(edges["highway"].dropna().unique()) <= {"footway", "residential"}

    assert nodes is not None and len(nodes) > 0
    assert "id" in nodes.columns
    assert not nodes.geometry.is_empty.any()
    assert set(nodes.geometry.geom_type.unique()) == {"Point"}


def test_get_network_custom_filter_graph_export():
    """#181 — the (nodes, edges) from a custom_filter network must export to a
    graph. With retain_all=True no nodes are dropped (the default retain_all=False
    prunes weakly-connected components, so the strict count needs retain_all)."""
    import pytest

    pytest.importorskip("networkx")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(
        custom_filter={"highway": ["footway", "residential"]},
        filter_type="keep",
        nodes=True,
    )

    graph = OSM.to_graph(nodes, edges, graph_type="networkx", retain_all=True)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_nodes() == len(nodes)

    # Default connectivity pruning keeps a subset of the returned node ids.
    pruned = OSM.to_graph(nodes, edges, graph_type="networkx")
    assert set(pruned.nodes()).issubset(set(nodes["id"]))


def test_get_network_custom_filter_exclude():
    """#118/#181 — a 'exclude' custom_filter drops the matching ways, and is the
    complement of the matching 'keep' filter."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    excluded = osm.get_network(
        custom_filter={"highway": ["footway"]}, filter_type="exclude", nodes=False
    )
    kept = osm.get_network(
        custom_filter={"highway": ["footway"]}, filter_type="keep", nodes=False
    )

    assert "footway" not in set(excluded["highway"].dropna().unique())
    assert len(excluded) > len(kept)


def test_get_network_custom_filter_validation_and_extra_keys():
    """#118/#181 — get_network rejects an invalid filter_type, and a custom_filter
    key outside the default 'highway' tag set is accepted (and added as a column
    when present in the data)."""
    import pytest
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf

    osm = OSM(get_data("test_pbf"))

    with pytest.raises(ValueError):
        osm.get_network(
            custom_filter={"highway": ["footway"]}, filter_type="not-a-filter"
        )

    # 'moped' is not one of the default highway tag columns; the filter key must
    # still be accepted and routed through (it becomes a column only if present).
    assert "moped" not in list(Conf.tags.highway)
    edges = osm.get_network(
        custom_filter={"highway": ["footway", "residential"], "moped": ["yes"]},
        filter_type="keep",
    )
    assert edges is not None and len(edges) > 0
    assert set(edges["highway"].dropna().unique()) <= {"footway", "residential"}


def test_get_network_without_custom_filter_unchanged():
    """Parity guard: the default (no custom_filter) get_network path is unchanged.
    Pins row count, column set, network_type metadata, and a platform-stable hash
    of the per-row id/highway values for the bundled extract."""
    import hashlib
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    edges = osm.get_network(network_type="driving")

    assert len(edges) == 200
    assert edges["id"].nunique() == 200
    assert sorted(edges.columns) == [
        "access",
        "bridge",
        "geometry",
        "highway",
        "id",
        "int_ref",
        "lanes",
        "length",
        "lit",
        "maxspeed",
        "name",
        "oneway",
        "osm_type",
        "ref",
        "service",
        "surface",
        "tags",
        "timestamp",
        "version",
    ]
    assert edges._metadata[-1] == "driving"

    rows = sorted(f"{i}:{h}" for i, h in zip(edges["id"], edges["highway"].fillna("")))
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    assert digest == "8ce8c63b51ee14f8008a6af4d642cb3adf95e83a6f926fa6c9e4381bb3f3b072"


def test_single_key_keep_filter_unchanged():
    """Parity guard: single-key keep filters and {key: True} are unaffected by the
    OR rewrite (only multi-key keep filters were buggy). Row counts are pinned to
    the current-master values so a regression in the unaffected paths is caught."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))

    # Single-key keep on "building" (get_buildings) is unchanged.
    buildings = osm.get_buildings()
    assert len(buildings) == 2208

    # {"building": True} keep recovers the same building ways.
    by_true = osm.get_data_by_custom_criteria(
        custom_filter={"building": True}, filter_type="keep"
    )
    assert len(by_true) == 2208

    # A single-key POI keep is unchanged (exercises the keep path for POIs).
    pois = osm.get_pois(custom_filter={"amenity": True})
    assert len(pois) == 20


def test_networkx_export_sets_street_count():
    """#117 — the exported NetworkX graph must carry a per-node 'street_count'
    attribute (number of streets incident to each intersection) so OSMnx's
    basic_stats works."""
    import pytest

    pytest.importorskip("networkx")
    import networkx as nx
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)
    graph = osm.to_graph(nodes, edges, graph_type="networkx")

    street_count = nx.get_node_attributes(graph, "street_count")
    # Present on every node and a positive integer.
    assert len(street_count) == graph.number_of_nodes()
    assert all(isinstance(c, int) and c >= 1 for c in street_count.values())


def test_street_count_matches_osmnx():
    """#117 — pyrosm's 'street_count' must equal osmnx.stats.count_streets_per_node
    across the export modes (osmnx_compatible x retain_all) the graph supports."""
    import pytest

    pytest.importorskip("networkx")
    osmnx = pytest.importorskip("osmnx")
    import networkx as nx
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    for osmnx_compatible in [True, False]:
        for retain_all in [True, False]:
            graph = osm.to_graph(
                nodes,
                edges,
                graph_type="networkx",
                osmnx_compatible=osmnx_compatible,
                retain_all=retain_all,
            )
            pyrosm_counts = nx.get_node_attributes(graph, "street_count")
            assert pyrosm_counts == osmnx.stats.count_streets_per_node(graph), (
                f"mismatch for osmnx_compatible={osmnx_compatible}, "
                f"retain_all={retain_all}"
            )


def test_networkx_export_works_with_osmnx_basic_stats():
    """#117 — the original symptom: osmnx.basic_stats(G) must run on the exported
    graph without raising (it failed because 'street_count' was missing)."""
    import pytest

    pytest.importorskip("networkx")
    osmnx = pytest.importorskip("osmnx")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)
    graph = osm.to_graph(nodes, edges, graph_type="networkx")

    stats = osmnx.basic_stats(graph)
    assert stats["n"] == graph.number_of_nodes()


def test_count_streets_per_node_known_topology():
    """#117 — pins the street_count semantics on a known layout: a path 1-2-3 with
    a branch 2-4 (bidirectional). Node 2 touches three streets; the leaves one."""
    import pytest

    pytest.importorskip("networkx")
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import Point, LineString
    from pyrosm import OSM

    nodes = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "lon": [0.0, 1.0, 2.0, 1.0],
            "lat": [0.0, 0.0, 0.0, 1.0],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "id": [10, 20, 30],
            "u": [1, 2, 2],
            "v": [2, 3, 4],
            "length": [1.0, 1.0, 1.0],
            "oneway": [None, None, None],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(1, 0), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )

    graph = OSM.to_graph(
        nodes, edges, graph_type="networkx", network_type="walking", retain_all=True
    )
    assert nx.get_node_attributes(graph, "street_count") == {1: 1, 2: 3, 3: 1, 4: 1}


def test_bbox_outside_extent_returns_empty_not_keyerror():
    """#241 — a bounding box that selects no nodes must not crash with
    KeyError "None of ['id'] are in the columns"; it should return empty data and
    warn instead."""
    import pytest
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), bounding_box=[0.0, 0.0, 0.001, 0.001])
    with pytest.warns(UserWarning):
        edges = osm.get_network(network_type="all")
    assert edges is None


def test_inverted_bbox_raises_valueerror_with_coord_order_hint():
    """#241 — an inverted/degenerate bounding box (e.g. min_x > max_x, the
    reproduction from the issue) is malformed input and must be rejected at OSM()
    construction with a ValueError that hints at the coordinate order, rather than
    crashing later with a cryptic KeyError."""
    import pytest
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    # Inverted x (min_x > max_x), the openbrian shape.
    with pytest.raises(ValueError, match="minx"):
        OSM(fp, bounding_box=[24.96, 60.16, 24.93, 60.20])
    # Degenerate (zero-width) bbox is also rejected.
    with pytest.raises(ValueError, match="minx"):
        OSM(fp, bounding_box=[24.93, 60.16, 24.93, 60.20])


def test_empty_bbox_pois_and_buildings_do_not_crash():
    """#241 — the node-using getters (get_pois, get_buildings) must also handle an
    empty bounding box gracefully (the original report used get_pois)."""
    import warnings
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), bounding_box=[0.0, 0.0, 0.001, 0.001])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = osm.get_pois(custom_filter={"amenity": True})
        buildings = osm.get_buildings()
    assert pois is None
    assert buildings is None


def test_valid_bbox_unchanged_by_empty_guard():
    """#241 guard — a valid in-extent bounding box still parses normally."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), bounding_box=[24.93, 60.16, 24.96, 60.20])
    edges = osm.get_network(network_type="all")
    assert edges is not None
    assert len(edges) == 2577


def test_bbox_straddling_building_ways_complete_not_cut():
    """#236 — building ways crossing the bounding-box edge must come back with
    their complete geometry, not cut. Pre-fix, a way straddling the edge lost its
    outside-bbox vertices and its area collapsed (some even became invalid)."""
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    full = OSM(fp).get_buildings()
    full = full[full["osm_type"] == "way"].set_index("id")

    sub = OSM(fp, bounding_box=[24.93, 60.16, 24.945, 60.18]).get_buildings()
    sub = sub[sub["osm_type"] == "way"].set_index("id")

    # The fix completes geometries; it does not change which ways are kept.
    assert len(sub) == 181
    common = full.index.intersection(sub.index)
    assert len(common) == len(sub)  # every bbox building way exists in the full set

    # No building way comes back with a smaller (cut) area than in the full dataset.
    cut = [
        i
        for i in common
        if full.loc[i].geometry.area > 0
        and sub.loc[i].geometry.area < full.loc[i].geometry.area * 0.999
    ]
    assert cut == [], f"cut/incomplete building ways: {cut}"


def test_bbox_does_not_introduce_invalid_building_ways():
    """#236 — the bounding-box parse must not turn a building way that is valid in
    the full dataset into an invalid one (the cut geometries were sometimes
    invalid). Source buildings that are already invalid without a bbox are
    tolerated."""
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    full = OSM(fp).get_buildings()
    full = full[full["osm_type"] == "way"].set_index("id")

    sub = OSM(fp, bounding_box=[24.93, 60.16, 24.945, 60.18]).get_buildings()
    sub = sub[sub["osm_type"] == "way"].set_index("id")

    common = full.index.intersection(sub.index)
    introduced = [
        i
        for i in common
        if full.loc[i].geometry.is_valid and not sub.loc[i].geometry.is_valid
    ]
    assert introduced == [], f"bbox introduced invalid geometries: {introduced}"


def test_bbox_network_nodes_cover_all_edge_endpoints():
    """#199 — with a bounding box, get_network(nodes=True) must return a nodes
    frame containing every endpoint referenced by the edges. Boundary endpoints
    of edges that straddle the box used to be clipped away, leaving dangling
    u/v."""
    from pyrosm import OSM, get_data

    fp = get_data("test_pbf")
    bounds = [26.94, 60.525, 26.96, 60.535]
    nodes, edges = OSM(filepath=fp, bounding_box=bounds).get_network(nodes=True)

    missing = (set(edges["u"]) | set(edges["v"])) - set(nodes["id"])
    assert missing == set(), f"edge endpoints missing from nodes: {missing}"


@pytest.mark.parametrize(
    "graph_type",
    [
        "networkx",
        "igraph",
        "pandarm",
        # pandana's compiled cyaccess uses C `long` buffers (32-bit on Windows),
        # but NumPy 2 makes the default integer int64, so its export is broken on
        # Windows regardless of this fix; skip there as the graph-export tests do.
        pytest.param(
            "pandana",
            marks=pytest.mark.skipif(
                sys.platform.startswith("win"),
                reason="pandana is incompatible with NumPy 2 on Windows (C long is 32-bit)",
            ),
        ),
    ],
)
def test_bbox_network_to_graph(graph_type):
    """#199 — to_graph must build from the get_network(nodes=True) output of a
    bbox reader without manual cleanup. Pre-fix the pandana export raised
    'Buffer dtype mismatch, expected long but got double'. Parametrized so each
    backend is exercised (or skipped) independently — a missing optional backend
    must not mask the others."""
    pytest.importorskip(graph_type)
    from pyrosm import OSM, get_data

    fp = get_data("test_pbf")
    bounds = [26.94, 60.525, 26.96, 60.535]
    osm = OSM(filepath=fp, bounding_box=bounds)
    nodes, edges = osm.get_network(nodes=True)

    g = osm.to_graph(nodes, edges, graph_type=graph_type)
    assert g is not None


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="pandana is incompatible with NumPy 2 on Windows (C long is 32-bit)",
)
def test_to_graph_pandana_emits_deprecation_warning():
    """#270 — graph_type='pandana' still works but warns that it is deprecated in
    favour of 'pandarm'."""
    pytest.importorskip("pandana")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    with pytest.warns(DeprecationWarning, match="pandarm"):
        g = osm.to_graph(nodes, edges, graph_type="pandana")
    assert g is not None


def test_parse_nodes_non_dense_pbf_matches_dense(tmp_path):
    """#274 — pyrosm parses non-dense PBF node groups (pbfreader.parse_nodes) and
    produces the same network and buildings as the dense encoding. Guards the
    regression where parse_nodes' dict result was spread into the all_nodes list
    as bare keys, crashing non-dense reads with 'str object has no attribute
    keys'. A non-dense copy of the bundled test PBF is generated with osmium."""
    import numpy as np

    osmium = pytest.importorskip("osmium")
    from pyrosm import OSM, get_data

    src = get_data("test_pbf")
    nondense = str(tmp_path / "test_nondense.osm.pbf")
    with osmium.SimpleWriter(
        osmium.io.File(nondense, "pbf,pbf_dense_nodes=false")
    ) as writer:
        for obj in osmium.FileProcessor(src):
            writer.add(obj)

    dense_edges = OSM(src).get_network()
    nondense_edges = OSM(nondense).get_network()
    assert nondense_edges is not None
    assert len(nondense_edges) == len(dense_edges)

    dense_buildings = OSM(src).get_buildings()
    nondense_buildings = OSM(nondense).get_buildings()
    assert len(nondense_buildings) == len(dense_buildings)
    # Same geometric extent -> node coordinates parsed identically.
    assert np.allclose(
        nondense_buildings.total_bounds, dense_buildings.total_bounds, atol=1e-6
    )

    # POIs come from standalone (tagged) nodes, so this checks parse_nodes emits
    # the 'tags' column with the same schema as parse_dense.
    dense_pois = OSM(src).get_pois()
    nondense_pois = OSM(nondense).get_pois()
    assert nondense_pois is not None
    assert len(nondense_pois) == len(dense_pois)
    assert set(nondense_pois.columns) == set(dense_pois.columns)

    # A bounding box exercises parse_nodes' bbox filter and the #236
    # boundary-completion (id-filter) pass on the non-dense node groups.
    bounds = [26.94, 60.525, 26.96, 60.535]
    dense_bbox = OSM(src, bounding_box=bounds).get_buildings()
    nondense_bbox = OSM(nondense, bounding_box=bounds).get_buildings()
    assert nondense_bbox is not None
    assert len(nondense_bbox) == len(dense_bbox)


def test_parse_nodes_non_dense_osh_history_with_timestamp(tmp_path):
    """#274 — parse_nodes must emit the 'visible' column like parse_dense, so a
    non-dense *history* PBF read with a timestamp filter (which selects on
    'visible') works instead of crashing on a missing column."""
    osmium = pytest.importorskip("osmium")
    from pyrosm import OSM, get_data

    history = get_data("helsinki_test_history_pbf")
    nondense = str(tmp_path / "history_nondense.osh.pbf")
    with osmium.SimpleWriter(
        osmium.io.File(nondense, "osh.pbf,pbf_dense_nodes=false")
    ) as writer:
        for obj in osmium.FileProcessor(history):
            writer.add(obj)

    dense = OSM(history).get_buildings(timestamp="2010-01-01")
    nondense_buildings = OSM(nondense).get_buildings(timestamp="2010-01-01")
    assert nondense_buildings is not None
    assert len(nondense_buildings) == len(dense)
    # Same geometry at the timestamp -> node versions newer than the timestamp
    # were dropped (parse_nodes honours unix_time_filter), not just the latest.
    import numpy as np

    dense_geom = dense.sort_values("id").reset_index(drop=True)
    nd_geom = nondense_buildings.sort_values("id").reset_index(drop=True)
    assert (dense_geom["id"].tolist()) == (nd_geom["id"].tolist())
    assert np.allclose(dense_geom.total_bounds, nd_geom.total_bounds, atol=1e-6)
    assert (dense_geom.geometry.area.round(10) == nd_geom.geometry.area.round(10)).all()


def test_version_attribute_and_unknown_fallback(monkeypatch):
    """#277 — pyrosm exposes __version__ from the installed distribution
    metadata, falling back to 'unknown' when pyrosm is not installed as a
    distribution (e.g. imported straight from a source tree)."""
    import importlib
    import importlib.metadata

    import pyrosm

    assert isinstance(pyrosm.__version__, str) and pyrosm.__version__

    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    try:
        importlib.reload(pyrosm)
        assert pyrosm.__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(pyrosm)


def test_get_bounding_box_returns_polygon_for_valid_pbf():
    """#160 — get_bounding_box must read the header bbox via protobuf field
    access, not the pyrobuf-only SerializeToDict() that the backend migration
    left behind (which silently returned None for every file)."""
    from shapely.geometry import Polygon
    from pyrosm import get_data
    from pyrosm.utils import get_bounding_box

    bbox = get_bounding_box(get_data("helsinki_pbf"))
    assert isinstance(bbox, Polygon)
    minx, miny, maxx, maxy = bbox.bounds
    # Helsinki sits around lon ~25, lat ~60.
    assert 24 < minx < maxx < 26
    assert 60 < miny < maxy < 61


def test_get_bounding_box_returns_none_without_header_bbox(tmp_path):
    """#160 — a valid OSM PBF whose header carries no bbox yields None, not an
    error and not a degenerate Polygon."""
    import struct
    import zlib
    from pyrosm.utils import get_bounding_box
    from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
    from pyrosm.proto.osmformat_pb2 import HeaderBlock

    header = HeaderBlock()
    header.required_features.append("OsmSchema-V0.6")
    blob_bytes = Blob(
        zlib_data=zlib.compress(header.SerializeToString())
    ).SerializeToString()
    blob_header = BlobHeader(
        type="OSMHeader", datasize=len(blob_bytes)
    ).SerializeToString()
    pbf = tmp_path / "no_bbox.pbf"
    pbf.write_bytes(struct.pack("!L", len(blob_header)) + blob_header + blob_bytes)

    assert get_bounding_box(pbf) is None


def test_node_coordinates_decoded_at_full_precision(tmp_path):
    """#245 — node coordinates must be decoded at full float64 precision (the
    exact OSM 7-decimal values, matching GDAL/osmium), not truncated to float32.
    Uses the exact coordinate of node 623850466 from the issue."""
    import numpy as np

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node
    from pyrosm import OSM

    path = str(tmp_path / "coords.osm.pbf")
    writer = osmium.SimpleWriter(path)
    writer.add_node(
        Node(id=1, location=(-80.4410082, 26.0914866), tags={"power": "tower"})
    )
    writer.close()

    osm = OSM(path)
    gdf = osm.get_data_by_custom_criteria(
        {"power": ["tower"]}, filter_type="keep", keep_nodes=True
    )
    geom = gdf[gdf["id"] == 1].geometry.iloc[0]
    assert (geom.x, geom.y) == (-80.4410082, 26.0914866)
    # The lon/lat columns carry the same full precision (not float32).
    assert gdf["lon"].dtype == np.float64
    assert gdf["lat"].dtype == np.float64
    assert gdf.loc[gdf["id"] == 1, "lon"].iloc[0] == -80.4410082


_METADATA_COLS = {"timestamp", "version", "changeset"}


def test_keep_metadata_false_drops_way_relation_metadata():
    """#87 (Bucket A / A5) — `OSM(keep_metadata=False)` drops the way/relation
    element metadata columns (timestamp, version, changeset) while keeping the
    rows, geometries and every other column byte-for-byte identical to the
    default. Default `keep_metadata=True` is unchanged."""
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    default = OSM(fp)
    minimal = OSM(fp, keep_metadata=False)

    # Buildings and the network are way/relation features -> metadata fully gated.
    for getter in ("get_buildings", "get_network"):
        full = getattr(default, getter)()
        lean = getattr(minimal, getter)()

        # Default keeps metadata; opt-in drops exactly those columns.
        assert _METADATA_COLS & set(full.columns)
        assert not (_METADATA_COLS & set(lean.columns))
        assert set(full.columns) - set(lean.columns) == (
            _METADATA_COLS & set(full.columns)
        )
        # Everything else is identical.
        assert len(full) == len(lean)
        assert full.geometry.equals(lean.geometry)


def test_keep_metadata_must_be_bool():
    """#87 (Bucket A / A5) — keep_metadata is validated as a boolean."""
    from pyrosm import OSM, get_data

    with pytest.raises(ValueError, match="keep_metadata"):
        OSM(get_data("helsinki_pbf"), keep_metadata="yes")


def test_keep_metadata_false_drops_node_metadata():
    """#150 — `OSM(keep_metadata=False)` also skips the per-node element metadata
    while parsing, so node features (POIs) drop the timestamp/version/changeset
    columns while keeping their rows, geometries and every other column identical
    to the default. Default `keep_metadata=True` is unchanged."""
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    full = OSM(fp).get_pois()
    lean = OSM(fp, keep_metadata=False).get_pois()

    # The POI set includes node elements, so node metadata is actually exercised.
    assert (full["osm_type"] == "node").any()

    # Default keeps node metadata; opt-in drops exactly those columns.
    assert _METADATA_COLS & set(full.columns)
    assert not (_METADATA_COLS & set(lean.columns))
    assert set(full.columns) - set(lean.columns) == (_METADATA_COLS & set(full.columns))

    # Everything else (rows, geometry) is identical.
    assert len(full) == len(lean)
    assert full.geometry.equals(lean.geometry)


def test_download_builds_ssl_context_from_certifi(tmp_path, monkeypatch):
    """Downloads must build the HTTPS context from certifi's CA bundle, not the
    OS trust store: on Windows, loading the system certificate store can raise
    ssl.SSLError [ASN1: NOT_ENOUGH_DATA] (a CPython bug on a malformed store
    entry), which aborted every download-backed test on the windows runners."""
    import io
    import os
    import ssl

    import certifi

    from pyrosm.utils import download as dl

    captured = {}

    def fake_create(*args, **kwargs):
        captured["cafile"] = kwargs.get("cafile")
        # Return a bare context; do NOT load the OS trust store (that is the very
        # Windows ssl bug under test). fake_urlopen ignores the context anyway.
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def fake_urlopen(url, context=None):
        captured["context"] = context
        return io.BytesIO(b"x" * 50000)

    monkeypatch.setattr(dl.ssl, "create_default_context", fake_create)
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)

    out = dl.download(
        "https://example.invalid/data.osm.pbf",
        "data.osm.pbf",
        True,
        str(tmp_path),
    )

    # The CA bundle came from certifi, and that context was handed to urlopen.
    assert captured["cafile"] == certifi.where()
    assert isinstance(captured["context"], ssl.SSLContext)
    assert os.path.exists(out)


def test_tags_to_keep_restricts_tag_columns():
    """#87 — get_*(tags_to_keep=[...]) materializes only the requested OSM tag
    keys as columns (replacing the default tag-column set), leaving rows,
    geometries and structural columns unchanged."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))

    full = osm.get_buildings()
    lean = osm.get_buildings(tags_to_keep=["building"])

    # The requested tag column is present...
    assert "building" in lean.columns
    # ...and the default-only tag columns (e.g. addr:*) are dropped from columns.
    default_only = {c for c in full.columns if c.startswith("addr:")}
    assert default_only and not (default_only & set(lean.columns))
    # Structural columns and rows/geometry are unchanged.
    assert "id" in lean.columns and "geometry" in lean.columns
    assert len(full) == len(lean)
    assert full.geometry.equals(lean.geometry)


def test_tags_to_keep_validates_input():
    """#87 — tags_to_keep is validated (must be a list, not a bare string)."""
    from pyrosm import OSM, get_data

    with pytest.raises(ValueError):
        OSM(get_data("helsinki_pbf")).get_buildings(tags_to_keep="building")


def test_tags_to_keep_applies_to_all_feature_methods():
    """#87 — tags_to_keep restricts the tag columns on every feature method, not
    only get_buildings: the requested key is kept, the other default tag columns
    are dropped, and rows/geometry are unchanged."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    cases = [
        (osm.get_network, {}, "highway"),
        (osm.get_landuse, {}, "landuse"),
        (osm.get_natural, {}, "natural"),
        (osm.get_boundaries, {}, "boundary"),
        (osm.get_pois, {"custom_filter": {"amenity": True}}, "amenity"),
    ]
    for method, base_kwargs, kept in cases:
        name = method.__name__
        full = method(**base_kwargs)
        lean = method(**base_kwargs, tags_to_keep=[kept])
        full_cols, lean_cols = set(full.columns), set(lean.columns)
        assert kept in lean_cols, f"{name}: '{kept}' column missing"
        # Restricting only removes columns, and it removed at least one.
        assert (
            lean_cols <= full_cols
        ), f"{name}: unexpected columns {lean_cols - full_cols}"
        assert len(lean_cols) < len(full_cols), f"{name}: tag columns not restricted"
        # Rows and geometry are unchanged by the column restriction.
        assert len(full) == len(lean), f"{name}: row count changed"
        assert full.geometry.equals(lean.geometry), f"{name}: geometry changed"


def test_compact_node_store_preserves_graph_node_attributes():
    """#53 — the compact node-coordinate store (a cykhash id->index map plus
    column arrays) that replaced the per-node dict-of-dicts must reproduce the
    graph node attributes unchanged. In particular the rebuilt records yield
    Python scalars, so the metadata columns keep their int64/float64 dtypes
    rather than the narrower array dtype."""
    import numpy as np
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    assert nodes["version"].dtype == np.int64
    assert nodes["changeset"].dtype == np.int64
    assert nodes["timestamp"].dtype == np.int64
    assert nodes["lon"].dtype == np.float64
    assert nodes["lat"].dtype == np.float64
    # Every graph node carries its id and a point geometry.
    assert "id" in nodes.columns
    assert nodes.geometry.notna().all()


def test_vectorized_network_geometry_graph_consistency():
    """#53 — the batched network-geometry path must keep graph export consistent:
    every directed edge's u/v are real graph-node ids and the edges have valid
    geometries, and a bounding-box network (which routes boundary ways through
    the per-way fallback) still builds valid geometries."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    nodes, edges = osm.get_network(nodes=True)
    assert len(edges) > 0 and len(nodes) > 0
    assert "u" in edges.columns and "v" in edges.columns

    node_ids = set(nodes["id"].tolist())
    # Every edge endpoint is a known graph node (from/to ids stay consistent).
    assert set(edges["u"].tolist()).issubset(node_ids)
    assert set(edges["v"].tolist()).issubset(node_ids)
    assert edges.geometry.notna().all()
    assert edges.geometry.is_valid.all()

    # A bounding-box network exercises the boundary-completeness path.
    bbox = [24.93, 60.16, 24.96, 60.18]
    bnet = OSM(get_data("helsinki_pbf"), bounding_box=bbox).get_network()
    assert bnet is not None and len(bnet) > 0
    assert bnet.geometry.notna().all()
    assert bnet.geometry.is_valid.all()


def test_streaming_block_reader_bbox_boundary_completeness():
    """#53 — blocks are read by a streaming generator (parsed and discarded one at
    a time, not all held in a list). A bounding box read smaller than the data
    extent must still return ways straddling the box edge complete, which is
    served by re-reading the file for the boundary node pass."""
    from pyrosm import OSM, get_data

    # Sub-box inside the data extent, so network ways cross the box edge.
    bbox = [24.945, 60.170, 24.950, 60.174]
    net = OSM(get_data("helsinki_pbf"), bounding_box=bbox).get_network()
    assert net is not None and len(net) > 0
    assert net.geometry.notna().all()
    assert net.geometry.is_valid.all()

    # Straddling ways are returned whole (#236), so the kept geometry extends
    # beyond the box -- proof the second (re-read) node pass supplied the
    # out-of-box boundary vertices.
    minx, miny, maxx, maxy = bbox
    b = net.total_bounds
    assert b[0] < minx or b[1] < miny or b[2] > maxx or b[3] > maxy


def test_node_tag_columns_built_only_when_nonempty():
    """#53 — node tag columns are materialized only for keys that actually occur in
    the data (empty candidate columns are not built and then dropped). Output is
    unchanged: a filter key lands in its own column, and no materialized tag column
    is entirely empty."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    pois = osm.get_pois(custom_filter={"amenity": True, "shop": True})
    assert pois is not None and len(pois) > 0
    assert "amenity" in pois.columns and pois["amenity"].notna().any()

    # Every column that is not a structural/metadata field is a materialized tag
    # column and must carry at least one value (empties are never built).
    structural = {
        "geometry",
        "osm_type",
        "id",
        "lon",
        "lat",
        "tags",
        "version",
        "changeset",
        "timestamp",
        "visible",
    }
    for c in pois.columns:
        if c not in structural:
            assert pois[c].notna().any(), f"tag column {c!r} is entirely empty"


def test_way_tag_columns_built_only_when_nonempty():
    """#53 — way/relation tag columns are materialized only for keys that actually
    occur in the data (empty candidate columns are not built and then dropped). Output
    is unchanged: the feature key lands in its own column, leftover tags go to the JSON
    'tags' column, and no materialized tag column is entirely empty."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    structural = {
        "geometry",
        "osm_type",
        "id",
        "nodes",
        "lon",
        "lat",
        "tags",
        "version",
        "changeset",
        "timestamp",
        "visible",
    }
    for feature, getter in (
        ("building", lambda o: o.get_buildings()),
        ("landuse", lambda o: o.get_landuse()),
        ("natural", lambda o: o.get_natural()),
    ):
        gdf = getter(osm)
        assert gdf is not None and len(gdf) > 0
        assert feature in gdf.columns and gdf[feature].notna().any()
        for c in gdf.columns:
            if c not in structural:
                assert (
                    gdf[c].notna().any()
                ), f"{feature}: tag column {c!r} is entirely empty"
