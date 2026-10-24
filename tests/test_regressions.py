"""Regression tests guarding against specific bugs reappearing."""

import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_get_methods_do_not_mutate_shared_tag_config(engine):
    """#252 — get_* must not mutate the shared Conf default-tag lists, nor the
    custom_filter they are given (True used to become [True]), with either engine."""
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf

    osm = OSM(get_data("test_pbf"), engine=engine, workers=1)

    building_before = list(Conf.tags.building)
    highway_before = list(Conf.tags.highway)
    natural_before = list(Conf.tags.natural)
    custom_filter = {"amenity": True, "shop": ["bakery"]}

    osm.get_buildings()
    osm.get_network()
    osm.get_natural()
    osm.get_buildings(extra_attributes=["my_extra_attr"])
    osm.get_pois(custom_filter=custom_filter)
    osm.get_data_by_custom_criteria(custom_filter)

    assert Conf.tags.building == building_before
    assert Conf.tags.highway == highway_before
    assert Conf.tags.natural == natural_before
    assert "my_extra_attr" not in Conf.tags.building
    assert custom_filter == {"amenity": True, "shop": ["bakery"]}


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
    import pandas as pd
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

    # Each node and edge holds the columns where it has a value, in column order, and
    # equal strings are one object.
    nodes["tags"] = [{"a": "1"}, None, None]
    edges["name"] = ["Main", "Main"]
    edges["ref"] = [None, "7"]
    edges["width"] = [2.5, float("nan")]
    graph = _create_nxgraph(nodes, edges, "u", "v", "id")
    first, second = (graph.edges[u, v, 0] for u, v in [(1000, 2000), (2000, 3000)])
    assert list(first) == ["u", "v", "geometry", "name", "width"]
    assert list(second) == ["u", "v", "geometry", "name", "ref"]
    assert (first["u"], first["width"], second["ref"]) == (1000, 2.5, "7")
    assert type(first["width"]) is float and first["name"] is second["name"]
    assert [list(graph.nodes[n]) for n in (1000, 2000)] == [
        ["id", "geometry", "tags", "street_count"],
        ["id", "geometry", "street_count"],
    ]
    # A repeated column takes the values of its last copy.
    edges.insert(len(edges.columns), "name", ["Side", None], allow_duplicates=True)
    graph = _create_nxgraph(nodes, edges, "u", "v", "id")
    assert graph.edges[1000, 2000, 0]["name"] == "Side"
    assert "name" not in graph.edges[2000, 3000, 0]

    # Nodes whose only column is the id, dropped as an attribute, are all added,
    # including one that no edge touches.
    only_ids = pd.DataFrame({"id": [1000, 2000, 3000, 4000]})
    graph = _create_nxgraph(only_ids, edges, "u", "v", "id", drop_node_id_attribute=True)
    assert sorted(graph.nodes()) == [1000, 2000, 3000, 4000]


_EXCLUDED_SERVICE = {"parking", "parking_aisle", "private", "emergency_access"}


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_exclude_filter_does_not_leak_secondary_keys(engine):
    """#112 — an exclude custom_filter that lists `service` values must drop those
    ways, with either engine. The old ways filter broke on the first filter key in the
    record, so a `highway=service` way leaked because `highway` was checked before
    `service`."""
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
    osm = OSM(get_data("test_pbf"), engine=engine, workers=1)
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

    graph = OSM.to_graph(
        nodes, edges, graph_type="networkx", retain_all=True, simplify=False
    )
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
    """#118/#181 — get_network and get_graph reject an invalid filter_type, and a custom_filter
    key outside the default 'highway' tag set is accepted (and added as a column
    when present in the data)."""
    import pytest
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf

    osm = OSM(get_data("test_pbf"))

    for read in (osm.get_network, osm.get_graph):
        with pytest.raises(ValueError):
            read(custom_filter={"highway": ["footway"]}, filter_type="not-a-filter")

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

    assert len(edges) == 171
    assert edges["id"].nunique() == 171
    assert sorted(edges.columns) == [
        "bridge",
        "geometry",
        "highway",
        "id",
        "int_ref",
        "lanes",
        "length",
        "maxspeed",
        "name",
        "oneway",
        "osm_type",
        "ref",
        "surface",
        "tags",
        "timestamp",
        "version",
        "visible",
    ]
    assert edges.attrs["network_type"] == "driving"

    rows = sorted(f"{i}:{h}" for i, h in zip(edges["id"], edges["highway"].fillna("")))
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    assert digest == "9d8d6995cb96d74f00422b5bb9e09d9e844342fa16781716860fc88b422a9566"


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

    osm = OSM(get_data("helsinki_pbf"), bounding_box=[24.93, 60.16, 24.96, 60.20], engine="in_memory")
    edges = osm.get_network(network_type="all")
    assert edges is not None
    assert len(edges) == 2469


def test_bbox_straddling_building_ways_complete_not_cut():
    """#236 — building ways crossing the bounding-box edge must come back with
    their complete geometry, not cut. Pre-fix, a way straddling the edge lost its
    outside-bbox vertices and its area collapsed (some even became invalid)."""
    from pyrosm import OSM, get_data

    fp = get_data("helsinki_pbf")
    full = OSM(fp, engine="in_memory").get_buildings()
    full = full[full["osm_type"] == "way"].set_index("id")

    sub = OSM(fp, bounding_box=[24.93, 60.16, 24.945, 60.18], engine="in_memory").get_buildings()
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
    full = OSM(fp, engine="in_memory").get_buildings()
    full = full[full["osm_type"] == "way"].set_index("id")

    sub = OSM(fp, bounding_box=[24.93, 60.16, 24.945, 60.18], engine="in_memory").get_buildings()
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
    """#270 — graph_type='pandana' still works but warns, as a FutureWarning, that it is
    deprecated in favour of 'pandarm' and removed in pyrosm 1.0."""
    pytest.importorskip("pandana")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    with pytest.warns(FutureWarning, match="removed in pyrosm 1.0"):
        g = osm.to_graph(nodes, edges, graph_type="pandana")
    assert g is not None


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_parse_nodes_non_dense_pbf_matches_dense(tmp_path, engine, fresh_cache):
    """#274 — pyrosm parses non-dense PBF node groups and returns the same frames as
    for the dense encoding, with both engines. Guards the regression where
    parse_nodes' dict result was spread into the all_nodes list as bare keys,
    crashing non-dense reads with 'str object has no attribute keys', and the
    out-of-core engine returning nothing because its decoder skipped plain nodes. A
    non-dense copy of the bundled test PBF is generated with osmium."""
    import pandas as pd

    osmium = pytest.importorskip("osmium")
    from pyrosm import OSM, get_data

    src = get_data("test_pbf")
    nondense = str(tmp_path / "test_nondense.osm.pbf")
    with osmium.SimpleWriter(
        osmium.io.File(nondense, "pbf,pbf_dense_nodes=false")
    ) as writer:
        for obj in osmium.FileProcessor(src):
            writer.add(obj)

    # POIs come from standalone (tagged) nodes; a bounding box exercises the node
    # filter by box (and, in memory, the #236 boundary-completion pass).
    bounds = [26.94, 60.525, 26.96, 60.535]
    reads = [
        ("get_network", {}),
        ("get_buildings", {}),
        ("get_pois", {}),
        ("get_buildings", {"bounding_box": bounds}),
    ]
    for method, options in reads:
        dense = getattr(OSM(src, engine=engine, workers=1, **options), method)()
        frame = getattr(OSM(nondense, engine=engine, workers=1, **options), method)()
        assert frame is not None and len(frame) > 0
        pd.testing.assert_frame_equal(
            frame.drop(columns="geometry"), dense.drop(columns="geometry")
        )
        assert (frame.geometry.to_wkb() == dense.geometry.to_wkb()).all()


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
    entry), which aborted every download-backed test on the windows runners.
    They also identify themselves with pyrosm's User-Agent."""
    import io
    import ssl

    import certifi

    from pyrosm.utils import download as dl

    captured = {}

    def fake_create(*args, **kwargs):
        captured["cafile"] = kwargs.get("cafile")
        # Return a bare context; do NOT load the OS trust store (that is the very
        # Windows ssl bug under test). fake_urlopen ignores the context anyway.
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def fake_urlopen(request, context=None, timeout=None):
        captured["context"] = context
        captured["user_agent"] = request.get_header("User-agent")
        response = io.BytesIO(b"x" * 50000)
        response.headers = {}
        return response

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
    assert captured["user_agent"] == dl.USER_AGENT
    assert Path(out).exists()


@pytest.mark.parametrize(
    "response, previous, error",
    [
        ("reset", None, "connection reset"),
        ("reset", b"old", "connection reset"),
        ("short", b"old", "stopped after 5 of 100 bytes"),
        ("empty", b"old", "was empty"),
        ("503", b"old", "HTTP Error 503"),
        ("loop", b"old", "infinite loop"),
    ],
)
def test_failed_download_leaves_no_partial_file(
    tmp_path, monkeypatch, response, previous, error
):
    """A download that breaks midway, ends before the announced size or is empty must
    not leave a partial file that later calls would reuse, and a failed update keeps
    the previous copy. A network failure or busy server is tried three times, a redirect
    loop once, and the last error is raised as a DownloadError."""
    import io
    from urllib.error import HTTPError

    from pyrosm.exceptions import DownloadError, ExtractDownloadError
    from pyrosm.utils import download as dl

    class Reset(io.BytesIO):
        headers = {}

        def read(self, *args):
            raise OSError("connection reset")

    calls = []

    def urlopen(request, context=None, timeout=None):
        calls.append(request.full_url)
        if response == "503":
            raise HTTPError(request.full_url, 503, "Service Unavailable", {}, None)
        if response == "loop":
            # What urllib raises when a server keeps redirecting a URL to itself.
            reason = "The HTTP server returned a redirect error that would lead to an infinite loop."
            raise HTTPError(request.full_url, 301, reason, {}, None)
        if response == "reset":
            return Reset()
        body = io.BytesIO(b"x" * 5 if response == "short" else b"")
        body.headers = {"Content-Length": "100"} if response == "short" else {}
        return body

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    target = tmp_path / "x.osm.pbf"
    if previous:
        target.write_bytes(previous)
    url = "https://example.invalid/x.osm.pbf"
    with pytest.raises(DownloadError, match=error) as info:
        dl.download(url, "x.osm.pbf", True, str(tmp_path))
    # Still caught by handlers of the ValueError and OSError download used to raise.
    assert isinstance(info.value, ValueError) and isinstance(info.value, OSError)
    assert isinstance(info.value, ExtractDownloadError)
    got = (info.value.url, info.value.status, info.value.attempts, len(calls))
    # A redirect loop is not retried; the other failures are tried three times.
    status, tries = {"503": (503, 3), "loop": (301, 1)}.get(response, (None, 3))
    assert got == (url, status, tries, tries)
    assert sorted(p.name for p in tmp_path.iterdir()) == (
        ["x.osm.pbf"] if previous else []
    )
    if previous:
        assert target.read_bytes() == previous


def test_download_does_not_retry_a_full_disk(tmp_path, monkeypatch):
    """An error writing the local file (a full disk) propagates as itself after one
    attempt; it is not retried or reported as a download failure."""
    import errno
    import io

    from pyrosm.utils import download as dl

    class Full(io.BytesIO):
        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

    calls = []
    monkeypatch.setattr(
        dl.urllib.request,
        "urlopen",
        lambda request, context=None, timeout=None: calls.append(1)
        or _headered(io.BytesIO(b"x" * 50000)),
    )
    monkeypatch.setattr(dl, "write_atomic", lambda path, write: write(Full()))
    with pytest.raises(OSError) as info:
        dl.download("https://example.invalid/x.osm.pbf", "x.osm.pbf", True, tmp_path)
    assert type(info.value) is OSError and info.value.errno == errno.ENOSPC
    assert calls == [1]


def test_download_creates_missing_target_dir(tmp_path, monkeypatch):
    """download(target_dir=...) creates a directory that does not exist yet."""
    import io

    from pyrosm.utils import download as dl

    monkeypatch.setattr(
        dl.urllib.request,
        "urlopen",
        lambda url, context=None, timeout=None: _headered(io.BytesIO(b"x" * 100)),
    )
    missing = tmp_path / "does" / "not" / "exist"
    out = dl.download(
        "https://example.invalid/x.osm.pbf", "x.osm.pbf", False, str(missing)
    )
    assert (
        Path(out) == missing.resolve() / "x.osm.pbf" and Path(out).stat().st_size == 100
    )


def _headered(response, headers=None):
    response.headers = headers or {}
    return response


@pytest.mark.parametrize("length", [None, "50000", "9" * 5000, "\u00b2"])
def test_download_update_replaces_existing_file(
    tmp_path, monkeypatch, caplog, capsys, length
):
    """download(update=True) replaces the cached file with the new download, and reports it
    through the pyrosm logger, not stdout. A Content-Length that is not a plain number is
    ignored."""
    import logging

    import io

    from pyrosm.utils import download as dl

    target = tmp_path / "x.osm.pbf"
    target.write_bytes(b"old")
    fresh = b"x" * 50000  # large enough that download's empty-file guard passes

    monkeypatch.setattr(dl.ssl, "create_default_context", lambda *a, **k: None)
    monkeypatch.setattr(
        dl.urllib.request,
        "urlopen",
        lambda url, context=None, timeout=None: _headered(
            io.BytesIO(fresh), {"Content-Length": length} if length else None
        ),
    )

    caplog.set_level(logging.INFO, logger="pyrosm")
    out = dl.download(
        "https://example.invalid/x.osm.pbf", "x.osm.pbf", True, str(tmp_path)
    )
    assert Path(out).read_bytes() == fresh
    assert "download file=x.osm.pbf megabytes=" in caplog.text
    assert capsys.readouterr().out == ""


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

    # The region extract is used because the small helsinki_pbf has only
    # incomplete boundaries (all dropped, #154), leaving get_boundaries empty. The
    # out-of-core engine's tags_to_keep is covered by its parity tests.
    osm = OSM(get_data("helsinki_region_pbf"), engine="in_memory")
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
    """#53 — the compact node-coordinate store (an id->index hash map plus
    column arrays) that replaced the per-node dict-of-dicts must reproduce the
    graph node attributes unchanged. In particular the rebuilt records yield
    Python scalars, so the metadata columns keep their int64/float64 dtypes
    rather than the narrower array dtype."""
    import numpy as np
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine="in_memory")
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
    net = OSM(get_data("helsinki_pbf"), bounding_box=bbox, engine="in_memory").get_network()
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


def test_incomplete_boundaries_dropped_not_force_closed():
    """#154 — administrative boundaries whose member ways run off the PBF extent
    cannot form a closed ring. They are dropped (matching how osmium and GDAL skip
    areas they cannot assemble), not force-closed into polygons with a spurious
    straight edge bridging the gap (the stray lines that slashed across
    get_boundaries() plots). Every boundary in the bundled Helsinki extract crosses
    its edge, so none can be assembled and get_boundaries returns nothing; pre-fix
    it returned 8 force-closed stray polygons."""
    from pyrosm import OSM, get_data

    assert OSM(get_data("helsinki_pbf")).get_boundaries() is None


def test_get_data_disambiguates_ambiguous_region_name(monkeypatch):
    """#162 — 'georgia' is both a country (europe) and a US state
    (north-america/us), which resolve to different Geofabrik files. The bare name
    used to silently return the country. It is now rejected as ambiguous, and a
    region-qualified name resolves each one; get_data routes a qualified name to
    the correct file."""
    import pyrosm.data as data
    from pyrosm.data import search_source

    # The two regions resolve to different files.
    state = search_source("usa/georgia")["url"]
    country = search_source("europe/georgia")["url"]
    assert state.endswith("north-america/us/georgia-latest.osm.pbf")
    assert country.endswith("europe/georgia-latest.osm.pbf")
    assert state != country

    # The bare, ambiguous name raises with guidance (search_source and get_data).
    with pytest.raises(ValueError, match="ambiguous"):
        search_source("georgia")
    with pytest.raises(ValueError, match="ambiguous"):
        data.get_data("georgia")

    # A benign collision (same file reachable via two groupings) is not ambiguous.
    assert search_source("oxfordshire")["url"].endswith(
        "england/oxfordshire-latest.osm.pbf"
    )

    # An unknown region qualifier is rejected.
    with pytest.raises(ValueError, match="Could not retrieve"):
        search_source("nowhere/georgia")

    # get_data routes a region-qualified name to the resolved file (download stubbed).
    captured = {}

    def fake_retrieve(d, update, directory, **net):
        captured["url"] = d["url"]
        return "/fake/path"

    monkeypatch.setattr(data, "retrieve", fake_retrieve)
    data.get_data("usa/georgia")
    assert captured["url"].endswith("north-america/us/georgia-latest.osm.pbf")

    # A source-registry entry that is neither a region list nor a subregion dict is
    # skipped rather than crashing the lookup.
    monkeypatch.setattr(
        data.sources, "available", {**data.sources.available, "_stray": 42}
    )
    assert data._find_sources("finland")


def test_get_data_dispatch_and_lazy_attrs(monkeypatch):
    """Exercise get_data's dispatch table -- bundled files, gist-hosted test data,
    the name-normalisation variants (spaces, underscores, dashes) and the
    not-available error -- plus the deprecated get_path alias and the lazy
    pyrosm.data attribute loader."""
    import pyrosm.data as data

    # Non-string input is rejected up front.
    with pytest.raises(ValueError, match="should be text"):
        data.get_data(123)

    # An unknown name reports that it is not available.
    with pytest.raises(ValueError, match="is not available"):
        data.get_data("not_a_real_place_xyz")

    # Bundled data is returned as a local path, no download.
    assert Path(data.get_data("test_pbf")).exists()

    # Stub the network so the retrieve-backed branches (and retrieve itself) run
    # offline; capture the resolved filename for each.
    seen = []

    def fake_download(url, filename, update, target_dir, **net):
        seen.append(filename)
        return "/fake/" + filename

    monkeypatch.setattr(data, "download", fake_download)

    # Gist-hosted test datasets dispatch to retrieve directly.
    for name in (
        "helsinki_region_pbf",
        "helsinki_history_pbf",
        "helsinki_test_history_pbf",
        "ulanbator_test_pbf",
    ):
        data.get_data(name)

    # A plain region name resolves through search_source.
    data.get_data("finland")

    # Name-normalisation variants: spaces removed (a city), space->underscore and
    # dash->underscore (a region).
    data.get_data("Brandenburg Havel")
    data.get_data("new zealand")
    data.get_data("new-zealand")
    assert seen  # everything above went through the stubbed download

    # get_path is the deprecated alias for get_data, removed in pyrosm 1.0.
    with pytest.warns(FutureWarning, match="get_path.*pyrosm 1.0"):
        assert Path(data.get_path("test_pbf")).exists()

    # The optional geo helpers are exposed lazily via module __getattr__.
    assert callable(data.get_data_by_bbox)
    assert callable(data.geocode)
    assert callable(data.get_data_by_geocoding)
    with pytest.raises(AttributeError):
        data.this_attr_does_not_exist


def test_engine_collect_relation_ways_handles_no_way_members(tmp_path):
    """A building relation whose members are all non-way (node/relation) leaves the
    way-member id set empty; the out-of-core way lookup must short-circuit to None
    (relation dropped) instead of indexing the empty id array."""
    import numpy as np
    from pyrosm.engine.collect import _collect_relation_ways

    shard = str(tmp_path / "shard_0.npz")
    np.savez(
        shard,
        all_id=np.array([1, 2, 3], dtype=np.int64),
        all_refs=np.array([10, 11, 12], dtype=np.int64),
        all_refs_off=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    assert _collect_relation_ways([shard], np.empty(0, np.int64)) is None


def test_networkx_export_survives_osmnx_round_trip(tmp_path):
    """#370 — an 'osmnx_compatible' graph must work with OSMnx. 'key' must not be
    an edge attribute (it collides with the parallel-edge key of add_edge) and
    'osmid' must not be repeated as a node attribute (it is the key of the node)."""
    pytest.importorskip("networkx")
    osmnx = pytest.importorskip("osmnx")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    for simplify in [False, True]:
        graph = osm.to_graph(nodes, edges, graph_type="networkx", simplify=simplify)

        assert all("key" not in attr for *_, attr in graph.edges(data=True))
        assert all("osmid" not in attr for _, attr in graph.nodes(data=True))
        assert graph.graph["simplified"] is simplify

        # Node tags are stored as a JSON string, like the edge tags, as OSMnx
        # cannot handle a dict as a node attribute.
        node_tags = [attr["tags"] for _, attr in graph.nodes(data=True) if "tags" in attr]
        assert not any(isinstance(tags, dict) for tags in node_tags)
        assert any(isinstance(tags, str) for tags in node_tags)

        # The OSMnx entry points a user runs on an exported graph.
        projected = osmnx.projection.project_graph(graph)
        osmnx.convert.to_undirected(graph)
        osmnx.convert.graph_to_gdfs(graph)
        osmnx.io.save_graph_geopackage(graph, tmp_path / f"graph_{simplify}.gpkg")
        osmnx.simplification.consolidate_intersections(projected, tolerance=10)
        osmnx.stats.basic_stats(graph)
        osmnx.distance.nearest_nodes(graph, 26.94, 60.52)
        osmnx.routing.shortest_path(
            graph, list(graph.nodes)[0], list(graph.nodes)[-1], weight="length"
        )
        osmnx.truncate.truncate_graph_bbox(graph, (26.9, 60.5, 27.0, 60.6))

        if simplify:
            # A pyrosm-simplified graph reports itself as simplified, so OSMnx
            # refuses to simplify it again instead of failing on the merged
            # list-valued attributes.
            with pytest.raises(osmnx._errors.GraphSimplificationError):
                osmnx.simplification.simplify_graph(graph)
        else:
            osmnx.simplification.simplify_graph(graph)


def test_networkx_export_keeps_parallel_edges():
    """#370 — parallel edges between the same pair of nodes must get separate keys.
    They were all written with key 0, so all but the last one were dropped."""
    pytest.importorskip("networkx")
    from pyrosm import OSM, get_data
    from pyrosm.graph_connectivity import get_connected_edges
    from pyrosm.graph_simplify import simplify_graph
    from pyrosm.graphs import get_directed_edges

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    # The steps to_graph(simplify=True) runs before building the graph.
    directed_nodes, directed_edges = get_directed_edges(nodes, edges)
    directed_nodes, directed_edges = simplify_graph(
        directed_nodes,
        directed_edges,
        from_id_col="u",
        to_id_col="v",
        node_id_col="id",
    )
    directed_nodes, directed_edges = get_connected_edges(directed_nodes, directed_edges)
    assert directed_edges.duplicated(subset=["u", "v"], keep=False).any()

    graph = osm.to_graph(nodes, edges, graph_type="networkx", simplify=True)
    assert graph.number_of_edges() == len(directed_edges)

    # Within each pair of nodes the keys run 0, 1, ...
    parallel_pairs = [
        (u, v) for u, v in graph.edges() if graph.number_of_edges(u, v) > 1
    ]
    assert parallel_pairs
    for u, v in parallel_pairs:
        assert sorted(graph[u][v]) == list(range(graph.number_of_edges(u, v)))


def test_networkx_export_accepts_edges_with_gaps_in_index():
    """#370 — the edge attributes are read positionally: a filtered edges
    GeoDataFrame has gaps in its index, which the label-based read could not handle."""
    pytest.importorskip("networkx")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    subset = edges[edges["highway"] == "footway"]
    assert subset.index.max() > len(subset)

    graph = osm.to_graph(
        nodes,
        subset,
        graph_type="networkx",
        network_type="walking",
        retain_all=True,
        simplify=False,
    )
    # Walking edges are bidirectional, so each row gives an edge to both directions.
    assert graph.number_of_edges() == 2 * len(subset)


def test_directed_edges_handle_duplicate_index_labels():
    """#370 — the one-way masks are applied positionally. Selecting them by index
    label raised 'IndexError: indices are out-of-bounds' on an edges GeoDataFrame
    with duplicate labels (e.g. two edge frames concatenated)."""
    pytest.importorskip("networkx")
    import pandas as pd
    from pyrosm import OSM, get_data
    from pyrosm.graphs import get_directed_edges

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)

    # Include an edge that is only allowed to the opposite direction.
    edges = edges.copy()
    edges.loc[edges.index[0], "oneway"] = "-1"
    against_u = edges["u"].iloc[0]
    against_v = edges["v"].iloc[0]

    duplicated = pd.concat([edges, edges])
    assert duplicated.index.has_duplicates

    _, directed = get_directed_edges(nodes, duplicated, network_type="driving")
    graph = osm.to_graph(
        nodes,
        duplicated,
        graph_type="networkx",
        network_type="driving",
        retain_all=True,
        simplify=False,
    )
    assert graph.number_of_edges() == len(directed)

    # The against-direction edge travels from v to u, not from u to v.
    assert graph.has_edge(against_v, against_u)
    assert not graph.has_edge(against_u, against_v)


def test_networkx_export_renames_edge_tag_named_key():
    """#370 — an edge attribute named 'key' collides with the parallel-edge key of
    the graph, so a tag with that name is surfaced as 'key_tag'."""
    pytest.importorskip("networkx")
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    nodes, edges = osm.get_network(nodes=True)
    edges["key"] = "value of a tag named key"

    graph = osm.to_graph(nodes, edges, graph_type="networkx")
    edge_attributes = [attr for *_, attr in graph.edges(data=True)]
    assert edge_attributes
    assert all("key" not in attr for attr in edge_attributes)
    assert all(
        attr["key_tag"] == "value of a tag named key" for attr in edge_attributes
    )


def test_driving_network_excludes_service_roads_by_default():
    """#369 — 'driving' follows OSMnx 'drive' and leaves out service roads;
    'driving+service' follows OSMnx 'drive_service' and keeps them, apart from the
    service roads that only provide parking, private or emergency access."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))

    driving = osm.get_network(network_type="driving")
    assert "service" not in set(driving["highway"].dropna())

    with_service = osm.get_network(network_type="driving+service")
    assert "service" in set(with_service["highway"].dropna())
    excluded = {"parking", "parking_aisle", "private", "emergency_access"}
    assert not (set(with_service["service"].dropna()) & excluded)

    # Service roads are the only difference between the two.
    assert set(driving["id"]) < set(with_service["id"])


def test_network_types_exclude_private_access():
    """#369 — every mode-specific network leaves out access=private ways, as OSMnx
    does. The unrestricted 'all' keeps them; 'all_public' does not."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))

    for network_type in ("driving", "driving+service", "walking", "cycling"):
        edges = osm.get_network(network_type=network_type)
        # The column is missing altogether when no way of the network is access-tagged.
        values = set(edges["access"].dropna()) if "access" in edges.columns else set()
        assert "private" not in values

    assert "private" in set(osm.get_network(network_type="all")["access"].dropna())
    public = osm.get_network(network_type="all_public")
    assert "private" not in set(public["access"].dropna())


def test_osmnx_network_type_aliases():
    """#369 — the OSMnx spellings select the same network as the pyrosm names."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    pairs = [
        ("drive", "driving"),
        ("drive_service", "driving+service"),
        ("walk", "walking"),
        ("bike", "cycling"),
    ]
    for alias, name in pairs:
        assert set(osm.get_network(network_type=alias)["id"]) == set(
            osm.get_network(network_type=name)["id"]
        )


def test_unknown_network_type_raises():
    """#369 — the accepted network types are listed explicitly. 'driving_psv' was
    accepted because the list was built from the filter attribute names, and it fell
    through the resolver to return the whole unfiltered network."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    for network_type in ("driving_psv", "driving+psv", "not_a_network"):
        with pytest.raises(ValueError, match="network_type"):
            osm.get_network(network_type=network_type)


def test_all_network_excludes_areas_and_lifecycle_ways():
    """#369 — 'all' follows OSMnx and leaves out areas and the ways that are not (or
    no longer) a physical way. It used to apply no filter at all."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    edges = osm.get_network(network_type="all")

    areas = set(edges["area"].dropna()) if "area" in edges.columns else set()
    assert "yes" not in areas
    lifecycle = {"construction", "proposed", "planned", "razed", "abandoned", "no"}
    assert not (set(edges["highway"].dropna()) & lifecycle)
    assert "platform" not in set(edges["highway"].dropna())


def test_network_filters_match_multi_value_tags():
    """#369 — a tag can hold several values ("access=private;delivery"). OSMnx matches
    its filters against the raw value with a regular expression, so the predefined
    networks exclude such a way as well. A custom_filter keeps matching exactly, which
    is what the regex filter values of #116 are for."""
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf
    from pyrosm.config.osm_filters import get_osm_filter
    from pyrosm.data_filter import element_should_be_kept
    from pyrosm.data_manager import _get_osm_ways_and_relations

    ways = [
        {"id": 1, "highway": "residential", "nodes": [1, 2]},
        {"id": 2, "highway": "residential", "access": "private", "nodes": [2, 3]},
        {
            "id": 3,
            "highway": "residential",
            "access": "private;delivery",
            "nodes": [3, 4],
        },
        {
            "id": 4,
            "highway": "residential",
            "access": "delivery;destination",
            "nodes": [4, 5],
        },
    ]
    driving = get_osm_filter("driving")

    # The way path (get_network) and the node/relation path (also used by the
    # out-of-core and streaming readers) both drop the multi-value private way.
    kept, _ = _get_osm_ways_and_relations(
        ways, None, ["highway"], list(Conf.tags.highway), driving, "exclude"
    )[:2]
    assert sorted(int(i) for i in kept["id"]) == [1, 4]
    assert [
        element_should_be_kept(w, ["highway"], driving, "exclude") for w in ways
    ] == [
        True,
        False,
        False,
        True,
    ]

    # A custom_filter is unaffected: the bundled extract has surface="paved;cobblestone",
    # which an exact-value filter does not select.
    osm = OSM(get_data("helsinki_pbf"))
    literal = osm.get_network(
        custom_filter={"surface": ["cobblestone"]}, filter_type="keep"
    )
    assert "paved;cobblestone" not in set(literal["surface"].dropna())


def test_network_filters_match_osmnx_way_filters():
    """#369 — the values each network type excludes, as OSMnx's way filters exclude
    them. OSMnx writes its exclusions as unanchored regular expressions, so its "motor"
    stands for the motor/motorway/motorway_link values written out here. The exceptions:
    the access tags follow OSM's access hierarchy, and driving also excludes busway,
    ladder and trail."""
    from pyrosm.config.osm_filters import get_osm_filter

    lifecycle = {
        "abandoned",
        "construction",
        "no",
        "planned",
        "platform",
        "proposed",
        "raceway",
        "razed",
        "rest_area",
        "services",
    }
    motor = {"motor", "motorway", "motorway_link"}
    drive_highway = lifecycle | {
        "bridleway",
        "bus_guideway",
        "busway",
        "corridor",
        "cycleway",
        "elevator",
        "escalator",
        "footway",
        "ladder",
        "path",
        "pedestrian",
        "steps",
        "track",
        "trail",
    }
    expected = {
        "driving": {
            "area": {"yes"},
            "highway": drive_highway | {"service"},
            "service": {
                "alley",
                "driveway",
                "emergency_access",
                "parking",
                "parking_aisle",
                "private",
            },
        },
        "driving+service": {
            "area": {"yes"},
            "highway": drive_highway,
            "service": {"emergency_access", "parking", "parking_aisle", "private"},
        },
        "walking": {
            "area": {"yes"},
            "highway": lifecycle | {"bus_guideway", "cycleway"} | motor,
            "service": {"private"},
            "sidewalk": {"separate"},
            "sidewalk:both": {"separate"},
            "sidewalk:left": {"separate"},
            "sidewalk:right": {"separate"},
        },
        "cycling": {
            "area": {"yes"},
            "highway": lifecycle
            | {"bus_guideway", "corridor", "elevator", "escalator", "footway", "steps"}
            | motor,
            "service": {"private"},
        },
        # OSMnx keeps bus_guideway and the privately accessible ways in 'all'.
        "all": {"area": {"yes"}, "highway": lifecycle},
        "all_public": {
            "area": {"yes"},
            "highway": lifecycle,
            "service": {"private"},
        },
    }
    # The access tags each network type reads, from the most specific one.
    vehicle = ("motorcar", "motor_vehicle", "vehicle", "access")
    access = {
        "driving": vehicle,
        "driving+service": vehicle,
        "walking": ("foot", "access"),
        "cycling": ("bicycle", "vehicle", "access"),
        "all": None,
        "all_public": ("access",),
    }

    for network_type, exclusions in expected.items():
        network_filter = dict(get_osm_filter(network_type))
        rule = network_filter.pop("access", None)
        assert (rule and rule.keys) == access[network_type], network_type
        assert {
            k: set(v) for k, v in network_filter.items()
        } == exclusions, network_type


@pytest.mark.parametrize(
    "network_type, access_tags, kept",
    [
        ("walking", {"access": "no"}, False),
        ("walking", {"access": "no", "foot": "yes"}, True),
        ("walking", {"access": "no", "foot": "private"}, False),
        ("walking", {"access": "private", "foot": "yes"}, True),
        ("walking", {"access": "private;delivery"}, False),
        ("cycling", {"access": "no", "vehicle": "yes"}, True),
        ("cycling", {"vehicle": "no", "bicycle": "designated"}, True),
        ("driving", {"access": "no", "motor_vehicle": "yes"}, True),
        ("driving", {"motor_vehicle": "no", "motorcar": "yes"}, True),
        ("driving", {"access": "yes", "motorcar": "no"}, False),
        ("all_public", {"access": "no"}, False),
    ],
)
def test_network_access_follows_the_most_specific_tag(network_type, access_tags, kept):
    """The predefined networks follow OSM's access hierarchy: a tag of the network's
    mode overrides the general access tag, the most specific tag present decides, and
    the way is excluded when it is 'no' or 'private'. access=no used to be kept, so the
    Suomenlinna maintenance tunnel (access=no, no foot tag) joined the island to the
    mainland in the walking network of Helsinki."""
    from pyrosm.config import Conf
    from pyrosm.config.osm_filters import get_osm_filter
    from pyrosm.data_filter import element_should_be_kept
    from pyrosm.data_manager import _get_osm_ways_and_relations

    way = {"id": 1, "highway": "residential", "nodes": [1, 2], **access_tags}
    network_filter = get_osm_filter(network_type)
    # The way path of the in-memory reader, and the element path of the out-of-core one.
    ways = _get_osm_ways_and_relations(
        [way], None, ["highway"], list(Conf.tags.highway), network_filter, "exclude"
    )[0]
    assert (ways is not None) == kept
    assert element_should_be_kept(way, ["highway"], network_filter, "exclude") == kept


@pytest.mark.parametrize("kernel", ["cython", "reference"])
def test_simplify_keeps_one_of_duplicate_ways(monkeypatch, kernel):
    """Two one-way OSM ways overlapping node for node (way 11 duplicates the end of
    way 10 on 3->4->5, as on Vääksyntie, Helsinki, in 2018-2024) collapse into one
    simplified edge built from the first way, as osmnx.simplification.simplify_graph
    does, instead of two parallel copies."""
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString, Point

    import pyrosm.graph_simplify as gs

    if kernel == "reference":
        monkeypatch.setattr(gs, "_cython_walk_chains", None)
    rows = [(1, 2, 10), (2, 3, 10), (3, 4, 10), (4, 5, 10), (3, 4, 11), (4, 5, 11)]
    nodes = GeoDataFrame(
        {"id": [1, 2, 3, 4, 5], "geometry": [Point(i - 1, 0) for i in range(1, 6)]},
        crs="epsg:4326",
    )
    edges = GeoDataFrame(
        {
            "u": [u for u, _, _ in rows],
            "v": [v for _, v, _ in rows],
            "id": [way for _, _, way in rows],
            "length": [1.0] * len(rows),
            "geometry": [LineString([(u - 1, 0), (v - 1, 0)]) for u, v, _ in rows],
        },
        crs="epsg:4326",
    )
    _, simplified = gs.simplify_graph(nodes, edges)
    assert sorted(
        zip(simplified["u"], simplified["v"], simplified["id"], simplified["length"])
    ) == [(1, 3, 10, 2.0), (3, 5, 10, 2.0)]


def test_merge_pbf_ignores_metadata_only_changes(tmp_path, monkeypatch):
    """A sync client such as OneDrive touches a file's metadata soon after it is
    written, which changes its ctime but not its contents; merge_pbf must not
    reject such an input as changed during the merge."""
    import os
    import shutil
    import time

    import pyrosm.pbf_export as pbf_export
    from pyrosm import get_data, merge_pbf

    def touch_metadata(path):
        mode = os.stat(path).st_mode
        os.chmod(path, mode ^ 0o040)
        os.chmod(path, mode)

    source = tmp_path / "a.osm.pbf"
    shutil.copy(get_data("helsinki_pbf"), source)
    before = os.stat(source).st_ctime_ns
    touch_metadata(source)
    if os.stat(source).st_ctime_ns == before:
        pytest.skip("chmod does not change the ctime on this platform")

    touched = []

    class TouchedInput(pbf_export._SortedInput):
        def __init__(self, filepath, progress=None):
            # Repeat until the ctime moves, for filesystems with coarse timestamps.
            before = os.stat(filepath).st_ctime_ns
            deadline = time.monotonic() + 3
            while (
                os.stat(filepath).st_ctime_ns == before and time.monotonic() < deadline
            ):
                touch_metadata(filepath)
                time.sleep(0.01)
            touched.append(os.stat(filepath).st_ctime_ns != before)
            super().__init__(filepath, progress)

    monkeypatch.setattr(pbf_export, "_SortedInput", TouchedInput)
    out = merge_pbf([str(source)], str(tmp_path / "out.osm.pbf"))
    assert touched and all(touched)
    assert os.path.getsize(out) > 0


@pytest.mark.parametrize(
    "marked, error, outcome",
    [
        (
            True,
            'DownloadError("Geofabrik is down", url="u", status=404, attempts=1)',
            "skipped",
        ),
        (True, "AssertionError('a real failure')", "failed"),
        (
            False,
            'DownloadError("Geofabrik is down", url="u", status=404, attempts=1)',
            "failed",
        ),
    ],
)
def test_live_download_test_skips_only_when_the_service_fails(
    pytester, monkeypatch, marked, error, outcome
):
    """A live_download test that fails because the outside service failed is skipped; any
    other failure, or the same error in an unmarked test, still fails."""
    monkeypatch.setenv("RUN_DOWNLOAD_TESTS", "true")
    pytester.makeconftest(
        (Path(__file__).parent / "conftest.py")
        .read_text()
        .replace('pytest_plugins = ["pytester"]', "")
    )
    marker = "@pytest.mark.live_download\n" if marked else ""
    pytester.makepyfile(
        "import pytest\n"
        "from pyrosm.exceptions import DownloadError\n\n"
        f"{marker}def test_download():\n"
        f"    raise {error}\n"
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "-r", "s")
    result.assert_outcomes(**{outcome: 1})
    if outcome == "skipped":
        result.stdout.fnmatch_lines(["*The outside service failed: Geofabrik is down*"])


def test_get_mask_by_osmid_without_np_bool(monkeypatch):
    """#377: get_mask_by_osmid (used by create_nodes_gdf with osmids_to_keep) works where
    NumPy has no ``np.bool`` (NumPy 1.24-1.26)."""
    import numpy as np

    from pyrosm.data_filter import get_mask_by_osmid

    monkeypatch.delattr(np, "bool", raising=False)
    mask = get_mask_by_osmid(
        np.array([1, 2, 3], dtype=np.int64), np.array([2], dtype=np.int64)
    )
    assert mask.tolist() == [False, True, False]


def test_node_locations_gather_from_an_empty_store():
    """#378: gather on an empty coordinate store marks every id absent (-1, NaN) instead of
    raising IndexError, and leaves found ids' coordinates unchanged."""
    import numpy as np
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    def store(ids, lon, lat):
        frame = {"id": np.array(ids, dtype=np.int64), "lon": lon, "lat": lat}
        return NodeLocations(pd.DataFrame(frame).astype({"lon": float, "lat": float}))

    idx, lon, lat = store([], [], []).gather(np.array([123], dtype=np.int64))
    assert idx.tolist() == [-1] and np.isnan(lon).all() and np.isnan(lat).all()
    idx, lon, lat = store([7], [24.9], [60.2]).gather(np.array([7, 8], dtype=np.int64))
    assert idx.tolist() == [0, -1]
    assert lon[0] == 24.9 and lat[0] == 60.2 and np.isnan(lon[1]) and np.isnan(lat[1])


def test_node_columns_come_in_a_fixed_order():
    """Readers that return nodes put the node columns in one fixed order, the one the
    out-of-core engine uses, instead of an order that changed with Python's hash seed."""
    from pyrosm import OSM, get_data

    fp = get_data("test_pbf")
    columns = list(OSM(fp, engine="in_memory").get_pois().columns)
    node_columns = ["id", "lon", "lat", "tags", "visible", "version", "timestamp"]
    assert columns[:8] == node_columns + ["changeset"]
    assert columns == list(OSM(fp, engine="out_of_core").get_pois().columns)


def test_write_pbf_frame_without_tag_columns_keeps_tags(tmp_path):
    """#383: a frame that names elements by id only, with no tag columns, edits no tags,
    so a subset written from it keeps the tags of every node, way and relation in it."""
    import geopandas as gpd

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    buildings, pois = osm.get_buildings(), osm.get_pois()
    way = buildings[buildings["osm_type"] == "way"].iloc[0]
    relation = buildings[buildings["osm_type"] == "relation"].iloc[0]
    node = pois[(pois["osm_type"] == "node") & pois["name"].notna()].iloc[0]
    rows = [way, relation, node]
    frame = gpd.GeoDataFrame(
        {"id": [r["id"] for r in rows], "osm_type": [r["osm_type"] for r in rows]},
        geometry=[None] * 3,
        crs="EPSG:4326",
    )
    out = tmp_path / "subset.osm.pbf"
    osm.write_pbf(frame, str(out), subset_only=True)

    back = OSM(str(out))
    written = back.get_buildings().set_index(["osm_type", "id"])["building"]
    assert written[("way", way["id"])] == way["building"]
    assert written[("relation", relation["id"])] == relation["building"]
    names = back.get_pois().set_index(["osm_type", "id"])["name"]
    assert names[("node", node["id"])] == node["name"]


def test_history_state_is_each_elements_latest_version():
    """#379: an element's state at a timestamp is its latest version, taken whole: a tag
    that the newest version removed stays missing instead of coming back from an older
    version, and an element whose latest version is a deletion is left out."""
    import pandas as pd

    from pyrosm.data_filter import get_latest_version

    versions = pd.DataFrame(
        {
            "version": [1, 1, 2, 1, 2],
            "id": [2, 1, 1, 3, 3],
            "visible": [True, True, True, True, False],
            "name": ["b", "old", None, "c", None],
        }
    )
    latest = get_latest_version(versions)
    assert list(latest.columns) == ["id", "version", "visible", "name"]
    assert latest["id"].tolist() == [1, 2] and latest["version"].tolist() == [2, 1]
    assert pd.isna(latest.loc[0, "name"]) and latest.loc[1, "name"] == "b"


def test_history_read_with_a_bounding_box_returns_the_state_in_the_box():
    """A history read with a bounding box returns the elements whose state at the
    timestamp lies in the box, at that state: none deleted by then, none that had left
    the box, and nodes at their position at that time."""
    import numpy as np
    import pandas as pd

    from pyrosm import get_data
    from pyrosm.pbfreader import parse_osm_data
    from pyrosm.utils import datetime_to_unix_time

    fp = get_data("helsinki_test_history_pbf")
    when = datetime_to_unix_time(pd.to_datetime("2014-01-01", utc=True))
    nodes, ways, _, _, _, _ = parse_osm_data(fp, None, False, None)
    columns = ("id", "timestamp", "visible", "lon", "lat")
    node_versions = pd.DataFrame({c: np.asarray(nodes[c]) for c in columns})
    state = node_versions[node_versions["timestamp"] <= when]
    state = state.drop_duplicates("id", keep="last")
    state = state[state["visible"]]
    (xmin, ymin), (xmax, ymax) = state[["lon", "lat"]].quantile([0.3, 0.7]).values
    inside = state[state["lon"].between(xmin, xmax) & state["lat"].between(ymin, ymax)]
    way_versions = pd.DataFrame(ways)
    way_state = way_versions[way_versions["timestamp"] <= when]
    way_state = way_state.drop_duplicates("id", keep="last")
    way_state = way_state[way_state["visible"]]
    in_box = set(inside["id"])
    expected_ways = {
        way_id
        for way_id, refs in zip(way_state["id"], way_state["nodes"])
        if in_box.intersection(refs)
    }

    got_nodes, got_ways, _, _, _, _ = parse_osm_data(fp, [xmin, ymin, xmax, ymax], False, when)
    got = pd.DataFrame({c: np.asarray(got_nodes[c]) for c in ("id", "lon", "lat")})
    expected = inside[["id", "lon", "lat"]]
    pd.testing.assert_frame_equal(
        got.sort_values("id").reset_index(drop=True),
        expected.sort_values("id").reset_index(drop=True),
        check_dtype=False,
    )
    assert {way["id"] for way in got_ways} == expected_ways


@pytest.mark.parametrize("bounding_box", [None, [24.9, 60.1, 25.0, 60.2]])
def test_history_read_before_the_first_way_finds_no_network(bounding_box):
    """A history read at a timestamp when only nodes existed (the test file's first way
    and relation came later) finds no network, instead of failing on the empty way and
    relation tables."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_test_history_pbf"), bounding_box=bounding_box)
    with pytest.warns(UserWarning, match="Could not find any edges"):
        assert osm.get_network(timestamp="2007-01-15") is None


def test_parse_joins_block_arrays_in_each_keys_dtype():
    """The parse joins its per-block arrays per key, in block order and in each key's
    dtype; tag dicts stay the same objects, a key without values is left out, and blocks
    whose columns differ in length are refused."""
    import numpy as np
    from pyrosm._arrays import concatenate_dicts_of_arrays

    tags = np.empty(2, dtype=object)
    tags[:] = [{"amenity": "cafe"}, None]
    first = {
        "id": np.array([1, 2]),
        "version": np.array([3, 4]),
        "timestamp": np.array([5, 6]),
        "changeset": np.array([7, 8]),
        "lon": np.array([24.5, 24.6]),
        "visible": np.array([True, False]),
        "tags": tags,
        "empty": np.empty(0),
    }
    second = {key: values[:1] for key, values in first.items()}
    joined = concatenate_dicts_of_arrays([first, second])

    expected = {
        "id": ("int64", [1, 2, 1]),
        "version": ("int32", [3, 4, 3]),
        "timestamp": ("uint32", [5, 6, 5]),
        "changeset": ("int32", [7, 8, 7]),
        "lon": ("float64", [24.5, 24.6, 24.5]),
        "visible": ("bool", [True, False, True]),
        "tags": ("object", [tags[0], None, tags[0]]),
    }
    assert list(joined) == list(expected)
    for key, (dtype, values) in expected.items():
        assert joined[key].dtype == dtype
        assert joined[key].tolist() == values
    assert joined["tags"][0] is tags[0] and joined["tags"][2] is tags[0]
    with pytest.raises(AssertionError):
        concatenate_dicts_of_arrays([{"id": np.array([1]), "lon": np.array([1.0, 2.0])}])


def test_orient_polygons_returns_points_and_lines_as_they_are():
    """Ring orientation (#230) applies to polygons, multipolygons and the polygons of a
    geometry collection; points, lines and missing geometries come back as the same
    objects instead of copies."""
    import numpy as np
    import shapely
    from shapely.geometry import GeometryCollection, LineString, Point, Polygon
    from pyrosm.geometry import orient_polygons

    clockwise = Polygon(
        [(0, 0), (0, 4), (4, 4), (4, 0)], [[(1, 1), (2, 1), (2, 2), (1, 2)]]
    )
    geometries = np.array(
        [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            clockwise,
            GeometryCollection([clockwise, Point(5, 5)]),
            None,
        ],
        dtype=object,
    )
    oriented = orient_polygons(geometries)

    assert oriented[0] is geometries[0]
    assert oriented[1] is geometries[1]
    assert oriented[4] is None
    expected = shapely.orient_polygons(geometries[2:4], exterior_cw=False)
    assert shapely.equals_exact(oriented[2:4], expected, tolerance=0).all()
    assert oriented[2].exterior.is_ccw and not oriented[2].interiors[0].is_ccw
    assert oriented[3].geoms[0].exterior.is_ccw


def test_network_edges_are_one_linestring_per_way():
    """Each network edge is one LineString through its way's nodes that have valid
    coordinates, in the way's order, and its length is the haversine sum of its segments
    (millimetres per segment, metres per edge)."""
    import numpy as np
    from pyrosm import OSM, get_data
    from pyrosm.distance import Unit, haversine

    osm = OSM(get_data("helsinki_pbf"), keep_node_info=True, engine="in_memory")
    edges = osm.get_network("all")
    assert set(edges.geometry.geom_type) == {"LineString"}

    partial = 0
    for nodes, line, length in zip(edges["nodes"], edges.geometry, edges["length"]):
        idx, lon, lat = osm._node_coordinates.gather(np.asarray(nodes, dtype=np.int64))
        kept = (idx >= 0) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
        partial += not kept.all()
        lon, lat = lon[kept], lat[kept]
        assert np.array_equal(np.asarray(line.coords), np.column_stack([lon, lat]))
        segments = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:], unit=Unit.METERS)
        assert length == segments.round(3).sum().round(0)
    assert partial > 0


def test_result_cache_key_changes_with_its_format(monkeypatch):
    """A result cached under another cache format is not served: the format number is part
    of the result's cache key."""
    from pyrosm import get_data
    from pyrosm.engine import cache

    fp = get_data("test_pbf")
    before = cache.result_path(fp, {"network": True})
    monkeypatch.setattr(cache, "_RESULT_FORMAT", cache._RESULT_FORMAT + 1)
    assert cache.result_path(fp, {"network": True}) != before


def test_line_ways_of_other_layers_are_one_linestring():
    """Outside the network, a way that is not an area is one LineString through its nodes
    that have valid coordinates: an open way, a closed highway ring, a closed ``area=no``
    way and a closed way without an area tag alike; a closed building stays a Polygon."""
    import numpy as np
    import pandas as pd
    from pyrosm.geometry import create_way_geometries
    from pyrosm.node_lookup import NodeLocations

    xy = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0)}
    store = NodeLocations(
        pd.DataFrame(
            {
                "id": np.array(list(xy), dtype=np.int64),
                "lon": [x for x, _ in xy.values()],
                "lat": [y for _, y in xy.values()],
            }
        )
    )
    ring = [1, 2, 3, 4, 1]
    ways = [
        ([1, 2, 3], None, None, None, [1, 2, 3], "LineString"),
        ([1, 2], None, None, None, [1, 2], "LineString"),
        (ring, "residential", None, None, ring, "LineString"),
        (ring, None, "no", "yes", ring, "LineString"),
        (ring, None, None, None, ring, "LineString"),
        (ring, None, None, "yes", ring, "Polygon"),
        ([1, 99, 3], None, None, None, [1, 3], "LineString"),
    ]
    nodes = np.empty(len(ways), dtype=object)
    nodes[:] = [np.array(w[0], dtype=np.int64) for w in ways]
    elements = {
        "id": np.arange(len(ways), dtype=np.int64),
        "nodes": nodes,
        "highway": np.array([w[1] for w in ways], dtype=object),
        "area": np.array([w[2] for w in ways], dtype=object),
        "building": np.array([w[3] for w in ways], dtype=object),
    }
    _, geometries, _, _, _ = create_way_geometries(store, elements, False)

    for geometry, (*_, expected, kind) in zip(geometries, ways):
        assert geometry.geom_type == kind
        coords = geometry.exterior.coords if kind == "Polygon" else geometry.coords
        assert [tuple(c) for c in coords] == [xy[n] for n in expected]


def test_graph_edges_join_consecutive_nodes_of_each_way():
    """``get_network(nodes=True)`` returns one 2-point edge per pair of consecutive way
    nodes with valid coordinates (from ``u`` to ``v``), and one node row per node those
    edges use, with the node columns' dtypes."""
    import numpy as np
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), keep_node_info=True, engine="in_memory")
    nodes, edges = osm.get_network("all", nodes=True)

    expected_u, expected_v = [], []
    for way_nodes in edges.drop_duplicates("id")["nodes"]:
        idx, lon, lat = osm._node_coordinates.gather(np.asarray(way_nodes, dtype=np.int64))
        kept = np.asarray(way_nodes)[(idx >= 0) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)]
        expected_u.extend(kept[:-1])
        expected_v.extend(kept[1:])
    assert edges["u"].tolist() == expected_u
    assert edges["v"].tolist() == expected_v
    assert (edges.geometry.apply(lambda g: len(g.coords)) == 2).all()

    assert nodes["id"].is_unique
    assert set(nodes["id"]) == set(edges["u"]) | set(edges["v"])
    assert nodes.dtypes[["lon", "lat"]].tolist() == ["float64", "float64"]
    assert nodes.dtypes[["version", "timestamp", "changeset", "id"]].tolist() == ["int64"] * 4
    assert nodes["visible"].dtype == bool


def test_decode_spills_every_ways_refs_only_for_relation_lookups():
    """The out-of-core decode spills every way's id and node refs, which only the lookup of
    relation members reads, unless told not to (as for a network read)."""
    from pyrosm import get_data
    from pyrosm.engine import decode
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.primitive_block_decoder import decode_primitive_block

    fp = get_data("test_pbf")
    with open(fp, "rb") as f:
        blocks = [
            decode_primitive_block(_read_block(f, offset, size))
            for kind, offset, size in _index_blobs(fp)
            if kind == "OSMData"
        ]
    spilled = {}
    for spill in (True, False):
        settings = decode._DecodeSettings(
            fp, None, [b"highway"], include_nodes=False, spill_all_ways=spill
        )
        shards = [decode._decode_one_block(settings, *block) for block in blocks]
        spilled[spill] = sum(len(shard["all_refs"]) for shard in shards)
        assert sum(len(shard["way_id"]) for shard in shards) > 0
    assert spilled[True] > 0
    assert spilled[False] == 0


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_graph_export_uses_the_network_type_of_its_frames(engine, fresh_cache):
    """Graph export takes the network type recorded on the frames it gets, not that of the
    latest network read: driving frames exported after a walking read stay a directed
    driving graph, and out-of-core frames export without naming their type."""
    from pyrosm import OSM, get_data

    pytest.importorskip("networkx")
    osm = OSM(get_data("test_pbf"), engine=engine, workers=1)
    nodes, edges = osm.get_network("driving", nodes=True)
    osm.get_network("walking")
    graph = OSM.to_graph(nodes, edges, graph_type="networkx", simplify=False)
    assert graph.number_of_edges() == 1237


def test_graph_export_reads_the_network_type_of_the_node_frame():
    """A network read records its type by its pyrosm name. Graph export uses a given
    network_type (refusing an unknown one), else falls back to the type of the node frame
    when the edges carry none, and asks for network_type when neither frame does."""
    from pyrosm import OSM, get_data

    pytest.importorskip("networkx")
    nodes, edges = OSM(get_data("test_pbf")).get_network("Drive", nodes=True)
    assert edges.attrs["network_type"] == nodes.attrs["network_type"] == "driving"
    with pytest.raises(ValueError, match="should be one of"):
        OSM.to_graph(nodes, edges, graph_type="networkx", network_type="drivng")
    edges.attrs.clear()
    graph = OSM.to_graph(nodes, edges, graph_type="networkx", simplify=False)
    assert graph.number_of_edges() == 1237
    nodes.attrs.clear()
    with pytest.raises(ValueError, match="Could not detect the network type"):
        OSM.to_graph(nodes, edges, graph_type="networkx")


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_network_read_with_custom_filter_refuses_unknown_network_type(engine):
    """A network read validates network_type even with a custom_filter, with either
    engine, so the type it records for graph export is a known one."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"), engine=engine, workers=1)
    with pytest.raises(ValueError, match="should be one of"):
        osm.get_network("drivng", custom_filter={"highway": ["primary"]})


@pytest.mark.parametrize("nodes", [False, True])
def test_cached_network_read_records_its_network_type(nodes, fresh_cache, monkeypatch):
    """An out-of-core network read served from the result cache records its network type
    on the frames like the read that built the cache."""
    from pyrosm import OSM, get_data
    from pyrosm.engine import readers

    pytest.importorskip("networkx")
    fp = get_data("test_pbf")
    first = OSM(fp, engine="out_of_core", workers=1).get_network("driving", nodes=nodes)
    # The second read must come from the cache, not from decoding the file again.
    monkeypatch.setattr(readers, "_decode_and_run", None)
    second = OSM(fp, engine="out_of_core", workers=1).get_network("driving", nodes=nodes)
    for result in (first, second):
        frames = result if nodes else (result,)
        assert [frame.attrs["network_type"] for frame in frames] == ["driving"] * len(
            frames
        )
    if nodes:
        graph = OSM.to_graph(*second, graph_type="networkx", simplify=False)
        assert graph.number_of_edges() == 1237


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
@pytest.mark.parametrize(
    "read, found",
    [
        (lambda osm: osm.get_network(), "edges"),
        (lambda osm: osm.get_network(nodes=True)[1], "edges"),
        (lambda osm: osm.get_buildings(), "building elements"),
        (lambda osm: osm.get_landuse(), "landuse elements"),
        (lambda osm: osm.get_natural(), "natural elements"),
        (lambda osm: osm.get_pois(), "POIs"),
        (lambda osm: osm.get_boundaries(), "boundaries"),
        (lambda osm: osm.get_data_by_custom_criteria({"shop": True}), "OSM data"),
    ],
    ids=[
        "network",
        "network_segments",
        "buildings",
        "landuse",
        "natural",
        "pois",
        "boundaries",
        "custom",
    ],
)
def test_read_that_finds_nothing_warns(engine, read, found, fresh_cache):
    """A read that finds nothing warns so and returns None, with either engine (the
    out-of-core engine used to return None silently)."""
    from pyrosm import OSM, get_data

    osm = OSM(
        get_data("test_pbf"), engine=engine, workers=1, bounding_box=[0, 0, 0.01, 0.01]
    )
    with pytest.warns(
        UserWarning, match="Could not find any %s for given area" % found
    ):
        assert read(osm) is None


def _malformed_block(case):
    """The bytes of a PrimitiveBlock whose arrays break the PBF schema as `case` says."""
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    block = PrimitiveBlock()
    block.stringtable.s.append(b"")
    group = block.primitivegroup.add()
    if case == "way with more values than keys":
        group.ways.add(id=1, keys=[1], vals=[1, 2])
    elif case == "relation with fewer member types than ids":
        group.relations.add(id=1, memids=[1, 1], types=[0], roles_sid=[0, 0])
    elif case == "dense nodes with fewer latitudes than ids":
        group.dense.id.extend([1, 1])
        group.dense.lat.extend([1])
        group.dense.lon.extend([1, 1])
    elif case == "dense tags without the 0 that ends a node":
        group.dense.id.extend([1])
        group.dense.lat.extend([1])
        group.dense.lon.extend([1])
        group.dense.keys_vals.extend([1, 1])
    elif case == "dense metadata with more versions than nodes":
        group.dense.id.extend([1])
        group.dense.lat.extend([1])
        group.dense.lon.extend([1])
        group.dense.denseinfo.version.extend([1, 1])
    return block.SerializeToString()


@pytest.mark.parametrize(
    "case, data, match",
    [
        ("varint cut off", b"\xff\xff", "does not fit"),
        ("field longer than the block", b"\x12\x10\x0a", "does not fit"),
        ("string longer than the string table", b"\x0a\x04\x0a\x05ab", "does not fit"),
        ("varint longer than 10 bytes", b"\x88\x01" + b"\x80" * 10 + b"\x01", "does not fit"),
        ("varint above 64 bits", b"\x88\x01" + b"\xff" * 9 + b"\x02", "does not fit"),
        ("field number 0", b"\x00\x00", "does not fit"),
        ("field number 2**32 + 17", b"\x88\x81\x80\x80\x80\x01\x64", "does not fit"),
        ("unknown wire type", b"\x0f\x00", "does not fit"),
        ("fixed64 field cut off", b"\x99\x06abc", "does not fit"),
        ("way refs cut off", b"\x12\x08\x1a\x06\x08\x01\x42\x02\x02\x80", "does not fit"),
        ("way with more values than keys", None, "a way has 1 tag keys but 2 values"),
        ("relation with fewer member types than ids", None, "2 member ids but 1 member types"),
        ("dense nodes with fewer latitudes than ids", None, "2 ids but 1 latitudes"),
        ("dense tags without the 0 that ends a node", None, "do not end once for each"),
        ("dense metadata with more versions than nodes", None, "1 nodes but 2 values of version"),
    ],
)
def test_block_decoder_rejects_malformed_blocks(case, data, match):
    """#483 — the raw block decoder raises on a block that breaks the wire format or the
    PBF schema, instead of reading or writing past its data."""
    import numpy as np

    from pyrosm.exceptions import InvalidOSMFileError
    from pyrosm.primitive_block_decoder import decode_primitive_block

    if data is None:
        data = _malformed_block(case)
    for keep_metadata in (True,) if "metadata" in case else (True, False):
        with pytest.raises(InvalidOSMFileError, match=match):
            decode_primitive_block(np.frombuffer(data, dtype=np.uint8), keep_metadata)


def test_block_decoder_defaults_a_missing_id_or_coordinate():
    """An element without its id, or a plain node without its latitude, decodes with 0,
    the proto2 default, as protobuf reads it."""
    import numpy as np

    from pyrosm.primitive_block_decoder import decode_primitive_block

    def field(tag, payload):
        return bytes([tag, len(payload)]) + payload

    groups = field(0x12, field(0x0A, b"\x48\x02"))  # a plain node with only lon 1
    groups += field(0x12, field(0x1A, b"\x42\x01\x02"))  # a way with only ref 1
    groups += field(0x12, field(0x22, b"\x4A\x01\x02\x52\x01\x00\x42\x01\x00"))
    data = np.frombuffer(field(0x0A, b"") + groups, dtype=np.uint8)
    for keep_metadata in (True, False):
        _, _, nodes, ways, relations = decode_primitive_block(data, keep_metadata)
        assert (nodes["id"].tolist(), nodes["lat"].tolist()) == ([0], [0])
        assert ways["id"].tolist() == [0] and relations["id"].tolist() == [0]


def test_block_decoder_joins_split_packed_runs():
    """#483 — a packed field written in several runs is read as one array, with the
    delta coding continuing from one run to the next."""
    import numpy as np

    from pyrosm.primitive_block_decoder import decode_primitive_block

    def field(tag, payload):
        return bytes([tag, len(payload)]) + payload

    def runs(tag, *payloads):
        return b"".join(field(tag, payload) for payload in payloads)

    info = runs(0x0A, b"\x05", b"\x07\x09")  # versions 5, 7, 9
    info += runs(0x12, b"\x14", b"\x02", b"\x02")  # timestamp deltas 10, 1, 1
    dense = runs(0x0A, b"\x02", b"\x04", b"\x06")  # id deltas 1, 2, 3
    dense += runs(0x42, b"\x02\x02\x02")  # lat deltas 1, 1, 1
    dense += runs(0x4A, b"\x06", b"\x01\x02")  # lon deltas 3, -1, 1
    dense += runs(0x52, b"\x00", b"\x00", b"\x00") + field(0x2A, info)
    way = b"\x08\x05" + runs(0x12, b"\x01", b"\x02") + runs(0x1A, b"\x03\x04")
    way += runs(0x42, b"\x14", b"\x02")  # ref deltas 10, 1
    relation = b"\x08\x07" + runs(0x42, b"\x00", b"\x00")
    relation += runs(0x4A, b"\x14", b"\x0a")  # member id deltas 10, 5
    relation += runs(0x52, b"\x00", b"\x01")
    groups = field(0x12, dense) + field(0x1A, way), field(0x22, relation)
    data = field(0x0A, b"") + runs(0x12, *groups)

    _, _, nodes, ways, relations = decode_primitive_block(
        np.frombuffer(data, dtype=np.uint8), True
    )
    assert ways["keys"].tolist() == [1, 2] and ways["vals"].tolist() == [3, 4]
    assert ways["refs"].tolist() == [10, 11]
    assert relations["memids"].tolist() == [10, 15]
    assert relations["types"].tolist() == [0, 1]
    assert relations["roles"].tolist() == [0, 0]
    assert nodes["id"].tolist() == [1, 3, 6]
    assert nodes["lon"].tolist() == [3, 2, 3]
    assert nodes["keys_vals"].tolist() == [0, 0, 0]
    assert nodes["version"].tolist() == [5, 7, 9]
    assert nodes["timestamp"].tolist() == [10, 11, 12]


def test_block_decoder_wraps_delta_sums_around():
    """Way refs and relation member ids whose deltas sum past the int64 range wrap around
    as numpy's int64 arithmetic does (a signed overflow is undefined in C)."""
    import numpy as np

    from pyrosm.primitive_block_decoder import decode_primitive_block
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    deltas = [2**62, 2**62, -5]
    block = PrimitiveBlock()
    block.stringtable.s.append(b"")
    block.primitivegroup.add().ways.add(id=1, refs=deltas)
    block.primitivegroup.add().relations.add(
        id=2, memids=deltas, types=[1, 1, 1], roles_sid=[0, 0, 0]
    )
    data = np.frombuffer(block.SerializeToString(), dtype=np.uint8)
    _, _, _, ways, relations = decode_primitive_block(data, True)
    expected = [2**62, -(2**63), 2**63 - 5]
    assert ways["refs"].tolist() == expected
    assert relations["memids"].tolist() == expected


@pytest.mark.parametrize("alias", ["same path", "hard link"])
def test_crop_refuses_to_overwrite_its_source(tmp_path, alias):
    """crop_pbf and OSM.to_pbf refuse an output path that is their source, under the same
    name or through a hard link, and leave the source as it was; they used to truncate it."""
    import os
    import shutil

    from pyrosm import OSM, get_data
    from pyrosm.pbf_export import crop_pbf

    source = tmp_path / "source.osm.pbf"
    shutil.copy(get_data("test_pbf"), source)
    before = source.read_bytes()
    output = source
    if alias == "hard link":
        output = tmp_path / "link.osm.pbf"
        os.link(source, output)
    box = [26.94, 60.52, 26.96, 60.53]
    for crop in (
        lambda: crop_pbf(str(source), str(output), bounding_box=box),
        lambda: OSM(str(source), bounding_box=box).to_pbf(str(output)),
    ):
        with pytest.raises(ValueError, match="is the input"):
            crop()
        assert source.read_bytes() == before


def test_multipolygon_with_thousands_of_rings_assembles_quickly(tmp_path):
    """A multipolygon with thousands of rings (Finland's largest lakes have 5,000-9,500)
    is assembled in seconds with all its holes; overlaying the rings one at a time onto
    the growing area took minutes for such relations."""
    import time

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Relation, Way
    from pyrosm import OSM

    n_holes, side, cell = 4000, 64, 0.001
    # The lake's shell, then a small square hole in each grid cell.
    squares = [(0, 0, side * cell)] + [
        ((h % side + 0.25) * cell, (h // side + 0.25) * cell, cell / 2)
        for h in range(n_holes)
    ]
    path = str(tmp_path / "lake.osm.pbf")
    writer = osmium.SimpleWriter(path)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for s, (x0, y0, size) in enumerate(squares):
        for c, (dx, dy) in enumerate(corners):
            location = (25 + x0 + dx * size, 60 + y0 + dy * size)
            writer.add_node(Node(id=4 * s + c + 1, location=location))
    for s in range(len(squares)):
        ring = [4 * s + 1, 4 * s + 2, 4 * s + 3, 4 * s + 4, 4 * s + 1]
        writer.add_way(Way(id=s + 1, nodes=ring))
    members = [("w", 1, "outer")]
    members += [("w", s + 1, "inner") for s in range(1, len(squares))]
    tags = {"type": "multipolygon", "natural": "water"}
    writer.add_relation(Relation(id=1, members=members, tags=tags))
    writer.close()

    start = time.perf_counter()
    gdf = OSM(path, workers=1).get_natural()
    seconds = time.perf_counter() - start
    lake = gdf[gdf["osm_type"] == "relation"].geometry.iloc[0]
    assert lake.geom_type == "Polygon" and len(lake.interiors) == n_holes
    expected = (side * cell) ** 2 - n_holes * (cell / 2) ** 2
    assert lake.area == pytest.approx(expected, rel=1e-6)
    assert seconds < 10


@pytest.mark.parametrize("keep_ways", [True, False], ids=["with ways", "nodes only"])
def test_nodes_without_leftover_tags_have_no_tags_value(keep_ways):
    """Nodes whose tags all became columns have no ``tags`` value, as other elements; both
    readers used to keep each node's tag dict there, and the out-of-core reader could not
    write it next to the ways' JSON (ArrowTypeError)."""
    import pandas as pd

    from pyrosm import OSM, get_data

    frames = [
        OSM(get_data("test_pbf"), engine=engine, workers=1).get_data_by_custom_criteria(
            {"shop": True}, keep_ways=keep_ways, keep_relations=keep_ways
        )
        for engine in ("in_memory", "out_of_core")
    ]
    for gdf in frames:
        assert list(gdf.columns)[:4] == ["id", "lon", "lat", "tags"]
        tags = [None if pd.isna(value) else value for value in gdf["tags"]]
        leftovers = ['{"landuse":"greenhouse_horticulture","building":"yes"}']
        assert tags == [None, None] + (leftovers if keep_ways else [])


@pytest.mark.parametrize(
    "codec, raw_size, engine",
    [("zlib", None, "in_memory"), ("zlib", 7, "out_of_core"), ("lzma", None, "in_memory")],
    ids=["zlib without raw_size", "zlib with a wrong raw_size", "lzma"],
)
def test_reading_refuses_a_blob_larger_than_32_mib(tmp_path, codec, raw_size, engine):
    """A data blob that decompresses to more than the 32 MiB the PBF format allows raises
    InvalidOSMFileError; the readers used to decompress it whole, however large."""
    import lzma
    import zlib
    from struct import pack

    from pyrosm import OSM
    from pyrosm.exceptions import InvalidOSMFileError
    from pyrosm.proto.fileformat_pb2 import Blob, BlobHeader
    from pyrosm.proto.osmformat_pb2 import HeaderBlock

    zeros = bytes(33 * 2**20)
    payload = {"zlib": zlib.compress, "lzma": lzma.compress}[codec](zeros)
    header = HeaderBlock(required_features=["OsmSchema-V0.6", "DenseNodes"])
    header = header.SerializeToString()
    blobs = [
        ("OSMHeader", Blob(raw=header, raw_size=len(header))),
        ("OSMData", Blob(raw_size=raw_size, **{codec + "_data": payload})),
    ]
    path = tmp_path / "large.osm.pbf"
    with open(path, "wb") as f:
        for kind, blob in blobs:
            data = blob.SerializeToString()
            head = BlobHeader(type=kind, datasize=len(data)).SerializeToString()
            f.write(pack("!L", len(head)) + head + data)
    with pytest.raises(InvalidOSMFileError, match="larger than 32 MiB"):
        OSM(str(path), engine=engine, workers=1).get_buildings()


def test_reads_decoding_at_the_same_time_keep_their_own_settings(fresh_cache, monkeypatch):
    """Two out-of-core reads decoding in one process at the same time, each in its own
    thread, give what each gives alone. Read A (buildings) is held after its first block
    until read B (POIs of the same file) has begun decoding, and B is held until A is done;
    A used to decode its other blocks with B's filter keys and return fewer rows. The
    threaded decode decodes one block per batch here, so it is held the same way."""
    import shutil
    import threading

    import pandas as pd

    from pyrosm import OSM, get_data
    from pyrosm.engine import decode

    fp = get_data("test_pbf")
    layers = {"A": "get_buildings", "B": "get_pois"}
    alone = {name: getattr(OSM(fp), layer)() for name, layer in layers.items()}
    shutil.rmtree(fresh_cache)
    fresh_cache.mkdir()
    monkeypatch.setattr(decode, "_THREAD_BATCH", 1)
    a_blocks = []
    a_held, b_decoding, a_done = threading.Event(), threading.Event(), threading.Event()

    def held(real):
        def call(*args, **kwargs):
            name = threading.current_thread().name
            if name == "A":
                a_blocks.append(1)
                if len(a_blocks) == 2:
                    a_held.set()
                    b_decoding.wait(30)
            elif name == "B" and not b_decoding.is_set():
                b_decoding.set()
                a_done.wait(30)
            return real(*args, **kwargs)

        return call

    for step in ("_decode_one_block", "decode_batch_nogil"):
        monkeypatch.setattr(decode, step, held(getattr(decode, step)))
    together = {}

    def read(name):
        try:
            together[name] = getattr(OSM(fp), layers[name])()
        except Exception as e:
            together[name] = e
        finally:
            if name == "A":
                a_done.set()

    a = threading.Thread(target=read, args=("A",), name="A")
    a.start()
    assert a_held.wait(30)
    b = threading.Thread(target=read, args=("B",), name="B")
    b.start()
    a.join(60)
    b.join(60)
    assert b_decoding.is_set()
    for name in layers:
        frame, expected = together[name], alone[name]
        assert not isinstance(frame, Exception), frame
        pd.testing.assert_frame_equal(
            frame.drop(columns="geometry"), expected.drop(columns="geometry")
        )
        assert (frame.geometry.to_wkb() == expected.geometry.to_wkb()).all()


@pytest.mark.parametrize(
    "engine, backend, chunk_size",
    [
        ("in_memory", None, None),
        ("out_of_core", "threads", None),
        ("out_of_core", "threads", 1),
        ("out_of_core", "processes", None),
    ],
    ids=["in_memory", "threads", "threads_chunks_in_c", "processes"],
)
def test_area_rule_reads_tags_that_are_not_columns(
    tmp_path, monkeypatch, fresh_cache, engine, backend, chunk_size
):
    """The OSM wiki's polygon features, as OSMnx reads them, decide whether a closed way is
    a Polygon, also from tags that are not columns of the read: an area tag such as
    ``building`` or ``leisure`` makes it one despite a ``barrier``, ``highway`` or
    ``route`` tag (0.23.0 returned fenced pitches and other fenced areas as LineStrings),
    while ``area=no`` and a ``natural`` value the list leaves out (``coastline``) make it a
    line. Of a key given twice, the last value counts. The tags stay in the JSON ``tags``
    column."""
    import json

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Way
    from pyrosm import OSM
    from pyrosm.engine import geoparquet

    ways = {
        1: ({"building": "yes"}, "Polygon"),
        2: ({"building": "yes", "area": "no"}, "LineString"),
        3: ({"building": "yes", "barrier": "wall"}, "Polygon"),
        4: ({"building": "yes", "highway": "pedestrian", "area": "yes"}, "Polygon"),
        5: ({"building": "yes", "route": "ferry", "area": "unknown"}, "Polygon"),
        6: ({"leisure": "pitch", "barrier": "fence"}, "Polygon"),
        7: ({"natural": "wood", "barrier": "retaining_wall"}, "Polygon"),
        8: ({"natural": "coastline"}, "LineString"),
        9: ([("natural", "wood"), ("natural", "coastline")], "LineString"),
        10: ([("natural", "coastline"), ("natural", "wood")], "Polygon"),
    }
    path = str(tmp_path / "closed_ways.osm.pbf")
    with osmium.SimpleWriter(path) as writer:
        for w in ways:
            corners = [(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]
            for k, (dx, dy) in enumerate(corners):
                writer.add_node(Node(id=10 * w + k, location=(25 + w / 100 + dx, 60 + dy)))
        for w, (tags, _) in ways.items():
            nodes = [10 * w + k for k in range(4)]
            writer.add_way(Way(id=w, nodes=nodes + nodes[:1], tags=tags))
    if backend is not None:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    if chunk_size is not None:
        monkeypatch.setattr(geoparquet, "_OUTPUT_CHUNK_SIZE", chunk_size)

    gdf = (
        OSM(path, engine=engine, workers=1)
        .get_data_by_custom_criteria({"building": True, "leisure": True, "natural": True})
        .set_index("id")
    )
    assert {w: gdf.geometry[w].geom_type for w in ways} == {
        w: expected for w, (_, expected) in ways.items()
    }
    assert "area" not in gdf.columns
    assert json.loads(gdf.loc[2, "tags"])["area"] == "no"


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_fenced_school_grounds_are_polygons(engine):
    """The schools of the Helsinki region extract that are not nodes are Polygons and
    MultiPolygons: 0.23.0 returned the school grounds of way 161966214, tagged
    ``amenity=school`` and ``barrier=fence``, as a LineString."""
    from pyrosm import OSM, get_data

    schools = OSM(
        get_data("helsinki_region_pbf"), engine=engine, progress=False
    ).get_data_by_custom_criteria({"amenity": ["school"]}, keep_nodes=False)
    assert schools.geometry.geom_type.value_counts().to_dict() == {
        "Polygon": 420,
        "MultiPolygon": 1,
    }
    assert schools.set_index("id").geometry[161966214].geom_type == "Polygon"


@pytest.mark.parametrize(
    "engine, backend",
    [("in_memory", None), ("out_of_core", "threads"), ("out_of_core", "processes")],
    ids=["in_memory", "threads", "processes"],
)
def test_relation_id_tag_does_not_overwrite_the_relation_id(
    tmp_path, monkeypatch, fresh_cache, engine, backend
):
    """A relation's tag keyed ``id`` is surfaced as ``id_tag``, as a way's is, and the
    relation keeps its id."""
    import numpy as np

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Relation, Way
    from pyrosm import OSM

    path = str(tmp_path / "relation_id_tag.osm.pbf")
    with osmium.SimpleWriter(path) as writer:
        for k, (dx, dy) in enumerate([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]):
            writer.add_node(Node(id=k + 1, location=(25 + dx, 60 + dy)))
        writer.add_way(Way(id=10, nodes=[1, 2, 3, 4, 1], tags={}))
        writer.add_relation(
            Relation(
                id=100,
                members=[("w", 10, "outer")],
                tags={"type": "multipolygon", "building": "yes", "id": "B-7"},
            )
        )
    if backend is not None:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)

    gdf = OSM(path, engine=engine, workers=1).get_buildings(extra_attributes=["id_tag"])
    assert gdf["id"].tolist() == [100] and gdf["id"].dtype == np.int64
    assert gdf["id_tag"].tolist() == ["B-7"]


@pytest.mark.parametrize(
    "engine, backend",
    [("in_memory", None), ("out_of_core", "threads"), ("out_of_core", "processes")],
    ids=["in_memory", "threads", "processes"],
)
def test_node_id_tag_does_not_overwrite_the_node_id(
    tmp_path, monkeypatch, fresh_cache, engine, backend
):
    """A node's tag keyed ``id`` is surfaced as ``id_tag``, as a way's is: requesting
    ``id`` as a column keeps the node ids, ``id_tag`` is a column when requested and
    a key of the JSON ``tags`` otherwise."""
    import json

    import numpy as np
    import pandas as pd

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node
    from pyrosm import OSM

    path = str(tmp_path / "node_id_tag.osm.pbf")
    with osmium.SimpleWriter(path) as writer:
        writer.add_node(
            Node(id=100, location=(25, 60), tags={"amenity": "cafe", "id": "stray"})
        )
        writer.add_node(Node(id=101, location=(25.001, 60), tags={"amenity": "cafe"}))
    if backend is not None:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    osm = OSM(path, engine=engine, workers=1)

    gdf = osm.get_pois(custom_filter={"amenity": True}, extra_attributes=["id", "id_tag"])
    assert gdf["id"].tolist() == [100, 101] and gdf["id"].dtype == np.int64
    assert gdf["id_tag"][0] == "stray" and pd.isna(gdf["id_tag"][1])
    gdf = osm.get_pois(custom_filter={"amenity": True})
    assert json.loads(gdf["tags"][0]) == {"id_tag": "stray"}


@pytest.mark.parametrize("backend", ["threads", "processes"])
def test_lone_surrogate_keys_read_out_of_core(monkeypatch, fresh_cache, backend):
    """A requested tag key UTF-8 cannot encode (a lone surrogate) reads out of core as it
    does in memory: no PBF tag has it, so its column is absent and a filter by it keeps
    nothing."""
    import pandas as pd
    from pyrosm import OSM, get_data

    monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    fp = get_data("test_pbf")
    key = "\ud800"
    reads = [
        lambda osm: osm.get_buildings(extra_attributes=[key]),
        lambda osm: osm.get_data_by_custom_criteria(
            custom_filter={"building": True}, extra_attributes=[key], keep_other_tags=False
        ),
    ]
    for i, read in enumerate(reads):
        gdf = read(OSM(fp, engine="out_of_core", workers=1))
        if i == 0:
            ref = read(OSM(fp, engine="in_memory"))
        else:
            # The in-memory reader keeps every tag; without them the JSON column goes.
            ref = ref.drop(columns="tags")
        assert key not in gdf.columns
        # The in-memory reader's ``visible`` column holds Python bools (object dtype).
        pd.testing.assert_frame_equal(
            gdf.drop(columns="geometry"), ref.drop(columns="geometry"), check_dtype=False
        )
        assert (gdf.geometry.to_wkb() == ref.geometry.to_wkb()).all()
    osm = OSM(fp, engine="out_of_core", workers=1)
    with pytest.warns(UserWarning, match="Could not find any OSM data"):
        assert osm.get_data_by_custom_criteria(custom_filter={key: True}) is None


@pytest.mark.parametrize(
    "engine, backend",
    [("in_memory", None), ("out_of_core", "threads"), ("out_of_core", "processes")],
    ids=["in_memory", "threads", "processes"],
)
def test_missing_visible_flag_reads_as_visible(
    tmp_path, monkeypatch, fresh_cache, engine, backend
):
    """An element whose metadata has no ``visible`` flag, as every element of a file that
    is not a history file, is visible, as the PBF format reads it: nodes, ways and
    relations alike."""
    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Relation, Way
    from pyrosm import OSM

    path = str(tmp_path / "no_visible_flags.osm.pbf")
    with osmium.SimpleWriter(path) as writer:
        writer.add_node(Node(id=100, location=(25.01, 60), tags={"amenity": "cafe"}))
        for k, (dx, dy) in enumerate([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]):
            writer.add_node(Node(id=k + 1, location=(25 + dx, 60 + dy)))
        writer.add_way(Way(id=10, nodes=[1, 2, 3, 4, 1], tags={"building": "yes"}))
        writer.add_way(Way(id=11, nodes=[1, 2, 3, 4, 1], tags={}))
        writer.add_relation(
            Relation(
                id=200,
                members=[("w", 11, "outer")],
                tags={"type": "multipolygon", "building": "yes"},
            )
        )
    if backend is not None:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    osm = OSM(path, engine=engine, workers=1)

    pois = osm.get_pois(custom_filter={"amenity": True})
    assert pois["visible"].tolist() == [True]
    buildings = osm.get_buildings().set_index("osm_type")
    assert buildings["visible"].to_dict() == {"way": True, "relation": True}


def test_block_decoders_read_the_visible_flag():
    """Both block decoders read a way's ``visible`` flag as it is written, and as 1 when
    its Info has no flag or the way has no Info."""
    import numpy as np

    from pyrosm.primitive_block_decoder import decode_batch_nogil, decode_primitive_block
    from pyrosm.proto.osmformat_pb2 import PrimitiveBlock

    block = PrimitiveBlock()
    block.stringtable.s.extend([b"", b"building", b"yes"])
    ways = block.primitivegroup.add().ways
    for way_id in (1, 2, 3, 4):
        way = ways.add(id=way_id, keys=[1], vals=[2], refs=[1, 1])
        if way_id < 3:
            way.info.visible = way_id == 2
        elif way_id == 3:
            way.info.version = 1
    data = block.SerializeToString()
    _, _, _, decoded, _ = decode_primitive_block(np.frombuffer(data, dtype=np.uint8))
    assert decoded["visible"].tolist() == [0, 1, 1, 1]
    out = decode_batch_nogil(
        np.frombuffer(data, dtype=np.uint8),
        np.array([0, len(data)], dtype=np.int64),
        keys=None,
    )
    assert out["way_meta"][:, 2].tolist() == [0, 1, 1, 1]


@pytest.mark.parametrize(
    "engine, backend",
    [("in_memory", None), ("out_of_core", "threads"), ("out_of_core", "processes")],
    ids=["in_memory", "threads", "processes"],
)
def test_filters_read_the_id_tag_as_id_tag(
    tmp_path, monkeypatch, fresh_cache, engine, backend
):
    """A filter reads a tag keyed ``id`` as ``id_tag``, the column it is returned in, on
    nodes, ways and relations; ``id``, the element's OSM id, is refused as a filter key.
    Out of core, ``keep_other_tags=False`` returns the requested ``id_tag`` column."""
    import pandas as pd

    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Relation, Way
    from pyrosm import OSM

    path = str(tmp_path / "id_tags.osm.pbf")
    with osmium.SimpleWriter(path) as writer:
        writer.add_node(Node(id=100, location=(25, 60), tags={"amenity": "cafe", "id": "n"}))
        writer.add_node(Node(id=101, location=(25.001, 60), tags={"amenity": "cafe"}))
        for k, (dx, dy) in enumerate([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]):
            writer.add_node(Node(id=k + 1, location=(25 + dx, 60.01 + dy)))
        writer.add_way(Way(id=10, nodes=[1, 2], tags={"highway": "path", "id": "w"}))
        writer.add_way(Way(id=11, nodes=[2, 3], tags={"highway": "path"}))
        writer.add_way(Way(id=12, nodes=[1, 2, 3, 4, 1], tags={}))
        for rel_id, tags in ((200, {"id": "r"}), (201, {})):
            writer.add_relation(
                Relation(
                    id=rel_id,
                    members=[("w", 12, "outer")],
                    tags={"type": "multipolygon", "building": "yes", **tags},
                )
            )
    if backend is not None:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
    osm = OSM(path, engine=engine, workers=1)
    tagged = [("node", 100, "n"), ("relation", 200, "r"), ("way", 10, "w")]

    def kept(gdf, column="id_tag"):
        return sorted(zip(gdf["osm_type"], gdf["id"], gdf[column]))

    assert kept(osm.get_data_by_custom_criteria(custom_filter={"id_tag": True})) == tagged
    found = osm.get_data_by_custom_criteria(
        custom_filter={"amenity": True, "highway": True, "building": True},
        osm_keys_to_keep=["id_tag"],
        extra_attributes=["id_tag"],
    )
    assert kept(found) == tagged
    assert kept(osm.get_data_by_custom_criteria(custom_filter='["id_tag"="w"]')) == [
        ("way", 10, "w")
    ]
    for refused in ({"id": True}, '["id"="w"]'):
        with pytest.raises(ValueError, match="'id' is the element's OSM id"):
            osm.get_data_by_custom_criteria(custom_filter=refused)
    for custom_filter in ({"amenity": True}, None):
        with pytest.raises(ValueError, match="'id' is the element's OSM id"):
            osm.get_data_by_custom_criteria(
                custom_filter=custom_filter, osm_keys_to_keep="id"
            )
    for flt in ({"id_tag": True}, '["highway"]["id_tag"]'):
        edges = osm.get_network(custom_filter=flt, filter_type="keep")
        assert edges["id"].tolist() == [10], flt
    if engine == "out_of_core":
        lean = osm.get_data_by_custom_criteria(
            custom_filter={"amenity": True},
            extra_attributes=["id_tag"],
            keep_other_tags=False,
        )
        lean = lean.set_index("id")["id_tag"]
        assert lean[100] == "n" and pd.isna(lean[101])


def test_cache_takes_the_bare_name_of_a_downloaded_file(tmp_path, monkeypatch, fresh_cache):
    """``OSM.list_cache`` and ``OSM.clear_cache`` take the bare name of a file in the
    download directory, as ``OSM.clear_downloads`` does, and still take the bare name of a
    local file read from the working directory, also after that file is deleted. The name
    was read only as a path in the working directory, so a downloaded file's name matched no
    cached layer and nothing was cleared."""
    import shutil

    import pyrosm.utils.download as dl
    from pyrosm import OSM, get_data

    downloads = tmp_path / "downloads"
    work = tmp_path / "work"
    downloads.mkdir()
    work.mkdir()
    monkeypatch.setattr(dl, "download_dir", lambda: downloads)
    monkeypatch.chdir(work)
    shutil.copy(get_data("test_pbf"), downloads / "sample-latest.osm.pbf")
    shutil.copy(get_data("test_pbf"), work / "local.osm.pbf")
    OSM(str(downloads / "sample-latest.osm.pbf"), engine="out_of_core").get_buildings()
    OSM("local.osm.pbf", engine="out_of_core").get_buildings()
    (work / "local.osm.pbf").unlink()

    assert len(OSM.list_cache("sample-latest.osm.pbf")) == 1
    assert OSM.clear_cache("sample-latest.osm.pbf") == 1
    assert OSM.clear_cache("local.osm.pbf") == 1
    assert OSM.list_cache() == []


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_custom_filter_narrows_the_layer_readers(engine):
    """A ``custom_filter`` without the layer key keeps only elements of the layer:
    ``get_boundaries`` the boundaries of ``boundary_type`` that match it, ``get_buildings``
    the buildings that match it. The layer key was added as an OR term, so the filter widened
    the result instead (all ``boundary=*`` features plus everything with ``admin_level=10``)."""
    from pyrosm import OSM, get_data

    region = OSM(get_data("helsinki_region_pbf"), engine=engine, workers=1)
    level10 = region.get_boundaries(custom_filter={"admin_level": ["10"]})
    assert len(level10) == 109
    assert set(level10["boundary"]) == {"administrative"}
    assert set(level10["admin_level"]) == {"10"}

    sample = OSM(get_data("helsinki_pbf"), engine=engine, workers=1)
    with_amenity = sample.get_buildings(custom_filter={"amenity": True})
    assert len(with_amenity) == 26
    assert with_amenity["building"].notna().all()

    # an empty filter keeps the whole layer
    assert len(region.get_boundaries(custom_filter={})) == 248
    assert len(sample.get_buildings(custom_filter={})) == len(sample.get_buildings())


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_driving_network_leaves_out_ways_cars_cannot_use(engine):
    """``get_network("driving")`` leaves out ``highway=trail`` ways, which cars cannot use;
    the walking network keeps them. The driving filter followed OSMnx's ``drive``, which
    does not exclude trails: 426 of the 1,926 driving edges of the Helsinki sample were
    trails."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine=engine, workers=1)
    driving = osm.get_network(network_type="driving")
    walking = osm.get_network(network_type="walking")
    assert "trail" not in set(driving["highway"])
    assert "trail" in set(walking["highway"])


@pytest.mark.parametrize("run", ["crop", "merge"])
def test_crop_and_merge_take_auto_workers(run, tmp_path, monkeypatch):
    """``workers="auto"``, the default, chooses a worker count for crops and merges as it
    does for out-of-core reads, and writes the same file as ``workers=1``, as a NumPy
    integer does; an unknown value raises ``ValueError``. ``"auto"`` was compared with
    ``1`` and raised ``TypeError``, and a NumPy integer was refused."""
    import numpy as np

    import pyrosm.engine.pool as pool
    import pyrosm.pbf_export as pbf_export
    from pyrosm import get_data, merge_pbf
    from pyrosm.pbf_export import crop_pbf

    opened = []

    class RecordingPool(pbf_export._ThreadPool):
        def __init__(self, workers, initargs):
            opened.append(workers)
            super().__init__(workers, initargs)

    monkeypatch.setattr(pbf_export, "_ThreadPool", RecordingPool)
    monkeypatch.delenv("PYROSM_DECODE_BACKEND", raising=False)

    fp = get_data("helsinki_pbf")
    box = [24.9375, 60.1631, 24.9466, 60.1691]
    runs = {
        "crop": lambda out, **kw: crop_pbf(fp, str(out), box, **kw),
        "merge": lambda out, **kw: merge_pbf([fp], str(out), box, **kw),
    }
    expected = Path(runs[run](tmp_path / "one.osm.pbf", workers=1)).read_bytes()
    assert Path(runs[run](tmp_path / "auto.osm.pbf", workers="auto")).read_bytes() == expected
    assert opened == []  # the sample is too small for "auto" to go parallel
    # As if the file were large, on 4 cores: "auto" takes 2 threads for the 4 data blocks
    monkeypatch.setattr(pool, "_PARALLEL_MIN_FILE_BYTES", 1)
    monkeypatch.setattr(pool, "_available_cores", lambda: 4)
    assert Path(runs[run](tmp_path / "par.osm.pbf", workers="auto")).read_bytes() == expected
    assert Path(runs[run](tmp_path / "default.osm.pbf")).read_bytes() == expected
    assert opened == [2, 2]
    numpy_workers = runs[run](tmp_path / "numpy.osm.pbf", workers=np.int64(2))
    assert Path(numpy_workers).read_bytes() == expected and opened == [2, 2, 2]
    with pytest.raises(ValueError, match="'workers' should be"):
        runs[run](tmp_path / "bad.osm.pbf", workers="many")


def test_geocode_retries_a_cut_off_reply(monkeypatch):
    """A geocoding reply cut off part way (``IncompleteRead``) is fetched again, as downloads
    are. It raised the bare ``http.client.IncompleteRead``."""
    import http.client
    import json

    import pyrosm

    payload = json.dumps(
        [{"display_name": "Somewhere", "boundingbox": ["50.79", "50.81", "-0.15", "-0.13"]}]
    ).encode()
    calls = []

    class Reply:
        def __init__(self, cut):
            self.cut = cut

        def read(self):
            if self.cut:
                raise http.client.IncompleteRead(payload[:10], len(payload) - 10)
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, context=None, timeout=None):
        calls.append(request)
        return Reply(cut=len(calls) == 1)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    area = pyrosm.geocode("Somewhere")
    assert len(calls) == 2
    assert area.bounds == pytest.approx((-0.15, 50.79, -0.13, 50.81))


@pytest.mark.parametrize(
    "functions, expected",
    [
        ({"process_cpu_count": lambda: 3, "cpu_count": lambda: 64}, 3),
        ({"sched_getaffinity": lambda pid: {0, 1}, "cpu_count": lambda: 64}, 2),
        ({"cpu_count": lambda: 6}, 6),
        ({"process_cpu_count": lambda: None, "cpu_count": lambda: None}, 1),
    ],
    ids=["process_cpu_count", "affinity", "cpu_count", "unknown"],
)
def test_auto_workers_count_the_cores_the_process_may_use(monkeypatch, functions, expected):
    """``workers="auto"`` counts the CPU cores this process may run on (Python 3.13's
    ``os.process_cpu_count``, else its CPU affinity), not every core of the machine. It used
    ``os.cpu_count()``, so a job limited to 4 cores of a 64-core node got 62 threads."""
    import types

    import pyrosm.engine.pool as pool

    monkeypatch.setattr(pool, "os", types.SimpleNamespace(**functions))
    assert pool._available_cores() == expected


@pytest.mark.parametrize(
    "case, expected",
    [
        (
            "only_name",
            "486 elements would no longer be found by get_buildings(): their 'building' "
            "tag is not in the frame",
        ),
        ("note_gone", "486 elements are written without their 'building' tag"),
        ("building_kept", None),
        ("value_changed", None),
        ("network_nodes", None),
        ("deleted", None),
        (
            "unreadable_note",
            "1 element is written without its 'building' tag; get_buildings() will not "
            "find it",
        ),
    ],
)
def test_write_pbf_warns_when_elements_lose_the_tag_they_were_read_by(
    case, expected, tmp_path
):
    """#588: ``write_pbf`` writes the tags a frame carries and warns when elements lose the
    tag their read selected them by: naming the reader while the frame notes its read, and
    the tags pyrosm's readers detect features by once the note is gone (e.g. after a
    merge). It gave no warning. A changed value is an edit, the nodes of a network are kept
    for their ways and a deleted element is not written, so none of these warns."""
    import json
    import warnings

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    buildings = osm.get_buildings()
    json.dumps(buildings.attrs)  # plain JSON, which pandas needs to write attrs to Parquet
    named = buildings[["id", "osm_type", "name", "geometry"]]
    delete = None
    if case == "only_name":
        frame = [named, named.copy()]  # an element in two frames is counted once
    elif case == "note_gone":
        frame = named.copy()
        frame.attrs.clear()
    elif case == "building_kept":
        frame = buildings[["id", "osm_type", "name", "building", "geometry"]]
    elif case == "value_changed":
        yes = osm.get_buildings(custom_filter={"building": ["yes"]})
        frame = yes[["id", "osm_type", "building", "geometry"]].iloc[:1].copy()
        frame["building"] = "house"
    elif case == "network_nodes":
        nodes, _ = osm.get_network(nodes=True)
        frame = nodes[["id", "geometry"]].assign(note="checked")
    elif case == "deleted":
        frame = named[named["osm_type"] == "way"].iloc[:1]
        delete = [("way", int(frame["id"].iloc[0]))]
    else:
        frame = named.iloc[:1].copy()
        frame.attrs["pyrosm_read"] = "unreadable"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        osm.write_pbf(
            frame, str(tmp_path / "out.osm.pbf"), subset_only=True, delete=delete
        )
    lost = [
        w
        for w in caught
        if "no longer be found" in str(w.message) or "written without" in str(w.message)
    ]
    if expected is None:
        assert lost == []
    else:
        assert len(lost) == 1 and expected in str(lost[0].message), lost
        assert lost[0].filename == __file__  # it points at the user's call



@pytest.mark.parametrize(
    "case", ["one_segment", "segment_cleared", "several_ways", "conflict", "layers"]
)
def test_write_pbf_merges_the_rows_of_an_element(case, tmp_path):
    """``write_pbf`` wrote an element that is in several rows from its last row, so a
    change in one segment of a way, or in the first of two layers holding an element,
    was lost. A change in some rows is written for the whole element, with a warning
    when other rows of the same frame keep the old value; different new values raise."""
    import json
    import warnings

    import pandas as pd

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    out = str(tmp_path / "out.osm.pbf")
    if case == "layers":
        buildings, pois = osm.get_buildings(), osm.get_pois()
        both = buildings.merge(pois[["osm_type", "id"]]).iloc[0]
        element = (both["osm_type"], both["id"])
        edited = (buildings["osm_type"] == element[0]) & (buildings["id"] == element[1])
        buildings.loc[edited, "building"] = "edited"
        ids = buildings[["id", "osm_type", "geometry"]]  # keeps the element's own tags
        for frames in ([buildings, pois, ids], [ids, pois, buildings]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                osm.write_pbf(frames, out, subset_only=True)
            assert not [w for w in caught if "rows" in str(w.message)]
            written = OSM(out).get_buildings().set_index(["osm_type", "id"])
            assert written.loc[element, "building"] == "edited"
        return
    nodes, edges = osm.get_network("driving", nodes=True)
    def tag_keys(tags):
        return list(json.loads(tags)) if isinstance(tags, str) else []

    usable = edges["maxspeed"].notna() & edges["tags"].map(tag_keys).astype(bool)
    segments = edges[usable].groupby("id").size()
    way = segments[segments >= 3].index[0]
    rows = edges.index[edges["id"] == way]
    if case == "several_ways":
        for other in segments[segments >= 3].index[:4]:
            edges.loc[edges.index[edges["id"] == other][0], "maxspeed"] = "80"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            osm.write_pbf([nodes, edges], out)
        message = str(caught[0].message)
        assert len(caught) == 1 and message.startswith("4 tags were changed"), caught
        assert message.endswith("rows) and 1 more.")
        return
    if case == "conflict":
        edges.loc[rows[0], "maxspeed"], edges.loc[rows[1], "maxspeed"] = "80", "60"
        with pytest.raises(ValueError, match="different new values"):
            osm.write_pbf([nodes, edges], out)
        return
    if case == "one_segment":
        edges.loc[rows[0], "maxspeed"] = "80"
    else:  # removed from a column and from the JSON tags of one segment
        edges.loc[rows[0], "maxspeed"] = None
        other_tags = json.loads(edges.loc[rows[0], "tags"])
        removed = tag_keys(edges.loc[rows[0], "tags"])[0]
        del other_tags[removed]
        edges.loc[rows[0], "tags"] = json.dumps(other_tags)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        osm.write_pbf([nodes, edges], out)
    partial = [w for w in caught if "rows" in str(w.message)]
    assert len(partial) == 1, partial
    assert partial[0].filename == __file__  # it points at the user's call
    written = OSM(out).get_network("driving").set_index("id").loc[way]
    if case == "one_segment":
        message = f"of way {way} was changed in 1 of its {len(rows)} rows"
        assert message in str(partial[0].message)
        assert written["maxspeed"] == "80"
    else:
        assert str(partial[0].message).startswith("2 tags were changed")
        assert pd.isna(written["maxspeed"])
        assert removed not in json.loads(written["tags"])


@pytest.mark.parametrize("read", ["network", "network_segments", "length_column"])
def test_write_pbf_writes_the_mapper_tags_not_pyrosm_columns(read, tmp_path):
    """``write_pbf`` wrote a network's end node ids ``u`` and ``v`` as tags, dropped a
    way's own ``length`` and ``members`` tags and wrote a tag keyed ``id`` as
    ``id_tag``. It writes the way's tags as mapped, without the columns pyrosm
    computes."""
    import numpy as np

    from pyrosm import OSM
    from pyrosm.pbf_export import write_pbf_from_records

    osmium = pytest.importorskip("osmium")
    tags = {
        "highway": "residential",
        "id": "W-9",
        "length": "25",
        "members": "1",
        "u": "x",
    }
    nodes = {
        "id": np.array([1, 2, 3], dtype=np.int64),
        "lon": np.array([24.940, 24.941, 24.942]),
        "lat": np.array([60.170, 60.171, 60.170]),
        "version": np.ones(3, dtype=np.int64),
        "timestamp": np.full(3, 1600000000, dtype=np.int64),
        "changeset": np.zeros(3, dtype=np.int64),
        "tags": [None, None, None],
    }
    way = {"id": 10, "refs": [1, 2, 3], "version": 1, "timestamp": 1600000000}
    source = write_pbf_from_records(
        nodes,
        [dict(way, tags=tags)],
        [],
        str(tmp_path / "source.osm.pbf"),
        (24.94, 60.17, 24.942, 60.171),
    )
    osm = OSM(source)
    if read == "network":
        frame = osm.get_network("all")
    elif read == "network_segments":
        frame = list(osm.get_network("all", nodes=True))
    else:
        frame = osm.get_data_by_custom_criteria(
            {"highway": True}, tags_as_columns=["highway", "length", "u"]
        )
        assert frame[["length", "u"]].values.tolist() == [["25", "x"]]
    out = str(tmp_path / "out.osm.pbf")
    osm.write_pbf(frame, out)
    written = {}
    for element in osmium.FileProcessor(out):
        if element.is_way():
            written[element.id] = {t.k: t.v for t in element.tags}
    assert written == {10: tags}


def test_to_pbf_crops_to_the_polygon_with_the_objects_workers(tmp_path, monkeypatch):
    """``OSM(bounding_box=polygon).to_pbf()`` cropped to the polygon's bounding
    rectangle, so the file held features outside the polygon that reading with it
    leaves out, and it ignored ``OSM(workers=...)``. It crops to the polygon with the
    object's workers, and reading the file returns the features of the read."""
    from shapely.geometry import Polygon

    import pyrosm.pbf_export
    from pyrosm import OSM, get_data

    crop_pbf = pyrosm.pbf_export.crop_pbf
    workers = []

    def recording_crop(*args, **kwargs):
        workers.append(kwargs["workers"])
        return crop_pbf(*args, **kwargs)

    monkeypatch.setattr(pyrosm.pbf_export, "crop_pbf", recording_crop)
    triangle = Polygon([(24.935, 60.165), (24.945, 60.165), (24.935, 60.170)])
    osm = OSM(get_data("helsinki_pbf"), bounding_box=triangle, workers=2)
    read = osm.get_buildings()
    cropped = OSM(osm.to_pbf(str(tmp_path / "triangle.osm.pbf"))).get_buildings()
    assert workers == [2]
    assert sorted(zip(cropped["osm_type"], cropped["id"])) == sorted(
        zip(read["osm_type"], read["id"])
    )


def test_extracts_with_the_same_file_name_download_to_their_own_files(tmp_path):
    """``get_data("usa/georgia")`` and ``get_data("europe/georgia")`` both saved
    ``georgia-latest.osm.pbf``, so the second returned the first one's file, and the
    Geofabrik downloads of ``get_data_by_bbox`` and ``get_data_by_area`` shared the name
    too. A file name that several extracts have gets the extract's region in front."""
    import io

    from pyrosm import get_data
    from pyrosm.data.geofabrik_index import extract_file_name

    class Response(io.BytesIO):
        status = 200

        def __init__(self, url):
            super().__init__(url.encode())
            self.headers = {"Content-Length": str(len(url))}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    class Opener:
        def open(self, request, timeout=None):
            return Response(request.full_url)

    paths = [
        Path(get_data(name, directory=tmp_path, opener=Opener(), progress=False))
        for name in ("usa/georgia", "europe/georgia")
    ]
    assert [p.name for p in paths] == [
        "us-georgia-latest.osm.pbf",
        "europe-georgia-latest.osm.pbf",
    ]
    urls = [p.read_bytes().decode() for p in paths]  # each file holds its own download
    assert urls[0].endswith("download.geofabrik.de/north-america/us/georgia-latest.osm.pbf")
    assert urls[1].endswith("download.geofabrik.de/europe/georgia-latest.osm.pbf")
    finland = "https://download.geofabrik.de/europe/finland-latest.osm.pbf"
    assert extract_file_name(finland) == "finland-latest.osm.pbf"


@pytest.mark.parametrize("graph_type", ["networkx", "igraph", "pandarm"])
def test_graph_exports_hold_the_given_nodes_and_edges(graph_type):
    """Graph exports went wrong for frames edited before the export. With nodes cropped
    to an area, igraph gave edges another edge's attributes and pandarm failed; with
    edges filtered (index gaps), pandarm got extra edges and wrong weights; projected
    frames gave nodes lon/lat ``x``/``y`` in a projected graph; a node id column named
    ``node_id`` gave igraph no edges and was ignored by pandarm. Each graph holds the
    edges between the given nodes, in the frames' CRS."""
    pytest.importorskip(graph_type)
    from shapely.geometry import box

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    nodes, edges = osm.get_network("walking", nodes=True)
    edges = edges[edges["highway"] != "footway"]  # leaves gaps in the index
    lon, lat = nodes["lon"], nodes["lat"]
    area = box(lon.quantile(0.25), lat.quantile(0.25), lon.median(), lat.median())
    nodes = nodes[nodes.within(area)].rename(columns={"id": "node_id"})
    nodes = nodes.assign(x=nodes["lon"], y=nodes["lat"])  # degrees, as lon/lat
    nodes, edges = nodes.to_crs(3067), edges.to_crs(3067)
    inside = edges["u"].isin(nodes["node_id"]) & edges["v"].isin(nodes["node_id"])
    # A walking network is two-way: each edge between the given nodes, both ways.
    count, length = 2 * int(inside.sum()), 2 * edges.loc[inside, "length"].sum()
    graph = osm.to_graph(
        nodes,
        edges,
        graph_type=graph_type,
        node_id_col="node_id",
        retain_all=True,
        simplify=False,
    )
    x = dict(zip(nodes["node_id"], nodes.geometry.x))
    if graph_type == "networkx":
        assert graph.number_of_edges() == count
        weights = [d["length"] for _, _, d in graph.edges(data=True)]
        assert all(d["x"] == x[node] for node, d in graph.nodes(data=True))
    elif graph_type == "igraph":
        ids = graph.vs["node_id"]
        assert graph.ecount() == count
        assert all((ids[e.source], ids[e.target]) == (e["u"], e["v"]) for e in graph.es)
        weights = graph.es["length"]
    else:
        assert len(graph.edges_df) == count
        weights = graph.edges_df["length"]
        assert (graph.nodes_df["x"] == graph.nodes_df.index.map(x)).all()
    assert sum(weights) == pytest.approx(length)


def test_history_read_without_timestamp_parses_once(monkeypatch):
    """Reading a history file without a timestamp took a new "now" on every call, so
    every call parsed the file again and warned again, and ``keep_other_tags=False``
    failed asking for the out-of-core engine already in use. The object chooses "now"
    once and warns once, at the user's call, and the error names history reads."""
    import warnings

    import pyrosm.pyrosm
    from pyrosm import OSM, get_data

    parse = pyrosm.pyrosm.parse_osm_data
    parses = []

    def counting_parse(*args, **kwargs):
        parses.append(kwargs["unix_time_filter"])
        return parse(*args, **kwargs)

    monkeypatch.setattr(pyrosm.pyrosm, "parse_osm_data", counting_parse)
    osm = OSM(get_data("helsinki_test_history_pbf"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first, second = osm.get_buildings(), osm.get_buildings()
    implicit = [w for w in caught if "without user-defined timestamp" in str(w.message)]
    assert len(parses) == 1 and parses[0] is not None
    assert len(implicit) == 1 and implicit[0].filename == __file__
    assert len(first) == len(second)
    with pytest.raises(ValueError, match="not supported for history reads"):
        osm.get_data_by_custom_criteria({"building": True}, keep_other_tags=False)


@pytest.mark.parametrize(
    "network_type, tags_to_keep, simplify",
    [("driving", ["name"], True), ("cycling", ["name", "oneway"], False)],
)
def test_get_graph_with_tags_to_keep_builds_the_default_graph(
    network_type, tags_to_keep, simplify
):
    """``get_graph(tags_to_keep=...)`` read only the listed tags, so a driving graph
    without ``oneway`` was two-way everywhere, a cycling graph without
    ``oneway:bicycle`` lost the one-way streets open to cycling both ways, and the
    simplification merged edges across changes of ``maxspeed`` and ``highway``. The tags
    the graph is built from are always read, so the graph is the default one."""
    pytest.importorskip("networkx")
    from pyrosm import OSM, get_data
    from pyrosm.graphs import direction_tags

    osm = OSM(get_data("helsinki_pbf"))
    options = dict(graph_type="networkx", simplify=simplify, retain_all=True)
    expected = osm.get_graph(network_type, **options)
    graph = osm.get_graph(network_type, tags_to_keep=tags_to_keep, **options)
    assert sorted(graph.edges()) == sorted(expected.edges())
    assert direction_tags("oneway", network_type) == {
        "driving": ["oneway", "junction"],
        "cycling": ["oneway", "oneway:bicycle", "junction"],
    }[network_type]
    assert direction_tags("oneway", "walking") == []  # every edge is two-way


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_osm_keys_to_keep_alone_keeps_the_elements_with_the_keys(engine):
    """``get_data_by_custom_criteria(osm_keys_to_keep=...)`` without a ``custom_filter``
    returned every element of the file. The keys are the first level of filtering, so
    only the elements with one of them are returned, whatever ``filter_type`` says."""
    import warnings

    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"), engine=engine)
    buildings = sorted(zip(*osm.get_buildings()[["osm_type", "id"]].values.T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for filter_type in ("keep", "exclude"):
            read = osm.get_data_by_custom_criteria(
                osm_keys_to_keep="building", filter_type=filter_type
            )
            assert sorted(zip(*read[["osm_type", "id"]].values.T)) == buildings
            # The read notes the key it selected by (see write_pbf's lost-tag warning).
            assert "building" in str(read.attrs["pyrosm_read"]["filter"])


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_get_boundaries_finds_the_name_as_plain_text(engine, tmp_path):
    """``get_boundaries(name=...)`` matched the name as a case-sensitive regular
    expression, so "Frankfurt (Oder)" was never found and "(" raised an error, and no
    match gave an empty frame without a warning. The name is plain text, case is ignored,
    and no match gives None with the usual warning."""
    import numpy as np

    from pyrosm import OSM
    from pyrosm.pbf_export import write_pbf_from_records

    corners = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
    lon = [24.90 + x for x in (0, 0.02) for x, _ in corners]
    lat = [60.10 + y for _ in (0, 0.02) for _, y in corners]
    nodes = {
        "id": np.arange(1, 9, dtype=np.int64),
        "lon": np.array(lon),
        "lat": np.array(lat),
        "version": np.ones(8, dtype=np.int64),
        "timestamp": np.full(8, 1600000000, dtype=np.int64),
        "changeset": np.zeros(8, dtype=np.int64),
        "tags": [None] * 8,
    }
    ways = [
        {"id": 10 + i, "refs": refs, "version": 1, "timestamp": 1600000000, "tags": {}}
        for i, refs in enumerate(([1, 2, 3, 4, 1], [5, 6, 7, 8, 5]))
    ]
    relations = [
        {
            "id": 20 + i,
            "members": [("way", 10 + i, "outer")],
            "version": 1,
            "timestamp": 1600000000,
            "changeset": 0,
            "tags": {"type": "boundary", "boundary": "administrative", "name": name},
        }
        for i, name in enumerate(("Frankfurt (Oder)", "Kamppi"))
    ]
    path = write_pbf_from_records(
        nodes,
        ways,
        relations,
        str(tmp_path / "boundaries.osm.pbf"),
        (24.9, 60.1, 24.93, 60.13),
    )
    osm = OSM(path, engine=engine)
    assert osm.get_boundaries(name="frankfurt (oder)")["name"].tolist() == [
        "Frankfurt (Oder)"
    ]
    assert osm.get_boundaries(name="(")["name"].tolist() == ["Frankfurt (Oder)"]
    assert osm.get_boundaries(name="KAMP")["name"].tolist() == ["Kamppi"]
    with pytest.warns(UserWarning, match="Could not find any boundaries"):
        assert osm.get_boundaries(name="Helsinki") is None


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_readers_keep_the_key_they_select_by_as_a_column(engine):
    """With ``tags_to_keep``, the readers left out the key they select by, so a row did
    not show what it was; ``get_data_by_custom_criteria`` had no ``tags_to_keep``, and in
    memory it added ``extra_attributes`` to the caller's ``tags_as_columns`` list. Every
    reader keeps its filter's keys as columns and leaves the caller's lists as they are."""
    import warnings

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine=engine)
    columns = ["name"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reads = {
            "building": osm.get_buildings(tags_to_keep=columns),
            "amenity": osm.get_pois(
                custom_filter={"amenity": ["cafe"]}, tags_to_keep=columns
            ),
            "highway": osm.get_network(tags_to_keep=columns),
            "landuse": osm.get_landuse(tags_to_keep=columns),
            "shop": osm.get_data_by_custom_criteria({"shop": True}, tags_to_keep=columns),
            "cuisine": osm.get_data_by_custom_criteria(
                {"cuisine": True},
                tags_as_columns=columns,
                extra_attributes=["opening_hours"],
            ),
        }
    for key, frame in reads.items():
        assert frame[key].notna().all(), key
        assert "name" in frame.columns
    assert columns == ["name"]
    with pytest.raises(ValueError, match="not both"):
        osm.get_data_by_custom_criteria(
            {"shop": True}, tags_as_columns=columns, tags_to_keep=columns
        )


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_get_network_plain_dict_filter_keeps_its_matches(engine):
    """``get_network(custom_filter={"highway": ["footway"]})`` returned every way but the
    footways, while the same filter as a regular expression or a bracket filter, and a
    plain dict in every other reader, keeps the matches. Every form keeps them;
    ``filter_type="exclude"`` gives the other ways."""
    import re

    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"), engine=engine)
    forms = (
        {"highway": ["footway"]},
        {"highway": [re.compile("^footway$")]},
        '["highway"="footway"]',
    )
    ids = [sorted(osm.get_network(custom_filter=form)["id"]) for form in forms]
    assert ids[0] == ids[1] == ids[2] and len(ids[0]) > 0
    others = osm.get_network(custom_filter=forms[0], filter_type="exclude")
    assert "footway" not in set(others["highway"]) and not set(others["id"]) & set(ids[0])


def test_progress_of_several_passes_keeps_moving(monkeypatch):
    """A bar of several passes was reset at each pass, which restarted its 2-second
    delay, so it stood at 0 % through passes shorter than that, and a long step without
    reports looked frozen. Steps report one progress from 0 to 100 % that names the
    current step and never goes back, also when the steps to come change; a reset bar
    keeps its start time; and an open bar is redrawn while no report comes."""
    import time

    from tqdm import std

    from pyrosm.utils import progress

    calls = []
    steps = progress.Steps(lambda *call: calls.append(call), "Cropping x", {"b": 3.0})
    steps.plan(["a", "b"])  # a takes 25 %, b 75 %
    steps.start_step("a")(50, 100)
    steps.plan(["c", "d"])  # the rest, from 12.5 %, in two equal steps
    advance = steps.start_step("c")
    advance(0, 0)  # an empty step: no report
    advance(1, 1)
    steps.start_step("a")(0, 1)  # not planned: it takes the rest, and nothing goes back
    steps.finish()
    done = [d for d, _ in calls]
    assert done[:3] == [0.0, 12.5, 56.2] and done[-1] == 100.0 and done == sorted(done)
    assert all(total == 100 for _, total in calls)

    monkeypatch.setattr(progress, "_bar_class", lambda: (std.tqdm, False))
    monkeypatch.setattr(progress, "_clock", iter([0.0, 5.0, 6.0]).__next__)
    bar = progress.Bar("Cropping x", delay=2.0, unit="%")
    steps = progress.Steps(bar, "Cropping x")
    steps.start_step("finding the ways")(1, 2)
    assert bar.bar.desc.startswith("Cropping x: finding the ways")
    started = bar.bar.start_t
    bar(10, 100)  # back: a reset after the delay keeps the start time
    assert bar.bar.start_t == started
    bar.close()
    # A hidden bar (no terminal) is reset after its delay as well.
    monkeypatch.setattr(progress, "_bar_class", lambda: (std.tqdm, True))
    monkeypatch.setattr(progress, "_clock", iter([0.0, 5.0, 6.0]).__next__)
    hidden = progress.Bar("Cropping x", delay=2.0, unit="%")
    hidden(50, 100)
    hidden(10, 100)
    hidden.close()

    refreshed = []

    class Counted(std.tqdm):
        def refresh(self, *args, **kwargs):
            refreshed.append(True)
            return super().refresh(*args, **kwargs)

    def wait_for(condition):
        deadline = time.monotonic() + 10.0
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)

    monkeypatch.setattr(progress, "_bar_class", lambda: (Counted, False))
    monkeypatch.setattr(progress, "_clock", time.monotonic)
    monkeypatch.setattr(progress, "_REDRAW_SECONDS", 0.01)
    bar = progress.Bar("Cropping x", unit="%")
    bar(1, 100)
    before = len(refreshed)
    wait_for(lambda: len(refreshed) > before)
    bar.close()
    assert len(refreshed) > before
    # Not before its delay, through several redraw rounds.
    bar = progress.Bar("Cropping x", delay=60.0, unit="%")
    woken = []
    wait = bar._closed.wait
    bar._closed.wait = lambda timeout: woken.append(True) or wait(timeout)
    before = len(refreshed)
    bar(1, 100)  # starts the redraw thread
    wait_for(lambda: len(woken) >= 3)
    bar.close()
    assert len(woken) >= 3 and len(refreshed) == before


_BUILD = ["collecting the elements", "building the ways", "building the relations"]


@pytest.mark.parametrize(
    "engine, read, steps",
    [
        (
            "out_of_core",
            "buildings",
            [
                "indexing the file",
                "decoding the blocks",
                "collecting the elements",
                "building the geometries",
                "loading the result",
            ],
        ),
        (
            "out_of_core",
            "network",
            [
                "indexing the file",
                "decoding the blocks",
                "building the edges",
                "writing the result",
                "loading the result",
            ],
        ),
        (
            "out_of_core",
            "network with nodes",
            [
                "indexing the file",
                "decoding the blocks",
                "gathering the nodes",
                "building the edges",
                "writing the result",
                "loading the result",
            ],
        ),
        ("out_of_core", "second buildings", ["loading the result"]),
        ("in_memory", "buildings", ["decoding the blocks"] + _BUILD),
        (
            "in_memory",
            "network with nodes",
            ["decoding the blocks", "collecting the elements", "building the edges"],
        ),
        ("in_memory", "second buildings", _BUILD),
        (
            "in_memory",
            "box buildings",
            [
                "decoding the blocks",
                "completing the relations",
                "completing the ways",
            ]
            + _BUILD,
        ),
        (
            "in_memory",
            "history buildings",
            ["decoding the blocks", "selecting the versions"] + _BUILD,
        ),
    ],
)
def test_read_progress_covers_the_whole_read(
    engine, read, steps, fresh_cache, monkeypatch
):
    """A read reported only its decoding (12-20 % of an out-of-core read of Paris, 17 %
    of one in memory), then went silent while it built the geometries, wrote the cache
    and loaded the result, or selected the versions of a history read (92 % of a read
    of the Helsinki history sample); a second read on the same object showed nothing.
    The whole read is one progress through named steps, from 0 to 100 %."""
    from pyrosm import OSM, get_data
    from pyrosm.utils import progress

    fp, options = get_data("test_pbf"), {}
    if read == "box buildings":
        fp = get_data("helsinki_pbf")
        options = {"bounding_box": [24.93, 60.16, 24.95, 60.18], "complete_relations": True}
    elif read == "history buildings":
        fp = get_data("helsinki_test_history_pbf")
    calls = {
        "buildings": lambda osm: osm.get_buildings(),
        "network": lambda osm: osm.get_network("driving"),
        "network with nodes": lambda osm: osm.get_network("driving", nodes=True),
        "history buildings": lambda osm: osm.get_buildings(timestamp="2019-01-01"),
    }
    seen = []
    osm = OSM(fp, engine=engine, progress=lambda *report: seen.append(report), **options)
    if read == "second buildings":
        calls["buildings"](osm)
        seen.clear()
    started = []
    real = progress.Steps.start_step

    def recorded(self, name):
        started.append(name)
        return real(self, name)

    monkeypatch.setattr(progress.Steps, "start_step", recorded)
    calls.get(read, calls["buildings"])(osm)
    assert list(dict.fromkeys(started)) == steps
    done = [d for d, _ in seen]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100) and done == sorted(done)


def test_history_ways_are_chosen_without_a_frame_of_their_records(monkeypatch):
    """A history read put every version of every way into one table with a column for
    each tag key of the file (1,211 columns for 10 tags per way in the Helsinki sample)
    and cleaned it again: 21 of the 23 s of a read. The versions are chosen from the
    ways' ids and deletion flags, and each way is its last version's record as read."""
    import pandas as pd

    import pyrosm.pbfreader as reader
    from pyrosm import get_data
    from pyrosm.utils import datetime_to_unix_time

    fp = get_data("helsinki_test_history_pbf")
    when = datetime_to_unix_time(pd.to_datetime("2014-01-01", utc=True))
    latest = {}
    for way in reader.parse_osm_data(fp, None, False, None)[1]:
        if way["timestamp"] <= when:
            latest[way["id"]] = way
    expected = [way for _, way in sorted(latest.items()) if way["visible"]]

    class NoRecordFrames:
        def __getattr__(self, name):
            return getattr(pd, name)

        @staticmethod
        def DataFrame(data=None, *args, **kwargs):
            assert not isinstance(data, list), "a frame of way records"
            return pd.DataFrame(data, *args, **kwargs)

    monkeypatch.setattr(reader, "pd", NoRecordFrames())
    ways = reader.parse_osm_data(fp, None, False, when)[1]
    assert ways == expected and len(ways) > 0


@pytest.mark.parametrize("parsed", [True, False])
def test_write_pbf_reports_progress_through_its_steps(tmp_path, monkeypatch, parsed):
    """write_pbf showed no progress and took no progress parameter (33 s without a sign
    for the buildings of Helsinki). It reports one progress from 0 to 100 % through its
    steps, after the read of the file when it is not read yet; by default it follows
    the object's progress, progress=False shows nothing, and other values are refused."""
    from pyrosm import OSM, get_data
    from pyrosm.utils import progress

    fp = get_data("test_pbf")
    seen = []
    osm = OSM(fp, engine="in_memory", progress=lambda *report: seen.append(report))
    buildings = osm.get_buildings()
    if not parsed:
        osm = OSM(fp, engine="in_memory", progress=lambda *report: seen.append(report))
    seen.clear()
    started = []
    real = progress.Steps.start_step

    def recorded(self, name):
        started.append(name)
        return real(self, name)

    monkeypatch.setattr(progress.Steps, "start_step", recorded)
    osm.write_pbf(buildings, str(tmp_path / "all.osm.pbf"))
    steps = [
        "applying the edits",
        "preparing the nodes",
        "preparing the ways",
        "preparing the relations",
        "writing the file",
    ]
    assert list(dict.fromkeys(started)) == ([] if parsed else ["decoding the blocks"]) + steps
    done = [d for d, _ in seen]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100) and done == sorted(done)
    seen.clear()
    osm.write_pbf(buildings, str(tmp_path / "quiet.osm.pbf"), progress=False)
    assert seen == []
    with pytest.raises(ValueError, match="progress must be"):
        osm.write_pbf(buildings, str(tmp_path / "refused.osm.pbf"), progress="yes")


@pytest.mark.parametrize("keep_metadata", [True, False])
def test_write_pbf_reads_rows_and_nodes_as_columns(tmp_path, monkeypatch, keep_metadata):
    """write_pbf went through the frame with iterrows() and copied every node of the
    file with one call each: 33 s for the buildings of Helsinki (17 s now). The rows
    and the nodes are read as columns, also without metadata, and the written file
    holds the same elements."""
    import geopandas as gpd

    import pyrosm.pbf_writer as writer
    from pyrosm import OSM, get_data

    osm = OSM(
        get_data("test_pbf"),
        engine="in_memory",
        keep_metadata=keep_metadata,
        progress=False,
    )
    buildings = osm.get_buildings()

    def refused(*args, **kwargs):
        raise AssertionError("one Python call per row")

    added = []
    add_node = writer._RecordBuilder.add_node
    monkeypatch.setattr(gpd.GeoDataFrame, "iterrows", refused)
    monkeypatch.setattr(
        writer._RecordBuilder,
        "add_node",
        lambda self, *node: added.append(node) or add_node(self, *node),
    )
    path = osm.write_pbf(buildings, str(tmp_path / "out.osm.pbf"))
    assert added == []
    written = OSM(path, engine="in_memory", progress=False).get_buildings()
    assert sorted(written["id"]) == sorted(buildings["id"])


def test_write_pbf_copies_the_nodes_of_any_store():
    """The copy of the nodes as arrays writes what the copy one node at a time wrote: a
    repeated id in each of its rows, edited in each, and large integer metadata exactly
    where some nodes have none."""
    from types import SimpleNamespace

    import numpy as np

    import pyrosm.pbf_writer as writer
    from pyrosm.node_lookup import NodeLocations

    store = NodeLocations(
        {
            "id": np.array([7, 7, 8]),
            "lon": [1.0, 1.0, 2.0],
            "lat": [3.0, 3.0, 4.0],
            "changeset": np.array([9007199254740993, None, 5], dtype=object),
            "tags": np.array([None, {"a": "b"}, None], dtype=object),
        }
    )
    plan = SimpleNamespace(
        removed_nodes=set(), node_coords={7: (5.0, 6.0)}, node_tags={8: {"c": "d"}}
    )
    builder = writer._RecordBuilder()
    writer._add_base_nodes(builder, store, {}, plan)
    assert builder.node_ids == [7, 7, 8] and builder.lons == [5.0, 5.0, 2.0]
    assert builder.css == [9007199254740993, 0, 5] and builder.vers == [0, 0, 0]
    assert builder.ntags == [None, {"a": "b"}, {"c": "d"}]


_GRAPH_STEPS = ["directing the edges", "keeping the largest connected part"]


@pytest.mark.parametrize(
    "engine, simplify, read",
    [
        (
            "out_of_core",
            True,
            [
                "indexing the file",
                "decoding the blocks",
                "gathering the nodes",
                "building the edges",
            ],
        ),
        (
            "out_of_core",
            False,
            [
                "indexing the file",
                "decoding the blocks",
                "gathering the nodes",
                "building the edges",
                "writing the result",
                "loading the result",
            ],
        ),
        (
            "in_memory",
            True,
            ["decoding the blocks", "collecting the elements", "building the edges"],
        ),
    ],
)
def test_get_graph_reports_one_progress(engine, simplify, read, fresh_cache, monkeypatch):
    """get_graph showed the read and the graph as two progresses, the graph's counting
    steps, and the graph steps ignored the progress of the OSM object (progress=False
    still drew a bar). One progress from 0 to 100 % covers the read and the graph
    steps, and progress=False draws no bar."""
    from pyrosm import OSM, get_data
    from pyrosm.utils import progress

    started = []
    real = progress.Steps.start_step
    monkeypatch.setattr(
        progress.Steps,
        "start_step",
        lambda self, name: started.append(name) or real(self, name),
    )
    fp = get_data("test_pbf")
    seen = []
    osm = OSM(fp, engine=engine, progress=lambda *report: seen.append(report))
    osm.get_graph("driving", graph_type="networkx", simplify=simplify)
    graph = _GRAPH_STEPS + (["simplifying"] if simplify else [])
    assert list(dict.fromkeys(started)) == read + graph + ["building the graph"]
    done = [d for d, _ in seen]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100) and done == sorted(done)

    bars = []
    monkeypatch.setattr(progress.Bar, "__init__", lambda self, *a, **k: bars.append(a))
    OSM(fp, engine=engine, progress=False).get_graph("driving", graph_type="networkx")
    assert bars == []


def test_find_extracts_reports_reading_the_indexes(tmp_path, monkeypatch):
    """On first use find_extracts downloaded Movisda's indexes (29 MB, 8 s for an area
    in Helsinki) without a sign, and took no progress. It reports reading the indexes,
    by their bytes, and then the extract sizes, from 0 to 100 %."""
    from urllib.error import URLError

    from shapely.geometry import box

    import pyrosm.data.extract_index as ei
    from pyrosm import find_extracts
    from pyrosm.utils import progress

    def cached_index(url, path, update=False, check=None, net=None, progress=None):
        if "/admin/" in url:
            # A server that gives no size, and no copy to take one from.
            for received in (0, 5, 10):
                progress(received, None)
            raise URLError("down")
        for received in (0, 5, 10):
            progress(received, 10)
        return ei._MOVISDA_GRID_PATH

    def download_sizes(urls, directory, update=False, net=None, fallback=None, progress=None):
        progress(0, 1)
        progress(1, 1)
        return {}

    monkeypatch.setattr(ei, "_cached_index", cached_index)
    monkeypatch.setattr(ei, "_download_sizes", download_sizes)
    started = []
    real = progress.Steps.start_step
    monkeypatch.setattr(
        progress.Steps,
        "start_step",
        lambda self, name: started.append(name) or real(self, name),
    )
    seen = []
    with pytest.warns(UserWarning, match="administrative index is unavailable"):
        find_extracts(
            box(24.9, 60.15, 25.0, 60.2),
            directory=tmp_path,
            progress=lambda *report: seen.append(report),
        )
    assert started == ["reading the extract indexes", "reading the extract sizes"]
    done = [d for d, _ in seen]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100) and done == sorted(done)
    assert len(set(done)) > 4


@pytest.mark.parametrize(
    "function, place",
    [
        ("get_data", "test_pbf"),
        ("get_data_by_bbox", [24.9, 60.15, 25.0, 60.2]),
        ("get_data_by_geocoding", "Helsinki"),
        ("get_data_by_area", [24.9, 60.15, 25.0, 60.2]),
        ("find_extracts", [24.9, 60.15, 25.0, 60.2]),
    ],
)
def test_download_functions_refuse_an_invalid_progress_first(
    function, place, monkeypatch
):
    """A progress that is not True, False, None or a function was refused only when the
    first download started: after the extract indexes or the geocoding request, and
    never for a file bundled with pyrosm. Every download function refuses it before
    any request."""
    import pyrosm
    import pyrosm.data.geocoding as geocoding
    import pyrosm.utils.download as dl

    def no_request(*args, **kwargs):
        pytest.fail("a request was sent")

    monkeypatch.setattr(dl._Net, "open", no_request)
    monkeypatch.setattr(geocoding, "open_url", no_request)
    with pytest.raises(ValueError, match="progress must be"):
        getattr(pyrosm, function)(place, progress="yes")


def test_index_download_keeps_its_failures(tmp_path, monkeypatch):
    """Reading an index in chunks for its progress must not hide two failures: an index
    that stops short of its Content-Length is fetched again, and left out when it keeps
    failing, and an error of the progress function reaches the caller of find_extracts
    unchanged, after one request."""
    import io

    from shapely.geometry import box

    import pyrosm.data.extract_index as ei
    import pyrosm.utils.download as dl

    class Response(io.BytesIO):
        headers = {"Content-Length": "10"}

    calls = []

    def short(url, method="GET", headers=None, timeout=None, opener=None):
        calls.append(url)
        return Response(b"abc")

    monkeypatch.setattr(dl, "open_url", short)
    monkeypatch.setattr(dl, "_sleep", lambda seconds: None)
    with pytest.raises(OSError, match="stopped after 3 of 10 bytes"):
        ei._cached_index("https://x/index.geojson", tmp_path / "index.geojson")
    assert len(calls) == dl._ATTEMPTS

    def stop(done, total):
        if done > 0:
            raise ValueError("stop")

    calls.clear()
    with pytest.raises(ValueError, match="stop"):
        ei.find_extracts(box(24.9, 60.15, 25.0, 60.2), directory=tmp_path, progress=stop)
    assert len(calls) == 1


def test_crops_merges_and_writes_log_a_line(tmp_path, caplog, monkeypatch):
    """Every read logs a line, but crops, merges and writes logged nothing. Each logs one
    INFO line with its files, its output and the seconds it took."""
    import logging

    from pyrosm import OSM, get_data, merge_pbf
    from pyrosm.pbf_export import crop_pbf

    monkeypatch.setattr(logging.getLogger("pyrosm"), "propagate", True)
    caplog.set_level(logging.INFO, logger="pyrosm")
    fp = get_data("test_pbf")
    crop_pbf(
        fp,
        str(tmp_path / "crop.osm.pbf"),
        bounding_box=[26.94, 60.525, 26.96, 60.535],
        progress=False,
    )
    merge_pbf([fp], str(tmp_path / "merged.osm.pbf"), progress=False)
    osm = OSM(fp, engine="in_memory", progress=False)
    osm.write_pbf(osm.get_buildings(), str(tmp_path / "written.osm.pbf"))
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("pyrosm")]
    lines = {message.split()[0]: message for message in messages}
    ops = [message.split()[0] for message in messages]
    assert [ops.count(op) for op in ("crop_pbf", "merge_pbf", "write_pbf")] == [1, 1, 1]
    assert "file=test.osm.pbf" in lines["crop_pbf"]
    assert "output=crop.osm.pbf" in lines["crop_pbf"]
    assert "files=test.osm.pbf " in lines["merge_pbf"]
    assert "output=merged.osm.pbf" in lines["merge_pbf"]
    assert "output=written.osm.pbf" in lines["write_pbf"]
    assert all("seconds=" in lines[op] for op in ("crop_pbf", "merge_pbf", "write_pbf"))


def test_read_from_the_cache_warns_as_its_build(fresh_cache, tmp_path):
    """A bounding-box read warned that relations reach beyond the box only when it was
    built; the same read served from the cache gave no warning. It gives the warnings
    of the build again. The record follows its result: a rebuild without warnings
    removes it, an unreadable one gives nothing, and builds running at once in two
    threads each record only their own warnings."""
    import threading
    import warnings

    from pyrosm import OSM, get_data
    from pyrosm.engine import cache
    from pyrosm.utils import recording_warnings, warn_recorded

    fp = get_data("helsinki_pbf")
    for read in ("built", "from the cache"):
        with pytest.warns(UserWarning, match="extend beyond the bounding box"):
            OSM(fp, bounding_box=[24.93, 60.16, 24.95, 60.18], progress=False).get_buildings()

    path = tmp_path / "result.parquet"
    cache._keep_warnings(path, ["old"])
    cache._keep_warnings(path, [])
    assert not cache._warnings_file(path).exists()
    with pytest.raises(TypeError):
        cache._keep_warnings(path, [object()])
    assert list(tmp_path.iterdir()) == []
    cache._warnings_file(path).write_text("{not json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cache._warn_again(path)
    assert caught == []

    both = threading.Barrier(2)
    recorded = {}

    def build(name):
        with recording_warnings() as messages:
            both.wait()
            warn_recorded(name)
            both.wait()
        recorded[name] = messages

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        threads = [threading.Thread(target=build, args=(name,)) for name in "ab"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert recorded == {"a": ["a"], "b": ["b"]}


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_empty_box_warns_once_in_either_engine(engine, fresh_cache):
    """A bounding box without data gave the in-memory engine two warnings and the
    out-of-core engine one. Both give the same single warning."""
    import warnings

    from pyrosm import OSM, get_data

    osm = OSM(
        get_data("test_pbf"),
        engine=engine,
        bounding_box=[0.0, 0.0, 0.01, 0.01],
        progress=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert osm.get_buildings() is None
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert messages == ["Could not find any building elements for given area."]


def test_area_crop_is_used_again(tmp_path, monkeypatch):
    """get_data_by_area cropped again on every call, so each call wrote a new file and
    the result cache filled with copies, and crop=False ignored output_path for one
    extract. A file pyrosm names after the area is used again while it, its extract and
    the area are unchanged; update=True, a newer or other extract, another area of the same
    name or an explicit output_path crop again, and crop=False copies the extract to
    output_path."""
    import os
    import shutil

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    import pyrosm
    import pyrosm.data.extract_index as ei
    import pyrosm.pbf_export
    from pyrosm import get_data

    area = [24.93, 60.16, 24.96, 60.18]
    source = tmp_path / "bbbike_Helsinki.osm.pbf"
    shutil.copy(get_data("helsinki_pbf"), source)
    os.utime(source, (1e9, 1e9))
    found = gpd.GeoDataFrame(
        {
            "provider": ["BBBike"],
            "id": ["Helsinki"],
            "name": ["Helsinki"],
            "url": ["https://b/Helsinki/Helsinki.osm.pbf"],
            "bytes": pd.array([50], dtype="Int64"),
            "contains": [True],
        },
        geometry=[box(*area)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(ei, "find_extracts", lambda *a, **k: found)
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: str(source))
    crops = []
    crop_pbf = pyrosm.pbf_export.crop_pbf
    monkeypatch.setattr(
        pyrosm.pbf_export,
        "crop_pbf",
        lambda *a, **k: crops.append(a[1]) or crop_pbf(*a, **k),
    )

    def read(where=area, **options):
        return pyrosm.get_data_by_area(
            where, directory=tmp_path, progress=False, **options
        ).path

    first = read()
    assert read() == first and len(crops) == 1
    read(update=True)
    os.utime(source)
    read()
    assert len(crops) == 3
    # Another extract for the same box, or a crop changed since it was written.
    other = tmp_path / "geofabrik_finland.osm.pbf"
    shutil.copy(source, other)
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: str(other))
    read()
    os.utime(first)
    read()
    assert read() == first and len(crops) == 5
    # A box that differs from the area only beyond the five decimals of the name.
    assert read([24.93, 60.16, 24.96, 60.180001]) == first and len(crops) == 6
    own = str(tmp_path / "own.osm.pbf")
    assert read(output_path=own) == read(output_path=own) == own and len(crops) == 8

    copied = tmp_path / "copied.osm.pbf"
    assert read(crop=False, output_path=str(copied)) == str(copied)
    assert copied.read_bytes() == other.read_bytes() and len(crops) == 8

    # A copy where the file system has no hard links, and none left when it fails too.
    import pyrosm.utils.download as dl

    dl.copy_atomic(copied, copied)
    no_links = tmp_path / "no_links"

    def refused(*args):
        raise OSError("refused")

    monkeypatch.setattr(dl.os, "link", refused)
    source.chmod(0o600)
    dl.copy_atomic(source, no_links / "copy.osm.pbf")
    assert (no_links / "copy.osm.pbf").read_bytes() == source.read_bytes()
    assert (no_links / "copy.osm.pbf").stat().st_mode == source.stat().st_mode
    monkeypatch.setattr(dl.shutil, "copy", refused)
    with pytest.raises(OSError, match="refused"):
        dl.copy_atomic(source, no_links / "failed.osm.pbf")
    assert [p.name for p in no_links.iterdir()] == ["copy.osm.pbf"]


def test_geocode_takes_the_network_options(monkeypatch):
    """geocode and get_data_by_geocoding waited without limit for the geocoding service
    and took no headers, timeout or opener. The request now waits 60 s by default, as
    downloads do, and the three options reach it."""
    import io
    import json

    import pyrosm

    reply = json.dumps(
        [{"display_name": "Helsinki", "boundingbox": ["60.1", "60.2", "24.9", "25.0"]}]
    ).encode()
    sent = []

    class Opener:
        def open(self, request, timeout=None):
            sent.append((request, timeout))
            return io.BytesIO(reply)

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, context=None, timeout=None: Opener().open(request, timeout),
    )
    pyrosm.geocode("Helsinki")
    pyrosm.geocode("Helsinki", headers={"User-Agent": "app/1", "X-Key": "k"}, timeout=5)
    pyrosm.geocode("Helsinki", user_agent="me/2", headers={"user-agent": "app/1"})
    pyrosm.geocode("Helsinki", timeout=7, opener=Opener())
    agents = [request.get_header("User-agent") for request, _ in sent]
    assert agents[0].startswith("pyrosm") and agents[1:] == ["app/1", "me/2", agents[0]]
    assert sent[1][0].get_header("X-key") == "k"
    assert [timeout for _, timeout in sent] == [60, 5, 60, 7]


@pytest.mark.parametrize("wrapper", ["get_data_by_bbox", "get_data_by_geocoding"])
def test_area_wrappers_get_the_data_as_get_data_by_area(wrapper, tmp_path, monkeypatch):
    """get_data_by_bbox and get_data_by_geocoding chose only among Geofabrik extracts,
    took none of the options of get_data_by_area and stopped at the first failed
    download. Both now get the data as get_data_by_area does: they search every
    provider with its options, choose again without a failed extract and crop with the
    workers given."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    import pyrosm
    import pyrosm.data.extract_index as ei
    import pyrosm.data.geocoding as gc
    import pyrosm.pbf_export
    from pyrosm.exceptions import DownloadError

    area = [24.93, 60.16, 24.96, 60.18]
    urls = ["https://b/Helsinki/Helsinki.osm.pbf", "https://m/grid/N60E024.osm.pbf"]
    found = gpd.GeoDataFrame(
        {
            "provider": ["BBBike", "Movisda"],
            "id": ["Helsinki", "N60E024"],
            "name": ["Helsinki", "N60E024"],
            "url": urls,
            "bytes": pd.array([50, 66], dtype="Int64"),
            "contains": [True, True],
        },
        geometry=[box(*area)] * 2,
        crs="EPSG:4326",
    )
    searched, geocoded, downloads, crops = [], [], [], []
    monkeypatch.setattr(ei, "find_extracts", lambda geom, **k: searched.append(k) or found)
    monkeypatch.setattr(gc, "geocode", lambda query, **k: geocoded.append(k) or box(*area))
    helsinki = pyrosm.get_data("helsinki_pbf")

    def download(url, filename, update, directory, **net):
        downloads.append((url, net))
        if url == urls[0]:
            raise DownloadError("unavailable", url=url, status=503, attempts=3)
        return helsinki

    monkeypatch.setattr("pyrosm.utils.download.download", download)
    crop_pbf = pyrosm.pbf_export.crop_pbf
    monkeypatch.setattr(
        pyrosm.pbf_export, "crop_pbf", lambda *a, **k: crops.append(k) or crop_pbf(*a, **k)
    )
    net = {"headers": {"User-Agent": "t"}, "timeout": 30, "opener": object()}
    stops = box(24.94, 60.165, 24.95, 60.17)
    place = area if wrapper == "get_data_by_bbox" else "Helsinki"
    with pytest.warns(UserWarning, match="choosing again without it"):
        out = getattr(pyrosm, wrapper)(
            place,
            directory=tmp_path,
            progress=False,
            strategy="smallest_total",
            must_cover=stops,
            workers=2,
            **net,
        )
    options = [{k: s[k] for k in (*net, "must_cover", "contains_only")} for s in searched]
    assert options == [{**net, "must_cover": stops, "contains_only": False}]
    assert downloads == [(url, {**net, "progress": False}) for url in urls]
    assert [c["workers"] for c in crops] == [2]
    if wrapper == "get_data_by_bbox":
        assert "bounding_box" in crops[0]
        assert Path(out) == tmp_path / "bbox_24.93_60.16_24.96_60.18.osm.pbf"
    else:
        assert geocoded == [{"base_url": gc._NOMINATIM_URL, "user_agent": None, **net}]
        assert "polygon" in crops[0]
        assert Path(out) == tmp_path / "area_helsinki.osm.pbf"


def test_update_refreshes_the_indexes_with_a_fallback(tmp_path, monkeypatch):
    """find_extracts(update=True) fetched Geofabrik's index without a timeout, retry or
    fallback, and did not keep it, and BBBike's list of cities was never refreshed. Both
    are now fetched with the network options, kept next to the downloads, and replaced by
    the kept copy, or else pyrosm's own, with a warning when they cannot be fetched."""
    import io
    import json
    from urllib.error import URLError

    from shapely.geometry import Point, box, mapping

    import pyrosm.data.extract_index as ei
    import pyrosm.utils.download as dl

    bbbike = "https://download.bbbike.org/osm/bbbike"
    finland = "https://download.geofabrik.de/europe/finland-latest.osm.pbf"
    feature = {
        "type": "Feature",
        "geometry": mapping(box(20, 59, 32, 71)),
        "properties": {"id": "finland-new", "name": "Finland", "urls": {"pbf": finland}},
    }
    cities = [c for c in ei._bbbike_extracts(None)["id"] if c != "Basel"]
    poly = "Newtown\n1\n 24 60\n 25 60\n 25 61\n 24 61\nEND\nEND\n"
    pages = {
        "https://download.geofabrik.de/index-v1.json": json.dumps(
            {"type": "FeatureCollection", "features": [feature]}
        ),
        bbbike + "/": "".join(
            '<a href="%s/">' % c for c in cities + ["Newtown", "Broken"]
        ),
        bbbike + "/Newtown/Newtown.poly": poly,
        bbbike + "/Broken/Broken.poly": "not a boundary",
    }
    timeouts, online = [], [True]

    class Response(io.BytesIO):
        headers = {}

    def fake(url, method="GET", headers=None, timeout=None, opener=None):
        timeouts.append(timeout)
        if not online[0]:
            raise URLError("down")
        return Response(pages[url].encode())

    monkeypatch.setattr(dl, "open_url", fake)
    net = dl._Net(None, 5, None)
    with pytest.warns(UserWarning, match="leaving Broken out"):
        found = [
            ei._geofabrik_extracts(tmp_path, True, net),
            ei._bbbike_extracts(tmp_path, True, net),
        ]
    assert set(timeouts) == {5}
    online[0] = False
    with pytest.warns(UserWarning, match="Could not refresh|leaving Broken out"):
        kept = [
            ei._geofabrik_extracts(tmp_path, True, net),
            ei._bbbike_extracts(tmp_path, True, net),
        ]
    for geofabrik, cities_found in (found, kept):
        assert list(zip(geofabrik["id"], geofabrik["url"])) == [("finland-new", finland)]
        ids = set(cities_found["id"])
        assert "Newtown" in ids and not {"Basel", "Broken"} & ids
        newtown = cities_found[cities_found["id"] == "Newtown"].iloc[0]
        assert newtown.url == bbbike + "/Newtown/Newtown.osm.pbf"
        assert newtown.geometry.covers(Point(24.5, 60.5))
    with pytest.warns(UserWarning, match="using pyrosm's copy") as caught:
        geofabrik = ei._geofabrik_extracts(tmp_path / "other", True, net)
        cities_found = ei._bbbike_extracts(tmp_path / "other", True, net)
    assert len(caught) == 2 and len(geofabrik) > 500
    assert cities_found is ei._bbbike_extracts(None)
    # A list with no new city.
    online[0] = True
    pages[bbbike + "/"] = "".join('<a href="%s/">' % c for c in cities)
    assert set(ei._bbbike_extracts(tmp_path / "third", True, net)["id"]) == set(cities)


def test_recipe_force_downloads_again_and_geocodes_with_its_options(
    tmp_path, monkeypatch
):
    """pyrosm run --force reran the extract stage on the extract an earlier run had
    downloaded, and a recipe's place was geocoded without the headers and timeout of its
    extract section. --force now downloads again, and the geocoding takes those options."""
    import functools

    from shapely.geometry import box

    import pyrosm.data.extract_index as ei
    import pyrosm.data.geocoding as gc
    from pyrosm import recipes

    seen = []

    class Stop(Exception):
        pass

    @functools.wraps(ei.get_data_by_area)
    def get_data_by_area(area, output_path=None, **keywords):
        seen.append(keywords)
        raise Stop()

    monkeypatch.setattr(ei, "get_data_by_area", get_data_by_area)
    monkeypatch.setattr(
        gc, "geocode", lambda place, **net: seen.append(net) or box(24.9, 60.1, 25, 60.2)
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "recipe: pyrosm\n"
        "extract: {area: {place: Kallio}, headers: {User-Agent: app/1}, timeout: 20}\n"
        "outputs: {extract: a.osm.pbf}\n"
    )
    for force in (False, True):
        with pytest.raises(Stop):
            recipes.run(recipe, force=force)
    net = {"headers": {"User-Agent": "app/1"}, "timeout": 20}
    assert seen == [net, net, net, dict(net, update=True)]


def test_get_data_downloads_over_https():
    """get_data downloaded every Geofabrik extract over plain HTTP, so anything on the way
    could change the file. Its Geofabrik and BBBike downloads use https."""
    from pyrosm.data import _resolve_dataset

    for name in ("finland", "usa/georgia", "greater_london", "helsinki"):
        assert _resolve_dataset(name)[1]["url"].startswith("https://download.")


def test_clear_downloads_removes_the_file_given_and_partial_files(tmp_path, monkeypatch):
    """clear_downloads(path) removed the file of that name in the download folder and kept
    the file at the path, and the partial files of interrupted downloads were never
    removed. A path removes exactly that file, a bare name the downloaded one, each with
    its partial files and crop record, and only PBF files can be named."""
    import pyrosm.utils.download as dl
    from pyrosm import OSM

    downloads, own = tmp_path / "pyrosm", tmp_path / "my_data"
    monkeypatch.setattr(dl, "download_dir", lambda: downloads)
    files = [
        downloads / "monaco-latest.osm.pbf",
        own / "monaco-latest.osm.pbf",
        downloads / "monaco-latest.osm.pbf.k2j3h4.part",
        downloads / "bbox_1_2_3_4.osm.pbf",
        downloads / "bbox_1_2_3_4.osm.pbf.sources",
        downloads / "estonia-latest.osm.pbf.x8y9z0.part",
        downloads / "monaco-latest.osm.pbf.copy.osm.pbf.ab12cd34.part",
        downloads / "geofabrik" / "index-v1.json",
    ]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"x")
    assert OSM.clear_downloads(own / "monaco-latest.osm.pbf") == 1
    assert files[0].exists() and not files[1].exists()
    assert OSM.clear_downloads("monaco-latest.osm.pbf") == 2
    assert files[-2].exists()
    assert OSM.clear_downloads() == 3
    left = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    assert left == ["pyrosm/geofabrik/index-v1.json"]
    with pytest.raises(ValueError, match="removes .osm.pbf and .osh.pbf files only"):
        OSM.clear_downloads(files[-1])


def test_dataset_names_take_every_spelling_and_get_suggestions():
    """An unknown dataset name listed about 800 names in no order, and a region-qualified
    name did not take the spellings a bare name takes. The error names the closest
    datasets, and both forms take spaces, dashes and underscores."""
    from pyrosm.data import _resolve_dataset

    for name in ("usa/new york", "USA / New-York", "usa/new_york"):
        assert _resolve_dataset(name)[1]["url"].endswith("/us/new-york-latest.osm.pbf")
    assert _resolve_dataset("Rio-de-Janeiro")[1]["url"].endswith("/RiodeJaneiro.osm.pbf")
    for name, hint in (
        ("finlnd", "'finland'"),
        ("test_pb", "'test_pbf'"),
        ("asia/georgia", "'europe/georgia' or 'usa/georgia'"),
    ):
        with pytest.raises(ValueError, match="Did you mean") as info:
            _resolve_dataset(name)
        assert hint in str(info.value) and len(str(info.value)) < 200


def test_get_data_knows_every_bbbike_city():
    """get_data lacked three cities (Damaskus, LosAngeles, Minsk) that BBBike lists and
    get_data_by_area finds. Every city of pyrosm's BBBike index is a get_data dataset."""
    from pyrosm.data import _resolve_dataset
    from pyrosm.data.extract_index import _bbbike_extracts

    cities = _bbbike_extracts(None)
    for city, url in zip(cities["id"], cities["url"]):
        name = "NewYorkCity" if city == "NewYork" else city
        assert _resolve_dataset("cities/" + name)[1]["url"] == url


@pytest.mark.parametrize(
    "write", ["to_pbf", "to_pbf in parallel", "write_pbf", "write_pbf subset"]
)
def test_written_files_keep_the_completed_relations(write, tmp_path):
    """With complete_relations=True a read with a bounding box fetches the member ways of
    the relations reaching into the box, also those outside it, but to_pbf and write_pbf
    left them out, so the written file read back with other relation geometries. Both
    write those member ways and their nodes now."""
    from pyrosm import OSM, get_data

    box = [24.94758, 60.16839, 24.94787, 60.16888]
    osm = OSM(get_data("helsinki_pbf"), bounding_box=box, complete_relations=True)
    buildings = osm.get_buildings()
    out = str(tmp_path / "out.osm.pbf")
    if write.startswith("to_pbf"):
        osm.to_pbf(out, workers=2 if write.endswith("parallel") else 1)
    else:
        osm.write_pbf(buildings, out, subset_only=write.endswith("subset"))

    def relations(gdf):
        return gdf[gdf["osm_type"] == "relation"].set_index("id").geometry

    read, again = relations(buildings), relations(OSM(out).get_buildings())
    assert 1689811 in read.index
    areas = [g.area for g in read]
    assert [g.area for g in again.loc[read.index]] == pytest.approx(areas, abs=1e-12)


def test_crop_completes_every_version_of_a_relation(tmp_path):
    """A crop keeps a relation by its id, so a file with several records of one relation
    (its versions) gets every record written; the crop completes the way members of all
    of them, also of a record that does not reach into the area."""
    import numpy as np

    from pyrosm.pbf_export import _iter_decoded_blocks, crop_pbf, write_pbf_from_records

    corners = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
    squares = (0, 0.02, 0.04)
    nodes = {
        "id": np.arange(1, 13, dtype=np.int64),
        "lon": np.array([24.90 + dx + x for dx in squares for x, _ in corners]),
        "lat": np.array([60.10 + y for _ in squares for _, y in corners]),
        "version": np.ones(12, dtype=np.int64),
        "timestamp": np.full(12, 1600000000, dtype=np.int64),
        "changeset": np.zeros(12, dtype=np.int64),
        "tags": [None] * 12,
    }
    ways = [
        {"id": 10 + i, "refs": [s, s + 1, s + 2, s + 3, s], "version": 1, "tags": {}}
        for i, s in enumerate((1, 5, 9))
    ]
    relations = [
        {"id": 20, "members": members, "version": version, "timestamp": 1600000000 + version,
         "changeset": 0, "tags": {"type": "multipolygon", "building": "yes"}}
        for version, members in (
            (1, [("way", 10, "outer"), ("way", 11, "outer")]),
            (2, [("way", 12, "outer")]),
        )
    ]
    source = write_pbf_from_records(
        nodes, ways, relations, str(tmp_path / "versions.osm.pbf"), (24.9, 60.1, 24.95, 60.11)
    )
    out = crop_pbf(
        source,
        str(tmp_path / "crop.osm.pbf"),
        bounding_box=[24.899, 60.099, 24.911, 60.111],
        complete_relations=True,
        progress=False,
    )
    kept_ways = np.concatenate(
        [ways["id"] for _, _, ways, _ in _iter_decoded_blocks(out) if ways is not None]
    )
    assert sorted(kept_ways.tolist()) == [10, 11, 12]


def test_bounding_box_write_keeps_the_relations_of_the_box(tmp_path):
    """write_pbf of a read with a bounding box wrote every relation of the whole file,
    most of them without a member in the written file. It writes the relations with a
    member in the file, as to_pbf does."""
    from pyrosm import OSM, get_data
    from pyrosm.pbf_writer import _relations_with_members

    fp = get_data("helsinki_pbf")
    osm = OSM(fp, bounding_box=[24.94, 60.165, 24.945, 60.17])
    with pytest.warns(UserWarning, match="extend beyond the bounding box"):
        buildings = osm.get_buildings()
    written = osm.write_pbf(buildings, str(tmp_path / "written.osm.pbf"))
    cropped = osm.to_pbf(str(tmp_path / "cropped.osm.pbf"))

    def relation_ids(path):
        read = OSM(path)
        read._read_pbf()
        return set(read._relations["id"].tolist())

    assert relation_ids(written) == relation_ids(cropped) < relation_ids(fp)
    assert _relations_with_members({}, [], []) == set()
    text = {"id": [1, 2], "members": [{"member_id": [5], "member_type": ["way"]}] * 2}
    assert _relations_with_members(text, [5], [7]) == set()
    assert _relations_with_members(text, [], [5]) == {1, 2}


@pytest.mark.parametrize(
    "form",
    ["list", "tuple", "array", "polygon", "GeoDataFrame", "GeoSeries", "projected", "no CRS"],
)
def test_bounding_box_forms(form):
    """Each function took its own bounding-box forms: OSM() refused a tuple, an array
    and a GeoDataFrame, crop_pbf failed on them with unrelated errors, and
    get_data_by_bbox read a projected GeoDataFrame's metres as degrees. One check takes
    every form, a GeoDataFrame or GeoSeries in lon/lat."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    from pyrosm.utils import validate_bounding_box

    bounds = [24.94, 60.165, 24.945, 60.17]
    gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=4326)
    given = {
        "list": bounds,
        "tuple": tuple(bounds),
        "array": np.array(bounds),
        "polygon": box(*bounds),
        "GeoDataFrame": gdf,
        "GeoSeries": gdf.geometry,
        "projected": gdf.to_crs(3067),
        "no CRS": gdf.set_crs(None, allow_override=True),
    }[form]
    checked = validate_bounding_box(given)
    got = checked if isinstance(checked, list) else list(checked.bounds)
    assert got == pytest.approx(bounds, abs=1e-6)


def test_every_function_takes_a_geodataframe_box(tmp_path, monkeypatch):
    """OSM(), the out-of-core readers, crop_pbf, merge_pbf and get_data_by_bbox all take
    a GeoDataFrame as the box, and read or crop what its bounds give."""
    import geopandas as gpd
    from shapely.geometry import box

    import pyrosm.data.extract_index as ei
    from pyrosm import OSM, get_data, get_data_by_bbox, merge_pbf
    from pyrosm.pbf_export import crop_pbf

    fp = get_data("helsinki_pbf")
    bounds = [24.94, 60.165, 24.945, 60.17]
    gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=4326)
    ids = sorted(OSM(fp, bounding_box=bounds).get_buildings()["id"])
    for engine in ("in_memory", "out_of_core"):
        read = OSM(fp, bounding_box=gdf, engine=engine).get_buildings()
        assert sorted(read["id"]) == ids
    crop = crop_pbf(fp, str(tmp_path / "crop.osm.pbf"), bounding_box=gdf, progress=False)
    merged = merge_pbf([fp], str(tmp_path / "merged.osm.pbf"), polygon=gdf, progress=False)
    for written in (crop, merged):
        assert set(ids) <= set(OSM(written).get_buildings()["id"])
    searched = []
    monkeypatch.setattr(ei, "find_extracts", lambda geom, **k: searched.append(geom) or [])
    with pytest.raises(Exception):
        get_data_by_bbox(gdf, download=False)
    assert list(searched[0].bounds) == pytest.approx(bounds)


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize(
    "bounding_box, problem",
    [
        ([24.95, 60.16, 24.93, 60.17], "minx < maxx"),
        ([385000, 6672000, 386000, 6673000], "lon/lat range"),
        ((24.93, 60.16, 24.95), "should be four numbers"),
        ([10.0, 50.0, 10.1, 50.1], "No node of"),
    ],
)
def test_crops_refuse_unusable_boxes_and_warn_when_empty(
    bounding_box, problem, merge, tmp_path
):
    """crop_pbf and merge_pbf wrote an empty file, without a word, for an inverted box,
    one in metres or one outside the data, and failed with unrelated errors on others.
    They refuse a box OSM() refuses and warn when they keep no node."""
    from pyrosm import get_data, merge_pbf
    from pyrosm.pbf_export import crop_pbf

    fp, out = get_data("helsinki_pbf"), str(tmp_path / "out.osm.pbf")

    def write():
        if merge:
            return merge_pbf([fp], out, bounding_box=bounding_box, progress=False)
        return crop_pbf(fp, out, bounding_box=bounding_box, progress=False)

    if problem == "No node of":
        with pytest.warns(UserWarning, match="No node of Helsinki.osm.pbf lies in"):
            write()
    else:
        with pytest.raises(ValueError, match=problem):
            write()


def test_merge_of_files_without_nodes_warns(tmp_path):
    """A merge without a box warns, as a crop does, when the file it writes holds no
    node."""
    import numpy as np

    from pyrosm import merge_pbf
    from pyrosm.pbf_export import write_pbf_from_records

    columns = ("id", "lat", "lon", "version", "timestamp", "changeset")
    nodes = {name: np.empty(0, dtype=np.int64) for name in columns}
    nodes.update(lat=np.empty(0), lon=np.empty(0), tags=[])
    empty = write_pbf_from_records(nodes, [], [], str(tmp_path / "empty.osm.pbf"), (0, 0, 1, 1))
    with pytest.warns(UserWarning, match="empty.osm.pbf hold no node"):
        merge_pbf([empty], str(tmp_path / "merged.osm.pbf"), progress=False)


def test_write_pbf_writes_new_elements_in_order(tmp_path):
    """write_pbf wrote a new way after the ways of the source, so the file was not sorted
    by type then id and merge_pbf refused it. It writes every element sorted by id, new
    ones included."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from pyrosm import OSM, get_data, merge_pbf
    from pyrosm.pbf_export import _iter_decoded_blocks

    osm = OSM(get_data("test_pbf"))
    footway = gpd.GeoDataFrame(
        {"highway": ["footway"]},
        geometry=[LineString([(26.94, 60.52), (26.945, 60.525)])],
        crs=4326,
    )
    written = osm.write_pbf([footway], str(tmp_path / "written.osm.pbf"))
    merged = merge_pbf([written], str(tmp_path / "merged.osm.pbf"), progress=False)
    way_ids = [
        i
        for _, _, ways, _ in _iter_decoded_blocks(merged)
        if ways is not None
        for i in ways["id"].tolist()
    ]
    assert way_ids[0] < 0 and way_ids == sorted(way_ids)


def _metadata_source(path, mixed=False):
    """A PBF with a building relation (id 20) of two square ways: way 10 (changeset 88)
    on nodes 1-4 and way 11 (changeset 99) on nodes 5-8, 0.02 degrees east; the nodes
    have changeset 77, or when ``mixed`` nodes 5-8 have no metadata."""
    import numpy as np

    from pyrosm.pbf_export import write_pbf_from_records

    corners = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
    nodes = {
        "id": np.arange(1, 9, dtype=np.int64),
        "lon": np.array([24.90 + dx + x for dx in (0, 0.02) for x, _ in corners]),
        "lat": np.array([60.10 + y for _ in (0, 0.02) for _, y in corners]),
        "version": np.full(8, 3, dtype=np.int64),
        "timestamp": np.full(8, 1600000000, dtype=np.int64),
        "changeset": np.full(8, 77, dtype=np.int64),
        "tags": [None] * 8,
    }
    if mixed:
        for name in ("version", "timestamp", "changeset"):
            nodes[name][4:] = 0
    ways = [
        {"id": 10 + i, "refs": [s, s + 1, s + 2, s + 3, s], "version": 2,
         "timestamp": 1600000000, "changeset": changeset, "tags": {}}
        for i, (s, changeset) in enumerate(((1, 88), (5, 99)))
    ]
    relation = {
        "id": 20,
        "members": [("way", 10, "outer"), ("way", 11, "outer")],
        "version": 1,
        "timestamp": 1600000000,
        "changeset": 66,
        "tags": {"type": "multipolygon", "building": "yes"},
    }
    return write_pbf_from_records(
        nodes, ways, [relation], str(path), (24.9, 60.1, 24.93, 60.11)
    )


@pytest.mark.parametrize(
    "read", ["whole", "no node metadata", "box, complete relations", "mixed nodes"]
)
def test_write_pbf_writes_the_metadata_it_read(read, tmp_path):
    """write_pbf wrote every way with changeset 0 and, for nodes read with
    keep_metadata=False, invented version 1, timestamp 1970 and changeset 0. It writes
    the metadata it read, also for the member ways a read with complete_relations adds,
    and none where none was read, node by node."""
    from pyrosm import OSM
    from pyrosm.pbfreader import iter_decoded_blocks

    source = _metadata_source(tmp_path / "source.osm.pbf", mixed=read == "mixed nodes")
    options = {
        "whole": {},
        "no node metadata": {"keep_metadata": False},
        "box, complete relations": {
            "bounding_box": [24.899, 60.099, 24.911, 60.111],
            "complete_relations": True,
        },
        "mixed nodes": {},
    }[read]
    osm = OSM(source, engine="in_memory", **options)
    out = osm.write_pbf(osm.get_buildings(), str(tmp_path / "out.osm.pbf"))
    nodes, ways = [], {}
    for _, _, block_nodes, block_ways, _ in iter_decoded_blocks(out):
        if block_nodes is not None:
            nodes.append(block_nodes)
        if block_ways is not None:
            ways.update(zip(block_ways["id"].tolist(), block_ways["changeset"].tolist()))
    assert ways == {10: 88, 11: 99}
    with_metadata = [
        (i, c)
        for block in nodes
        if len(block["changeset"])
        for i, c in zip(block["id"].tolist(), block["changeset"].tolist())
    ]
    expected = {"no node metadata": [], "mixed nodes": [(i, 77) for i in range(1, 5)]}
    assert with_metadata == expected.get(read, [(i, 77) for i in range(1, 9)])


def test_write_pbf_of_a_history_read_keeps_the_way_changesets(tmp_path):
    """A write of a history read gives each way the changeset of the version read."""
    import warnings

    import numpy as np

    from pyrosm import OSM, get_data
    from pyrosm.pbfreader import iter_decoded_blocks

    def ways_of(path):
        return [w for _, _, _, w, _ in iter_decoded_blocks(path) if w is not None]

    fp = get_data("helsinki_test_history_pbf")
    latest = {}
    for ways in ways_of(fp):
        for i, version, changeset in zip(
            ways["id"].tolist(), ways["version"].tolist(), ways["changeset"].tolist()
        ):
            if version >= latest.get(i, (0, 0))[0]:
                latest[i] = (version, changeset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        osm = OSM(fp, engine="in_memory")
        out = osm.write_pbf(osm.get_buildings(), str(tmp_path / "out.osm.pbf"))
    written = [
        (i, c)
        for ways in ways_of(out)
        for i, c in zip(ways["id"].tolist(), ways["changeset"].tolist())
    ]
    assert written and np.all([latest[i][1] == c for i, c in written])


@pytest.mark.parametrize("write", ["crop", "crop in parallel", "write_pbf"])
def test_interrupted_writes_leave_the_earlier_file(write, tmp_path):
    """Crops and write_pbf wrote straight into output_path, so an interrupted call left a
    truncated file that read as a smaller valid one, and a failed crop without
    output_path left its temporary file. They write next to it and replace it when the
    file is complete."""
    import os

    from pyrosm import OSM, get_data
    from pyrosm.pbf_export import crop_pbf

    fp, box = get_data("helsinki_pbf"), [24.93, 60.16, 24.95, 60.18]
    out = tmp_path / "out.osm.pbf"
    out.write_bytes(b"earlier")

    def stop(done, total):
        if done > total * 0.97:
            raise KeyboardInterrupt

    def run(progress):
        if write == "write_pbf":
            osm = OSM(fp, engine="in_memory", progress=False)
            osm.write_pbf(osm.get_buildings(), str(out), progress=progress)
        else:
            workers = 2 if write.endswith("parallel") else 1
            crop_pbf(fp, str(out), bounding_box=box, workers=workers, progress=progress)

    out.chmod(0o640)
    with pytest.raises(KeyboardInterrupt):
        run(stop)
    assert [p.name for p in tmp_path.iterdir()] == ["out.osm.pbf"]
    assert out.read_bytes() == b"earlier"
    # A complete write replaces the file and keeps its permissions (POSIX), read-only ones
    # too.
    for mode in (0o640, 0o444) if os.name != "nt" else (0o666,):
        out.chmod(mode)
        run(False)
        assert out.stat().st_size > 1000 and out.stat().st_mode & 0o777 == mode
    out.chmod(0o644)


def test_partial_file_names_skip_taken_ones(tmp_path, monkeypatch):
    """A partial file is created under a name nothing else holds."""
    import pyrosm.utils.download as dl

    names = iter(["taken", "fresh"])
    monkeypatch.setattr(dl.secrets, "token_hex", lambda n: next(names))
    (tmp_path / "x.osm.pbf.taken.part").write_bytes(b"other")
    assert dl._new_partial(tmp_path / "x.osm.pbf").name == "x.osm.pbf.fresh.part"
    assert (tmp_path / "x.osm.pbf.taken.part").read_bytes() == b"other"


@pytest.mark.parametrize("simplify", [True, False])
def test_networkx_graphs_carry_the_osmnx_booleans(simplify, tmp_path):
    """NetworkX graphs kept oneway as the OSM text and had no reversed, so OSMnx could
    not load a saved graph. With osmnx_compatible they carry oneway and reversed as
    booleans (a list where the ways of a simplified edge differ), the OSM text as
    oneway_tag."""
    osmnx = pytest.importorskip("osmnx")
    from pyrosm import OSM, get_data

    graph = OSM(get_data("helsinki_pbf"), progress=False).get_graph(
        "driving", graph_type="networkx", simplify=simplify
    )

    def booleans(value):
        values = value if isinstance(value, list) else [value]
        return all(type(v) is bool for v in values)

    edges = list(graph.edges(data=True))
    assert all(booleans(d["oneway"]) and booleans(d["reversed"]) for *_, d in edges)
    assert {d["oneway"] for *_, d in edges} == {True, False}
    assert all(graph.has_edge(v, u) for u, v, d in edges if d["oneway"] is False)
    assert "yes" in {d.get("oneway_tag") for *_, d in edges}
    osmnx.save_graphml(graph, tmp_path / "graph.graphml")
    loaded = osmnx.load_graphml(tmp_path / "graph.graphml")
    assert loaded.number_of_edges() == graph.number_of_edges()


def test_node_attrs_include_finds_node_tags():
    """simplify_kwargs node_attrs_include looked only at node columns, and the network's
    node tags are in their tags, so naming a tag kept no node. It finds node tags too,
    in both engines, and warns about a name no node has."""
    import json

    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine="in_memory", progress=False)
    nodes, edges = osm.get_network("driving", nodes=True)
    signals = {
        i
        for i, tags in zip(nodes["id"], nodes["tags"])
        if isinstance(tags, dict) and tags.get("highway") == "traffic_signals"
    }
    full = osm.to_graph(nodes, edges, graph_type="networkx", simplify=False)
    signals &= set(full)
    keep = {"node_attrs_include": ["highway"]}
    default = osm.to_graph(nodes, edges, graph_type="networkx")
    graph = osm.to_graph(nodes, edges, graph_type="networkx", simplify_kwargs=keep)
    assert signals - set(default) and signals <= set(graph)
    engine = OSM(get_data("helsinki_pbf"), progress=False)
    engine_graph = engine.get_graph("driving", graph_type="networkx", simplify_kwargs=keep)
    assert set(engine_graph) == set(graph)
    # Tags as JSON text, as a frame read back from a file holds them.
    text = nodes.assign(
        tags=[json.dumps(t) if isinstance(t, dict) else t for t in nodes["tags"]]
    )
    text_graph = osm.to_graph(text, edges, graph_type="networkx", simplify_kwargs=keep)
    assert set(text_graph) == set(graph)
    # A column of the name that holds the values of some nodes only.
    partial = nodes.assign(highway=None)
    partial_graph = osm.to_graph(partial, edges, graph_type="networkx", simplify_kwargs=keep)
    assert set(partial_graph) == set(graph)
    with pytest.warns(UserWarning, match="named 'no_such_tag'"):
        osm.to_graph(
            nodes,
            edges,
            graph_type="networkx",
            simplify_kwargs={"node_attrs_include": ["no_such_tag"]},
        )


@pytest.mark.parametrize("network_type", ["driving", "cycling"])
def test_default_graphs_are_fully_simplified(network_type):
    """A graph was simplified first and its largest connected part kept afterwards, so a
    node whose side street the second step removed was left between two edges of one
    street. Graphs are fully simplified also where that step removes edges, in both
    readers."""
    from pyrosm import OSM, get_data
    from pyrosm.graph_simplify import simplify_graph
    from pyrosm.graphs import graph_tables

    osm = OSM(get_data("helsinki_pbf"), engine="in_memory", progress=False)
    nodes, edges = osm.get_network(network_type, nodes=True)
    graph_nodes, graph_edges = graph_tables(nodes, edges, network_type=network_type)
    assert len(simplify_graph(graph_nodes, graph_edges)[0]) == len(graph_nodes)
    read = OSM(get_data("helsinki_pbf"), progress=False)
    graph = read.get_graph(network_type, graph_type="networkx")
    assert set(graph) == set(graph_nodes["id"])


@pytest.mark.parametrize("network_type", ["driving", "walking"])
def test_unsimplified_edges_run_from_u_to_v(network_type):
    """In a graph built with simplify=False, an edge that runs against the node order of
    its way kept the way's line, which started at its v node. Every edge's line runs
    from its u node to its v node."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine="in_memory", progress=False)
    nodes, edges = osm.get_network(network_type, nodes=True)
    graph = osm.to_graph(nodes, edges, graph_type="networkx", simplify=False)

    def at(node):
        return pytest.approx((graph.nodes[node]["x"], graph.nodes[node]["y"]), abs=1e-7)

    for u, v, data in graph.edges(data=True):
        line = data["geometry"].coords
        assert line[0] == at(u) and line[-1] == at(v)


@pytest.mark.parametrize(
    "call, name",
    [
        (lambda osm: osm.get_buildings(extra_attributes="name"), "extra_attributes"),
        (lambda osm: osm.get_buildings(tags_to_keep="name"), "tags_to_keep"),
        (
            lambda osm: osm.get_data_by_custom_criteria(
                {"highway": True}, tags_as_columns="name"
            ),
            "tags_as_columns",
        ),
        (
            lambda osm: osm.get_data_by_custom_criteria(
                {"highway": True}, tags_to_keep=["name", 1]
            ),
            "tags_to_keep",
        ),
        (lambda osm: osm.get_graph("driving", tags_to_keep="name"), "tags_to_keep"),
    ],
)
def test_tag_list_errors_name_the_parameter_given(call, name):
    """A tag list that is not a list of strings was refused as 'tags_as_columns', a
    parameter most of the methods do not have; the error names the parameter given."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"), progress=False)
    with pytest.raises(ValueError, match=f"'{name}' should be"):
        call(osm)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("get_pois", {}),
        ("get_network", {}),
        (
            "get_data_by_custom_criteria",
            {
                "custom_filter": {"amenity": True},
                "keep_ways": False,
                "keep_relations": False,
            },
        ),
    ],
)
def test_both_engines_give_visible_and_the_metadata_the_same_dtypes(method, kwargs):
    """Out-of-core node rows had int64 ``version``, ``timestamp`` and ``changeset`` where
    the in-memory reader has int32, uint32 and int32, the in-memory ``visible`` was an
    object column, and the ways of reads without a ``visible`` column held it in the JSON
    ``tags``. Both engines give the same columns with the same dtypes, and ``visible`` is
    a bool column of every read, never a key of ``tags``."""
    from pyrosm import OSM, get_data

    frames = [
        getattr(OSM(get_data("helsinki_pbf"), engine=engine, progress=False), method)(
            **kwargs
        )
        for engine in ("in_memory", "out_of_core")
    ]
    dtypes = [list(frame.dtypes.astype(str).items()) for frame in frames]
    assert dtypes[0] == dtypes[1]
    for frame in frames:
        assert frame["visible"].dtype == bool
        assert not frame["tags"].fillna("").str.contains('"visible"').any()


def test_box_reads_leave_out_tag_columns_of_rows_outside_the_box():
    """With a box and ``complete_relations``, the in-memory reader read the boundary
    Honkasuo, which lies outside the box, left it out by the box, and kept its
    ``start_date`` column with no value. Both engines give the same columns, none of them
    empty."""
    from pyrosm import OSM, get_data

    box = [24.85, 60.15, 25.05, 60.25]
    frames = [
        OSM(
            get_data("helsinki_region_pbf"),
            engine=engine,
            bounding_box=box,
            complete_relations=True,
            progress=False,
        ).get_boundaries()
        for engine in ("in_memory", "out_of_core")
    ]
    assert list(frames[0].columns) == list(frames[1].columns)
    assert not [c for c in frames[0].columns if frames[0][c].isna().all()]


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_get_buildings_keeps_building_nodes_on_request(engine):
    """``get_buildings`` returned no building mapped as a single node. With
    ``keep_nodes=True`` it returns them as Points, without the nodes tagged
    ``building=entrance`` (33 in the sample), and its ways and relations stay those of the
    default read, which still has no nodes."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"), engine=engine, progress=False)
    default = osm.get_buildings()
    buildings = osm.get_buildings(keep_nodes=True)
    points = buildings[buildings["osm_type"] == "node"]
    assert sorted(points["building"]) == ["kiosk", "roof", "yes"]
    assert set(points.geometry.geom_type) == {"Point"}
    assert "node" not in set(default["osm_type"])
    others = buildings[buildings["osm_type"] != "node"]
    assert list(zip(others["osm_type"], others["id"])) == list(
        zip(default["osm_type"], default["id"])
    )
    with pytest.raises(ValueError, match="'keep_nodes' should be boolean"):
        osm.get_buildings(keep_nodes="yes")


def test_building_parts_are_left_out_of_building_nodes():
    """Of the building nodes, those tagged ``building=entrance``, ``window`` or ``no`` are
    left out, a way with such a value is kept, a frame without them is returned as it is,
    and nothing left gives None with a warning."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from pyrosm.layers import without_building_parts

    rows = [
        ("node", 1, "yes", Point(0, 0)),
        ("node", 2, "entrance", Point(0, 1)),
        ("node", 3, "window", Point(0, 2)),
        ("node", 4, "no", Point(0, 3)),
        ("way", 5, "entrance", LineString([(0, 0), (1, 1)])),
    ]
    gdf = gpd.GeoDataFrame(rows, columns=["osm_type", "id", "building", "geometry"])
    assert without_building_parts(gdf)["id"].tolist() == [1, 5]
    buildings = gdf[gdf["id"].isin([1, 5])]
    assert without_building_parts(buildings) is buildings
    with pytest.warns(UserWarning):
        assert without_building_parts(gdf[gdf["id"].isin([2, 3])]) is None
    assert without_building_parts(None) is None
