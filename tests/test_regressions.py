"""Regression tests guarding against specific bugs reappearing."""


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


def test_custom_filter_highway_does_not_linestringify_polygons():
    """#144 — a custom_filter that includes 'highway' must not flip unrelated
    closed-way polygons (buildings, etc.) into (Multi)LineStrings. The old code
    keyed the polygon-vs-line decision on whether 'highway' existed as a column
    in the batch, not on the individual way's tags."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("test_pbf"))
    buildings = osm.get_data_by_custom_criteria(
        custom_filter={"building": True}, filter_type="keep"
    )
    assert set(buildings.geometry.geom_type.unique()) == {"Polygon"}
    assert len(buildings) == 2208

    # Adding 'highway' to the filter must not change the building geometries.
    combined = osm.get_data_by_custom_criteria(
        custom_filter={"building": True, "highway": True}, filter_type="keep"
    )
    building_rows = combined[combined["building"].notna()]
    assert len(building_rows) == 2208
    assert set(building_rows.geometry.geom_type.unique()) == {"Polygon"}


def test_closed_highway_without_area_is_linestring():
    """#144 — in feature extraction, a closed 'highway' way WITHOUT area=yes (e.g.
    a highway=service roundabout) must stay a line, not become a polygon."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": True}, filter_type="keep"
    )
    way = gdf[(gdf["osm_type"] == "way") & (gdf["id"] == 8035241)]
    assert len(way) == 1
    assert way.iloc[0].geometry.geom_type in ("LineString", "MultiLineString")


def test_closed_highway_area_yes_is_polygon():
    """#144 — in feature extraction, a closed 'highway' way tagged area=yes (a
    pedestrian/footway plaza) must be typed as a Polygon."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": True}, filter_type="keep"
    )
    ways = gdf[gdf["osm_type"] == "way"]
    for plaza_id in (4369051, 18379563):
        row = ways[ways["id"] == plaza_id]
        assert len(row) == 1
        assert row.iloc[0]["area"] == "yes"
        assert row.iloc[0].geometry.geom_type == "Polygon"

    # The area=yes plazas as a group come back as polygons.
    area_yes = ways[ways["area"] == "yes"]
    assert (area_yes.geometry.geom_type == "Polygon").all()
    assert len(area_yes) > 0


def test_network_extraction_keeps_areas_as_lines():
    """#144 guard — network extraction must NEVER produce polygons, even for
    highway ways tagged area=yes (plazas). A Polygon is not routable, so the
    parse_network path keeps every closed way linear. get_network('all') does not
    exclude area=yes ways, so the plazas are present and must be lines."""
    from pyrosm import OSM, get_data

    osm = OSM(get_data("helsinki_pbf"))
    edges = osm.get_network(network_type="all")
    geom_types = set(edges.geometry.geom_type.unique())
    assert not any("Polygon" in t for t in geom_types), geom_types

    # The area=yes plazas are present (not excluded by the 'all' filter) and linear.
    for plaza_id in (4369051, 18379563):
        row = edges[edges["id"] == plaza_id]
        assert len(row) >= 1
        assert all("LineString" in t for t in row.geometry.geom_type.unique())


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
