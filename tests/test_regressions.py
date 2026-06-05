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
