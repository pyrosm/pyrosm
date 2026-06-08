"""Geometry-focused tests: way/relation geometry creation, polygon-vs-line
typing, and polygon/multipolygon ring orientation (cross-checked against
osmium). Consolidated here so geometry-correctness tests live in one place."""
import pytest
from pyrosm import get_data


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


def test_creating_building_geometries(test_pbf):
    from pyrosm import OSM
    from pyrosm.data_manager import get_osm_data
    from pyrosm.geometry import create_way_geometries
    from shapely import Geometry

    osm = OSM(filepath=test_pbf)
    osm._read_pbf()
    custom_filter = {"building": True}
    nodes, ways, relation_ways, relations = get_osm_data(
        None,
        osm._way_records,
        osm._relations,
        osm.conf.tags.building,
        custom_filter,
        filter_type="keep",
    )
    assert isinstance(ways, dict)

    ways, geometries, lengths, from_ids, to_ids = create_way_geometries(
        osm._node_coordinates, ways, parse_network=False
    )
    assert isinstance(geometries, list), f"Type should be list, got {type(geometries)}."
    assert isinstance(geometries[0], Geometry)
    assert len(geometries) == len(ways["id"])


def test_custom_filter_highway_does_not_linestringify_polygons():
    """#144 — a custom_filter that includes 'highway' must not flip unrelated
    closed-way polygons (buildings, etc.) into (Multi)LineStrings. The old code
    keyed the polygon-vs-line decision on whether 'highway' existed as a column
    in the batch, not on the individual way's tags."""
    from pyrosm import OSM

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
    from pyrosm import OSM

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
    from pyrosm import OSM

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
    from pyrosm import OSM

    osm = OSM(get_data("helsinki_pbf"))
    edges = osm.get_network(network_type="all")
    geom_types = set(edges.geometry.geom_type.unique())
    assert not any("Polygon" in t for t in geom_types), geom_types

    # The area=yes plazas are present (not excluded by the 'all' filter) and linear.
    for plaza_id in (4369051, 18379563):
        row = edges[edges["id"] == plaza_id]
        assert len(row) >= 1
        assert all("LineString" in t for t in row.geometry.geom_type.unique())


def test_polygon_ring_orientation_follows_right_hand_rule():
    """#230 — exterior rings must be CCW and holes CW (OGC/GeoJSON right-hand
    rule, matching osmium and QGIS), regardless of the OSM way node order."""
    from pyrosm import OSM

    osm = OSM(get_data("helsinki_pbf"))
    checked = 0
    for meth in ("get_buildings", "get_natural", "get_landuse"):
        gdf = getattr(osm, meth)()
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        for g in polys.geometry:
            parts = [g] if g.geom_type == "Polygon" else list(g.geoms)
            for part in parts:
                # Winding is ill-defined for invalid (e.g. self-touching) rings.
                if not part.is_valid:
                    continue
                assert part.exterior.is_ccw, f"{meth}: exterior ring not CCW"
                for hole in part.interiors:
                    assert not hole.is_ccw, f"{meth}: interior ring (hole) not CW"
                checked += 1
    assert checked > 100


def _osmium_relation_areas(path):
    """Assemble reference multipolygon areas (relation id -> shapely geom) with
    osmium's area handler."""
    import osmium
    from osmium.geom import WKBFactory
    from shapely import wkb

    fab = WKBFactory()
    ref = {}

    class _H(osmium.SimpleHandler):
        def area(self, a):
            if a.from_way():
                return
            try:
                ref[a.orig_id()] = wkb.loads(fab.create_multipolygon(a), hex=True)
            except Exception:
                pass

    _H().apply_file(path, locations=True)
    return ref


def test_relation_polygons_match_osmium_orientation():
    """#230 — pyrosm relation multipolygons match osmium's orientation (CCW
    exteriors) and geometry (within the float32-coordinate tolerance)."""
    pytest.importorskip("osmium")
    from pyrosm import OSM

    fp = get_data("helsinki_pbf")
    ref = _osmium_relation_areas(fp)

    osm = OSM(fp)
    pyr = {}
    for meth in ("get_natural", "get_landuse", "get_buildings", "get_boundaries"):
        gdf = getattr(osm, meth)()
        if gdf is None or "osm_type" not in gdf:
            continue
        sub = gdf[gdf.osm_type == "relation"]
        for _id, g in zip(sub["id"], sub.geometry):
            if g is not None:
                pyr[int(_id)] = g

    common = sorted(set(ref) & set(pyr))
    assert len(common) > 20
    for i in common:
        p = pyr[i]
        parts = [p] if p.geom_type == "Polygon" else list(p.geoms)
        for part in parts:
            if part.is_valid:
                assert part.exterior.is_ccw, f"relation {i}: exterior not CCW"
        err = ref[i].symmetric_difference(p).area / ref[i].area
        assert err < 0.1, f"relation {i} differs from osmium by {err:.3f}"


def test_synthetic_multipolygon_matches_osmium(tmp_path):
    """#230 — a multipolygon with two separate outer rings, an outer ring split
    across two ways, and a hole assembles into the correct 3-part MultiPolygon
    with CCW exteriors / CW hole, matching osmium."""
    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Way, Relation
    from pyrosm import OSM

    path = str(tmp_path / "synthetic_mp.osm.pbf")
    writer = osmium.SimpleWriter(path)

    def node(i, lon, lat):
        writer.add_node(Node(id=i, location=(lon, lat)))

    # Outer A (with a hole), separate outer B, and outer C split across two ways.
    node(1, 25.000, 60.000), node(2, 25.002, 60.000)
    node(3, 25.002, 60.002), node(4, 25.000, 60.002)
    node(5, 25.0005, 60.0005), node(6, 25.0015, 60.0005)
    node(7, 25.0015, 60.0015), node(8, 25.0005, 60.0015)
    node(9, 25.003, 60.000), node(10, 25.004, 60.000)
    node(11, 25.004, 60.001), node(12, 25.003, 60.001)
    node(13, 25.000, 60.003), node(14, 25.001, 60.003)
    node(15, 25.001, 60.004), node(16, 25.000, 60.004)
    writer.add_way(Way(id=101, nodes=[1, 2, 3, 4, 1]))
    writer.add_way(Way(id=102, nodes=[5, 6, 7, 8, 5]))
    writer.add_way(Way(id=103, nodes=[9, 10, 11, 12, 9]))
    writer.add_way(Way(id=104, nodes=[13, 14, 15]))
    writer.add_way(Way(id=105, nodes=[15, 16, 13]))
    writer.add_relation(
        Relation(
            id=1000,
            members=[
                ("w", 101, "outer"),
                ("w", 102, "inner"),
                ("w", 103, "outer"),
                ("w", 104, "outer"),
                ("w", 105, "outer"),
            ],
            tags={"type": "multipolygon", "natural": "wood"},
        )
    )
    writer.close()

    ref = _osmium_relation_areas(path)[1000]

    rel = OSM(path).get_natural()
    rel = rel[rel.osm_type == "relation"]
    assert len(rel) == 1
    g = rel.geometry.iloc[0]
    assert g.geom_type == "MultiPolygon"
    assert len(g.geoms) == 3
    assert g.is_valid
    assert sum(len(p.interiors) for p in g.geoms) == 1
    for part in g.geoms:
        assert part.exterior.is_ccw
        for hole in part.interiors:
            assert not hole.is_ccw
    err = ref.symmetric_difference(g).area / ref.area
    assert err < 0.05
