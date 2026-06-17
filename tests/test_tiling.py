import pytest
import geopandas as gpd
import pandas as pd
import shapely
from pyrosm import OSM, get_data, read_tiled, generate_tiles
from pyrosm.tiling import LAYER_METHODS
from pyrosm.utils import get_bounding_box
from shapely.geometry import box, Point, Polygon


def _assert_geom_exact(a, b):
    # Exact coordinate equality (tolerance 0), but order-canonical via normalize: a
    # relation assembled from (in-tile + completed) member ways may list its rings or
    # vertices in a different order than the untiled read while being the same
    # geometry. shapely.normalize canonicalises that ordering.
    na = gpd.GeoSeries(shapely.normalize(a.values), index=a.index, crs=a.crs)
    nb = gpd.GeoSeries(shapely.normalize(b.values), index=b.index, crs=b.crs)
    assert na.geom_equals_exact(nb, tolerance=0).all()


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


@pytest.fixture
def helsinki_history_pbf():
    return get_data("helsinki_test_history_pbf")


def _extent(fp):
    return list(get_bounding_box(fp).bounds)


def _clip_to(gdf, geom):
    """The untiled ``gdf`` clipped to ``geom`` the way OSM's bounding_box does: keep
    whole features that intersect it (no geometry cutting)."""
    cols = list(gdf.columns)
    filt = gpd.GeoDataFrame({"geometry": [geom]}, crs="epsg:4326", index=[0])
    return gpd.sjoin(gdf, filt, how="inner")[cols].reset_index(drop=True)


def _assert_matches_untiled(
    fp, layer, tile_size, relations="error", check_tags=True, **kw
):
    """read_tiled() must reproduce the untiled layer read."""
    full = getattr(OSM(fp), LAYER_METHODS[layer])(**kw)
    tiled = read_tiled(fp, layer, tile_size=tile_size, relations=relations, **kw)

    if relations == "drop" and full is not None:
        full = full[full["osm_type"] != "relation"]

    if full is None or len(full) == 0:
        assert tiled is None or len(tiled) == 0
        return

    a = full.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = tiled.sort_values(["osm_type", "id"]).reset_index(drop=True)

    # Same identity rows.
    assert a[["osm_type", "id"]].equals(b[["osm_type", "id"]])
    # Same column set (order may differ).
    assert set(a.columns) == set(b.columns)
    b = b[a.columns]
    # Same geometries (exact coordinates), row-aligned by the sorted key.
    _assert_geom_exact(a.geometry, b.geometry)
    # Every non-string column keeps its exact dtype. Only free-form string columns
    # are exempt: pandas infers object vs StringDtype from content, which is not
    # stable even between two untiled reads, so they are compared dtype-insensitively.
    for c in a.columns:
        if c == "geometry":
            continue
        if a[c].dtype == object or isinstance(a[c].dtype, pd.StringDtype):
            continue
        assert a[c].dtype == b[c].dtype, "dtype mismatch on column {0}".format(c)
    # Same non-geometry values. The "tags" column is excluded for node-heavy layers
    # whose bounding-box reads differ from the unfiltered read (see check_tags).
    skip = {"geometry"} | (set() if check_tags else {"tags"})
    cols = [c for c in a.columns if c not in skip]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)


# --- generate_tiles -------------------------------------------------------


def test_generate_tiles_covers_extent(test_pbf):
    ext = _extent(test_pbf)
    tiles = generate_tiles(ext, 0.015)
    assert len(tiles) > 1
    minx, miny, maxx, maxy = ext
    assert min(t[0] for t in tiles) == pytest.approx(minx)
    assert min(t[1] for t in tiles) == pytest.approx(miny)
    assert max(t[2] for t in tiles) == pytest.approx(maxx)
    assert max(t[3] for t in tiles) == pytest.approx(maxy)


def test_generate_tiles_single_tile_when_large(test_pbf):
    assert len(generate_tiles(_extent(test_pbf), 10.0)) == 1


def test_generate_tiles_mask_reduces(test_pbf):
    ext = _extent(test_pbf)
    minx, miny, maxx, maxy = ext
    mask = box(minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)
    full = generate_tiles(ext, 0.008)
    reduced = generate_tiles(ext, 0.008, mask=mask)
    assert 0 < len(reduced) < len(full)
    assert all(box(*t).intersects(mask) for t in reduced)


def test_generate_tiles_invalid():
    with pytest.raises(ValueError):
        generate_tiles([0, 0, 1, 1], 0)
    with pytest.raises(ValueError):
        generate_tiles([1, 0, 0, 1], 0.1)


# --- read_tiled parity (way-based layers stitch exactly) ------------------


def test_read_tiled_buildings_matches_untiled(test_pbf):
    _assert_matches_untiled(test_pbf, "buildings", 1.0)


def test_read_tiled_network_matches_untiled(test_pbf):
    _assert_matches_untiled(test_pbf, "network", 1.0)


def test_read_tiled_landuse_matches_untiled(helsinki_pbf):
    _assert_matches_untiled(helsinki_pbf, "landuse", 0.25, relations="drop")


def test_read_tiled_custom_criteria_single_tile_matches_untiled(test_pbf):
    _assert_matches_untiled(
        test_pbf, "custom_criteria", 10000.0, custom_filter={"amenity": True}
    )


def test_read_tiled_custom_criteria_multitile_ids_and_geoms(test_pbf):
    _assert_matches_untiled(
        test_pbf,
        "custom_criteria",
        1.0,
        custom_filter={"amenity": True},
        check_tags=False,
    )


def test_read_tiled_boundaries_no_boundary_data_returns_none(helsinki_pbf):
    # The bundled Helsinki extract has no assemblable boundary relations (incomplete
    # boundaries are dropped, #154), so a tiled boundaries read yields no features.
    assert read_tiled(helsinki_pbf, "boundaries", tile_size=0.25) is None


def test_read_tiled_natural_multitile_ids_and_geoms(helsinki_pbf):
    # 'natural' includes node features (trees, peaks), so like POIs its tags
    # column is subject to the pyrosm bounding-box node behaviour; ids/geometries
    # still stitch exactly.
    _assert_matches_untiled(
        helsinki_pbf, "natural", 0.25, relations="drop", check_tags=False
    )


def test_read_tiled_single_tile_equals_untiled(test_pbf):
    _assert_matches_untiled(test_pbf, "buildings", 10000.0)


def test_read_tiled_several_tile_sizes(test_pbf):
    for ts in (0.5, 1.0, 2.0):
        _assert_matches_untiled(test_pbf, "network", ts)


# --- POIs (node-heavy): single tile is exact; multi-tile keeps ids/geoms --


def test_read_tiled_pois_single_tile_matches_untiled(test_pbf):
    _assert_matches_untiled(test_pbf, "pois", 10000.0, custom_filter={"amenity": True})


def test_read_tiled_pois_multitile_ids_and_geoms(test_pbf):
    _assert_matches_untiled(
        test_pbf, "pois", 1.0, custom_filter={"amenity": True}, check_tags=False
    )


# --- relations ------------------------------------------------------------


def test_read_tiled_relations_error_raises(helsinki_pbf):
    with pytest.raises(ValueError, match="relation"):
        read_tiled(helsinki_pbf, "buildings", tile_size=0.25, relations="error")


def test_read_tiled_relations_drop_matches_nonrelation(helsinki_pbf):
    _assert_matches_untiled(helsinki_pbf, "buildings", 0.25, relations="drop")


def test_read_tiled_relations_complete_buildings_matches_untiled(helsinki_pbf):
    # The default: relations rebuilt from their full member set across tiles, so the
    # stitched result (relations included) equals the untiled read.
    _assert_matches_untiled(helsinki_pbf, "buildings", 0.25, relations="complete")


def test_read_tiled_relations_complete_landuse_matches_untiled(helsinki_pbf):
    _assert_matches_untiled(helsinki_pbf, "landuse", 0.25, relations="complete")


def test_read_tiled_relations_complete_natural_matches_untiled(helsinki_pbf):
    # Exercises the natural layer's completion path (no natural relations in the
    # bundled data, so it equals the untiled read).
    _assert_matches_untiled(
        helsinki_pbf, "natural", 0.25, relations="complete", check_tags=False
    )


def test_read_tiled_relations_complete_pois_matches_untiled(test_pbf):
    _assert_matches_untiled(
        test_pbf,
        "pois",
        1.0,
        relations="complete",
        custom_filter={"amenity": True},
        check_tags=False,
    )


def test_read_tiled_relations_complete_custom_criteria_matches_untiled(helsinki_pbf):
    # custom_criteria with a building filter rebuilds the building relations.
    _assert_matches_untiled(
        helsinki_pbf,
        "custom_criteria",
        0.25,
        relations="complete",
        custom_filter={"building": True},
        osm_keys_to_keep="building",
        check_tags=False,
    )


def test_read_tiled_relations_complete_is_default(helsinki_pbf):
    # No relations= argument uses "complete", so building relations are present and
    # match the untiled geometries.
    full = OSM(helsinki_pbf).get_buildings()
    tiled = read_tiled(helsinki_pbf, "buildings", tile_size=0.25)
    full_rel = set(full[full["osm_type"] == "relation"]["id"])
    tiled_rel = set(tiled[tiled["osm_type"] == "relation"]["id"])
    assert full_rel == tiled_rel
    assert len(full_rel) > 0


def test_read_tiled_suppresses_per_tile_incomplete_relation_warning(helsinki_pbf):
    # Each tile is a bbox read, so straddling relations trigger the OSM reader's
    # "incomplete relation" warning -- but read_tiled rebuilds them once per call, so
    # that internal warning must not leak out to the caller.
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        read_tiled(helsinki_pbf, "buildings", tile_size=0.25)
    assert not [w for w in record if "extend beyond the bounding box" in str(w.message)]


def test_read_tiled_relations_complete_keep_metadata_false(helsinki_pbf):
    # Completion honours keep_metadata=False (the probe and the member-only read use
    # the same setting) and still rebuilds the relations.
    full = OSM(helsinki_pbf, keep_metadata=False).get_buildings()
    tiled = read_tiled(helsinki_pbf, "buildings", tile_size=0.25, keep_metadata=False)
    assert "timestamp" not in tiled.columns
    full_rel = set(full[full["osm_type"] == "relation"]["id"])
    tiled_rel = set(tiled[tiled["osm_type"] == "relation"]["id"])
    assert full_rel == tiled_rel
    assert len(full_rel) > 0


def test_parse_relations_only_includes_relations_dropped_by_layer(helsinki_pbf):
    # Relation definitions come from an unfiltered relation parse, so relations whose
    # tiled/bbox geometry is dropped (or that belong to another layer) are still
    # present as candidates.
    from pyrosm.pbfreader import parse_relations_only

    all_rel = parse_relations_only(helsinki_pbf)
    n_all = len(all_rel.get("id", []))
    n_building_rel = (OSM(helsinki_pbf).get_buildings()["osm_type"] == "relation").sum()
    assert n_all > n_building_rel > 0


def _assert_bbox_clips_like_untiled(fp, layer, bounding_box, clip_geom, tile_size=0.25):
    """read_tiled(bounding_box=...) equals the whole-file layer read clipped to the
    box: same (osm_type, id) rows and exact geometries (relations rebuilt whole)."""
    tiled = read_tiled(
        fp, layer, tile_size=tile_size, bounding_box=bounding_box, relations="complete"
    )
    oracle = _clip_to(getattr(OSM(fp), LAYER_METHODS[layer])(), clip_geom)
    a = oracle.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = tiled.sort_values(["osm_type", "id"]).reset_index(drop=True)
    assert a[["osm_type", "id"]].equals(b[["osm_type", "id"]])
    _assert_geom_exact(a.geometry, b.geometry)
    return b


def test_read_tiled_bounding_box_clips_like_untiled(helsinki_pbf):
    # A list bounding_box restricts the read to features intersecting it and keeps whole
    # (un-clipped) geometries -- identical to the whole-file read clipped to the box,
    # with relations rebuilt whole (never cut at the box edge). Covers the way-based and
    # node-heavy/relation-aware layers the plan names.
    geom = box(24.94, 60.165, 24.95, 60.175)
    for layer in ("buildings", "landuse", "natural"):
        b = _assert_bbox_clips_like_untiled(
            helsinki_pbf, layer, list(geom.bounds), geom
        )
        if layer == "buildings":
            # Relations are kept whole, not clipped: at least one kept relation extends
            # beyond the box (clipping the geometry to the box would have cut it).
            rel = b[b["osm_type"] == "relation"].geometry
            assert len(rel) > 0
            assert any(not geom.contains(g) for g in rel)


def test_read_tiled_polygon_bounding_box_clips_like_untiled(helsinki_pbf):
    # A polygon bounding_box clips the stitched result to the polygon (by intersection),
    # matching the whole-file read clipped to the same polygon, across layers.
    geom = box(24.93, 60.16, 24.95, 60.175)
    for layer in ("buildings", "landuse", "natural"):
        _assert_bbox_clips_like_untiled(helsinki_pbf, layer, geom, geom)


def test_layer_relation_filters_reused_from_feature_modules():
    # The per-layer relation filters completion reuses match the feature defaults.
    from pyrosm.boundary import boundary_relation_filter
    from pyrosm.buildings import building_relation_filter
    from pyrosm.landuse import landuse_relation_filter
    from pyrosm.natural import natural_relation_filter
    from pyrosm.pois import poi_relation_filter

    # Defaults.
    assert building_relation_filter(None) == {"building": [True]}
    assert landuse_relation_filter(None) == {"landuse": [True]}
    assert natural_relation_filter(None) == {"natural": [True]}
    assert boundary_relation_filter(None, "administrative") == {
        "boundary": ["administrative"]
    }
    assert boundary_relation_filter(None, "all") == {"boundary": [True]}
    assert poi_relation_filter(None) == {
        "amenity": [True],
        "shop": [True],
        "tourism": [True],
    }

    # A user custom_filter lacking the layer's key gets it added (and is validated).
    assert building_relation_filter({"amenity": True}) == {
        "amenity": [True],
        "building": [True],
    }
    assert landuse_relation_filter({"amenity": True}) == {
        "amenity": [True],
        "landuse": [True],
    }
    assert natural_relation_filter({"amenity": True}) == {
        "amenity": [True],
        "natural": [True],
    }
    assert boundary_relation_filter({"admin_level": ["8"]}, "administrative") == {
        "admin_level": ["8"],
        "boundary": [True],
    }
    assert poi_relation_filter({"shop": ["bakery"]}) == {"shop": ["bakery"]}

    # A user custom_filter that already has the layer's key is left as-is.
    assert building_relation_filter({"building": ["house"]}) == {"building": ["house"]}
    assert landuse_relation_filter({"landuse": ["forest"]}) == {"landuse": ["forest"]}
    assert natural_relation_filter({"natural": ["wood"]}) == {"natural": ["wood"]}


def test_read_tiled_complete_does_not_overfetch_non_layer_relations(
    helsinki_pbf, monkeypatch
):
    # Completion's expensive node fetch is scoped to the requested layer's relations,
    # not every relation touching the tiles.
    import pyrosm.pbfreader as pbf
    import pyrosm.tiling as tiling

    fetched = set()
    orig = pbf.fetch_member_nodes

    def spy(filepath, node_ids, *args, **kwargs):
        fetched.update(int(n) for n in node_ids)
        return orig(filepath, node_ids, *args, **kwargs)

    monkeypatch.setattr(tiling, "fetch_member_nodes", spy)

    gdf = read_tiled(helsinki_pbf, "buildings", tile_size=0.25)
    assert (gdf["osm_type"] == "relation").sum() > 0

    # Upper bound: the member nodes of *all* relations intersecting the file.
    all_member_ids = set()
    all_rel = pbf.parse_relations_only(helsinki_pbf)
    for members in all_rel["members"]:
        all_member_ids.update(tiling._relation_member_way_ids(members))
    all_member_ways = pbf.fetch_member_ways(helsinki_pbf, all_member_ids)
    all_nodes = set()
    for way in all_member_ways:
        all_nodes.update(way["nodes"])

    # The layer-scoped fetch is non-empty and strictly smaller than the all-relation
    # member-node set (the route/boundary members are no longer fetched).
    assert 0 < len(fetched) < len(all_nodes)


def test_read_tiled_custom_criteria_keep_relations_false_skips_completion(
    helsinki_pbf, monkeypatch
):
    # keep_relations=False returns no relation rows, so completion (and its node
    # fetch) is short-circuited entirely.
    import pyrosm.pbfreader as pbf
    import pyrosm.tiling as tiling

    calls = {"n": 0}
    orig = pbf.fetch_member_nodes

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(tiling, "fetch_member_nodes", spy)

    gdf = read_tiled(
        helsinki_pbf,
        "custom_criteria",
        tile_size=0.25,
        custom_filter={"building": True},
        keep_relations=False,
    )
    assert calls["n"] == 0
    if gdf is not None:
        assert (gdf["osm_type"] == "relation").sum() == 0


def test_read_tiled_complete_with_no_relation_definitions(test_pbf, monkeypatch):
    # When the file has no relation definitions, completion finds nothing to rebuild
    # and the result carries no relation rows.
    import pyrosm.tiling as tiling

    monkeypatch.setattr(tiling, "parse_relations_only", lambda *a, **k: {})
    gdf = read_tiled(test_pbf, "buildings", tile_size=1.0)
    if gdf is not None:
        assert (gdf["osm_type"] == "relation").sum() == 0


def test_fetch_relation_members_returns_empty_for_no_match(test_pbf):
    from pyrosm.pbfreader import fetch_relation_members

    member_ways, coords = fetch_relation_members(test_pbf, set())
    assert member_ways == []
    assert coords is None


def test_custom_criteria_filter_spec_osm_keys(test_pbf):
    from pyrosm.tiling import _layer_relation_filter_spec

    # A string osm_keys_to_keep is normalised to a list; absent -> None (derived).
    _, keys, ft = _layer_relation_filter_spec(
        "custom_criteria",
        {"custom_filter": {"building": True}, "osm_keys_to_keep": "building"},
    )
    assert keys == ["building"]
    assert ft == "keep"
    _, keys2, _ = _layer_relation_filter_spec(
        "custom_criteria", {"custom_filter": {"building": True}}
    )
    assert keys2 is None


def test_read_relations_from_members_without_coords_returns_none(test_pbf):
    from pyrosm.tiling import _read_relations_from_members

    # No member node coordinates -> nothing to assemble.
    assert (
        _read_relations_from_members(
            test_pbf, "get_buildings", {}, [], None, True, None, {}
        )
        is None
    )


def test_candidate_relations_edge_cases(helsinki_pbf):
    from pyrosm.pbfreader import parse_relations_only
    from pyrosm.tiling import _candidate_relations

    # No relations at all -> no candidates.
    assert _candidate_relations({}, {1, 2}) == (None, set())
    # Relations present but none has a member in the kept tiles -> no candidates.
    relations = parse_relations_only(helsinki_pbf)
    candidates, ids = _candidate_relations(relations, set())
    assert candidates is None
    assert ids == set()


def test_relation_completion_reader_on_history_file(helsinki_history_pbf):
    # Exercise the relation-only reader passes (incl. their OSH latest-version
    # branches) on a history file at a fixed timestamp.
    from pyrosm.pbfreader import fetch_relation_members, parse_relations_only

    probe = OSM(helsinki_history_pbf)
    probe._set_current_time("2021-01-01")
    unix_time = probe._current_timestamp

    relations = parse_relations_only(helsinki_history_pbf, unix_time)
    assert len(relations.get("id", [])) > 0

    member_ids = set()
    for members in relations["members"]:
        for mid, mtype in zip(members["member_id"], members["member_type"]):
            if mtype == b"way":
                member_ids.add(int(mid))

    member_ways, coords = fetch_relation_members(
        helsinki_history_pbf, member_ids, unix_time
    )
    assert len(member_ways) > 0
    assert coords is not None


# --- input handling -------------------------------------------------------


def test_read_tiled_does_not_mutate_custom_filter(test_pbf):
    cf = {"amenity": True}
    read_tiled(test_pbf, "pois", tile_size=1.0, custom_filter=cf)
    assert cf == {"amenity": True}


def test_read_tiled_rejects_network_nodes_true(test_pbf):
    with pytest.raises(ValueError, match="nodes=True"):
        read_tiled(test_pbf, "network", tile_size=1.0, nodes=True)


def test_read_tiled_rejects_unsupported_layer(test_pbf):
    with pytest.raises(ValueError, match="[Uu]nsupported layer"):
        read_tiled(test_pbf, "roads", tile_size=1.0)


def test_read_tiled_rejects_bad_relations(test_pbf):
    with pytest.raises(ValueError, match="relations"):
        read_tiled(test_pbf, "buildings", tile_size=1.0, relations="keep")


def test_generate_tiles_step_rounds_to_zero():
    with pytest.raises(ValueError, match="too small"):
        generate_tiles([0, 0, 1, 1], 1e-9)


def test_read_tiled_full_bounding_box_matches_untiled(test_pbf):
    # A bounding_box equal to the data extent clips nothing, so the result matches the
    # untiled read.
    ext = _extent(test_pbf)
    full = OSM(test_pbf).get_buildings().sort_values("id").reset_index(drop=True)
    tiled = read_tiled(test_pbf, "buildings", tile_size=1.0, bounding_box=ext)
    tiled = tiled.sort_values("id").reset_index(drop=True)
    assert full["id"].equals(tiled["id"])


def test_read_tiled_returns_none_when_no_data(test_pbf):
    # A bounding_box away from the data leaves every tile empty.
    assert (
        read_tiled(
            test_pbf, "buildings", tile_size=1.0, bounding_box=[0, 0, 0.02, 0.02]
        )
        is None
    )


def test_read_tiled_rejects_bad_bounding_box(test_pbf):
    with pytest.raises(ValueError, match="4"):
        read_tiled(test_pbf, "buildings", tile_size=1.0, bounding_box=[0, 0, 1])
    with pytest.raises(ValueError, match="Invalid bounding_box"):
        read_tiled(test_pbf, "buildings", tile_size=1.0, bounding_box=[1, 0, 0, 1])
    with pytest.raises(ValueError, match="bounding_box"):
        read_tiled(test_pbf, "buildings", tile_size=1.0, bounding_box="nope")


def test_read_tiled_requires_bounding_box_without_header_bbox(monkeypatch, test_pbf):
    monkeypatch.setattr("pyrosm.tiling.get_bounding_box", lambda fp: None)
    with pytest.raises(ValueError, match="no bounding box"):
        read_tiled(test_pbf, "buildings", tile_size=1.0)


class _FakeOSM:
    """Stand-in for OSM whose get_<layer> returns a fixed frame, to exercise the
    fail-closed guards that real supported layers cannot trigger."""

    allowed_bbox_types = OSM.allowed_bbox_types

    def __init__(self, frame, *args, **kwargs):
        self._frame = frame

    def __call__(self, *args, **kwargs):
        return self

    def get_buildings(self, **kwargs):
        return self._frame


def _patch_osm(monkeypatch, frame):
    monkeypatch.setattr("pyrosm.tiling.OSM", _FakeOSM(frame))


def test_read_tiled_fails_closed_without_identity_columns(monkeypatch, test_pbf):
    frame = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    _patch_osm(monkeypatch, frame)
    with pytest.raises(ValueError, match="osm_type"):
        read_tiled(
            test_pbf,
            "buildings",
            tile_size=1.0,
            relations="drop",
        )


def test_read_tiled_fails_closed_on_duplicate_key_in_tile(monkeypatch, test_pbf):
    frame = gpd.GeoDataFrame(
        {"osm_type": ["way", "way"], "id": [1, 1]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    _patch_osm(monkeypatch, frame)
    with pytest.raises(ValueError, match="multiple rows"):
        read_tiled(
            test_pbf,
            "buildings",
            tile_size=1.0,
            relations="drop",
        )


def test_read_tiled_handles_frame_without_tags_column(monkeypatch, test_pbf):
    frame = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    _patch_osm(monkeypatch, frame)
    out = read_tiled(
        test_pbf,
        "buildings",
        tile_size=10000.0,
        relations="drop",
    )
    assert "tags" not in out.columns
    assert list(out["id"]) == [1]
    assert out.columns[-1] == "geometry"


def test_read_tiled_returns_none_when_clip_removes_all(monkeypatch, test_pbf):
    # A polygon bounding_box whose envelope drives a tile that yields a feature, but
    # whose polygon excludes that feature -> the post-clip result is empty.
    frame = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [1]}, geometry=[Point(0.3, 0.3)], crs="EPSG:4326"
    )
    _patch_osm(monkeypatch, frame)
    triangle = Polygon(
        [(0, 0), (0.4, 0), (0, 0.4)]
    )  # envelope holds the point; it does not
    assert (
        read_tiled(
            test_pbf,
            "buildings",
            tile_size=10000.0,
            bounding_box=triangle,
            relations="drop",
        )
        is None
    )


# --- km^2 tile_size and auto-sizing ---------------------------------------


def test_read_tiled_tile_size_is_km2_area(helsinki_pbf):
    # A km^2 tile_size produces full tiles whose ground area matches the target.
    import math
    import pyrosm.tiling as T

    ext = _extent(helsinki_pbf)
    target = 0.25
    dlon, dlat = T._km2_to_degree_steps(target, T._extent_centre_latitude(ext))
    tiles = T._build_tile_grid(ext, dlon, dlat)
    assert len(tiles) > 1

    areas = []
    for x0, y0, x1, y1 in tiles:
        w_km = (
            (x1 - x0)
            * T._KM_PER_DEG_LON_EQUATOR
            * math.cos(math.radians((y0 + y1) / 2))
        )
        h_km = (y1 - y0) * T._KM_PER_DEG_LAT
        areas.append(w_km * h_km)
    # The full (unclamped) tiles match the target area to within the E7 snapping.
    assert max(areas) == pytest.approx(target, rel=0.02)


def test_read_tiled_auto_tile_size_matches_untiled(helsinki_pbf):
    # tile_size=None auto-sizes; the stitched result -- including the completed
    # relations -- equals the untiled read (geometries, columns and dtypes).
    _assert_matches_untiled(helsinki_pbf, "buildings", None, relations="complete")


def test_auto_tile_km2_units(monkeypatch):
    import math
    import sys
    import types

    import pyrosm.tiling as T

    # 1000 MB available, 100 MB file -> budget = 1000*SAFETY, peak = K_MB_PER_MB*100.
    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=1000 * 1e6)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(T.os.path, "getsize", lambda fp: 100 * 1e6)

    area = T._auto_tile_km2("dummy.osm.pbf", 300.0)
    n = max(1, math.ceil(T.K_MB_PER_MB * 100 / (1000 * T.SAFETY)))
    assert area == pytest.approx(300.0 / n)


def test_auto_tile_km2_fallback_without_psutil(monkeypatch):
    import sys

    import pyrosm.tiling as T

    # No psutil available -> fixed fallback.
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert T._auto_tile_km2("dummy.osm.pbf", 300.0) == T.DEFAULT_TILE_KM2


def test_auto_tile_km2_fallback_on_unreadable_file_size(monkeypatch):
    import sys
    import types

    import pyrosm.tiling as T

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=1000 * 1e6)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    def _raise(fp):
        raise OSError("no such file")

    monkeypatch.setattr(T.os.path, "getsize", _raise)
    assert T._auto_tile_km2("dummy.osm.pbf", 300.0) == T.DEFAULT_TILE_KM2


def test_read_tiled_rejects_nonpositive_tile_size(test_pbf):
    with pytest.raises(ValueError, match="km"):
        read_tiled(test_pbf, "buildings", tile_size=0, bounding_box=_extent(test_pbf))


def test_auto_tile_km2_fallback_on_zero_available_memory(monkeypatch):
    import sys
    import types

    import pyrosm.tiling as T

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=0)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(T.os.path, "getsize", lambda fp: 100 * 1e6)
    assert T._auto_tile_km2("dummy.osm.pbf", 300.0) == T.DEFAULT_TILE_KM2


def test_auto_tile_km2_fallback_on_nonpositive_extent_area(monkeypatch):
    import sys
    import types

    import pyrosm.tiling as T

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=1000 * 1e6)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(T.os.path, "getsize", lambda fp: 100 * 1e6)
    assert T._auto_tile_km2("dummy.osm.pbf", 0.0) == T.DEFAULT_TILE_KM2
