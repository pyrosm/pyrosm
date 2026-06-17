import pytest
import geopandas as gpd
import pandas as pd
from pyrosm import OSM, get_data, read_tiled, generate_tiles
from pyrosm.tiling import LAYER_METHODS
from pyrosm.utils import get_bounding_box
from shapely.geometry import box, Point


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _extent(fp):
    return list(get_bounding_box(fp).bounds)


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
    # Same geometries, row-aligned by the sorted key.
    assert a.geometry.geom_equals(b.geometry).all()
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


def test_read_tiled_explicit_extent_matches_untiled(test_pbf):
    ext = _extent(test_pbf)
    full = OSM(test_pbf).get_buildings().sort_values("id").reset_index(drop=True)
    tiled = read_tiled(test_pbf, "buildings", tile_size=1.0, extent=ext)
    tiled = tiled.sort_values("id").reset_index(drop=True)
    assert full["id"].equals(tiled["id"])


def test_read_tiled_returns_none_when_no_data(test_pbf):
    # An explicit extent away from the data leaves every tile empty.
    assert (
        read_tiled(test_pbf, "buildings", tile_size=1.0, extent=[0, 0, 0.02, 0.02])
        is None
    )


def test_read_tiled_requires_extent_without_header_bbox(monkeypatch, test_pbf):
    monkeypatch.setattr("pyrosm.tiling.get_bounding_box", lambda fp: None)
    with pytest.raises(ValueError, match="no bounding box"):
        read_tiled(test_pbf, "buildings", tile_size=1.0)


class _FakeOSM:
    """Stand-in for OSM whose get_<layer> returns a fixed frame, to exercise the
    fail-closed guards that real supported layers cannot trigger."""

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
        read_tiled(test_pbf, "buildings", tile_size=1.0, extent=_extent(test_pbf))


def test_read_tiled_fails_closed_on_duplicate_key_in_tile(monkeypatch, test_pbf):
    frame = gpd.GeoDataFrame(
        {"osm_type": ["way", "way"], "id": [1, 1]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    _patch_osm(monkeypatch, frame)
    with pytest.raises(ValueError, match="multiple rows"):
        read_tiled(test_pbf, "buildings", tile_size=1.0, extent=_extent(test_pbf))


def test_read_tiled_handles_frame_without_tags_column(monkeypatch, test_pbf):
    frame = gpd.GeoDataFrame(
        {"osm_type": ["way"], "id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    _patch_osm(monkeypatch, frame)
    out = read_tiled(test_pbf, "buildings", tile_size=10000.0, extent=_extent(test_pbf))
    assert "tags" not in out.columns
    assert list(out["id"]) == [1]
    assert out.columns[-1] == "geometry"


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
    full = OSM(helsinki_pbf).get_buildings()
    full = full[full["osm_type"] != "relation"]
    tiled = read_tiled(helsinki_pbf, "buildings", tile_size=None, relations="drop")
    assert set(tiled["id"]) == set(full["id"])


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
        read_tiled(test_pbf, "buildings", tile_size=0, extent=_extent(test_pbf))


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
