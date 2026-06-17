"""Tests for OSM(complete_relations=True): completing the member set of relations
that straddle a bounding box, so their geometries are correct (not cut at the box
edge). Uses the real bundled Helsinki extract and the downloadable history file."""

import pytest
import shapely
from pyrosm import OSM, get_data


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_history_pbf():
    return get_data("helsinki_test_history_pbf")


# A sub-bbox of the bundled Helsinki extract that cuts through several building
# multipolygons so that some of their member ways fall outside the box.
STRADDLING_BBOX = [24.94338, 60.17089, 24.95068, 60.17687]


def _relation_geoms(gdf):
    return gdf[gdf["osm_type"] == "relation"].set_index("id").geometry


def _geom_exact(a, b):
    # Vertex-for-vertex coordinate equality, but order-independent: a relation
    # assembled from (in-box + completed) member ways may list its rings/vertices
    # in a different order than the whole-file read while being the same geometry.
    return shapely.normalize(a).equals_exact(shapely.normalize(b), tolerance=0)


def test_complete_relations_fixes_straddling_building_geometries(helsinki_pbf):
    whole_rel = _relation_geoms(OSM(helsinki_pbf).get_buildings())

    partial_rel = _relation_geoms(
        OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX).get_buildings()
    )
    complete_rel = _relation_geoms(
        OSM(
            helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=True
        ).get_buildings()
    )

    common = sorted(set(complete_rel.index) & set(whole_rel.index))
    assert len(common) > 0

    # Every relation present in the completed read matches the whole-file geometry
    # exactly (vertex-for-vertex), not merely topologically.
    for rid in common:
        assert _geom_exact(complete_rel.loc[rid], whole_rel.loc[rid])

    # ...and at least one of them was broken without completion, proving the option
    # does real work.
    n_fixed = sum(
        1
        for rid in common
        if not (
            rid in partial_rel.index
            and _geom_exact(partial_rel.loc[rid], whole_rel.loc[rid])
        )
    )
    assert n_fixed > 0


def test_complete_relations_defaults_to_off(helsinki_pbf):
    # The option is opt-in: the default is False and the cached completion set is
    # empty, so existing reads are unchanged.
    osm = OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX)
    assert osm.complete_relations is False
    osm.get_buildings()
    assert osm._relation_member_ways == []


def test_complete_relations_off_reproduces_partial_output(helsinki_pbf):
    from pandas.testing import assert_frame_equal

    default = OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX).get_buildings()
    off = OSM(
        helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=False
    ).get_buildings()

    assert_frame_equal(default.drop(columns="geometry"), off.drop(columns="geometry"))
    assert default.geometry.geom_equals_exact(off.geometry, tolerance=0).all()


def test_complete_relations_does_not_leak_into_other_layers(helsinki_pbf):
    from pandas.testing import assert_frame_equal

    # Completed member ways (fetched outside the box) must not appear in a layer
    # that does not assemble them, e.g. the driving network.
    plain = OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX).get_network("driving")
    completed = OSM(
        helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=True
    ).get_network("driving")

    assert_frame_equal(
        plain.drop(columns="geometry"), completed.drop(columns="geometry")
    )
    assert plain.geometry.geom_equals_exact(completed.geometry, tolerance=0).all()


def test_complete_relations_fetches_member_ways_outside_box(helsinki_pbf):
    osm = OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=True)
    osm.get_buildings()

    completed = osm._relation_member_ways
    assert len(completed) > 0

    # The completed ways are kept out of the in-box way records and at least one of
    # them is a way that lies outside the box (its id is not among the box's ways).
    in_box_ids = {w["id"] for w in osm._way_records}
    completed_ids = {w["id"] for w in completed}
    assert completed_ids - in_box_ids


def test_complete_relations_noop_without_bounding_box(helsinki_pbf):
    from pandas.testing import assert_frame_equal

    # A whole-file read already holds every member, so the option is a no-op there.
    whole = OSM(helsinki_pbf).get_buildings()
    osm = OSM(helsinki_pbf, complete_relations=True)
    whole_complete = osm.get_buildings()

    assert osm._relation_member_ways == []
    assert_frame_equal(
        whole.drop(columns="geometry"), whole_complete.drop(columns="geometry")
    )
    assert whole.geometry.geom_equals_exact(whole_complete.geometry, tolerance=0).all()


def test_complete_relations_noop_when_box_contains_whole_extent(helsinki_pbf):
    # A box covering the entire data extent contains every relation in full, so no
    # member ways are missing and nothing extra is fetched.
    extent = OSM(helsinki_pbf)._data_bounding_box.bounds
    big_box = [extent[0] - 0.01, extent[1] - 0.01, extent[2] + 0.01, extent[3] + 0.01]

    osm = OSM(helsinki_pbf, bounding_box=big_box, complete_relations=True)
    osm.get_buildings()
    assert osm._relation_member_ways == []


def test_complete_relations_invalid_type_raises(test_pbf):
    with pytest.raises(ValueError, match="complete_relations"):
        OSM(test_pbf, complete_relations="yes")


def test_complete_relations_with_history_file(helsinki_history_pbf):
    # complete_relations also fixes straddling relations in a history (.osh) read,
    # exercising the latest-version selection on the completed member ways. A fixed
    # timestamp keeps the read deterministic.
    bbox = [24.933271, 60.157689, 24.934307, 60.158535]
    ts = "2021-01-01"

    whole_rel = _relation_geoms(OSM(helsinki_history_pbf).get_buildings(timestamp=ts))
    partial_rel = _relation_geoms(
        OSM(helsinki_history_pbf, bounding_box=bbox).get_buildings(timestamp=ts)
    )
    osm = OSM(helsinki_history_pbf, bounding_box=bbox, complete_relations=True)
    complete_rel = _relation_geoms(osm.get_buildings(timestamp=ts))

    # The completion pass actually fetched member ways for this box.
    assert len(osm._relation_member_ways) > 0

    # The relations completion changed (relative to the partial read) are the ones it
    # repaired; each must now match the whole-file geometry. (Relations completion did
    # not touch are left as the partial read produced them, so they are not asserted.)
    common = set(complete_rel.index) & set(whole_rel.index)
    changed = [
        rid
        for rid in sorted(common)
        if rid not in partial_rel.index
        or not _geom_exact(complete_rel.loc[rid], partial_rel.loc[rid])
    ]
    assert len(changed) > 0
    for rid in changed:
        assert _geom_exact(complete_rel.loc[rid], whole_rel.loc[rid])


def test_complete_relations_available_for_custom_criteria(helsinki_pbf):
    # The shared completion machinery is reachable through every relation-using
    # layer; get_data_by_custom_criteria is the user-defined path.
    whole_rel = _relation_geoms(
        OSM(helsinki_pbf).get_data_by_custom_criteria(
            custom_filter={"building": True}, osm_keys_to_keep="building"
        )
    )
    osm = OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=True)
    complete_rel = _relation_geoms(
        osm.get_data_by_custom_criteria(
            custom_filter={"building": True}, osm_keys_to_keep="building"
        )
    )

    common = sorted(set(complete_rel.index) & set(whole_rel.index))
    assert len(common) > 0
    for rid in common:
        assert _geom_exact(complete_rel.loc[rid], whole_rel.loc[rid])


def _incomplete_relation_warnings(record):
    return [r for r in record if "extend beyond the bounding box" in str(r.message)]


def test_complete_relations_warns_when_box_cuts_relations(helsinki_pbf):
    # A bounding-box read that cuts relations warns and points to the option, when
    # completion was not requested.
    with pytest.warns(UserWarning, match="extend beyond the bounding box"):
        OSM(helsinki_pbf, bounding_box=STRADDLING_BBOX).get_buildings()


def test_complete_relations_no_warning_when_enabled(helsinki_pbf):
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        OSM(
            helsinki_pbf, bounding_box=STRADDLING_BBOX, complete_relations=True
        ).get_buildings()
    assert _incomplete_relation_warnings(record) == []


def test_complete_relations_no_warning_for_whole_file(helsinki_pbf):
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        OSM(helsinki_pbf).get_buildings()
    assert _incomplete_relation_warnings(record) == []
