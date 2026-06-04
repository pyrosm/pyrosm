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
