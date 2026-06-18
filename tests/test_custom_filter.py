import pytest

from pyrosm import get_data
from pyrosm.utils.download import download


@pytest.fixture
def california_highway_motorway_pbf():
    filename = "california_highway_motorway.osm.pbf"
    url = f"https://github.com/eracle/pyrosm-test-data/raw/refs/heads/main/{filename}"
    return download(url=url, filename=filename, update=None, target_dir=None)


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def helsinki_region_pbf():
    pbf_path = get_data("helsinki_region_pbf")
    return pbf_path


@pytest.fixture
def default_filter():
    return {
        "amenity": True,
        "craft": True,
        "historic": True,
        "leisure": True,
        "shop": True,
        "tourism": True,
    }


@pytest.fixture
def test_output_dir():
    import os, tempfile

    return os.path.join(tempfile.gettempdir(), "pyrosm_test_results")


def test_get_data_by_custom_criteria_custom_filter(california_highway_motorway_pbf):
    from geopandas import GeoDataFrame
    from pyrosm import OSM

    osm = OSM(filepath=california_highway_motorway_pbf)
    gdf = osm.get_data_by_custom_criteria(custom_filter={"highway": ["motorway"]})

    assert isinstance(gdf, GeoDataFrame)


def test_parsing_osm_with_custom_filter_by_excluding_tags(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    import pyproj

    osm = OSM(filepath=test_pbf)

    # Keep only building as column
    tags_as_columns = ["building"]
    # Get all buildings except "residential"
    custom_filter = {"building": ["residential"]}
    filter_type = "exclude"
    osm_type = "building"
    gdf = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter,
        filter_type=filter_type,
        osm_keys_to_keep=osm_type,
        tags_as_columns=tags_as_columns,
    )

    assert isinstance(gdf, GeoDataFrame)

    # Only following columns should exist after specifying tags_as_columns
    allowed_columns = [
        "geometry",
        "tags",
        "building",
        "id",
        "osm_type",
        "version",
        "timestamp",
        "changeset",
    ]
    for col in gdf.columns:
        assert col in allowed_columns

    # Building columns should not have any "residential" tags
    assert "residential" not in gdf["building"].tolist()

    # Required keys
    required = ["id", "geometry"]
    for col in required:
        assert col in gdf.columns

    # Test shape
    assert len(gdf) == 1054
    assert gdf.crs == pyproj.CRS.from_epsg(4326)


def test_parsing_osm_with_custom_filter_by_including_tags(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    import pyproj

    osm = OSM(filepath=test_pbf)

    # Keep only building as column
    tags_as_columns = ["building"]
    # Get all buildings that are "retail"
    custom_filter = {"building": ["retail"]}
    filter_type = "keep"
    osm_type = "building"
    gdf = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter,
        filter_type=filter_type,
        osm_keys_to_keep=osm_type,
        tags_as_columns=tags_as_columns,
    )

    assert isinstance(gdf, GeoDataFrame)

    # Only following columns should exist after specifying tags_as_columns
    allowed_columns = [
        "geometry",
        "tags",
        "building",
        "id",
        "osm_type",
        "version",
        "timestamp",
        "changeset",
    ]
    for col in gdf.columns:
        assert col in allowed_columns

    # Building columns should not have any "residential" tags
    assert len(gdf["building"].unique()) == 1
    assert gdf["building"].unique()[0] == "retail"

    # Required keys
    required = ["id", "geometry"]
    for col in required:
        assert col in gdf.columns

    # Test shape
    assert len(gdf) == 2
    assert gdf.crs == pyproj.CRS.from_epsg(4326)


def test_using_incorrect_filter(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)

    # Test that passing incorrect data works as should
    # (custom_filter=None is now valid and means "return everything", see #113)
    # 1.
    custom_filter = {"building": [1]}
    try:
        gdf = osm.get_data_by_custom_criteria(custom_filter=custom_filter)
    except ValueError as e:
        if "string" in str(e):
            pass
        else:
            raise e

    custom_filter = {"building": ["correct_string", 1]}
    # 2.
    try:
        gdf = osm.get_data_by_custom_criteria(custom_filter=custom_filter)
    except ValueError as e:
        if "string" in str(e):
            pass
        else:
            raise e
    # 3.
    custom_filter = {0: ["residential"]}
    try:
        gdf = osm.get_data_by_custom_criteria(custom_filter=custom_filter)
    except ValueError as e:
        if "string" in str(e):
            pass
        else:
            raise e


def test_using_incorrect_tags(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)

    # Incorrect tags
    # --------------
    tags_as_columns = [1]
    custom_filter = {"building": ["retail"]}
    # Test that passing incorrect data works as should
    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, tags_as_columns=tags_as_columns
        )
    except ValueError as e:
        if "All tags listed in 'tags_as_columns' should be strings" in str(e):
            pass
        else:
            raise e


def test_using_incorrect_filter_type(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)

    custom_filter = {"building": ["retail"]}
    filter_type = "incorrect_test"
    # Test that passing incorrect data works as should
    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, filter_type=filter_type
        )
    except ValueError as e:
        if "should be either 'keep' or 'exclude'" in str(e):
            pass
        else:
            raise e


def test_using_incorrect_booleans(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)

    custom_filter = {"building": ["retail"]}
    incorrect_bool = "foo"
    # Test that passing incorrect data works as should
    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, keep_nodes=incorrect_bool
        )
    except ValueError as e:
        if "'keep_nodes' should be boolean type: True or False" in str(e):
            pass
        else:
            raise e

    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, keep_ways=incorrect_bool
        )
    except ValueError as e:
        if "'keep_ways' should be boolean type: True or False" in str(e):
            pass
        else:
            raise e

    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, keep_relations=incorrect_bool
        )
    except ValueError as e:
        if "'keep_relations' should be boolean type: True or False" in str(e):
            pass
        else:
            raise e

    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter,
            keep_relations=False,
            keep_ways=False,
            keep_nodes=False,
        )
    except ValueError as e:
        if "At least on of the following parameters should be True" in str(e):
            pass
        else:
            raise e


def test_using_incorrect_osm_keys(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)

    osm_keys = 1
    custom_filter = {"building": ["retail"]}
    # Test that passing incorrect data works as should
    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, osm_keys_to_keep=osm_keys
        )
    except ValueError as e:
        if "'osm_keys_to_keep' -parameter should be of type str or list." in str(e):
            pass
        else:
            raise e


def test_reading_with_custom_filters_with_including(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame

    # Get first all data
    osm = OSM(filepath=test_pbf)
    gdf_all = osm.get_buildings()

    # Find out all 'building' tags
    cnts = gdf_all["building"].value_counts()
    for filter_, cnt in cnts.items():
        # Use the custom filter
        filtered = osm.get_data_by_custom_criteria(
            custom_filter={"building": [filter_]}, filter_type="keep"
        )

        assert isinstance(filtered, GeoDataFrame)
        assert isinstance(filtered.loc[0, "geometry"], Polygon)
        assert len(filtered) == cnt
        # Now should only have buildings with given key
        assert len(filtered["building"].unique()) == 1

        required_cols = ["building", "id", "timestamp", "version", "geometry"]

        for col in required_cols:
            assert col in filtered.columns


def test_reading_with_custom_filters_with_excluding(test_pbf):
    from pyrosm import OSM
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame

    # Get first all data
    osm = OSM(filepath=test_pbf)
    gdf_all = osm.get_buildings()

    # Find out all 'building' tags
    cnts = gdf_all["building"].value_counts()
    n = len(gdf_all)
    for filter_, cnt in cnts.items():
        # Use the custom filter
        filtered = osm.get_data_by_custom_criteria(
            custom_filter={"building": [filter_]}, filter_type="exclude"
        )

        assert isinstance(filtered, GeoDataFrame)
        assert isinstance(filtered.loc[0, "geometry"], Polygon)
        assert len(filtered) == n - cnt
        # Now should not have the filter_ in buildings
        assert filter_ not in filtered["building"].unique()

        required_cols = ["building", "id", "timestamp", "version", "geometry"]

        for col in required_cols:
            assert col in filtered.columns


def test_reading_with_custom_filters_selecting_specific_osm_element(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Get first all data
    osm = OSM(filepath=helsinki_pbf)

    # Test getting only relations
    # ---------------------------
    filtered = osm.get_data_by_custom_criteria(
        custom_filter={"building": True},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=False,
        keep_relations=True,
    )
    assert isinstance(filtered, GeoDataFrame)

    # Now should only have 'relation' osm_type
    assert len(filtered["osm_type"].unique()) == 1
    assert filtered["osm_type"].unique()[0] == "relation"
    assert len(filtered) == 67

    # Test getting only ways
    # ---------------------------
    filtered = osm.get_data_by_custom_criteria(
        custom_filter={"building": True},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
    )
    assert isinstance(filtered, GeoDataFrame)

    # Now should only have 'way' osm_type
    assert len(filtered["osm_type"].unique()) == 1
    assert filtered["osm_type"].unique()[0] == "way"
    assert len(filtered) == 425

    # Test getting only nodes
    # ---------------------------
    filtered = osm.get_data_by_custom_criteria(
        custom_filter={"building": True},
        filter_type="keep",
        keep_nodes=True,
        keep_ways=False,
        keep_relations=False,
    )
    assert isinstance(filtered, GeoDataFrame)

    # Now should only have 'node' osm_type
    assert len(filtered["osm_type"].unique()) == 1
    assert filtered["osm_type"].unique()[0] == "node"
    assert len(filtered) == 36


def test_custom_filters_with_custom_keys(helsinki_region_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Get first all data
    osm = OSM(filepath=helsinki_region_pbf)

    # Test reading public transport related data
    filtered = osm.get_data_by_custom_criteria(
        custom_filter={"public_transport": True},
        filter_type="keep",
    )
    assert isinstance(filtered, GeoDataFrame)
    assert len(filtered) == 5542

    # Combining a True ("any value") filter with an explicit-value filter
    # should keep all buildings AND railway=station features (issue #224).
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"building": True, "railway": ["station"]},
        filter_type="keep",
    )
    # Assert on row/content counts (stable), not the column count: the latter
    # can be polluted by extra_attributes leaking into the shared config when
    # other tests run first.
    assert len(gdf) == 176742
    assert gdf["building"].notna().sum() == 176675
    assert (gdf["railway"] == "station").sum() == 67

    # Test a more complicated query
    # -----------------------------

    # Test reading all transit related data (bus, trains, trams, metro etc.)
    # Exclude nodes (not keeping stops, etc.)
    routes = ["bus", "ferry", "railway", "subway", "train", "tram", "trolleybus"]
    rails = ["tramway", "light_rail", "rail", "subway", "tram"]
    # 'express' comes with routes
    bus = ["yes", "express"]

    transit = osm.get_data_by_custom_criteria(
        custom_filter={"route": routes, "railway": rails, "bus": bus},
        filter_type="keep",
        keep_nodes=False,
    )

    required_columns = ["railway", "bus", "route"]
    for col in required_columns:
        assert col in transit.columns

    # Check individual counts
    correct_counts = {"railway": 1456, "route": 824, "bus": 79}

    for col in required_columns:
        cnt = len(transit[col].dropna())
        correct = correct_counts[col]
        assert cnt == correct, (
            f"Incorrect count for {col}. " f"Should have {correct}, found {cnt}."
        )

    # Ensure that the data contains only data specified in the filters
    unique_route = transit["route"].unique()
    for v in unique_route:
        if v is None:
            continue
        elif str(v) == "nan":
            continue
        assert v in routes

    unique_rails = transit["railway"].unique()
    for v in unique_rails:
        if v is None:
            continue
        elif str(v) == "nan":
            continue
        assert v in rails

    unique_bus = transit["bus"].unique()
    for v in unique_bus:
        if v is None:
            continue
        elif str(v) == "nan":
            continue

        assert v in bus

    assert isinstance(transit, GeoDataFrame)
    assert len(transit) == 2357

    # When using custom filters all records should have a value
    # at least on one of the attributes specified in the custom_filter
    selected = transit[required_columns]
    # Try dropping out rows with NaNs on all columns
    no_nans = selected.dropna(subset=required_columns, how="all")
    assert selected.shape == no_nans.shape


def test_reading_custom_from_area_having_none(helsinki_pbf):
    from pyrosm import OSM

    # Bounding box for area that does not have any data
    bbox = [24.940514, 60.173849, 24.942, 60.175892]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)

    # The tool should warn if no buildings were found
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_data_by_custom_criteria({"highway": ["primary"]})
        # Check the warning text
        if "could not find any OSM data" in str(w):
            pass

    # Result should be None
    assert gdf is None


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_data_by_custom_criteria({"highway": True})
    extra_col = "wikidata"
    extra = osm.get_data_by_custom_criteria(
        {"highway": True}, extra_attributes=[extra_col]
    )

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1] + 1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)


def test_using_multiple_filters(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_data_by_custom_criteria({"shop": ["alcohol"], "amenity": ["pub"]})

    # shop and amenity columns should only contain alcohol and pub as requested
    # (in addition to None values)
    shop = gdf["shop"].unique().tolist()
    shop = [item for item in shop if isinstance(item, str)]
    amenity = gdf["amenity"].unique().tolist()
    amenity = [item for item in amenity if isinstance(item, str)]

    assert isinstance(gdf, GeoDataFrame)
    assert shop == ["alcohol"]
    assert amenity == ["pub"]
    assert gdf.shape == (59, 33)


def test_using_two_level_custom_filter(helsinki_region_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=helsinki_region_pbf)
    osm_keys = ["building"]
    custom_filter = {"amenity": ["school"]}
    gdf = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter, osm_keys_to_keep=osm_keys
    )

    assert gdf.shape == (72, 26)

    # Now 'building' and 'amenity' should not have NaNs
    assert not gdf["building"].hasnans
    assert not gdf["amenity"].hasnans


def test_exclude_filtering_nodes_and_relations(helsinki_pbf):
    from pyrosm import OSM

    # Initialize the reader
    osm = OSM(helsinki_pbf)
    custom_filter = {"amenity": ["library"]}

    gdf = osm.get_data_by_custom_criteria(
        custom_filter,
        filter_type="exclude",
    )
    assert gdf.shape == (1083, 38)
    assert "library" not in gdf["amenity"].unique().tolist()

    # There should be nodes, ways and relations
    assert gdf["osm_type"].unique().tolist() == ["node", "way", "relation"]

    # Test other way around
    gdf = osm.get_data_by_custom_criteria(
        custom_filter,
        filter_type="keep",
    )
    assert gdf.shape == (7, 24)
    assert gdf["amenity"].unique().tolist() == ["library"]

    # There should be nodes and ways (no relations)
    assert gdf["osm_type"].unique().tolist() == ["node", "way"]


def test_custom_criteria_keep_true_valued_key(helsinki_pbf):
    """#272 — a True-valued custom_filter key keeps every feature carrying that
    key, across nodes/ways/relations."""
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(helsinki_pbf)
    gdf = osm.get_data_by_custom_criteria({"amenity": True}, filter_type="keep")
    assert isinstance(gdf, GeoDataFrame)
    assert "amenity" in gdf.columns
    # Every kept feature actually has an amenity tag value.
    assert gdf["amenity"].notna().all()
    assert set(gdf["osm_type"]) == {"node", "way", "relation"}
    assert len(gdf) == 1090


def test_custom_criteria_keep_specific_values(helsinki_pbf):
    """#272 — keep filtering by explicit tag values returns only those values."""
    from pyrosm import OSM

    osm = OSM(helsinki_pbf)
    gdf = osm.get_data_by_custom_criteria(
        {"amenity": ["restaurant", "cafe"]}, filter_type="keep"
    )
    assert set(gdf["amenity"].dropna()) == {"restaurant", "cafe"}
    assert len(gdf) == 303


def test_custom_criteria_exclude_removes_matching_values(helsinki_pbf):
    """#272 — exclude filtering drops features whose tag matches the listed
    values while keeping the rest of that key."""
    from pyrosm import OSM

    osm = OSM(helsinki_pbf)
    kept_all = osm.get_data_by_custom_criteria({"amenity": True}, filter_type="keep")
    excluded = osm.get_data_by_custom_criteria(
        {"amenity": ["restaurant", "cafe"]},
        osm_keys_to_keep="amenity",
        filter_type="exclude",
    )
    excluded_values = set(excluded["amenity"].dropna())
    assert "restaurant" not in excluded_values
    assert "cafe" not in excluded_values
    # Exclude keeps strictly fewer amenity features than keep-all.
    assert 0 < len(excluded) < len(kept_all)


def test_custom_criteria_element_toggle_keeps_only_nodes(helsinki_pbf):
    """#272 — keep_ways/keep_relations=False restrict the result to node
    features (filter_node_indices path)."""
    from pyrosm import OSM

    osm = OSM(helsinki_pbf)
    nodes_only = osm.get_data_by_custom_criteria(
        {"amenity": True},
        filter_type="keep",
        keep_ways=False,
        keep_relations=False,
    )
    assert set(nodes_only["osm_type"]) == {"node"}
    assert len(nodes_only) == 1006


def test_get_all_data_with_custom_filter_none(test_pbf):
    """#113 — custom_filter=None returns every tagged element without having to
    enumerate keys: tagged nodes, ways, and relations across all geometry types."""
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_data_by_custom_criteria(custom_filter=None)

    assert isinstance(gdf, GeoDataFrame)
    assert len(gdf) > 0
    assert set(gdf["osm_type"].unique()) == {"node", "way", "relation"}
    assert {"Point", "LineString", "Polygon"}.issubset(set(gdf.geom_type.unique()))
    # Calling with no arguments is equivalent
    assert len(OSM(filepath=test_pbf).get_data_by_custom_criteria()) == len(gdf)
    # filter_type is ignored in this mode -- even an invalid value is accepted
    assert len(
        OSM(filepath=test_pbf).get_data_by_custom_criteria(
            custom_filter=None, filter_type="ignored-value"
        )
    ) == len(gdf)
    # Common keys are materialised as columns; the rest live in the JSON 'tags'
    assert "building" in gdf.columns and "highway" in gdf.columns
    assert gdf["tags"].notna().any()


def test_get_all_data_is_superset_of_single_key(test_pbf):
    """The unfiltered result contains every element a single-key query returns."""
    from pyrosm import OSM

    everything = OSM(filepath=test_pbf).get_data_by_custom_criteria(custom_filter=None)
    buildings = OSM(filepath=test_pbf).get_data_by_custom_criteria(
        custom_filter={"building": True}
    )
    assert len(buildings) > 0
    assert set(buildings["id"]).issubset(set(everything["id"]))


def test_get_all_data_element_toggles(test_pbf):
    """keep_nodes / keep_ways / keep_relations scope the unfiltered result."""
    from pyrosm import OSM

    no_nodes = OSM(filepath=test_pbf).get_data_by_custom_criteria(
        custom_filter=None, keep_nodes=False
    )
    assert "node" not in set(no_nodes["osm_type"].unique())
    assert "Point" not in set(no_nodes.geom_type)

    no_ways = OSM(filepath=test_pbf).get_data_by_custom_criteria(
        custom_filter=None, keep_ways=False, keep_relations=False
    )
    assert "way" not in set(no_ways["osm_type"].unique())


def test_get_all_data_respects_keep_metadata(test_pbf):
    """The all-data path honours the existing keep_metadata=False option."""
    from pyrosm import OSM

    gdf = OSM(filepath=test_pbf, keep_metadata=False).get_data_by_custom_criteria(
        custom_filter=None
    )
    for col in ("version", "timestamp", "changeset"):
        assert col not in gdf.columns


def test_get_all_data_drops_untagged_ways(test_pbf):
    """Every way in the unfiltered result carries at least one OSM tag (as a column
    or in the JSON 'tags'); untagged standalone ways are not included."""
    import json
    import pandas as pd
    from pyrosm import OSM

    gdf = OSM(filepath=test_pbf).get_data_by_custom_criteria(custom_filter=None)
    structural = {
        "id",
        "version",
        "timestamp",
        "visible",
        "nodes",
        "changeset",
        "geometry",
        "osm_type",
        "tags",
        "length",
    }
    tag_cols = [c for c in gdf.columns if c not in structural]
    ways = gdf[gdf["osm_type"] == "way"]
    assert len(ways) > 0
    for _, row in ways.iterrows():
        has_column_tag = any(
            not (isinstance(row[c], float) and pd.isna(row[c])) and row[c] is not None
            for c in tag_cols
        )
        raw = row.get("tags")
        has_json_tag = (
            isinstance(raw, str) and raw not in ("", "{}") and bool(json.loads(raw))
        )
        assert has_column_tag or has_json_tag


def test_get_all_data_empty_area_returns_none(test_pbf):
    """An extract with no data still warns and returns None under custom_filter=None."""
    from pyrosm import OSM

    # Bounding box far outside the data extent (open ocean) -> nothing matches
    osm = OSM(filepath=test_pbf, bounding_box=[0.0, 0.0, 0.01, 0.01])
    with pytest.warns(UserWarning):
        gdf = osm.get_data_by_custom_criteria(custom_filter=None)
    assert gdf is None


def test_keep_all_drops_untagged_standalone_way():
    """#113 — the keep_all way gate drops a standalone way carrying only structural
    fields (no OSM tag) while keeping a tagged way. test.pbf has no untagged
    standalone ways, so this exercises that branch with synthetic records."""
    from pyrosm.data_manager import _get_osm_ways_and_relations

    tagged = {
        "id": 1,
        "version": 1,
        "timestamp": 0,
        "visible": True,
        "nodes": [1, 2, 3],
        "highway": "primary",
    }
    untagged = {
        "id": 2,
        "version": 1,
        "timestamp": 0,
        "visible": True,
        "nodes": [4, 5, 6],
    }
    # _get_osm_ways_and_relations(way_records, relations, osm_keys, tags_as_columns,
    #   data_filter, filter_type, keep_metadata, relation_member_ways, keep_all)
    ways, relation_ways, relations = _get_osm_ways_and_relations(
        [tagged, untagged], None, [], [], {}, "keep", True, None, True
    )
    assert list(ways["id"]) == [1]
    assert relation_ways is None
    assert relations is None


def test_custom_filter_key_outside_conf_uses_key_as_column(test_pbf):
    """A custom_filter key with no predefined Conf.tags entry still works: the
    getattr lookup falls through and the key itself becomes the column."""
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_data_by_custom_criteria(custom_filter={"source": True})
    assert gdf is not None
    assert "source" in gdf.columns
    assert gdf["source"].notna().any()


def test_custom_criteria_non_string_filter_type_raises(test_pbf):
    """A non-string filter_type is rejected by the isinstance guard."""
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    with pytest.raises(ValueError, match="should be either"):
        osm.get_data_by_custom_criteria(
            custom_filter={"building": True}, filter_type=123
        )
