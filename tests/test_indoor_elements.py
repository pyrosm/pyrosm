import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import shapely

from pyrosm import OSM, get_data
from pyrosm.indoor import PLATFORM_FILTER, is_indoor_element
from pyrosm.levels import level_list
from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _keys(frame):
    """The ``(osm_type, id)`` of each row of ``frame``."""
    return set(zip(frame["osm_type"], frame["id"]))


def _tags(frame):
    """The tags that write_pbf writes for each row of ``frame``, by ``(osm_type,
    id)``."""
    tags = _frame_tags(frame, _frame_tag_columns(frame))
    return dict(zip(zip(frame["osm_type"], frame["id"]), tags))


def test_get_indoor_elements_of_helsinki(helsinki_pbf):
    """The indoor elements of central Helsinki: every element with an indoor tag,
    every entrance and lift, every platform, every bus and tram stop (none of them
    is a stop position here), and the shops and amenities that have a level, each
    once; ``levels`` stands next to ``level`` and lists its levels."""
    osm = OSM(helsinki_pbf)
    elements = osm.get_indoor_elements()
    assert len(elements) == 823 and elements.crs.to_epsg() == 4326
    assert Counter(elements["osm_type"]) == {"node": 689, "way": 133, "relation": 1}
    assert Counter(elements["indoor"].dropna()) == {"yes": 22, "room": 2, "wall": 2}
    keys = _keys(elements)
    assert len(keys) == len(elements)

    by_reason = set()
    for reason in ({"indoor": True}, {"entrance": True}, {"highway": ["elevator"]}):
        by_reason |= _keys(osm.get_data_by_custom_criteria(reason))
    platforms = osm.get_data_by_custom_criteria(PLATFORM_FILTER)
    assert Counter(platforms["osm_type"]) == {"node": 12, "way": 107, "relation": 1}
    by_reason |= _keys(platforms)
    stops = osm.get_data_by_custom_criteria(
        {"highway": ["bus_stop"], "railway": ["tram_stop"]},
        extra_attributes=["public_transport"],
    )
    assert Counter(stops["public_transport"].fillna("")) == {"": 116, "platform": 12}
    by_reason |= _keys(stops)
    places = osm.get_data_by_custom_criteria(
        {"shop": True, "amenity": True}, extra_attributes=["level"]
    )
    assert len(places) == 1601
    by_reason |= _keys(places[places["level"].notna()])
    assert keys == by_reason

    columns = list(elements.columns)
    assert columns.index("levels") == columns.index("level") + 1
    assert elements["levels"].notna().sum() == elements["level"].notna().sum() == 231
    expected = [level_list(level) or "" for level in elements["level"]]
    assert elements["levels"].fillna("").tolist() == expected


def test_get_indoor_elements_is_the_same_with_both_readers(helsinki_pbf):
    """The in-memory reader gives the rows, columns and geometries of the out-of-core
    reader, also with the walking areas."""
    readers = [OSM(helsinki_pbf, engine=name) for name in ("out_of_core", "in_memory")]
    for walking_areas in (False, True):
        ours, theirs = (osm.get_indoor_elements(walking_areas) for osm in readers)
        pd.testing.assert_frame_equal(
            pd.DataFrame(ours.drop(columns="geometry")),
            pd.DataFrame(theirs.drop(columns="geometry")),
        )
        assert shapely.equals(ours.geometry, theirs.geometry).all()


def test_get_indoor_elements_with_walking_areas(helsinki_pbf):
    """With walking_areas the walking areas follow the indoor elements, in the same
    columns: polygons with a highway tag and no indoor tag. Every area that the
    indoor network draws lines across is among the rows. With fewer tag columns the
    rows and the tags of each element are the same."""
    osm = OSM(helsinki_pbf)
    elements = osm.get_indoor_elements()
    both = osm.get_indoor_elements(walking_areas=True)
    assert list(both.columns) == list(elements.columns)
    assert both["id"].tolist()[: len(elements)] == elements["id"].tolist()
    areas = both.iloc[len(elements) :]
    assert Counter(areas["osm_type"]) == {"way": 51, "relation": 22}
    assert set(areas.geom_type) == {"Polygon", "MultiPolygon"}
    assert areas["highway"].notna().all() and areas["indoor"].isna().all()
    assert len(_keys(both)) == len(both)

    nodes, edges = osm.get_network(
        "walking", nodes=True, indoor_elements=True, walking_areas=True
    )
    crossed = _keys(edges[edges["walking_area"]])
    assert len(crossed) == 113 and crossed <= _keys(both)
    assert len(crossed & _keys(areas)) == 51

    narrow = osm.get_indoor_elements(walking_areas=True, tags_to_keep=["name"])
    assert "level" not in narrow.columns and "area" in narrow.columns
    assert _tags(narrow) == _tags(both)


def test_get_indoor_elements_where_there_are_few_or_none(test_pbf, helsinki_pbf):
    """The small test file has one entrance, 36 bus stops and no walking area. A box
    that holds a square and no indoor element gives the square with walking_areas and
    nothing without. A read that finds nothing warns and returns None. Without a level
    column, levels stands before the geometry. walking_areas is a boolean, and the
    OSM tag levels cannot be asked for as a column."""
    osm = OSM(test_pbf)
    for walking_areas in (False, True):
        elements = osm.get_indoor_elements(walking_areas=walking_areas)
        assert Counter(elements["highway"].fillna("")) == {"bus_stop": 36, "": 1}
        entrance = elements[elements["entrance"].notna()]
        assert entrance[["id", "entrance"]].values.tolist() == [[1909030939, "main"]]
        assert elements["levels"].isna().all()
        columns = list(elements.columns)
        assert "level" not in columns
        assert columns.index("levels") == columns.index("geometry") - 1

    square = OSM(helsinki_pbf, bounding_box=[24.94964, 60.17827, 24.9499, 60.1786])
    with pytest.warns(UserWarning, match="Could not find any indoor elements"):
        assert square.get_indoor_elements() is None
    areas = square.get_indoor_elements(walking_areas=True)
    assert (areas["osm_type"].tolist(), areas["id"].tolist()) == (["way"], [22956114])
    assert areas["levels"].isna().all()

    empty = OSM(test_pbf, bounding_box=[26.94, 60.52, 26.9401, 60.5201])
    for walking_areas in (False, True):
        with pytest.warns(UserWarning, match="Could not find any indoor elements"):
            assert empty.get_indoor_elements(walking_areas=walking_areas) is None
    with pytest.raises(ValueError, match="'walking_areas' should be boolean"):
        osm.get_indoor_elements(walking_areas="yes")
    for asked in ({"extra_attributes": ["levels"]}, {"tags_to_keep": ["levels"]}):
        with pytest.raises(ValueError, match="computes the column 'levels'"):
            osm.get_indoor_elements(**asked)


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_walking_areas_without_the_member_ways_of_indoor_areas(engine, tmp_path):
    """A walking area that an indoor multipolygon takes in as its outline, such as a
    concourse mapped both as a pedestrian area and as an indoor area with a hole, is
    not a row of its own, as in the other reads; a walking area in the hole (an inner
    way) and one beside the area are."""
    from pyrosm.pbf_export import write_pbf_from_records

    rings = {  # way id: (west, south, east, north)
        20: (24.9000, 60.1000, 24.9010, 60.1005),
        21: (24.9004, 60.1002, 24.9006, 60.1003),
        22: (24.9020, 60.1000, 24.9030, 60.1005),
    }
    lon, lat, ways = [], [], []
    for way, (west, south, east, north) in rings.items():
        first = len(lon) + 1
        lon += [west, east, east, west]
        lat += [south, south, north, north]
        refs = [first, first + 1, first + 2, first + 3, first]
        tags = {"highway": "pedestrian", "area": "yes"}
        ways.append(
            {"id": way, "refs": refs, "version": 1, "timestamp": 1600000000, "tags": tags}
        )
    nodes = {
        "id": np.arange(1, len(lon) + 1, dtype=np.int64),
        "lon": np.array(lon),
        "lat": np.array(lat),
        "version": np.ones(len(lon), dtype=np.int64),
        "timestamp": np.full(len(lon), 1600000000, dtype=np.int64),
        "changeset": np.zeros(len(lon), dtype=np.int64),
        "tags": [None] * len(lon),
    }
    concourse = {
        "id": 30,
        "members": [("way", 20, "outer"), ("way", 21, "inner")],
        "version": 1,
        "timestamp": 1600000000,
        "changeset": 0,
        "tags": {"type": "multipolygon", "indoor": "area", "level": "-1"},
    }
    path = write_pbf_from_records(
        nodes,
        ways,
        [concourse],
        str(tmp_path / "concourse.osm.pbf"),
        (24.9, 60.1, 24.91, 60.11),
    )
    osm = OSM(path, engine=engine)
    elements = osm.get_indoor_elements()
    assert _keys(elements) == {("relation", 30)}
    assert len(elements.geometry.iloc[0].interiors) == 1
    both = osm.get_indoor_elements(walking_areas=True)
    rows = list(zip(both["osm_type"], both["id"], both["levels"].fillna("")))
    assert rows == [("relation", 30, "-1"), ("way", 21, ""), ("way", 22, "")]


def test_write_pbf_of_indoor_elements(test_pbf, helsinki_pbf, tmp_path):
    """write_pbf does not write the computed ``levels`` column of the indoor elements as
    a tag, also after rows were picked and a level was edited, and warns when elements
    lose a tag they were read by: a shop its level, a walking area its highway tag. In
    a frame without the note of this read, such as the buildings, ``levels`` is the
    OSM tag of that name and is written."""
    osm = OSM(helsinki_pbf)
    elements = osm.get_indoor_elements()
    shops = elements[elements["levels"].notna() & elements["shop"].notna()].head(5)
    edited = shops.assign(level="7")
    assert "levels" not in _frame_tag_columns(edited)
    osm.write_pbf(edited, tmp_path / "shops.osm.pbf", subset_only=True)
    written = OSM(tmp_path / "shops.osm.pbf").get_data_by_custom_criteria(
        {"level": True}
    )
    assert _keys(written) == _keys(shops) and set(written["level"]) == {"7"}
    assert not any("levels" in json.loads(tags) for tags in written["tags"])

    square = osm.get_indoor_elements(walking_areas=True).tail(1)
    lost = r"%s would no longer be found by get_indoor_elements\(\): their '%s' tag"
    for frame, count, tag in (
        (shops.drop(columns="level"), "5 elements", "level"),
        (square.drop(columns="highway"), "1 element", "highway"),
    ):
        with pytest.warns(UserWarning, match=lost % (count, tag)):
            osm.write_pbf(frame, tmp_path / "lost.osm.pbf", subset_only=True)
    unnoted = edited.copy()
    unnoted.attrs.clear()
    assert "levels" in _frame_tag_columns(unnoted)

    small = OSM(test_pbf)
    buildings = small.get_buildings().head(3).assign(levels=["3", "0;1", None])
    small.write_pbf(buildings, tmp_path / "buildings.osm.pbf", subset_only=True)
    written = OSM(tmp_path / "buildings.osm.pbf").get_buildings()
    assert dict(zip(written["id"], written["levels"].fillna(""))) == dict(
        zip(buildings["id"], ["3", "0;1", ""])
    )


@pytest.mark.parametrize(
    "level, levels",
    [
        ("0-2", "0;1;2"),
        ("-2;0", "-2;0"),
        ("0.5-2", "0.5;1;2"),
        (" 2 ; -1 ", "-1;2"),
        ("1;1;0-1", "0;1"),
        ("-4--2", "-4;-3;-2"),
        ("0-500;3", "3"),
        ("0.0000001;9007199254740993", "0.0000001;9007199254740993"),
        ("01.50;-0.0", "0;1.5"),
        ("ground", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_level_list(level, levels):
    """A level value written out: the ends of each range and the whole levels between
    them, each once, lowest first; a range over more than 200 levels is left out, and
    a value that names no level gives None. Every digit of a level is kept, without
    the zeros that say nothing and the sign of zero."""
    assert level_list(level) == levels


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"indoor": "room"}, True),
        ({"room": "stairs", "building": "roof"}, True),
        ({"room": "office"}, False),
        ({"door": "hinged"}, True),
        ({"entrance": "main"}, True),
        ({"highway": "elevator"}, True),
        ({"railway": "platform"}, True),
        ({"public_transport": "platform"}, True),
        ({"highway": "bus_stop"}, True),
        ({"railway": "tram_stop"}, True),
        ({"railway": "tram_stop", "public_transport": "stop_position"}, False),
        ({"shop": "kiosk", "level": "-1"}, True),
        ({"amenity": "cafe", "level": "ground"}, False),
        ({"amenity": "cafe"}, False),
        ({"highway": "pedestrian", "level": "0"}, False),
    ],
)
def test_is_indoor_element(tags, expected):
    """An indoor element has an indoor tag, is a room tagged as a walkway, a door, an
    entrance, a lift, a platform or a bus or tram stop other than a stop position on
    the track, or is a shop or amenity whose level tag names a level."""
    assert is_indoor_element(tags) is expected
