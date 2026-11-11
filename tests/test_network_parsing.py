from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyrosm import get_data


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def helsinki_history_pbf():
    pbf_path = get_data("helsinki_test_history_pbf")
    return pbf_path


@pytest.fixture
def test_output_dir():
    import tempfile

    return str(Path(tempfile.gettempdir()) / "pyrosm_test_results")


# Columns of every network frame
BASE_COLUMNS = ["id", "geometry", "tags", "osm_type", "length"]
WALKING_COLUMNS = [
    "bridge",
    "foot",
    "highway",
    "lanes",
    "lit",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "service",
    "surface",
]
DRIVING_COLUMNS = [
    "bridge",
    "highway",
    "int_ref",
    "lanes",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "surface",
]
CYCLING_COLUMNS = [
    "bicycle",
    "bridge",
    "foot",
    "highway",
    "lanes",
    "lit",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "service",
    "surface",
    "tunnel",
]


@pytest.mark.parametrize(
    "network_type, shape, columns, excluded_highways",
    [
        ("walking", (318, 21), WALKING_COLUMNS, ["motorway"]),
        ("driving", (171, 17), DRIVING_COLUMNS, ["footway", "path"]),
        (
            "driving+service",
            (199, 19),
            DRIVING_COLUMNS + ["lit", "service"],
            ["footway", "path"],
        ),
        ("cycling", (289, 21), CYCLING_COLUMNS, ["motorway", "motorway_link"]),
        ("all", (331, 23), CYCLING_COLUMNS + ["access"], []),
    ],
    ids=["walking", "driving", "driving+service", "cycling", "all"],
)
def test_filter_network_by_type(
    test_pbf, network_type, shape, columns, excluded_highways
):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString

    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type=network_type)

    assert isinstance(gdf.loc[0, "geometry"], LineString)
    assert isinstance(gdf, GeoDataFrame)
    assert gdf.shape == shape
    for col in columns + BASE_COLUMNS:
        assert col in gdf.columns

    # Highway types the network type leaves out by default
    highways = gdf["highway"].unique()
    for highway in excluded_highways:
        assert highway not in highways


def test_saving_network_to_shapefile(test_pbf, test_output_dir):
    from pyrosm import OSM
    import geopandas as gpd
    import shutil
    import numpy as np

    if not Path(test_output_dir).exists():
        Path(test_output_dir).mkdir(parents=True)

    temp_path = str(Path(test_output_dir) / "pyrosm_test.shp")
    osm = OSM(filepath=test_pbf)
    gdf = osm.get_network(network_type="cycling")
    gdf.to_file(temp_path)

    # Ensure it can be read and matches with original one
    gdf2 = gpd.read_file(temp_path)

    cols = gdf.columns
    for col in cols:
        # Geometry col might contain different types of geoms
        # (due to saving MultiLineGeometries which might be read as a "single")
        if col == "geometry":
            continue

        try:
            assert gdf[col].tolist() == gdf2[col].tolist()
        except AssertionError:
            # Skip if the column contains only None values (to avoid conflict between None and np.nan)
            if gdf[col].unique().tolist() == [None]:
                continue

    # Clean up
    shutil.rmtree(test_output_dir)


@pytest.mark.parametrize("as_polygon", [False, True], ids=["list", "shapely box"])
def test_parse_network_with_bbox(test_pbf, as_polygon):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString, box

    bounds = [26.94, 60.525, 26.96, 60.535]
    bounding_box = box(*bounds) if as_polygon else bounds
    osm = OSM(filepath=test_pbf, bounding_box=bounding_box)
    gdf = osm.get_network()

    assert isinstance(gdf.loc[0, "geometry"], LineString)
    assert isinstance(gdf, GeoDataFrame)

    # Test shape (#236: edges crossing the bbox edge are now returned complete
    # rather than cut, so boundary-crossing edges are included)
    assert gdf.shape == (109, 21)
    for col in WALKING_COLUMNS + BASE_COLUMNS:
        assert col in gdf.columns

    # Should not include 'motorway' ways by default
    assert "motorway" not in gdf["highway"].unique()

    # Every returned edge intersects the bounding box. Edges that straddle the edge
    # are returned complete (#236), so the result may extend slightly beyond the
    # filter; clip with GeoPandas afterwards if exact bounds are required.
    assert gdf.intersects(box(*bounds)).all()


def test_passing_incorrect_bounding_box(test_pbf):
    from pyrosm import OSM

    wrong_format = "[26.94, 60.525, 26.96, 60.535]"
    try:
        osm = OSM(filepath=test_pbf, bounding_box=wrong_format)
    except ValueError as e:
        if "bounding_box should be" in str(e):
            pass
        else:
            raise (e)
    except Exception as e:
        raise e


def test_passing_incorrect_net_type(test_pbf):
    from pyrosm import OSM

    osm = OSM(filepath=test_pbf)
    try:
        osm.get_network("wrong_network")
    except ValueError as e:
        if "'network_type' should be one of the following" in str(e):
            pass
        else:
            raise (e)
    except Exception as e:
        raise e

    try:
        osm.get_network(42)
    except ValueError as e:
        if "'network_type' should be one of the following" in str(e):
            pass
        else:
            raise (e)
    except Exception as e:
        raise e


def test_reading_network_from_area_without_data(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # Bounding box for area that does not have any data
    bbox = [24.940514, 60.173849, 24.942, 60.175892]

    osm = OSM(filepath=helsinki_pbf, bounding_box=bbox)

    # The tool should warn if no buildings were found
    with pytest.warns(UserWarning) as w:
        gdf = osm.get_network()
        # Check the warning text
        if "could not find any network data" in str(w):
            pass

    # Result should be None
    assert gdf is None


def test_adding_extra_attribute(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_pbf)
    gdf = osm.get_network()
    extra_col = "wikidata"
    extra = osm.get_network(extra_attributes=[extra_col])

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1] + 1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)


@pytest.mark.parametrize(
    "bounding_box, edges_shape, nodes_shape",
    [
        (None, (1532, 23), (1397, 9)),
        # #236: edges crossing the box edge are returned complete, with their nodes.
        # #199: the nodes frame keeps every endpoint of the kept edges, also those
        # just outside the box, so the nodes/edges pair stays consistent for graph
        # export.
        ([26.94, 60.525, 26.96, 60.535], (484, 23), (451, 9)),
    ],
    ids=["whole file", "bounding box"],
)
def test_getting_nodes_and_edges(test_pbf, bounding_box, edges_shape, nodes_shape):
    from pyrosm import OSM
    from geopandas import GeoDataFrame
    from shapely.geometry import Point, LineString

    osm = OSM(filepath=test_pbf, bounding_box=bounding_box)

    nodes, edges = osm.get_network(nodes=True)
    nodes = nodes.reset_index(drop=True)

    assert isinstance(edges, GeoDataFrame)
    assert isinstance(edges.loc[0, "geometry"], LineString)

    assert isinstance(nodes, GeoDataFrame)
    assert isinstance(nodes.loc[0, "geometry"], Point)

    assert edges.shape == edges_shape
    assert nodes.shape == nodes_shape

    # Edges should have "u" and "v" columns
    for col in ["u", "v", "length"]:
        assert col in edges.columns

    # Nodes should have (at least) "id", "lat", and "lon" columns
    for col in ["id", "lat", "lon"]:
        assert col in nodes.columns


def test_reading_network_from_osh(helsinki_history_pbf):
    from pyrosm import OSM
    from pyrosm.utils import datetime_to_unix_time
    import pandas as pd
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString

    timestamp = "2010-01-01"
    dt = pd.to_datetime(timestamp, utc=True)
    unix_time = datetime_to_unix_time(dt)

    osm = OSM(filepath=helsinki_history_pbf)
    gdf = osm.get_network(timestamp=timestamp)

    assert isinstance(gdf, GeoDataFrame)
    assert isinstance(gdf.loc[0, "geometry"], LineString)
    assert gdf.shape == (179, 22)

    required_cols = ["highway", "id", "timestamp", "version", "geometry"]

    for col in required_cols:
        assert col in gdf.columns

    # None of the features should be newer than the given timestamp
    assert gdf["timestamp"].max() <= unix_time

    # There should be only a single version per id
    cnt = len(gdf)
    assert cnt == len(gdf.drop_duplicates(subset=["id"]))

    # Changing the timestamp should parse a different set of elements
    new_timestamp = "2015-01-01"
    dt2 = pd.to_datetime(new_timestamp, utc=True)
    unix_time2 = datetime_to_unix_time(dt2)

    gdf2 = osm.get_network(timestamp=new_timestamp)

    # Now the "current_timestamp" should have been updated
    assert osm._current_timestamp == unix_time2

    # Number of elements should (likely) be higher
    assert len(gdf2) > len(gdf)

    # There should be newer timestamps in the data
    assert gdf2["timestamp"].max() > unix_time

    # But is shouldn't be higher than the given timestamp
    assert gdf2["timestamp"].max() <= unix_time2

    # Test reading network with nodes
    n, e = osm.get_network(timestamp=new_timestamp, nodes=True)
    assert isinstance(n, GeoDataFrame)
    assert isinstance(e, GeoDataFrame)
    # Timestamp shouldn't be higher than the given timestamp
    assert e["timestamp"].max() <= unix_time2
    assert n["timestamp"].max() <= unix_time2


def test_osh_network_does_not_leak_empty_tag_columns(helsinki_history_pbf):
    """Regression test for the pandas 3.0 string-dtype / Copy-on-Write change.

    Missing tag values became ``pd.NA`` instead of ``None``, so the OSH
    way-record cleaner stopped stripping empty tags. Every way record then
    carried the union of all tag keys (NA-valued), inflating the network
    output from (210, 25) to (383, 27) and admitting extra ways through the
    highway filter.
    """
    from pyrosm import OSM

    osm = OSM(filepath=helsinki_history_pbf)
    gdf = osm.get_network(timestamp="2010-01-01")

    # An all-NA column is a "phantom" tag that the cleaner failed to strip.
    phantom = [c for c in gdf.columns if c != "geometry" and gdf[c].isna().all()]
    assert phantom == [], f"empty tag columns leaked into OSH output: {phantom}"

    # The column set must stay lean (22 with the ``visible`` column; 21 since the latest
    # version of each way is taken whole (#379); 24 before that, 25 before the network-filter fix and 27 with the
    # #248 NA-leak bug). The row count dropped by one with the OSMnx-parity filters
    # (#369), by one more with #379, as a way lost its highway tag before 2010, and by
    # 33 ways that were deleted before 2010. It grew by 8 cycle ways that the foot tag
    # opens to pedestrians.
    assert gdf.shape == (179, 22)


def _walking_area_frames():
    """Network nodes and edges around a hand-made walking area (a relation of two parts,
    the first a square with a hole), in units of 1e-4 degrees near Helsinki."""
    import geopandas as gpd
    from shapely.geometry import LineString, MultiPolygon, Polygon

    def lonlat(x, y):
        return 24.9 + x * 1e-4, 60.2 + y * 1e-4

    outline = [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (0, 4), (0, 3), (0, 2)]
    hole = [(1.5, 1.8), (2.5, 1.8), (2.5, 3.0), (1.5, 3.0)]
    second = [(6, 0), (7, 0), (7, 1), (6, 1)]
    u_shape = [(10, 0), (13, 0), (13, 3), (12, 3), (12, 1), (11, 1), (11, 3), (10, 3)]
    area = MultiPolygon(
        [
            Polygon([lonlat(*xy) for xy in outline], [[lonlat(*xy) for xy in hole]]),
            Polygon([lonlat(*xy) for xy in second]),
            Polygon([lonlat(*xy) for xy in u_shape]),
        ]
    )
    points = {
        1: (0, 2),  # A, a corner of the outline
        2: (4, 2),  # B
        22: (4, 2),  # a second node at B
        3: (2, 0),  # C
        4: (0.5, 3.5),  # D, the end of a way inside the area
        5: (0, 3),  # E
        6: (0, 4),  # F
        7: (4, 1),  # on the outline but not at a corner of it
        8: (1, 1),  # inside, on a way that goes on
        9: (6, 0),
        10: (7, 1),
        11: (10, 3),  # the tips of the U, which see each other only around its corners
        12: (13, 3),
    }
    nodes = pd.DataFrame(
        [(i, *lonlat(*xy)) for i, xy in points.items()], columns=["id", "lon", "lat"]
    )
    ends = [(4, 1), (8, 1), (8, 3), (7, 2), (9, 10), (5, 6)]
    edges = gpd.GeoDataFrame(
        {
            "id": range(len(ends)),
            "osm_type": "way",
            "highway": "footway",
            "tags": None,
            "u": [u for u, _ in ends],
            "v": [v for _, v in ends],
            "length": 1.0,
        },
        geometry=[
            LineString([lonlat(*points[u]), lonlat(*points[v])]) for u, v in ends
        ],
        crs="EPSG:4326",
    )
    areas = gpd.GeoDataFrame(
        {"id": [500], "osm_type": "relation", "highway": "pedestrian", "tags": None},
        geometry=[area],
        crs="EPSG:4326",
    )
    return nodes, edges, areas, lonlat


@pytest.mark.parametrize("capped", [False, True])
def test_walking_area_crossings(capped, monkeypatch):
    """The network nodes at corners of an area's outline or of its holes, and the ends of
    ways inside it, are joined in pairs along the shortest line inside the area, which
    bends at the corners of the hole; a pair whose line passes another of them, and the
    parts of a multipolygon between each other, get no edge. Over the point limit, the
    lines bend only at other entrances, so the tips of a U get no edge. Every other pair
    gets its line here, with no detour allowed (MAX_ROUTE_STRETCH 1)."""
    from pyrosm import walking_areas
    from pyrosm.distance import Unit, haversine

    nodes, edges, areas, lonlat = _walking_area_frames()
    monkeypatch.setattr(walking_areas, "MAX_ROUTE_STRETCH", 1.0)
    if capped:
        monkeypatch.setattr(walking_areas, "MAX_VISIBILITY_POINTS", 3)
    rows = walking_areas.crossing_rows(nodes, edges, areas)

    pairs = set(zip(rows["u"], rows["v"]))
    ends_at_b = {(1, 2), (1, 22), (2, 4), (22, 4), (2, 5), (22, 5), (2, 6), (22, 6)}
    others = {(1, 3), (1, 4), (1, 5), (2, 3), (22, 3), (3, 4), (3, 5), (3, 6)}
    others |= {(4, 5), (4, 6), (5, 6), (9, 10)}
    assert pairs == (others if capped else others | ends_at_b | {(11, 12)})
    assert list(rows.columns) == list(edges.columns)
    assert (rows["id"] == 500).all() and (rows["highway"] == "pedestrian").all()
    if capped:
        return
    bends = {(1, 2): [(0, 2), (1.5, 1.8), (2.5, 1.8), (4, 2)]}
    bends[(2, 4)] = [(4, 2), (2.5, 3.0), (0.5, 3.5)]
    bends[(11, 12)] = [(10, 3), (11, 1), (12, 1), (13, 3)]
    for (u, v), path in bends.items():
        row = rows[(rows["u"] == u) & (rows["v"] == v)].iloc[0]
        xy = [lonlat(*p) for p in path]
        np.testing.assert_allclose(row.geometry.coords, xy)
        lon, lat = np.array(xy).T
        pieces = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:], unit=Unit.METERS)
        assert row["length"] == pytest.approx(pieces.round(3).sum(), abs=1e-6)
    without_tags = walking_areas.crossing_rows(nodes, edges.drop(columns="tags"), areas)
    assert "tags" not in without_tags.columns
    shrunk = areas.assign(geometry=[areas.geometry.iloc[0].geoms[1].buffer(-1e-6)])
    assert walking_areas.crossing_rows(nodes, edges, shrunk) is None


@pytest.mark.parametrize("offset, direct", [(1, False), (2, True)])
def test_walking_areas_leave_out_lines_that_shorter_ones_cover(offset, direct):
    """A square walking area with entrances at two corners (nodes 1 and 3, 10 units
    apart) and a dead end inside it (node 2), ``offset`` units off the middle of the line
    between them: the line from 1 to 3 is left out when the two shorter lines through
    node 2 are at most 5 % longer (offset 1: 2 * sqrt(26) = 10.2), and kept otherwise
    (offset 2: 2 * sqrt(29) = 10.8)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    from pyrosm import walking_areas

    def lonlat(x, y):
        return x * 1e-4, y * 1e-4

    points = {1: (0, 0), 2: (offset, 5), 3: (0, 10), 4: (-5, 0), 5: (-5, 10), 6: (20, 5)}
    nodes = pd.DataFrame(
        [(i, *lonlat(*xy)) for i, xy in points.items()], columns=["id", "lon", "lat"]
    )
    ends = [(1, 4), (3, 5), (2, 6)]
    edges = gpd.GeoDataFrame(
        {"id": range(len(ends)), "osm_type": "way", "highway": "footway", "tags": None,
         "u": [u for u, _ in ends], "v": [v for _, v in ends], "length": 1.0},
        geometry=[LineString([lonlat(*points[u]), lonlat(*points[v])]) for u, v in ends],
        crs="EPSG:4326",
    )
    square = Polygon([lonlat(*xy) for xy in [(0, 0), (10, 0), (10, 10), (0, 10)]])
    areas = gpd.GeoDataFrame(
        {"id": [500], "osm_type": "way", "highway": "pedestrian", "tags": None},
        geometry=[square],
        crs="EPSG:4326",
    )
    rows = walking_areas.crossing_rows(nodes, edges, areas)
    pairs = set(map(tuple, map(sorted, zip(rows["u"], rows["v"]))))
    assert pairs == {(1, 2), (2, 3)} | ({(1, 3)} if direct else set())


@pytest.mark.parametrize(
    "end_tags, area_tags, joined",
    [
        ({}, {}, True),
        ({"level": "-1"}, {}, False),
        ({"tunnel": "yes"}, {}, False),
        ({"tunnel": "building_passage"}, {}, True),
        ({"layer": "1"}, {}, False),
        ({"level": "1"}, {"level": "0-2"}, True),
        ({"level": "-1;0"}, {}, True),
        ({"level": "-1-1"}, {}, True),
        ({"level": "-2"}, {"level": "-1"}, False),
        ({"layer": "-1", "tunnel": "yes"}, {"level": "-1"}, True),
        ({"level": "B1", "layer": "x"}, {}, True),
        ({"layer": "nan"}, {}, True),
    ],
)
def test_walking_areas_join_dead_ends_on_their_level(end_tags, area_tags, joined):
    """A dead end inside an area is joined only on the area's level: by overlapping
    level ranges when both have a level, else by the side of street level from the
    levels, layer and tunnel; no such tag (or a level or layer that is not a number) is
    street level."""
    from rapidjson import dumps

    from pyrosm import walking_areas

    nodes, edges, areas, _ = _walking_area_frames()
    edges.loc[0, "tags"] = dumps(end_tags)  # the way that ends at node 4, inside
    areas.loc[0, "tags"] = dumps(area_tags)
    rows = walking_areas.crossing_rows(nodes, edges, areas)
    assert ((rows["u"] == 4) | (rows["v"] == 4)).any() == joined


@pytest.mark.parametrize(
    "stair_tags, other_end_level, joined",
    [
        ({"level": "0;1"}, "1", False),
        ({"level": "0;1"}, "0", True),
        ({"level": "0;1", "incline": "up"}, None, False),
        ({"level": "0;1", "incline": "down"}, None, True),
        ({"level": "0;1"}, None, False),
    ],
)
def test_walking_areas_join_stair_ends_on_their_level(stair_tags, other_end_level, joined):
    """The dead end of stairs with a level range is joined to an area only on the level
    of that end: the other end of the range from a walkway at the stairs' other end, or
    the end that the stairs' incline gives; with neither, while both ends lie in areas
    of level 1, the stairs rise along their drawn direction, so their start is on 0."""
    from rapidjson import dumps

    from pyrosm import walking_areas
    from pyrosm.indoor import stair_way_ends

    nodes, edges, areas, _ = _walking_area_frames()
    edges.loc[0, "highway"] = "steps"  # the way from node 4, inside, to node 1
    edges.loc[0, "tags"] = dumps(stair_tags)
    if other_end_level is not None:
        edges.loc[1, "tags"] = dumps({"level": other_end_level})  # a way at node 1
    areas.loc[0, "tags"] = dumps({"level": "1"})
    rows = walking_areas.crossing_rows(nodes, edges, areas)
    assert ((rows["u"] == 4) | (rows["v"] == 4)).any() == joined
    # The way starts at node 4 and ends at node 1, with no node between them
    assert stair_way_ends(edges, [4]) == {4: (1, True, 0)}


STEPS = {"highway": "steps", "level": "0;1"}


@pytest.mark.parametrize(
    "way, way_tags, own_level, stacked, continues, node, entered",
    [
        (5, {"level": "1"}, None, False, False, 5, {501}),
        (5, {}, "1", False, False, 5, {501}),
        (5, STEPS, None, False, False, 5, {501}),
        (5, STEPS, None, False, True, 5, {501}),
        (5, {**STEPS, "incline": "up"}, None, False, False, 5, {500}),
        (5, {**STEPS, "level": "-1;0"}, None, True, False, 5, set()),
        (5, {**STEPS, "level": "-1;0"}, None, True, False, 6, {500}),
        (0, {**STEPS, "incline": "up"}, None, False, False, 1, set()),
        ((1, 2), STEPS, None, False, False, 3, {500}),
        ((1, 2), STEPS, None, False, False, 1, {500}),
        (5, STEPS, None, True, False, 5, {500}),
        (5, STEPS, None, True, False, 6, {501}),
        (0, {**STEPS, "incline": "up"}, None, True, False, 1, {501}),
        (5, {}, None, True, False, 5, {500, 501}),
    ],
)
def test_walking_areas_join_a_node_on_one_level(
    way, way_tags, own_level, stacked, continues, node, entered
):
    """A node that would enter areas on two levels (area 500 on level 0, area 501 on
    level 1, either around node 5 or on the same outline) enters only those on its
    level: its walkway's, else at an end of stairs with a level range the level of that
    end, else its own level tag. Such a stair end enters only areas on its level, also
    when it touches one area (node 1 on the outline of area 500). The end of such stairs
    in areas on both levels takes the other end of the range from the one level of the
    areas at its other end (also where a walkway goes on from there), else the low end at
    the start of the way and the high end at its end, as do the two ends of stairs that
    both lie in areas of one level (level=-1;0 in the stacked layout), unless the stairs
    have a node between their ends (ways 1 and 2 as one way through node 8), where they
    dip to the other level. A node without levels enters both."""
    import geopandas as gpd
    from rapidjson import dumps
    from shapely.geometry import box

    from pyrosm import walking_areas

    nodes, edges, areas, lonlat = _walking_area_frames()
    nodes.loc[len(nodes)] = [30, *lonlat(0.5, 3)]  # inside area 501 in both layouts
    edges.loc[len(edges)] = {**edges.iloc[1].to_dict(), "id": 30, "u": 30, "v": 2}
    edges.loc[len(edges) - 1, "tags"] = dumps({"level": "1"})
    if continues:  # a walkway without a level from node 6, the stairs' other end
        edges.loc[len(edges)] = {**edges.iloc[1].to_dict(), "id": 31, "u": 6, "v": 3}
    nodes["tags"] = [{"level": own_level} if i == node else None for i in nodes["id"]]
    rows = list(way) if isinstance(way, tuple) else [way]
    edges.loc[rows, "id"] = edges.loc[rows[0], "id"]
    edges.loc[rows, "highway"] = way_tags.get("highway", "footway")
    edges.loc[rows, "tags"] = dumps({k: v for k, v in way_tags.items() if k != "highway"})
    areas.loc[0, "tags"] = dumps({"level": "0"})
    small = box(*lonlat(-0.5, 2.6), *lonlat(0.8, 3.3))
    second = areas.assign(
        id=501,
        tags=dumps({"level": "1"}),
        geometry=[areas.geometry.iloc[0] if stacked else small],
    )
    areas = gpd.GeoDataFrame(pd.concat([areas, second], ignore_index=True), crs=4326)
    rows = walking_areas.crossing_rows(nodes, edges, areas)
    assert set(rows.loc[(rows["u"] == node) | (rows["v"] == node), "id"]) == entered


def test_walking_areas_leave_a_landing_of_two_stairs_open():
    """A node where two stair ways with level ranges meet (node 8, ways tagged 0;1 and
    1;2) takes no level from their drawn direction, whatever the order of the edges;
    the far ends of the two ways do."""
    from rapidjson import dumps

    from pyrosm.indoor import node_levels
    from pyrosm.walking_areas import _entrance_positions

    nodes, edges, _, _ = _walking_area_frames()
    edges.loc[[1, 2], "highway"] = "steps"
    edges.loc[1, "tags"] = dumps({"level": "0;1"})
    edges.loc[2, "tags"] = dumps({"level": "1;2"})
    ids = nodes["id"].to_numpy()
    for frame in (edges, edges.iloc[::-1]):
        _, open_ends, *_ = _entrance_positions(frame, ids, np.array([3]), node_levels(nodes))
        assert 8 not in open_ends and {1, 3} <= set(open_ends)


@pytest.mark.parametrize(
    "network_type, kept",
    [("walking", [1, 4, 8, 9]), ("cycling", [1, 8, 9]), ("driving", [])],
)
def test_walking_areas_pass_the_network_filter(network_type, kept, monkeypatch):
    """A walking area is a closed way tagged area=yes or a multipolygon relation whose
    tags the network's filter keeps. One built for another mode, without a tag of the
    network's mode, counts where its country lets the mode use it: cyclists may use foot
    ways in Norway, not in Finland."""
    import geopandas as gpd
    from shapely.geometry import LineString, box

    from pyrosm.data import countries
    from pyrosm.walking_areas import area_filter, select_areas

    # With the read's area unknown, each area is decided by its own country.
    monkeypatch.setattr(countries, "read_extent", lambda *args: None)
    helsinki = box(24.940, 60.170, 24.941, 60.171)
    oslo = box(10.750, 59.910, 10.751, 59.911)
    multipolygon, route = '{"type": "multipolygon"}', '{"type": "route"}'
    rows = [
        (1, "way", "pedestrian", "yes", None, None, None, helsinki),
        (2, "way", "pedestrian", None, None, None, None, helsinki),
        (3, "way", "footway", "yes", None, None, None, LineString([(24.94, 60.17)] * 2)),
        (4, "relation", "footway", None, None, None, multipolygon, helsinki),
        (5, "relation", "pedestrian", None, None, None, route, helsinki),
        (6, "way", "footway", "yes", "no", None, None, helsinki),
        (7, "way", "platform", "yes", None, None, None, helsinki),
        (8, "way", "footway", "yes", None, None, None, oslo),
        (9, "way", "footway", "yes", None, "yes", None, helsinki),
    ]
    columns = ["id", "osm_type", "highway", "area", "foot", "bicycle", "tags"]
    candidates = gpd.GeoDataFrame(
        [row[:-1] for row in rows],
        columns=columns,
        geometry=[row[-1] for row in rows],
        crs="EPSG:4326",
    )
    _, keep, rule = area_filter(
        network_type, None, None, "local", get_data("test_pbf"), None
    )
    assert select_areas(candidates, keep, rule)["id"].tolist() == kept


@pytest.mark.parametrize("case", ["walking", "driving", "footway areas", "node lists"])
def test_get_network_with_walking_areas(case):
    """walking_areas=True adds the edges across the walking areas that the network's
    filter keeps to the edges of the network, marked by the column walking_area, with the
    area's id, type and tags; both engines return the same frames."""
    from geopandas.testing import assert_geodataframe_equal

    from pyrosm import OSM

    kwargs = {"network_type": "walking"}
    if case == "driving":
        kwargs = {"network_type": "driving"}
    elif case == "footway areas":
        kwargs = {"custom_filter": {"highway": ["footway"]}}
    frames = []
    for engine in ("in_memory", "out_of_core"):
        osm = OSM(
            get_data("helsinki_pbf"), engine=engine, keep_node_info=case == "node lists"
        )
        frames.append(osm.get_network(nodes=True, walking_areas=True, **kwargs))
    assert_geodataframe_equal(frames[0][1], frames[1][1], check_dtype=False)
    _, plain = osm.get_network(nodes=True, **kwargs)
    nodes, edges = frames[1]

    crossing = edges[edges["walking_area"]]
    assert edges["walking_area"].dtype == bool
    assert len(edges) == len(plain) + len(crossing)
    assert crossing["u"].isin(nodes["id"]).all() and crossing["v"].isin(nodes["id"]).all()
    if case == "driving":
        assert crossing.empty
    elif case == "footway areas":
        assert len(crossing) > 0 and (crossing["highway"] == "footway").all()
    elif case == "node lists":
        assert crossing["nodes"].map(list).tolist() == [
            [u, v] for u, v in zip(crossing["u"], crossing["v"])
        ]
    else:
        import json

        import networkx as nx
        import shapely

        areas = osm.get_data_by_custom_criteria(
            {"highway": True}, keep_nodes=False
        ).set_index(["osm_type", "id"])
        outlines = areas.geometry.loc[pd.MultiIndex.from_frame(crossing[["osm_type", "id"]])]
        assert shapely.covers(
            shapely.buffer(outlines.to_numpy(), 1e-7), crossing.geometry.to_numpy()
        ).all()
        ways = crossing[crossing["osm_type"] == "way"]
        assert len(ways) > 0
        assert all(json.loads(tags)["area"] == "yes" for tags in ways["tags"])
        square = crossing[(crossing["osm_type"] == "relation") & (crossing["id"] == 2919118)]
        assert (square["name"] == "Rautatientori").all()
        assert (square["highway"] == "pedestrian").all()
        assert nx.is_connected(nx.Graph(list(zip(square["u"], square["v"]))))
        assert "walking_area" not in plain.columns


@pytest.mark.parametrize(
    "read, kwargs, message",
    [
        ("get_network", {"walking_areas": True}, "needs nodes=True"),
        (
            "get_network",
            {"walking_areas": "yes", "nodes": True},
            "'walking_areas' should be boolean",
        ),
        ("get_graph", {"walking_areas": None}, "'walking_areas' should be boolean"),
        ("get_network", {"indoor_elements": True}, "indoor_elements=True needs nodes"),
        (
            "get_network",
            {"indoor_elements": 1, "nodes": True},
            "'indoor_elements' should be boolean",
        ),
        (
            "get_network",
            {"network_type": "driving", "indoor_elements": True, "nodes": True},
            "not with 'driving'",
        ),
        ("get_graph", {"indoor_elements": "yes"}, "'indoor_elements' should be boolean"),
        (
            "get_graph",
            {"network_type": "cycling", "indoor_elements": True},
            "not with 'cycling'",
        ),
        ("get_network", {"escalator_slope": True}, "escalator_slope should be an angle"),
        ("get_graph", {"escalator_slope": 90}, "escalator_slope should be an angle"),
        ("get_network", {"escalator_slope": "30"}, "escalator_slope should be an angle"),
    ],
)
def test_network_reads_refuse_unusable_walking_areas(test_pbf, read, kwargs, message):
    """walking_areas and indoor_elements are booleans, need the network's nodes, and
    indoor_elements needs a walking network type or a custom filter; escalator_slope is
    None or an angle of at least 0 and below 90 degrees, not a boolean."""
    from pyrosm import OSM

    with pytest.raises(ValueError, match=message):
        getattr(OSM(test_pbf), read)(**kwargs)


@pytest.mark.parametrize("walking_areas", [False, True])
def test_get_network_with_indoor_elements(walking_areas):
    """indoor_elements=True adds the platforms, also those the walking network leaves
    out (way 35744431, railway=platform, has no highway tag), once each and with their
    nodes, crosses the platform areas (way 8137136), and adds u_level, v_level and
    vertical (way 23653232 is stairs, a platform is neither), and splits lift node
    2039713549 per level; both engines return the same frames: the escalators
    of ways 18378805 (level=-3;-2) and 655180399 (level=0;2) run between the levels at
    their ends; without the option the columns and platforms are not there. A read
    whose filter keeps no other way returns the platforms, and with walking_areas also
    the edges across the walking areas that the filter keeps."""
    import warnings

    from geopandas.testing import assert_geodataframe_equal

    from pyrosm import OSM

    frames = [
        OSM(get_data("helsinki_pbf"), engine=engine).get_network(
            "walking", nodes=True, indoor_elements=True, walking_areas=walking_areas
        )
        for engine in ("in_memory", "out_of_core")
    ]
    for first, second in zip(*frames):
        assert_geodataframe_equal(first, second, check_dtype=False)
    nodes, edges = frames[1]
    assert 35744431 in set(edges["id"])
    assert not edges.duplicated(["osm_type", "id", "u", "v"]).any()
    assert edges["u"].isin(nodes["id"]).all() and edges["v"].isin(nodes["id"]).all()
    assert edges["walking_area"].dtype == bool
    platform_area = (edges["osm_type"] == "way") & (edges["id"] == 8137136)
    assert (platform_area & edges["walking_area"]).any()
    note = edges.attrs["pyrosm_read"]["filter"]
    assert [["railway", "eq", "platform", 0]] in note
    ends = edges.set_index("id")[["u_level", "v_level", "vertical"]]
    assert sorted(ends.loc[18378805, ["u_level", "v_level"]]) == [-3.0, -2.0]
    assert sorted(ends.loc[655180399, ["u_level", "v_level"]]) == [0.0, 2.0]
    assert set(ends.loc[[18378805, 23653232], "vertical"]) == {"escalator", "stairs"}
    assert ends.loc[[35744431], "vertical"].isna().all()
    # The lift node 2039713549 joins walkways on levels -2 and 0
    lift = edges[(edges["vertical"] == "lift") & (edges["id"] == 2039713549)]
    assert lift[["u", "v"]].values.tolist() == [[2039713549, -2039713549001]]
    assert -2039713549001 in set(edges.loc[edges["id"] == 470012326, "v"])
    assert -2039713549001 in set(nodes["id"])
    _, plain = OSM(get_data("helsinki_pbf")).get_network("walking", nodes=True)
    assert "u_level" not in plain.columns and 35744431 not in set(plain["id"])
    # A filter that keeps no way: the platforms alone, without an empty-data warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, platforms = OSM(get_data("helsinki_pbf")).get_network(
            custom_filter={"highway": ["no-such-value"]},
            nodes=True,
            indoor_elements=True,
        )
    assert 35744431 in set(platforms["id"]) and "u_level" in platforms.columns
    if walking_areas:
        # It crosses the walking areas that the filter keeps, here every multipolygon,
        # such as Elielinaukio (relation 116162), whose outline holds platform nodes
        _, kept = OSM(get_data("helsinki_pbf")).get_network(
            custom_filter={"type": ["multipolygon"]},
            nodes=True,
            indoor_elements=True,
            walking_areas=True,
        )
        across = kept[kept["walking_area"] & (kept["osm_type"] == "relation")]
        assert 116162 in set(across["id"])


def test_escalator_length_along_its_slope():
    """With indoor_elements=True, the length of an escalator is its length on the map
    divided by the cosine of escalator_slope (30 degrees by default), also in get_graph;
    None keeps the map length, and the other edges keep theirs."""
    import math

    from pyrosm import OSM

    osm = OSM(get_data("helsinki_pbf"))
    lengths = {}
    for slope in (30, None):
        _, edges = osm.get_network(
            "walking", nodes=True, indoor_elements=True, escalator_slope=slope
        )
        lengths[slope] = edges.set_index(["osm_type", "id", "u", "v"])
    sloped, flat = lengths[30], lengths[None]
    escalator = flat["vertical"] == "escalator"
    assert escalator.sum() > 0
    expected = flat.loc[escalator, "length"] / math.cos(math.radians(30))
    assert sloped.loc[escalator, "length"].to_numpy() == pytest.approx(
        expected.to_numpy(), abs=1e-3
    )
    assert sloped.loc[~escalator, "length"].equals(flat.loc[~escalator, "length"])
    graph = osm.get_graph(
        "walking",
        graph_type="networkx",
        indoor_elements=True,
        retain_all=True,
        simplify=False,
        escalator_slope=60,
    )
    edges = graph.edges(data=True)
    steep = [d["length"] for _, _, d in edges if d.get("osmid") == 18378805]
    # cos(60 degrees) = 0.5, and the walking graph has both directions of each edge
    map_lengths = flat.xs(18378805, level="id")["length"].tolist()
    assert sorted(steep) == pytest.approx(sorted([2 * x for x in map_lengths] * 2))


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_indoor_read_that_finds_nothing(engine):
    """An indoor read whose filter keeps no way, of a file without platforms, returns
    no frames and warns once that it found nothing."""
    from pyrosm import OSM

    osm = OSM(get_data("test_pbf"), engine=engine)
    with pytest.warns(UserWarning, match="Could not find any edges") as caught:
        result = osm.get_network(
            custom_filter={"highway": ["no-such-value"]},
            nodes=True,
            indoor_elements=True,
        )
    assert result == (None, None) and len(caught) == 1


def test_indoor_platforms_merge():
    """A read without platforms stays as it is; the platforms add the ways that are not
    edges yet (an edge across a walking area with the same id is a relation's) and
    their new nodes (without a walking_area column when the read has none); the
    recorded filter keeps the platforms next to the network's keys."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from pyrosm.indoor import with_platform_filter, with_platforms

    points = [Point(0, 0), Point(1, 0), Point(2, 0)]
    nodes = gpd.GeoDataFrame({"id": [1, 2]}, geometry=points[:2], crs=4326)
    line = LineString(points[:2])
    edges = gpd.GeoDataFrame({"id": [10], "u": [1], "v": [2]}, geometry=[line], crs=4326)
    assert with_platforms(nodes, edges, None, None)[1] is edges
    only = with_platforms(None, None, nodes, edges, walking_areas=True)[1]
    assert only["walking_area"].tolist() == [False]
    platform_nodes = gpd.GeoDataFrame({"id": [2, 3]}, geometry=points[1:], crs=4326)
    platforms = gpd.GeoDataFrame(
        {"id": [10, 20], "u": [1, 2], "v": [2, 3]},
        geometry=[line, LineString(points[1:])],
        crs=4326,
    )
    added = with_platforms(nodes, edges, platform_nodes, platforms)
    assert added[1]["id"].tolist() == [10, 20] and added[0]["id"].tolist() == [1, 2, 3]
    assert "walking_area" not in added[1].columns
    # An edge across a walking area numbered like a platform way
    crossing = edges.assign(osm_type="relation", id=20, walking_area=True)
    ways = gpd.GeoDataFrame(pd.concat([edges.assign(osm_type="way"), crossing]))
    nodes, edges = with_platforms(nodes, ways, platform_nodes, platforms)
    assert edges["id"].tolist() == [10, 20, 20]
    assert edges["walking_area"].tolist()[-1] is False
    assert with_platform_filter({"highway": ["footway"]}) == {
        "highway": ["footway", "platform"],
        "railway": ["platform"],
        "public_transport": ["platform"],
    }
    assert with_platform_filter({"highway": True, "area": True}) == {
        "highway": [True],
        "area": [True],
        "railway": ["platform"],
        "public_transport": ["platform"],
    }


def test_indoor_area_crossings_enter_through_doors():
    """Indoor areas are crossed as walking areas are, entered also through a door on the
    outline that is not a node of the network, which then joins the nodes; a walking
    area is not entered through it; without areas the edges get walking_area False; a
    walking area that is also an indoor area is crossed once, and one that is not a
    polygon is not crossed."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from pyrosm.indoor import indoor_areas, with_indoor_crossings

    nodes, edges, areas, lonlat = _walking_area_frames()
    points = gpd.points_from_xy(nodes["lon"], nodes["lat"])
    nodes = gpd.GeoDataFrame(nodes, geometry=points, crs=4326)
    x, y = lonlat(4, 4)  # a corner of the outline, not a node of the network
    door = gpd.GeoDataFrame(
        {"id": [99], "lon": [x], "lat": [y], "door": ["yes"]},
        geometry=[Point(x, y)],
        crs=4326,
    )
    same, plain = with_indoor_crossings(nodes, edges, None, door)
    assert same is nodes and not plain["walking_area"].any()
    walking_only, _ = with_indoor_crossings(nodes, edges, areas, door)
    assert 99 not in set(walking_only["id"])
    areas = areas.assign(tags='{"indoor": "area"}')
    with_door, crossed = with_indoor_crossings(nodes, edges, areas, door)
    across = crossed[crossed["walking_area"]]
    assert 99 in set(across["u"]) | set(across["v"])
    assert 99 in set(with_door["id"]) and "door" not in with_door.columns
    assert with_indoor_crossings(nodes, edges, areas, None)[1]["walking_area"].any()
    # The same lines across another area that is crossed already are not repeated
    other = across.assign(id=501)
    again = gpd.GeoDataFrame(pd.concat([plain, other], ignore_index=True), crs=4326)
    assert len(with_indoor_crossings(nodes, again, areas, door)[1]) == len(again)
    inner = areas.assign(geometry=[areas.geometry.iloc[0].geoms[1].buffer(-1e-6)])
    assert not with_indoor_crossings(nodes, edges, inner, None)[1]["walking_area"].any()
    assert len(indoor_areas(areas, areas)[0]) == 1
    assert indoor_areas(None) == (None, None)
    # A tag in the JSON of a walking area fills the column the indoor areas have for it
    walking = areas.assign(tags='{"railway": "platform"}')
    both, _ = indoor_areas(areas.assign(id=501, railway="platform"), walking)
    assert both["railway"].tolist() == ["platform", "platform"]
    lines = areas.assign(geometry=[LineString([(0, 0), (1, 1)])])
    assert indoor_areas(lines) == (None, None)
    # A lift area (highway=elevator) is not crossed; the polygons of the lift read
    # come back as the lifts
    lift = areas.assign(highway="elevator")
    assert indoor_areas(lift) == (None, None)
    crossed, lifts = indoor_areas(areas, None, pd.concat([lift, lines]))
    assert len(crossed) == 1 and lifts["id"].tolist() == areas["id"].tolist()
    assert indoor_areas(None, None, lines) == (None, None)


def test_indoor_lifts_split_per_level():
    """A lift node where edges on three levels meet becomes one node per level: the
    lowest keeps the id, each edge ends at the copy of its level (an end without a level
    stays at the lowest), and lift edges join neighbouring levels; a lift node with one
    level (tags as JSON text) and a node that is not a lift stay as they are. The lift
    edges take the lift node's metadata in the edges' dtypes."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from pyrosm.indoor import lift_copy_id, split_lifts

    tags = [{"highway": "elevator"}, {"highway": "elevator"}, {}, {}]
    nodes = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "tags": tags, "version": [7, 1, 1, 1],
         "timestamp": [1600000000] * 4, "visible": [True] * 4},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs=4326,
    )
    ends = [(3, 1, -1.0), (1, 4, 0.0), (4, 1, 2.0), (1, 3, np.nan), (2, 3, 0.0)]
    edges = gpd.GeoDataFrame(
        {
            "id": [10, 11, 12, 13, 14],
            "osm_type": "way",
            "u": [e[0] for e in ends],
            "v": [e[1] for e in ends],
            "u_level": [e[2] for e in ends],
            "v_level": [e[2] for e in ends],
            "version": np.array([2] * 5, dtype=np.int32),
            "visible": True,
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 5,
        crs=4326,
    )
    nodes, edges = split_lifts(nodes, edges)
    copy0, copy2 = lift_copy_id(1, 1), lift_copy_id(1, 2)
    assert edges.loc[:4, ["u", "v"]].values.tolist() == [
        [3, 1], [copy0, 4], [4, copy2], [1, 3], [2, 3]
    ]
    lifts = edges[edges["vertical"] == "lift"]
    assert lifts[["id", "osm_type", "u", "v", "length"]].values.tolist() == [
        [1, "node", 1, copy0, 0.0],
        [1, "node", copy0, copy2, 0.0],
    ]
    assert lifts["version"].tolist() == [7, 7] and lifts["visible"].tolist() == [True] * 2
    assert edges["version"].dtype == np.int32 and edges["visible"].dtype == bool
    assert "timestamp" not in edges.columns
    assert sorted(nodes["id"]) == [copy2, copy0, 1, 2, 3, 4]
    assert split_lifts(nodes.drop(columns="tags"), edges)[1] is edges
    one_level = nodes.assign(tags=["{}", '{"highway": "elevator"}'] + ["{}"] * 4)
    assert split_lifts(one_level, edges)[1] is edges
    # A copy's id taken by another node
    taken = gpd.GeoDataFrame(
        {"id": [1, 2, copy0], "tags": [{"highway": "elevator"}, {}, {}]},
        geometry=[Point(0, 0)] * 3,
        crs=4326,
    )
    two = gpd.GeoDataFrame(
        {"u": [2, 1], "v": [1, copy0], "u_level": [-1.0, 0.0], "v_level": [-1.0, 0.0]},
        geometry=[LineString([(0, 0), (1, 1)])] * 2,
        crs=4326,
    )
    with pytest.raises(ValueError, match="cannot be split per level"):
        split_lifts(taken, two)
    # A lift node that pyrosm wrote (negative id) is not split
    written = taken.assign(id=[-1, 2, 3])
    assert split_lifts(written, two.replace({1: -1}))[1]["u"].tolist() == [2, -1]


def test_lift_areas_joined_per_level(monkeypatch):
    """A lift area that the network enters on two of its levels gets a node per level in
    its middle, an edge to it from each entrance there (on that level, with that one
    level as its ``level``, along a line inside the area: bent in an L-shaped lift) and
    a lift edge between the levels; an area without a level tag takes any level, a
    relation's nodes have their own ids, and each part of a multipolygon is a lift of
    its own. Not joined: an area
    entered on one listed level (an entrance on an unlisted level and a dead end without
    a level do not count), one with a lift node split per level in it, one that pyrosm
    wrote (negative id) and one whose node ids are taken. Without lift areas the frames
    stay as they are; the walkways of closed lift ways go, with the nodes only they
    used."""
    import json

    import geopandas as gpd
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon

    from pyrosm.indoor import (
        join_lift_areas,
        lift_area_copy_id,
        lift_copy_id,
        without_lift_area_ways,
    )

    def at(x, y):
        return 24.9 + x * 1e-4, 60.2 + y * 1e-4

    def square(x):
        return Polygon([at(x, 0), at(x + 1, 0), at(x + 1, 1), at(x, 1)])

    # (id, osm_type, level tag, its area): A, B (a relation of two parts, without a
    # level), C (one listed level reached), D (a split lift node in it), E (negative id),
    # F (ids taken), G (L-shaped)
    l_shape = Polygon([at(14, 0), at(16, 0), at(16, 0.4), at(14.4, 0.4), at(14.4, 2),
                       at(14, 2)])
    areas = [(7, "way", "0;1", square(0)),
             (8, "relation", None, MultiPolygon([square(2), square(12)])),
             (9, "way", "0;1", square(4)), (10, "way", "0;1", square(6)),
             (-11, "way", "0;1", square(8)), (12, "way", "0;1", square(10)),
             (13, "way", "0;1", l_shape)]
    lifts = gpd.GeoDataFrame(
        {"id": [a[0] for a in areas], "osm_type": [a[1] for a in areas],
         "level": [a[2] for a in areas], "indoor": "room", "version": 3},
        geometry=[a[3] for a in areas],
        crs=4326,
    )
    # (entrance node, its place, the level of its edge): node 3 is on level 2, which A
    # does not list, and node 4 is a dead end in A without a level
    entrances = [(1, (0, 0), 0.0), (2, (1, 1), 1.0), (3, (1, 0), 2.0),
                 (4, (0.5, 0.5), np.nan), (5, (2, 0), 0.0), (6, (3, 1), 2.0),
                 (7, (4, 0), 0.0), (8, (6, 0), 0.0), (9, (7, 1), 1.0),
                 (10, (8, 0), 0.0), (11, (9, 1), 1.0), (12, (10, 0), 0.0),
                 (13, (11, 1), 1.0), (14, (12, 0), 0.0), (15, (13, 1), 1.0),
                 (16, (16, 0), 0.0), (17, (14, 2), 1.0)]
    far = [100 + e[0] for e in entrances]
    taken = lift_area_copy_id(12, "way", 1)
    ids = [e[0] for e in entrances] + far + [taken]
    places = [at(*e[1]) for e in entrances] + [at(20, 20)] * (len(far) + 1)
    nodes = gpd.GeoDataFrame(
        {"id": ids, "lon": [p[0] for p in places], "lat": [p[1] for p in places],
         "version": 1},
        geometry=[Point(p) for p in places],
        crs=4326,
    )
    split = (50, lift_copy_id(50, 1), 0.0, 1.0)  # a lift node split per level, in D
    edges = gpd.GeoDataFrame(
        {"id": [200 + e[0] for e in entrances] + [50],
         "osm_type": ["way"] * len(entrances) + ["node"],
         "u": far + [split[0]], "v": [e[0] for e in entrances] + [split[1]],
         "u_level": [e[2] for e in entrances] + [split[2]],
         "v_level": [e[2] for e in entrances] + [split[3]],
         "vertical": [None] * len(entrances) + ["lift"],
         "walking_area": False, "version": 1},
        geometry=[LineString([at(20, 20), at(*e[1])]) for e in entrances]
        + [LineString([at(6.5, 0.5), at(6.5, 0.5)])],
        crs=4326,
    )
    assert join_lift_areas(nodes, edges, None) == (nodes, edges)
    a = [lift_area_copy_id(7, "way", rank) for rank in (1, 2)]
    b = [lift_area_copy_id(8, "relation", rank) for rank in (1, 2, 3, 4)]
    g = [lift_area_copy_id(13, "way", rank) for rank in (1, 2)]
    for columns in ({"level": None}, {"tags": None}, {}):
        joined_nodes, joined = join_lift_areas(nodes, edges.assign(**columns), lifts)
        new = joined.iloc[len(edges):]
        lift = new[new["vertical"] == "lift"]
        assert lift[["id", "osm_type", "u", "v", "u_level", "v_level"]].values.tolist() == [
            [7, "way", a[0], a[1], 0.0, 1.0], [8, "relation", b[0], b[1], 0.0, 2.0],
            [8, "relation", b[2], b[3], 0.0, 1.0], [13, "way", g[0], g[1], 0.0, 1.0],
        ]
        assert (lift["length"] == 0).all() and not lift["walking_area"].any()
        into = new[new["walking_area"]]
        assert into[["id", "u", "v", "u_level", "v_level"]].values.tolist() == [
            [8, 5, b[0], 0.0, 0.0], [8, 6, b[1], 2.0, 2.0],
            [8, 14, b[2], 0.0, 0.0], [8, 15, b[3], 1.0, 1.0],
            [7, 1, a[0], 0.0, 0.0], [7, 2, a[1], 1.0, 1.0],
            [13, 16, g[0], 0.0, 0.0], [13, 17, g[1], 1.0, 1.0],
        ]
        assert sorted(joined_nodes["id"].iloc[len(nodes):]) == sorted(a + b + g)
        lines = into.geometry.iloc[-2:]
        assert all(l_shape.buffer(1e-9).covers(line) for line in lines)
        assert sorted(len(line.coords) for line in lines) == [2, 3]
        if "level" in columns:
            assert into["level"].tolist() == ["0", "2"] + ["0", "1"] * 3
        elif "tags" in columns:
            tags = [json.loads(t) for t in into["tags"]]
            assert [t["level"] for t in tags] == ["0", "2"] + ["0", "1"] * 3
            assert tags[0]["indoor"] == "room"
    assert joined["version"].dtype == edges["version"].dtype
    # A taken node id of one part leaves the whole multipolygon as it is
    one_taken = gpd.GeoDataFrame(
        pd.concat([nodes, nodes.iloc[[-1]].assign(id=b[2])], ignore_index=True), crs=4326
    )
    added = join_lift_areas(one_taken, edges, lifts)[1].iloc[len(edges):]
    assert 8 not in set(added["id"]) and 7 in set(added["id"])
    # Without a path from the visibility search, only a straight line inside counts, and
    # the L-shaped lift is entered on one level then
    import pyrosm.walking_areas

    from pyrosm.indoor import _line_inside

    monkeypatch.setattr(pyrosm.walking_areas, "_part_crossings", lambda *args: [])
    assert _line_inside(l_shape, at(14.2, 0.2), at(14.2, 1.8)) is not None
    assert _line_inside(l_shape, at(16, 0), at(14, 2)) is None
    only_l = lifts[lifts["id"] == 13].reset_index(drop=True)
    assert join_lift_areas(nodes, edges, only_l) == (nodes, edges)
    monkeypatch.undo()
    # The walkways of a closed lift way go with the nodes only they used
    ring = edges.iloc[[0]].assign(id=7, u=1, v=2)
    with_ring = gpd.GeoDataFrame(pd.concat([edges, ring], ignore_index=True), crs=4326)
    kept_nodes, kept = without_lift_area_ways(nodes, with_ring, lifts)
    assert 7 not in set(kept["id"]) and len(kept) == len(edges)
    assert len(kept_nodes) == len(nodes)
    lone = with_ring.assign(u=with_ring["u"].where(with_ring["id"] != 7, 999))
    lone_nodes = gpd.GeoDataFrame(
        pd.concat([nodes, nodes.iloc[[0]].assign(id=999)], ignore_index=True), crs=4326
    )
    assert 999 not in set(without_lift_area_ways(lone_nodes, lone, lifts)[0]["id"])
    assert without_lift_area_ways(nodes, edges, lifts) == (nodes, edges)
    assert without_lift_area_ways(nodes, edges, None) == (nodes, edges)


def test_lift_areas_mapped_floor_by_floor_make_one_lift():
    """Lift areas mapped one floor at a time make one lift: S on levels 0, 1 and 3
    (level 2 missing) and F on levels 0 to 4, the areas slightly apart, with lift edges
    from the middle of one floor's area to the next, each with the id of the lower area.
    Not joined: two lifts
    side by side (N), areas three levels apart (G), a second area on a floor that the
    lift has already (T, which keeps its lift from levels 0 to 1), an area without a
    level tag and the area above it (U), an area whose only node is on its outline
    between two corners, not an entrance, and the area above it (V), a small area in a
    large area above it whose middle lies outside the small one (W)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    from pyrosm.indoor import join_lift_areas, lift_area_copy_id

    def at(x, y):
        return 24.9 + x * 1e-4, 60.2 + y * 1e-4

    def square(x, y=0.0, size=1.0):
        return Polygon([at(x, y), at(x + size, y), at(x + size, y + size), at(x, y + size)])

    # (id, level tag, its square's corner, the entrance at that corner on that level)
    areas = [(20, "0", (0, 0)), (21, "1", (0.1, 0.1)), (23, "3", (0.2, 0)),
             (30, "0", (5, 0)), (31, "1", (5.9, 0)), (40, "0", (10, 0)),
             (43, "3", (10.1, 0.1)), (50, "0", (15, 0)), (51, "1", (15.1, 0)),
             (52, "1", (15, 0.1)), (60, None, (20, 0)), (61, "1", (20.1, 0.1)),
             (70, "0", (25, 0)), (71, "1", (25.1, 0.1)), (80, "0", (30, 0)),
             (81, "1", (30.2, -3), 4.0), (90, "0", (40, 0)), (91, "1", (40.1, 0.1)),
             (92, "2", (40, 0)), (93, "3", (40.1, 0.1)), (94, "4", (40, 0))]
    lifts = gpd.GeoDataFrame(
        {"id": [a[0] for a in areas], "osm_type": "way",
         "level": [a[1] for a in areas]},
        geometry=[square(*a[2], *a[3:]) for a in areas],
        crs=4326,
    )
    # V's node lies on its outline between two corners; the areas above U and V, and the
    # large area of W, are entered at corners outside the area below them
    place = {a[0]: a[2] for a in areas} | {70: (25.5, 0), 61: (21.1, 1.1),
                                           71: (26.1, 1.1), 81: (34.2, 1)}
    nodes = gpd.GeoDataFrame(
        {"id": [a[0] for a in areas] + [999],
         "lon": [at(*place[a[0]])[0] for a in areas] + [at(30, 30)[0]],
         "lat": [at(*place[a[0]])[1] for a in areas] + [at(30, 30)[1]]},
        geometry=[Point(at(*place[a[0]])) for a in areas] + [Point(at(30, 30))],
        crs=4326,
    )
    levels = [float(a[1]) if a[1] else 0.0 for a in areas]
    edges = gpd.GeoDataFrame(
        {"id": range(100, 100 + len(areas)), "osm_type": "way", "u": 999,
         "v": [a[0] for a in areas], "u_level": levels, "v_level": levels,
         "vertical": None, "walking_area": False},
        geometry=[LineString([at(30, 30), at(*place[a[0]])]) for a in areas],
        crs=4326,
    )
    _, joined = join_lift_areas(nodes, edges, lifts)
    lift = joined[joined["vertical"] == "lift"]
    ids = (20, 21, 23, 50, 51, *range(90, 95))
    node = {i: lift_area_copy_id(i, "way", 1) for i in ids}
    assert lift[["id", "u", "v", "u_level", "v_level"]].values.tolist() == [
        [20, node[20], node[21], 0.0, 1.0], [21, node[21], node[23], 1.0, 3.0],
        [50, node[50], node[51], 0.0, 1.0],
        *([i, node[i], node[i + 1], i - 90.0, i - 89.0] for i in range(90, 94)),
    ]
    first = lift.geometry.iloc[0]
    middles = [lifts.geometry.iloc[i].representative_point() for i in (0, 1)]
    assert [Point(c) for c in first.coords] == middles
    into = joined[joined["walking_area"]]
    assert sorted(into["id"]) == [20, 21, 23, 50, 51, *range(90, 95)]


def test_indoor_levels_of_edge_ends():
    """An edge with one level has it at both ends; stairs and escalators with a level
    range run between the levels of the walkways at their ends, also when one end is
    unknown (none when both are, or when both ends touch only the same level), and
    those with no level or one level take the walkways' levels when they differ; a
    walkway tagged with several levels and an edge without one have none. Moving steps
    are escalators, other steps stairs, other ways that move moving walkways, and a line
    across a moving area is none of them (a frame without such lines has none)."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from pyrosm.indoor import edge_levels, edge_vertical, with_indoor_columns

    rows = [
        (1, 2, {"highway": "corridor", "level": "-1"}),
        (3, 4, {"highway": "corridor", "level": "0"}),
        (2, 3, {"highway": "steps", "conveying": "yes", "level": "0;-1"}),
        (4, 5, {"highway": "steps", "level": "0-1"}),
        (5, 6, {"highway": "corridor", "level": "0;1"}),
        (1, 4, {"highway": "steps"}),
        (1, 7, {"highway": "steps", "level": "-1"}),
        (7, 8, {"highway": "footway"}),
        (9, 4, {"highway": "steps", "level": "0;1"}),
        (10, 11, {"highway": "steps", "level": "2;3"}),
        (2, 1, {"highway": "steps", "level": "-1;0"}),
        (3, 2, {"highway": "steps", "level": "-1;0"}),
    ]
    edges = pd.DataFrame({"u": [r[0] for r in rows], "v": [r[1] for r in rows]})
    u_level, v_level = edge_levels(edges, [r[2] for r in rows])
    expected = [(-1, -1), (0, 0), (-1, 0), (0, 1), (None, None), (-1, 0), (-1, -1)]
    expected += [(None, None), (1, 0), (None, None), (None, None), (0, -1)]
    moving = [
        {"highway": "steps", "conveying": "no"},
        {"highway": "footway", "conveying": "yes"},
        {},
    ]
    assert edge_vertical(moving) == ["stairs", "moving_walkway", None]
    across = gpd.GeoDataFrame(
        {
            "u": [1, 2],
            "v": [2, 3],
            "length": 1.0,
            "walking_area": [False, True],
            "indoor": [None, "area"],
            "highway": ["footway", None],
            "conveying": "forward",
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 2,
        crs=4326,
    )
    vertical = with_indoor_columns(across, None)["vertical"]
    assert vertical[0] == "moving_walkway" and pd.isna(vertical[1])
    plain = with_indoor_columns(across.drop(columns="walking_area"), None)["vertical"]
    assert plain.tolist() == ["moving_walkway"] * 2
    assert [
        (None if np.isnan(a) else a, None if np.isnan(b) else b)
        for a, b in zip(u_level, v_level)
    ] == expected


def test_indoor_levels_along_ways_of_several_edges():
    """Stairs split into several edges get their levels per way: the two ends from the
    walkways there or, for a level range, from the end node's own level, a walkway
    joined in between (or an inner node's own level) is kept, and the other inner nodes
    get levels in proportion to the distance along the way, whichever way an edge is
    drawn. An end without a level takes the other end of the range from the nearest
    inner level; ends on one level reach the other end of the range in between; a way
    without a range keeps its outermost levels up to its ends; a way with an edge of
    length 0 counts each edge as one. A way whose edges branch or close a ring is taken
    edge by edge. Where nothing gives an end, incline=up puts the low end of the range at
    the start of the way (its drawn direction) and incline=down the high end, but not
    against two known ends."""
    import geopandas as gpd

    from pyrosm.indoor import edge_levels, node_levels

    walkway = {"highway": "corridor"}
    rows = [
        # Way 30, -1;0 over 10-11-12-13 (lengths 1, 1, 2), walkways at both ends
        (30, 10, 11, 1, {"highway": "steps", "level": "-1;0"}),
        (30, 12, 11, 1, {"highway": "steps", "level": "-1;0"}),
        (30, 12, 13, 2, {"highway": "steps", "level": "-1;0"}),
        (1, 1, 10, 1, {**walkway, "level": "0"}),
        (2, 13, 2, 1, {**walkway, "level": "-1"}),
        # Way 31, -1;0 over 20-21-22, node 20 tagged level=0, no walkways
        (31, 20, 21, 1, {"highway": "steps", "level": "-1;0"}),
        (31, 21, 22, 1, {"highway": "steps", "level": "-1;0"}),
        # Way 32, -2;0 over 40-41-42 (lengths 1, 3), a walkway at -1 in between
        (32, 40, 41, 1, {"highway": "steps", "level": "-2;0"}),
        (32, 41, 42, 3, {"highway": "steps", "level": "-2;0"}),
        (3, 3, 40, 1, {**walkway, "level": "0"}),
        (4, 41, 4, 1, {**walkway, "level": "-1"}),
        (5, 42, 5, 1, {**walkway, "level": "-2"}),
        # Way 33 without a level tag over 50-51-52
        (33, 50, 51, 1, {"highway": "steps"}),
        (33, 51, 52, 1, {"highway": "steps"}),
        (6, 6, 50, 1, {**walkway, "level": "0"}),
        (7, 52, 7, 1, {**walkway, "level": "-1"}),
        # Way 34 branches at 71, way 35 is a line and a ring: edge by edge
        (34, 70, 71, 1, {"highway": "steps", "level": "-1;0"}),
        (34, 71, 72, 1, {"highway": "steps", "level": "-1;0"}),
        (34, 71, 73, 1, {"highway": "steps", "level": "-1;0"}),
        (35, 80, 81, 1, {"highway": "steps", "level": "-1;0"}),
        (35, 82, 83, 1, {"highway": "steps", "level": "-1;0"}),
        (35, 83, 84, 1, {"highway": "steps", "level": "-1;0"}),
        (35, 84, 82, 1, {"highway": "steps", "level": "-1;0"}),
        (8, 8, 70, 1, {**walkway, "level": "0"}),
        (9, 9, 80, 1, {**walkway, "level": "0"}),
        # Way 36, -1;0 with both ends at -1; way 37 the same through node 94 (level=0)
        (36, 90, 91, 1, {"highway": "steps", "level": "-1;0"}),
        (36, 91, 92, 1, {"highway": "steps", "level": "-1;0"}),
        (37, 93, 94, 1, {"highway": "steps", "level": "-1;0"}),
        (37, 94, 95, 3, {"highway": "steps", "level": "-1;0"}),
        (10, 111, 90, 1, {**walkway, "level": "-1"}),
        (11, 92, 112, 1, {**walkway, "level": "-1"}),
        (12, 113, 93, 1, {**walkway, "level": "-1"}),
        (13, 95, 114, 1, {**walkway, "level": "-1"}),
        # Way 38, 0;1 over 96-99, ends unknown, a walkway at 1 at node 98
        (38, 96, 97, 1, {"highway": "steps", "level": "0;1"}),
        (38, 97, 98, 1, {"highway": "steps", "level": "0;1"}),
        (38, 98, 99, 1, {"highway": "steps", "level": "0;1"}),
        (14, 98, 115, 1, {**walkway, "level": "1"}),
        # Way 39 without a level tag over 100-103, walkways at 101 and 102 only
        (39, 100, 101, 1, {"highway": "steps"}),
        (39, 101, 102, 1, {"highway": "steps"}),
        (39, 102, 103, 1, {"highway": "steps"}),
        (15, 116, 101, 1, {**walkway, "level": "0"}),
        (16, 102, 117, 1, {**walkway, "level": "-1"}),
        # Way 40, -1;0 over 120-121-122 with an edge of length 0: steps of one
        (40, 120, 121, 0, {"highway": "steps", "level": "-1;0"}),
        (40, 121, 122, 1, {"highway": "steps", "level": "-1;0"}),
        (17, 118, 120, 1, {**walkway, "level": "0"}),
        (18, 122, 119, 1, {**walkway, "level": "-1"}),
        # Ways 41-44 with incline: rising, falling over two edges listed in reverse,
        # one end known, and two known ends on one level
        (41, 200, 201, 1, {"highway": "steps", "level": "-1;0", "incline": "up"}),
        (42, 211, 212, 1, {"highway": "steps", "level": "0;1", "incline": "down"}),
        (42, 210, 211, 1, {"highway": "steps", "level": "0;1", "incline": "down"}),
        (43, 220, 221, 1, {"highway": "steps", "level": "-3;-1", "incline": "up"}),
        (19, 222, 220, 1, {**walkway, "level": "-2"}),
        (44, 230, 231, 1, {"highway": "steps", "level": "-1;0", "incline": "up"}),
        (20, 232, 230, 1, {**walkway, "level": "-1"}),
        (21, 231, 233, 1, {**walkway, "level": "-1"}),
    ]
    edges = pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "u": [r[1] for r in rows],
            "v": [r[2] for r in rows],
            "length": [float(r[3]) for r in rows],
        }
    )
    nodes = gpd.GeoDataFrame(
        {
            "id": [20, 21, 22, 23, 94],
            "tags": [{"level": "0"}, '{"level": "-1;0"}', {}, None, {"level": "0"}],
        }
    )
    assert node_levels(nodes) == {20: (0.0,), 94: (0.0,)}
    u_level, v_level = edge_levels(edges, [r[4] for r in rows], node_levels(nodes))
    got = {
        (r[0], r[1], r[2]): (a, b)
        for r, a, b in zip(rows, u_level, v_level)
        if r[0] >= 30
    }
    nan = (np.nan, np.nan)
    expected = {
        (30, 10, 11): (0.0, -0.25), (30, 12, 11): (-0.5, -0.25), (30, 12, 13): (-0.5, -1.0),
        (31, 20, 21): (0.0, -0.5), (31, 21, 22): (-0.5, -1.0),
        (32, 40, 41): (0.0, -1.0), (32, 41, 42): (-1.0, -2.0),
        (33, 50, 51): (0.0, -0.5), (33, 51, 52): (-0.5, -1.0),
        (34, 70, 71): (0.0, -1.0), (34, 71, 72): nan, (34, 71, 73): nan,
        (35, 80, 81): (0.0, -1.0), (35, 82, 83): nan, (35, 83, 84): nan, (35, 84, 82): nan,
        (36, 90, 91): (-1.0, 0.0), (36, 91, 92): (0.0, -1.0),
        (37, 93, 94): (-1.0, 0.0), (37, 94, 95): (0.0, -1.0),
        (38, 96, 97): (0.0, 0.5), (38, 97, 98): (0.5, 1.0), (38, 98, 99): (1.0, 0.0),
        (39, 100, 101): (0.0, 0.0), (39, 101, 102): (0.0, -1.0), (39, 102, 103): (-1.0, -1.0),
        (40, 120, 121): (0.0, -0.5), (40, 121, 122): (-0.5, -1.0),
        (41, 200, 201): (-1.0, 0.0),
        (42, 211, 212): (0.5, 0.0), (42, 210, 211): (1.0, 0.5),
        (43, 220, 221): (-2.0, -1.0),
        (44, 230, 231): nan,
    }
    assert got.keys() == expected.keys()
    for key, levels in expected.items():
        np.testing.assert_array_equal(got[key], levels, err_msg=str(key))


@pytest.mark.parametrize("engine", ["in_memory", "out_of_core"])
def test_get_network_with_walking_areas_reports_one_progress(engine, monkeypatch):
    """The network read and the read of the walking areas report one progress from 0 to
    100 %, which stays below 100 % until the edges across the areas are built."""
    from pyrosm import OSM, walking_areas

    seen = []
    at_crossings = []
    real = walking_areas.add_crossings

    def add_crossings(*args):
        at_crossings.append(seen[-1][0])
        return real(*args)

    monkeypatch.setattr(walking_areas, "add_crossings", add_crossings)
    osm = OSM(get_data("helsinki_pbf"), engine=engine, progress=lambda *r: seen.append(r))
    osm.get_network(nodes=True, walking_areas=True)
    done = [d for d, _ in seen]
    assert seen[0] == (0.0, 100) and seen[-1] == (100.0, 100) and done == sorted(done)
    assert at_crossings[0] < 100
