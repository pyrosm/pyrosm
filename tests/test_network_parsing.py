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
    lines bend only at other entrances, so the tips of a U get no edge."""
    from pyrosm import walking_areas
    from pyrosm.distance import Unit, haversine

    nodes, edges, areas, lonlat = _walking_area_frames()
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
    ],
)
def test_network_reads_refuse_unusable_walking_areas(test_pbf, read, kwargs, message):
    """walking_areas and indoor_elements are booleans, need the network's nodes, and
    indoor_elements needs a walking network type or a custom filter."""
    from pyrosm import OSM

    with pytest.raises(ValueError, match=message):
        getattr(OSM(test_pbf), read)(**kwargs)


@pytest.mark.parametrize("walking_areas", [False, True])
def test_get_network_with_indoor_elements(walking_areas):
    """indoor_elements=True adds the platforms, also those the walking network leaves
    out (way 35744431, railway=platform, has no highway tag), once each and with their
    nodes, crosses the platform areas (way 8137136), and adds u_level, v_level and
    vertical (way 23653232 is stairs, a platform is neither); both engines return the
    same frames: the escalators
    of ways 18378805 (level=-3;-2) and 655180399 (level=0;2) run between the levels at
    their ends; without the option the columns and platforms are not there. A read
    whose filter keeps no other way returns the platforms."""
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
    outline that is not a node of the network, which then joins the nodes; without
    areas the edges get walking_area False; an area already crossed, or not a polygon,
    is not crossed again."""
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
    assert indoor_areas(areas, crossed) is None and indoor_areas(None, plain) is None
    lines = areas.assign(geometry=[LineString([(0, 0), (1, 1)])])
    assert indoor_areas(lines, plain) is None


def test_indoor_levels_of_edge_ends():
    """An edge with one level has it at both ends; stairs and escalators with a level
    range run between the levels of the walkways at their ends, also when one end is
    unknown (none when both are, or when both ends touch only the same level), and
    those with no level or one level take the walkways' levels when they differ; a
    walkway tagged with several levels and an edge without one have none. Moving steps
    are escalators, other steps stairs."""
    from pyrosm.indoor import edge_levels, edge_vertical

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
    assert edge_vertical(moving) == ["stairs", None, None]
    assert [
        (None if np.isnan(a) else a, None if np.isnan(b) else b)
        for a, b in zip(u_level, v_level)
    ] == expected


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
