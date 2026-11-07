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
        assert list(row.geometry.coords) == pytest.approx(xy)
        lon, lat = np.array(xy).T
        pieces = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:], unit=Unit.METERS)
        assert row["length"] == pytest.approx(pieces.round(3).sum(), abs=1e-6)
    without_tags = walking_areas.crossing_rows(nodes, edges.drop(columns="tags"), areas)
    assert "tags" not in without_tags.columns
    shrunk = areas.assign(geometry=[areas.geometry.iloc[0].geoms[1].buffer(-1e-6)])
    assert walking_areas.crossing_rows(nodes, edges, shrunk) is None


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
    ],
)
def test_network_reads_refuse_unusable_walking_areas(test_pbf, read, kwargs, message):
    """walking_areas is a boolean, and the edges across areas need the network's nodes."""
    from pyrosm import OSM

    with pytest.raises(ValueError, match=message):
        getattr(OSM(test_pbf), read)(**kwargs)


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
