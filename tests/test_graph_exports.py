import sys
import pytest
from pyrosm import get_data
from pyrosm.config import Conf


def _ulanbator_pbf():
    """Path to the UlanBator test network, a pinned snapshot hosted on a gist.

    The data must stay uncropped: the export tests assert ``ecount == 2 * n_edges``
    and ``vcount == n_nodes``, which only hold when no boundary nodes/edges are
    dropped.
    """
    return get_data("ulanbator_test_pbf")


# pandana's compiled cyaccess uses C `long` buffers. On Windows C `long` is
# 32-bit, but NumPy 2 makes the default integer int64 ("long long"), so pandana
# rejects the arrays it builds internally ("expected 'long' but got 'long
# long'"). pandana is unmaintained, so skip its export tests on Windows; they
# run normally on Linux/macOS where C `long` is 64-bit.
_pandana_unsupported_on_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="pandana is incompatible with NumPy 2 on Windows (C long is 32-bit)",
)

# The values used to determine oneway road in OSM
oneway_values = Conf.oneway_values
oneway_col = "oneway"


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_history_pbf():
    pbf_path = get_data("helsinki_test_history_pbf")
    return pbf_path


@pytest.fixture
def walk_nodes_and_edges():
    from pyrosm import OSM

    # UlanBator is good small dataset for testing
    # (unmodified, i.e. not cropped)
    pbf_path = _ulanbator_pbf()
    osm = OSM(pbf_path)
    return osm.get_network(nodes=True)


@pytest.fixture
def bike_nodes_and_edges():
    from pyrosm import OSM

    # UlanBator is good small dataset for testing
    # (unmodified, i.e. not cropped)
    pbf_path = _ulanbator_pbf()
    osm = OSM(pbf_path)
    return osm.get_network(nodes=True, network_type="cycling")


@pytest.fixture
def driving_nodes_and_edges():
    from pyrosm import OSM

    pbf_path = _ulanbator_pbf()
    osm = OSM(pbf_path)
    return osm.get_network(network_type="driving", nodes=True)


@pytest.fixture
def immutable_nodes_and_edges():
    from pyrosm import OSM

    pbf_path = get_data("test_pbf")
    osm = OSM(pbf_path)
    return osm.get_network(nodes=True)


def test_igraph_export_by_walking(walk_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph

    nodes, edges = walk_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=True, simplify=False)
    n_edges = len(edges)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)

    # In case of walking/cycling there should be 2x the num of edges
    # as in the orig gdf (one edge for each direction)
    # --> assuming the data has not been cropped (which might drop nodes/edges)
    assert g.ecount() == 2 * n_edges

    # The number of nodes should be the same
    assert g.vcount() == n_nodes

    # Ensure that all attributes were transfered to graph
    ecolumns = edges.columns
    ncolumns = nodes.columns
    eattributes = g.edge_attributes()
    nattributes = g.vertex_attributes()

    for col in ecolumns:
        assert col in eattributes

    for col in ncolumns:
        assert col in nattributes

    # Check that all edge attributes have same length
    for col in eattributes:
        assert len(g.es[col]) == g.ecount()


def test_igraph_export_by_cycling(bike_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph

    nodes, edges = bike_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=True, simplify=False)
    n_edges = len(edges)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)

    # Cycling is now directed (it honours oneway / oneway:bicycle). Two-way edges
    # are still duplicated (one per direction) but one-way cycleways/streets are
    # not, so the directed edge count is strictly between n_edges and 2*n_edges
    # (assuming the data is uncropped and contains a mix of one-way and two-way).
    assert n_edges < g.ecount() < 2 * n_edges

    # The number of nodes should be the same
    assert g.vcount() == n_nodes

    # Ensure that all attributes were transfered to graph
    ecolumns = edges.columns
    ncolumns = nodes.columns
    eattributes = g.edge_attributes()
    nattributes = g.vertex_attributes()

    for col in ecolumns:
        assert col in eattributes

    for col in ncolumns:
        assert col in nattributes

    # Check that all edge attributes have same length
    for col in eattributes:
        assert len(g.es[col]) == g.ecount()


def test_igraph_export_by_driving(driving_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph

    nodes, edges = driving_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=True, simplify=False)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)

    # The number of nodes should be the same in the graph
    assert g.vcount() == n_nodes

    # Ensure that all attributes were transfered to graph
    ecolumns = edges.columns
    ncolumns = nodes.columns
    eattributes = g.edge_attributes()
    nattributes = g.vertex_attributes()

    for col in ecolumns:
        assert col in eattributes

    for col in ncolumns:
        assert col in nattributes

    # Check that all edge attributes have same length
    for col in eattributes:
        assert len(g.es[col]) == g.ecount()

    # Calculate the number of edges that should be oneway + bidirectional
    mask = edges[oneway_col].isin(oneway_values)

    # Check that the edge count matches
    # TODO: The following fails, check why later
    # assert g.ecount() == 44296


def test_igraph_immutable_counts(test_pbf):
    """
    A simple check to ensure that
    the graph shape is always the
    same with unmutable data.
    """
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph
    from pyrosm import OSM

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    g = to_igraph(nodes, edges, retain_all=True, simplify=False)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)
    # Check that the edge count matches
    assert g.ecount() == 2050
    assert g.vcount() == n_nodes


def test_nxgraph_export_by_walking(walk_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_networkx
    import networkx as nx

    nodes, edges = walk_nodes_and_edges
    edges = edges.drop_duplicates(["u", "v"])
    g = to_networkx(nodes, edges, retain_all=True, osmnx_compatible=False, simplify=False)
    n_edges = len(edges)
    n_nodes = len(nodes)

    assert isinstance(g, nx.MultiDiGraph)

    # In case of walking/cycling there should be 2x the num of edges
    # as in the orig gdf (one edge for each direction)
    # --> assuming the data has not been cropped (which might drop nodes/edges)

    # Add a small threshold for the difference in the number of edges (allow 1 per mille diff)
    # as networkx automatically drops duplicates and otherwise seems
    # to filter somehow "incorrect" or duplicate edges
    assert abs(1 - (nx.number_of_edges(g) / (2 * n_edges))) < 0.001

    # The number of nodes should be the same
    # TODO: For some reason the number of nodes is getting duplicated here.
    #  Check why this happens and how to avoid
    #  (does not happen always so something to do with UlanBatar data.)
    # assert nx.number_of_nodes(g) == n_nodes

    # Ensure that all attributes were transfered to graph
    ecolumns = edges.columns
    ncolumns = nodes.columns

    for fr, to, attr in g.edges(data=True):
        eattributes = list(attr.keys())
        break

    for id, attr in g.nodes(data=True):
        nattributes = list(attr.keys())
        break

    for col in ecolumns:
        assert col in eattributes

    for col in ncolumns:
        assert col in nattributes


def test_nxgraph_immutable_counts(test_pbf):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_networkx
    import networkx as nx
    from pyrosm import OSM

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    g = to_networkx(nodes, edges, retain_all=True, simplify=False)
    n_nodes = len(nodes)

    assert isinstance(g, nx.MultiDiGraph)
    # Check that the edge count matches
    assert nx.number_of_edges(g) == 2050
    assert nx.number_of_nodes(g) == n_nodes


def test_directed_edge_generator(test_pbf):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import generate_directed_edges
    from pyrosm import OSM

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)

    # Calculate the number of edges that should be oneway + bidirectional
    mask = edges[oneway_col].isin(oneway_values)
    oneway_edge_cnt = len(edges.loc[mask])
    twoway_edge_cnt = len(edges.loc[~mask])

    # Bidirectional edges
    bidir_edges = generate_directed_edges(
        edges,
        direction="oneway",
        direction_suffix=None,
        from_id_col="u",
        to_id_col="v",
        force_bidirectional=True,
    )

    assert len(bidir_edges) == 2 * len(edges)

    # Directed edges according the rules in "oneway" column
    dir_edges = generate_directed_edges(
        edges,
        direction="oneway",
        direction_suffix=None,
        from_id_col="u",
        to_id_col="v",
        force_bidirectional=False,
    )

    assert len(dir_edges) == oneway_edge_cnt + twoway_edge_cnt * 2


def test_connected_component(immutable_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import generate_directed_edges
    from pyrosm.graph_connectivity import get_connected_edges

    nodes, edges = immutable_nodes_and_edges

    # Bidirectional edges
    bidir_edges = generate_directed_edges(
        edges,
        direction="oneway",
        direction_suffix=None,
        from_id_col="u",
        to_id_col="v",
        force_bidirectional=True,
    )

    # After filtering the unconnected edges, the number of edges/nodes should always be lower (or equal)
    cn, ce = get_connected_edges(nodes, bidir_edges, "u", "v", "id")
    assert len(ce) <= len(bidir_edges)
    assert len(cn) <= len(nodes)

    # Test exact shape
    assert ce.shape == (1650, 20)
    assert cn.shape == (781, 9)


def test_igraph_connectivity(immutable_nodes_and_edges):
    from pyrosm.graphs import to_igraph
    import igraph
    import numpy as np

    nodes, edges = immutable_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=False, simplify=False)

    # Test that graph source and target nodes matches with the ones in attribute table
    for edge in g.es:
        assert edge.source == edge.attributes()["u_seq"]
        assert edge.target == edge.attributes()["v_seq"]

    # Test that finding shortest paths works for all nodes
    N = g.vcount()
    shortest_paths = g.distances(
        source=5, target=[i for i in range(N)], weights="length"
    )

    # Check couple of exact lengths (allow some flexibility due to floating point calculations)
    assert round(shortest_paths[0][0], 0) in [499, 500]
    assert round(shortest_paths[0][-1], 0) in [2802, 2803]

    # Check summaries
    arr = np.array(shortest_paths[0])
    arr[arr == np.inf] = 0
    assert arr.min() == 0
    assert arr.max().round(0) == 3126
    assert arr.mean().round(0) == 1429


def test_nxgraph_connectivity(immutable_nodes_and_edges):
    from pyrosm.graphs import to_networkx
    import networkx as nx
    import numpy as np

    nodes, edges = immutable_nodes_and_edges
    g = to_networkx(nodes, edges, retain_all=False, simplify=False)

    # Test that graph source and target nodes matches with the ones in attribute table
    for fr, to, edge in g.edges(data=True):
        assert fr == edge["u"]
        assert to == edge["v"]

    # Test that finding shortest paths works for all nodes
    node_ids = [n for n in g.nodes()]
    source = node_ids[5]
    shortest_paths = []
    for target in node_ids:
        shortest_path_length = nx.shortest_path_length(
            g, source=source, target=target, weight="length"
        )
        shortest_paths.append(shortest_path_length)

    # Check couple of exact lengths
    assert round(shortest_paths[0], 0) in [499, 500]
    assert round(shortest_paths[-1], 0) in [2802, 2803]

    # Check summaries
    arr = np.array(shortest_paths)
    arr[arr == np.inf] = 0
    assert arr.min() == 0
    assert arr.max().round(0) == 3126
    assert arr.mean().round(0) == 1429


@_pandana_unsupported_on_win
def test_pdgraph_connectivity():
    """Pandana graph export.

    Skipped when ``pandana`` is not installed (it has no Python 3.13 build on
    conda-forge yet), so the suite stays green on Python 3.13.
    """
    pytest.importorskip("pandana")
    from pyrosm.graphs import to_pandana
    import pandas as pd
    from pyrosm import OSM

    osm = OSM(get_data("helsinki_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    # Prerare some test data for aggregations
    restaurants = osm.get_pois(custom_filter={"amenity": ["restaurant"]})
    restaurants = restaurants.loc[restaurants["osm_type"] == "node"]
    restaurants["employee_cnt"] = 1
    x = restaurants["lon"]
    y = restaurants["lat"]

    g = to_pandana(nodes, edges, retain_all=False, simplify=False)

    # Nodes and edges should be in DataFrames
    assert isinstance(g.nodes_df, pd.DataFrame)
    assert isinstance(g.edges_df, pd.DataFrame)

    # Precompute up to 1000 meters
    g.precompute(1000)

    # Link restaurants to graph
    g.set_pois("restaurants", 1000, 5, x, y)

    # Find the distance to nearest 5 restaurants from each node
    nearest_restaurants = g.nearest_pois(1000, "restaurants", num_pois=5)
    assert isinstance(nearest_restaurants, pd.DataFrame)
    assert nearest_restaurants.shape == (5262, 5)

    # Get closest node_ids for each restaurant
    node_ids = g.get_node_ids(x, y)
    assert isinstance(node_ids, pd.Series)
    assert node_ids.min() > 0
    restaurants["node_id"] = node_ids

    # Attach employee counts to the graph
    g.set(node_ids, variable=restaurants.employee_cnt, name="employee_cnt")

    # Aggregate the number of employees within 500 meters from each node
    access = g.aggregate(500, type="sum", decay="linear", name="employee_cnt")
    assert isinstance(access, pd.Series)
    assert len(access) == 5262

    # Test shortest path calculations
    shortest_distances = g.shortest_path_lengths(
        node_ids[0:100], node_ids[100:200], imp_name="length"
    )
    assert isinstance(shortest_distances, list)
    assert len(shortest_distances) == 100
    shortest_distances = pd.Series(shortest_distances)
    assert round(shortest_distances.min(), 0) == 22
    assert round(shortest_distances.max(), 0) in [2457, 2458]
    assert round(shortest_distances.mean(), 0) in [865, 866, 867]


@_pandana_unsupported_on_win
def test_to_graph_api(test_pbf):
    """Smoke-test the to_graph() API for igraph, networkx and pandana.

    Skipped when ``pandana`` is not installed (no Python 3.13 build on
    conda-forge yet); igraph and networkx exports are covered by other tests.
    """
    pytest.importorskip("pandana")
    from pyrosm import OSM
    import networkx as nx
    import igraph
    import pandana

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    # igraph is the default
    ig = osm.to_graph(nodes, edges)
    nxg = osm.to_graph(nodes, edges, graph_type="networkx")
    pdg = osm.to_graph(nodes, edges, graph_type="pandana")
    assert isinstance(nxg, nx.MultiDiGraph)
    assert isinstance(ig, igraph.Graph)
    assert isinstance(pdg, pandana.Network)


def test_graph_exports_correct_number_of_nodes(test_pbf):
    """
    Check issue: #97
    """
    from pyrosm import OSM

    osm = OSM(test_pbf)
    # NetworkX
    nodes, edges = osm.get_network(nodes=True)
    node_cnt = len(nodes)
    nxg = osm.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        osmnx_compatible=False,
        retain_all=True,
        simplify=False,
    )
    assert node_cnt == nxg.number_of_nodes()


def test_graph_export_works_without_oneway_column(test_pbf):
    """
    Check issue: #100
    """
    from pyrosm import OSM

    osm = OSM(test_pbf)
    # NetworkX
    nodes, edges = osm.get_network(nodes=True)
    # Drop "oneway" column to test
    edges = edges.drop("oneway", axis=1)

    with pytest.warns(UserWarning) as w:
        nxg = osm.to_graph(nodes, edges, graph_type="networkx")
        # Check the warning text
        if "missing in the edges" in str(w):
            pass


def test_nxgraph_export_from_osh(helsinki_history_pbf):
    from pyrosm import OSM
    from pyrosm.utils import datetime_to_unix_time
    import pandas as pd
    from geopandas import GeoDataFrame
    from shapely.geometry import MultiLineString
    import networkx as nx

    timestamp = "2010-01-01"
    osm = OSM(filepath=helsinki_history_pbf)
    nodes, edges = osm.get_network(timestamp=timestamp, nodes=True)

    g = osm.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        retain_all=False,
        osmnx_compatible=True,
        simplify=False,
    )
    assert isinstance(g, nx.MultiDiGraph)

    # Test that graph source and target nodes matches with the ones in attribute table
    for fr, to, edge in g.edges(data=True):
        assert fr == edge["u"]
        assert to == edge["v"]

    # Test that finding shortest paths works for all nodes
    source = 25292470
    shortest_paths = {
        target: nx.shortest_path_length(g, source=source, target=target, weight="length")
        for target in g.nodes()
    }

    # Check couple of exact lengths
    # Windows gives a slightly different result
    # most likely due to float handling differences between Unix and Windows
    assert round(shortest_paths[60072314], 0) in [478, 470]
    assert round(shortest_paths[472662939], 0) in [810]


def _edge_pairs(directed_edges, from_id_col="u", to_id_col="v"):
    return set(zip(directed_edges[from_id_col], directed_edges[to_id_col]))


def test_generate_directed_edges_honours_oneway_bicycle():
    """generate_directed_edges: oneway:bicycle overrides oneway per edge.

    Covers a two-way street, a one-way street, a contraflow street
    (oneway=yes + oneway:bicycle=no -> two-way for bikes), a bike-only one-way
    (oneway:bicycle=yes), and a reverse one-way (oneway=-1).
    """
    import pandas as pd
    from pyrosm.graph_export import generate_directed_edges

    edges = pd.DataFrame(
        {
            "u": [1, 3, 5, 7, 9],
            "v": [2, 4, 6, 8, 10],
            "oneway": [None, "yes", "yes", None, "-1"],
            "oneway:bicycle": [None, None, "no", "yes", None],
        }
    )
    out = generate_directed_edges(edges, "oneway", "bicycle", "u", "v", False)
    pairs = _edge_pairs(out)

    # 2 (two-way) + 1 + 2 (contraflow) + 1 + 1 = 7 directed edges
    assert len(out) == 7
    # two-way street -> both directions
    assert (1, 2) in pairs and (2, 1) in pairs
    # one-way street -> single direction
    assert (3, 4) in pairs and (4, 3) not in pairs
    # contraflow (oneway=yes, oneway:bicycle=no) -> both directions for bikes
    assert (5, 6) in pairs and (6, 5) in pairs
    # bike-only one-way (oneway:bicycle=yes) -> single direction
    assert (7, 8) in pairs and (8, 7) not in pairs
    # reverse one-way (oneway=-1) -> flipped
    assert (10, 9) in pairs and (9, 10) not in pairs


def test_generate_directed_edges_without_suffix_is_oneway_only():
    """Without a direction_suffix the base 'oneway' column drives everything
    (the driving path is unaffected by the oneway:bicycle change)."""
    import pandas as pd
    from pyrosm.graph_export import generate_directed_edges

    edges = pd.DataFrame(
        {
            "u": [1, 3],
            "v": [2, 4],
            "oneway": [None, "yes"],
            # present but must be ignored when no suffix is requested
            "oneway:bicycle": ["no", "no"],
        }
    )
    out = generate_directed_edges(edges, "oneway", None, "u", "v", False)
    pairs = _edge_pairs(out)
    assert len(out) == 3  # two-way + one-way
    assert (1, 2) in pairs and (2, 1) in pairs
    assert (3, 4) in pairs and (4, 3) not in pairs


def _toy_network():
    import geopandas as gpd
    from shapely.geometry import Point, LineString

    nodes = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs="epsg:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "u": [1, 2, 3],
            "v": [2, 3, 4],
            "oneway": [None, "yes", "yes"],
            "oneway:bicycle": [None, None, "no"],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
        crs="epsg:4326",
    )
    return nodes, edges


def test_get_directed_edges_cycling_is_directed_with_contraflow():
    """Cycling is directed by default and honours oneway:bicycle; walking stays
    bidirectional."""
    from pyrosm.graphs import get_directed_edges

    # twoway + oneway + contraflow
    nodes, edges = _toy_network()
    _, cyc = get_directed_edges(nodes, edges, network_type="cycling")
    pairs = _edge_pairs(cyc)
    assert len(cyc) == 5  # 2 + 1 + 2
    assert (2, 3) in pairs and (3, 2) not in pairs  # one-way respected
    assert (3, 4) in pairs and (4, 3) in pairs  # contraflow both ways

    # Walking ignores oneway entirely (bidirectional)
    nodes, edges = _toy_network()
    _, walk = get_directed_edges(nodes, edges, network_type="walking")
    assert len(walk) == 6  # every edge both ways


def test_get_directed_edges_explicit_oneway_bicycle_direction():
    """An explicit direction='oneway:bicycle' is split into base + override.

    This exercises the "<base>:<suffix>" parsing path independently of the
    cycling auto-default (here the network type is driving).
    """
    from pyrosm.graphs import get_directed_edges

    nodes, edges = _toy_network()
    _, out = get_directed_edges(
        nodes, edges, direction="oneway:bicycle", network_type="driving"
    )
    pairs = _edge_pairs(out)
    assert len(out) == 5  # twoway(2) + oneway(1) + contraflow(2)
    assert (2, 3) in pairs and (3, 2) not in pairs  # base oneway respected
    assert (3, 4) in pairs and (4, 3) in pairs  # oneway:bicycle=no -> both ways


def test_pandarm_connectivity():
    """pandarm graph export (#270).

    pandarm is the maintained, NumPy 2-compatible fork of pandana, so unlike the
    pandana tests this is NOT skipped on Windows. Skipped only when pandarm is
    not installed.
    """
    pytest.importorskip("pandarm")
    from pyrosm.graphs import to_pandarm
    import pandas as pd
    from pyrosm import OSM

    osm = OSM(get_data("helsinki_pbf"))
    nodes, edges = osm.get_network(nodes=True)

    restaurants = osm.get_pois(custom_filter={"amenity": ["restaurant"]})
    restaurants = restaurants.loc[restaurants["osm_type"] == "node"]
    restaurants["employee_cnt"] = 1
    x = restaurants["lon"]
    y = restaurants["lat"]

    g = to_pandarm(nodes, edges, retain_all=False, simplify=False)

    assert isinstance(g.nodes_df, pd.DataFrame)
    assert isinstance(g.edges_df, pd.DataFrame)

    g.precompute(1000)
    g.set_pois("restaurants", 1000, 5, x, y)

    nearest_restaurants = g.nearest_pois(1000, "restaurants", num_pois=5)
    assert isinstance(nearest_restaurants, pd.DataFrame)
    assert nearest_restaurants.shape == (5262, 5)

    node_ids = g.get_node_ids(x, y)
    assert isinstance(node_ids, pd.Series)
    assert node_ids.min() > 0
    restaurants["node_id"] = node_ids

    g.set(node_ids, variable=restaurants.employee_cnt, name="employee_cnt")

    access = g.aggregate(500, type="sum", decay="linear", name="employee_cnt")
    assert isinstance(access, pd.Series)
    assert len(access) == 5262

    shortest_distances = g.shortest_path_lengths(
        node_ids[0:100], node_ids[100:200], imp_name="length"
    )
    assert isinstance(shortest_distances, list)
    assert len(shortest_distances) == 100
    shortest_distances = pd.Series(shortest_distances)
    assert round(shortest_distances.min(), 0) == 22
    assert round(shortest_distances.max(), 0) in [2457, 2458]
    assert round(shortest_distances.mean(), 0) in [865, 866, 867]


def test_to_graph_api_pandarm(test_pbf):
    """Smoke-test to_graph(graph_type="pandarm") (#270). Not skipped on Windows."""
    pytest.importorskip("pandarm")
    from pyrosm import OSM
    import pandarm

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    pdg = osm.to_graph(nodes, edges, graph_type="pandarm")
    assert isinstance(pdg, pandarm.Network)


def test_validate_graph_type_rejects_unknown_type():
    """#270 — validate_graph_type raises for unknown graph types and lists the
    supported ones (incl. pandarm)."""
    from pyrosm.utils import validate_graph_type

    with pytest.raises(ValueError, match="pandarm"):
        validate_graph_type("bogus")


def test_to_graph_pandarm_retain_all(test_pbf):
    """#270 — pandarm export with retain_all=True skips the connected-component
    pruning and still builds a Network."""
    pytest.importorskip("pandarm")
    from pyrosm import OSM
    import pandarm

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    g = osm.to_graph(nodes, edges, graph_type="pandarm", retain_all=True)
    assert isinstance(g, pandarm.Network)


def test_has_pandarm_false_without_pandarm(monkeypatch):
    """#270 — _compat.HAS_PANDARM falls back to False when pandarm is absent."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandarm":
            raise ImportError("simulated missing pandarm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import pyrosm.utils._compat as compat

    reloaded = importlib.reload(compat)
    try:
        assert reloaded.HAS_PANDARM is False
    finally:
        monkeypatch.undo()
        importlib.reload(compat)


@pytest.mark.parametrize(
    "flag,graph_type,message",
    [
        ("HAS_IGRAPH", "igraph", "python-igraph"),
        ("HAS_NETWORKX", "networkx", "networkx"),
        ("HAS_PANDANA", "pandana", "pandana"),
        ("HAS_PANDARM", "pandarm", "pandarm"),
    ],
)
def test_to_graph_raises_when_backend_missing(
    test_pbf, monkeypatch, flag, graph_type, message
):
    """#272 — each backend export raises an ImportError naming the missing
    optional dependency."""
    import warnings
    from pyrosm import OSM
    import pyrosm.graph_export as ge

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)

    monkeypatch.setattr(ge, flag, False)
    with pytest.raises(ImportError, match=message):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pandana emits a DeprecationWarning
            osm.to_graph(nodes, edges, graph_type=graph_type)


def test_graph_export_skips_edges_with_missing_nodes(test_pbf):
    """#272 — the igraph/networkx builders skip edges whose endpoint node is
    absent from the nodes frame (KeyError -> continue) instead of failing. This
    is the "data was cropped manually" branch; the builders are exercised
    directly with a real network minus one referenced node, since the public
    to_graph pipeline reconciles nodes/edges before they reach the builders."""
    from pyrosm import OSM
    from pyrosm.graph_export import _create_igraph, _create_nxgraph

    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)

    # Drop a node referenced by an edge -> dangling endpoint the builder skips.
    referenced = set(edges["u"]) | set(edges["v"])
    drop_id = next(i for i in nodes["id"].tolist() if i in referenced)
    nodes_missing = nodes[nodes["id"] != drop_id].reset_index(drop=True)

    pytest.importorskip("igraph")
    ig = _create_igraph(nodes_missing, edges, "u", "v", "id")
    assert ig is not None

    pytest.importorskip("networkx")
    nxg = _create_nxgraph(nodes_missing, edges, "u", "v", "id")
    assert nxg is not None


@pytest.mark.parametrize(
    "modname,flag",
    [
        ("igraph", "HAS_IGRAPH"),
        ("networkx", "HAS_NETWORKX"),
        ("pandana", "HAS_PANDANA"),
    ],
)
def test_compat_flag_false_without_dependency(monkeypatch, modname, flag):
    """#272 — _compat.HAS_* falls back to False when the optional backend is
    not importable (the pandarm case is covered by
    test_has_pandarm_false_without_pandarm above in this file)."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == modname:
            raise ImportError("simulated missing %s" % modname)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import pyrosm.utils._compat as compat

    reloaded = importlib.reload(compat)
    try:
        assert getattr(reloaded, flag) is False
    finally:
        monkeypatch.undo()
        importlib.reload(compat)


def _comparable(value):
    """``value`` with NaN as a string and geometries as WKB, so that ``==`` compares two
    graphs' attributes, in order."""
    import math
    import numpy as np
    import shapely

    if isinstance(value, dict):
        return [(k, _comparable(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_comparable(v) for v in value]
    if isinstance(value, shapely.Geometry):
        return shapely.to_wkb(value)
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def _graph_contents(graph):
    """The nodes and edges of a graph of any exported type, with their attributes."""
    if hasattr(graph, "get_edgelist"):
        vertices = [v.attributes() for v in graph.vs]
        return graph.get_edgelist(), vertices, [e.attributes() for e in graph.es]
    if hasattr(graph, "nodes_df"):
        return graph.nodes_df.to_dict("list"), graph.edges_df.to_dict("list")
    return graph.graph, list(graph.nodes(data=True)), list(graph.edges(keys=True, data=True))


@pytest.mark.parametrize(
    "osm_kwargs, network_type",
    [({}, "driving"), ({"bounding_box": [26.94, 60.525, 26.96, 60.535], "keep_node_info": True}, "walking")],
)
def test_graph_network_reads_the_network_tables(test_pbf, osm_kwargs, network_type):
    """The readers of a graph read hold the tables of get_network(nodes=True)."""
    import numpy as np
    from geopandas.testing import assert_geodataframe_equal
    from pyrosm import OSM
    from pyrosm.engine.readers import graph_network

    nodes, edges = OSM(test_pbf, **osm_kwargs).get_network(network_type, nodes=True)
    node_reader, edge_reader = graph_network(test_pbf, network_type, **osm_kwargs)
    segments = np.arange(len(edge_reader.way_of))
    assert_geodataframe_equal(edge_reader.rows(segments), edges, check_crs=False)
    assert_geodataframe_equal(node_reader.table(), nodes)
    assert_geodataframe_equal(node_reader.empty(), nodes.iloc[:0])


@pytest.mark.parametrize(
    "graph_type, osm_kwargs, kwargs",
    [
        (
            "networkx",
            {},
            {
                "network_type": "driving",
                "extra_attributes": ["lit"],
                "custom_filter": {"highway": ["residential", "service", "tertiary"]},
                "filter_type": "keep",
            },
        ),
        (
            "networkx",
            {"keep_metadata": False},
            {
                "network_type": "drive",
                "force_bidirectional": True,
                "simplify_kwargs": {"length_cols": (), "track_merged": True},
            },
        ),
        (
            "networkx",
            {},
            {
                "network_type": "walking",
                "osmnx_compatible": False,
                "tags_to_keep": ["highway", "name"],
                "simplify_kwargs": {
                    "node_attrs_include": ["tags"],
                    "edge_attrs_differ": ("highway", "geometry"),
                },
            },
        ),
        ("igraph", {}, {"network_type": "cycling", "retain_all": True}),
        ("pandarm", {}, {"network_type": "driving"}),
        pytest.param("pandana", {}, {"network_type": "walking"}, marks=_pandana_unsupported_on_win),
        ("networkx", {}, {"network_type": "driving", "simplify": False}),
        ("networkx", {"engine": "in_memory"}, {"network_type": "driving"}),
        ("networkx", {}, {"network_type": "driving", "timestamp": "2010-01-01"}),
    ],
)
def test_get_graph_is_the_exported_network(request, graph_type, osm_kwargs, kwargs):
    """get_graph builds the graph that to_graph builds from get_network(nodes=True)."""
    import warnings
    from pyrosm import OSM

    if graph_type in ("pandarm", "pandana"):
        pytest.importorskip(graph_type)
    history = "timestamp" in kwargs
    pbf = request.getfixturevalue("helsinki_history_pbf" if history else "test_pbf")
    read_keys = (
        "network_type",
        "tags_to_keep",
        "extra_attributes",
        "custom_filter",
        "filter_type",
        "timestamp",
    )
    read = {k: v for k, v in kwargs.items() if k in read_keys}
    export = {k: v for k, v in kwargs.items() if k not in read_keys}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nodes, edges = OSM(pbf, **osm_kwargs).get_network(nodes=True, **read)
        expected = OSM.to_graph(nodes, edges, graph_type, **export)
        graph = OSM(pbf, **osm_kwargs).get_graph(graph_type=graph_type, **kwargs)
    assert _comparable(_graph_contents(graph)) == _comparable(_graph_contents(expected))


@pytest.mark.parametrize("engine", ["out_of_core", "in_memory"])
def test_get_graph_of_an_empty_network_is_none(test_pbf, engine):
    from pyrosm import OSM

    osm = OSM(test_pbf, engine=engine)
    with pytest.warns(UserWarning):
        graph = osm.get_graph(custom_filter={"highway": ["no-such-road"]}, filter_type="keep")
    assert graph is None
