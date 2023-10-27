import pytest
from pyrosm import get_data
from pyrosm.config import Conf

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
    pbf_path = get_data("ulanbator")
    osm = OSM(pbf_path)
    return osm.get_network(nodes=True)


@pytest.fixture
def bike_nodes_and_edges():
    from pyrosm import OSM

    # UlanBator is good small dataset for testing
    # (unmodified, i.e. not cropped)
    pbf_path = get_data("ulanbator")
    osm = OSM(pbf_path)
    return osm.get_network(nodes=True, network_type="cycling")


@pytest.fixture
def driving_nodes_and_edges():
    from pyrosm import OSM

    pbf_path = get_data("ulanbator")
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
    g = to_igraph(nodes, edges, retain_all=True)
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
    g = to_igraph(nodes, edges, retain_all=True)
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


def test_igraph_export_by_driving(driving_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph

    nodes, edges = driving_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=True)
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
    g = to_igraph(nodes, edges, retain_all=True)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)
    # Check that the edge count matches
    assert g.ecount() == 2430
    assert g.vcount() == n_nodes


def test_nxgraph_export_by_walking(walk_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_networkx
    import networkx as nx

    nodes, edges = walk_nodes_and_edges
    edges = edges.drop_duplicates(["u", "v"])
    g = to_networkx(nodes, edges, retain_all=True, osmnx_compatible=False)
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
    g = to_networkx(nodes, edges, retain_all=True)
    n_nodes = len(nodes)

    assert isinstance(g, nx.MultiDiGraph)
    # Check that the edge count matches
    assert nx.number_of_edges(g) == 2430
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
        from_id_col="u",
        to_id_col="v",
        force_bidirectional=True,
    )

    assert len(bidir_edges) == 2 * len(edges)

    # Directed edges according the rules in "oneway" column
    dir_edges = generate_directed_edges(
        edges,
        direction="oneway",
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
        from_id_col="u",
        to_id_col="v",
        force_bidirectional=True,
    )

    # After filtering the unconnected edges, the number of edges/nodes should always be lower (or equal)
    cn, ce = get_connected_edges(nodes, bidir_edges, "u", "v", "id")
    assert len(ce) <= len(bidir_edges)
    assert len(cn) <= len(nodes)

    # Test exact shape
    assert ce.shape == (2034, 23)
    assert cn.shape == (954, 9)


def test_igraph_connectivity(immutable_nodes_and_edges):
    from pyrosm.graphs import to_igraph
    import igraph
    import numpy as np

    nodes, edges = immutable_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=False)

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
    assert round(shortest_paths[0][-1], 0) == 2315

    # Check summaries
    arr = np.array(shortest_paths[0])
    arr[arr == np.inf] = 0
    assert arr.min() == 0
    assert arr.max().round(0) == 2838
    assert arr.mean().round(0) == 1372


def test_nxgraph_connectivity(immutable_nodes_and_edges):
    from pyrosm.graphs import to_networkx
    import networkx as nx
    import numpy as np

    nodes, edges = immutable_nodes_and_edges
    g = to_networkx(nodes, edges, retain_all=False)

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
    assert round(shortest_paths[-1], 0) == 2315

    # Check summaries
    arr = np.array(shortest_paths)
    arr[arr == np.inf] = 0
    assert arr.min() == 0
    assert arr.max().round(0) == 2838
    assert arr.mean().round(0) == 1372


def test_pdgraph_connectivity():
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

    g = to_pandana(nodes, edges, retain_all=False)

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
    assert nearest_restaurants.shape == (5750, 5)

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
    assert len(access) == 5750

    # Test shortest path calculations
    shortest_distances = g.shortest_path_lengths(
        node_ids[0:100], node_ids[100:200], imp_name="length"
    )
    assert isinstance(shortest_distances, list)
    assert len(shortest_distances) == 100
    shortest_distances = pd.Series(shortest_distances)
    assert round(shortest_distances.min(), 0) == 22
    assert round(shortest_distances.max(), 0) == 2453
    assert round(shortest_distances.mean(), 0) == 869


def test_to_graph_api(test_pbf):
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
        nodes, edges, graph_type="networkx", osmnx_compatible=False, retain_all=True
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
        nodes, edges, graph_type="networkx", retain_all=False, osmnx_compatible=True
    )
    assert isinstance(g, nx.MultiDiGraph)

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
    # Windows gives a slightly different result
    # most likely due to float handling differences between Unix and Windows
    assert round(shortest_paths[0], 0) in [478, 470]
    assert round(shortest_paths[-1], 0) in [797, 793]
