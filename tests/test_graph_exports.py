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
def walk_nodes_and_edges():
    from pyrosm import OSM
    # UlanBator is good small dataset for testing
    # (unmodified, i.e. not cropped)
    pbf_path = get_data("ulanbator")
    osm = OSM(pbf_path)
    return osm.get_network(to_graph=True)


@pytest.fixture
def bike_nodes_and_edges():
    from pyrosm import OSM
    # UlanBator is good small dataset for testing
    # (unmodified, i.e. not cropped)
    pbf_path = get_data("ulanbator")
    osm = OSM(pbf_path)
    return osm.get_network(to_graph=True, network_type="cycling")


@pytest.fixture
def driving_nodes_and_edges():
    from pyrosm import OSM
    pbf_path = get_data("ulanbator")
    osm = OSM(pbf_path)
    return osm.get_network(network_type="driving", to_graph=True)


@pytest.fixture
def immutable_nodes_and_edges():
    from pyrosm import OSM
    pbf_path = get_data("test_pbf")
    osm = OSM(pbf_path)
    return osm.get_network(to_graph=True)


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
    oneway_edge_cnt = len(edges.loc[mask])
    twoway_edge_cnt = len(edges.loc[~mask])

    # Check that the edge count matches
    assert g.ecount() == oneway_edge_cnt + twoway_edge_cnt * 2


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
    nodes, edges = osm.get_network(to_graph=True)
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
    assert abs(1-(nx.number_of_edges(g) / (2 * n_edges))) < 0.001

    # The number of nodes should be the same
    # TODO: For some reason the number of nodes is getting duplicated here.
    #  Check why this happens and how to avoid
    #  (does not happen always so something to do with UlanBatar data.)
    #assert nx.number_of_nodes(g) == n_nodes

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
    nodes, edges = osm.get_network(to_graph=True)
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
    nodes, edges = osm.get_network(to_graph=True)

    # Calculate the number of edges that should be oneway + bidirectional
    mask = edges[oneway_col].isin(oneway_values)
    oneway_edge_cnt = len(edges.loc[mask])
    twoway_edge_cnt = len(edges.loc[~mask])

    # Bidirectional edges
    bidir_edges = generate_directed_edges(edges,
                                          direction="oneway",
                                          from_id_col="u",
                                          to_id_col="v",
                                          force_bidirectional=True
                                          )

    assert len(bidir_edges) == 2 * len(edges)

    # Directed edges according the rules in "oneway" column
    dir_edges = generate_directed_edges(edges,
                                        direction="oneway",
                                        from_id_col="u",
                                        to_id_col="v",
                                        force_bidirectional=False
                                        )

    assert len(dir_edges) == oneway_edge_cnt + twoway_edge_cnt*2


def test_connected_component(immutable_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import generate_directed_edges
    from pyrosm.graph_connectivity import get_connected_edges

    nodes, edges = immutable_nodes_and_edges

    # Bidirectional edges
    bidir_edges = generate_directed_edges(edges,
                                          direction="oneway",
                                          from_id_col="u",
                                          to_id_col="v",
                                          force_bidirectional=True
                                          )

    # After filtering the unconnected edges, the number of edges/nodes should always be lower (or equal)
    cn, ce = get_connected_edges(nodes, bidir_edges, "u", "v", "id")
    assert len(ce) <= len(bidir_edges)
    assert len(cn) <= len(nodes)

    # Test exact shape
    assert ce.shape == (2034, 23)
    assert cn.shape == (954, 8)


def test_igraph_connectivity(immutable_nodes_and_edges):
    from pyrosm.graphs import to_igraph
    import igraph
    import numpy as np

    nodes, edges = immutable_nodes_and_edges
    g = to_igraph(nodes, edges, retain_all=False)

    # Test that graph source and target nodes matches with the ones in attribute table
    for edge in g.es:
        assert edge.source == edge.attributes()['u_seq']
        assert edge.target == edge.attributes()['v_seq']

    # Test that finding shortest paths works for all nodes
    N = g.vcount()
    shortest_paths = g.shortest_paths_dijkstra(source=5, target=[i for i in range(N)], weights='length')

    # Check couple of exact lengths
    assert round(shortest_paths[0][0], 0) == 807
    assert round(shortest_paths[0][-1], 0) == 1940

    # Check summaries
    arr = np.array(shortest_paths[0])
    arr[arr == np.inf] = 0
    assert arr.min() == 0
    assert arr.max().round(0) == 2343
    assert arr.mean().round(0) == 1141