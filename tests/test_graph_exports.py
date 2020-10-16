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


def test_igraph_export_by_walking(walk_nodes_and_edges):
    from geopandas import GeoDataFrame
    from pyrosm.graphs import to_igraph
    import igraph

    nodes, edges = walk_nodes_and_edges
    g = to_igraph(nodes, edges)
    n_edges = len(edges)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)

    # In case of walking/cycling there should be 2x the num of edges
    # as in the orig gdf (one edge for each direction)
    # --> assuming the data has not been cropped (which might drop nodes/edges)
    assert g.ecount() == 2*n_edges

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
    g = to_igraph(nodes, edges)
    n_edges = len(edges)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)

    # In case of walking/cycling there should be 2x the num of edges
    # as in the orig gdf (one edge for each direction)
    # --> assuming the data has not been cropped (which might drop nodes/edges)
    assert g.ecount() == 2*n_edges

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
    g = to_igraph(nodes, edges)
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
    assert g.ecount() == oneway_edge_cnt + twoway_edge_cnt*2


def test_igraph_unmutable_counts(test_pbf):
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
    g = to_igraph(nodes, edges)
    n_nodes = len(nodes)

    assert isinstance(g, igraph.Graph)
    # Check that the edge count matches
    assert g.ecount() == 462
    assert g.vcount() == n_nodes

