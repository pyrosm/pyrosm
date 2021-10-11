from pyrosm.graph_export import (
    _create_igraph,
    _create_nxgraph,
    _create_pdgraph,
    generate_directed_edges,
)
from pyrosm.graph_connectivity import get_connected_edges
from pyrosm.utils import validate_edge_gdf, validate_node_gdf
from pyrosm.config import Conf
import warnings


def get_directed_edges(
    nodes,
    edges,
    direction="oneway",
    from_id_col="u",
    to_id_col="v",
    node_id_col="id",
    force_bidirectional=False,
    network_type=None,
):
    """Prepares the edges and nodes for exporting to different graphs."""
    allowed_network_types = Conf._possible_network_filters

    # Validate nodes and edges
    validate_node_gdf(nodes)
    validate_edge_gdf(edges)

    for col in [from_id_col, to_id_col]:
        if col not in edges.columns:
            raise ValueError(
                "Required column '{col}' does not exist in edges.".format(col=col)
            )

    if direction not in edges.columns:
        warnings.warn(
            f"Column '{direction}' missing in the edges GeoDataFrame. "
            f"Assuming all edges to be bidirectional "
            f"(travel allowed to both directions).",
            UserWarning,
            stacklevel=2,
        )
        edges[direction] = None

    if node_id_col not in nodes.columns:
        raise ValueError(
            "Required column '{col}' does not exist in nodes.".format(col=node_id_col)
        )

    # Check the network_type
    if network_type is not None:
        net_type = network_type
    else:
        net_type = edges._metadata[-1]

    # Check if network type is stored with edges or nodes
    if net_type not in allowed_network_types:
        net_type = nodes._metadata[-1]
        if net_type not in allowed_network_types:
            txt = ", ".join(allowed_network_types)
            raise ValueError(
                "Could not detect the network type from the edges. "
                "In order to save the graph, specify the type of your network"
                "with 'network_type' -parameter."
                "Possible network types are: " + txt
            )

    edges = edges.copy()
    nodes = nodes.copy()

    # Generate directed edges
    # Check if user wants to force bidirectional graph
    # or if the graph is walking, cycling or all
    if force_bidirectional or net_type in ["walking", "cycling", "all"]:
        edges = generate_directed_edges(
            edges, direction, from_id_col, to_id_col, force_bidirectional=True
        )
    else:
        edges = generate_directed_edges(
            edges, direction, from_id_col, to_id_col, force_bidirectional=False
        )

    return nodes, edges


def to_networkx(
    nodes,
    edges,
    direction="oneway",
    from_id_col="u",
    to_id_col="v",
    edge_id_col="id",
    node_id_col="id",
    force_bidirectional=False,
    network_type=None,
    retain_all=False,
    osmnx_compatible=True,
):
    """
    Creates a NetworkX.MultiDiGraph from given OSM GeoDataFrame.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing road network data.

    network_type : str
        The type of the network. Possible values:
              - `'walking'`
              - `'cycling'`
              - `'driving'`
              - `'driving+service'`
              - `'all'`.

    direction : str
        Name for the column containing information about the allowed driving directions

    from_id_col : str
        Name for the column having the from-node-ids of edges.

    to_id_col : str
        Name for the column having the to-node-ids of edges.

    edge_id_col : str
        Name for the column having the unique id for edges.

    node_id_col : str
        Name for the column having the unique id for nodes.

    force_bidirectional : bool
        If True, all edges will be created as bidirectional (allow travel to both directions).

    network_type : str
        Network type for the given data. Determines how the graph will be constructed.
        By default, bidirectional graph is created for walking, cycling and all,
        and directed graph for driving (i.e. oneway streets are taken into account).
        Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'.

    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the connected edges.

    osmnx_compatible : bool (default True)
        if True, modifies the edge and node-attribute naming to be compatible with OSMnx
        (allows utilizing all OSMnx functionalities).

    Returns
    -------
    networkx.MultiDiGraph

    """

    # Prepare the data
    nodes, edges = get_directed_edges(
        nodes,
        edges,
        direction,
        from_id_col,
        to_id_col,
        node_id_col,
        force_bidirectional,
        network_type,
    )

    # Keep only strongly connected component if not specifically requested otherwise
    if not retain_all:
        nodes, edges = get_connected_edges(
            nodes, edges, from_id_col, to_id_col, node_id_col
        )

    if osmnx_compatible:
        # add 'key' attribute which is needed by OSMnx
        if "key" not in edges.columns:
            edges["key"] = 0

        # Follow the naming convention of OSMnx
        nodes = nodes.rename(columns={node_id_col: "osmid", "lat": "y", "lon": "x"})
        edges = edges.rename(columns={edge_id_col: "osmid"})
        node_id_col = "osmid"

    # Add node-id as index
    nodes = nodes.set_index(node_id_col, drop=False)
    nodes = nodes.rename_axis([None])

    # Create NetworkX graph
    return _create_nxgraph(nodes, edges, from_id_col, to_id_col, node_id_col)


def to_igraph(
    nodes,
    edges,
    direction="oneway",
    from_id_col="u",
    to_id_col="v",
    node_id_col="id",
    force_bidirectional=False,
    network_type=None,
    retain_all=False,
):
    """
    Creates an iGraph from given OSM GeoDataFrame.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing road network data.

    network_type : str
        The type of the network. Possible values:
              - `'walking'`
              - `'cycling'`
              - `'driving'`
              - `'driving+service'`
              - `'all'`.

    direction : str
        Name for the column containing information about the allowed driving directions

    from_id_col : str
        Name for the column having the from-node-ids of edges.

    to_id_col : str
        Name for the column having the to-node-ids of edges.

    edge_id_col : str
        Name for the column having the unique id for edges.

    node_id_col : str
        Name for the column having the unique id for nodes.

    force_bidirectional : bool
        If True, all edges will be created as bidirectional (allow travel to both directions).

    network_type : str
        Network type for the given data. Determines how the graph will be constructed.
        By default, bidirectional graph is created for walking, cycling and all,
        and directed graph for driving (i.e. oneway streets are taken into account).
        Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'.

    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the connected edges.

    Returns
    -------
    igraph.Graph

    """
    # Prepare the data
    nodes, edges = get_directed_edges(
        nodes,
        edges,
        direction,
        from_id_col,
        to_id_col,
        node_id_col,
        force_bidirectional,
        network_type,
    )

    # Keep only strongly connected component if not specifically requested otherwise
    if not retain_all:
        nodes, edges = get_connected_edges(
            nodes, edges, from_id_col, to_id_col, node_id_col
        )

    return _create_igraph(nodes, edges, from_id_col, to_id_col, node_id_col)


def to_pandana(
    nodes,
    edges,
    direction="oneway",
    from_id_col="u",
    to_id_col="v",
    node_id_col="id",
    force_bidirectional=False,
    network_type=None,
    retain_all=False,
    weight_cols=["length"],
):
    # Prepare the data
    nodes, edges = get_directed_edges(
        nodes,
        edges,
        direction,
        from_id_col,
        to_id_col,
        node_id_col,
        force_bidirectional,
        network_type,
    )

    # Keep only strongly connected component if not specifically requested otherwise
    if not retain_all:
        nodes, edges = get_connected_edges(
            nodes, edges, from_id_col, to_id_col, node_id_col
        )

    nodes = nodes.rename(columns={"lat": "y", "lon": "x"})
    nodes = nodes.set_index("id", drop=False)
    nodes = nodes.rename_axis([None])

    return _create_pdgraph(nodes, edges, from_id_col, to_id_col, weight_cols)
