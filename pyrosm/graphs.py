from pyrosm.graph_export import _create_igraph
from pyrosm.config import Conf

def to_networkx(nodes, edges, network_type=None):
    pass


def to_pandana(nodes, edges, network_type=None):
    pass


def to_igraph(nodes,
              edges,
              direction='oneway',
              from_id_col='u',
              to_id_col='v',
              force_bidirectional=False,
              network_type=None):
    """
    Creates a Graph from given OSM GeoDataFrame.

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

    force_bidirectional : bool
        If True, all edges will be created as bidirectional (allow travel to both directions).

    network_type : str
        Network type for the given data. Determines how the graph will be constructed.
        By default, bidirectional graph is created for walking, cycling and all,
        and directed graph for driving (i.e. oneway streets are taken into account).
        Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'.
    """
    allowed_network_types = Conf._possible_network_filters

    for col in [direction, from_id_col, to_id_col]:
        if col not in edges.columns:
            raise ValueError(
                "Required column '{col}' does not exist in edges.".format(
                    col=col)
            )

    # Check if user wants to force bidirectional graph
    if force_bidirectional:
        return _create_igraph(nodes, edges, direction, from_id_col, to_id_col, force_bidirectional)

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
            raise ValueError("Could not detect the network type from the edges. "
                             "In order to save the graph, specify the type of your network"
                             "with 'network_type' -parameter."
                             "Possible network types are: " + txt)

    # For cycling, walking and all create bidirectional graph
    if net_type in ["walking", "cycling", "all"]:
        return _create_igraph(nodes, edges, direction, from_id_col, to_id_col, True)

    # Otherwise, create directed graph as specified in OSM
    return _create_igraph(nodes, edges, direction, from_id_col, to_id_col, False)


