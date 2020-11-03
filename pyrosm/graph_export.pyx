from pyrosm.utils._compat import HAS_IGRAPH, HAS_NETWORKX, HAS_PANDANA
from pyrosm.config import Conf
import geopandas as gpd

# The values used to determine oneway road in OSM
oneway_values = Conf.oneway_values

cpdef _create_igraph(nodes,
                     edges,
                     from_id_col,
                     to_id_col,
                     node_id_col):
    """
    Creates a iGraph from directed edges and nodes.
    NOTE: Assumes that the input edges GeoDataFrame is directed.  
    """
    if not HAS_IGRAPH:
        raise ImportError("'python-igraph' needs to be installed "
                          "in order to export the network for igraph.")
    import igraph

    cdef long long i

    nodes = nodes.copy()
    edges = edges.copy()

    from_id_int = from_id_col + "_seq"
    to_id_int = to_id_col + "_seq"

    edge_list = []

    n_edges = len(edges)
    n_nodes = len(nodes)

    # Add columns for sequential ids
    edges[from_id_int] = None
    edges[to_id_int] = None

    # Convert edges to dict
    edges = edges.to_dict(orient="list")

    # Node-ids needs to be sequential for igraph
    nodes = nodes.reset_index(drop=True)
    nodes["node_id"] = nodes.index

    # Prepare dictionary for fast lookups
    node_dict = {k: v for k, v in zip(nodes[node_id_col].to_list(), nodes["node_id"].to_list())}

    # Node attributes
    node_attributes = nodes.to_dict(orient='list')

    # Generate edge dictionary
    for i in range(0, n_edges):

        # Get nodeids for the edge
        # ------------------------
        # Note: In some cases the node for from/to_id might not exist
        # on the "edge" of the network (e.g. if data has been cropped manually).
        try:
            from_node_id = edges[from_id_col][i]
            from_seq_id = node_dict[from_node_id]
        except KeyError:
            continue
        except Exception as e:
            raise e

        try:
            to_node_id = edges[to_id_col][i]
            to_seq_id = node_dict[to_node_id]
        except KeyError:
            continue
        except Exception as e:
            raise e

        # Add sequential ids to edge_list
        edge_list.append([from_seq_id, to_seq_id])

        # Update the edge attributes
        edges[from_id_int][i] = from_seq_id
        edges[to_id_int][i] = to_seq_id

    del node_dict

    # Create directed graph
    graph = igraph.Graph(n=n_nodes, directed=True, edges=edge_list,
                         vertex_attrs=node_attributes,
                         edge_attrs=edges)
    return graph

cpdef _create_nxgraph(nodes,
                      edges,
                      from_id_col,
                      to_id_col,
                      node_id_col):
    """
    Creates a NetworkX graph from directed edges and nodes.
    NOTE: Assumes that the input edges GeoDataFrame is directed.
    """
    if not HAS_NETWORKX:
        raise ImportError("'networkx' needs to be installed "
                          "in order to export the network for networkx / osmnx.")
    import networkx as nx

    cdef long long i

    nodes = nodes.copy()
    edges = edges.copy()

    crs = edges.crs
    n_edges = len(edges)
    edge_list = []

    # Convert edges to dict
    edge_attributes = edges.to_dict(orient="index")

    # Prepare node dictionary for fast lookups
    node_dict = {k: None for k in nodes[node_id_col].to_list()}

    # Node attributes
    node_attributes = nodes.to_dict(orient="index")
    node_attributes = [(k, v) for k, v in node_attributes.items()]

    # Generate edge dictionary
    for i in range(0, n_edges):

        # Get nodeids for the edge
        # ------------------------
        # Note: In some cases the node for from/to_id might not exist
        # on the "edge" of the network (e.g. if data has been cropped manually).
        try:
            from_node_id = edge_attributes[i][from_id_col]

            # Check if the data for node exists
            node_dict[from_node_id]

        except KeyError:
            continue
        except Exception as e:
            raise e

        try:
            to_node_id = edge_attributes[i][to_id_col]

            # Check if the data for node exists
            node_dict[to_node_id]
        except KeyError:
            continue
        except Exception as e:
            raise e

        # Create edges
        # ------------
        edge_list.append([from_node_id, to_node_id, 0, edge_attributes[i]])

    del node_dict

    # Create directed graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_attributes)
    graph.add_edges_from(edge_list)
    graph.graph["crs"] = crs
    graph.graph["name"] = "Made with Pyrosm library."

    return graph

cpdef _create_pdgraph(nodes,
                      edges,
                      from_id_col,
                      to_id_col,
                      weight_cols):
    """
    Creates a Pandana Network from directed edges and nodes.
    NOTE: Assumes that the input edges GeoDataFrame is directed.
    """
    if not HAS_PANDANA:
        raise ImportError("'pandana' needs to be installed "
                          "in order to export the network for it.")
    from pandana import Network
    return Network(node_x=nodes["x"],
                   node_y=nodes["y"],
                   edge_from=edges[from_id_col],
                   edge_to=edges[to_id_col],
                   edge_weights=edges[weight_cols],
                   twoway=False)

cpdef generate_directed_edges(edges,
                              direction,
                              from_id_col,
                              to_id_col,
                              force_bidirectional):
    """
    Generates directed set of edges from network 
    following rules specified in 'direction' column.
    
    If 'force_bidirectional=True' travel to both direction is allowed for all edges.
    """
    cdef int i, n_edges = len(edges)

    # Convert edges to dict
    edge_attributes = edges.to_dict(orient="records")

    # Generate directed set of edges
    for i in range(0, n_edges):

        # Get nodeids for the edge
        # ------------------------
        from_node_id = edge_attributes[i][from_id_col]
        to_node_id = edge_attributes[i][to_id_col]

        # Create edges
        # ------------

        # Oneway streets
        if edge_attributes[i][direction] in oneway_values and not force_bidirectional:
            # When travelling is allowed only against digitization direction
            # flip the order of link nodes
            if edge_attributes[i][direction] in ['-1', 'T']:
                edge_attributes[i]["u"] = to_node_id
                edge_attributes[i]["v"] = from_node_id

            # Do nothing if travel is allowed along the digitization direction
            else:
                continue

        # Roundabouts are oneways
        elif 'junction' in edge_attributes[i].keys() \
                and edge_attributes[i]['junction'] == 'roundabout' \
                and not force_bidirectional:
            # Do nothing
            continue

        else:

            # If road is bi-directional add the "against" direction
            # -----------------------------------------------------
            # Take a deepcopy with dict-comprehension,
            # so that from_node_id and to_node_id attribute info stays correct
            # in the dictionary. This is needed only with bi-directional road segments.
            e_attributes = {k: v for k, v in edge_attributes[i].items()}

            # Append the opposite direction link nodes and attributes
            e_attributes["u"] = to_node_id
            e_attributes["v"] = from_node_id
            edge_attributes.append(e_attributes)

    return gpd.GeoDataFrame(edge_attributes, geometry="geometry", crs=edges.crs)
