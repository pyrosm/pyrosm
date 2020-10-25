from pyrosm.utils._compat import HAS_IGRAPH, HAS_NETWORKX
from pyrosm.config import Conf

# The values used to determine oneway road in OSM
oneway_values = Conf.oneway_values

cpdef _create_igraph(nodes,
                     edges,
                     direction,
                     from_id_col,
                     to_id_col,
                     force_bidirectional):
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

    edge_columns = edges.columns.to_list()
    n_edges = len(edges)
    n_nodes = len(nodes)
    n_cols = len(edge_columns)

    # Convert edges to dict
    edges = edges.to_dict(orient="list")

    # Add columns for sequential ids
    edges[from_id_int] = [None for x in range(n_edges)]
    edges[to_id_int] = [None for x in range(n_edges)]

    # Node-ids needs to be sequential for igraph
    nodes = nodes.reset_index(drop=True)
    nodes["node_id"] = nodes.index

    # Prepare dictionary for fast lookups
    node_dict = {k: v for k, v in zip(nodes["id"].to_list(), nodes["node_id"].to_list())}

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

        # Oneway streets
        if edges[direction][i] in oneway_values and not force_bidirectional:
            # When travelling is allowed only against digitization direction
            # flip the order of link nodes
            if edges[direction][i] in ['-1', 'T']:

                edge_list.append([to_seq_id, from_seq_id])
                edges[from_id_int][i] = to_seq_id
                edges[to_id_int][i] = from_seq_id

            # In other cases add edge along the digitization direction
            else:
                edge_list.append([from_seq_id, to_seq_id])
                edges[from_id_int][i] = from_seq_id
                edges[to_id_int][i] = to_seq_id

        # Roundabouts are oneways
        elif 'junction' in edge_columns \
                and edges['junction'][i] == 'roundabout' \
                and not force_bidirectional:

            edge_list.append([from_seq_id, to_seq_id])
            edges[from_id_int][i] = from_seq_id
            edges[to_id_int][i] = to_seq_id

        else:
            # If road is bi-directional add it in both ways
            # ---------------------------------------------
            # Along
            edge_list.append([from_seq_id, to_seq_id])
            edges[from_id_int][i] = from_seq_id
            edges[to_id_int][i] = to_seq_id

            # Against - Flip the link nodes
            edge_list.append([to_seq_id, from_seq_id])

            # Append the opposite direction link nodes and attributes
            edges[from_id_int].append(to_seq_id)
            edges[to_id_int].append(from_seq_id)
            edges[from_id_col].append(to_node_id)
            edges[to_id_col].append(from_node_id)

            for key in edge_columns:
                # Skip edge nodes which were added separately (in opposite direction)
                if key == from_id_col or key == to_id_col:
                    continue
                edges[key].append(edges[key][i])

    del node_dict

    # Create directed graph
    graph = igraph.Graph(n=n_nodes, directed=True, edges=edge_list,
                         vertex_attrs=node_attributes,
                         edge_attrs=edges)
    return graph

cpdef _create_nxgraph(nodes,
                      edges,
                      direction,
                      from_id_col,
                      to_id_col,
                      force_bidirectional):
    if not HAS_NETWORKX:
        raise ImportError("'networkx' needs to be installed "
                          "in order to export the network for networkx / osmnx.")
    import networkx as nx

    cdef long long i

    nodes = nodes.copy()
    edges = edges.copy()

    crs = edges.crs
    n_edges = len(edges)
    oneway_values = ['yes', 'true', '1', '-1', 'T', 'F']
    edge_list = []

    # Convert edges to dict
    edge_attributes = edges.to_dict(orient="index")

    # Prepare node dictionary for fast lookups
    node_dict = {k: None for k in nodes["osmid"].to_list()}

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

        e_attributes = edge_attributes[i]

        # Oneway streets
        if edge_attributes[i][direction] in oneway_values and not force_bidirectional:
            # When travelling is allowed only against digitization direction
            # flip the order of link nodes
            if edge_attributes[i][direction] in ['-1', 'T']:
                e_attributes["u"] = to_node_id
                e_attributes["v"] = from_node_id
                edge_list.append([to_node_id, from_node_id, 0, e_attributes])

            # In other cases add edge along the digitization direction
            else:
                edge_list.append([from_node_id, to_node_id, 0, e_attributes])

        # Roundabouts are oneways
        elif 'junction' in edge_attributes[i].keys() \
                and edge_attributes[i]['junction'] == 'roundabout' \
                and not force_bidirectional:

            edge_list.append([from_node_id, to_node_id, 0, e_attributes])

        else:

            # If road is bi-directional add it in both ways
            # ---------------------------------------------

            # Along
            edge_list.append([from_node_id, to_node_id, 0, e_attributes])

            # Append the opposite direction link nodes and attributes
            e_attributes["u"] = to_node_id
            e_attributes["v"] = from_node_id
            edge_list.append([to_node_id, from_node_id, 0, e_attributes])

    del node_dict

    # Create directed graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_attributes)
    graph.add_edges_from(edge_list)
    graph.graph["crs"] = crs
    graph.graph["name"] = "NetworkX graph"

    return graph
