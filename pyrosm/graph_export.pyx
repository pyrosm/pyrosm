from pyrosm.utils._compat import HAS_IGRAPH, HAS_NETWORKX, HAS_PANDANA
from pyrosm.config import Conf
import geopandas as gpd
import pandas as pd
import numpy as np

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

    crs = f"EPSG:{edges.crs.to_epsg(min_confidence=25)}"
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
                              direction_suffix,
                              from_id_col,
                              to_id_col,
                              force_bidirectional):
    """
    Generates directed set of edges from network 
    following rules specified in 'direction' column.
    
    If 'force_bidirectional=True' travel to both direction is allowed for all edges.
    """
    if force_bidirectional:
        # Flip from/to values
        edges_dir2 = edges.copy(deep=True).rename(
            columns={to_id_col: from_id_col,
                     from_id_col: to_id_col}
        )
        return pd.concat([edges, edges_dir2], ignore_index=True)

    # ========================================
    # Directed edges according 'oneway' rules
    # ========================================

    if "junction" in edges.columns:
        roundabouts = True
    else:
        roundabouts = False

    if direction_suffix:
        direction_suffix = edges[direction + ":" + direction_suffix]
    else:
        direction_suffix = pd.DataFrame()
    direction = edges[direction]

    oneway_mask = direction_suffix.isin(oneway_values)
    if not direction_suffix.empty:
        oneway_mask |= (direction.isin(oneway_values) & direction_suffix.isna())

    if roundabouts:
        # Edge is oneway if it is tagged as such OR if it tagged as roundabout
        oneway_mask |= (edges["junction"] == "roundabout")

    edge_cnt = len(edges)
    oneway_edges = edges.loc[oneway_mask].copy()
    twoway_edges = edges.loc[~oneway_mask].copy()
    twoway_edges_dir2 = twoway_edges.copy(deep=True).rename(columns={to_id_col: from_id_col, from_id_col: to_id_col})
    twoway_edges_dir2.index = np.arange(edge_cnt, edge_cnt + len(twoway_edges))

    # Select edges that are allowed only to opposite direction
    against_mask = direction.isin(["-1", "T"])
    against_edges = oneway_edges.loc[against_mask].copy()
    along_edges = oneway_edges.loc[~against_mask].copy()  # Nothing needs to be done for these

    # Flip the from/to ids for against edges
    against_edges = against_edges.rename(columns={from_id_col: to_id_col, to_id_col: from_id_col})

    # Stack everything (keep order)
    return pd.concat([along_edges, against_edges, twoway_edges, twoway_edges_dir2])
