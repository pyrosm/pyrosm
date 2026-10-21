from collections import defaultdict

import numpy as np


def _strongly_connected_components(list_of_nodes, node_successors):
    """
    Generate nodes in strongly connected components of graph.

    Source: https://networkx.org/documentation/stable/reference/algorithms/component.html

    Uses Tarjan's algorithm [1] with Nuutila's modifications [2].
    Nonrecursive version of algorithm.

    References
    ----------

    [1] Depth-first search and linear graph algorithms, R. Tarjan
        SIAM Journal of Computing 1(2):146-160, (1972).

    [2] On finding the strongly connected components in a directed graph.
        E. Nuutila and E. Soisalon-Soinen
        Information Processing Letters 49(1): 9-14, (1994).

    """
    preorder = {}
    lowlink = {}
    scc_found = {}
    scc_queue = []
    i = 0  # Preorder counter
    for source in list_of_nodes:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = 1
                v_nbrs = node_successors[v]
                for w in v_nbrs:
                    if w not in preorder:
                        queue.append(w)
                        done = 0
                        break
                if done == 1:
                    lowlink[v] = preorder[v]
                    for w in v_nbrs:
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc_found[v] = True
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc_found[k] = True
                            scc.add(k)
                        yield scc
                    else:
                        scc_queue.append(v)


def largest_component(node_ids, from_ids, to_ids):
    """The ids of the nodes of the largest strongly connected component of the nodes
    ``node_ids`` and the directed edges ``from_ids[i]`` -> ``to_ids[i]``, as an array.
    Edges with an end that is not among ``node_ids`` are left out.
    """
    node_ids = np.asarray(node_ids)
    from_ids, to_ids = np.asarray(from_ids), np.asarray(to_ids)
    known = np.isin(from_ids, node_ids) & np.isin(to_ids, node_ids)
    node_successors = defaultdict(list)
    for from_id, to_id in zip(from_ids[known].tolist(), to_ids[known].tolist()):
        node_successors[from_id].append(to_id)
    scc = max(
        _strongly_connected_components(node_ids.tolist(), node_successors), key=len
    )
    return np.fromiter(scc, dtype=node_ids.dtype, count=len(scc))


def get_connected_edges(nodes, edges, from_id_col="u", to_id_col="v", node_id_col="id"):
    """Filters the network data (directed) to include only connected edges and nodes."""
    scc = largest_component(
        nodes[node_id_col].to_numpy(), edges[from_id_col], edges[to_id_col]
    )
    # Filter nodes and edges accordingly
    n = nodes[nodes[node_id_col].isin(scc)]
    e = edges[(edges[from_id_col].isin(scc)) & (edges[to_id_col].isin(scc))]
    return n, e.reset_index(drop=True)
