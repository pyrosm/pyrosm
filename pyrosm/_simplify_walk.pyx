# cython: language_level=3
"""Cython chain-walk kernel for graph simplification (see pyrosm/graph_simplify.py).

Walks interstitial chains between endpoint nodes over a directed CSR adjacency,
emitting one chain per walk as a flat array of original directed-row ids plus a
per-chain offset array. Each directed row is consumed by exactly one chain. This is
the only non-vectorized stage; everything else is numpy/pandas/shapely.

Arrays are typed as ``long long`` (format 'q'); the Python caller casts to
``np.longlong`` so binding is portable (np.int64 is 'l' on LP64 platforms).
"""
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def walk_chains(long long[::1] indptr,
                long long[::1] indices,
                long long[::1] edge_id,
                unsigned char[::1] is_endpoint,
                long long[::1] src,
                bint remove_rings):
    cdef Py_ssize_t m = indices.shape[0]
    cdef Py_ssize_t n_nodes = indptr.shape[0] - 1
    cdef Py_ssize_t e, p, q, cur, prev, nxt, guard

    out_ids_arr = np.empty(m, dtype=np.longlong)
    out_ptr_arr = np.empty(m + 1, dtype=np.longlong)
    visited_arr = np.zeros(m, dtype=np.uint8)
    cdef long long[::1] out_ids = out_ids_arr
    cdef long long[::1] out_ptr = out_ptr_arr
    cdef unsigned char[::1] visited = visited_arr
    cdef Py_ssize_t n_out = 0
    cdef Py_ssize_t n_chain = 0

    out_ptr[0] = 0
    with nogil:
        # Chains that start at an endpoint: interstitial chains and, when the
        # endpoint's successor is itself an endpoint, length-1 pass-through edges.
        for e in range(n_nodes):
            if is_endpoint[e] == 0:
                continue
            for p in range(indptr[e], indptr[e + 1]):
                if visited[p]:
                    continue
                visited[p] = 1
                out_ids[n_out] = edge_id[p]
                n_out += 1
                prev = e
                cur = indices[p]
                guard = 0
                while is_endpoint[cur] == 0:
                    nxt = -1
                    for q in range(indptr[cur], indptr[cur + 1]):
                        if visited[q] == 0 and indices[q] != prev:
                            nxt = q
                            break
                    if nxt == -1:
                        break
                    visited[nxt] = 1
                    out_ids[n_out] = edge_id[nxt]
                    n_out += 1
                    prev = cur
                    cur = indices[nxt]
                    guard += 1
                    if guard > m:
                        break
                n_chain += 1
                out_ptr[n_chain] = n_out

        # Endpoint-free cycles: any still-unvisited edge starts one ring chain.
        if not remove_rings:
            for p in range(m):
                if visited[p]:
                    continue
                visited[p] = 1
                out_ids[n_out] = edge_id[p]
                n_out += 1
                prev = src[p]
                cur = indices[p]
                guard = 0
                while is_endpoint[cur] == 0:
                    nxt = -1
                    for q in range(indptr[cur], indptr[cur + 1]):
                        if visited[q] == 0 and indices[q] != prev:
                            nxt = q
                            break
                    if nxt == -1:
                        break
                    visited[nxt] = 1
                    out_ids[n_out] = edge_id[nxt]
                    n_out += 1
                    prev = cur
                    cur = indices[nxt]
                    guard += 1
                    if guard > m:
                        break
                n_chain += 1
                out_ptr[n_chain] = n_out

    return out_ids_arr[:n_out].copy(), out_ptr_arr[:n_chain + 1].copy()
