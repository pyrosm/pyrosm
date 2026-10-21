"""Topological graph simplification for pyrosm's directed graph-export edges.

Removes interstitial nodes (degree-2 nodes that only carry geometry) and collapses
each chain of them into a single edge that keeps the full original geometry, matching
the semantics of ``osmnx.simplification.simplify_graph``. Operates on the *directed*
representation pyrosm builds in ``generate_directed_edges`` (two reciprocal rows per
two-way street), before the data is handed to an exporter.

The module is graph-library free: it uses only numpy/pandas/shapely plus one Cython
kernel (``pyrosm._simplify_walk``) for the inherently sequential chain walk. A pure
Python reference walk (``_reference_walk``) is kept as a correctness oracle and as a
fallback when the compiled kernel is unavailable.

The endpoint-detection and chain-collapsing rules follow OSMnx
(``osmnx.simplification.simplify_graph``) and the topological simplification method
described in:

    Boeing, G. (2025). Topological Graph Simplification Solutions to the Street
    Intersection Miscount Problem. Transactions in GIS, 29: e70037.
    https://doi.org/10.1111/tgis.70037
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rapidjson import loads

from pyrosm.distance import segment_lengths
from pyrosm.frames import network_node_frame
from pyrosm.geometry import segment_lines

# The edge attributes a simplified edge has one value of by default: its speed limit and
# its road class (a node where they change stays).
EDGE_ATTRS_DIFFER = ("maxspeed", "highway")

try:
    from pyrosm._simplify_walk import walk_chains as _cython_walk_chains
except Exception:  # pragma: no cover - exercised only before the extension is built
    _cython_walk_chains = None


def _index_dtype(n):
    """The integer type of indices below ``n``: int32 when it holds them, else int64."""
    return np.int32 if n <= np.iinfo(np.int32).max else np.int64


def _factorize_nodes(src, dst):
    """Map the (sparse, large) OSM node ids of the directed edges' sources ``src`` and
    targets ``dst`` onto contiguous indices, int32 when they fit (see
    :func:`_index_dtype`).

    Returns (u, v, uniques) where u/v are one index per directed edge and
    ``uniques[i]`` is the original node id for factor ``i``.
    """
    m = len(src)
    both = pd.concat([pd.Series(src), pd.Series(dst)], ignore_index=True)
    codes, uniques = pd.factorize(both, sort=False)
    index = _index_dtype(2 * m)
    u = np.ascontiguousarray(codes[:m], dtype=index)
    v = np.ascontiguousarray(codes[m:], dtype=index)
    return u, v, np.asarray(uniques)


def _distinct_neighbour_count(u, v, n_nodes):
    """Number of distinct undirected neighbours per node (vectorized): each pair of
    neighbouring nodes is one int64 key, counted once for each of its nodes."""
    low = np.minimum(u, v).astype(np.int64)
    high = np.maximum(u, v).astype(np.int64)
    pairs = pd.unique(low * n_nodes + high)
    low, high = np.divmod(pairs, n_nodes)
    counts = np.bincount(low, minlength=n_nodes)
    counts += np.bincount(high[high != low], minlength=n_nodes)
    return counts


def detect_endpoints(u, v, n_nodes, *, edge_attr_values=None, node_include_mask=None):
    """Boolean array marking which factor-indexed nodes are endpoints (kept).

    Mirrors OSMnx ``_is_endpoint`` rules 1-5:
      1. self-loop, 2. dead-end (no in- or no out-edges),
      3. not a clean pass-through (distinct neighbours == 2 and total degree in {2,4}),
      4. ``node_include_mask`` (node carries a relaxation attribute),
      5. ``edge_attr_values`` (incident edges disagree on a relaxation column).
    """
    out_deg = np.bincount(u, minlength=n_nodes)
    in_deg = np.bincount(v, minlength=n_nodes)

    self_loop = np.zeros(n_nodes, dtype=bool)
    self_loop[u[u == v]] = True

    distinct_nbr = _distinct_neighbour_count(u, v, n_nodes)
    total_deg = in_deg + out_deg
    is_pass_through = (distinct_nbr == 2) & ((total_deg == 2) | (total_deg == 4))
    dead_end = (in_deg == 0) | (out_deg == 0)

    endpoint = self_loop | dead_end | ~is_pass_through

    # Rule 5: a node is an endpoint if its incident (in+out) edges disagree on any
    # named column, i.e. their codes for it (-1 when missing) have a different lowest and
    # highest value. ``edge_attr_values`` is a list of per-row value arrays or Series.
    if edge_attr_values:
        node_of = np.concatenate([u, v])
        for values in edge_attr_values:
            codes = pd.factorize(values, sort=False)[0]
            codes = np.concatenate([codes, codes])
            low = np.full(n_nodes, np.iinfo(np.int64).max)
            high = np.full(n_nodes, np.iinfo(np.int64).min)
            np.minimum.at(low, node_of, codes)
            np.maximum.at(high, node_of, codes)
            endpoint |= high > low

    # Rule 4: a node is an endpoint if it carries any named node attribute.
    if node_include_mask is not None:
        endpoint |= node_include_mask

    return endpoint


def _build_csr(u, v, n_nodes):
    """Directed CSR adjacency keyed by source node.

    Returns (indptr, indices, edge_id, src) where for node ``a`` its out-edges occupy
    CSR positions ``[indptr[a]:indptr[a+1]]``; ``indices[p]`` is the target node and
    ``edge_id[p]`` is the original directed-row index of that out-edge. ``src[p]`` is
    the source node of position ``p`` (for ring handling).
    """
    order = np.argsort(u, kind="stable")
    indices = np.ascontiguousarray(v[order])
    edge_id = order.astype(u.dtype)
    counts = np.bincount(u, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=u.dtype)
    np.cumsum(counts, out=indptr[1:])
    src = np.repeat(np.arange(n_nodes, dtype=u.dtype), counts)
    return indptr, indices, edge_id, src


def _reference_walk(indptr, indices, edge_id, is_endpoint, src, remove_rings):
    """Pure-Python chain walk (correctness oracle / fallback for the Cython kernel).

    Returns (chain_edge_ids, chain_ptr): chain ``k`` consists of the original
    directed-row indices ``chain_edge_ids[chain_ptr[k]:chain_ptr[k+1]]``, in walk
    order. Each directed row is emitted in at most one chain: where several rows run
    between the same two consecutive nodes of a chain that starts at an endpoint, the
    chain keeps the first and the others are dropped, as in OSMnx.
    """
    m = indices.shape[0]
    n_nodes = indptr.shape[0] - 1
    visited = np.zeros(m, dtype=bool)
    chain_edge_ids = []
    chain_ptr = [0]

    def _drop_parallels(a, b):
        for q in range(indptr[a], indptr[a + 1]):
            if indices[q] == b:
                visited[q] = True

    def _walk_from(start_pos, start_node, drop_duplicates):
        visited[start_pos] = True
        chain = [int(edge_id[start_pos])]
        prev = start_node
        cur = int(indices[start_pos])
        if drop_duplicates and not is_endpoint[cur]:
            _drop_parallels(start_node, cur)
        while not is_endpoint[cur]:
            nxt = -1
            for q in range(indptr[cur], indptr[cur + 1]):
                if not visited[q] and indices[q] != prev:
                    nxt = q
                    break
            if nxt == -1:
                break  # OSM digitization quirk / one-way dead structure
            visited[nxt] = True
            chain.append(int(edge_id[nxt]))
            if drop_duplicates:
                _drop_parallels(cur, int(indices[nxt]))
            prev = cur
            cur = int(indices[nxt])
        chain_edge_ids.extend(chain)
        chain_ptr.append(len(chain_edge_ids))

    # Chains that start at an endpoint (interstitial chains + endpoint->endpoint edges).
    # A walk halts at endpoints, so an endpoint's out-edge is never consumed mid-chain;
    # it is visited already only when it duplicates the first step of an earlier chain.
    for e in range(n_nodes):
        if not is_endpoint[e]:
            continue
        for p in range(indptr[e], indptr[e + 1]):
            if not visited[p]:
                _walk_from(p, e, True)

    # Remaining unvisited edges belong to endpoint-free rings.
    if not remove_rings:
        for p in range(m):
            if not visited[p]:
                _walk_from(p, int(src[p]), False)

    return (
        np.asarray(chain_edge_ids, dtype=np.int64),
        np.asarray(chain_ptr, dtype=np.int64),
    )


def _differs_from_next(edges, col, rows, pairs):
    """Per edge row of ``rows`` but the last, where ``pairs`` marks it, whether the next one
    has another value of ``col``: a different value, or a value where this one has none or
    none where it has one. ``edges`` is an edge reader of :func:`_simplify_rows`."""
    try:
        codes = edges.codes(col)[rows]
    except TypeError:  # unhashable values, such as lists
        missing = edges.missing(col)[rows]
        differ = pairs & (missing[1:] != missing[:-1])
        both = np.flatnonzero(pairs & ~missing[1:] & ~missing[:-1])
        left = edges.take(col, rows[both]).array
        right = edges.take(col, rows[both + 1]).array
        differ[both] = np.asarray(left != right, dtype=bool)
        return differ
    return pairs & (codes[1:] != codes[:-1])


def _value_objects(series):
    """The values of ``series`` as ``series.tolist()`` gives them, in an object array that
    holds equal strings as one object."""
    if isinstance(series.dtype, pd.StringDtype):
        codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=False)
        objects = np.empty(len(uniques), dtype=object)
        objects[:] = uniques.tolist()
        return objects[codes]
    return series.to_numpy(dtype=object, copy=True)


def _geometry_coords(geoms):
    """All coordinates of ``geoms``, and where each geometry's run of them starts:
    ``(coords, offsets)``, geometry ``i`` holding ``coords[offsets[i]:offsets[i + 1]]``.
    """
    coords = shapely.get_coordinates(geoms)
    offsets = np.zeros(len(geoms) + 1, dtype=np.int64)
    np.cumsum(shapely.get_num_coordinates(geoms), out=offsets[1:])
    return coords, offsets


def _compute_geom_reversed(coords, offsets, rows, u, v, node_x, node_y):
    """Per directed edge, True when the geometry of its row (``rows``, into the geometries
    of ``coords``/``offsets``) runs labelled-v -> labelled-u.

    pyrosm's reciprocal rows reuse the forward geometry (verified), so a row's
    coordinates do not necessarily run from labelled ``u`` to labelled ``v``. Decide
    by comparing the geometry's endpoints to the u/v node coordinates.
    """
    # first and last coordinate of each directed edge's geometry
    first = coords[offsets[:-1][rows]]
    last = coords[offsets[1:][rows] - 1]
    # Squared endpoint mismatch for the two orientations, then pick the smaller.
    # Using both endpoints (not just the first) keeps the decision robust even if a
    # geometry endpoint does not exactly equal its node coordinate, where EPSG:4326
    # anisotropy could otherwise bias a single-endpoint distance comparison.
    forward = (first[:, 0] - node_x[u]) ** 2 + (first[:, 1] - node_y[u]) ** 2
    forward += (last[:, 0] - node_x[v]) ** 2 + (last[:, 1] - node_y[v]) ** 2
    reverse = (first[:, 0] - node_x[v]) ** 2 + (first[:, 1] - node_y[v]) ** 2
    reverse += (last[:, 0] - node_x[u]) ** 2 + (last[:, 1] - node_y[u]) ** 2
    return reverse < forward


def _stitch_geometries(coords, offsets, rows, geom_reversed, chain_edge_ids, chain_ptr):
    """Build one merged LineString per chain, in walk order, dropping shared vertices.

    Each consumed directed edge takes the coordinates of its row (``rows``, into the
    geometries of ``coords``/``offsets``); a segment whose stored geometry runs against the
    walk is reversed.
    """
    s = len(chain_edge_ids)
    if s == 0:
        return shapely.linestrings(np.empty((0, 2)), indices=np.empty(0, np.int64))

    # Per consumed segment (in walk order): its coord run start, length, and direction.
    index = _index_dtype(len(coords))
    segs = rows[chain_edge_ids]
    lens = np.diff(offsets)[segs].astype(index)
    starts = offsets[segs].astype(index)
    del segs
    rev = geom_reversed[chain_edge_ids]

    # Every segment but the first in its chain drops one vertex (shared with the prev).
    skip = np.ones(s, dtype=index)
    skip[chain_ptr[:-1]] = 0
    out_len = lens - skip

    out_off = np.zeros(s + 1, dtype=np.int64)
    np.cumsum(out_len, out=out_off[1:])
    total = int(out_off[-1])

    # Map each output coordinate back to a coordinate: a segment's run is read from its
    # first output vertex on, forwards, or backwards where the segment is reversed.
    seg_of_pos = np.repeat(np.arange(s, dtype=_index_dtype(s)), out_len)
    first = np.where(rev, starts + lens - 1 - skip, starts + skip)
    step = np.where(rev, np.int8(-1), np.int8(1))
    del lens, starts, rev, skip, out_len
    src = np.arange(total, dtype=np.int64)
    src -= out_off[seg_of_pos]
    src *= step[seg_of_pos]
    src += first[seg_of_pos]
    del out_off, first, step
    points = coords[src]
    del src
    n_chains = len(chain_ptr) - 1
    seg_chain = np.repeat(np.arange(n_chains), np.diff(chain_ptr))
    indices = seg_chain[seg_of_pos]
    del seg_chain, seg_of_pos
    return shapely.linestrings(points, indices=indices)


def simplify_graph(
    nodes,
    directed_edges,
    *,
    from_id_col="u",
    to_id_col="v",
    node_id_col="id",
    length_cols=("length",),
    edge_attrs_differ=EDGE_ATTRS_DIFFER,
    node_attrs_include=None,
    remove_rings=True,
    track_merged=False,
    edge_attr_aggs=None,
):
    """Topologically simplify pyrosm's *directed* graph-export edges.

    Removes interstitial nodes (degree-2 nodes that only carry geometry) and collapses
    each chain of them into a single edge that keeps the full original geometry, matching
    the semantics of ``osmnx.simplification.simplify_graph``. Operates on the *directed*
    representation pyrosm builds in ``get_directed_edges`` (two reciprocal rows per two-way
    street), before the data is handed to an exporter.

    A chain is not merged across a change of ``maxspeed`` or ``highway`` (the default
    ``edge_attrs_differ``; a missing value next to a set one counts as a change), so each
    simplified edge has one speed limit and one road class; ``edge_attrs_differ=None``
    merges across them as OSMnx does by default.

    A node with a value for one of ``node_attrs_include``, a node column or a tag in the
    nodes' ``tags``, stays an end of its edges (a warning names a name no node has).

    A simplified edge sums the ``length_cols`` of its segments. Other columns keep one
    value when it is the same along the chain, else become a list of the segments' values;
    ``edge_attr_aggs`` aggregates a column instead, as in OSMnx: a dict of column ->
    function, called with the list of the segments' values (e.g. ``{"travel_time_s": sum,
    "maxspeed_kmh": max}``). By default (``None``) ``travel_time`` is summed, as OSMnx
    does.

    Where two or more edges run between the same two consecutive nodes of a chain
    (typically duplicate OSM ways drawn over one street), only the first is kept, as in
    OSMnx. Parallel edges between two endpoints stay separate.

    The implementation is graph-library free: it uses only numpy/pandas/shapely plus one
    Cython kernel (``pyrosm._simplify_walk``) for the inherently sequential chain walk. A
    pure Python reference walk (``_reference_walk``) is kept as a correctness oracle and as
    a fallback when the compiled kernel is unavailable.

    The endpoint-detection and chain-collapsing rules follow OSMnx
    (``osmnx.simplification.simplify_graph``) and the topological simplification method
    described in:

        Boeing, G. (2025). Topological Graph Simplification Solutions to the Street
        Intersection Miscount Problem. Transactions in GIS, 29: e70037.
        https://doi.org/10.1111/tgis.70037

    ``directed_edges`` must be the directed representation (two reciprocal rows per
    two-way street), e.g. the output of ``pyrosm.graphs.get_directed_edges`` -- NOT the
    single-row ``OSM.get_network(nodes=True)`` edges. Returns
    ``(simplified_nodes, simplified_edges)`` in the same schema the exporters consume.

    Examples
    --------
    Most users simplify as part of graph export (``OSM.to_graph(..., simplify=True)``)::

        from pyrosm import OSM, get_data

        osm = OSM(get_data("helsinki_pbf"))
        nodes, edges = osm.get_network(network_type="walking", nodes=True)
        graph = osm.to_graph(nodes, edges, graph_type="networkx", simplify=True)

    To run it directly on the directed edges and get GeoDataFrames back::

        from pyrosm.graphs import get_directed_edges
        from pyrosm.graph_simplify import simplify_graph

        directed_nodes, directed_edges = get_directed_edges(
            nodes, edges, network_type="walking"
        )
        simple_nodes, simple_edges = simplify_graph(directed_nodes, directed_edges)
    """
    m = len(directed_edges)
    return _simplify_rows(
        _TableNodes(nodes, node_id_col),
        _TableEdges(directed_edges),
        np.arange(m),
        np.zeros(m, dtype=bool),
        from_id_col=from_id_col,
        to_id_col=to_id_col,
        length_cols=length_cols,
        edge_attrs_differ=edge_attrs_differ,
        node_attrs_include=node_attrs_include,
        remove_rings=remove_rings,
        track_merged=track_merged,
        edge_attr_aggs=edge_attr_aggs,
    )


# Numeric columns of pyrosm's network readers whose values are expected to become lists
# along a merged edge, as in OSMnx: the OSM ids (``osmid`` in OSMnx-compatible edges) and
# the metadata.
_READER_NUMERIC_COLUMNS = ("id", "osmid", "timestamp", "version", "changeset", "uid")


def _is_numeric_column(edges, col):
    """Whether the column ``col`` of ``edges`` holds numbers (not booleans), by its dtype."""
    dtype = edges.take(col, np.arange(0)).dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
        dtype
    )


def _warn_listed_numbers(listed, n_chains):
    """Warn that the numeric columns of ``listed`` (``(column, number of edges)`` pairs)
    became lists along merged edges, and how to aggregate them instead."""
    from pyrosm.utils import _stacklevel_outside_pyrosm

    where = ", ".join("%r in %d of %d edges" % (c, n, n_chains) for c, n in listed)
    example = "{%s}" % ", ".join("%r: sum" % c for c, _ in listed)
    warnings.warn(
        "Simplifying the graph made lists of the numeric values that differ along a "
        "merged edge: %s. To aggregate them, give edge_attr_aggs=%s (or another "
        "function; in to_graph through simplify_kwargs), or do not simplify, to keep "
        "every segment." % (where, example),
        UserWarning,
        stacklevel=_stacklevel_outside_pyrosm(),
    )


def _edge_aggregations(edge_attr_aggs, length_cols, structural):
    """The aggregation function of each column ``edge_attr_aggs`` names (a dict of column
    -> function, as in OSMnx), other than the ``length_cols``, which are always summed;
    ``None`` sums ``travel_time``, as OSMnx does. Raises ``ValueError`` for anything but
    such a dict, for one of the ``structural`` columns (the endpoints, the geometry and
    ``merged_edges``, which the simplification sets), or for a function other than ``sum`` on one of the
    ``length_cols``."""
    if edge_attr_aggs is None:
        edge_attr_aggs = {"travel_time": sum}
    if not isinstance(edge_attr_aggs, dict) or not all(
        callable(func) for func in edge_attr_aggs.values()
    ):
        raise ValueError(
            "'edge_attr_aggs' should be a dict of column -> aggregation function, "
            f"e.g. {{'travel_time': sum}}. Got {edge_attr_aggs!r}."
        )
    fixed = [c for c in structural if c in edge_attr_aggs]
    if fixed:
        raise ValueError(
            f"{fixed} are set by the simplification, so 'edge_attr_aggs' cannot "
            "aggregate them."
        )
    clash = [
        c for c in length_cols if c in edge_attr_aggs and edge_attr_aggs[c] is not sum
    ]
    if clash:
        raise ValueError(
            f"{clash} are always summed along a merged edge ('length_cols'), so "
            "'edge_attr_aggs' can only give them sum."
        )
    return {c: func for c, func in edge_attr_aggs.items() if c not in length_cols}


class _TableEdges:
    """The edges of a table (a GeoDataFrame), read by row position as
    :func:`_simplify_rows` reads them."""

    def __init__(self, frame):
        self.frame = frame
        self.columns = list(frame.columns)
        self.geometry_name = frame.geometry.name
        self.crs = frame.crs

    def values(self, col):
        return self.frame[col].to_numpy()

    def codes(self, col):
        return pd.factorize(self.frame[col], sort=False)[0]

    def floats(self, col):
        return self.frame[col].to_numpy(dtype="float64")

    def missing(self, col):
        return self.frame[col].isna().to_numpy()

    def take(self, col, positions):
        return self.frame[col].take(positions)

    def coords(self):
        return _geometry_coords(self.frame.geometry.values)

    def rows(self, positions):
        return self.frame.iloc[positions].reset_index(drop=True)


def _present_in(frame, col):
    """Whether each node of ``frame`` has a value for ``col``: in the column ``col`` or as
    a tag in the nodes' ``tags`` (dicts or JSON text); None when the frame has neither.
    """
    if col not in frame.columns and "tags" not in frame.columns:
        return None
    present = np.zeros(len(frame), dtype=bool)
    if col in frame.columns:
        present |= frame[col].notna().to_numpy()
    if "tags" in frame.columns:
        for i, tags in enumerate(frame["tags"].tolist()):
            if present[i]:
                continue
            if isinstance(tags, str):
                tags = loads(tags) if tags else None
            present[i] = isinstance(tags, dict) and tags.get(col) is not None
    return present


class _TableNodes:
    """The nodes of a table (a GeoDataFrame) keyed by ``node_id_col``, as
    :func:`_simplify_rows` reads them."""

    def __init__(self, frame, node_id_col):
        self.frame = frame
        self.node_id_col = node_id_col
        self.columns = list(frame.columns)

    def ids(self):
        return self.frame[self.node_id_col].to_numpy()

    def coordinates(self):
        geom = self.frame.geometry.values
        return shapely.get_x(geom), shapely.get_y(geom)

    def present(self, col):
        """Whether each node has a value for ``col``, as a column or a tag (see
        :func:`_present_in`)."""
        return _present_in(self.frame, col)

    def rows(self, ids):
        """The nodes whose id is one of ``ids``, in table order."""
        keep = self.frame[self.node_id_col].isin(ids)
        return self.frame[keep].reset_index(drop=True)

    def empty(self):
        return self.frame.iloc[:0].copy()

    def table(self):
        return self.frame


class _WayEdges:
    """The segments of the ways of the table ``ways`` (one row per way) as edges, read by
    segment position as :func:`_simplify_rows` reads them: segment ``i`` is of way row
    ``way_of[i]`` and runs from node ``u[i]`` at the lon/lat ``start[i]`` to node ``v[i]``
    at ``end[i]``. The edges have the columns of the ways, then ``geometry``, ``u``, ``v``
    and ``length`` in metres, as the per-segment table of ``get_network(nodes=True)``.
    """

    geometry_name = "geometry"
    crs = "epsg:4326"

    def __init__(self, ways, way_of, u, v, start, end):
        self.ways = ways
        self.way_of = way_of
        self.start = start
        self.end = end
        self._segment_columns = {
            "u": pd.Series(u),
            "v": pd.Series(v),
            "length": pd.Series(segment_lengths(start, end)),
        }
        self.columns = list(
            dict.fromkeys([*ways.columns, self.geometry_name, *self._segment_columns])
        )

    def add_empty_column(self, col):
        """Add ``col``, with no values, as the last column."""
        self.ways[col] = None
        self.columns.append(col)

    def release(self):
        """Drop the ways and segments; the edges cannot be read after this."""
        self.ways = self.way_of = self.start = self.end = self._segment_columns = None

    def _segment_column(self, col):
        """The values of ``col`` by segment, or ``None`` for a column of the ways."""
        if col == self.geometry_name:
            return gpd.GeoSeries(segment_lines(self.start, self.end), crs=self.crs)
        return self._segment_columns.get(col)

    def _segment_values(self, col, read):
        """``read`` of the values of ``col``, one per segment."""
        values = self._segment_column(col)
        if values is not None:
            return read(values)
        return read(self.ways[col])[self.way_of]

    def values(self, col):
        return self._segment_values(col, lambda series: series.to_numpy())

    def codes(self, col):
        return self._segment_values(
            col, lambda series: pd.factorize(series, sort=False)[0]
        )

    def floats(self, col):
        return self._segment_values(
            col, lambda series: series.to_numpy(dtype="float64")
        )

    def missing(self, col):
        return self._segment_values(col, lambda series: series.isna().to_numpy())

    def take(self, col, positions):
        values = self._segment_column(col)
        if values is not None:
            return values.take(positions)
        return self.ways[col].take(self.way_of[positions])

    def coords(self):
        coords = np.empty((2 * len(self.way_of), 2))
        coords[0::2] = self.start
        coords[1::2] = self.end
        return coords, np.arange(0, len(coords) + 1, 2)

    def rows(self, positions):
        columns = {}
        for col in self.columns:
            if col == self.geometry_name:
                columns[col] = segment_lines(self.start[positions], self.end[positions])
            else:
                columns[col] = self.take(col, positions).reset_index(drop=True)
        return gpd.GeoDataFrame(columns, geometry=self.geometry_name, crs=self.crs)


class _StoreNodes:
    """The nodes ``ids`` at ``rows`` of the coordinate store ``store`` (a
    :class:`~pyrosm.node_lookup.NodeLocations` of node records), read as
    :func:`_simplify_rows` reads them; their table is the node table of
    ``get_network(nodes=True)``."""

    def __init__(self, store, ids, rows):
        self.store = store
        self.node_ids = ids
        self.node_rows = rows
        self.columns = [*store.columns_at(rows[:0]), "id", "geometry"]

    def ids(self):
        return self.node_ids

    def coordinates(self):
        _, lon, lat = self.store.gather(self.node_ids)
        return lon, lat

    def present(self, col):
        """Whether each node has a value for ``col``, as a column or a tag (see
        :func:`_present_in`)."""
        return _present_in(self.table(), col)

    def rows(self, ids):
        """The nodes whose id is one of ``ids``, in table order."""
        keep = np.isin(self.node_ids, ids)
        return network_node_frame(self.store, self.node_rows[keep], self.node_ids[keep])

    def empty(self):
        return self.table().iloc[:0].copy()

    def table(self):
        return network_node_frame(self.store, self.node_rows, self.node_ids)

    def release(self):
        """Drop the store; the nodes cannot be read after this."""
        self.store = self.node_ids = self.node_rows = None


def _chain_flags(flags, chain_edge_ids, chain_ptr, seg_chain):
    """The boolean ``flags`` of the directed edges, one value per simplified edge (the
    chains of ``chain_edge_ids``/``chain_ptr``, ``seg_chain`` the chain of each consumed
    edge): the value of its edges where they agree, else the sorted list of their
    values, as OSMnx gives a simplified edge."""
    per_edge = np.asarray(flags, dtype=bool)[chain_edge_ids]
    values = per_edge[chain_ptr[:-1]].tolist()
    inner = seg_chain[1:] == seg_chain[:-1]
    mixed = np.unique(seg_chain[1:][inner & (per_edge[1:] != per_edge[:-1])])
    for k in mixed.tolist():
        values[k] = sorted(set(per_edge[chain_ptr[k] : chain_ptr[k + 1]].tolist()))
    return values


def _simplify_rows(
    nodes,
    edges,
    rows,
    swapped,
    *,
    from_id_col="u",
    to_id_col="v",
    length_cols=("length",),
    edge_attrs_differ=EDGE_ATTRS_DIFFER,
    node_attrs_include=None,
    remove_rings=True,
    track_merged=False,
    edge_attr_aggs=None,
    directions=None,
):
    """:func:`simplify_graph` of the directed edges ``rows``/``swapped`` of the edges that
    ``edges`` reads (a :class:`_TableEdges` or :class:`_WayEdges`), with the nodes that
    ``nodes`` reads (a :class:`_TableNodes` or :class:`_StoreNodes`): directed edge ``i``
    is edge row ``rows[i]``, travelled from its
    ``to_id_col`` node to its ``from_id_col`` node where ``swapped[i]``, as
    :func:`pyrosm.graph_export.directed_rows` gives them. Every column is read through the
    row numbers, so the directed edges are never copied into a table; only the simplified
    edges are. ``directions`` maps a column name to a boolean of each directed edge,
    which each simplified edge gets by :func:`_chain_flags`."""
    structural = (from_id_col, to_id_col, edges.geometry_name)
    if track_merged:
        structural += ("merged_edges",)
    aggregations = _edge_aggregations(edge_attr_aggs, tuple(length_cols), structural)
    m = len(rows)
    if m == 0:
        return nodes.table(), edges.rows(rows)

    from_ids = edges.values(from_id_col)
    to_ids = edges.values(to_id_col)
    u, v, uniques = _factorize_nodes(
        np.where(swapped, to_ids[rows], from_ids[rows]),
        np.where(swapped, from_ids[rows], to_ids[rows]),
    )
    n_nodes = len(uniques)

    # node id -> factor index, and node coordinates indexed by factor
    nid = nodes.ids()
    nid_to_factor = pd.Series(np.arange(n_nodes), index=uniques)
    x, y = nodes.coordinates()
    node_x = np.full(n_nodes, np.nan)
    node_y = np.full(n_nodes, np.nan)
    present = nid_to_factor.reindex(nid).to_numpy()
    keep = ~np.isnan(present)
    fpos = present[keep].astype(np.int64)
    node_x[fpos] = x[keep]
    node_y[fpos] = y[keep]

    edge_attr_values = None
    if edge_attrs_differ is not None and len(edge_attrs_differ):
        edge_attr_values = [
            edges.codes(c)[rows] for c in edge_attrs_differ if c in edges.columns
        ]
    node_include_mask = None
    if node_attrs_include is not None and len(node_attrs_include):
        nm = np.zeros(n_nodes, dtype=bool)
        unknown = []
        for c in node_attrs_include:
            has = nodes.present(c)
            if has is None or not has.any():
                unknown.append(c)
                continue
            f = nid_to_factor.reindex(nid[has]).to_numpy()
            f = f[~np.isnan(f)].astype(np.int64)
            nm[f] = True
        if unknown:
            from pyrosm.utils import _stacklevel_outside_pyrosm

            warnings.warn(
                "node_attrs_include: no node has a column or a tag named %s, which "
                "therefore keeps no node." % ", ".join(repr(c) for c in unknown),
                UserWarning,
                stacklevel=_stacklevel_outside_pyrosm(),
            )
        node_include_mask = nm
    del nid, nid_to_factor, x, y, present, keep, fpos

    is_endpoint = detect_endpoints(
        u,
        v,
        n_nodes,
        edge_attr_values=edge_attr_values,
        node_include_mask=node_include_mask,
    )
    del edge_attr_values, node_include_mask

    indptr, indices, edge_id, src = _build_csr(u, v, n_nodes)

    if _cython_walk_chains is not None:
        index = np.intc if u.dtype == np.int32 else np.longlong
        chain_edge_ids, chain_ptr = _cython_walk_chains(
            indptr.astype(index, copy=False),
            indices.astype(index, copy=False),
            edge_id.astype(index, copy=False),
            np.ascontiguousarray(is_endpoint, dtype=np.uint8),
            src.astype(index, copy=False),
            remove_rings,
        )
    else:
        chain_edge_ids, chain_ptr = _reference_walk(
            indptr, indices, edge_id, is_endpoint, src, remove_rings
        )
    chain_edge_ids = np.asarray(chain_edge_ids, dtype=u.dtype)
    chain_ptr = np.asarray(chain_ptr, dtype=np.int64)
    del indptr, indices, edge_id, src

    n_chains = len(chain_ptr) - 1
    if n_chains == 0:
        # Nothing survived (e.g. a pure endpoint-free ring with remove_rings=True):
        # return empty node/edge frames preserving the input schema and CRS.
        return nodes.empty(), edges.rows(rows[:0])

    # First/last node (factor) of each chain -> original node id.
    first_pos = chain_ptr[:-1]
    last_pos = chain_ptr[1:] - 1
    new_u_factor = u[chain_edge_ids[first_pos]]
    new_v_factor = v[chain_edge_ids[last_pos]]

    geom_col = edges.geometry_name
    coords, offsets = edges.coords()
    geom_reversed = _compute_geom_reversed(coords, offsets, rows, u, v, node_x, node_y)
    del node_x, node_y
    new_geom = _stitch_geometries(
        coords, offsets, rows, geom_reversed, chain_edge_ids, chain_ptr
    )
    del coords, offsets, geom_reversed
    chain_id = np.repeat(np.arange(n_chains), np.diff(chain_ptr))
    # The row of ``edges`` of each consumed segment, aligned to chain_edge_ids.
    seg_rows = rows[chain_edge_ids]

    # Assemble the simplified edge frame: keep the first segment's row per chain, then
    # overwrite u/v/length and the merged geometry. Assign the geometry to the actual
    # active geometry column (not assumed to be "geometry") so exporters that read
    # edges[geom_col] see the collapsed chain, not the first original segment.
    out = edges.rows(seg_rows[first_pos])
    out[from_id_col] = uniques[new_u_factor]
    out[to_id_col] = uniques[new_v_factor]
    out[geom_col] = gpd.GeoSeries(new_geom, index=out.index, crs=edges.crs)
    out = out.set_geometry(geom_col)

    seg_chain = chain_id  # chain id per consumed segment, aligned to chain_edge_ids
    for col in length_cols:
        if col in edges.columns:
            vals = edges.floats(col)[seg_rows]
            out[col] = np.bincount(seg_chain, weights=vals, minlength=n_chains)
    # Aggregated columns: a float column summed by bincount, any other by calling the
    # function with each chain's list of values.
    for col, func in aggregations.items():
        if col not in edges.columns:
            continue
        values = np.asarray(edges.take(col, seg_rows))
        if func is sum and values.dtype.kind == "f":
            out[col] = np.bincount(seg_chain, weights=values, minlength=n_chains)
        else:
            per_chain = np.split(values, chain_ptr[1:-1])
            out[col] = pd.Series(
                [func(chunk.tolist()) for chunk in per_chain], index=out.index
            )

    # Other columns: scalar if uniform within a chain, else a list (object dtype).
    # A chain is "mixed" iff two adjacent consumed segments of it disagree on the column:
    # one has a value and the other none, or both have values that differ. Only the mixed
    # chains' values are read out for their lists.
    merge_cols = [
        c
        for c in edges.columns
        if c not in (from_id_col, to_id_col, geom_col)
        and c not in length_cols
        and c not in aggregations
    ]
    inner = seg_chain[1:] == seg_chain[:-1]
    listed = []
    for col in merge_cols:
        differ = _differs_from_next(edges, col, seg_rows, inner)
        if not differ.any():
            continue
        mixed_ids = np.unique(seg_chain[1:][differ])
        if col not in _READER_NUMERIC_COLUMNS and _is_numeric_column(edges, col):
            listed.append((col, len(mixed_ids)))
        starts = chain_ptr[mixed_ids]
        sizes = chain_ptr[mixed_ids + 1] - starts
        # The positions of the mixed chains' segments, chain after chain.
        positions = np.repeat(starts - np.cumsum(sizes) + sizes, sizes)
        positions += np.arange(positions.size)
        values = _value_objects(edges.take(col, seg_rows[positions]))
        colvals = _value_objects(out[col])
        at = 0
        for k, size in zip(mixed_ids.tolist(), sizes.tolist()):
            colvals[k] = values[at : at + size].tolist()
            at += size
        out[col] = pd.Series(colvals, index=out.index, dtype=object)
    if listed:
        _warn_listed_numbers(listed, n_chains)
    for name, flags in (directions or {}).items():
        out[name] = pd.Series(
            _chain_flags(flags, chain_edge_ids, chain_ptr, seg_chain),
            index=out.index,
            dtype=object,
        )

    if track_merged:
        merged_pairs = []
        for k in range(n_chains):
            seg = chain_edge_ids[chain_ptr[k] : chain_ptr[k + 1]]
            merged_pairs.append([(int(uniques[u[e]]), int(uniques[v[e]])) for e in seg])
        out["merged_edges"] = merged_pairs

    # Output nodes: every node that is an endpoint of a simplified edge. This is the
    # retained endpoint set plus any ring pseudo-endpoint emitted when
    # remove_rings=False (a pure ring's start node is not flagged in is_endpoint, so
    # keying off is_endpoint alone would drop a node the ring edge still references).
    kept_factors = np.unique(np.concatenate([new_u_factor, new_v_factor]))
    kept_ids = uniques[kept_factors]
    out_nodes = nodes.rows(kept_ids)

    return out_nodes, out
