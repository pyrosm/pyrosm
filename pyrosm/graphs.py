from pyrosm.graph_export import (
    _create_igraph,
    _create_nxgraph,
    _create_pdgraph,
    _create_pandarm_graph,
    directed_rows,
    generate_directed_edges,
)
from pyrosm.graph_connectivity import get_connected_edges
from pyrosm.graph_simplify import (
    _TableEdges,
    _TableNodes,
    _WayEdges,
    _simplify_rows,
)
from pyrosm.utils import (
    set_frame_attr,
    validate_edge_gdf,
    validate_node_gdf,
    warn_future,
)
from pyrosm.utils.progress import Bar, reporting
from pyrosm.config import Conf
from pyrosm.config.osm_filters import _UNKNOWN_NETWORK_TYPE, normalize_network_type
from rapidjson import dumps
import contextlib
import pandas as pd
import warnings


def _step_names(simplify, retain_all, last=None):
    """The steps of making a graph's tables, and ``last`` when given."""
    names = ["directing the edges"]
    if simplify:
        names.append("simplifying")
    if not retain_all:
        names.append("keeping the largest connected part")
    if last is not None:
        names.append(last)
    return names


@contextlib.contextmanager
def _steps(desc, names):
    """A function ``step(name)`` that reports the start of each of the steps ``names`` of the
    work described ``desc``, as :func:`pyrosm.enable_logging` sets the progress: a bar of
    steps on stderr that names the current one, shown after 2 seconds, or calls of the
    function set as ``progress(done, total)`` with the steps done and the steps."""
    with reporting(
        None, desc, delay=2.0, leave=False, timed=True, unit="step"
    ) as report:

        def step(name):
            if report is None:
                return
            if isinstance(report, Bar):
                report.describe("%s: %s" % (desc, name))
            report(names.index(name), len(names))

        yield step
        if report is not None:
            report(len(names), len(names))


def record_network_type(result, network_type):
    """Record ``network_type`` (lower-cased, an OSMnx alias as its pyrosm name) in the
    ``attrs`` of the frames ``get_network`` returns (the edges, or the ``(nodes, edges)``
    pair), where graph export looks it up. Entries that are not frames (an empty read's
    ``None``) are left alone. Returns ``result``."""
    return set_frame_attr(result, "network_type", normalize_network_type(network_type))


def _summing_weights(simplify_kwargs, weight_cols):
    """``simplify_kwargs`` with ``weight_cols`` added to the columns a simplified edge sums
    (``length_cols``), so each merged edge carries the totals of its segments' weights.
    """
    kwargs = dict(simplify_kwargs or {})
    summed = list(kwargs.get("length_cols", ("length",)))
    kwargs["length_cols"] = tuple(dict.fromkeys(summed + list(weight_cols)))
    return kwargs


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
    direction, direction_suffix, both_ways = _direction_rules(
        nodes,
        edges,
        direction,
        from_id_col,
        to_id_col,
        node_id_col,
        force_bidirectional,
        network_type,
    )
    edges = edges.copy()
    nodes = nodes.copy()
    edges = generate_directed_edges(
        edges,
        direction,
        direction_suffix,
        from_id_col,
        to_id_col,
        force_bidirectional=both_ways,
    )
    return nodes, edges


def _direction_rules(
    nodes,
    edges,
    direction,
    from_id_col,
    to_id_col,
    node_id_col,
    force_bidirectional,
    network_type,
):
    """Validate the nodes and edges of :func:`get_directed_edges` and return how its edges
    are directed: ``(direction, direction_suffix, both_ways)``, the direction column, its
    mode-specific suffix (``"bicycle"`` for cycling) and whether every edge is two-way. A
    missing direction column is added to ``edges`` as empty, with a warning."""
    allowed_network_types = Conf._possible_network_filters

    # Validate nodes and edges
    validate_node_gdf(nodes)
    validate_edge_gdf(edges)

    for col in [from_id_col, to_id_col]:
        if col not in edges.columns:
            raise ValueError(
                "Required column '{col}' does not exist in edges.".format(col=col)
            )

    direction, direction_suffix = _direction_column(direction, edges.columns)
    if direction not in edges.columns:
        edges[direction] = None

    if node_id_col not in nodes.columns:
        raise ValueError(
            "Required column '{col}' does not exist in nodes.".format(col=node_id_col)
        )

    # The network type given, else the one recorded on the edges or nodes
    if network_type is not None:
        net_type = network_type
        if normalize_network_type(net_type) not in allowed_network_types:
            raise ValueError(_UNKNOWN_NETWORK_TYPE)
    else:
        net_type = edges.attrs.get("network_type")
        if net_type not in allowed_network_types:
            net_type = nodes.attrs.get("network_type")
        if net_type not in allowed_network_types:
            txt = ", ".join(allowed_network_types)
            raise ValueError(
                "Could not detect the network type from the edges. "
                "In order to save the graph, specify the type of your network"
                "with 'network_type' -parameter."
                "Possible network types are: " + txt
            )

    return (direction, *_travel_rules(direction_suffix, net_type, force_bidirectional))


def _direction_column(direction, columns):
    """The column of ``direction`` and its mode-specific suffix: ``"oneway:bicycle"`` is
    the column ``oneway`` with the suffix ``bicycle``, whose column overrides it per edge.
    Warns when ``columns`` lack the column, as every edge is then two-way."""
    direction_suffix = None
    if ":" in direction:
        direction, direction_suffix = direction.split(":", 1)
    if direction not in columns:
        warnings.warn(
            f"Column '{direction}' missing in the edges GeoDataFrame. "
            f"Assuming all edges to be bidirectional "
            f"(travel allowed to both directions).",
            UserWarning,
            stacklevel=4,
        )
    return direction, direction_suffix


def _travel_rules(direction_suffix, network_type, force_bidirectional):
    """The direction suffix that a network of ``network_type`` (a pyrosm name or an OSMnx
    alias) honours, and whether all its edges are two-way: ``(direction_suffix,
    both_ways)``."""
    net_type = normalize_network_type(network_type)

    # Cycling honours bicycle-specific direction tags (oneway:bicycle) so that
    # contraflow cycling on one-way streets is modelled correctly.
    if direction_suffix is None and net_type == "cycling":
        direction_suffix = "bicycle"

    # Walking and the unrestricted networks are bidirectional by default; driving and
    # cycling are directed (they honour oneway, and cycling additionally honours
    # oneway:bicycle). force_bidirectional overrides this for any type.
    both_ways = force_bidirectional or net_type in ["walking", "all", "all_public"]
    return direction_suffix, both_ways


def _segment_rows(edges, direction, force_bidirectional, network_type):
    """:func:`directed_rows` of the segments that ``edges`` (a
    :class:`~pyrosm.graph_simplify._WayEdges`) reads, each travelled as its way is. A
    missing direction column is added to them as empty, with a warning."""
    direction, direction_suffix = _direction_column(direction, edges.columns)
    if direction not in edges.columns:
        edges.add_empty_column(direction)
    direction_suffix, both_ways = _travel_rules(
        direction_suffix, network_type, force_bidirectional
    )
    return directed_rows(
        edges.ways, direction, direction_suffix, both_ways, edges.way_of
    )


def _edges_between(nodes, edges, from_id_col, to_id_col, node_id_col):
    """The rows of ``edges`` whose two end nodes are among ``nodes``: an edge to a node the
    frame leaves out, e.g. after the nodes were cropped, cannot be in the graph."""
    ids = nodes[node_id_col]
    known = (edges[from_id_col].isin(ids) & edges[to_id_col].isin(ids)).to_numpy()
    return edges if known.all() else edges[known]


def _xy_from_geometry(nodes):
    """``nodes`` with ``x`` and ``y`` from their geometry, in the CRS of the graph, in place
    of the ``lon`` and ``lat`` degrees of the read and of any ``x`` and ``y`` they had.
    """
    nodes = nodes.drop(columns=["x", "y"], errors="ignore")
    nodes = nodes.rename(columns={"lon": "x", "lat": "y"})
    return nodes.assign(x=nodes.geometry.x.to_numpy(), y=nodes.geometry.y.to_numpy())


def _routing_nodes(nodes, node_id_col):
    """The nodes of a pandana or pandarm network: ``x`` and ``y`` from their geometry
    and indexed by their id."""
    nodes = nodes.set_index(node_id_col, drop=False).rename_axis([None])
    return _xy_from_geometry(nodes)


def graph_tables(
    nodes,
    edges,
    direction="oneway",
    from_id_col="u",
    to_id_col="v",
    node_id_col="id",
    force_bidirectional=False,
    network_type=None,
    retain_all=False,
    simplify=True,
    simplify_kwargs=None,
):
    """The nodes and edges of a graph as GeoDataFrames, as the graph exporters build it: the
    edges directed (see :func:`get_directed_edges`), simplified when ``simplify`` and, unless
    ``retain_all``, only the connected part kept. Edges with an end node that is not among
    ``nodes`` are left out. The parameters are those of
    :func:`to_networkx`. The progress is shown as :func:`pyrosm.enable_logging` sets it, by
    step."""
    with _steps(
        "Preparing the graph tables", _step_names(simplify, retain_all)
    ) as step:
        return _tables(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
            retain_all,
            simplify,
            simplify_kwargs,
            step,
        )


def _tables(
    nodes,
    edges,
    direction,
    from_id_col,
    to_id_col,
    node_id_col,
    force_bidirectional,
    network_type,
    retain_all,
    simplify,
    simplify_kwargs,
    step,
):
    """:func:`graph_tables`, reporting the start of each step to ``step(name)``. Edges whose
    end nodes are not among ``nodes`` are left out (see :func:`_edges_between`), and the
    edges returned are numbered from 0.

    With ``simplify``, the directed edges are row numbers of ``edges`` and a direction each
    (:func:`directed_rows`), and only the simplified edges become rows of a new table.
    ``simplify_kwargs`` are the optional parameters of :func:`simplify_graph`
    (``edge_attrs_differ``, ``node_attrs_include``, ``remove_rings``, ``track_merged``,
    ``length_cols``, ``edge_attr_aggs``). ``nodes`` and ``edges`` may also be the readers of a network read
    for graph export (see :func:`pyrosm.engine.readers.graph_network`), always
    simplified, with the given ``network_type`` and the default id columns."""
    step("directing the edges")
    if not isinstance(edges, _WayEdges):
        edges = _edges_between(nodes, edges, from_id_col, to_id_col, node_id_col)
    if isinstance(edges, _WayEdges):
        rows, swapped = _segment_rows(
            edges, direction, force_bidirectional, network_type
        )
    elif simplify:
        direction, direction_suffix, both_ways = _direction_rules(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
        )
        rows, swapped = directed_rows(edges, direction, direction_suffix, both_ways)
        nodes, edges = _TableNodes(nodes, node_id_col), _TableEdges(edges)
    else:
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
    if simplify:
        step("simplifying")
        simplified = _simplify_rows(
            nodes,
            edges,
            rows,
            swapped,
            from_id_col=from_id_col,
            to_id_col=to_id_col,
            **(simplify_kwargs or {}),
        )
        if isinstance(edges, _WayEdges):
            nodes.release()
            edges.release()
        nodes, edges = simplified
    # Keep only strongly connected component if not specifically requested otherwise
    if not retain_all:
        step("keeping the largest connected part")
        nodes, edges = get_connected_edges(
            nodes, edges, from_id_col, to_id_col, node_id_col
        )
    return nodes, edges.reset_index(drop=True)


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
    simplify=True,
    simplify_kwargs=None,
):
    """
    Creates a NetworkX.MultiDiGraph from given OSM GeoDataFrame.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing road network data.

    network_type : str
        The type of the network. Possible values:
              - `'walking'` (OSMnx `'walk'`)
              - `'cycling'` (OSMnx `'bike'`)
              - `'driving'` (OSMnx `'drive'`)
              - `'driving+service'` (OSMnx `'drive_service'`)
              - `'all'`
              - `'all_public'`.

            The OSMnx name of a network type is accepted as well.

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
        By default, a bidirectional graph is created for walking and all, and a
        directed graph for driving and cycling (oneway streets are taken into
        account; cycling additionally honours oneway:bicycle for contraflow).
        Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'
        and 'all_public', plus the OSMnx names 'walk', 'bike', 'drive' and
        'drive_service'.

    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the connected edges.

    osmnx_compatible : bool (default True)
        if True, modifies the edge and node-attribute naming to be compatible with OSMnx
        (allows utilizing all OSMnx functionalities): the node/edge id is renamed to
        ``osmid``, the nodes get ``x``/``y`` from their geometry, in the CRS of the frames,
        in place of ``lon``/``lat``, and node ``tags`` are stored as a JSON string. The node ``osmid`` is the key of the node in the graph and is not
        repeated as a node attribute. No ``key`` edge attribute is added; an edge column
        named ``key`` is renamed to ``key_tag``.

    Returns
    -------
    networkx.MultiDiGraph

    """

    with _steps(
        "Building the NetworkX graph",
        _step_names(simplify, retain_all, "building the graph"),
    ) as step:
        # Prepare the data
        nodes, edges = _tables(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
            retain_all,
            simplify,
            simplify_kwargs,
            step,
        )

        if osmnx_compatible:
            # OSMnx carries the parallel-edge key in the graph structure, not as an
            # edge attribute, and an attribute named 'key' collides with it. Surface
            # a tag with that name as 'key_tag' instead (as is done for 'id').
            if "key" in edges.columns:
                edges = edges.rename(columns={"key": "key_tag"})

            # Follow the naming convention of OSMnx
            nodes = _xy_from_geometry(nodes.rename(columns={node_id_col: "osmid"}))
            edges = edges.rename(columns={edge_id_col: "osmid"})
            node_id_col = "osmid"

            # OSMnx handles node attributes as scalars, so store the free-form tags
            # as a JSON string like the edge tags already are.
            if "tags" in nodes.columns:
                nodes["tags"] = [
                    dumps(tags) if isinstance(tags, dict) else tags
                    for tags in nodes["tags"]
                ]

        # Create NetworkX graph (nodes are keyed by node_id_col internally, so the
        # input frame's index does not need to be set to the node id).
        step("building the graph")
        graph = _create_nxgraph(
            nodes,
            edges,
            from_id_col,
            to_id_col,
            node_id_col,
            drop_node_id_attribute=osmnx_compatible,
        )
    graph.graph["simplified"] = bool(simplify)
    return graph


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
    simplify=True,
    simplify_kwargs=None,
):
    """
    Creates an iGraph from given OSM GeoDataFrame.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing road network data.

    network_type : str
        The type of the network. Possible values:
              - `'walking'` (OSMnx `'walk'`)
              - `'cycling'` (OSMnx `'bike'`)
              - `'driving'` (OSMnx `'drive'`)
              - `'driving+service'` (OSMnx `'drive_service'`)
              - `'all'`
              - `'all_public'`.

            The OSMnx name of a network type is accepted as well.

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
        By default, a bidirectional graph is created for walking and all, and a
        directed graph for driving and cycling (oneway streets are taken into
        account; cycling additionally honours oneway:bicycle for contraflow).
        Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'
        and 'all_public', plus the OSMnx names 'walk', 'bike', 'drive' and
        'drive_service'.

    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the connected edges.

    Returns
    -------
    igraph.Graph

    """
    with _steps(
        "Building the igraph graph",
        _step_names(simplify, retain_all, "building the graph"),
    ) as step:
        # Prepare the data
        nodes, edges = _tables(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
            retain_all,
            simplify,
            simplify_kwargs,
            step,
        )
        step("building the graph")
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
    simplify=True,
    simplify_kwargs=None,
):
    """
    Creates a pandana Network from given OSM GeoDataFrames (deprecated: use
    :func:`to_pandarm`). The edges are directed, simplified and kept as
    :func:`to_igraph` does, whose parameters these are; ``weight_cols`` are the edge
    columns used as impedances, summed over the edges a simplification merges. The
    nodes are indexed by ``node_id_col`` at the ``x``/``y`` of their geometry.

    Returns
    -------
    pandana.Network
    """
    warn_future(
        "The pandana graph backend (graph_type='pandana', pyrosm.graphs.to_pandana) "
        "is deprecated because pandana is unmaintained and incompatible with NumPy 2 "
        "on Windows; it will be removed in pyrosm 1.0. Use graph_type='pandarm' "
        "(pyrosm.graphs.to_pandarm) instead."
    )
    with _steps(
        "Building the pandana network",
        _step_names(simplify, retain_all, "building the network"),
    ) as step:
        # Prepare the data
        nodes, edges = _tables(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
            retain_all,
            simplify,
            _summing_weights(simplify_kwargs, weight_cols),
            step,
        )

        nodes = _routing_nodes(nodes, node_id_col)
        step("building the network")
        return _create_pdgraph(nodes, edges, from_id_col, to_id_col, weight_cols)


def to_pandarm(
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
    simplify=True,
    simplify_kwargs=None,
):
    """
    Creates a pandarm Network from given OSM GeoDataFrames. The edges are directed,
    simplified and kept as :func:`to_igraph` does, whose parameters these are;
    ``weight_cols`` are the edge columns used as impedances, summed over the edges a
    simplification merges. The nodes are indexed by ``node_id_col`` at the ``x``/``y``
    of their geometry, in the CRS of the frames, which is the network's ``crs``.

    Returns
    -------
    pandarm.Network
    """
    with _steps(
        "Building the pandarm network",
        _step_names(simplify, retain_all, "building the network"),
    ) as step:
        # Prepare the data
        nodes, edges = _tables(
            nodes,
            edges,
            direction,
            from_id_col,
            to_id_col,
            node_id_col,
            force_bidirectional,
            network_type,
            retain_all,
            simplify,
            _summing_weights(simplify_kwargs, weight_cols),
            step,
        )

        nodes = _routing_nodes(nodes, node_id_col)
        step("building the network")
        return _create_pandarm_graph(nodes, edges, from_id_col, to_id_col, weight_cols)
