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
    EDGE_ATTRS_DIFFER,
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
from pyrosm.utils.progress import stepping, validate_progress
from pyrosm.config import Conf
from pyrosm.config.osm_filters import _UNKNOWN_NETWORK_TYPE, normalize_network_type
from rapidjson import dumps
import contextlib
import numpy as np
import pandas as pd
import warnings

# The shares of the steps of a graph export, in seconds of exports of the driving network
# of Paris (242 MB), and the step that builds the graph of each graph type with its share.
_STEP_SHARES = {
    "directing the edges": 0.1,
    "simplifying": 1.2,
    "keeping the largest connected part": 0.35,
}
_BUILDING_STEPS = {
    "igraph": ("building the graph", 1.1),
    "networkx": ("building the graph", 1.9),
    "pandarm": ("building the network", 9.7),
    "pandana": ("building the network", 5.0),
}


def step_shares(simplify, retain_all, graph_type=None):
    """The shares of the steps of making a graph's tables, and of building the graph of
    ``graph_type`` when given."""
    names = ["directing the edges"]
    if simplify:
        names.append("simplifying")
    if not retain_all:
        names.append("keeping the largest connected part")
    shares = {name: _STEP_SHARES[name] for name in names}
    if graph_type is not None:
        name, share = _BUILDING_STEPS[graph_type]
        shares[name] = share
    return shares


@contextlib.contextmanager
def _steps(desc, shares, progress):
    """A function ``step(name)`` that starts each step of ``shares`` (step name ->
    share) of the work described ``desc``, reported as ``progress`` says (see
    :func:`~pyrosm.utils.progress.stepping`): a bar that names the current step, shown
    after 2 seconds, or ``progress(done, 100)`` calls."""
    validate_progress(progress)
    with stepping(progress, desc, shares, delay=2.0, leave=False, timed=True) as steps:
        yield steps.start_step


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
    osmnx_flags=False,
):
    """Prepares the edges and nodes for exporting to different graphs; with
    ``osmnx_flags`` the directed edges get the columns of :func:`_osmnx_flags`."""
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
    flags = None
    if osmnx_flags:
        flags = _osmnx_flags(
            *directed_rows(edges, direction, direction_suffix, both_ways)
        )
    edges = generate_directed_edges(
        edges,
        direction,
        direction_suffix,
        from_id_col,
        to_id_col,
        force_bidirectional=both_ways,
    )
    for name, values in (flags or {}).items():
        edges[name] = pd.Series(values.tolist(), index=edges.index, dtype=object)
    return nodes, edges


# The columns of the OSMnx ``oneway`` and ``reversed`` of directed edges, named apart
# from the OSM tags of those names until the export renames them.
_OSMNX_ONEWAY = "osmnx:oneway"
_OSMNX_REVERSED = "osmnx:reversed"


def _osmnx_flags(rows, swapped):
    """The OSMnx ``oneway`` and ``reversed`` of the directed edges ``rows``/``swapped``
    (see :func:`~pyrosm.graph_export.directed_rows`), by column: an edge is one-way when
    its row is travelled one way only, and reversed when it runs against its node order.
    """
    rows = np.asarray(rows)
    counts = np.bincount(rows) if len(rows) else np.zeros(0, dtype=np.int64)
    return {
        _OSMNX_ONEWAY: counts[rows] == 1,
        _OSMNX_REVERSED: np.asarray(swapped, dtype=bool),
    }


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
    _validate_frames(nodes, edges, from_id_col, to_id_col, node_id_col)

    direction, direction_suffix = _direction_column(direction, edges.columns)
    if direction not in edges.columns:
        edges[direction] = None

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


def _split_direction(direction):
    """``direction`` as its column and its mode-specific suffix: ``"oneway:bicycle"`` is
    the column ``oneway`` with the suffix ``bicycle``, whose column overrides it per edge.
    """
    if ":" in direction:
        column, suffix = direction.split(":", 1)
        return column, suffix
    return direction, None


def direction_tags(direction, network_type, force_bidirectional=False):
    """The tags a graph of ``network_type`` reads its edges' directions from: the
    ``direction`` tag (``oneway``), the form of it that the network honours (such as
    ``oneway:bicycle`` for cycling) and ``junction`` (a roundabout is one-way); none when
    every edge is two-way."""
    column, suffix = _split_direction(direction)
    suffix, both_ways = _travel_rules(suffix, network_type, force_bidirectional)
    if both_ways:
        return []
    tags = [column, "junction"]
    if suffix:
        tags.insert(1, "%s:%s" % (column, suffix))
    return tags


# The columns of a network frame that are not tags, which it has whatever tags are read.
_NETWORK_FIELDS = {
    "id",
    "osm_type",
    "version",
    "timestamp",
    "changeset",
    "visible",
    "tags",
    "nodes",
    "geometry",
    "length",
    "u",
    "v",
}


def graph_tags(direction, network_type, force_bidirectional, simplify, simplify_kwargs):
    """The tags a graph of ``network_type`` is built from: those its edges' directions are
    read from (see :func:`direction_tags`) and, when ``simplify``, the tags a simplified
    edge has one value of (``edge_attrs_differ`` of ``simplify_kwargs``, by default
    ``maxspeed`` and ``highway``; a column every network has is not a tag)."""
    tags = direction_tags(direction, network_type, force_bidirectional)
    if simplify:
        differ = (simplify_kwargs or {}).get("edge_attrs_differ", EDGE_ATTRS_DIFFER)
        tags += [t for t in differ or () if t not in tags and t not in _NETWORK_FIELDS]
    return tags


def _direction_column(direction, columns):
    """The column of ``direction`` and its mode-specific suffix (see
    :func:`_split_direction`). Warns when ``columns`` lack the column, as every edge is
    then two-way."""
    direction, direction_suffix = _split_direction(direction)
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


def _validate_frames(nodes, edges, from_id_col, to_id_col, node_id_col):
    """Raise ``ValueError`` unless ``nodes`` and ``edges`` are GeoDataFrames with the
    node id column and the end node columns of a graph."""
    validate_node_gdf(nodes)
    validate_edge_gdf(edges)
    for col in [from_id_col, to_id_col]:
        if col not in edges.columns:
            raise ValueError(
                "Required column '{col}' does not exist in edges.".format(col=col)
            )
    if node_id_col not in nodes.columns:
        raise ValueError(
            "Required column '{col}' does not exist in nodes.".format(col=node_id_col)
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
    progress=None,
):
    """The nodes and edges of a graph as GeoDataFrames, as the graph exporters build it: the
    edges directed (see :func:`get_directed_edges`), unless ``retain_all`` only their
    largest connected part kept, and simplified when ``simplify``, as when the connected
    part is kept before simplifying (as OSMnx does). Edges with an end node that is not
    among ``nodes`` are left out; nodes and edges are numbered from 0. The parameters are
    those of :func:`to_networkx`, also ``progress``."""
    with _steps(
        "Preparing the graph tables", step_shares(simplify, retain_all), progress
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
    osmnx_flags=False,
):
    """:func:`graph_tables`, reporting the start of each step to ``step(name)``. Edges whose
    end nodes are not among ``nodes`` are left out (see :func:`_edges_between`), and the
    edges returned are numbered from 0. With ``osmnx_flags`` the edges get the OSMnx
    ``oneway`` and ``reversed`` (see :func:`_osmnx_flags`) in their own columns.

    With ``simplify``, the directed edges are row numbers of ``edges`` and a direction each
    (:func:`directed_rows`), and only the simplified edges become rows of a new table. The
    connected part is found on the simplified graph; when it leaves out edges, the
    directed edges between the nodes of the edges kept are simplified again.
    ``simplify_kwargs`` are the optional parameters of :func:`simplify_graph`
    (``edge_attrs_differ``, ``node_attrs_include``, ``remove_rings``, ``track_merged``,
    ``length_cols``, ``edge_attr_aggs``). ``nodes`` and ``edges`` may also be the readers of a network read
    for graph export (see :func:`pyrosm.engine.readers.graph_network`), always
    simplified, with the given ``network_type`` and the default id columns."""
    step("directing the edges")
    if not isinstance(edges, _WayEdges):
        _validate_frames(nodes, edges, from_id_col, to_id_col, node_id_col)
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
            osmnx_flags,
        )
    if simplify:
        step("simplifying")
        flags = _osmnx_flags(rows, swapped) if osmnx_flags else {}

        def simplified(kept):
            return _simplify_rows(
                nodes,
                edges,
                rows[kept],
                swapped[kept],
                from_id_col=from_id_col,
                to_id_col=to_id_col,
                directions={name: values[kept] for name, values in flags.items()},
                with_chains=True,
                **(simplify_kwargs or {}),
            )

        graph_nodes, graph_edges, chains = simplified(np.arange(len(rows)))
    else:
        graph_nodes, graph_edges = nodes, edges
    # Keep only strongly connected component if not specifically requested otherwise
    if not retain_all:
        step("keeping the largest connected part")
        connected_nodes, connected_edges = get_connected_edges(
            graph_nodes, graph_edges, from_id_col, to_id_col, node_id_col
        )
        if simplify and len(connected_edges) < len(graph_edges):
            # A node of a removed edge may now join two edges of one street. Simplify
            # again the directed edges between the nodes of the edges kept, as when the
            # connected part is kept before simplifying (as OSMnx does).
            ids = connected_nodes[node_id_col]
            kept = graph_edges[from_id_col].isin(ids) & graph_edges[to_id_col].isin(ids)
            in_kept = np.concatenate(
                [chains[k] for k in np.flatnonzero(kept.to_numpy())]
            )
            from_ids, to_ids = edges.values(from_id_col), edges.values(to_id_col)
            u = np.where(swapped, to_ids[rows], from_ids[rows])
            v = np.where(swapped, from_ids[rows], to_ids[rows])
            connected = np.union1d(u[in_kept], v[in_kept])
            positions = np.flatnonzero(np.isin(u, connected) & np.isin(v, connected))
            connected_nodes, connected_edges, _ = simplified(positions)
        graph_nodes, graph_edges = connected_nodes, connected_edges
    if simplify and isinstance(edges, _WayEdges):
        nodes.release()
        edges.release()
    return graph_nodes.reset_index(drop=True), graph_edges.reset_index(drop=True)


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
    progress=None,
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
        named ``key`` is renamed to ``key_tag``. The edges get OSMnx's ``oneway`` and
        ``reversed`` booleans: ``oneway`` when the edge can be travelled only in its own
        direction (by the rule the graph is directed by), ``reversed`` when it runs against
        the node order of its way (a simplified edge whose ways differ gets the list of
        their values, as in OSMnx); the OSM text of the ``oneway`` tag is ``oneway_tag``.

    progress : bool | callable | None (default: None)
        How the export shows its progress: ``None`` follows
        ``pyrosm.enable_logging(progress=...)``, ``True`` shows a bar on stderr once it
        has run 2 seconds, from 0 to 100 % with its current step in the label, ``False``
        nothing, and a callable is called as ``progress(done, 100)``.

    Returns
    -------
    networkx.MultiDiGraph

    """

    with _steps(
        "Building the NetworkX graph",
        step_shares(simplify, retain_all, "networkx"),
        progress,
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
            osmnx_compatible,
        )

        if osmnx_compatible:
            # OSMnx carries the parallel-edge key in the graph structure, not as an
            # edge attribute, and an attribute named 'key' collides with it. Surface
            # a tag with that name as 'key_tag' instead (as is done for 'id'), and the
            # oneway and reversed tags as 'oneway_tag' and 'reversed_tag', as the
            # graph's own booleans take their names.
            edges = edges.rename(
                columns={
                    "key": "key_tag",
                    "oneway": "oneway_tag",
                    "reversed": "reversed_tag",
                }
            ).rename(columns={_OSMNX_ONEWAY: "oneway", _OSMNX_REVERSED: "reversed"})

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
    progress=None,
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

    progress : bool | callable | None (default: None)
        How the export shows its progress: ``None`` follows
        ``pyrosm.enable_logging(progress=...)``, ``True`` shows a bar on stderr once it
        has run 2 seconds, from 0 to 100 % with its current step in the label, ``False``
        nothing, and a callable is called as ``progress(done, 100)``.

    Returns
    -------
    igraph.Graph

    """
    with _steps(
        "Building the igraph graph",
        step_shares(simplify, retain_all, "igraph"),
        progress,
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
    progress=None,
):
    """
    Creates a pandana Network from given OSM GeoDataFrames (deprecated: use
    :func:`to_pandarm`). The edges are directed, simplified and kept as
    :func:`to_igraph` does, whose parameters these are, also ``progress``;
    ``weight_cols`` are the edge columns used as impedances, summed over the edges a
    simplification merges. The nodes are indexed by ``node_id_col`` at the ``x``/``y``
    of their geometry.

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
        step_shares(simplify, retain_all, "pandana"),
        progress,
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
    progress=None,
):
    """
    Creates a pandarm Network from given OSM GeoDataFrames. The edges are directed,
    simplified and kept as :func:`to_igraph` does, whose parameters these are, also
    ``progress``; ``weight_cols`` are the edge columns used as impedances, summed over
    the edges a simplification merges. The nodes are indexed by ``node_id_col`` at the
    ``x``/``y`` of their geometry, in the CRS of the frames, which is the network's
    ``crs``.

    Returns
    -------
    pandarm.Network
    """
    with _steps(
        "Building the pandarm network",
        step_shares(simplify, retain_all, "pandarm"),
        progress,
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
