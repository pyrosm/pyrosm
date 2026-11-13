"""A 3D view of an indoor network, drawn with plotly (install it with
``pip install "pyrosm[3d]"``).

The edges of :meth:`pyrosm.OSM.get_network` or :meth:`pyrosm.OSM.get_graph` with
``indoor_elements=True`` are drawn at their levels: walkways, stairs, escalators, moving
walkways, lifts, platforms and the lines across areas, each kind a legend entry that hides
or shows it.
A walkway tagged with several levels is drawn flat at the lowest, dotted; an edge without
a level at its layer when that is below street level (a tunnel at least one below), else
at street level. The vertical axis counts OSM levels, not metres.
"""

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame

from pyrosm.indoor import float_column
from pyrosm.levels import level_ranges, tag_layer

# (name in the legend, colour, line width, dash) of each kind of edge
_STYLES = {
    "street": ("Walkways at street level", "#c3c7ce", 1.5, "solid"),
    "levels": ("Walkways on other levels", "#3987e5", 3.5, "solid"),
    "multi": ("Walkways tagged with several levels", "#3987e5", 3.0, "dot"),
    "escalator": ("Escalators", "#c98500", 6.0, "solid"),
    "moving_walkway": ("Moving walkways", "#c98500", 6.0, "dash"),
    "stairs": ("Stairs", "#444b55", 5.0, "solid"),
    "lift": ("Lifts", "#8e44ad", 5.0, "dash"),
    "platform": ("Platforms", "#199e70", 5.0, "solid"),
    "crossing": ("Lines across areas", "#b8bcc4", 1.5, "solid"),
    "area": ("Area outlines", "#8a8f98", 2.5, "dot"),
}
# The route: red, wide and solid, drawn over the edges; the escalators are amber so that
# they stay apart from it
_ROUTE_COLOUR = "#e8002d"
_PLATFORM_TAGS = ("railway", "public_transport", "highway")
_ENTRANCES = ("subway_entrance", "train_station_entrance")


def _plotly():
    """The ``plotly.graph_objects`` module, or an ImportError that names the extra."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            'plot_3d needs plotly: install it with pip install "pyrosm[3d]".'
        ) from error
    return go


def _graph_frames(graph):
    """``(edges, nodes)`` frames of an igraph or networkx ``graph`` of get_graph, each
    edge once whatever its directions (parallel edges of other ways kept); a pandana or
    pandarm network (no line geometry) raises a ValueError."""
    module = type(graph).__module__.split(".")[0]
    if module == "networkx":
        edges = pd.DataFrame(
            [{**data, "u": u, "v": v} for u, v, data in graph.edges(data=True)]
        )
        if "id" not in edges.columns:
            edges = edges.rename(columns={"osmid": "id"})  # pyrosm's networkx name
        nodes = pd.DataFrame([{**data, "id": n} for n, data in graph.nodes(data=True)])
    elif module == "igraph":
        ids = graph.vs["id"] if "id" in graph.vs.attributes() else graph.vs["name"]
        edges = pd.DataFrame(
            [
                {**edge.attributes(), "u": ids[edge.source], "v": ids[edge.target]}
                for edge in graph.es
            ]
        )
        nodes = pd.DataFrame([vertex.attributes() for vertex in graph.vs])
        nodes["id"] = ids
    else:
        raise ValueError(
            "plot_3d draws the frames of get_network and igraph or networkx graphs, "
            "not a %s object; pass the frames of get_network(..., nodes=True)."
            % type(graph).__name__
        )
    if edges.empty:
        return GeoDataFrame(geometry=[], crs=4326), None
    edges = GeoDataFrame(edges, geometry="geometry", crs=4326)
    # a directed graph has a two-way edge in both directions, with the same end nodes,
    # line and ids (a simplified edge's id list reversed): draw it once
    ends = np.sort(edges[["u", "v"]].to_numpy(), axis=1)
    shapes = shapely.to_wkb(shapely.normalize(edges.geometry.to_numpy()))
    ids = edges.get("id", pd.Series(None, index=edges.index))
    ids = ids.map(lambda i: str(sorted(i) if isinstance(i, list) else i))
    keys = pd.DataFrame(
        {"a": ends[:, 0], "b": ends[:, 1], "shape": shapes, "id": ids.to_numpy()}
    )
    edges = edges[~keys.duplicated().to_numpy()]
    for x, y in (("x", "y"), ("lon", "lat")):
        if x in nodes.columns and y in nodes.columns:
            points = shapely.points(nodes[x].astype(float), nodes[y].astype(float))
            return edges, GeoDataFrame(nodes, geometry=points, crs=4326)
    return edges, None


def _edge_kind(tags, vertical, u_level, v_level, crossing):
    """``(kind, z at u, z at v)`` of an edge with the OSM ``tags``, ``vertical`` value
    and end levels."""
    if vertical in ("escalator", "moving_walkway", "stairs", "lift"):
        kind = vertical
    elif crossing:
        kind = "crossing"
    elif any(tags.get(key) == "platform" for key in _PLATFORM_TAGS):
        kind = "platform"
    else:
        kind = None
    if np.isnan(u_level) and np.isnan(v_level):
        ranges = level_ranges(tags.get("level"))
        if ranges and (len(ranges) > 1 or ranges[0][0] != ranges[0][1]):
            low = min(r[0] for r in ranges)
            return kind or "multi", low, low
        z = ranges[0][0] if ranges else min(tag_layer(tags), 0.0)
        return kind or ("street" if z == 0 else "levels"), z, z
    u_level = v_level if np.isnan(u_level) else u_level
    v_level = u_level if np.isnan(v_level) else v_level
    flat = u_level == v_level == 0
    return kind or ("street" if flat else "levels"), u_level, v_level


def _edge_kinds(edges):
    """``(row, tags, kind, z at u, z at v)`` of each of ``edges`` (see
    :func:`_edge_kind`), the row a named tuple of the frame."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    tags = _frame_tags(edges, _frame_tag_columns(edges))
    u_levels, v_levels = (float_column(edges, col) for col in ("u_level", "v_level"))
    verticals = edges.get("vertical", pd.Series(None, index=edges.index)).to_numpy()
    crossings = (
        edges.get("walking_area", pd.Series(False, index=edges.index))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    for i, (row_tags, row) in enumerate(zip(tags, edges.itertuples())):
        kind, z_u, z_v = _edge_kind(
            row_tags, verticals[i], u_levels[i], v_levels[i], crossings[i]
        )
        yield row, row_tags, kind, z_u, z_v


def _line_points(line, z_u, z_v, origin, vertical=False):
    """x, y (metres from ``origin``) and z of the points of ``line``, z running from
    ``z_u`` to ``z_v`` along it; a vertical line at its first point when ``vertical``
    or when it has no length, as a lift."""
    xy = shapely.get_coordinates(line)
    steps = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if vertical or steps[-1] == 0:
        xy, z = xy[:1].repeat(2, axis=0), np.array([z_u, z_v])
    else:
        z = z_u + (z_v - z_u) * steps / steps[-1]
    return xy[:, 0] - origin[0], xy[:, 1] - origin[1], z


class _Trace:
    """The points of one legend entry, lines separated by gaps."""

    def __init__(self):
        self.x, self.y, self.z, self.text = [], [], [], []

    def add(self, x, y, z, text):
        """Add one line through the points ``x``, ``y``, ``z`` (rounded to hundredths),
        labelled ``text``."""
        self.x += np.round(x, 2).tolist() + [None]
        self.y += np.round(y, 2).tolist() + [None]
        self.z += np.round(z, 2).tolist() + [None]
        self.text += [text] * len(x) + [None]


def plot_3d(edges, nodes=None, center=None, radius=None, route=None, areas=None):
    """A plotly Figure of an indoor network in 3D (see the module docstring).

    Parameters
    ----------
    edges : GeoDataFrame, igraph.Graph or networkx.MultiDiGraph
        The edges of ``get_network(..., nodes=True, indoor_elements=True)``, or a graph
        of ``get_graph(indoor_elements=True)`` (igraph or networkx).
    nodes : GeoDataFrame (optional)
        The nodes of the same read: their station entrances (``railway`` tagged
        ``subway_entrance`` or ``train_station_entrance``) are drawn as markers, and the
        ``route`` is drawn through them. Taken from the graph when ``edges`` is one.
    center : (lon, lat) or shapely Point (optional)
        The middle of the view; by default the middle of the edges.
    radius : float (optional)
        Draw only the edges and areas that reach within this many metres (at least 0)
        of ``center``, and the entrances within it; the ``route`` is drawn in full.
    route : list of node ids (optional)
        A path through the network, such as a shortest path of a networkx graph, drawn
        as a wide red line along the edges between its nodes (``u`` and ``v``), also
        beyond ``radius``, with markers at its two ends; a click on "Route" in the
        legend hides it.
    areas : GeoDataFrame (optional)
        Polygons, such as platform areas or squares, drawn as outlines at their
        ``level`` (street level without one); lines are drawn as they are.

    Returns
    -------
    plotly.graph_objects.Figure
        Show it in a notebook with ``fig.show()``, in a browser window with
        ``fig.show(renderer="browser")``, or save it with ``fig.write_html(path)``.
    """
    go = _plotly()
    if radius is not None and not radius >= 0:
        raise ValueError("radius must be at least 0 metres, not %r." % (radius,))
    if not isinstance(edges, GeoDataFrame):
        edges, graph_nodes = _graph_frames(edges)
        nodes = graph_nodes if nodes is None else nodes
    if edges.empty:
        raise ValueError("plot_3d got no edges to draw.")
    crs = edges.estimate_utm_crs()
    metric = edges.to_crs(crs)
    if center is None:
        middle = shapely.box(*metric.total_bounds).centroid
    else:
        point = center if hasattr(center, "x") else shapely.Point(*center)
        middle = GeoDataFrame(geometry=[point], crs=4326).to_crs(crs).geometry[0]
    origin = (middle.x, middle.y)
    on_route = metric.iloc[:0]
    if route is not None and "u" in metric.columns and "v" in metric.columns:
        steps = set(zip(route[:-1], route[1:]))
        steps |= {(b, a) for a, b in steps}
        on_route = metric[[pair in steps for pair in zip(metric["u"], metric["v"])]]
    if radius is not None:
        metric = metric[metric.distance(middle) <= radius]
    traces = {kind: _Trace() for kind in _STYLES}
    for row, row_tags, kind, z_u, z_v in _edge_kinds(metric):
        x, y, z = _line_points(row.geometry, z_u, z_v, origin, vertical=kind == "lift")
        text = "%s %s · %s<br>level %s" % (
            getattr(row, "osm_type", "way"),
            getattr(row, "id", ""),
            row_tags.get("highway", kind),
            row_tags.get("level", "not tagged"),
        )
        traces[kind].add(x, y, z, text)
    ends = {
        (row.u, row.v): (row.geometry, z_u, z_v, kind == "lift")
        for row, _, kind, z_u, z_v in _edge_kinds(on_route)
    }
    figure = go.Figure()
    if areas is not None:
        _add_area_outlines(traces["area"], areas.to_crs(crs), middle, radius)
    for kind, (name, colour, width, dash) in _STYLES.items():
        if traces[kind].x:
            figure.add_trace(_lines(go, traces[kind], name, colour, width, dash))
    if route is not None:
        _add_route(go, figure, route, ends, origin)
    if nodes is not None:
        _add_entrances(go, figure, nodes.to_crs(crs), middle, radius)
    figure.update_layout(
        scene=dict(
            xaxis_title="east (m)",
            yaxis_title="north (m)",
            zaxis_title="level",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        legend=dict(itemclick="toggle"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return figure


def _add_route(go, figure, route, ends, origin):
    """Add the ``route`` (node ids) as a line along the edges between consecutive nodes,
    either way (``ends``: ``{(u, v): (line, z at u, z at v, lift)}``), with markers at
    its first and last point; steps without an edge are left out."""
    line_trace = _Trace()
    for a, b in zip(route[:-1], route[1:]):
        if (a, b) in ends:
            line, z_a, z_b, lift = ends[(a, b)]
        elif (b, a) in ends:
            line, z_b, z_a, lift = ends[(b, a)]
            line = line.reverse()
        else:
            continue
        line_trace.add(*_line_points(line, z_a, z_b, origin, vertical=lift), "route")
    if not line_trace.x:
        return
    figure.add_trace(_lines(go, line_trace, "Route", _ROUTE_COLOUR, 10.0, "solid"))
    points = [i for i, x in enumerate(line_trace.x) if x is not None]
    first, last = points[0], points[-1]
    figure.add_trace(
        go.Scatter3d(
            x=[line_trace.x[first], line_trace.x[last]],
            y=[line_trace.y[first], line_trace.y[last]],
            z=[line_trace.z[first], line_trace.z[last]],
            text=["route start", "route end"],
            mode="markers",
            name="Route ends",
            legendgroup="Route",
            showlegend=False,
            marker=dict(size=7, symbol="circle", color=_ROUTE_COLOUR),
            hovertemplate="%{text}<extra></extra>",
        )
    )


def _lines(go, trace, name, colour, width, dash):
    """A plotly line trace of the points of ``trace``, its own group in the legend."""
    return go.Scatter3d(
        x=trace.x,
        y=trace.y,
        z=trace.z,
        text=trace.text,
        mode="lines",
        name=name,
        legendgroup=name,
        line=dict(color=colour, width=width, dash=dash),
        hovertemplate="%{text}<extra></extra>",
    )


def _near(geometry, middle, radius):
    """Whether ``geometry`` reaches within ``radius`` metres of the point ``middle``
    (always without a radius)."""
    return radius is None or geometry.distance(middle) <= radius


def _add_area_outlines(trace, areas, middle, radius):
    """Add the outlines of the polygons and the lines among ``areas`` (in metres) that
    reach within ``radius`` of ``middle``, at the lowest level of their ``level`` tag;
    points are left out."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    area_tags = _frame_tags(areas, _frame_tag_columns(areas))
    for geometry, tags in zip(areas.geometry, area_tags):
        if geometry is None or not _near(geometry, middle, radius):
            continue
        ranges = level_ranges(tags.get("level"))
        z = min(r[0] for r in ranges) if ranges else 0.0
        for part in shapely.get_parts(geometry):
            if part.geom_type == "Polygon":
                lines = shapely.get_rings(part)
            elif part.geom_type == "LineString":
                lines = [part]
            else:
                continue
            for line in lines:
                points = _line_points(line, z, z, (middle.x, middle.y))
                trace.add(*points, "area, level %g" % z)


def _add_entrances(go, figure, nodes, middle, radius):
    """Add the station entrances among ``nodes`` (in metres) within ``radius`` of
    ``middle`` as markers."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    x, y, z, text = [], [], [], []
    node_tags = _frame_tags(nodes, _frame_tag_columns(nodes))
    for row_tags, row in zip(node_tags, nodes.itertuples()):
        if row_tags.get("railway") not in _ENTRANCES:
            continue
        point = row.geometry
        if not _near(point, middle, radius):
            continue
        ranges = level_ranges(row_tags.get("level"))
        x.append(point.x - middle.x)
        y.append(point.y - middle.y)
        z.append(min(r[0] for r in ranges) if ranges else 0.0)
        text.append("entrance %s" % row.id)
    if x:
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=text,
                mode="markers",
                name="Station entrances",
                marker=dict(size=3.5, symbol="diamond", color="#1f2329"),
                hovertemplate="%{text}<extra></extra>",
            )
        )
