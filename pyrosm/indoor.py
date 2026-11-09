"""The indoor elements of a network read with ``indoor_elements=True``.

The platforms (:data:`PLATFORM_FILTER`) are edges of the network, also those that its
type leaves out, such as ``highway=platform`` in a walking network. The platform areas
and the indoor corridors and areas (:data:`INDOOR_AREA_FILTER`) are crossed as walking
areas are, each on its own level and together with the walking areas of a read with
``walking_areas=True``, and they are entered also through the doors on their outlines,
which connect areas that share them; the walking areas are not. The edges get the level
at each of their two ends, ``u_level`` and ``v_level``, from the OSM ``level`` tags (see
:func:`pyrosm.levels.level_ranges`). An edge with one level has it at both ends. A level
list or range on stairs, an escalator or a moving walkway runs between the levels of the
walkways at the two ends of its way or, at an end where no walkway has a level in the
range, the end node's own ``level``, and where neither gives an end, the way's
``incline`` (``up`` rises along the way's drawn direction, ``down`` falls); on another
walkway, such as a corridor tagged ``level=0;1``, the data does not say which part is on
which level, and both ends stay empty. A way of several edges also keeps the level of a
walkway that joins it in between (or of an inner node's own ``level``), an end without a
level then takes the other end of the range from the nearest such level, a way whose two
ends are on one level reaches the other end of its range in between, and the other inner
nodes get levels in proportion to the distance along the way. Stairs, escalators and
moving walkways without a level range (no ``level`` tag or a single level) take the
levels of the walkways that join their way, when these give two different levels, and
keep the outermost ones up to the way's ends. An edge without a ``level`` tag otherwise
has no level. The column ``vertical`` marks escalators, moving walkways and stairs; the
``length`` of an escalator is measured along its slope (see
:func:`with_indoor_columns`), and in a graph an escalator or moving walkway runs only in
the direction it moves (see :func:`moving_one_way`). A lift node (``highway=elevator``)
where walkways on several levels meet is split into one node per level, joined by lift
edges (see :func:`split_lifts`).
"""

import json
import math
import numbers

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from pyrosm.levels import level_ranges

# The network types whose reads take indoor_elements=True.
INDOOR_NETWORK_TYPES = ("walking", "all", "all_public")

# The ways a read with indoor_elements=True adds to the network as platforms.
PLATFORM_FILTER = {
    "railway": ["platform"],
    "public_transport": ["platform"],
    "highway": ["platform"],
}

# The areas a read with indoor_elements=True crosses, each on its own level: the platform
# areas and the indoor corridors and areas of Simple Indoor Tagging.
INDOOR_AREA_FILTER = {"indoor": ["corridor", "area"], **PLATFORM_FILTER}

# The ``vertical`` values of the edges that move, in the direction ``conveying`` gives.
MOVING = ("escalator", "moving_walkway")

# The nodes through which an indoor area is also entered: its doors and entrances.
DOOR_FILTER = {"door": [True], "entrance": [True]}


def check_indoor_read(network_type, custom_filter, nodes):
    """Raise a ValueError when a network read cannot take ``indoor_elements=True``: a
    network type other than :data:`INDOOR_NETWORK_TYPES` without a ``custom_filter``, or
    a read without ``nodes``."""
    if custom_filter is None and network_type not in INDOOR_NETWORK_TYPES:
        raise ValueError(
            "indoor_elements=True works with the network types %s and with a "
            "custom_filter, not with %r."
            % (", ".join(repr(t) for t in INDOOR_NETWORK_TYPES), network_type)
        )
    if not nodes:
        raise ValueError(
            "indoor_elements=True needs nodes=True: the levels are those of the edges' "
            "two nodes."
        )


def _is_moving(tags):
    """Whether an edge with the OSM tags ``tags`` moves: ``conveying`` other than no."""
    return tags.get("conveying") not in (None, "no")


def is_vertical(tags):
    """Whether an edge with the OSM tags ``tags`` is stairs, an escalator or a moving
    walkway."""
    return tags.get("highway") == "steps" or _is_moving(tags)


def edge_levels(edges, tags, node_levels=None):
    """``(u_level, v_level)``: float arrays of the level at each end of ``edges`` (a
    frame with ``u`` and ``v``, and ``id``, ``length`` and ``walking_area`` when known),
    whose OSM tags are ``tags`` (a list of dicts in their order), as the module docstring
    describes; NaN where the level is not known. ``node_levels`` holds the nodes' own
    levels (see :func:`node_levels`)."""
    n = len(edges)
    low, high = np.full(n, np.nan), np.full(n, np.nan)
    vertical = np.zeros(n, dtype=bool)
    for i, row_tags in enumerate(tags):
        ranges = level_ranges(row_tags.get("level"))
        if ranges:
            low[i] = min(r[0] for r in ranges)
            high[i] = max(r[1] for r in ranges)
        vertical[i] = is_vertical(row_tags)
    rise = np.array([_rise(row_tags) for row_tags in tags], dtype=np.int8)
    single = low == high
    listed = high > low
    u_level = np.where(single, low, np.nan)
    v_level = u_level.copy()
    u, v = edges["u"].to_numpy(), edges["v"].to_numpy()
    flat = single & ~vertical
    ends = pd.DataFrame(
        {"node": np.concatenate([u[flat], v[flat]]), "level": np.tile(low[flat], 2)}
    )
    levels_at = ends.groupby("node")["level"].unique().to_dict()
    node_levels = node_levels or {}
    lengths = float_column(edges, "length")
    for rows, nodes in _vertical_ways(edges, vertical):
        first = rows[0]
        if listed[first]:
            points = _range_points(
                nodes, low[first], high[first], levels_at, node_levels, rise[first]
            )
        else:
            points = _joined_points(nodes, levels_at)
        if points is None:
            continue
        levels = _levels_along(nodes, lengths[rows], points)
        u_level[rows] = [levels[node] for node in u[rows]]
        v_level[rows] = [levels[node] for node in v[rows]]
    return u_level, v_level


def _range_stair_rows(edges, ends):
    """The rows of ``edges`` (a frame with ``id``) that are stairs, escalators or moving
    walkways with a level range and end at one of the nodes ``ends``."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    u, v = edges["u"].to_numpy(), edges["v"].to_numpy()
    at_end = np.flatnonzero(np.isin(u, ends) | np.isin(v, ends))
    if "id" not in edges.columns or not len(at_end):
        return []
    stairs = []
    for row, tags in zip(
        at_end, _frame_tags(edges.iloc[at_end], _frame_tag_columns(edges))
    ):
        ranges = level_ranges(tags.get("level"))
        if (
            is_vertical(tags)
            and ranges
            and len({r for pair in ranges for r in pair}) > 1
        ):
            stairs.append(row)
    return stairs


def stair_end_levels(edges, ends, own_levels=None):
    """``{node: level}`` for the nodes ``ends`` at an end of the way of stairs,
    escalators or moving walkways with a level range in ``edges`` (see
    :func:`stair_way_ends`), where :func:`edge_levels` finds the level
    of that end from the rest of the way: from the walkways at its other end (or the
    nodes' ``own_levels``, see :func:`node_levels`), or from its incline."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    wanted = stair_way_ends(edges, ends)
    if not wanted:
        return {}
    u, v = edges["u"].to_numpy(), edges["v"].to_numpy()
    ids = edges["id"].to_numpy()
    ways = np.isin(ids, ids[_range_stair_rows(edges, ends)])
    way_nodes = np.concatenate([u[ways], v[ways]])
    near = np.flatnonzero(np.isin(u, way_nodes) | np.isin(v, way_nodes))
    subset = edges.iloc[near]
    u_level, v_level = edge_levels(
        subset, _frame_tags(subset, _frame_tag_columns(edges)), own_levels
    )
    found = {}
    for a, b, level_a, level_b, way in zip(
        u[near], v[near], u_level, v_level, ways[near]
    ):
        for node, level in ((a, level_a), (b, level_b)):
            if way and node in wanted and not np.isnan(level):
                found[node] = float(level)
    return found


def stair_way_ends(edges, ends):
    """``{node: (other end, starts, inner)}`` for the nodes ``ends`` that end stairs,
    escalators or moving walkways with a level range in ``edges``: the node at the other
    end of the way (see :func:`_vertical_ways`), whether the way starts at the node, in
    its drawn direction, and the number of nodes between its ends."""
    stairs = _range_stair_rows(edges, ends)
    if not stairs:
        return {}
    ids = edges["id"].to_numpy()
    rows = np.flatnonzero(np.isin(ids, ids[stairs]))
    vertical = np.zeros(len(edges), dtype=bool)
    vertical[rows] = True
    wanted = set(np.asarray(ends).tolist())
    found = {}
    for _, nodes in _vertical_ways(edges, vertical):
        first, last, inner = nodes[0], nodes[-1], len(nodes) - 2
        if first in wanted:
            found[first] = (last, True, inner)
        if last in wanted:
            found[last] = (first, False, inner)
    return found


def _rise(tags):
    """1 when a way with the OSM tags ``tags`` rises along its drawn direction
    (``incline=up``), -1 when it falls (``incline=down``), else 0."""
    return {"up": 1, "down": -1}.get(tags.get("incline"), 0)


def float_column(frame, col):
    """The column ``col`` of ``frame`` as an array of floats, NaN where a value is
    missing or not a number (all NaN without the column)."""
    if col not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)


def _vertical_ways(edges, vertical):
    """The stairs, escalators and moving walkways among ``edges`` (the rows ``vertical``
    marks) as ``(rows, nodes)``: the edges of a way in their order along it and its nodes
    in that order, when the edges of the way form one line (see :func:`_line_order`); else
    each edge on its own, as are the lines across areas (``walking_area`` True)."""
    u, v = edges["u"].to_numpy(), edges["v"].to_numpy()
    alone = np.full(len(edges), "id" not in edges.columns)
    if "walking_area" in edges.columns:
        alone |= edges["walking_area"].fillna(False).to_numpy(dtype=bool)
    ways, found = {}, []
    for i in np.flatnonzero(vertical):
        if alone[i]:
            found.append(([i], [u[i], v[i]]))
        else:
            ways.setdefault(edges["id"].iat[i], []).append(i)
    for rows in ways.values():
        line = _line_order(rows, u, v)
        found += [line] if line else [([i], [u[i], v[i]]) for i in rows]
    return found


def _line_order(rows, u, v):
    """``(rows, nodes)``: the edges ``rows`` (with the end nodes ``u`` and ``v``) in their
    order along the line they form, and its nodes in that order, from the end where an
    edge starts (its ``u``), so along the drawn direction of the way; None when they do
    not form one line (a branch, a ring or a gap)."""
    at = {}
    for i in rows:
        at.setdefault(u[i], []).append(i)
        at.setdefault(v[i], []).append(i)
    ends = [node for node, edges in at.items() if len(edges) == 1]
    if len(ends) != 2 or len(at) != len(rows) + 1:
        return None
    ends.sort(key=lambda node: u[at[node][0]] != node)
    order, nodes = [], [ends[0]]
    while len(order) < len(rows):
        step = [i for i in at[nodes[-1]] if i not in order]
        if not step:
            return None
        order.append(step[0])
        nodes.append(v[step[0]] if u[step[0]] == nodes[-1] else u[step[0]])
    return order, nodes


def _range_points(nodes, low, high, levels_at, node_levels, rise=0):
    """The known levels along a way with a level range from ``low`` to ``high`` through
    ``nodes`` (in the way's drawn direction), as ``{position in nodes: level}``. A node
    takes the levels in the range of the walkways there (``levels_at``) or, without one,
    its own level (``node_levels``). The two ends are chosen as :func:`_span_ends`
    chooses them; an inner node keeps a single level. An end without a level takes the
    other end of the range from the nearest inner level, and when the two ends have the
    same level and no inner node has one, the inner nodes take the other end of the
    range. An end still without a level takes the low end of the range at the start of a
    way that rises (``rise`` 1, see :func:`_rise`) and the high end at its end, the other
    way round when it falls (-1). None when the ends are not known."""

    def at(node):
        joined = _span_levels(levels_at.get(node, ()), low, high)
        return joined or _span_levels(node_levels.get(node, ()), low, high)

    inner = {
        k: found[0] for k in range(1, len(nodes) - 1) if len(found := at(nodes[k])) == 1
    }
    first, last = at(nodes[0]), at(nodes[-1])
    ends = _span_ends(first, last, low, high)
    if first and last and ends is None:
        other = _other_end(first[0], low, high)
        if not inner and other is not None:
            inner = {k: other for k in range(1, len(nodes) - 1)}
        ends = (first[0], last[0]) if inner else None
    elif inner and not (first and last):
        ends = (
            first[0] if first else _other_end(inner[min(inner)], low, high),
            last[0] if last else _other_end(inner[max(inner)], low, high),
        )
    if (ends is None or None in ends) and rise and not (first and last):
        bottom, top = (low, high) if rise > 0 else (high, low)
        ends = (first[0] if first else bottom, last[0] if last else top)
    if ends is None or None in ends or (ends[0] == ends[1] and not inner):
        return None
    return {0: ends[0], len(nodes) - 1: ends[1], **inner}


def _joined_points(nodes, levels_at):
    """The known levels along a way without a level range through ``nodes``, as
    ``{position in nodes: level}``: each node where the walkways (``levels_at``) have one
    level, when they give two different levels or more; None otherwise."""
    points = {
        k: found[0]
        for k, node in enumerate(nodes)
        if len(found := levels_at.get(node, ())) == 1
    }
    return points if len(set(points.values())) > 1 else None


def _other_end(level, low, high):
    """The other end of the range from ``low`` to ``high`` from ``level``, when ``level``
    is one of its ends; else None."""
    return high if level == low else low if level == high else None


def _levels_along(nodes, lengths, points):
    """``{node: level}`` for ``nodes``, joined by edges of ``lengths`` (each counted as 1
    when one of them is not a finite length above 0): the levels of ``points``
    (``{position in nodes: level}``), levels in proportion to the distance along the
    edges between them, rounded to 0.001, and beyond the outermost points their
    levels."""
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        lengths = np.ones(len(lengths))
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    known = sorted(points)
    levels = np.interp(distance, distance[known], [points[k] for k in known])
    return dict(zip(nodes, np.round(levels, 3)))


def _span_levels(levels, low, high):
    """The ``levels`` of the walkways at an end of stairs or an escalator spanning
    ``low`` to ``high`` that lie in the span, the span's own ends first."""
    inside = [level for level in levels if low <= level <= high]
    return [level for level in inside if level in (low, high)] + [
        level for level in inside if level not in (low, high)
    ]


def _span_ends(at_u, at_v, low, high):
    """``(u level, v level)`` of stairs or an escalator spanning ``low`` to ``high``,
    from the span levels ``at_u`` and ``at_v`` of the walkways at its two ends: two
    different ones when they allow it; with one end unknown, the other end of the span
    when the known level is a span end; else None."""
    for a in at_u:
        for b in at_v:
            if a != b:
                return a, b
    if at_u and not at_v and at_u[0] in (low, high):
        return at_u[0], _other_end(at_u[0], low, high)
    if at_v and not at_u and at_v[0] in (low, high):
        return _other_end(at_v[0], low, high), at_v[0]
    return None


def check_escalator_slope(escalator_slope):
    """Raise a ValueError unless ``escalator_slope`` is None or an angle in degrees of
    at least 0 and below 90; True and False are refused, which Python counts as 1 and
    0."""
    if escalator_slope is None:
        return
    if (
        isinstance(escalator_slope, bool)
        or not isinstance(escalator_slope, numbers.Real)
        or not 0 <= escalator_slope < 90
    ):
        raise ValueError(
            "escalator_slope should be an angle in degrees of at least 0 and below 90, "
            "or None to keep the length of escalators on the map, not %r."
            % (escalator_slope,)
        )


def edge_vertical(tags):
    """The vertical movement of edges with the OSM tags ``tags`` (a list of dicts):
    ``"escalator"`` for steps that move (``conveying`` other than no), ``"stairs"`` for
    other steps, ``"moving_walkway"`` for other ways that move, None for the rest."""
    kinds = []
    for row_tags in tags:
        kind = None
        if row_tags.get("highway") == "steps":
            kind = "escalator" if _is_moving(row_tags) else "stairs"
        elif _is_moving(row_tags):
            kind = "moving_walkway"
        kinds.append(kind)
    return kinds


def moving_one_way(edges):
    """``(forward, backward)`` by position in ``edges``: whether each edge is an escalator
    or a moving walkway (:data:`MOVING`) that moves only from its ``u`` to its ``v`` node
    (``conveying=forward``, the direction the way is drawn in) or only from ``v`` to ``u``
    (``conveying=backward``); None when ``edges`` have neither."""
    if "vertical" not in edges.columns:
        return None
    moving = edges["vertical"].isin(MOVING).to_numpy()
    if not moving.any():
        return None
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    rows = np.flatnonzero(moving)
    tags = _frame_tags(edges.iloc[rows], _frame_tag_columns(edges))
    conveying = np.full(len(edges), None, dtype=object)
    conveying[rows] = [row_tags.get("conveying") for row_tags in tags]
    return conveying == "forward", conveying == "backward"


def with_indoor_columns(edges, escalator_slope, nodes=None):
    """``edges`` (the frame of a network read with ``nodes=True``) with the columns
    ``u_level`` and ``v_level`` (see :func:`edge_levels`, with the own levels of
    ``nodes``, the read's nodes) and ``vertical`` (see
    :func:`edge_vertical`; none on the lines across areas, ``walking_area`` True, which
    do not move whatever the area's tags say). The ``length`` of an escalator is its
    length on the map divided by the cosine of ``escalator_slope`` (degrees), its length
    along a slope of that angle, in millimetres; None keeps the length on the map."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    tags = _frame_tags(edges, _frame_tag_columns(edges))
    own = None if nodes is None else node_levels(nodes)
    u_level, v_level = edge_levels(edges, tags, own)
    vertical = edge_vertical(tags)
    if "walking_area" in edges.columns:
        across = edges["walking_area"].fillna(False).to_numpy(dtype=bool)
        vertical = [None if line else kind for kind, line in zip(vertical, across)]
    edges = edges.assign(u_level=u_level, v_level=v_level, vertical=vertical)
    if escalator_slope is not None:
        escalator = edges["vertical"] == "escalator"
        along = edges.loc[escalator, "length"] / math.cos(math.radians(escalator_slope))
        edges.loc[escalator, "length"] = along.round(3)
    return edges


def with_platform_filter(read_filter):
    """The keep filter ``read_filter`` (a dict of value lists, or of single values) that
    also keeps the platforms of :data:`PLATFORM_FILTER`."""
    combined = {
        key: list(values) if isinstance(values, (list, tuple, set)) else [values]
        for key, values in read_filter.items()
    }
    for key, values in PLATFORM_FILTER.items():
        kept = combined.setdefault(key, [])
        if True not in kept:
            kept.extend(value for value in values if value not in kept)
    return combined


def with_platforms(nodes, edges, platform_nodes, platforms, walking_areas=False):
    """The ``nodes`` and ``edges`` of a network read with the ``platforms`` (the edges of
    a read of :data:`PLATFORM_FILTER`) whose ways are not edges yet (an edge across a
    walking area carries a relation's id, which does not count), after them, and the
    ``platform_nodes`` they add; the platforms alone when ``edges`` is None, with a
    ``walking_area`` column of False with ``walking_areas``."""
    if platforms is None or platforms.empty:
        return nodes, edges
    if edges is None:
        if walking_areas:
            platforms = platforms.assign(walking_area=False)
        return platform_nodes, platforms
    ways = edges["id"]
    if "osm_type" in edges.columns:
        ways = ways[edges["osm_type"] == "way"]
    new = platforms[~platforms["id"].isin(ways)]
    if "walking_area" in edges.columns:
        new = new.assign(walking_area=False)
    attrs = dict(edges.attrs)
    edges = GeoDataFrame(
        pd.concat([edges, new], ignore_index=True), geometry="geometry", crs=edges.crs
    )
    edges.attrs = attrs
    used = pd.unique(np.concatenate([new["u"].to_numpy(), new["v"].to_numpy()]))
    added = platform_nodes[
        platform_nodes["id"].isin(used) & ~platform_nodes["id"].isin(nodes["id"])
    ]
    nodes = GeoDataFrame(
        pd.concat([nodes, added], ignore_index=True), geometry="geometry", crs=nodes.crs
    )
    return nodes, edges


def indoor_areas(candidates, walking=None):
    """The polygons and multipolygons among ``candidates`` (a read of
    :data:`INDOOR_AREA_FILTER`) after the ``walking`` areas of the read (see
    :func:`pyrosm.walking_areas.select_areas`), each area once and with the tag columns
    of both (see :func:`_with_tag_columns`); None when there are none."""
    frames = [] if walking is None else [walking]
    if candidates is not None:
        polygonal = candidates.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        areas = candidates[polygonal]
        if walking is not None:
            done = set(zip(walking["osm_type"], walking["id"]))
            areas = areas[
                [key not in done for key in zip(areas["osm_type"], areas["id"])]
            ]
        frames.append(areas)
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return None
    columns = [col for frame in frames for col in frame.columns]
    frames = [_with_tag_columns(frame, columns) for frame in frames]
    areas = pd.concat(frames, ignore_index=True)
    return GeoDataFrame(areas, geometry="geometry", crs=frames[0].crs)


def _with_tag_columns(frame, columns):
    """``frame`` with those of ``columns`` that it lacks, filled from its tags (see
    :func:`pyrosm.pbf_writer._frame_tags`), so that a tag of the JSON ``tags`` is kept
    where another frame has a column for it."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags, _tag_key

    missing = list(dict.fromkeys(col for col in columns if col not in frame.columns))
    if not missing:
        return frame
    tags = _frame_tags(frame, _frame_tag_columns(frame))
    return frame.assign(
        **{col: [row.get(_tag_key(col)) for row in tags] for col in missing}
    )


def with_indoor_crossings(nodes, edges, areas, doors):
    """The ``nodes`` and ``edges`` of a network read with the edges across the indoor and
    walking ``areas`` (see :func:`pyrosm.walking_areas.crossing_rows`), marked True in
    the column ``walking_area`` (False on the other edges), entered through the
    network's nodes on the areas' outlines and, for the areas of
    :data:`INDOOR_AREA_FILTER`, through the ``doors`` (a frame of nodes with ``id``,
    ``lon``, ``lat`` and ``tags``) on their outlines, without those between two nodes an
    edge already joins; the doors that the new edges use join the nodes."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags
    from pyrosm.walking_areas import crossing_rows

    if "walking_area" not in edges.columns:
        edges = edges.assign(walking_area=False)
    if areas is None:
        return nodes, edges
    columns = [col for col in ("id", "lon", "lat", "tags") if col in nodes.columns]
    entrances = nodes[columns]
    if doors is not None:
        new_doors = doors[~doors["id"].isin(nodes["id"])]
        new_doors = new_doors[[col for col in columns if col in new_doors.columns]]
        entrances = pd.concat([entrances, new_doors], ignore_index=True)
    plain = edges[~edges["walking_area"]]
    door_areas = [
        any(tags.get(key) in values for key, values in INDOOR_AREA_FILTER.items())
        for tags in _frame_tags(areas, _frame_tag_columns(areas))
    ]
    rows = crossing_rows(entrances, plain, areas, door_areas)
    if rows is not None:
        # A line along the outline of an area that is an edge itself (a platform's ring)
        joined = set(zip(edges["u"], edges["v"])) | set(zip(edges["v"], edges["u"]))
        rows = rows[[pair not in joined for pair in zip(rows["u"], rows["v"])]]
    if rows is None or rows.empty:
        return nodes, edges
    rows = rows.assign(walking_area=True)
    attrs = dict(edges.attrs)
    edges = GeoDataFrame(
        pd.concat([edges, rows], ignore_index=True), geometry="geometry", crs=edges.crs
    )
    edges.attrs = attrs
    if doors is not None:
        used = set(rows["u"]) | set(rows["v"])
        added = doors[doors["id"].isin(used) & ~doors["id"].isin(nodes["id"])]
        if len(added):
            added = added[[col for col in nodes.columns if col in added.columns]]
            nodes = GeoDataFrame(
                pd.concat([nodes, added], ignore_index=True),
                geometry="geometry",
                crs=nodes.crs,
            )
    return nodes, edges


# The largest lift node id whose copies' ids (see lift_copy_id) fit in a signed 64-bit
# integer.
_MAX_LIFT_ID = (2**63 - 1) // 1000 - 1


# The element metadata a lift edge takes from its lift node.
LIFT_METADATA = ("version", "timestamp", "changeset", "visible")


def lift_copy_id(lift, rank):
    """The id of the copy of the lift node ``lift`` at its ``rank``-th level from the
    lowest (1 or more): negative and the same on every read."""
    return -(int(lift) * 1000 + rank)


def _tags_by_node(nodes):
    """``(id, tags)`` of each of ``nodes`` with tags, from their ``tags`` column (dicts,
    or JSON text)."""
    if "tags" not in nodes.columns:
        return
    for node, tags in zip(nodes["id"], nodes["tags"]):
        if isinstance(tags, str):
            tags = json.loads(tags)
        if isinstance(tags, dict):
            yield int(node), tags


def _lift_ids(nodes):
    """The ids of the lift nodes (``highway=elevator``) among ``nodes``."""
    return {
        node for node, tags in _tags_by_node(nodes) if tags.get("highway") == "elevator"
    }


def node_levels(nodes):
    """``{id: (level,)}``: the level of each of ``nodes`` whose ``level`` tag names one
    level."""
    found = {}
    for node, tags in _tags_by_node(nodes):
        ranges = level_ranges(tags.get("level"))
        if ranges and len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
            found[node] = (ranges[0][0],)
    return found


def split_lifts(nodes, edges):
    """``nodes`` and ``edges`` (with ``u_level`` and ``v_level``) with each lift node
    where edges on two or more levels meet split into one node per level: the lowest
    keeps the node's id, the others get :func:`lift_copy_id`, and each edge ends at the
    copy of its level there (an end without a level stays at the lowest). Lift edges of
    length 0 join the copies of neighbouring levels, with the lift node's id and metadata
    (those of :data:`LIFT_METADATA` that ``edges`` has), ``osm_type`` "node",
    ``highway`` "elevator" and ``vertical`` "lift". A copy id that another node has
    already raises a ValueError; a lift node with a negative id (one pyrosm wrote) is
    not split."""
    lifts = _lift_ids(nodes)
    if not lifts:
        return nodes, edges
    u, v = edges["u"].to_numpy().copy(), edges["v"].to_numpy().copy()
    u_level, v_level = edges["u_level"].to_numpy(), edges["v_level"].to_numpy()
    levels = {}
    for ends, end_levels in ((u, u_level), (v, v_level)):
        for i in np.flatnonzero(np.isin(ends, list(lifts)) & ~np.isnan(end_levels)):
            levels.setdefault(int(ends[i]), set()).add(float(end_levels[i]))
    at = nodes.set_index("id")
    metadata = [
        col for col in LIFT_METADATA if col in edges.columns and col in at.columns
    ]
    taken = set(nodes["id"].tolist())
    copies, lift_rows = [], []
    for lift, found in sorted(levels.items()):
        if len(found) < 2 or not 0 < lift <= _MAX_LIFT_ID:
            continue
        ordered = sorted(found)
        ids = {level: lift_copy_id(lift, rank) for rank, level in enumerate(ordered)}
        ids[ordered[0]] = lift
        clash = [node for node in list(ids.values())[1:] if node in taken]
        if clash or len(ordered) >= 1000:
            raise ValueError(
                "Lift node %d cannot be split per level: the ids of its copies, %s, "
                "are taken or too many." % (lift, sorted(ids.values())[:3])
            )
        taken.update(ids.values())
        for ends, end_levels in ((u, u_level), (v, v_level)):
            for i in np.flatnonzero((ends == lift) & ~np.isnan(end_levels)):
                ends[i] = ids[float(end_levels[i])]
        row = at.loc[lift]
        point = row.geometry
        for level in ordered[1:]:
            copies.append(dict(row, id=ids[level]))
        for low, high in zip(ordered, ordered[1:]):
            lift_rows.append(
                {
                    "id": lift,
                    "osm_type": "node",
                    "highway": "elevator",
                    "u": ids[low],
                    "v": ids[high],
                    "length": 0.0,
                    "u_level": low,
                    "v_level": high,
                    "vertical": "lift",
                    "geometry": LineString([point, point]),
                    **{col: row[col] for col in metadata},
                }
            )
    if not lift_rows:
        return nodes, edges
    edges = edges.assign(u=u, v=v)
    if "walking_area" in edges.columns:
        for row in lift_rows:
            row["walking_area"] = False
    attrs = dict(edges.attrs)
    lift_edges = pd.DataFrame(lift_rows).astype(
        {col: edges[col].dtype for col in metadata}
    )
    edges = GeoDataFrame(
        pd.concat([edges, lift_edges], ignore_index=True),
        geometry="geometry",
        crs=edges.crs,
    )
    edges.attrs = attrs
    nodes = GeoDataFrame(
        pd.concat([nodes, pd.DataFrame(copies)], ignore_index=True),
        geometry="geometry",
        crs=nodes.crs,
    )
    return nodes, edges
