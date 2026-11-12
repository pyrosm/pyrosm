"""The indoor elements of a network read with ``indoor_elements=True``.

The platforms (:data:`PLATFORM_FILTER`) are edges of the network, also those that its
type leaves out, such as ``highway=platform`` in a walking network. The platform areas
and the indoor corridors and areas, also stairs and escalators mapped as rooms
(:data:`INDOOR_AREA_FILTER`, :func:`is_indoor_area`), are crossed as walking areas are,
each on its own level and together with the walking areas of a read with
``walking_areas=True``, and they are entered also through the doors on their outlines,
which connect areas that share them; the walking areas are not. Corridors, indoor areas
and platform areas that share outline nodes on a common level, which Simple Indoor
Tagging maps without walls, are entered through those nodes too, also when one is mapped
as a multipolygon; where the spaces at a node have no level in common, such as stairs on
levels 1 and 2 drawn with the nodes of both floors, each level's spaces join through a
copy of the node of their own (see :func:`border_entrances`). The end of stairs or an
escalator that joins nothing else also enters the platform or indoor area on its level
that it lies outside of within
:data:`pyrosm.walking_areas.STAIR_END_REACH` metres. The edges get the level at each of
their two ends, ``u_level`` and ``v_level``, from the OSM ``level`` tags (see
:func:`pyrosm.levels.level_ranges`). An edge with one level has it at both ends. A level
list or range on stairs, an escalator or a moving walkway runs between the levels of the
walkways at the two ends of its way or, at an end where no walkway has a level in the
range, the end node's own ``level``, and where neither gives an end, the way's
``incline`` (``up`` rises along the way's drawn direction, ``down`` falls); on another
walkway, such as a corridor tagged ``level=0;1``, the data does not say which part is on
which level, and both ends stay empty. A line across an area with a level list or range
takes at each end the level in the range of the other edges there or, without one, the
end node's own ``level``; an end with none or with several stays empty, and a line
between entrances on two levels has these two levels; such lines are kept only across
stairs or an escalator mapped as an area (see :func:`without_lines_between_levels`). A
way of several edges also keeps the level of a walkway that joins it in between (or of
an inner node's own ``level``), an end without a level then takes the other end of the
range from the nearest such level, a way whose two ends are on one level reaches the
other end of its range in between, and the other inner nodes get levels in proportion to
the distance along the way. Stairs, escalators and moving walkways without a level range
(no ``level`` tag or a single level) take the levels of the walkways that join their
way, when these give two different levels, and keep the outermost ones up to the way's
ends. A platform without a ``level`` tag, its way and the lines across its area, takes
the level that the other edges give at its nodes, when the platforms joined to it
through shared nodes meet exactly one level. An edge without a ``level`` tag otherwise
has no level. The column ``vertical`` marks escalators, moving walkways and stairs; the
``length`` of an escalator is measured along its slope (see
:func:`with_indoor_columns`), and in a graph an escalator or moving walkway runs only in
the direction it moves (see :func:`moving_one_way`). A lift node (``highway=elevator``)
where walkways on several levels meet is split into one node per level, joined by lift
edges (see :func:`split_lifts`), whose length is that of an escalator or stairs between
their levels plus 1 m (see :func:`with_climb_lengths`); a lift node whose own ``level``
tag names levels that its walkways do not reach also enters, from inside, the platform
and indoor areas of one level on those levels that it lies in, when it then reaches two
or more levels, unless it lies inside a lift area on one of its levels (see
:func:`pyrosm.walking_areas.crossing_rows`), and gets lift edges to them; a lift mapped
as an area (``highway=elevator`` on a closed way or multipolygon) is not a walkway
around the lift but joins the levels at which the network enters it in the same way, and
lift areas mapped one floor at a time make one lift together (see
:func:`join_lift_areas`). A door tagged ``repeat_on`` is a door on each of its levels, a
copy per level (see :func:`door_copies`).
"""

import json
import math
import numbers

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from rapidjson import dumps
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
# areas, the indoor corridors and areas of Simple Indoor Tagging, the rooms tagged as
# corridors, entrance rooms, lobbies, halls and waiting rooms, and stairs and escalators
# mapped as rooms (see is_indoor_area, which the elements read with this filter pass).
INDOOR_AREA_FILTER = {
    "indoor": ["corridor", "area"],
    "room": ["corridor", "entrance", "lobby", "hall", "waiting", "stairs"],
    "stairs": [True],
    **PLATFORM_FILTER,
}

# The lifts mapped as areas, closed ways and multipolygons, that a read with
# indoor_elements=True joins as lifts instead of crossing them (see join_lift_areas); an
# open way with this tag is an inclined lift, a walkway. Read on their own: a lift is
# often a hole in the multipolygon of the corridor around it, and a read with that
# relation leaves its member ways out.
LIFT_AREA_FILTER = {"highway": ["elevator"]}

# The walls mapped as ways (Simple Indoor Tagging): where one runs along the side two open
# indoor spaces share at a node, they do not join there (see border_entrances).
WALL_FILTER = {"indoor": ["wall"]}

# The ``vertical`` values of the edges that move, in the direction ``conveying`` gives.
MOVING = ("escalator", "moving_walkway")

# The nodes through which an indoor area is also entered: its doors and entrances, and
# the lift nodes, which enter it on the levels of their own level tag that their ways do
# not reach (see pyrosm.walking_areas.crossing_rows).
ENTRANCE_FILTER = {"door": [True], "entrance": [True], "highway": ["elevator"]}


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


def is_indoor_area(tags):
    """Whether an area with the OSM tags ``tags`` is one that indoor reads cross: a
    platform, an ``indoor=corridor`` or ``indoor=area``, a room tagged as a walkway
    (``room=corridor``, ``entrance``, ``lobby``, ``hall``, ``waiting`` or ``stairs``), or
    stairs or an escalator mapped as a room (``indoor=room`` with ``stairs`` other than
    no), as Simple Indoor Tagging maps staircases."""
    return (
        _is_platform(tags)
        or tags.get("indoor") in ("corridor", "area")
        or tags.get("room") in INDOOR_AREA_FILTER["room"]
        or (tags.get("indoor") == "room" and tags.get("stairs") not in (None, "no"))
    )


def _is_stair_area(tags):
    """Whether an area with the OSM tags ``tags`` is stairs or an escalator mapped as an
    area: stairs or an escalator by :func:`is_vertical`, ``stairs`` other than no, or
    ``room=stairs``."""
    return (
        is_vertical(tags)
        or tags.get("stairs") not in (None, "no")
        or tags.get("room") == "stairs"
    )


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
    across = np.flatnonzero(listed & ~vertical & _area_lines(edges))
    if len(across):
        known = pd.DataFrame({"u": u, "v": v, "u_level": u_level, "v_level": v_level})
        end_levels_at = _levels_at_nodes(known, np.concatenate([u[across], v[across]]))
        for ends, end_levels in ((u, u_level), (v, v_level)):
            for i in across:
                found = _levels_in_range(
                    ends[i], low[i], high[i], end_levels_at, node_levels
                )
                if len(found) == 1:
                    end_levels[i] = found[0]
    platforms = np.isnan(low) & np.array([_is_platform(t) for t in tags], dtype=bool)
    if platforms.any():
        _fill_platform_levels(u, v, u_level, v_level, platforms)
    return u_level, v_level


def _is_platform(tags):
    """Whether an edge with the OSM tags ``tags`` is a platform (:data:`PLATFORM_FILTER`)."""
    return any(tags.get(key) in values for key, values in PLATFORM_FILTER.items())


def _fill_platform_levels(u, v, u_level, v_level, platforms):
    """Set ``u_level`` and ``v_level`` (arrays of the edges with the end nodes ``u`` and
    ``v``) in place on the edges that ``platforms`` marks, platforms without a level: the
    platforms joined to each other through shared nodes take the level that the other
    edges give at their nodes, when these give exactly one level."""
    rows = np.flatnonzero(platforms)
    parent = {node: node for node in np.concatenate([u[rows], v[rows]]).tolist()}
    for i in rows:
        a, b = _root(parent, u[i]), _root(parent, v[i])
        if a != b:
            parent[a] = b
    found = {}
    for ends, end_levels in ((u, u_level), (v, v_level)):
        for i in np.flatnonzero(~platforms & ~np.isnan(end_levels)):
            if ends[i] in parent:
                found.setdefault(_root(parent, ends[i]), set()).add(
                    float(end_levels[i])
                )
    for i in rows:
        levels = found.get(_root(parent, u[i]), ())
        if len(levels) == 1:
            u_level[i] = v_level[i] = next(iter(levels))


def _root(parent, node):
    """The root of ``node`` in the union-find forest ``parent`` (a dict or list of each
    node's parent, a root its own), halving the path to it on the way."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


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
    alone = np.full(len(edges), "id" not in edges.columns) | _area_lines(edges)
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


def _area_lines(edges):
    """Whether each of ``edges`` is a line across an area (``walking_area`` True); all
    False without that column."""
    if "walking_area" not in edges.columns:
        return np.zeros(len(edges), dtype=bool)
    return edges["walking_area"].fillna(False).to_numpy(dtype=bool)


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
        return _levels_in_range(node, low, high, levels_at, node_levels)

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


def _levels_in_range(node, low, high, levels_at, node_levels):
    """The levels from ``low`` to ``high`` at ``node`` (see :func:`_span_levels`): those
    of the edges there (``levels_at``) or, without one, the node's own level
    (``node_levels``)."""
    joined = _span_levels(levels_at.get(node, ()), low, high)
    return joined or _span_levels(node_levels.get(node, ()), low, high)


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


def with_indoor_columns(edges, escalator_slope, nodes=None, levels=None):
    """``edges`` (the frame of a network read with ``nodes=True``) with the columns
    ``u_level`` and ``v_level`` (see :func:`edge_levels`, with the own levels of
    ``nodes``, the read's nodes, and for the nodes without one the ``levels``, ``{id:
    (level,)}``, such as those of border entrances) and ``vertical`` (see
    :func:`edge_vertical`; none on the lines across areas, ``walking_area`` True, which
    do not move whatever the area's tags say). The ``length`` of an escalator is its
    length on the map divided by the cosine of ``escalator_slope`` (degrees), its length
    along a slope of that angle, in millimetres; None keeps the length on the map."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    tags = _frame_tags(edges, _frame_tag_columns(edges))
    own = {**(levels or {}), **({} if nodes is None else node_levels(nodes))}
    u_level, v_level = edge_levels(edges, tags, own)
    vertical = edge_vertical(tags)
    vertical = [
        None if line else kind for kind, line in zip(vertical, _area_lines(edges))
    ]
    edges = edges.assign(u_level=u_level, v_level=v_level, vertical=vertical)
    if escalator_slope is not None:
        escalator = edges["vertical"] == "escalator"
        along = edges.loc[escalator, "length"] / math.cos(math.radians(escalator_slope))
        edges.loc[escalator, "length"] = along.round(3)
    return edges


def without_lines_between_levels(nodes, edges):
    """``nodes`` and ``edges`` (with ``u_level`` and ``v_level``) without the lines across
    areas (``walking_area`` True) whose two ends are on two different levels, except
    across stairs or an escalator mapped as an area (see :func:`_is_stair_area`), and
    without the doors that only those lines used."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    u_level = edges["u_level"].to_numpy(dtype=float)
    v_level = edges["v_level"].to_numpy(dtype=float)
    rows = np.flatnonzero(
        _area_lines(edges)
        & ~np.isnan(u_level)
        & ~np.isnan(v_level)
        & (u_level != v_level)
    )
    tags = _frame_tags(edges.iloc[rows], _frame_tag_columns(edges))
    rows = rows[[not _is_stair_area(row_tags) for row_tags in tags]]
    if not len(rows):
        return nodes, edges
    drop = np.zeros(len(edges), dtype=bool)
    drop[rows] = True
    kept = edges[~drop].reset_index(drop=True)
    kept.attrs = dict(edges.attrs)
    used = set(kept["u"]) | set(kept["v"])
    doors = set(edges["u"].iloc[rows]) | set(edges["v"].iloc[rows])
    left = nodes[~nodes["id"].isin(doors - used)].reset_index(drop=True)
    left.attrs = dict(nodes.attrs)
    return left, kept


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


def indoor_areas(candidates, walking=None, lifts=None):
    """The areas of a read with ``indoor_elements=True`` as ``(areas, lifts)``: the
    polygons and multipolygons among ``candidates`` (a read of
    :data:`INDOOR_AREA_FILTER`) that indoor reads cross (see :func:`is_indoor_area`)
    after the ``walking`` areas of the read (see
    :func:`pyrosm.walking_areas.select_areas`), each area once and with the tag columns
    of both (see :func:`_with_tag_columns`) and the ``nodes`` lists of ``candidates``
    when it has them, without the lift areas
    (``highway=elevator``), which are not crossed; and the polygons and multipolygons
    among ``lifts`` (a read of :data:`LIFT_AREA_FILTER`, see :func:`join_lift_areas`).
    None for either when there are none."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    if lifts is not None:
        lifts = lifts[lifts.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        lifts = lifts.reset_index(drop=True) if len(lifts) else None
    frames = [] if walking is None else [walking]
    if candidates is not None:
        polygonal = candidates.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        crossed = [
            is_indoor_area(tags)
            for tags in _frame_tags(candidates, _frame_tag_columns(candidates))
        ]
        areas = candidates[polygonal.to_numpy() & np.array(crossed, dtype=bool)]
        if walking is not None:
            keys = list(zip(areas["osm_type"], areas["id"]))
            if "nodes" in areas.columns:
                rings = dict(zip(keys, areas["nodes"]))
                walking = walking.assign(
                    nodes=[
                        rings.get(key)
                        for key in zip(walking["osm_type"], walking["id"])
                    ]
                )
                frames = [walking]
            done = set(zip(walking["osm_type"], walking["id"]))
            areas = areas[[key not in done for key in keys]]
        frames.append(areas)
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return None, lifts
    columns = [col for frame in frames for col in frame.columns]
    frames = [_with_tag_columns(frame, columns) for frame in frames]
    areas = pd.concat(frames, ignore_index=True)
    areas = GeoDataFrame(areas, geometry="geometry", crs=frames[0].crs)
    lift = np.array(
        [
            tags.get("highway") == "elevator"
            for tags in _frame_tags(areas, _frame_tag_columns(areas))
        ],
        dtype=bool,
    )
    areas = areas[~lift]
    return (areas if len(areas) else None), lifts


def without_member_nodes(areas):
    """``(areas, nodes)``: ``areas`` without their ``member_nodes`` column (see
    :func:`pyrosm.engine.relation_geometry.member_nodes`) and the nodes it lists as a
    frame with ``id``, ``lon`` and ``lat``, None when it lists none."""
    if areas is None or "member_nodes" not in areas.columns:
        return areas, None
    rows = [
        row for listed in areas["member_nodes"].dropna() for row in json.loads(listed)
    ]
    nodes = pd.DataFrame(rows, columns=["id", "lon", "lat"]) if rows else None
    return areas.drop(columns="member_nodes"), nodes


# The indoor spaces that Simple Indoor Tagging maps without walls; neighbouring ones on a
# level join where their outlines share nodes (see border_entrances), as platform areas do.
OPEN_INDOOR = ("corridor", "area")


def border_entrances(areas, walls=None, anchors=()):
    """The nodes that the outlines of two open indoor spaces among ``areas`` share on a
    common level, as a frame with ``id``, ``node``, ``lon``, ``lat``, ``level`` (that
    level when there is one, else NaN), ``tags`` (JSON with that ``level``), ``spaces`` (the
    ``(osm_type, id)`` of those spaces, the only areas the node enters) and ``geometry``;
    None when there are none. Open spaces are the polygons and multipolygons among
    ``areas`` that are platforms or tagged ``indoor=corridor`` or ``indoor=area``, which
    have no walls: people pass between two of them where they meet. Spaces meet at the
    vertices their outlines and holes share, and the node there is the one whose place
    the reads tell (see :func:`_node_places`, with the ``anchors`` and ``walls``). Where
    the spaces at a vertex have no level in common, each group of them on a common
    level joins there (see :func:`_level_groups`): the lowest group through the node,
    each other group through a copy of it on its level (:func:`node_copy_id`), and the
    frame's ``node`` holds the node each row is or copies. An ``indoor=wall`` way among
    ``walls`` (with ``nodes`` lists) through the node keeps the spaces on its levels
    apart there (all of them when it has no ``level`` tag) where it runs along the
    sides of the outlines they share at the node, or anywhere through the node where
    they share no side (see :func:`_walls_keep_apart`); a wall that only ends at the
    node, beside an opening, does not. A vertex is left out when no two spaces at it
    share a level or when no single node has its place."""
    from pyrosm.new_elements import _coordinate_key
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags, _tag_str

    if areas is None:
        return None
    spaces = []
    for row, tags in zip(
        areas.itertuples(index=False), _frame_tags(areas, _frame_tag_columns(areas))
    ):
        open_space = tags.get("indoor") in OPEN_INDOOR or _is_platform(tags)
        if not open_space or tags.get("indoor") == "room":
            continue
        if row.geometry is None or row.geometry.geom_type not in (
            "Polygon",
            "MultiPolygon",
        ):
            continue
        ring = getattr(row, "nodes", None)
        spaces.append(
            (
                (row.osm_type, row.id),
                row.geometry,
                level_ranges(tags.get("level")) or [],
                ring if isinstance(ring, (list, np.ndarray)) else None,
            )
        )
    places, walled = _node_places(spaces, walls, anchors)
    node_at = {}
    for node, (x, y) in places.items():
        key = _coordinate_key(x, y)
        node_at[key] = None if node_at.get(key, node) != node else node
    meeting, sides = {}, {}
    for key, geometry, ranges, _ in spaces:
        for vertex, next_to in _outline_sides(geometry).items():
            meeting.setdefault(vertex, {})[key] = ranges
            sides.setdefault(vertex, {})[key] = next_to
    rows = []
    for vertex, by_space in meeting.items():
        node = node_at.get(vertex)
        if len(by_space) < 2 or node is None:
            continue
        x, y = places[node]
        groups = [
            (spaces_on, level)
            for spaces_on, level in _level_groups(by_space)
            if not _walls_keep_apart(
                walled.get(node, ()),
                [by_space[k] for k in spaces_on],
                (x, y),
                _shared_sides((x, y), [sides[vertex][k] for k in spaces_on]),
            )
        ]
        for rank, (spaces_on, level) in enumerate(groups):
            if rank and not (0 < node <= _MAX_LIFT_ID and rank < MAX_REPEATED_LEVELS):
                break
            rows.append(
                {
                    "id": node_copy_id(node, rank) if rank else node,
                    "node": node,
                    "lon": x,
                    "lat": y,
                    "level": level,
                    "spaces": spaces_on,
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["tags"] = [
        None if level != level else dumps({"level": _tag_str(level)})
        for level in frame["level"]
    ]
    return GeoDataFrame(
        frame,
        geometry=shapely.points(frame["lon"], frame["lat"]),
        crs=areas.crs,
    )


def without_clashing_copies(borders, nodes, doors):
    """The ``borders`` (see :func:`border_entrances`) without the copies of the nodes
    that are doors tagged ``repeat_on`` or lift nodes among ``doors`` and ``nodes``
    (frames with ``id`` and tags), whose copies per level :func:`door_copies` and
    :func:`split_lifts` make with the same ids, and without the copies whose id a node
    of ``nodes`` or ``doors``, or another border node, has already."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    if borders is None:
        return None
    doors_ids, split = set(), _lift_ids(nodes)
    if doors is not None:
        doors_ids = set(doors["id"])
        split |= {
            int(door)
            for door, tags in zip(
                doors["id"], _frame_tags(doors, _frame_tag_columns(doors))
            )
            if tags.get("highway") == "elevator" or tags.get("repeat_on")
        }
    copy = (borders["id"] != borders["node"]).to_numpy()
    taken = set(nodes["id"]) | doors_ids | set(borders["id"][~copy])
    clash = borders["node"].isin(split) | borders["id"].isin(taken)
    kept = borders[~copy | ~clash.to_numpy()]
    return kept if len(kept) else None


def _walls_keep_apart(walls, range_lists, at, shared):
    """Whether the ``indoor=wall`` ways ``walls`` through a node at ``at`` keep apart the
    spaces with the level ``range_lists`` that meet there and whose outlines share the
    sides from ``at`` towards the vertices ``shared`` (see :func:`_shared_sides`).
    ``walls`` holds ``(level ranges, vertices)`` per wall (see :func:`_node_places`):
    None as ranges for a wall without a ``level`` tag, which stands on every level, and
    as vertices when the wall's vertices next to the node are not known. A wall on a
    level that all the spaces cover keeps them apart when the spaces share no side
    (they touch only at the node) or its vertices are not known; otherwise the walls
    keep them apart when each shared side has a wall along it."""
    common = _common_ranges(range_lists)
    on_level = [
        next_to
        for ranges, next_to in walls
        if ranges is None
        or any(max(la, lb) <= min(ha, hb) for la, ha in ranges for lb, hb in common)
    ]
    if not on_level:
        return False
    if not shared or any(next_to is None for next_to in on_level):
        return True
    along = [xy for next_to in on_level for xy in next_to]
    return all(any(_same_direction(at, side, xy) for xy in along) for side in shared)


def _shared_sides(at, sides):
    """The vertices next to ``at`` that mark the sides two or more outlines share there:
    of ``sides``, the vertices next to ``at`` along each space's outline, those in whose
    direction (see :func:`_same_direction`) the outline of a later space goes too."""
    sides = list(sides)
    return [
        xy
        for n, mine in enumerate(sides)
        for xy in mine
        if any(
            _same_direction(at, xy, other)
            for theirs in sides[n + 1 :]
            for other in theirs
        )
    ]


def _same_direction(at, a, b):
    """Whether the points ``a`` and ``b`` lie in the same direction from ``at``: the
    sine of the angle between the two directions is at most :data:`_SAME_DIRECTION`."""
    ax, ay = a[0] - at[0], a[1] - at[1]
    bx, by = b[0] - at[0], b[1] - at[1]
    length = math.hypot(ax, ay) * math.hypot(bx, by)
    return (
        length > 0
        and ax * bx + ay * by > 0
        and abs(ax * by - ay * bx) <= _SAME_DIRECTION * length
    )


# Two sides from a node go in the same direction (see _same_direction) when the sine of
# the angle between them is at most this (about three degrees).
_SAME_DIRECTION = 0.05


def _outline_sides(geometry):
    """``{coordinate key: [(x, y), ...]}``: for each vertex of the outlines and holes of
    ``geometry`` (a polygon or multipolygon), the vertices next to it along them."""
    from pyrosm.new_elements import _coordinate_key
    from pyrosm.walking_areas import _oriented_rings

    sides = {}
    for part in shapely.get_parts(geometry):
        for ring in _oriented_rings(part):
            before, after = np.roll(ring, 1, axis=0), np.roll(ring, -1, axis=0)
            for here, *next_to in zip(ring, before, after):
                sides.setdefault(_coordinate_key(*here), []).extend(
                    (float(x), float(y)) for x, y in next_to
                )
    return sides


def _level_groups(by_space):
    """The spaces that meet at a vertex, ``{(osm_type, id): level ranges}``, as
    ``[(spaces, level), ...]``, the groups that join there: all of them when they have a
    level in common, else, from the lowest level up, each largest group of two or more
    that shares a level (a staircase on levels 1 and 2 meets the floor of level 1 below
    and that of level 2 above at the same nodes). ``level`` is the one level a group
    has in common, NaN when it has several."""
    common = _common_ranges(by_space.values())
    if common:
        return [(set(by_space), _one_level(common))]
    ends = sorted(
        {end for ranges in by_space.values() for pair in ranges for end in pair}
    )
    groups = []
    for level in ends:
        group = {
            key
            for key, ranges in by_space.items()
            if any(low <= level <= high for low, high in ranges)
        }
        if len(group) > 1 and group not in groups:
            groups.append(group)
    largest = [group for group in groups if not any(group < other for other in groups)]
    return [
        (group, _one_level(_common_ranges(by_space[key] for key in group)))
        for group in largest
    ]


def _common_ranges(range_lists):
    """The level ranges, ``(low, high)`` each, that all the ``range_lists`` cover."""
    lists = list(range_lists)
    common = list(lists[0]) if lists else []
    for ranges in lists[1:]:
        common = [
            (max(la, lb), min(ha, hb))
            for la, ha in common
            for lb, hb in ranges
            if max(la, lb) <= min(ha, hb)
        ]
    return common


def _one_level(ranges):
    """The one level of the level ``ranges``, NaN when they cover several."""
    (low, high), *_ = ranges
    return low if len(set(ranges)) == 1 and low == high else np.nan


def _node_places(spaces, walls, anchors):
    """``(places, walled)``: ``{node id: (x, y)}``, the places of nodes that the reads
    tell, and ``{node id: [(level ranges, vertices), ...]}``, for each ``indoor=wall``
    way among ``walls`` (with ``nodes`` lists) through a node, its levels (None without
    a ``level`` tag) and its vertices next to the node (None when its coordinates do not
    follow its node list). A node of the ``anchors`` (frames with ``id``, ``lon`` and ``lat``, such as
    the network's nodes) and of the walls (lines whose coordinates follow their node
    lists) has its place. The node lists of the ways among ``spaces`` (``(key,
    geometry, ranges, node list or None)`` each, see :func:`border_entrances`) tell a
    place in two ways: pyrosm turns polygon outlines counter-clockwise, so a node can be
    at either of two places (see :func:`_ring_places`), and the places that all the ways
    with the node allow, when only one, is its place; a way with a node whose place is
    known at just one of its two places is drawn in that direction, which gives the
    place of each of its nodes. A way whose node list does not match its outline tells
    nothing, and a node with places that disagree has none."""
    from pyrosm.new_elements import _coordinate_key
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    places, conflicts, walled = {}, set(), {}

    def place(node, xy):
        node = int(node)
        if node in places and _coordinate_key(*places[node]) != _coordinate_key(*xy):
            conflicts.add(node)
        places.setdefault(node, (float(xy[0]), float(xy[1])))

    for frame in anchors:
        if frame is not None and len(frame):
            for node, x, y in zip(frame["id"], frame["lon"], frame["lat"]):
                place(node, (x, y))
    if walls is not None and "nodes" in walls.columns:
        wall_tags = _frame_tags(walls, _frame_tag_columns(walls))
        for ring, line, tags in zip(walls["nodes"], walls.geometry, wall_tags):
            if not isinstance(ring, (list, np.ndarray)):
                continue
            ranges = level_ranges(tags.get("level"))
            coords = None
            if line is not None and line.geom_type == "LineString":
                coords = shapely.get_coordinates(line)
                if len(coords) != len(ring):
                    coords = None
            for n, node in enumerate(ring):
                next_to = None
                if coords is not None:
                    place(node, coords[n])
                    next_to = [
                        (float(coords[i][0]), float(coords[i][1]))
                        for i in (n - 1, n + 1)
                        if 0 <= i < len(coords)
                    ]
                walled.setdefault(int(node), []).append((ranges, next_to))
    ways, allowed = [], {}
    for key, geometry, _, ring in spaces:
        if ring is None or key[0] != "way" or geometry.geom_type != "Polygon":
            continue
        coords = shapely.get_coordinates(geometry.exterior)
        if len(ring) != len(coords):
            continue
        candidates = list(_ring_places(ring, coords))
        ways.append(candidates)
        for node, pair in candidates:
            both = set(pair)
            mine = allowed.setdefault(int(node), {}).get(key, both)
            allowed[int(node)][key] = mine & both
    for node, by_way in allowed.items():
        common = set.intersection(*by_way.values())
        if len(common) == 1:
            place(node, next(iter(common)))
    oriented = [False] * len(ways)
    changed = True
    while changed:
        changed = False
        for n, candidates in enumerate(ways):
            if oriented[n]:
                continue
            side = _drawn_direction(candidates, places, conflicts)
            if side is None:
                continue
            oriented[n] = changed = True
            for node, pair in candidates:
                place(node, side(pair))
    return {n: xy for n, xy in places.items() if n not in conflicts}, walled


def _drawn_direction(candidates, places, conflicts):
    """For a way's nodes with their two places (see :func:`_ring_places`), a function
    that picks each node's place (the one at its position, or the mirrored one) when a
    node with a known place among ``places`` (not in ``conflicts``) is at just one of
    its two places; None when no node tells."""
    from pyrosm.new_elements import _coordinate_key

    for node, (at, mirrored) in candidates:
        node = int(node)
        if node not in places or node in conflicts:
            continue
        known = _coordinate_key(*places[node])
        at, mirrored = _coordinate_key(*at), _coordinate_key(*mirrored)
        if at == mirrored:
            continue
        if known == at:
            return lambda pair: pair[0]
        if known == mirrored:
            return lambda pair: pair[1]
    return None


def _ring_places(ring, coords):
    """``(node, (at, mirrored))`` for each node of the closed way ``ring`` (its node ids)
    whose polygon outline has the coordinates ``coords``: the two places the node can
    have, as pyrosm turns outlines counter-clockwise (the reverse of a way drawn
    clockwise), the one at its position in the way and the one at the mirrored
    position."""
    last = len(ring) - 1
    for i, node in enumerate(ring[:-1]):
        yield node, (tuple(coords[i]), tuple(coords[(last - i) % last]))


def without_lift_area_ways(nodes, edges, lifts):
    """``nodes`` and ``edges`` without the edges of the closed ways among the lift areas
    ``lifts`` (see :func:`indoor_areas`), which a network read takes for a walkway
    around the lift, and without the nodes that only those edges used."""
    if lifts is None:
        return nodes, edges
    drop = edges["id"].isin(lifts.loc[lifts["osm_type"] == "way", "id"])
    drop &= edges["osm_type"] == "way"
    if not drop.any():
        return nodes, edges
    kept = edges[~drop]
    used = set(kept["u"]) | set(kept["v"])
    gone = (set(edges.loc[drop, "u"]) | set(edges.loc[drop, "v"])) - used
    return nodes[~nodes["id"].isin(gone)], kept


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


def with_indoor_crossings(nodes, edges, areas, doors, borders=None, lifts=None):
    """The ``nodes`` and ``edges`` of a network read with the edges across the indoor and
    walking ``areas`` (see :func:`pyrosm.walking_areas.crossing_rows`), marked True in
    the column ``walking_area`` (False on the other edges), entered through the
    network's nodes on the areas' outlines and, for the areas of
    :data:`INDOOR_AREA_FILTER`, through the ``doors`` (a frame of nodes with ``id``,
    ``lon``, ``lat`` and tags, a read of :data:`ENTRANCE_FILTER`) on their outlines and
    the lift nodes among them (see :func:`pyrosm.walking_areas.crossing_rows`), a door
    tagged ``repeat_on`` once on each of its levels (see :func:`door_copies`; a copy id
    that a node has already raises a ValueError), and
    through the ``borders``, the nodes where open indoor spaces meet (see
    :func:`border_entrances`), into those spaces only unless the node is already one of
    the network's, without those between two nodes an edge already joins (a lift node
    inside one of the lift areas ``lifts`` is left to that area);
    the doors and lift nodes that the new edges use join the nodes, with all their tags
    in ``tags``, and so do the border nodes, without tags."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags
    from pyrosm.walking_areas import crossing_rows

    if "walking_area" not in edges.columns:
        edges = edges.assign(walking_area=False)
    if areas is None:
        return nodes, edges
    columns = [col for col in ("id", "lon", "lat", "tags") if col in nodes.columns]
    entrances = nodes[columns]
    repeated = set()
    if doors is not None:
        doors = doors.assign(
            tags=[dumps(t) for t in _frame_tags(doors, _frame_tag_columns(doors))]
        )
        copied = door_copies(doors)
        twice = set(copied["id"][copied["id"].duplicated()])
        new = set(copied["id"]) - set(doors["id"])
        taken = set(nodes["id"]) | (set() if borders is None else set(borders["id"]))
        clash = sorted(twice | (new & taken))
        if clash:
            raise ValueError(
                "Doors tagged repeat_on cannot be copied per level: the ids of their "
                "copies, %s, are taken." % clash[:3]
            )
        # The doors copied per level, and their copies, enter areas on their level only
        repeated = new | {door for door in doors["id"] if node_copy_id(door, 1) in new}
        doors = copied
    if borders is not None and doors is not None:
        borders = borders[~borders["id"].isin(doors["id"])]
    added = [frame for frame in (doors, borders) if frame is not None]
    if doors is not None and "tags" in entrances.columns:
        # A door of the network enters areas with its tags as a door, on its own level
        door_tags = dict(zip(doors["id"], doors["tags"]))
        entrances = entrances.assign(
            tags=[
                door_tags.get(i, tags)
                for i, tags in zip(entrances["id"], entrances["tags"])
            ]
        )
    for frame in added:
        new = frame[~frame["id"].isin(nodes["id"])]
        new = new[[col for col in columns if col in new.columns]]
        entrances = pd.concat([entrances, new], ignore_index=True)
    plain = edges[~edges["walking_area"]]
    door_areas = [
        is_indoor_area(tags) for tags in _frame_tags(areas, _frame_tag_columns(areas))
    ]
    only = None
    if borders is not None:
        new = borders[~borders["id"].isin(nodes["id"])]
        only = dict(zip(new["id"], new["spaces"]))
    rows = crossing_rows(entrances, plain, areas, door_areas, only, repeated, lifts)
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
    if borders is not None:
        added[-1] = borders.drop(columns=["spaces", "level"]).assign(tags=None)
    used = set(rows["u"]) | set(rows["v"])
    for frame in added:
        new = frame[frame["id"].isin(used) & ~frame["id"].isin(nodes["id"])]
        if len(new):
            new = new[[col for col in nodes.columns if col in new.columns]]
            nodes = GeoDataFrame(
                pd.concat([nodes, new], ignore_index=True),
                geometry="geometry",
                crs=nodes.crs,
            )
    return nodes, edges


# The largest lift node id whose copies' ids (see node_copy_id) fit in a signed 64-bit
# integer.
_MAX_LIFT_ID = (2**63 - 1) // 1000 - 1


# The element metadata a lift edge takes from its lift node.
LIFT_METADATA = ("version", "timestamp", "changeset", "visible")


def node_copy_id(node, rank):
    """The id of the copy of ``node``, a lift node or a door on several levels (see
    :func:`door_copies`), at its ``rank``-th level from the lowest (1 to 499): negative
    and the same on every read."""
    return -(int(node) * 1000 + rank)


def door_copies(doors):
    """``doors`` (a frame of nodes with ``id`` and ``tags``, JSON with all their tags)
    with each door tagged ``repeat_on`` as one row per level, as Simple Indoor Tagging
    maps a door that repeats on other floors, such as the door of a lift: the door's
    ``level`` keeps its id, the other levels get :func:`node_copy_id`, ranked from the
    lowest, and each row's ``tags`` have its one ``level`` and no ``repeat_on``. Left as
    they are: a door that is also a lift node (``highway=elevator``, split per level by
    :func:`split_lifts`), a door whose id is not a positive one the copy ids fit, and a
    door on more than :data:`MAX_REPEATED_LEVELS` levels or with a range spanning more.
    """
    from pyrosm.pbf_writer import _tag_str

    rows = []
    for _, row in doors.iterrows():
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else {}
        own = level_ranges(tags.get("level")) or []
        levels = _whole_levels(own + (level_ranges(tags.get("repeat_on")) or []))
        copied = (
            tags.get("repeat_on")
            and tags.get("highway") != "elevator"
            and 0 < row["id"] <= _MAX_LIFT_ID
            and 1 < len(levels) <= MAX_REPEATED_LEVELS
        )
        if not copied:
            rows.append(row)
            continue
        home = own[0][0] if own else min(levels)
        others = [level for level in levels if level != home]
        for rank, level in [(0, home)] + list(enumerate(others, 1)):
            copy = {k: v for k, v in tags.items() if k != "repeat_on"}
            copy["level"] = _tag_str(level)
            node = row["id"] if rank == 0 else node_copy_id(row["id"], rank)
            rows.append(row.copy())
            rows[-1]["id"], rows[-1]["tags"] = node, dumps(copy)
    return GeoDataFrame(rows, geometry="geometry", crs=doors.crs).reset_index(drop=True)


# A door's repeat_on is read as further levels only when its levels span at most this
# many (see door_copies).
MAX_REPEATED_LEVELS = 100


def _whole_levels(ranges):
    """The levels of the level ``ranges``, ``(low, high)`` each: the ends of each range
    and the whole levels between them; none when a range spans more than
    :data:`MAX_REPEATED_LEVELS` levels."""
    if any(high - low > MAX_REPEATED_LEVELS for low, high in ranges):
        return []
    found = set()
    for low, high in ranges:
        found |= {low, high} | set(range(math.ceil(low), math.floor(high) + 1))
    return sorted(float(level) for level in found)


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
    keeps the node's id, the others get :func:`node_copy_id`, and each edge ends at the
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
    levels = _levels_at_nodes(edges, lifts)
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
        ids = {level: node_copy_id(lift, rank) for rank, level in enumerate(ordered)}
        ids[ordered[0]] = lift
        clash = [node for node in list(ids.values())[1:] if node in taken]
        if clash or len(ordered) >= 500:
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
    return _with_lift_rows(nodes, edges.assign(u=u, v=v), copies, lift_rows, metadata)


def _with_lift_rows(nodes, edges, copies, lift_rows, metadata, rows=None):
    """``nodes`` with the lift nodes ``copies`` (dicts) after them, and ``edges`` with
    the edge ``rows`` (a frame) and the lift edges ``lift_rows`` (dicts, ``walking_area``
    False where ``edges`` has that column, the ``metadata`` columns in the edges'
    dtypes) after them."""
    if "walking_area" in edges.columns:
        for row in lift_rows:
            row["walking_area"] = False
    attrs = dict(edges.attrs)
    lift_edges = pd.DataFrame(lift_rows).astype(
        {col: edges[col].dtype for col in metadata}
    )
    added = [lift_edges] if rows is None else [rows, lift_edges]
    edges = GeoDataFrame(
        pd.concat([edges, *added], ignore_index=True),
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


# How many metres longer a lift edge is than the escalator or stairs whose length it
# takes (see with_climb_lengths), so that a route by length takes the escalator.
LIFT_EXTRA_LENGTH = 1.0

# The height of one level in metres, for a lift edge, or a step from floor to floor at
# one place in stairs, between levels that no escalator or stairs join (see
# with_climb_lengths): a rough storey height.
LEVEL_HEIGHT = 4.0


def with_climb_lengths(edges):
    """``edges`` (with ``vertical``, ``u_level``, ``v_level`` and ``length``) with the
    ``length`` of the edges that change floors at one place. A lift edge (``vertical``
    "lift") gets the length of a climb between its two levels along escalators among
    ``edges`` (see :func:`_climb_length`), plus :data:`LIFT_EXTRA_LENGTH`; without one,
    along escalators and stairs; else :data:`LEVEL_HEIGHT` for each level between them,
    plus :data:`LIFT_EXTRA_LENGTH`. A line across stairs from a door tagged ``repeat_on``
    to its copy on the next floor, from a place to itself (see
    :func:`pyrosm.walking_areas.crossing_rows`), gets that of a climb along stairs;
    without one, along stairs and escalators; else :data:`LEVEL_HEIGHT` for each level
    between them."""
    lifts = (edges["vertical"] == "lift").to_numpy()
    u_level = edges["u_level"].to_numpy(dtype=float)
    v_level = edges["v_level"].to_numpy(dtype=float)
    steps = (
        _area_lines(edges)
        & ~np.isnan(u_level)
        & ~np.isnan(v_level)
        & (u_level != v_level)
        & (shapely.length(edges.geometry.to_numpy()) == 0)
    )
    if not (lifts | steps).any():
        return edges
    escalators = _ways_between_levels(edges, "escalator")
    stairs = _ways_between_levels(edges, "stairs")
    both = dict(escalators)
    for pair, ways in stairs.items():
        both[pair] = both.get(pair, []) + ways
    lengths = edges["length"].to_numpy(dtype=float).copy()
    for i in np.flatnonzero(lifts | steps):
        low, high = sorted((u_level[i], v_level[i]))
        point = shapely.get_point(edges.geometry.iloc[i], 0)
        length = _climb_length(escalators if lifts[i] else stairs, low, high, point)
        if length is None:
            length = _climb_length(both, low, high, point)
        if length is None:
            length = LEVEL_HEIGHT * (high - low)
        lengths[i] = round(length + (LIFT_EXTRA_LENGTH if lifts[i] else 0.0), 3)
    attrs = dict(edges.attrs)
    edges = edges.assign(length=lengths)
    edges.attrs = attrs
    return edges


def _climb_length(ways, low, high, point):
    """The length of a climb from level ``low`` to ``high`` along ``ways`` (``{(low,
    high): [(geometry, length), ...]}``, see :func:`_ways_between_levels`): that of the
    way nearest to ``point`` (lon/lat) between the two levels, else of the shortest
    chain of ways between levels from ``low`` up to ``high``, each the nearest of the
    ways between its two levels; None when no chain reaches ``high``."""
    scale = math.cos(math.radians(point.y))

    def nearest(found):
        return min(found, key=lambda way: _scaled_distance(way[0], point, scale))[1]

    if ways.get((low, high)):
        return nearest(ways[(low, high)])
    steps = [
        (a, b, nearest(found))
        for (a, b), found in ways.items()
        if low <= a and b <= high
    ]
    # Every step climbs, so taking the steps from the lowest level up finds the
    # shortest chain
    best = {low: 0.0}
    for a, b, step in sorted(steps):
        if a in best and best[a] + step < best.get(b, math.inf):
            best[b] = best[a] + step
    return best.get(high)


def _ways_between_levels(edges, kind):
    """``{(low, high): [(geometry, length), ...]}``: the ways among ``edges`` whose
    ``vertical`` is ``kind`` (lines across areas aside), by the lowest and highest level
    of their edges when these differ, each with the geometry and the summed ``length``
    of its edges."""
    rows = edges[(edges["vertical"] == kind).to_numpy() & ~_area_lines(edges)]
    found = {}
    for _, way in rows.groupby(["osm_type", "id"] if "osm_type" in rows else "id"):
        levels = np.concatenate(
            [way["u_level"].to_numpy(dtype=float), way["v_level"].to_numpy(dtype=float)]
        )
        levels = levels[~np.isnan(levels)]
        if len(levels) and levels.min() < levels.max():
            found.setdefault((levels.min(), levels.max()), []).append(
                (shapely.union_all(way.geometry.to_numpy()), way["length"].sum())
            )
    return found


def _scaled_distance(geometry, point, scale):
    """The distance between ``geometry`` and ``point`` (lon/lat) with the longitudes
    multiplied by ``scale``, the cosine of the latitude, so that it orders distances as
    metres would."""
    return shapely.distance(
        shapely.transform(geometry, lambda xy: xy * [scale, 1.0]),
        shapely.Point(point.x * scale, point.y),
    )


def lift_area_copy_id(area, osm_type, rank):
    """The id of the node of the lift area ``area`` (the id of a way, or of a relation
    when ``osm_type`` is "relation") at its ``rank``-th level from the lowest (1 to
    249): negative, the same on every read, and apart from the copies of lift nodes
    (see :func:`node_copy_id`, ranks below 500)."""
    return -(int(area) * 1000 + (750 if osm_type == "relation" else 500) + rank)


def _levels_at_nodes(edges, nodes):
    """``{node id: {level, ...}}``: the levels of ``edges`` (``u_level`` and
    ``v_level``) at those of their end nodes that are among ``nodes``."""
    found = {}
    wanted = list(nodes)
    for ends, end_levels in (("u", "u_level"), ("v", "v_level")):
        ends = edges[ends].to_numpy()
        end_levels = edges[end_levels].to_numpy(dtype=float)
        for i in np.flatnonzero(np.isin(ends, wanted) & ~np.isnan(end_levels)):
            found.setdefault(int(ends[i]), set()).add(float(end_levels[i]))
    return found


def _lift_area_entrances(lifts, nodes, edges):
    """``{lift area row: [(part, {level: [(node id, x, y), ...]}), ...]}``: for each part
    (polygon) of each lift area of ``lifts``, its entrances on each of the area's levels
    that the network's edges have there. The entrances are the network's nodes at a
    corner of the part's outline and its dead ends inside the part (see
    :func:`pyrosm.walking_areas._entrances`); a level counts when the area's ``level``
    tag lists it, or any level when the area has none."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags
    from pyrosm.walking_areas import _entrances

    ids = nodes["id"].to_numpy()
    lon = nodes["lon"].to_numpy(dtype=float)
    lat = nodes["lat"].to_numpy(dtype=float)
    geometries = lifts.geometry.to_numpy()
    area_rows, node_rows = shapely.STRtree(shapely.points(lon, lat)).query(
        geometries, predicate="intersects"
    )
    levels_at = _levels_at_nodes(edges, ids[node_rows])
    ends, counts = np.unique(
        np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()]),
        return_counts=True,
    )
    dead_end = np.isin(ids, ends[counts == 1])
    tags = _frame_tags(lifts, _frame_tag_columns(lifts))
    found = {}
    for area in np.unique(area_rows):
        ranges = level_ranges(tags[area].get("level"))
        rows = node_rows[area_rows == area]
        parts = []
        for part in shapely.get_parts(geometries[area]):
            at = {}
            for x, y, node_ids in _entrances(
                part, ids[rows], lon[rows], lat[rows], dead_end[rows]
            ).values():
                for node in node_ids:
                    for level in sorted(levels_at.get(node, ())):
                        if ranges is None or any(a <= level <= b for a, b in ranges):
                            at.setdefault(level, []).append((node, x, y))
            parts.append((part, at))
        found[int(area)] = parts
    return found


def _line_inside(polygon, start, end):
    """The shortest line inside ``polygon`` from the point ``start`` to the point
    ``end`` (see :func:`pyrosm.walking_areas._part_crossings`), bending at its inward
    corners; when the search finds none, the straight line if ``polygon`` covers it,
    else None."""
    from pyrosm.new_elements import _coordinate_key
    from pyrosm.walking_areas import _TOLERANCE, _part_crossings

    ends = {_coordinate_key(*start): (*start, [0]), _coordinate_key(*end): (*end, [1])}
    found = _part_crossings(polygon, ends)
    if found:
        return found[0][2]
    line = LineString([start, end])
    return line if polygon.buffer(_TOLERANCE).covers(line) else None


def join_lift_areas(nodes, edges, lifts):
    """``nodes`` and ``edges`` (with ``u_level`` and ``v_level``) with the lifts of the
    lift areas ``lifts`` (see :func:`indoor_areas`) that the network enters on two or
    more levels joined (entrances: see :func:`_lift_area_entrances`): each part of a
    multipolygon is a lift of its own, and the parts of lift areas mapped one floor at a
    time make one lift together (see :func:`_lift_stacks`). Each level with entrances
    gets a node in the middle of its part (ids: :func:`lift_area_copy_id`, ranked over
    the area's parts) and an edge to it from each entrance there along the shortest line
    inside the part, with the area's id, ``osm_type`` and tags and that level
    (``walking_area`` True, and ``level`` that one level), and lift edges of length 0,
    with the id, ``osm_type`` and metadata of the area of the lower level, join the
    nodes of neighbouring levels, as :func:`split_lifts` joins the copies of a lift node
    (see :func:`_lift_area_plan`). A lift whose levels a lift node split per level
    inside it joins already, all of them, is left out (see :func:`_joined_by_lift_node`).
    An area of which a node id is in the read already (a file that
    :meth:`pyrosm.OSM.write_pbf` wrote) is left as it is, all its parts."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags, _tag_str
    from pyrosm.walking_areas import area_edge_rows

    if lifts is None:
        return nodes, edges
    metadata = [
        col for col in LIFT_METADATA if col in edges.columns and col in lifts.columns
    ]
    node_metadata = [
        col for col in LIFT_METADATA if col in nodes.columns and col in lifts.columns
    ]
    tags = _frame_tags(lifts, _frame_tag_columns(lifts))
    taken = set(nodes["id"].tolist())
    items = []
    for area, parts in _lift_area_entrances(lifts, nodes, edges).items():
        row = lifts.iloc[area]
        planned = _lift_area_plan(row, parts) if 0 < row["id"] <= _MAX_LIFT_ID else []
        new_ids = [node for *_, ids in planned for node in ids.values()]
        if len(new_ids) >= 250 or any(node in taken for node in new_ids):
            continue
        taken.update(new_ids)
        ranges = level_ranges(tags[area].get("level")) or []
        span = (
            (min(a for a, _ in ranges), max(b for _, b in ranges)) if ranges else None
        )
        items += [(area, span, *part) for part in planned]
    split = edges[(edges["vertical"] == "lift") & (edges["osm_type"] == "node")]
    copies, lift_rows, found, level_of = [], [], [], {}
    for stack in _lift_stacks(items):
        at = {level: item for item in stack for level in item[5]}
        if _joined_by_lift_node([item[2] for item in stack], set(at), split):
            continue
        ordered = sorted(at)
        for level in ordered:
            area, _, part, middle, lines, ids = at[level]
            copies.append(
                {"id": ids[level], "lon": middle.x, "lat": middle.y}
                | {"geometry": middle}
                | {col: lifts.iloc[area][col] for col in node_metadata}
            )
            level_of[ids[level]] = level
            for node, line in lines[level]:
                found.append((area, node, ids[level], line))
        for low, high in zip(ordered, ordered[1:]):
            row = lifts.iloc[at[low][0]]
            lift_rows.append(
                {
                    "id": row["id"],
                    "osm_type": row["osm_type"],
                    "highway": "elevator",
                    "u": at[low][5][low],
                    "v": at[high][5][high],
                    "length": 0.0,
                    "u_level": low,
                    "v_level": high,
                    "vertical": "lift",
                    "geometry": LineString([at[low][3], at[high][3]]),
                    **{col: row[col] for col in metadata},
                }
            )
    if not lift_rows:
        return nodes, edges
    rows = area_edge_rows(lifts, found, edges)
    levels = rows["v"].map(level_of).to_numpy(dtype=float)
    rows = rows.assign(u_level=levels, v_level=levels, walking_area=True, vertical=None)
    if "level" in rows.columns:
        rows["level"] = [_tag_str(level) for level in levels]
    elif "tags" in rows.columns:
        rows["tags"] = [
            _with_level_tag(tags, level) for tags, level in zip(rows["tags"], levels)
        ]
    return _with_lift_rows(nodes, edges, copies, lift_rows, metadata, rows)


def _joined_by_lift_node(parts, levels, lift_edges):
    """Whether one lift node split per level (see :func:`split_lifts`; its lift edges
    among ``lift_edges``, with ``u_level`` and ``v_level``) lies in one of ``parts`` and
    is on every one of ``levels``."""
    if lift_edges.empty:
        return False
    points = shapely.points([line.coords[0][:2] for line in lift_edges.geometry])
    inside = np.zeros(len(points), dtype=bool)
    for part in parts:
        inside |= shapely.covers(part, points)
    on = {}
    for node, low, high in zip(
        lift_edges["id"].to_numpy()[inside],
        lift_edges["u_level"].to_numpy(dtype=float)[inside],
        lift_edges["v_level"].to_numpy(dtype=float)[inside],
    ):
        on.setdefault(node, set()).update((low, high))
    return any(levels <= node_levels for node_levels in on.values())


def _lift_stacks(items):
    """The lifts among the planned parts ``items`` of lift areas, ``(area row, (lowest,
    highest) level of the area's level tag or None, part, middle, lines, ids)`` each (see
    :func:`_lift_area_plan`), as lists of items entered on two or more levels together.
    A part of an area with a level tag makes one lift with a part above it whose area's
    lowest level is at most two levels above its own highest (one floor may be missing
    between them, the lift passes it), when the middle of each part lies in the other
    and the two have no entrance level in common; a lift area mapped one floor at a time
    thus makes one lift, and two lifts side by side stay apart."""
    parent = list(range(len(items)))
    levels = [set(item[4]) for item in items]
    parts = shapely.STRtree([item[2] for item in items])
    for n, (_, span, part, middle, *_) in enumerate(items):
        if span is None:
            continue
        # The parts that the middle of this one lies in
        for m in sorted(parts.query(middle, predicate="intersects")):
            _, above, other, other_middle, *_ = items[m]
            if above is None or not 0 < above[0] - span[1] <= 2:
                continue
            a, b = _root(parent, n), _root(parent, m)
            if a == b or levels[a] & levels[b]:
                continue
            if part.covers(other_middle) and other.covers(middle):
                parent[a] = b
                levels[b] |= levels[a]
    stacks = {}
    for n, item in enumerate(items):
        stacks.setdefault(_root(parent, n), []).append(item)
    return [stack for root_, stack in stacks.items() if len(levels[root_]) >= 2]


def _lift_area_plan(row, parts):
    """The parts of the lift area ``row`` entered at ``parts`` (see
    :func:`_lift_area_entrances`) with a line inside the part (see :func:`_line_inside`)
    from an entrance on at least one level, as ``(part, middle, lines, ids)``: the part,
    its middle point, ``{level: [(entrance node, line to the middle), ...]}``, and the
    node id of each level (:func:`lift_area_copy_id`, ranked over the parts)."""
    planned, rank = [], 0
    for part, at in parts:
        middle = part.representative_point()
        lines = {}
        for level, entrances in at.items():
            for node, x, y in entrances:
                line = _line_inside(part, (x, y), (middle.x, middle.y))
                if line is not None:
                    lines.setdefault(level, []).append((node, line))
        if not lines:
            continue
        ids = {
            level: lift_area_copy_id(row["id"], row["osm_type"], rank + i)
            for i, level in enumerate(sorted(lines), 1)
        }
        rank += len(ids)
        planned.append((part, middle, lines, ids))
    return planned


def _with_level_tag(tags, level):
    """The JSON ``tags`` of an edge into a lift area with ``level`` set to the edge's
    one ``level`` (in place of the area's list of levels)."""
    from pyrosm.pbf_writer import _tag_str

    tags = json.loads(tags) if tags else {}
    tags["level"] = _tag_str(level)
    return json.dumps(tags)
