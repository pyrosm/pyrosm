"""The indoor elements of a network read with ``indoor_elements=True``.

The platforms (:data:`PLATFORM_FILTER`) are edges of the network, also those that its
type leaves out, such as ``highway=platform`` in a walking network. The platform areas and
the indoor corridors and areas (:data:`INDOOR_AREA_FILTER`) are crossed as walking areas
are, each on its own level, entered also through the doors on their outlines, which
connect areas that share them. The edges get the level at each of their two ends, ``u_level`` and ``v_level``, from the
OSM ``level`` tags (see :func:`pyrosm.levels.level_ranges`). An edge with one level has it
at both ends. A level list or range on stairs or an escalator runs between the levels of
the walkways at its two ends; on another walkway, such as a corridor tagged ``level=0;1``,
the data does not say which part is on which level, and both ends stay empty. Stairs and
an escalator without a level range (no ``level`` tag or a single level) take their ends
from the walkways they join, when each end has one level and the two differ. An edge
without a ``level`` tag otherwise has no level. The column ``vertical`` marks escalators,
moving walkways and stairs; the ``length`` of an escalator is measured along its slope
(see :func:`with_indoor_columns`), and in a graph an escalator or moving walkway runs
only in the direction it moves (see :func:`moving_one_way`). A lift node (``highway=elevator``) where walkways on several
levels meet is split into one node per level, joined by lift edges (see
:func:`split_lifts`).
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


def _is_vertical(tags):
    """Whether an edge with the OSM tags ``tags`` is stairs, an escalator or a moving
    walkway."""
    return tags.get("highway") == "steps" or _is_moving(tags)


def edge_levels(edges, tags):
    """``(u_level, v_level)``: float arrays of the level at each end of ``edges`` (a
    frame with ``u`` and ``v``), whose OSM tags are ``tags`` (a list of dicts in their
    order), as the module docstring describes; NaN where the level is not known."""
    n = len(edges)
    low, high = np.full(n, np.nan), np.full(n, np.nan)
    vertical = np.zeros(n, dtype=bool)
    for i, row_tags in enumerate(tags):
        ranges = level_ranges(row_tags.get("level"))
        if ranges:
            low[i] = min(r[0] for r in ranges)
            high[i] = max(r[1] for r in ranges)
        vertical[i] = _is_vertical(row_tags)
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
    for i in np.flatnonzero(vertical & listed):
        ends = _span_ends(
            _span_levels(levels_at.get(u[i], ()), low[i], high[i]),
            _span_levels(levels_at.get(v[i], ()), low[i], high[i]),
            low[i],
            high[i],
        )
        if ends is not None:
            u_level[i], v_level[i] = ends
    for i in np.flatnonzero(vertical & ~listed):
        a, b = levels_at.get(u[i], ()), levels_at.get(v[i], ())
        if len(a) == 1 and len(b) == 1 and a[0] != b[0]:
            u_level[i], v_level[i] = a[0], b[0]
    return u_level, v_level


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
        return at_u[0], high if at_u[0] == low else low
    if at_v and not at_u and at_v[0] in (low, high):
        return (high if at_v[0] == low else low), at_v[0]
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


def with_indoor_columns(edges, escalator_slope):
    """``edges`` (the frame of a network read with ``nodes=True``) with the columns
    ``u_level`` and ``v_level`` (see :func:`edge_levels`) and ``vertical`` (see
    :func:`edge_vertical`; none on the lines across areas, ``walking_area`` True, which
    do not move whatever the area's tags say). The ``length`` of an escalator is its
    length on the map divided by the cosine of ``escalator_slope`` (degrees), its length
    along a slope of that angle, in millimetres; None keeps the length on the map."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    tags = _frame_tags(edges, _frame_tag_columns(edges))
    u_level, v_level = edge_levels(edges, tags)
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


def indoor_areas(candidates, edges):
    """The areas among ``candidates`` (a read of :data:`INDOOR_AREA_FILTER`), their
    polygons and multipolygons, without those whose edges across are already in
    ``edges`` (with ``walking_areas=True``); None when there are none."""
    if candidates is None:
        return None
    areas = candidates[candidates.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if "walking_area" in edges.columns:
        crossed = edges[edges["walking_area"]]
        done = set(zip(crossed["osm_type"], crossed["id"]))
        areas = areas[[key not in done for key in zip(areas["osm_type"], areas["id"])]]
    return areas.reset_index(drop=True) if len(areas) else None


def with_indoor_crossings(nodes, edges, areas, doors):
    """The ``nodes`` and ``edges`` of a network read with the edges across the indoor
    ``areas`` (see :func:`pyrosm.walking_areas.crossing_rows`), marked True in the column
    ``walking_area`` (False on the other edges), entered through the network's nodes and
    the ``doors`` (a frame of nodes with ``id``, ``lon`` and ``lat``) on the areas'
    outlines, without those between two nodes an edge already joins; the doors that
    the new edges use join the nodes."""
    from pyrosm.walking_areas import crossing_rows

    if "walking_area" not in edges.columns:
        edges = edges.assign(walking_area=False)
    if areas is None:
        return nodes, edges
    entrances = nodes[["id", "lon", "lat"]]
    if doors is not None:
        new_doors = doors[~doors["id"].isin(nodes["id"])][["id", "lon", "lat"]]
        entrances = pd.concat([entrances, new_doors], ignore_index=True)
    plain = edges[~edges["walking_area"]]
    rows = crossing_rows(entrances, plain, areas)
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


def _lift_ids(nodes):
    """The ids of the lift nodes (``highway=elevator``) among ``nodes``, by their
    ``tags`` column (dicts, or JSON text)."""
    if "tags" not in nodes.columns:
        return set()
    found = set()
    for node, tags in zip(nodes["id"], nodes["tags"]):
        if isinstance(tags, str):
            tags = json.loads(tags)
        if isinstance(tags, dict) and tags.get("highway") == "elevator":
            found.add(int(node))
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
