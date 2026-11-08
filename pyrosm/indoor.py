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
without a ``level`` tag otherwise has no level. The column ``vertical`` marks escalators
and stairs.
"""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

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


def _is_vertical(tags):
    """Whether an edge with the OSM tags ``tags`` is stairs or an escalator."""
    return tags.get("highway") == "steps" or tags.get("conveying") not in (None, "no")


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


def edge_vertical(tags):
    """The vertical movement of edges with the OSM tags ``tags`` (a list of dicts):
    ``"escalator"`` for steps that move (``conveying`` other than no), ``"stairs"`` for
    other steps, None for the rest."""
    kinds = []
    for row_tags in tags:
        kind = None
        if row_tags.get("highway") == "steps":
            moving = row_tags.get("conveying") not in (None, "no")
            kind = "escalator" if moving else "stairs"
        kinds.append(kind)
    return kinds


def with_indoor_columns(edges):
    """``edges`` (the frame of a network read with ``nodes=True``) with the columns
    ``u_level`` and ``v_level`` (see :func:`edge_levels`) and ``vertical`` (see
    :func:`edge_vertical`)."""
    from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags

    tags = _frame_tags(edges, _frame_tag_columns(edges))
    u_level, v_level = edge_levels(edges, tags)
    return edges.assign(u_level=u_level, v_level=v_level, vertical=edge_vertical(tags))


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
