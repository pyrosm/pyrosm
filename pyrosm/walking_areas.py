"""The edges across walking areas that a network read adds with ``walking_areas=True``.

A walking area is a closed way tagged ``area=yes`` or a multipolygon relation, such as a
pedestrian square, whose tags pass the network's filter. Its entrances are the network's
nodes on its outline and the network's dead ends inside it on the area's level. A node
that would enter areas on two or more levels enters, of the areas with a level, only
those on its own levels: those of its walkways with one level, else, at an end of
stairs, an escalator or a moving walkway with a level range, the level of that end that
the rest of the way gives, else its own ``level`` tag, else the ``level`` tags of its
other edges. Such a stair end enters only areas on its level, also where it touches one
area. A stair end without that level, in or on areas on several levels of its range, is
on the other end of the range from the one level of the areas at the way's other end,
else on the low end at the start of the way (in its drawn direction) and the high end at
its end, as are the two ends of stairs of one edge that both lie in areas of one level,
an end of the range. In an indoor read, the end of stairs or an escalator that joins
nothing also enters the platform or indoor area on its level that it lies outside of
within :data:`STAIR_END_REACH` metres, and a lift node enters, from inside, the platform
and indoor areas on the levels of its own ``level`` tag that its walkways do not reach
(see :func:`crossing_rows`). Each pair of entrances gets one edge along the shortest
line inside the area, which bends only at the area's inward corners: the corners of the
outline that point into the area and those of its holes. A pair gets no edge when the
edges of shorter pairs already join it along a route at most :data:`MAX_ROUTE_STRETCH`
times as long, so that an area with many entrances does not get an edge for every pair.
"""

import math

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from rapidjson import dumps

from pyrosm._area_crossings import (
    OPEN,
    VISIBLE,
    entrance_paths,
    entrances_passed,
    ring_links,
    shortest_paths,
    spanning_pairs,
    visibility,
)
from pyrosm.distance import calculate_line_lengths
from pyrosm.int64_hash import Int64Set
from pyrosm.levels import level_ranges, tag_layer
from pyrosm.new_elements import (
    _coordinate_codes,
    _coordinate_key,
    _key_codes,
    _query_pairs,
)
from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags, _tag_key

# An area with more entrances and inward corners than this gets edges only between the
# entrances that a straight line inside it joins.
MAX_VISIBILITY_POINTS = 500

# A line across an area is left out when the lines kept, shorter ones first, already join
# its two entrances within this many times its length (see _part_crossings).
MAX_ROUTE_STRETCH = 1.05

# How many metres outside an area that ``door_areas`` marks (see crossing_rows), on its
# level, the end of stairs or an escalator that joins nothing else, or only its own stairs or
# escalator area, may lie and still enter it.
STAIR_END_REACH = 2.5

# The tags that _level_position and pyrosm.indoor.is_vertical read of an element
_POSITION_TAG_KEYS = frozenset({"level", "layer", "tunnel", "highway", "conveying"})

# A line is inside an area when it lies within the area widened by this many degrees
# (about 0.1 mm), so that rounding does not cut a line along the outline or through a
# corner.
_TOLERANCE = 1e-9


def area_filter(network_type, custom_filter, filter_type, shared_paths, filepath, box):
    """``(osm_keys, keep, rule)`` for the walking areas of a network read: the keys whose
    elements are candidates, ``keep(tags)``, whether an area's tags pass the network's
    filter, and the access rule that leaves areas built for another mode without a tag of
    the mode to their country (see :func:`~pyrosm.config.osm_filters.country_rule`), or
    None. The ``area`` group of a filter that excludes, that of the predefined network
    types or of a ``custom_filter`` dict, does not apply: the areas such a network
    leaves out as ways are still crossed. Any other ``custom_filter`` is used as
    given."""
    from pyrosm.config.osm_filters import country_rule, filter_for_read
    from pyrosm.engine.collect import _keep_fn
    from pyrosm.engine.readers import _network_spec

    osm_keys, data_filter, filter_type = _network_spec(
        network_type, None, custom_filter, filter_type, None, shared_paths
    )[1]
    if filter_type == "exclude" and isinstance(data_filter, dict):
        data_filter = {k: v for k, v in data_filter.items() if k != "area"}
    data_filter = filter_for_read(data_filter, filter_type, filepath, box)
    data_filter, rule = country_rule(data_filter, filter_type)
    return osm_keys, _keep_fn((osm_keys, data_filter, filter_type)), rule


def select_areas(candidates, keep, rule):
    """The rows of ``candidates`` (a frame of a read with an ``area`` column) that are
    walking areas: closed ways tagged ``area=yes`` and multipolygon relations whose tags
    pass ``keep``, without those that ``rule`` leaves to their country where the
    country, by the area's first vertex, does not let the mode use them."""
    polygonal = np.isin(shapely.get_type_id(candidates.geometry.to_numpy()), (3, 6))
    candidates = candidates[polygonal].reset_index(drop=True)
    rows = []
    for row, (osm_type, tags) in enumerate(
        zip(
            candidates["osm_type"],
            _frame_tags(candidates, _frame_tag_columns(candidates)),
        )
    ):
        if osm_type == "way" and tags.get("area") != "yes":
            continue
        if osm_type == "relation" and tags.get("type") != "multipolygon":
            continue
        if not keep(tags):
            continue
        if rule is not None and rule.lacks_mode_tag(tags):
            x, y = shapely.get_coordinates(candidates.geometry.iloc[row])[0]
            if not rule.allowed_in(
                np.array([tags["highway"]]), np.array([x]), np.array([y])
            )[0]:
                continue
        rows.append(row)
    return candidates.iloc[rows].reset_index(drop=True)


def add_crossings(nodes, edges, areas):
    """``edges`` with a ``walking_area`` column, False on its rows, and a row with
    ``walking_area`` True for each edge across ``areas`` (see :func:`crossing_rows`)."""
    crossings = crossing_rows(nodes, edges, areas)
    result = edges.assign(walking_area=False)
    if crossings is not None:
        crossings["walking_area"] = True
        result = pd.concat([result, crossings], ignore_index=True)
        result = GeoDataFrame(result, geometry="geometry", crs=edges.crs)
    result.attrs = dict(edges.attrs)
    return result


def crossing_rows(
    nodes, edges, areas, door_areas=None, only=None, strict=(), lift_areas=None
):
    """The edges across ``areas`` as rows with the columns of ``edges``: ``u`` and ``v``
    the entrances (among ``nodes``, with their tags in ``tags`` when known), ``length``
    in metres, the line as ``geometry``, and the area's ``id``, ``osm_type`` and tags in
    the other columns, its tags without a column of ``edges`` in the JSON ``tags``. None
    when no area gets an edge. A node of ``nodes`` without an edge in ``edges``, a
    door, enters only the areas that ``door_areas`` marks (one bool per area; by default
    every area). The end of stairs or an escalator that is a dead end, enters no area
    but stairs or an escalator mapped as an area, such as its own, and has the level its
    way gives also enters the nearest area with a level that ``door_areas`` marks, on
    that level, when it lies outside the area within
    :data:`STAIR_END_REACH` metres: at the closest point of the area's outline, and the
    edges from it begin at the stair end. A lift node also enters, from inside, the areas
    of one level that ``door_areas`` marks on the levels of its own ``level`` tag that
    its walkways do not reach (see :func:`_lift_entrances`). A node in ``only``, a dict
    of node id to the ``(osm_type, id)`` of areas, enters only those areas, and a node in
    ``strict`` only the areas on its own level; in stairs mapped as an area or room, the
    nodes of ``strict`` at one place (a door tagged ``repeat_on`` and its copies) are
    also joined from each floor to the next (see :func:`_floor_steps`). A lift node
    inside one of the
    ``lift_areas`` (a frame of lift areas with their tags) on a level of both is left
    to that area (see :func:`pyrosm.indoor.join_lift_areas`) and does not enter other
    areas as a lift."""
    from pyrosm.indoor import _is_stair_area, node_levels

    ids = nodes["id"].to_numpy()
    lon = nodes["lon"].to_numpy(dtype=float)
    lat = nodes["lat"].to_numpy(dtype=float)
    ends = np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()])
    end_ids, counts = np.unique(ends, return_counts=True)
    on_edges = Int64Set(end_ids)
    dead_end = Int64Set(end_ids[counts == 1]).isin(ids)

    geometries = areas.geometry.to_numpy()
    near = _rows_in_boxes(lon, lat, shapely.bounds(geometries))
    area_rows, node_rows = shapely.STRtree(shapely.points(lon[near], lat[near])).query(
        geometries, predicate="intersects"
    )
    # In the order of the areas and the nodes, whatever the tree's order
    order = np.lexsort((near[node_rows], area_rows))
    area_rows, node_rows = area_rows[order], near[node_rows][order]
    if door_areas is not None:
        door = ~on_edges.isin(ids)
        keep = ~door[node_rows] | np.asarray(door_areas, dtype=bool)[area_rows]
        area_rows, node_rows = area_rows[keep], node_rows[keep]
    if only:
        keys = list(zip(areas["osm_type"], areas["id"]))
        keep = [
            ids[n] not in only or keys[a] in only[ids[n]]
            for a, n in zip(area_rows, node_rows)
        ]
        area_rows, node_rows = area_rows[keep], node_rows[keep]
    area_tags = _frame_tags(areas, _frame_tag_columns(areas))
    area_positions = [_level_position(tags) for tags in area_tags]
    near = np.unique(node_rows)
    loose = np.array([], dtype=np.int64)
    if door_areas is not None:
        loose = _loose_stair_ends(edges, ids, dead_end)
    wanted = np.union1d(near, loose)
    positions, open_ends, stair_ends, walked = _entrance_positions(
        edges, ids[wanted], ids[wanted[dead_end[wanted]]], node_levels(nodes)
    )
    if open_ends:
        chosen = _stair_end_choices(
            edges, open_ends, ids[node_rows], area_rows, area_positions
        )
        positions.update(chosen)
        stair_ends |= set(chosen)
    lifts = _lift_entrances(
        nodes,
        walked,
        node_rows,
        area_rows,
        area_positions,
        door_areas,
        geometries,
        lift_areas,
    )
    if door_areas is not None:
        # A lift node on no edge enters an area only as a lift
        lone = np.isin(ids[node_rows], _lone_lifts(nodes, on_edges))
        lifted = [
            (area, node) in lifts for area, node in zip(area_rows, ids[node_rows])
        ]
        keep = ~lone | np.array(lifted, dtype=bool)
        area_rows, node_rows = area_rows[keep], node_rows[keep]
    # The rows are in the order of the areas
    with_nodes, starts = np.unique(area_rows, return_index=True)
    ends = np.append(starts[1:], len(area_rows))
    parts, part_area = shapely.get_parts(geometries[with_nodes], return_index=True)
    part_starts = np.searchsorted(part_area, np.arange(len(with_nodes) + 1))
    each_part = []
    for k, area in enumerate(with_nodes):
        candidates = node_rows[starts[k] : ends[k]]
        joinable = np.array(
            [
                (bool(end) and _on_same_level(positions[i], area_positions[area]))
                or (area, i) in lifts
                for i, end in zip(ids[candidates], dead_end[candidates])
            ],
            dtype=bool,
        )
        each_part.extend(
            (area, part, ids[candidates], lon[candidates], lat[candidates], joinable)
            for part in parts[part_starts[k] : part_starts[k + 1]]
        )
    entrances = [
        (area, part, at)
        for (area, part, *_), at in zip(
            each_part, _parts_entrances([entry[1:] for entry in each_part])
        )
    ]
    off_level = _off_level_entrances(entrances, positions, area_positions, stair_ends)
    off_level -= lifts
    if len(strict):
        off_level |= {
            (area, node)
            for area, _, at in entrances
            for _, _, node_ids in at.values()
            for node in node_ids
            if node in strict
            and not _on_same_level(
                positions.get(node, (None, set())), area_positions[area]
            )
        }
    kept = {}
    for area, part, at in entrances:
        within = {}
        for key, (x, y, node_ids) in at.items():
            node_ids = [node for node in node_ids if (area, node) not in off_level]
            if node_ids:
                within[key] = (x, y, node_ids)
        kept.setdefault(area, []).append((part, within))
    reached = {}
    if len(loose):
        # A stair end in its own stairs or escalator area still reaches the area next to it
        entered = {
            node
            for area, parts in kept.items()
            if not _is_stair_area(area_tags[area])
            for _, within in parts
            for _, _, node_ids in within.values()
            for node in node_ids
        }
        ends = [
            row
            for row in loose
            if ids[row] in stair_ends and int(ids[row]) not in entered
        ]
        for area, k, node, end, point in _areas_within_reach(
            geometries,
            area_positions,
            door_areas,
            ids[ends],
            lon[ends],
            lat[ends],
            positions,
        ):
            parts = kept.setdefault(
                area, [(part, {}) for part in shapely.get_parts(geometries[area])]
            )
            part, within = parts[k]
            within.setdefault(_coordinate_key(*point), (*point, []))[2].append(node)
            parts[k] = (part, dict(sorted(within.items(), key=lambda e: min(e[1][2]))))
            reached[area, k, node] = end
    found = []
    each_part = [
        (area, k, part, within)
        for area, parts in kept.items()
        for k, (part, within) in enumerate(parts)
    ]
    crossings = _parts_crossings([(part, within) for _, _, part, within in each_part])
    for (area, k, _, within), lines in zip(each_part, crossings):
        for u, v, line in lines:
            start, end = reached.get((area, k, u)), reached.get((area, k, v))
            if start or end:
                coords = list(line.coords)
                line = shapely.LineString(
                    ([start] if start else []) + coords + ([end] if end else [])
                )
            found.append((area, u, v, line))
        if _is_stair_area(area_tags[area]):
            found.extend(
                (area, u, v, line)
                for u, v, line in _floor_steps(within, strict, positions)
            )
    if not found:
        return None
    return area_edge_rows(areas, found, edges)


def _rows_in_boxes(lon, lat, boxes):
    """The rows of the points ``lon``, ``lat`` that lie in one of the ``boxes`` (rows of
    ``minx, miny, maxx, maxy``, edges included), ascending."""
    order = np.argsort(lon, kind="stable")
    ordered = lon[order]
    found = np.zeros(len(lon), dtype=bool)
    for minx, miny, maxx, maxy in boxes:
        rows = order[
            np.searchsorted(ordered, minx, side="left") : np.searchsorted(
                ordered, maxx, side="right"
            )
        ]
        found[rows[(lat[rows] >= miny) & (lat[rows] <= maxy)]] = True
    return np.flatnonzero(found)


def _floor_steps(entrances, repeated, positions):
    """``(u, v, line)`` for each two of the ``repeated`` nodes (a door tagged
    ``repeat_on`` and its copies, see :func:`pyrosm.indoor.door_copies`) at one of the
    ``entrances`` (see :func:`_entrances`) whose levels (``positions``, see
    :func:`_level_position`) follow each other, the lower one first, the line from that
    place to itself."""
    found = []
    for x, y, node_ids in entrances.values():
        doors = sorted(
            (min(low for low, _ in positions[node][0]), node)
            for node in node_ids
            if node in repeated and positions.get(node, ((), set()))[0]
        )
        line = shapely.LineString([(x, y), (x, y)])
        found.extend((u, v, line) for (a, u), (b, v) in zip(doors, doors[1:]) if a < b)
    return found


def _lift_entrances(
    nodes,
    walked,
    node_rows,
    area_rows,
    area_positions,
    door_areas,
    geometries,
    lift_areas=None,
):
    """``{(area, node id)}``: the areas that ``door_areas`` marks (none without it) that
    a lift node enters on a level of its own ``level`` tag that none of its walkways with
    one level is on (``walked``, see :func:`_entrance_positions`). A lift node is tagged
    ``highway=elevator`` and its ``level`` tag names two or more levels; it enters an
    area of one level (``area_positions``, ``geometries``) among its levels that it lies
    inside or at a corner of (see :func:`_inside_or_at_corner`; ``node_rows`` and
    ``area_rows`` pair the rows of ``nodes`` with the areas they touch), when the levels
    it then reaches, those of its walkways among its levels and those of these areas,
    are two or more. A lift node inside one of the ``lift_areas`` (a frame of lift areas
    with their tags) enters none when they share a level, or the area has no ``level``
    tag: the lift area joins the floors of that lift."""
    from pyrosm.indoor import _tags_by_node

    if door_areas is None:
        return set()
    lifts = {}
    for node, tags in _tags_by_node(nodes, {"highway"}):
        ranges = level_ranges(tags.get("level"))
        if tags.get("highway") == "elevator" and ranges:
            if len({level for pair in ranges for level in pair}) > 1:
                lifts[node] = ranges
    ids = nodes["id"].to_numpy()
    lon = nodes["lon"].to_numpy(dtype=float)
    lat = nodes["lat"].to_numpy(dtype=float)
    if lifts and lift_areas is not None and len(lift_areas):
        area_ranges = [
            level_ranges(tags.get("level"))
            for tags in _frame_tags(lift_areas, _frame_tag_columns(lift_areas))
        ]
        rows = np.flatnonzero(np.isin(ids, list(lifts)))
        inside, covering = shapely.STRtree(lift_areas.geometry.to_numpy()).query(
            shapely.points(lon[rows], lat[rows]), predicate="covered_by"
        )
        for row, area in zip(rows[inside], covering):
            node = int(ids[row])
            ranges = area_ranges[area]
            if node in lifts and (
                not ranges or _on_same_level((lifts[node], ()), (ranges, ()))
            ):
                del lifts[node]
    if not lifts:
        return set()
    marked = np.asarray(door_areas, dtype=bool)
    entered = {}
    for area, row in zip(area_rows, node_rows):
        node = int(ids[row])
        area_ranges = area_positions[area][0]
        if node not in lifts or not marked[area] or not area_ranges:
            continue
        if len(area_ranges) != 1 or area_ranges[0][0] != area_ranges[0][1]:
            continue
        level = area_ranges[0][0]
        if level in walked.get(node, ()) or not _in_ranges(level, lifts[node]):
            continue
        if _inside_or_at_corner(geometries[area], lon[row], lat[row]):
            entered.setdefault(node, []).append((area, level))
    found = set()
    for node, at in entered.items():
        own = {
            level for level in walked.get(node, ()) if _in_ranges(level, lifts[node])
        }
        if len(own | {level for _, level in at}) > 1:
            found.update((area, node) for area, _ in at)
    return found


def _lone_lifts(nodes, end_ids):
    """The ids of the lift nodes among ``nodes`` (tagged ``highway=elevator``, without a
    ``door`` or ``entrance`` tag) that are on none of the edges whose ends are
    ``end_ids`` (a set)."""
    from pyrosm.indoor import _tags_by_node

    return [
        node
        for node, tags in _tags_by_node(nodes, {"highway"})
        if tags.get("highway") == "elevator"
        and tags.get("door") is None
        and tags.get("entrance") is None
        and node not in end_ids
    ]


def _in_ranges(level, ranges):
    """Whether ``level`` lies in one of the level ``ranges``, ``(low, high)`` each."""
    return any(low <= level <= high for low, high in ranges)


def _loose_stair_ends(edges, ids, dead_end):
    """The rows of the nodes ``ids`` that are dead ends (``dead_end``) at the end of
    stairs or an escalator (``highway=steps``) among ``edges``."""
    u = edges["u"].to_numpy()
    v = edges["v"].to_numpy()
    ends = Int64Set(ids[dead_end])
    rows = np.flatnonzero(ends.isin(u) | ends.isin(v))
    subset = edges.iloc[rows]
    steps = [
        node
        for a, b, tags in zip(
            u[rows],
            v[rows],
            _frame_tags(subset, _frame_tag_columns(subset), only={"highway"}),
        )
        if tags.get("highway") == "steps"
        for node in (a, b)
    ]
    return np.flatnonzero(dead_end & np.isin(ids, steps))


def _areas_within_reach(geometries, area_positions, marked, ids, lon, lat, positions):
    """``(area, part, node id, end, point)`` for each stair end ``ids`` at ``lon``, ``lat``
    that lies outside, within :data:`STAIR_END_REACH` metres, of a part of the areas
    with a level among ``geometries`` that ``marked`` marks (one bool per area), on the
    node's level (``positions`` and ``area_positions``, see :func:`_level_position`):
    the nearest such part, the node's coordinates ``end`` and the closest ``point`` of the
    part's outline."""
    areas = np.flatnonzero(
        np.asarray(marked, dtype=bool) & np.array([bool(p[0]) for p in area_positions])
    )
    if not len(ids) or not len(areas):
        return []
    rows, hits = _query_pairs(
        shapely.STRtree(geometries[areas]), lon, lat, STAIR_END_REACH
    )
    nearest = {}
    for row, area in zip(rows, areas[hits]):
        node = int(ids[row])
        if not _on_same_level(positions[node], area_positions[area]):
            continue
        scale = math.cos(math.radians(lat[row]))
        end = (float(lon[row]), float(lat[row]))
        for k, part in enumerate(shapely.get_parts(geometries[area])):
            if part.covers(shapely.Point(end)):
                continue
            plane = shapely.transform(part, lambda xy: xy * [scale, 1.0])
            line = shapely.shortest_line(plane, shapely.Point(end[0] * scale, end[1]))
            x, y = line.coords[0]
            point = (x / scale, y)
            length = calculate_line_lengths([shapely.LineString([end, point])], 3)[0]
            if length <= STAIR_END_REACH and length < nearest.get(node, (np.inf,))[0]:
                nearest[node] = (length, area, k, end, point)
    return [
        (area, k, node, end, point)
        for node, (_, area, k, end, point) in sorted(nearest.items())
    ]


def area_edge_rows(areas, found, edges):
    """The edges ``found`` in ``areas``, ``(area row, u, v, line)`` each, as rows with
    the columns of ``edges``: ``length`` in metres, the line as ``geometry``, and the
    area's ``id``, ``osm_type`` and tags in the other columns, its tags without a column
    of ``edges`` in the JSON ``tags``; sorted by area and nodes."""
    area_index = np.array([f[0] for f in found])
    lines = np.array([f[3] for f in found], dtype=object)
    rows = areas.iloc[area_index].reset_index(drop=True)
    crossings = pd.DataFrame(
        {col: rows[col].to_numpy() if col in rows else None for col in edges.columns}
    )
    # The text columns keep the edges' type, also those no area has a value in
    crossings = crossings.astype(
        {
            col: edges[col].dtype
            for col in edges.columns
            if isinstance(edges[col].dtype, pd.StringDtype)
        }
    )
    crossings["u"] = np.array([f[1] for f in found], dtype=edges["u"].dtype)
    crossings["v"] = np.array([f[2] for f in found], dtype=edges["v"].dtype)
    crossings["length"] = calculate_line_lengths(lines, decimals=3)
    crossings["geometry"] = lines
    if "tags" in crossings:
        columns = {_tag_key(col) for col in _frame_tag_columns(edges)}
        other = [
            {k: v for k, v in tags.items() if k not in columns}
            for tags in _frame_tags(areas, _frame_tag_columns(areas))
        ]
        crossings["tags"] = [dumps(other[i]) if other[i] else None for i in area_index]
    if "nodes" in crossings:
        crossings["nodes"] = [[u, v] for u, v in zip(crossings["u"], crossings["v"])]
    crossings = crossings.sort_values(
        ["osm_type", "id", "u", "v"], kind="stable", ignore_index=True
    )
    return GeoDataFrame(crossings, geometry="geometry", crs=edges.crs)


def _entrance_positions(edges, node_ids, dead_ends, own_levels):
    """``(positions, open_ends, stair_ends, walked)`` of the network nodes ``node_ids`` in
    or on an area.
    ``positions`` is ``{node id: position}`` (see :func:`_level_position`) by, in this
    order: the levels of the node's walkways with one level among ``edges``; at an end
    of stairs, an escalator or a moving walkway with a level range, the level of that
    end when the rest of the way gives it (see :func:`pyrosm.indoor.stair_end_levels`);
    its ``own_levels`` (see :func:`pyrosm.indoor.node_levels`); the ``level`` tags of
    its other edges; for a dead end (one of ``dead_ends``), the side of street level of
    its edge's ``layer``. Other nodes without levels are left out. ``stair_ends`` holds
    the nodes at an end of such a way whose level the rest of the way gives, also when
    a walkway gives their position. ``open_ends`` is ``{node id:
    (low, high)}``: the nodes of one such way without that level and without a walkway
    with one level, with the way's range. ``walked`` is ``{node id: {level, ...}}``, the
    levels of the node's walkways with one level."""
    from pyrosm.indoor import is_vertical, stair_end_levels

    known = stair_end_levels(edges, node_ids, own_levels)
    u = edges["u"].to_numpy()
    v = edges["v"].to_numpy()
    in_areas = Int64Set(node_ids)
    rows = np.flatnonzero(in_areas.isin(u) | in_areas.isin(v))
    subset = edges.iloc[rows]
    wanted = set(np.asarray(node_ids).tolist())
    ends = set(np.asarray(dead_ends).tolist())
    way_ids = subset["id"].to_numpy() if "id" in subset.columns else rows
    single, other, sides, open_ends, stair_ways = {}, {}, {}, {}, {}
    for a, b, way, tags in zip(
        u[rows],
        v[rows],
        way_ids,
        _frame_tags(subset, _frame_tag_columns(subset), only=_POSITION_TAG_KEYS),
    ):
        ranges, side = _level_position(tags)
        vertical = is_vertical(tags)
        levels = {level for pair in ranges or () for level in pair}
        for node in (a, b):
            if node not in wanted:
                continue
            on_one = not vertical and len(levels) == 1
            (single if on_one else other).setdefault(node, []).extend(ranges or ())
            if node in ends:
                sides[node] = side
            if len(levels) > 1 and vertical and node not in known:
                open_ends[node] = (min(levels), max(levels))
                stair_ways.setdefault(node, set()).add(way)
    open_ends = {
        node: span
        for node, span in open_ends.items()
        if not single.get(node) and len(stair_ways[node]) == 1
    }
    positions, stair_ends = {}, set()
    for node in wanted:
        own = [(level, level) for level in own_levels.get(node, ())]
        if node in known:
            stair_ends.add(node)
        if single.get(node):
            positions[node] = _ranges_position(single[node])
        elif node in known:
            positions[node] = _single_level_position(known[node])
        elif own or other.get(node):
            positions[node] = _ranges_position(own or other[node])
        elif node in sides:
            positions[node] = (None, sides[node])
    walked = {node: {low for low, _ in ranges} for node, ranges in single.items()}
    return positions, open_ends, stair_ends, walked


def _area_levels_at(node_ids, area_rows, area_positions, ranges):
    """``{node id: levels}``: for the nodes with level ``ranges`` (``{node id: (low,
    high)}``), the levels within their range of the areas on one level that they lie in
    or on (the area of each node of ``node_ids`` in ``area_rows``, at
    ``area_positions``)."""
    found = {}
    for node, area in zip(node_ids, area_rows):
        levels = area_positions[area][0]
        if node not in ranges or not levels or len(levels) != 1:
            continue
        low, high = ranges[node]
        if levels[0][0] == levels[0][1] and low <= levels[0][0] <= high:
            found.setdefault(node, set()).add(levels[0][0])
    return found


def _stair_end_choices(edges, open_ends, node_ids, area_rows, area_positions):
    """``{node id: position}`` for the ``open_ends`` (see :func:`_entrance_positions`)
    at an end of their way, in or on areas on several levels of their range (see
    :func:`_area_levels_at`, with the nodes ``node_ids`` in or on the areas
    ``area_rows``): the other end of the range from the one level of the areas at the
    other end of the way, else the low end of the range at the start of the way and the
    high end at its end. Also the ends of a way of one edge whose two ends lie in areas
    of one and the same level, an end of its range: the low end at the start and the
    high end at the end, so that only one of them is on the areas' level (a longer way
    dips to the other level between its ends, see :func:`pyrosm.indoor.edge_levels`)."""
    from pyrosm.indoor import stair_way_ends

    ways = stair_way_ends(edges, list(open_ends))
    ranges = {way[0]: open_ends[node] for node, way in ways.items()}
    ranges.update(open_ends)
    levels_at = _area_levels_at(node_ids, area_rows, area_positions, ranges)
    chosen = {}
    for node, (low, high) in open_ends.items():
        if node not in ways:
            continue
        other, starts, inner = ways[node]
        here, at_other = levels_at.get(node, set()), levels_at.get(other, set())
        one_level = not inner and here and here == at_other and here <= {low, high}
        if (
            len(here) >= 2
            and len(at_other) == 1
            and next(iter(at_other)) in (low, high)
        ):
            level = high if next(iter(at_other)) == low else low
        elif len(here) >= 2 or one_level:
            level = low if starts else high
        else:
            continue
        chosen[node] = _single_level_position(level)
    return chosen


def _single_level_position(level):
    """The position (see :func:`_level_position`) of an element on the one ``level``."""
    return _ranges_position([(level, level)])


def _level_position(tags):
    """Where an element with the OSM tags ``tags`` is: ``(ranges, sides)``, the
    ranges of its ``level`` tag (see :func:`pyrosm.levels.level_ranges`) or None, and
    the sides of street level it is on, -1 below, 0 at and 1 above. The sides come from
    the levels when the element has them, else from ``layer``, and ``tunnel=yes``
    without a positive layer is below; an element with none of these tags is at street
    level."""
    ranges = level_ranges(tags.get("level"))
    if ranges:
        return _ranges_position(ranges)
    return None, {int(np.sign(tag_layer(tags)))}


def _ranges_position(ranges):
    """The position (see :func:`_level_position`) of an element on the level ``ranges``:
    the ranges and the sides of street level they reach."""
    sides = set()
    for low, high in ranges:
        sides.update((int(np.sign(low)), int(np.sign(high))))
        if low < 0 < high:
            sides.add(0)
    return ranges, sides


def _on_same_level(a, b):
    """Whether the positions ``a`` and ``b`` (see :func:`_level_position`) share a
    level: two level ranges that overlap when both have levels, else a side of street
    level."""
    if a[0] and b[0]:
        return any(
            low_a <= high_b and low_b <= high_a
            for low_a, high_a in a[0]
            for low_b, high_b in b[0]
        )
    return bool(a[1] & b[1])


def _off_level_entrances(entrances, positions, area_positions, stair_ends):
    """``{(area, node id)}``: the areas with a level that each node enters (see
    :func:`_entrances`; ``entrances`` as ``(area, part, entrances)``) off the node's
    levels (``positions``, see :func:`_entrance_positions`), for the nodes with levels
    that enter areas on two or more levels (``area_positions``), and for the
    ``stair_ends``, whose level their way gives, also when they enter one area."""
    entered = {}
    for area, _, at in entrances:
        if area_positions[area][0]:
            for _, _, node_ids in at.values():
                for node in node_ids:
                    entered.setdefault(node, set()).add(area)
    off = set()
    for node, areas in entered.items():
        position = positions.get(node)
        if not position or not position[0]:
            continue
        several = any(
            not _on_same_level(area_positions[a], area_positions[b])
            for a in areas
            for b in areas
        )
        if several or node in stair_ends:
            off.update(
                (area, node)
                for area in areas
                if not _on_same_level(position, area_positions[area])
            )
    return off


def _vertex_keys(polygon):
    """The coordinate keys (see :func:`pyrosm.new_elements._coordinate_key`) of the
    vertices of the outline and the holes of ``polygon``."""
    return {_coordinate_key(x, y) for x, y in shapely.get_coordinates(polygon).tolist()}


def _inside_or_at_corner(geometry, x, y):
    """Whether the point ``x``, ``y`` lies inside ``geometry`` (a polygon or
    multipolygon) or at a vertex of the outline or a hole of one of its parts, where
    :func:`_entrances` takes a node as an entrance."""
    point = shapely.Point(x, y)
    return any(
        part.contains_properly(point) or _coordinate_key(x, y) in _vertex_keys(part)
        for part in shapely.get_parts(geometry)
    )


def _entrances(polygon, ids, lon, lat, dead_end):
    """The entrances of ``polygon`` among the network nodes ``ids`` at ``lon``, ``lat``:
    ``{coordinate key: (x, y, [node ids])}`` of the nodes at a vertex of its outline or of
    a hole, and of the dead ends (``dead_end``, nodes of one edge) inside it, in the
    order of their first node id."""
    return _parts_entrances([(polygon, ids, lon, lat, dead_end)])[0]


def _parts_entrances(entries):
    """:func:`_entrances` of each ``(polygon, ids, lon, lat, dead_end)`` of ``entries``,
    for all of them at once."""
    if not entries:
        return []
    polygons = np.array([polygon for polygon, *_ in entries], dtype=object)
    node_part = np.repeat(np.arange(len(entries)), [len(e[1]) for e in entries])
    lon = np.concatenate([np.asarray(e[2], dtype=float) for e in entries])
    lat = np.concatenate([np.asarray(e[3], dtype=float) for e in entries])
    coords, vertex_part = shapely.get_coordinates(polygons, return_index=True)
    at_vertex = _pairs_in(
        node_part,
        _coordinate_codes(lon, lat),
        vertex_part,
        _coordinate_codes(coords[:, 0], coords[:, 1]),
    )
    inside = shapely.contains_properly(polygons[node_part], shapely.points(lon, lat))
    chosen = at_vertex | (inside & np.concatenate([e[4] for e in entries]))
    found, start = [], 0
    for _, ids, part_lon, part_lat, _ in entries:
        entrances = {}
        for row in np.argsort(ids, kind="stable"):
            if chosen[start + row]:
                key = _coordinate_key(part_lon[row], part_lat[row])
                entrances.setdefault(key, (part_lon[row], part_lat[row], []))[2].append(
                    int(ids[row])
                )
        found.append(entrances)
        start += len(ids)
    return found


def _pairs_in(part, code, other_part, other_code):
    """Whether each pair ``(part[i], code[i])`` of the integer arrays is one of the
    pairs ``(other_part[j], other_code[j])``. The codes are compared by their rank among
    all codes, which with the part fits one int64."""
    _, ranks = np.unique(np.concatenate([code, other_code]), return_inverse=True)
    width = len(ranks) + 1
    return np.isin(
        np.asarray(part) * width + ranks[: len(code)],
        np.asarray(other_part) * width + ranks[len(code) :],
    )


def _oriented_rings(polygon):
    """The outline and the holes of ``polygon`` as coordinate arrays without the closing
    point, the outline counterclockwise and the holes clockwise, so the area lies to the
    left of each."""
    polygon = shapely.orient_polygons(polygon)
    rings = [polygon.exterior] + list(polygon.interiors)
    return [np.asarray(ring.coords)[:-1, :2] for ring in rings]


def _inward_corners(vertices, following, preceding, scale):
    """Whether each of the ``vertices`` of a polygon's rings (see
    :func:`pyrosm._area_crossings.ring_links` for ``following`` and ``preceding``) is a
    vertex where its outline or a hole turns away from the area: where the area's angle
    is over 180 degrees. ``scale`` multiplies longitudes, so the turns are measured in a
    plane true to the area's latitude."""
    x, y = vertices[:, 0] * scale, vertices[:, 1]
    dx_in, dy_in = x - x[preceding], y - y[preceding]
    dx_out, dy_out = x[following] - x, y[following] - y
    return dx_in * dy_out - dy_in * dx_out < 0


def _part_crossings(polygon, entrances):
    """``(u, v, line)`` for each pair of node ids at two of the ``entrances`` of
    ``polygon`` (see :func:`_entrances`), the line a LineString, the shortest one inside
    the polygon. Over :data:`MAX_VISIBILITY_POINTS`, only the pairs that a straight line
    inside the polygon joins get an edge. A pair whose line passes another entrance gets
    no edge: the edges to and from that entrance cover it. Nor does a pair that the edges
    of shorter pairs already join within :data:`MAX_ROUTE_STRETCH` times its length (see
    :func:`pyrosm._area_crossings.spanning_pairs`)."""
    return _parts_crossings([(polygon, entrances)])[0]


def _parts_crossings(parts):
    """:func:`_part_crossings` of each ``(polygon, entrances)`` of ``parts``, for all of
    them at once: the rings, the inward corners, the pairs that rounding leaves open and
    the lines of the edges are worked out for all parts together."""
    found = [[] for _ in parts]
    live = [i for i, (_, entrances) in enumerate(parts) if len(entrances) >= 2]
    if not live:
        return found
    polygons = np.array([parts[i][0] for i in live], dtype=object)
    entrances = [parts[i][1] for i in live]
    points = [np.array([(x, y) for x, y, _ in at.values()]) for at in entrances]
    counts = [len(p) for p in points]
    scales = [math.cos(math.radians(p[:, 1].mean())) for p in points]
    outlines, corners = _part_outlines(polygons, scales, entrances)
    few = [n + len(c) <= MAX_VISIBILITY_POINTS for n, c in zip(counts, corners)]
    for k in np.flatnonzero(few):
        if len(corners[k]):
            points[k] = np.vstack([points[k], corners[k]])
    paths = []
    for k, (first, second) in enumerate(_visible_pairs(polygons, points, outlines)):
        plane = points[k] * [scales[k], 1.0]
        if few[k]:
            length = np.hypot(*(plane[first] - plane[second]).T)
            step, distance = shortest_paths(
                len(points[k]), first, second, length, 1 - 1e-12
            )
            first, second, lengths, offsets, path = entrance_paths(
                step, distance, counts[k]
            )
        else:
            lengths = np.array(
                [float(np.hypot(*(plane[a] - plane[b]))) for a, b in zip(first, second)]
            )
            offsets = np.arange(0, 2 * len(first) + 1, 2, dtype=np.intp)
            path = np.column_stack([first, second]).ravel()
        if not len(first):
            paths.append(None)
            continue
        passed = _entrances_passed(points[k], counts[k], offsets, path)
        candidates = np.flatnonzero(passed == 2)
        order = candidates[np.argsort(lengths[candidates], kind="stable")]
        kept = spanning_pairs(
            len(points[k]), first, second, lengths, order, MAX_ROUTE_STRETCH
        )
        paths.append((first, second, offsets, path, kept))
    lines = _path_lines(points, paths)
    for k, i in enumerate(live):
        if paths[k] is None:
            continue
        first, second, _, _, kept = paths[k]
        node_ids = [ids for _, _, ids in entrances[k].values()]
        found[i] = [
            (u, v, lines[k][j])
            for j, pair in enumerate(kept)
            for u in node_ids[first[pair]]
            for v in node_ids[second[pair]]
        ]
    return found


def _part_outlines(polygons, scales, entrances):
    """``(outlines, corners)`` of the ``polygons``: for each, ``(vertices, following,
    preceding)``, its rings as :func:`_oriented_rings` gives them, one after another, and
    the vertex after and before each (see :func:`pyrosm._area_crossings.ring_links`);
    and its inward corners (see :func:`_inward_corners`, with its longitude scale of
    ``scales``) that are not at one of its ``entrances`` (see :func:`_entrances`)."""
    rings, ring_part = shapely.get_rings(
        shapely.orient_polygons(polygons), return_index=True
    )
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    sizes = np.bincount(ring_of, minlength=len(rings)) - 1
    # Each ring without its closing point
    closing = np.cumsum(sizes + 1) - 1
    keep = np.ones(len(coords), dtype=bool)
    keep[closing] = False
    vertices = np.ascontiguousarray(coords[keep], dtype=float)
    following, preceding = ring_links(sizes.astype(np.intp))
    part_sizes = np.bincount(ring_part, weights=sizes, minlength=len(polygons))
    starts = np.concatenate([[0], np.cumsum(part_sizes.astype(np.intp))])
    vertex_part = np.repeat(np.arange(len(polygons)), np.diff(starts))
    inward = np.flatnonzero(
        _inward_corners(vertices, following, preceding, np.asarray(scales)[vertex_part])
    )
    # A corner at an entrance of its part is not a corner to bend at
    at_entrance = _pairs_in(
        vertex_part[inward],
        _coordinate_codes(vertices[inward, 0], vertices[inward, 1]),
        np.repeat(np.arange(len(entrances)), [len(at) for at in entrances]),
        _key_codes([key for at in entrances for key in at]),
    )
    inward = inward[~at_entrance]
    corner_starts = np.searchsorted(inward, starts)
    outlines, corners = [], []
    for k in range(len(polygons)):
        a, b = starts[k], starts[k + 1]
        outlines.append((vertices[a:b], following[a:b] - a, preceding[a:b] - a))
        corners.append(vertices[inward[corner_starts[k] : corner_starts[k + 1]]])
    return outlines, corners


def _path_lines(points, paths):
    """For each part, the LineStrings of its kept paths: ``paths[k]`` is ``(first,
    second, offsets, path, kept)`` (see :func:`_parts_crossings`) through ``points[k]``,
    or None for a part without paths."""
    coords, indices, counts, total = [], [], [], 0
    for k, entry in enumerate(paths):
        if entry is None:
            counts.append(0)
            continue
        _, _, offsets, path, kept = entry
        sizes = offsets[kept + 1] - offsets[kept]
        rows = np.repeat(offsets[kept] - (np.cumsum(sizes) - sizes), sizes)
        coords.append(points[k][path[rows + np.arange(sizes.sum())]])
        indices.append(np.repeat(np.arange(len(kept)) + total, sizes))
        counts.append(len(kept))
        total += len(kept)
    lines = np.empty(0, dtype=object)
    if total:
        lines = shapely.linestrings(
            np.concatenate(coords), indices=np.concatenate(indices)
        )
    return np.split(lines, np.cumsum(counts)[:-1])


def _entrances_passed(points, count, offsets, path):
    """How many of the first ``count`` ``points`` (lon/lat) lie within the tolerance of
    each path through ``points`` (``path[offsets[i]:offsets[i + 1]]``), its two ends
    included."""
    passed, undecided = entrances_passed(
        np.ascontiguousarray(points, dtype=float), count, offsets, path, _TOLERANCE
    )
    if undecided.any():
        entrances = shapely.points(points[:count])
        for i in np.flatnonzero(undecided):
            line = shapely.LineString(points[path[offsets[i] : offsets[i + 1]]])
            passed[i] = int(shapely.dwithin(line, entrances, _TOLERANCE).sum())
    return passed


def _visible_pairs(polygons, points, outlines):
    """For each of the ``polygons``, the pairs ``(first, second)`` of indices of its
    ``points`` (lon/lat), ``first`` below ``second``, that a straight line inside the
    polygon joins (by :func:`shapely.covers` on the polygon widened by the tolerance,
    which is asked only where :func:`pyrosm._area_crossings.visibility` leaves a pair
    open, for all polygons at once). Its ``outlines`` entry holds its ring vertices (see
    :func:`_oriented_rings`), one ring after another, and the vertex after and before
    each (see :func:`pyrosm._area_crossings.ring_links`)."""
    valid = shapely.is_valid(polygons)
    pairs, insides, segments, segment_part, open_rows = [], [], [], [], []
    # np.triu_indices by number of points, for the parts of this call
    upper = {}
    for k, (polygon_points, outline) in enumerate(zip(points, outlines)):
        n = len(polygon_points)
        if n not in upper:
            upper[n] = np.triu_indices(n, 1)
        first, second = upper[n]
        if valid[k]:
            status = visibility(
                *outline, np.ascontiguousarray(polygon_points, dtype=float)
            )
            inside = status == VISIBLE
            open_pairs = np.flatnonzero(status == OPEN)
        else:
            inside = np.zeros(len(first), dtype=bool)
            open_pairs = np.arange(len(first))
        if len(open_pairs):
            segments.append(
                np.stack(
                    [
                        polygon_points[first[open_pairs]],
                        polygon_points[second[open_pairs]],
                    ],
                    axis=1,
                )
            )
            segment_part.append(np.full(len(open_pairs), k))
            open_rows.append((k, open_pairs))
        pairs.append((first, second))
        insides.append(inside)
    if segments:
        with_open = np.array([k for k, _ in open_rows])
        widened = shapely.buffer(polygons[with_open], _TOLERANCE, join_style="mitre")
        shapely.prepare(widened)
        segment_part = np.concatenate(segment_part)
        covered = shapely.covers(
            widened[np.searchsorted(with_open, segment_part)],
            shapely.linestrings(np.concatenate(segments)),
        )
        for k, open_pairs in open_rows:
            insides[k][open_pairs] = covered[: len(open_pairs)]
            covered = covered[len(open_pairs) :]
    return [
        (first[inside], second[inside])
        for (first, second), inside in zip(pairs, insides)
    ]
