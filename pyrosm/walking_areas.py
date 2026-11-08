"""The edges across walking areas that a network read adds with ``walking_areas=True``.

A walking area is a closed way tagged ``area=yes`` or a multipolygon relation, such as a
pedestrian square, whose tags pass the network's filter. Its entrances are the network's
nodes on its outline and the network's dead ends inside it on the area's level. Each
pair of entrances gets one edge along the shortest line inside the area, which bends
only at the area's inward corners: the corners of the outline that point into the area
and those of its holes.
"""

import math

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from rapidjson import dumps

from pyrosm.distance import calculate_line_lengths
from pyrosm.levels import level_ranges
from pyrosm.new_elements import _coordinate_key
from pyrosm.pbf_writer import _frame_tag_columns, _frame_tags, _tag_key

# An area with more entrances and inward corners than this gets edges only between the
# entrances that a straight line inside it joins.
MAX_VISIBILITY_POINTS = 500

# A line is inside an area when it lies within the area widened by this many degrees
# (about 0.1 mm), so that rounding does not cut a line along the outline or through a
# corner.
_TOLERANCE = 1e-9


def area_filter(network_type, custom_filter, filter_type, shared_paths, filepath, box):
    """``(osm_keys, keep, rule)`` for the walking areas of a network read: the keys whose
    elements are candidates, ``keep(tags)``, whether an area's tags pass the network's
    filter, and the access rule that leaves areas built for another mode without a tag of
    the mode to their country (see :func:`~pyrosm.config.osm_filters.country_rule`), or
    None. The ``area=yes`` exclusion of the predefined network types does not apply; a
    ``custom_filter`` is used as given."""
    from pyrosm.config.osm_filters import country_rule, filter_for_read
    from pyrosm.engine.collect import _keep_fn
    from pyrosm.engine.readers import _network_spec

    osm_keys, data_filter, filter_type = _network_spec(
        network_type, None, custom_filter, filter_type, None, shared_paths
    )[1]
    if custom_filter is None:
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


def crossing_rows(nodes, edges, areas):
    """The edges across ``areas`` as rows with the columns of ``edges``: ``u`` and ``v``
    the entrances, ``length`` in metres, the line as ``geometry``, and the area's ``id``,
    ``osm_type`` and tags in the other columns, its tags without a column of ``edges`` in
    the JSON ``tags``. None when no area has two entrances."""
    ids = nodes["id"].to_numpy()
    lon = nodes["lon"].to_numpy(dtype=float)
    lat = nodes["lat"].to_numpy(dtype=float)
    ends = np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()])
    end_ids, counts = np.unique(ends, return_counts=True)
    dead_end = np.isin(ids, end_ids[counts == 1])

    geometries = areas.geometry.to_numpy()
    area_rows, node_rows = shapely.STRtree(shapely.points(lon, lat)).query(
        geometries, predicate="intersects"
    )
    positions = _dead_end_level_positions(edges, ids[node_rows[dead_end[node_rows]]])
    area_positions = [
        _level_position(tags) for tags in _frame_tags(areas, _frame_tag_columns(areas))
    ]
    found = []
    for area in np.unique(area_rows):
        candidates = node_rows[area_rows == area]
        joinable = np.array(
            [
                bool(end) and _on_same_level(positions[i], area_positions[area])
                for i, end in zip(ids[candidates], dead_end[candidates])
            ],
            dtype=bool,
        )
        for part in shapely.get_parts(geometries[area]):
            entrances = _entrances(
                part,
                ids[candidates],
                lon[candidates],
                lat[candidates],
                joinable,
            )
            for u, v, line in _part_crossings(part, entrances):
                found.append((area, u, v, line))
    if not found:
        return None

    area_index = np.array([f[0] for f in found])
    lines = np.array([f[3] for f in found], dtype=object)
    rows = areas.iloc[area_index].reset_index(drop=True)
    crossings = pd.DataFrame(
        {col: rows[col].to_numpy() if col in rows else None for col in edges.columns}
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


def _dead_end_level_positions(edges, dead_ends):
    """``{node id: position}`` of the nodes ``dead_ends``, each a node of one of
    ``edges``, by the tags of that edge (see :func:`_level_position`)."""
    u = edges["u"].to_numpy()
    v = edges["v"].to_numpy()
    rows = np.flatnonzero(np.isin(u, dead_ends) | np.isin(v, dead_ends))
    subset = edges.iloc[rows]
    wanted = set(dead_ends.tolist())
    positions = {}
    for a, b, tags in zip(
        u[rows], v[rows], _frame_tags(subset, _frame_tag_columns(subset))
    ):
        position = _level_position(tags)
        for node in (a, b):
            if node in wanted:
                positions[node] = position
    return positions


def _level_position(tags):
    """Where an element with the OSM tags ``tags`` is: ``(ranges, sides)``, the
    ranges of its ``level`` tag (see :func:`pyrosm.levels.level_ranges`) or None, and
    the sides of street level it is on, -1 below, 0 at and 1 above. The sides come from
    the levels when the element has them, else from ``layer``, and ``tunnel=yes``
    without a positive layer is below; an element with none of these tags is at street
    level."""
    ranges = level_ranges(tags.get("level"))
    if ranges:
        sides = set()
        for low, high in ranges:
            sides.update((int(np.sign(low)), int(np.sign(high))))
            if low < 0 < high:
                sides.add(0)
        return ranges, sides
    try:
        layer = float(tags.get("layer", 0))
    except (TypeError, ValueError):
        layer = 0.0
    if not math.isfinite(layer):
        layer = 0.0
    if tags.get("tunnel") == "yes" and layer <= 0:
        layer = -1.0
    return None, {int(np.sign(layer))}


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


def _entrances(polygon, ids, lon, lat, dead_end):
    """The entrances of ``polygon`` among the network nodes ``ids`` at ``lon``, ``lat``:
    ``{coordinate key: (x, y, [node ids])}`` of the nodes at a vertex of its outline or of
    a hole, and of the dead ends (``dead_end``, nodes of one edge) inside it, in the
    order of their first node id."""
    vertices = {
        _coordinate_key(x, y) for ring in _oriented_rings(polygon) for x, y in ring
    }
    inside = shapely.contains_properly(polygon, shapely.points(lon, lat)) & dead_end
    entrances = {}
    for row in np.argsort(ids, kind="stable"):
        key = _coordinate_key(lon[row], lat[row])
        if key in vertices or inside[row]:
            entrances.setdefault(key, (lon[row], lat[row], []))[2].append(int(ids[row]))
    return entrances


def _oriented_rings(polygon):
    """The outline and the holes of ``polygon`` as coordinate arrays without the closing
    point, the outline counterclockwise and the holes clockwise, so the area lies to the
    left of each."""
    polygon = shapely.orient_polygons(polygon)
    rings = [polygon.exterior] + list(polygon.interiors)
    return [np.asarray(ring.coords)[:-1, :2] for ring in rings]


def _inward_corners(polygon, scale):
    """The vertices of ``polygon`` where its outline or a hole turns away from the area:
    where the area's angle is over 180 degrees. ``scale`` multiplies longitudes, so the
    turns are measured in a plane true to the area's latitude."""
    corners = []
    for ring in _oriented_rings(polygon):
        x, y = ring[:, 0] * scale, ring[:, 1]
        dx_in, dy_in = x - np.roll(x, 1), y - np.roll(y, 1)
        dx_out, dy_out = np.roll(x, -1) - x, np.roll(y, -1) - y
        corners.append(ring[dx_in * dy_out - dy_in * dx_out < 0])
    return np.concatenate(corners)


def _part_crossings(polygon, entrances):
    """``(u, v, line)`` for each pair of node ids at two of the ``entrances`` of
    ``polygon`` (see :func:`_entrances`), the line a LineString, the shortest one inside
    the polygon. Over :data:`MAX_VISIBILITY_POINTS`, only the pairs that a straight line
    inside the polygon joins get an edge. A pair whose line passes another entrance gets
    no edge: the edges to and from that entrance cover it."""
    if len(entrances) < 2:
        return []
    points = np.array([(x, y) for x, y, _ in entrances.values()])
    count = len(points)
    scale = math.cos(math.radians(points[:, 1].mean()))
    corners = [
        xy
        for xy in _inward_corners(polygon, scale)
        if _coordinate_key(*xy) not in entrances
    ]
    if count + len(corners) <= MAX_VISIBILITY_POINTS:
        if corners:
            points = np.vstack([points, corners])
        step = _shortest_paths(points, *_visible_pairs(polygon, points), scale)
        pairs, paths = [], []
        for a in range(count):
            for b in range(a + 1, count):
                if step[a, b] >= 0:
                    path = [a]
                    while path[-1] != b:
                        path.append(step[path[-1], b])
                    pairs.append((a, b))
                    paths.append(path)
    else:
        pairs = list(zip(*_visible_pairs(polygon, points)))
        paths = [list(pair) for pair in pairs]
    if not pairs:
        return []
    lines = shapely.linestrings(
        points[np.concatenate(paths)],
        indices=np.repeat(np.arange(len(paths)), [len(path) for path in paths]),
    )
    passed = shapely.dwithin(
        lines[:, None], shapely.points(points[:count])[None, :], _TOLERANCE
    ).sum(axis=1)
    node_ids = [ids for _, _, ids in entrances.values()]
    return [
        (u, v, line)
        for (a, b), line, touched in zip(pairs, lines, passed)
        if touched == 2
        for u in node_ids[a]
        for v in node_ids[b]
    ]


def _visible_pairs(polygon, points):
    """The pairs ``(first, second)`` of indices of ``points`` (lon/lat), ``first`` below
    ``second``, that a straight line inside ``polygon`` joins."""
    first, second = np.triu_indices(len(points), 1)
    lines = shapely.linestrings(np.stack([points[first], points[second]], axis=1))
    widened = shapely.buffer(polygon, _TOLERANCE, join_style="mitre")
    shapely.prepare(widened)
    inside = shapely.covers(widened, lines)
    return first[inside], second[inside]


def _shortest_paths(points, first, second, scale):
    """The shortest paths between ``points`` (lon/lat) over the straight lines between the
    pairs ``first``, ``second``, measured with longitudes multiplied by ``scale``:
    ``step[i, j]`` is the point after ``i`` on the way to ``j``, -1 where no path joins
    them."""
    n = len(points)
    plane = points * [scale, 1.0]
    length = np.hypot(*(plane[first] - plane[second]).T)
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0.0)
    distance[first, second] = distance[second, first] = length
    step = np.tile(np.arange(n), (n, 1))
    for k in range(n):
        through = distance[:, k : k + 1] + distance[k : k + 1, :]
        shorter = through < distance * (1 - 1e-12)
        distance = np.where(shorter, through, distance)
        step = np.where(shorter, step[:, k : k + 1], step)
    return np.where(np.isfinite(distance), step, -1)
