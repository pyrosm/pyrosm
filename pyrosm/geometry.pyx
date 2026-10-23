import numpy as np
from shapely import linestrings, polygons, points, linearrings, \
    multilinestrings, multipolygons, get_geometry, symmetric_difference, \
    is_valid
from shapely import GEOSException
from shapely.linear import line_merge
from shapely.coordinates import get_coordinates
from shapely.predicates import is_geometry
from shapely.geometry import MultiPolygon
from shapely.ops import polygonize
from shapely import orient_polygons as _orient_polygons
from shapely import GeometryType, get_type_id
from pyrosm.config.polygon_features import POLYGON_FEATURES, is_area
from pyrosm.distance import Unit, haversine
from pyrosm.node_lookup cimport NodeLocations

# The geometry types whose rings orient_polygons orients (a collection may hold polygons).
_POLYGONAL_TYPES = [
    GeometryType.POLYGON,
    GeometryType.MULTIPOLYGON,
    GeometryType.GEOMETRYCOLLECTION,
]


cpdef orient_polygons(geometries):
    """Normalize Polygon/MultiPolygon ring orientation to the OGC/GeoJSON
    right-hand rule (exterior counter-clockwise, holes clockwise), matching
    osmium and QGIS (#230). Only polygons, multipolygons and geometry collections
    are oriented; points, lines and missing geometries are returned as the same
    objects.

    Delegates to shapely's vectorized ``orient_polygons`` (requires shapely
    >= 2.1) rather than a per-geometry Python loop."""
    geometries = np.asarray(geometries, dtype=object)
    polygonal = np.isin(get_type_id(geometries), _POLYGONAL_TYPES)
    if not polygonal.any():
        return geometries
    oriented = geometries.copy()
    oriented[polygonal] = _orient_polygons(geometries[polygonal], exterior_cw=False)
    return oriented


cpdef fix_geometry(geometry, diff_threshold=20):
    """
    Fix for invalid geometries using two strategies:
        1. buffer(0) --> works in most cases.
        2. bowtie fix --> used if buffer breaks down the geometry
        3. If the difference is still huge between the
        original geometry and fix-candidate, returns
        invalid geometry
    """
    # Then try fixing with buffer
    fix_candidate = geometry.buffer(0)
    if fix_candidate.is_valid:
        # Ensure that the area of the geometry
        # hasn't changed dramatically
        # Sometimes taking buffer 0 totally breaks down
        # the original geometry having hundred/thousand-fold
        # difference in area
        try:
            diff = abs(1 - geometry.area / fix_candidate.area)
            if diff < diff_threshold:
                return fix_candidate
        except ZeroDivisionError:
            pass
        except Exception as e:
            raise e

    # If geometry is MultiPolygon do not try fix bowtie
    if isinstance(geometry, MultiPolygon):
        return geometry

    # Try fixing "bowtie" geometry
    ext = geometry.exterior
    mls = ext.intersection(ext)
    polys = polygonize(mls)
    fix_candidate = MultiPolygon(polys)
    if fix_candidate.is_valid:
        try:
            diff = abs(1 - geometry.area / fix_candidate.area)
            if diff < diff_threshold:
                return fix_candidate
        except ZeroDivisionError:
            pass
        except Exception as e:
            raise e
    # Otherwise return original geometry
    return geometry


cdef get_way_coordinates_for_polygon(node_coordinate_lookup, way_elements):
    cdef NodeLocations nc = node_coordinate_lookup
    cdef int i, ii, nn, n = len(way_elements["id"])
    cdef long long node, idx
    cdef double lon, lat
    features = []
    for i in range(0, n):
        way_nodes = way_elements["nodes"][i]
        nn = len(way_nodes)
        coords = []
        for ii in range(0, nn):
            node = way_nodes[ii]
            if nc.contains(node):
                idx = nc.index(node)
                lon = nc.lon_at(idx)
                lat = nc.lat_at(idx)
                if is_valid_coordinate_pair(lat, lon):
                    coords.append((lon, lat))
        features.append(coords)
    return features


cpdef valid_coordinates(rows, lon, lat):
    """Whether each gathered node (its store row ``rows``, -1 when absent) has coordinates
    in range, as ``is_valid_coordinate_pair`` checks one pair."""
    return ((rows >= 0)
            & (lon <= 180.0) & (lon >= -180.0)
            & (lat <= 90.0) & (lat >= -90.0))


cpdef ring_polygon(coords):
    """The polygon of the closed ring ``coords``, fixed with ``buffer(0)`` when invalid, or
    ``None`` when GEOS cannot build it."""
    ring = create_linear_ring(coords)
    if ring is None:
        return None
    try:
        polygon = polygons(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
    except GEOSException:
        return None
    return polygon


cdef _geometry_or_none(constructor, coordinates):
    """``constructor(coordinates)`` (``shapely.linearrings`` or ``shapely.linestrings``), or
    ``None`` when the coordinates are too few for the geometry. Coordinates of the wrong
    shape raise."""
    try:
        return constructor(coordinates)
    except GEOSException as e:
        if "Invalid number of points" in str(e) or "point array must contain" in str(e):
            return None
        raise
    except ValueError as e:
        if "not have enough dimensions" in str(e):
            raise
        return None


cpdef create_linear_ring(coordinates):
    """The ring of ``coordinates``, or ``None`` when there are too few of them."""
    return _geometry_or_none(linearrings, coordinates)


cdef create_linestring(coordinates):
    """The line of ``coordinates``, or ``None`` when there are too few of them."""
    return _geometry_or_none(linestrings, coordinates)


cdef _create_point_geometries(xarray, yarray):
    cdef:
        int N = len(xarray)
        float x, y

    geometries = []
    for i in range(0, N):
        coords = (xarray[i], yarray[i])
        geometries.append(coords)

    return [
        points(geom) if geom is not None 
        else None for geom in geometries
    ]

cdef bint _is_closed_ring(coords):
    # A closed ring needs at least 4 positions and identical first/last points.
    # OSM closed ways repeat the same node id at both ends, so their coordinates
    # are exactly equal and a plain comparison is sufficient.
    if coords.shape[0] < 4:
        return False
    return coords[0, 0] == coords[-1, 0] and coords[0, 1] == coords[-1, 1]


cdef _line_components(merged_lines):
    # The LineString components of a (Multi)LineString returned by line_merge.
    if merged_lines is None:
        return []
    geom_type = merged_lines.geom_type
    if geom_type == "LineString":
        return [merged_lines]
    if geom_type == "MultiLineString" or geom_type == "GeometryCollection":
        return [part for part in merged_lines.geoms if part.geom_type == "LineString"]
    return []


cpdef _polygonal_only(geometry):
    """Reduce an even-odd overlay result to a Polygon/MultiPolygon, dropping any non-areal
    artifacts (stray lines/points from coincident boundaries); ``None`` when no polygonal
    area remains."""
    if geometry is None or geometry.is_empty:
        return None
    geom_type = geometry.geom_type
    if geom_type == "Polygon" or geom_type == "MultiPolygon":
        return geometry
    if geom_type == "GeometryCollection":
        polygon_parts = []
        for part in geometry.geoms:
            if part.is_empty:
                continue
            if part.geom_type == "Polygon":
                polygon_parts.append(part)
            elif part.geom_type == "MultiPolygon":
                polygon_parts.extend([sub for sub in part.geoms if not sub.is_empty])
        if len(polygon_parts) == 0:
            return None
        if len(polygon_parts) == 1:
            return polygon_parts[0]
        return MultiPolygon(polygon_parts)
    return None


cdef _even_odd_overlay(list ring_polygons):
    """The symmetric difference of all ring polygons, taken in pairs round by round:
    ``(r0 ^ r1) ^ (r2 ^ r3)`` and so on, an odd last part carried to the next round."""
    parts = np.asarray(ring_polygons, dtype=object)
    while len(parts) > 1:
        n_paired = len(parts) // 2 * 2
        merged = symmetric_difference(parts[0:n_paired:2], parts[1:n_paired:2])
        parts = np.concatenate([merged, parts[n_paired:]])
    return parts[0]


cdef _assemble_multipolygon(member_lines):
    """The multipolygon of ``member_lines``, assembled by geometry, not member role (#21):

    1. line_merge all member ways into maximal lines and keep the closed rings
       (open/incomplete rings are dropped, not force-closed with a spurious edge);
    2. one simple polygon per ring;
    3. even-odd overlay (symmetric_difference of the rings, in pairs): a point is
       inside iff it is covered by an odd number of rings. This yields outers, holes
       and islands-in-holes independent of roles and matches osmium for valid data.
    """
    merged_lines = line_merge(multilinestrings(member_lines))
    ring_polygons = []
    for line in _line_components(merged_lines):
        coords = get_coordinates(line)
        if not _is_closed_ring(coords):
            continue
        polygon = ring_polygon(coords)
        if polygon is None or polygon.is_empty:
            continue
        ring_polygons.append(polygon)

    if len(ring_polygons) == 0:
        return None

    # Even-odd overlay. GEOS can raise on pathological/self-touching source rings; rather
    # than abort the whole parse, drop this one relation (its rings are unassemblable).
    try:
        area = _even_odd_overlay(ring_polygons)
    except GEOSException:
        return None

    area = _polygonal_only(area)
    if area is None:
        return None
    if not is_valid(area):
        area = _polygonal_only(fix_geometry(area))
    return area


cdef create_relation_geometry(node_coordinates, ways,
                              member_roles, force_linestring,
                              make_multipolygon,
                              bint drop_if_open=False):
    cdef int i, n_members
    # Get coordinates for relation
    coordinates = get_way_coordinates_for_polygon(node_coordinates, ways)

    # One LineString per member way. Member roles are deliberately NOT used to
    # classify outer/inner here: the OSM multipolygon algorithm decides that by
    # geometry (#21). Two-point segments are kept -- line_merge stitches them into
    # rings; ``make_multipolygon`` and ``drop_if_open`` no longer affect assembly.
    member_lines = []
    n_members = len(member_roles)
    for i in range(0, n_members):
        coords = coordinates[i]
        if len(coords) < 2:
            continue
        line = create_linestring(coords)
        if line is not None:
            member_lines.append(line)

    if len(member_lines) == 0:
        return None

    # Routes and similar are linestrings, not areas.
    if force_linestring:
        if len(member_lines) == 1:
            return member_lines
        merged = line_merge(multilinestrings(member_lines))
        if isinstance(merged, np.ndarray):
            return merged.tolist()
        return [merged]

    area = _assemble_multipolygon(member_lines)
    if area is None:
        return None
    return [area]


cpdef create_point_geometries(xarray, yarray):
    return _create_point_geometries(xarray, yarray)


cdef is_valid_coordinate_pair(lat, lon):
    if lon > 180 or lon < -180:
        return False
    if lat > 90 or lat < -90:
        return False
    return True


cdef tuple _kept_nodes(nodes, NodeLocations nc):
    """The store rows, ids and coordinates of the way's nodes that are present with valid
    coordinates, in the way's order."""
    rows = []
    kept_nodes = []
    coords = []
    cdef int i, n = len(nodes)
    cdef long long node, idx
    cdef double lon, lat
    for i in range(0, n):
        node = nodes[i]
        if nc.contains(node):
            idx = nc.index(node)
            lon = nc.lon_at(idx)
            lat = nc.lat_at(idx)
            if is_valid_coordinate_pair(lat, lon):
                rows.append(idx)
                kept_nodes.append(node)
                coords.append((lon, lat))
    return rows, kept_nodes, coords


cdef create_line_geometry(nodes, node_coordinates):
    """One LineString through the way's nodes that are present with valid coordinates, in
    order; None when fewer than two remain."""
    cdef NodeLocations nc = node_coordinates
    _, _, coords = _kept_nodes(nodes, nc)
    if len(coords) < 2:
        return None
    return linestrings(coords)


cdef create_polygon_geometry(nodes, node_coordinates):
    cdef NodeLocations nc = node_coordinates
    cdef int i, n = len(nodes)
    cdef long long node, idx
    cdef double lon, lat
    coords = []
    for i in range(0, n):
        node = nodes[i]
        if nc.contains(node):
            # Ensure lat/lon values are valid
            idx = nc.index(node)
            lon = nc.lon_at(idx)
            lat = nc.lat_at(idx)
            if is_valid_coordinate_pair(lat, lon):
                coords.append((lon, lat))

    if len(coords) > 2:
        try:
            return polygons(coords)
        except GEOSException as e:
            # Some geometries might not be valid for creating a Polygon
            # These might occur e.g. at the edge of the spatial extent
            if "Invalid number of points in LinearRing" in str(e):
                return None
            else:
                raise e
        # pygeos 0.8.0 throws ValueError
        except ValueError as e:
            if "Provide at least 4 coordinates" in str(e):
                return None
        except Exception as e:
            raise e
    else:
        return None


# The key of ``way_elements`` that holds, per way, whether its tags make it an area when it
# is closed (see :func:`tags_make_area`). A tuple, so no tag column can have it; the way
# frame builder takes it out before it builds the columns.
AREA_BY_TAGS = ("area by tags",)


cpdef tags_make_area(tag_dicts):
    """Whether the tags of each way make it an area when it is closed, from its tags
    ``tag_dicts`` (one mapping per way, all of its tags), by the OSM wiki's polygon
    features as OSMnx reads them (see :func:`pyrosm.config.polygon_features.is_area`)."""
    cdef Py_ssize_t i, n = len(tag_dicts)
    out = np.empty(n, dtype=bool)
    for i in range(n):
        out[i] = is_area(tag_dicts[i])
    return out

cdef _concatenated_ranges(starts, counts):
    # Vectorised concatenation of ``[start, start + count)`` for each (start, count)
    # pair: e.g. starts=[10, 20], counts=[2, 3] -> [10, 11, 20, 21, 22].
    cdef long long total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out_off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=out_off[1:])
    return (np.arange(total, dtype=np.int64)
            - np.repeat(out_off[:-1], counts)
            + np.repeat(starts, counts))


cdef dict _keep_rows(dict way_elements, rows, Py_ssize_t n):
    """``way_elements`` restricted to ``rows`` (the positions of the ways that have
    nodes); unchanged when that is all ``n`` of them."""
    if len(rows) == n:
        return way_elements
    index = np.asarray(rows, dtype=np.intp)
    for key in list(way_elements.keys()):
        way_elements[key] = way_elements[key][index]
    return way_elements


cdef tuple _way_node_arrays(way_elements, NodeLocations nc):
    """The node ids of the ways that have nodes as one flat array with per-way offsets,
    with their store rows and coordinates (-1 and NaN for absent nodes) and whether each
    pair of coordinates is valid. Returns ``(nonempty, flat_nodes, offsets, rows, lon,
    lat, valid)``; ``nonempty`` lists the positions of those ways."""
    nodes_col = way_elements['nodes']
    cdef int i, n = len(nodes_col)
    flat_list = []
    offsets = [0]
    nonempty = []
    for i in range(n):
        wnodes = nodes_col[i]
        if len(wnodes) == 0:
            continue
        flat_list.extend(wnodes)
        offsets.append(len(flat_list))
        nonempty.append(i)
    flat_nodes = np.asarray(flat_list, dtype=np.int64)
    rows, lon, lat = nc.gather(flat_nodes)
    valid = valid_coordinates(rows, lon, lat)
    return nonempty, flat_nodes, np.asarray(offsets, dtype=np.int64), rows, lon, lat, valid


cdef _complete_ways(valid, offsets):
    """Whether each way has at least two nodes, all with valid coordinates."""
    cumulative = np.zeros(len(valid) + 1, dtype=np.int64)
    np.cumsum(valid, out=cumulative[1:])
    counts = np.diff(offsets)
    return (cumulative[offsets[1:]] - cumulative[offsets[:-1]] == counts) & (counts >= 2)


cdef _create_network_lines(node_coordinates, way_elements):
    # A plain network read: one LineString per way. The ways whose nodes are all
    # present with valid coordinates (the near-universal case) come from one batched
    # ``shapely.linestrings`` call, the others from the per-way builder, through the
    # nodes that remain.
    cdef NodeLocations nc = node_coordinates
    cdef Py_ssize_t w
    nodes_col = way_elements['nodes']
    nonempty, flat_nodes, offsets, rows, lon, lat, valid = _way_node_arrays(
        way_elements, nc
    )
    batched = _complete_ways(valid, offsets)
    line_counts = np.diff(offsets)[batched]
    positions = _concatenated_ranges(offsets[:-1][batched], line_counts)
    geometries = np.empty(len(nonempty), dtype=object)
    if len(positions) > 0:
        geometries[batched] = linestrings(
            np.column_stack([lon[positions], lat[positions]]),
            indices=np.repeat(np.arange(len(line_counts)), line_counts),
        )
    for w in np.flatnonzero(~batched):
        geometries[w] = create_line_geometry(nodes_col[nonempty[w]], nc)
    way_elements = _keep_rows(way_elements, nonempty, len(nodes_col))
    return way_elements, geometries, [], [], []


cpdef create_network_segments(node_coordinates, way_elements):
    """Graph-export edges: each way's segments between consecutive nodes, with their
    end coordinates and node ids, and the nodes the segments use.

    Returns ``(way_elements, segment_counts, start, end, from_ids, to_ids, node_ids,
    node_rows)``: the ways that have nodes, each way's number of segments (0 when fewer
    than two of its nodes have valid coordinates), the lon/lat of the start and the end
    of all segments in way order with their end node ids, and the ids and store rows of
    the nodes they use, in order of first use. A way with dropped nodes is cut through
    the nodes that remain."""
    cdef NodeLocations nc = node_coordinates
    cdef Py_ssize_t w, start
    nodes_col = way_elements['nodes']
    nonempty, flat_nodes, offsets, rows, lon, lat, valid = _way_node_arrays(
        way_elements, nc
    )
    way_elements = _keep_rows(way_elements, nonempty, len(nodes_col))
    way_lengths = np.diff(offsets)
    batched = _complete_ways(valid, offsets)

    # The nodes each way uses: all nodes of a complete way, the remaining nodes of a way
    # with dropped nodes when at least two remain.
    used_counts = np.where(batched, way_lengths, 0)
    remaining = {}
    for w in np.flatnonzero(~batched):
        kept_rows, kept_ids, kept_coords = _kept_nodes(nodes_col[nonempty[w]], nc)
        if len(kept_rows) > 1:
            remaining[w] = (kept_rows, kept_ids, kept_coords)
            used_counts[w] = len(kept_rows)
    used_offsets = np.zeros(len(used_counts) + 1, dtype=np.int64)
    np.cumsum(used_counts, out=used_offsets[1:])
    total = int(used_offsets[-1])
    used_ids = np.empty(total, dtype=np.int64)
    used_rows = np.empty(total, dtype=np.int64)
    used_xy = np.empty((total, 2), dtype=np.float64)
    source = _concatenated_ranges(offsets[:-1][batched], way_lengths[batched])
    target = _concatenated_ranges(used_offsets[:-1][batched], way_lengths[batched])
    used_ids[target] = flat_nodes[source]
    used_rows[target] = rows[source]
    used_xy[target, 0] = lon[source]
    used_xy[target, 1] = lat[source]
    for w, (kept_rows, kept_ids, kept_coords) in remaining.items():
        start = used_offsets[w]
        used_ids[start:start + len(kept_ids)] = kept_ids
        used_rows[start:start + len(kept_rows)] = kept_rows
        used_xy[start:start + len(kept_coords)] = kept_coords

    # One segment per pair of consecutive used nodes of a way
    segment_counts = np.maximum(used_counts - 1, 0)
    starts = _concatenated_ranges(used_offsets[:-1], segment_counts)
    first_use = np.unique(used_ids, return_index=True)[1]
    first_use.sort()
    return (way_elements, segment_counts, used_xy[starts], used_xy[starts + 1],
            used_ids[starts], used_ids[starts + 1], used_ids[first_use],
            used_rows[first_use])


cpdef segment_lines(start, end):
    """2-point LineStrings from the lon/lat coordinates ``start`` to ``end``."""
    if len(start) == 0:
        return np.empty(0, dtype=object)
    return linestrings(np.stack([start, end], axis=1))


cdef _area_by_tag_columns(way_elements, Py_ssize_t n):
    """:func:`tags_make_area` from the tag columns of ``way_elements``, for ways passed
    without their flags: a tag of the rule that is not a column counts as absent."""
    columns = {key: way_elements[key] for key in POLYGON_FEATURES if key in way_elements}
    out = np.empty(n, dtype=bool)
    for i in range(n):
        out[i] = is_area(
            {key: values[i] for key, values in columns.items() if values[i] is not None}
        )
    return out


cdef _single_area_geometry(nodes, node_coordinates, bint area_by_tags):
    """One way's non-network geometry: a Polygon when it is closed and its tags make it
    an area, otherwise a LineString through its kept nodes. Used for the ways the
    vectorised builder leaves out (ways with dropped nodes, areas with fewer than four
    nodes)."""
    if nodes[0] == nodes[-1] and area_by_tags:
        return create_polygon_geometry(nodes, node_coordinates)
    return create_line_geometry(nodes, node_coordinates)


cdef _create_area_geometries_vectorized(node_coordinates, way_elements, area_by_tags):
    """The geometries of ``way_elements``: the closed-area Polygons built with a single
    batched ``shapely.polygons(linearrings(...))`` call, and the lines (open ways, closed
    ways whose tags make no area) with a single ``shapely.linestrings`` call, instead of one call per
    way. A way takes a batched path only when every node is present with valid coordinates
    and it has enough of them (4 for a ring, 2 for a line). Every other way (ways with
    dropped nodes, too-short rings) falls back to the per-way builder. ``area_by_tags``
    tells whether each way's tags make it an area when closed (see
    :func:`tags_make_area`); without it, the tag columns decide."""
    cdef NodeLocations nc = node_coordinates
    cdef int n = len(way_elements['id'])
    cdef int i, w, W, poly_i
    nodes_col = way_elements['nodes']
    if area_by_tags is None:
        area_by_tags = _area_by_tag_columns(way_elements, n)

    # Flat node ids + per-way offsets, dropping empty ways (as the loop did), with a
    # per-way "closed area" flag decided from each way's own tags.
    flat_list = []
    offsets = [0]
    nonempty_idx = []
    is_area = []
    for i in range(n):
        wnodes = nodes_col[i]
        if len(wnodes) == 0:
            continue
        flat_list.extend(wnodes)
        offsets.append(len(flat_list))
        nonempty_idx.append(i)
        is_area.append(wnodes[0] == wnodes[-1] and bool(area_by_tags[i]))

    W = len(nonempty_idx)
    geometries = [None] * W

    if W > 0:
        flat_nodes = np.asarray(flat_list, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)          # length W + 1
        way_lengths = offsets[1:] - offsets[:-1]               # length W
        is_area = np.asarray(is_area, dtype=bool)

        # Vectorised coordinate gather + validity (mirrors is_valid_coordinate_pair).
        idx, lon, lat = nc.gather(flat_nodes)
        valid = valid_coordinates(idx, lon, lat)
        valid_count = np.add.reduceat(valid.astype(np.int64), offsets[:-1])

        vectorizable = is_area & (valid_count == way_lengths) & (way_lengths >= 4)

        coords = np.column_stack([lon, lat])                   # (T, 2) float64
        ring_lengths = way_lengths[vectorizable]
        ring_positions = _concatenated_ranges(offsets[:-1][vectorizable], ring_lengths)
        if len(ring_positions) > 0:
            ring_index = np.repeat(
                np.arange(len(ring_lengths), dtype=np.int64), ring_lengths
            )
            try:
                poly_geoms = polygons(
                    linearrings(coords[ring_positions], indices=ring_index)
                )
            except GEOSException:
                # A degenerate ring (e.g. all-identical coordinates) makes the
                # batched ring builder raise where the per-way builder returned
                # None; demote every batched way to the per-way path so the result
                # still matches exactly.
                vectorizable = np.zeros(W, dtype=bool)
                poly_geoms = None
            if poly_geoms is not None:
                poly_i = 0
                for w in range(W):
                    if vectorizable[w]:
                        geometries[w] = poly_geoms[poly_i]
                        poly_i += 1

        # The lines (open ways, and closed ways that are not areas) whose nodes are all
        # present and valid: one LineString each from a single call.
        line_batch = ~is_area & (valid_count == way_lengths) & (way_lengths >= 2)
        line_counts = way_lengths[line_batch]
        line_positions = _concatenated_ranges(offsets[:-1][line_batch], line_counts)
        if len(line_positions) > 0:
            lines = linestrings(
                coords[line_positions],
                indices=np.repeat(np.arange(len(line_counts)), line_counts),
            )
            for w, line in zip(np.flatnonzero(line_batch), lines):
                geometries[w] = line
        vectorizable |= line_batch

        for w in range(W):
            if not vectorizable[w]:
                i = nonempty_idx[w]
                geometries[w] = _single_area_geometry(nodes_col[i], nc, area_by_tags[i])

    way_elements = _keep_rows(way_elements, nonempty_idx, n)

    return way_elements, geometries, [], [], []


cdef _create_way_geometries(node_coordinates, way_elements, parse_network, area_by_tags):
    """The geometries of ``way_elements``: networks are linear and have their own
    vectorised builder; everything else (buildings, landuse, ...) goes through the
    vectorised area builder, ``area_by_tags`` deciding which closed ways are areas."""
    if parse_network:
        return _create_network_lines(node_coordinates, way_elements)
    return _create_area_geometries_vectorized(node_coordinates, way_elements, area_by_tags)


cpdef create_way_geometries(node_coordinates, way_elements, parse_network, area_by_tags=None):
    """The geometries of ``way_elements``; ``area_by_tags`` (per way, see
    :func:`tags_make_area`) decides which closed ways are areas, else the tag columns."""
    return _create_way_geometries(node_coordinates, way_elements, parse_network, area_by_tags)
