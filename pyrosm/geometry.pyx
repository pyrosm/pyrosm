import numpy as np
from shapely import linestrings, polygons, points, linearrings, \
    multilinestrings, multipolygons, get_geometry, symmetric_difference, \
    is_valid
from shapely import Geometry
from shapely import GEOSException
from shapely.linear import line_merge
from shapely.coordinates import get_coordinates
from shapely.predicates import is_geometry
from shapely.geometry import MultiPolygon
from shapely.ops import polygonize
from shapely import orient_polygons as _orient_polygons
from pyrosm.distance import Unit, haversine
from pyrosm.node_lookup cimport NodeLocations


cpdef orient_polygons(geometries):
    """Normalize Polygon/MultiPolygon ring orientation to the OGC/GeoJSON
    right-hand rule (exterior counter-clockwise, holes clockwise), matching
    osmium and QGIS (#230). Non-polygonal geometries pass through unchanged.

    Delegates to shapely's vectorized ``orient_polygons`` (requires shapely
    >= 2.1) rather than a per-geometry Python loop."""
    return _orient_polygons(geometries, exterior_cw=False)


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


cdef create_linear_ring(coordinates):
    try:
        return linearrings(coordinates)
    # pygeos 0.7.1 throws GEOSException
    except GEOSException as e:
        if "Invalid number of points" in str(e):
            return None
        elif "point array must contain" in str(e):
            return None
        raise e
    # pygeos 0.8.0 throws ValueError
    except ValueError as e:
        if "Provide at least 4 coordinates" in str(e):
            return None
    except Exception as e:
        raise e


cdef create_linestring(coordinates):
    try:
        return linestrings(coordinates)
    except GEOSException as e:
        if "Invalid number of points" in str(e):
            return None
        elif "point array must contain" in str(e):
            return None
        raise e
    except ValueError as e:
        if "Provide at least 2 coordinates" in str(e):
            return None
        if "not have enough dimensions" in str(e):
            raise e
    except Exception as e:
        raise e


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


cdef _line_components(merged):
    # The LineString components of a (Multi)LineString returned by line_merge.
    if merged is None:
        return []
    gt = merged.geom_type
    if gt == "LineString":
        return [merged]
    if gt == "MultiLineString" or gt == "GeometryCollection":
        return [g for g in merged.geoms if g.geom_type == "LineString"]
    return []


cdef _polygonal_only(geom):
    # Reduce an even-odd overlay result to a Polygon/MultiPolygon, dropping any
    # non-areal artifacts (stray lines/points from coincident boundaries). Returns
    # None when no polygonal area remains.
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "Polygon" or gt == "MultiPolygon":
        return geom
    if gt == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            if g.is_empty:
                continue
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend([p for p in g.geoms if not p.is_empty])
        if len(polys) == 0:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    return None


cdef _assemble_multipolygon(lines):
    # OSM multipolygon assembly by geometry, not member role (#21):
    #   1. line_merge all member ways into maximal lines and keep the closed rings
    #      (open/incomplete rings are dropped, not force-closed with a spurious edge);
    #   2. one simple polygon per ring;
    #   3. even-odd overlay (reduce symmetric_difference): a point is inside iff it is
    #      covered by an odd number of rings. This yields outers, holes and
    #      islands-in-holes independent of roles and matches osmium for valid data.
    cdef int i
    merged = line_merge(multilinestrings(lines))
    ring_polys = []
    for comp in _line_components(merged):
        coords = get_coordinates(comp)
        if not _is_closed_ring(coords):
            continue
        ring = create_linear_ring(coords)
        if ring is None:
            continue
        poly = polygons(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        ring_polys.append(poly)

    if len(ring_polys) == 0:
        return None

    geom = ring_polys[0]
    for i in range(1, len(ring_polys)):
        geom = symmetric_difference(geom, ring_polys[i])

    geom = _polygonal_only(geom)
    if geom is None:
        return None
    if not is_valid(geom):
        geom = _polygonal_only(fix_geometry(geom))
    return geom


cdef create_relation_geometry(node_coordinates, ways,
                              member_roles, force_linestring,
                              make_multipolygon,
                              bint drop_if_open=False):
    cdef int i, m_cnt
    # Get coordinates for relation
    coordinates = get_way_coordinates_for_polygon(node_coordinates, ways)

    # One LineString per member way. Member roles are deliberately NOT used to
    # classify outer/inner here: the OSM multipolygon algorithm decides that by
    # geometry (#21). Two-point segments are kept -- line_merge stitches them into
    # rings; ``make_multipolygon`` and ``drop_if_open`` no longer affect assembly.
    lines = []
    m_cnt = len(member_roles)
    for i in range(0, m_cnt):
        coords = coordinates[i]
        if len(coords) < 2:
            continue
        geometry = create_linestring(coords)
        if geometry is not None:
            lines.append(geometry)

    if len(lines) == 0:
        return None

    # Routes and similar are linestrings, not areas.
    if force_linestring:
        if len(lines) == 1:
            return lines
        geom = line_merge(multilinestrings(lines))
        if isinstance(geom, np.ndarray):
            return geom.tolist()
        return [geom]

    geom = _assemble_multipolygon(lines)
    if geom is None:
        return None
    return [geom]


cpdef create_point_geometries(xarray, yarray):
    return _create_point_geometries(xarray, yarray)


cdef is_valid_coordinate_pair(lat, lon):
    if lon > 180 or lon < -180:
        return False
    if lat > 90 or lat < -90:
        return False
    return True


cdef create_linestring_geometry(nodes, node_coordinates):
    cdef NodeLocations nc = node_coordinates
    coords = []
    kept_nodes = []
    node_data = []
    cdef int i, n = len(nodes)
    cdef long long node, idx
    cdef double lon, lat
    for i in range(0, n):
        node = nodes[i]
        if nc.contains(node):
            idx = nc.index(node)
            # Ensure coordinates are valid
            lon = nc.lon_at(idx)
            lat = nc.lat_at(idx)

            if is_valid_coordinate_pair(lat, lon):
                coords.append([(lon, lat)])
                kept_nodes.append(node)
                node_data.append(nc.record(idx, node))

    if len(coords) > 1:
        try:
            # Each geom segment should be constructed separately
            # (i.e. becomes a multilinestring)
            coords = np.array(coords, dtype=np.float64)
            coords = np.hstack([coords[:-1], coords[1:]])
            coord_cnt = len(coords)
            # Get an array of linestrings (segments of the way geometry)
            geom = linestrings(coords)

            # Get from and to-ids
            from_ids = kept_nodes[:-1]
            to_ids = kept_nodes[1:]

            return geom, from_ids, to_ids, node_data
        except GEOSException as e:
            if "Invalid number of points" in str(e):
                # node_data should always be a list
                return None, None, None, []
            else:
                raise e
        except Exception as e:
            raise e

    else:
        # node_data should always be a list
        return None, None, None, []


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


cdef _closed_way_is_polygon(area_value, has_linear_tag):
    # OSM area semantics for a closed way: an explicit 'area' tag wins
    # ('area=yes' -> Polygon, 'area=no' -> LineString); otherwise the way is an
    # area (Polygon) unless it carries a linear-feature tag (highway/barrier/route).
    if area_value == "yes":
        return True
    if area_value == "no":
        return False
    return not has_linear_tag

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


cdef _create_network_geometries_vectorized(node_coordinates, way_elements,
                                           bint build_node_data):
    # Vectorised network path: build every way's per-segment LineStrings with a
    # single batched ``shapely.linestrings`` call instead of one call per way.
    # A way takes the batched path only when all of its nodes are present and
    # have valid coordinates (the near-universal case); ways with dropped nodes
    # fall back to the exact per-way builder, so the kept-node subsequence (and
    # therefore the output) is identical to before. ``from``/``to`` ids and the
    # node-attribute records are only built when requested (graph export); plain
    # ``get_network`` discards them.
    cdef NodeLocations nc = node_coordinates
    cdef int n = len(way_elements['id'])
    cdef int i, w, W, vci
    cdef Py_ssize_t o0, o1, p
    keys = list(way_elements.keys())
    nodes_col = way_elements['nodes']

    # Flat node ids + per-way offsets, dropping empty ways (as the loop did).
    flat_list = []
    offsets = [0]
    nonempty_idx = []
    for i in range(n):
        wnodes = nodes_col[i]
        if len(wnodes) == 0:
            continue
        flat_list.extend(wnodes)
        offsets.append(len(flat_list))
        nonempty_idx.append(i)

    W = len(nonempty_idx)
    geometries = []
    from_ids = []
    to_ids = []
    node_attributes = []

    if W > 0:
        flat_nodes = np.asarray(flat_list, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)          # length W + 1
        way_lengths = offsets[1:] - offsets[:-1]               # length W

        # Vectorised coordinate gather + validity (mirrors is_valid_coordinate_pair).
        idx, lon, lat = nc.gather(flat_nodes)
        valid = ((idx >= 0)
                 & (lon <= 180.0) & (lon >= -180.0)
                 & (lat <= 90.0) & (lat >= -90.0))

        # Ways whose every node is present+valid (and >= 2 nodes) take the batched
        # path; the rest fall back to the exact per-way builder.
        valid_count = np.add.reduceat(valid.astype(np.int64), offsets[:-1])
        vectorizable = (valid_count == way_lengths) & (way_lengths >= 2)

        coords = np.column_stack([lon, lat])                   # (T, 2) float64
        seg_counts = way_lengths[vectorizable] - 1             # segments per vec way
        seg_start_pos = _concatenated_ranges(offsets[:-1][vectorizable], seg_counts)
        seg_geoms_all = None
        if len(seg_start_pos) > 0:
            segments = np.stack(
                [coords[seg_start_pos], coords[seg_start_pos + 1]], axis=1
            )
            seg_geoms_all = linestrings(segments)
        seg_cum = np.zeros(len(seg_counts) + 1, dtype=np.int64)
        np.cumsum(seg_counts, out=seg_cum[1:])

        vci = 0
        for w in range(W):
            if vectorizable[w]:
                geometries.append(seg_geoms_all[seg_cum[vci]:seg_cum[vci + 1]])
                if build_node_data:
                    o0 = offsets[w]
                    o1 = offsets[w + 1]
                    way_node_ids = flat_nodes[o0:o1]
                    from_ids.append(way_node_ids[:-1].tolist())
                    to_ids.append(way_node_ids[1:].tolist())
                    for p in range(o0, o1):
                        node_attributes.append(nc.record(idx[p], flat_nodes[p]))
                vci += 1
            else:
                geom, from_id, to_id, node_data = create_linestring_geometry(
                    nodes_col[nonempty_idx[w]], nc
                )
                geometries.append(geom)
                if build_node_data:
                    from_ids.append(from_id)
                    to_ids.append(to_id)
                    node_attributes += node_data

    for key in keys:
        way_elements[key] = way_elements[key][nonempty_idx]

    return way_elements, geometries, from_ids, to_ids, node_attributes


cdef _has_linear_tag(highway_arr, barrier_arr, route_arr, int i):
    # A closed way is linear (and so a LineString, not an area) when it carries a
    # highway/barrier/route tag -- read from THIS way's own tag values.
    return (
        (highway_arr is not None and highway_arr[i] is not None)
        or (barrier_arr is not None and barrier_arr[i] is not None)
        or (route_arr is not None and route_arr[i] is not None)
    )


cdef _single_area_geometry(nodes, node_coordinates, area_value, bint has_linear_tag):
    # One way's non-network geometry: a Polygon when it is a closed area, otherwise
    # a (Multi)LineString -- the exact per-way decision the loop made. Used for the
    # ways the vectorised builder leaves out (open ways, closed-but-linear ways,
    # ways with dropped nodes).
    if nodes[0] == nodes[-1] and _closed_way_is_polygon(area_value, has_linear_tag):
        return create_polygon_geometry(nodes, node_coordinates)
    geom = create_linestring_geometry(nodes, node_coordinates)[0]
    if isinstance(geom, Geometry) or geom is None:
        return geom
    # LineStrings come back as an array of segments.
    if len(geom) == 1:
        return geom[0]
    return multilinestrings(geom)


cdef _create_area_geometries_vectorized(node_coordinates, way_elements):
    # Vectorised area path: build the closed-area Polygons with a single batched
    # shapely.polygons(linearrings(...)) call instead of one call per way. A way
    # takes the batched path only when it is a closed area whose every node is
    # present with valid coordinates and it has at least 4 coordinates (a ring's
    # minimum) -- the dominant building/landuse case. Every other way (open ways,
    # closed ways tagged linear, ways with dropped nodes, too-short rings) falls
    # back to the exact per-way builder, so the output is identical to before.
    cdef NodeLocations nc = node_coordinates
    cdef int n = len(way_elements['id'])
    cdef int i, w, W, poly_i
    keys = list(way_elements.keys())
    nodes_col = way_elements['nodes']

    highway_arr = way_elements["highway"] if "highway" in way_elements else None
    barrier_arr = way_elements["barrier"] if "barrier" in way_elements else None
    route_arr = way_elements["route"] if "route" in way_elements else None
    area_arr = way_elements["area"] if "area" in way_elements else None

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
        if wnodes[0] == wnodes[-1]:
            area_value = area_arr[i] if area_arr is not None else None
            is_area.append(_closed_way_is_polygon(
                area_value,
                _has_linear_tag(highway_arr, barrier_arr, route_arr, i),
            ))
        else:
            is_area.append(False)

    W = len(nonempty_idx)
    geometries = [None] * W

    if W > 0:
        flat_nodes = np.asarray(flat_list, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)          # length W + 1
        way_lengths = offsets[1:] - offsets[:-1]               # length W
        is_area = np.asarray(is_area, dtype=bool)

        # Vectorised coordinate gather + validity (mirrors is_valid_coordinate_pair).
        idx, lon, lat = nc.gather(flat_nodes)
        valid = ((idx >= 0)
                 & (lon <= 180.0) & (lon >= -180.0)
                 & (lat <= 90.0) & (lat >= -90.0))
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

        for w in range(W):
            if not vectorizable[w]:
                i = nonempty_idx[w]
                area_value = area_arr[i] if area_arr is not None else None
                geometries[w] = _single_area_geometry(
                    nodes_col[i],
                    nc,
                    area_value,
                    _has_linear_tag(highway_arr, barrier_arr, route_arr, i),
                )

    for key in keys:
        way_elements[key] = way_elements[key][nonempty_idx]

    return way_elements, geometries, [], [], []


cdef _create_way_geometries(node_coordinates,
                            way_elements,
                            parse_network,
                            bint build_node_data=True):
    # Networks are linear and have their own vectorised builder; everything else
    # (buildings, landuse, ...) goes through the vectorised area builder.
    if parse_network:
        return _create_network_geometries_vectorized(
            node_coordinates, way_elements, build_node_data
        )
    return _create_area_geometries_vectorized(node_coordinates, way_elements)


cpdef create_way_geometries(node_coordinates, way_elements, parse_network,
                            bint build_node_data=True):
    return _create_way_geometries(node_coordinates, way_elements, parse_network,
                                  build_node_data)
