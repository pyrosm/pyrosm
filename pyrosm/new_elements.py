"""New OSM elements from the features of GeoDataFrames, in the form
``pyrosm.OSM.write_pbf(..., apply_geometry=True)`` writes, with the ends of new lines
joined to existing nodes or to new nodes on existing ways (``pyrosm.OSM.add_elements``).
"""

import math
import numbers
from collections import namedtuple

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame

from pyrosm.distance import Unit, haversine
from pyrosm.pbf_writer import (
    _FIELD_COLS,
    _as_frames,
    _infer_osm_type,
    _normalize_id,
    _normalize_osm_type,
    _record_tags,
    _reproject_to_wgs84,
    _tag_key,
)

# Metres per degree of latitude on the sphere of :func:`~pyrosm.distance.haversine`.
_METRES_PER_DEGREE = math.radians(6371008.8)
# The columns of an input frame that are replaced, not written as tags.
_REPLACED_COLS = _FIELD_COLS - {"tags", "geometry"}


def validate_inputs(data, snap_to, snap_distance):
    """Check the arguments of ``OSM.add_elements``: ``data`` a GeoDataFrame or a list of
    them, ``snap_to`` and ``snap_distance`` given together, as a GeoDataFrame and a
    positive number of metres."""
    if not all(isinstance(gdf, GeoDataFrame) for gdf in _as_frames(data)):
        raise ValueError(
            "add_elements: 'data' should be a GeoDataFrame or a list of GeoDataFrames."
        )
    if (snap_to is None) != (snap_distance is None):
        raise ValueError(
            "add_elements: give 'snap_to' and 'snap_distance' together, or neither."
        )
    if snap_to is None:
        return
    if not isinstance(snap_to, GeoDataFrame):
        raise ValueError("add_elements: 'snap_to' should be a GeoDataFrame.")
    if (
        isinstance(snap_distance, bool)
        or not isinstance(snap_distance, numbers.Real)
        or not math.isfinite(snap_distance)
        or snap_distance <= 0
    ):
        raise ValueError(
            "add_elements: 'snap_distance' should be a positive number of metres, "
            "got %r." % (snap_distance,)
        )


def _snap_targets(snap_to, way_by_id):
    """The ids of the ways of ``snap_to`` that the parsed file holds, sorted, and the
    ids of its nodes."""
    missing = [None] * len(snap_to)
    otypes = snap_to["osm_type"] if "osm_type" in snap_to.columns else missing
    oids = snap_to["id"] if "id" in snap_to.columns else missing
    way_ids, node_ids = set(), []
    for otype, oid, geom in zip(otypes, oids, snap_to.geometry):
        if isinstance(otype, (str, bytes)):
            otype = _normalize_osm_type(otype)
        else:
            otype = _infer_osm_type(geom)
        oid = _normalize_id(oid)
        if otype == "way" and oid in way_by_id:
            way_ids.add(oid)
        elif otype == "node" and oid is not None:
            node_ids.append(oid)
    return sorted(way_ids), node_ids


def _snap_candidates(way_ids, node_ids, way_by_id, node_coordinates):
    """The ids, longitudes and latitudes of the nodes of the ways ``way_ids`` and of the
    nodes ``node_ids`` that the parsed file holds, sorted by id."""
    ids = list(node_ids)
    for wid in way_ids:
        ids.extend(way_by_id[wid]["nodes"])
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    idx, lon, lat = node_coordinates.gather(ids)
    found = idx >= 0
    if not found.any():
        raise ValueError(
            "add_elements: none of the elements of 'snap_to' are in the data this OSM "
            "object reads, so there are no nodes to snap to."
        )
    return ids[found], lon[found], lat[found]


def _query_pairs(tree, lon, lat, max_distance):
    """The (point, tree item) index pairs of the points (``lon``, ``lat``) and the items
    of ``tree`` whose boxes meet a box of ``max_distance`` metres around the point."""
    dlat = max_distance / _METRES_PER_DEGREE
    cos = np.cos(np.radians(np.minimum(np.abs(lat) + dlat, 90.0)))
    dlon = np.minimum(dlat / np.maximum(cos, 1e-9), 180.0)
    # A box that crosses the antimeridian is queried again on its other side.
    pairs = []
    for shift, rows in (
        (0.0, np.arange(len(lon))),
        (360.0, np.flatnonzero(lon - dlon < -180.0)),
        (-360.0, np.flatnonzero(lon + dlon > 180.0)),
    ):
        x, y, dx = lon[rows] + shift, lat[rows], dlon[rows]
        p, c = tree.query(shapely.box(x - dx, y - dlat, x + dx, y + dlat))
        pairs.append((rows[p], c))
    return np.concatenate([p for p, _ in pairs]), np.concatenate([c for _, c in pairs])


def _closest_per_point(n, point, item, dist, *ties):
    """For each of ``n`` points, the ``item`` of its pair with the smallest ``dist``
    (ties broken by ``ties``, then the item), or -1 when it has no pair."""
    closest = np.full(n, -1, dtype=np.int64)
    order = np.lexsort((item,) + tuple(reversed(ties)) + (dist, point))
    point, item = point[order], item[order]
    first = np.ones(len(point), dtype=bool)
    first[1:] = point[1:] != point[:-1]
    closest[point[first]] = item[first]
    return closest


def _nearest_nodes(lon, lat, candidates, max_distance):
    """For each point (``lon``, ``lat``), the index into ``candidates`` (see
    :func:`_snap_candidates`) of the closest node within ``max_distance`` metres, or -1.
    Of nodes at the same distance, the one with the smallest id is chosen."""
    _, cand_lon, cand_lat = candidates
    if len(lon) == 0:
        return np.full(0, -1, dtype=np.int64)
    tree = shapely.STRtree(shapely.points(cand_lon, cand_lat))
    point, cand = _query_pairs(tree, lon, lat, max_distance)
    dist = haversine(
        lat[point], lon[point], cand_lat[cand], cand_lon[cand], unit=Unit.METERS
    )
    near = dist <= max_distance
    return _closest_per_point(len(lon), point[near], cand[near], dist[near])


class _Segments:
    """The segments between the consecutive nodes of the ways ``way_ids``: per segment
    its way, the position of its first node in the way, its two nodes and their
    coordinates. Segments with a node the parsed file does not hold, and segments that
    cross the antimeridian, are left out."""

    def __init__(self, way_ids, way_by_id, node_coordinates):
        way, pos, a, b = [], [], [], []
        for wid in way_ids:
            refs = [int(r) for r in way_by_id[wid]["nodes"]]
            way.extend([wid] * (len(refs) - 1))
            pos.extend(range(len(refs) - 1))
            a.extend(refs[:-1])
            b.extend(refs[1:])
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        ia, ax, ay = node_coordinates.gather(a)
        ib, bx, by = node_coordinates.gather(b)
        keep = (ia >= 0) & (ib >= 0) & (np.abs(bx - ax) <= 180.0)
        self.way = np.asarray(way, dtype=np.int64)[keep]
        self.pos = np.asarray(pos, dtype=np.int64)[keep]
        self.a, self.b = a[keep], b[keep]
        self.ax, self.ay, self.bx, self.by = ax[keep], ay[keep], bx[keep], by[keep]
        self._tree = shapely.STRtree(
            shapely.linestrings(
                np.stack([self.ax, self.ay, self.bx, self.by], axis=1).reshape(-1, 2, 2)
            )
        )

    def closest(self, lon, lat, max_distance):
        """For each point (``lon``, ``lat``), the segment with the closest point within
        ``max_distance`` metres strictly between its nodes (-1 when there is none), that
        point's longitude and latitude, and its fraction of the way from the segment's
        first node to its second. Ties go to the smallest way id and position."""
        n = len(lon)
        seg, x, y, frac = np.full(n, -1), np.zeros(n), np.zeros(n), np.zeros(n)
        if n == 0 or len(self.a) == 0:
            return seg, x, y, frac
        point, item = _query_pairs(self._tree, lon, lat, max_distance)
        # Project in a plane where a degree of longitude is cos(latitude) as long.
        scale = np.cos(np.radians(lat[point]))
        dx = (self.bx[item] - self.ax[item]) * scale
        dy = self.by[item] - self.ay[item]
        px = ((lon[point] - self.ax[item] + 180.0) % 360.0 - 180.0) * scale
        py = lat[point] - self.ay[item]
        length2 = dx * dx + dy * dy
        t = np.where(
            length2 > 0, (px * dx + py * dy) / np.where(length2 > 0, length2, 1), 0
        )
        cx = self.ax[item] + t * (self.bx[item] - self.ax[item])
        cy = self.ay[item] + t * (self.by[item] - self.ay[item])
        dist = haversine(lat[point], lon[point], cy, cx, unit=Unit.METERS)
        near = (dist <= max_distance) & (t > 0) & (t < 1)
        pair = np.flatnonzero(near)
        best = _closest_per_point(
            n, point[pair], pair, dist[pair], self.way[item[pair]], self.pos[item[pair]]
        )
        found = best >= 0
        seg[found] = item[best[found]]
        x[found], y[found], frac[found] = (
            cx[best[found]],
            cy[best[found]],
            t[best[found]],
        )
        return seg, x, y, frac


class _Feature:
    """A feature of the data: the ``label`` that names its row in messages, its
    ``kind`` ("node", "line" or "area"), its geometry, and the ``id`` and node ids
    (``refs``) it gets."""

    def __init__(self, label, geometry):
        self.label = label
        self.geometry = geometry
        self.kind = self._kind()
        minx, miny, maxx, maxy = geometry.bounds
        if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
            raise ValueError(
                "add_elements: %s has coordinates outside the longitude and latitude "
                "ranges (bounds %s); a frame without a CRS must be in EPSG:4326."
                % (label, geometry.bounds)
            )
        self.id = None
        self.refs = None

    def _kind(self):
        """ "node" for a Point, "line" for a LineString and "area" for a Polygon without
        holes; other geometries raise, as ``write_pbf`` cannot write them as new
        elements."""
        geom = self.geometry
        if geom is None or geom.is_empty:
            raise ValueError("add_elements: %s has no geometry." % self.label)
        gtype = geom.geom_type
        if gtype == "Point":
            return "node"
        if gtype == "LineString":
            return "line"
        if gtype == "Polygon" and len(geom.interiors) == 0:
            return "area"
        raise ValueError(
            "add_elements: %s is a %s; new elements can be Points, LineStrings and "
            "Polygons without holes." % (self.label, gtype)
        )

    def vertices(self):
        """The (lon, lat) of each vertex of a line or area."""
        geom = self.geometry
        coords = geom.coords if self.kind == "line" else geom.exterior.coords
        return [tuple(c[:2]) for c in coords]


def _features(frames):
    """The :class:`_Feature` of each row of ``frames``, in order."""
    out = []
    for frame, gdf in enumerate(frames):
        for row, geom in enumerate(gdf.geometry):
            label = "row %d" % row
            if len(frames) > 1:
                label += " of frame %d" % frame
            out.append(_Feature(label, geom))
    return out


def _coordinate_key(x, y):
    """The key of the coordinates (``x``, ``y``) at the precision of a PBF file, as
    (latitude, longitude) in 1e-7 degrees; longitude -180 is +180."""
    lon = round(x * 1e7)
    return round(y * 1e7), 1_800_000_000 if lon == -1_800_000_000 else lon


class _NewNodes:
    """Gives out the ids of new nodes: a vertex at the coordinates of an earlier new
    Point or vertex gets that node."""

    def __init__(self, below):
        self.next_id = below - 1
        self.vertices = []  # (id, lon, lat) of the nodes that are not Points of data
        self._at = {}

    def _claim(self, x, y):
        """A new node id for a node at (``x``, ``y``)."""
        nid = self.next_id
        self.next_id -= 1
        self._at.setdefault(_coordinate_key(x, y), nid)
        return nid

    def point(self, x, y):
        """The id of the node of a Point of the data; each Point is a node of its own."""
        return self._claim(x, y)

    def vertex(self, x, y):
        """The id of the node at a vertex of a new line or area."""
        nid = self._at.get(_coordinate_key(x, y))
        if nid is None:
            nid = self._claim(x, y)
            self.vertices.append((nid, x, y))
        return nid


def _first_ids(node_coordinates, way_records, used_ids):
    """The (node, way) ids below which new ids start: below every id of the parsed file,
    every id ``used_ids`` names and zero."""
    node_ids = node_coordinates.ids
    low_node = int(node_ids.min()) if len(node_ids) else 0
    low_way = min((int(w["id"]) for w in way_records), default=0)
    return min(low_node, used_ids[0], 0), min(low_way, used_ids[1], 0)


def _fixed_points(features):
    """The coordinate keys of the new Points, the inner vertices of the new lines and the
    vertices of the new areas: the places of new nodes that do not snap."""
    keys = set()
    for feature in features:
        if feature.kind == "node":
            keys.add(_coordinate_key(feature.geometry.x, feature.geometry.y))
            continue
        xy = feature.vertices()
        if feature.kind == "line":
            xy = xy[1:-1]
        keys.update(_coordinate_key(x, y) for x, y in xy)
    return keys


_Snap = namedtuple("_Snap", "node x y segment frac")
_Snap.__doc__ = """Where an end of a new line snaps: to the existing ``node`` at (``x``,
``y``), or, with ``node`` None, to a new node at (``x``, ``y``), ``frac`` of the way
along the ``segment`` (see :class:`_Segments`)."""


def _snapped_ends(features, candidates, segments, snap_distance):
    """Where each end of each line snaps, as ``{(feature index, 0 or -1): _Snap}``: to
    the closest node of ``candidates`` within ``snap_distance`` metres, or when there is
    none, to the closest point of ``segments`` within it. An end at a new Point, an inner
    vertex of a new line or a vertex of a new area stays there, joined to that
    feature."""
    fixed = _fixed_points(features)
    ends = [
        (n, end)
        for n, feature in enumerate(features)
        if feature.kind == "line"
        for end in (0, -1)
        if _coordinate_key(*feature.geometry.coords[end][:2]) not in fixed
    ]
    # The ends at one place (one coordinate key) snap as the first of them does.
    coords = [features[n].geometry.coords[end][:2] for n, end in ends]
    first = {}
    for i, xy in enumerate(coords):
        first.setdefault(_coordinate_key(*xy), i)
    group = [first[_coordinate_key(*xy)] for xy in coords]
    leaders = sorted(first.values())
    xy = np.array([coords[i] for i in leaders], dtype=np.float64).reshape(-1, 2)
    nearest = _nearest_nodes(xy[:, 0], xy[:, 1], candidates, snap_distance)
    loose = np.flatnonzero(nearest < 0)
    seg, sx, sy, frac = segments.closest(xy[loose, 0], xy[loose, 1], snap_distance)
    snaps = {}
    for i, c in zip(leaders, nearest.tolist()):
        if c >= 0:
            ids, lon, lat = candidates
            snaps[i] = _Snap(int(ids[c]), float(lon[c]), float(lat[c]), None, None)
    for j, s in enumerate(seg.tolist()):
        if s >= 0:
            snaps[leaders[loose[j]]] = _Snap(
                None, float(sx[j]), float(sy[j]), s, float(frac[j])
            )
    return {end: snaps[g] for end, g in zip(ends, group) if g in snaps}


def _assign_ids(features, nodes, first_way, snapped):
    """Give each feature its id, and each line and area its node ids: new nodes from
    ``nodes``, and at the ``snapped`` ends (see :func:`_snapped_ends`) the existing node
    or a new node on a segment. Returns the lowest way id given out, and the new nodes on
    segments as ``{segment: {node id: fraction along the segment}}``."""
    for feature in features:
        if feature.kind == "node":
            geom = feature.geometry
            feature.id = nodes.point(geom.x, geom.y)
    way_id = first_way
    on_segments = {}
    for n, feature in enumerate(features):
        if feature.kind == "node":
            continue
        xy = feature.vertices()
        ends = {}
        for end in (0, len(xy) - 1):
            snap = snapped.get((n, 0 if end == 0 else -1))
            if snap is None:
                continue
            xy[end] = (snap.x, snap.y)
            if snap.node is not None:
                ends[end] = snap.node
            else:
                ends[end] = nodes.vertex(snap.x, snap.y)
                on_segments.setdefault(snap.segment, {})[ends[end]] = snap.frac
        refs = [
            ends[i] if i in ends else nodes.vertex(x, y) for i, (x, y) in enumerate(xy)
        ]
        if len(set(refs)) < 2:
            raise ValueError(
                "add_elements: %s would be a way over one node (%d): its ends snap to "
                "one node, or its points are less than 1e-7 degrees apart."
                % (feature.label, refs[0])
            )
        if ends:
            feature.geometry = shapely.LineString(xy)
        way_id -= 1
        feature.id = way_id
        feature.refs = refs
    return way_id, on_segments


def _split_ways(on_segments, segments, way_by_id):
    """The node ids of the ways that get the new nodes ``on_segments`` (see
    :func:`_assign_ids`), as ``{way id: node ids}``: each new node goes between the two
    nodes of its segment in every way of ``segments`` with that segment, in order along
    it."""
    by_pair = {}  # (smaller node, larger node) -> {node id: fraction from the smaller}
    for s, new in on_segments.items():
        a, b = int(segments.a[s]), int(segments.b[s])
        pair = (min(a, b), max(a, b))
        for nid, frac in new.items():
            by_pair.setdefault(pair, {})[nid] = frac if a < b else 1.0 - frac
    inserts = {}  # (way id, position) -> node ids to put after that position
    for (low, high), new in by_pair.items():
        along = sorted(new, key=new.get)
        hits = ((segments.a == low) & (segments.b == high)) | (
            (segments.a == high) & (segments.b == low)
        )
        for s in np.flatnonzero(hits).tolist():
            ordered = along if segments.a[s] == low else along[::-1]
            inserts[(int(segments.way[s]), int(segments.pos[s]))] = ordered
    split = {}
    for wid in sorted({w for w, _ in inserts}):
        refs = [int(r) for r in way_by_id[wid]["nodes"]]
        out = []
        for i, ref in enumerate(refs):
            out.append(ref)
            out.extend(inserts.get((wid, i), ()))
        split[wid] = out
    return split


def _way_rows(split, way_by_id, node_coordinates, new_xy):
    """A row per way of ``split`` (see :func:`_split_ways`): its id, node ids, tags and
    a LineString through those of its nodes the parsed file or ``new_xy`` (``{node id:
    (lon, lat)}``) holds."""
    rows = []
    for wid, refs in split.items():
        ids = np.asarray(refs, dtype=np.int64)
        idx, lon, lat = node_coordinates.gather(ids)
        xy = []
        for nid, i, x, y in zip(refs, idx.tolist(), lon.tolist(), lat.tolist()):
            if i >= 0:
                xy.append((x, y))
            elif nid in new_xy:
                xy.append(new_xy[nid])
        rows.append((wid, refs, _record_tags(way_by_id[wid]), shapely.LineString(xy)))
    return rows


def _field_columns(ids, osm_types, refs):
    """A frame of the element fields: the ``id``, ``osm_type`` and ``nodes`` of each
    element."""
    return pd.DataFrame(
        {
            "id": np.array(ids, dtype=np.int64),
            "osm_type": list(osm_types),
            "nodes": pd.Series(list(refs), dtype=object),
        }
    )


def _elements_frame(frames, features, way_rows, vertices):
    """The frame of the new elements: a row per feature with its columns other than the
    replaced ones (see ``_REPLACED_COLS``), a row per way of ``way_rows`` (see
    :func:`_way_rows`) with all its tags, in the tag columns or else in the JSON
    ``tags`` column, then a Point row per new vertex node."""
    tables, start = [], 0
    for gdf in frames:
        part = features[start : start + len(gdf)]
        start += len(gdf)
        fields = _field_columns(
            [f.id for f in part],
            ["node" if f.kind == "node" else "way" for f in part],
            [f.refs for f in part],
        )
        dropped = [gdf.geometry.name] + [c for c in gdf.columns if c in _REPLACED_COLS]
        tags = pd.DataFrame(gdf.drop(columns=dropped)).reset_index(drop=True)
        tables.append(pd.concat([fields, tags], axis=1))
    if way_rows:
        columns = {
            _tag_key(col): col
            for table in tables
            for col in table.columns[3:]
            if col != "tags" and _tag_key(col) is not None
        }
        records = []
        for wid, refs, tags, _ in way_rows:
            record = {"id": wid, "osm_type": "way", "nodes": refs}
            rest = {}
            for key, value in tags.items():
                if key in columns:
                    record[columns[key]] = value
                else:
                    rest[key] = value
            record["tags"] = rest or None
            records.append(record)
        tables.append(pd.DataFrame(records))
    count = len(vertices)
    tables.append(
        _field_columns([v[0] for v in vertices], ["node"] * count, [None] * count)
    )
    geometry = [f.geometry for f in features] + [row[3] for row in way_rows]
    geometry += [shapely.Point(x, y) for _, x, y in vertices]
    return GeoDataFrame(
        pd.concat(tables, ignore_index=True), geometry=geometry, crs="EPSG:4326"
    )


def new_elements(
    data,
    node_coordinates,
    way_records,
    snap_to=None,
    snap_distance=None,
    used_ids=(0, 0),
):
    """The features of ``data`` (a GeoDataFrame or a list of them) as new OSM elements,
    and the lowest (node, way) ids they use.

    A Point becomes a node, a LineString a way and a Polygon without holes a closed way,
    with the feature's columns as tags and new negative ids below the ids of the parsed
    file (``node_coordinates`` and ``way_records``) and below ``used_ids``. The vertices
    of the lines and areas become new nodes, shared with the new Points and vertices at
    the same coordinates. With ``snap_to``, the first and last vertex of each line is the
    closest node of ``snap_to`` (see :func:`_snap_targets`) within ``snap_distance``
    metres, or when there is none, a new node at the closest point within it on a
    segment of a ``snap_to`` way, inserted into every ``snap_to`` way along that segment;
    a vertex at a new Point, an inner vertex of a new line or a vertex of a new area does
    not snap. The frame has a row per feature, whose ``nodes`` are the node ids of a way,
    a row per existing way that got a new node, and a Point row per new vertex node. The
    arguments are checked by :func:`validate_inputs`.
    """
    frames = _reproject_to_wgs84(_as_frames(data))
    features = _features(frames)
    first_node, first_way = _first_ids(node_coordinates, way_records, used_ids)
    snapped, segments, way_by_id = {}, None, None
    if snap_to is not None:
        way_by_id = {int(w["id"]): w for w in way_records}
        way_ids, node_ids = _snap_targets(snap_to, way_by_id)
        candidates = _snap_candidates(way_ids, node_ids, way_by_id, node_coordinates)
        segments = _Segments(way_ids, way_by_id, node_coordinates)
        snapped = _snapped_ends(features, candidates, segments, snap_distance)
    nodes = _NewNodes(first_node)
    low_way, on_segments = _assign_ids(features, nodes, first_way, snapped)
    way_rows = []
    if on_segments:
        split = _split_ways(on_segments, segments, way_by_id)
        new_xy = {nid: (x, y) for nid, x, y in nodes.vertices}
        way_rows = _way_rows(split, way_by_id, node_coordinates, new_xy)
    frame = _elements_frame(frames, features, way_rows, nodes.vertices)
    return frame, (nodes.next_id + 1, low_way)
