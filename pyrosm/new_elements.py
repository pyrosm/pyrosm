"""New OSM elements from the features of GeoDataFrames, in the form
``pyrosm.OSM.write_pbf(..., apply_geometry=True)`` writes, with the ends of new lines
joined to existing nodes (``pyrosm.OSM.add_elements``)."""

import math
import numbers

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
    _reproject_to_wgs84,
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


def _snap_candidates(snap_to, way_by_id, node_coordinates):
    """The ids, longitudes and latitudes of the nodes of the elements of ``snap_to``
    (the nodes of its ways, and its nodes) that the parsed file holds, sorted by id."""
    missing = [None] * len(snap_to)
    otypes = snap_to["osm_type"] if "osm_type" in snap_to.columns else missing
    oids = snap_to["id"] if "id" in snap_to.columns else missing
    ids = []
    for otype, oid, geom in zip(otypes, oids, snap_to.geometry):
        if isinstance(otype, (str, bytes)):
            otype = _normalize_osm_type(otype)
        else:
            otype = _infer_osm_type(geom)
        oid = _normalize_id(oid)
        if otype == "way" and oid in way_by_id:
            ids.extend(way_by_id[oid]["nodes"])
        elif otype == "node" and oid is not None:
            ids.append(oid)
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    idx, lon, lat = node_coordinates.gather(ids)
    found = idx >= 0
    if not found.any():
        raise ValueError(
            "add_elements: none of the elements of 'snap_to' are in the data this OSM "
            "object reads, so there are no nodes to snap to."
        )
    return ids[found], lon[found], lat[found]


def _nearest_nodes(lon, lat, candidates, max_distance):
    """For each point (``lon``, ``lat``), the index into ``candidates`` (see
    :func:`_snap_candidates`) of the closest node within ``max_distance`` metres, or -1.
    Of nodes at the same distance, the one with the smallest id is chosen."""
    _, cand_lon, cand_lat = candidates
    nearest = np.full(len(lon), -1, dtype=np.int64)
    if len(lon) == 0:
        return nearest
    tree = shapely.STRtree(shapely.points(cand_lon, cand_lat))
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
    point = np.concatenate([p for p, _ in pairs])
    cand = np.concatenate([c for _, c in pairs])
    dist = haversine(
        lat[point], lon[point], cand_lat[cand], cand_lon[cand], unit=Unit.METERS
    )
    near = dist <= max_distance
    point, cand, dist = point[near], cand[near], dist[near]
    order = np.lexsort((cand, dist, point))
    point, cand = point[order], cand[order]
    first = np.ones(len(point), dtype=bool)
    first[1:] = point[1:] != point[:-1]
    nearest[point[first]] = cand[first]
    return nearest


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


def _snapped_ends(features, candidates, snap_distance):
    """The node of ``candidates`` that each end of each line snaps to, as
    ``{(feature index, 0 or -1): candidate index}``. An end at a new Point, an inner
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
    snapped = dict(zip(leaders, nearest.tolist()))
    return {end: snapped[g] for end, g in zip(ends, group) if snapped[g] >= 0}


def _assign_ids(features, nodes, first_way, snapped, candidates):
    """Give each feature its id, and each line and area its node ids: new nodes from
    ``nodes``, and at the ``snapped`` ends the nodes of ``candidates``. Returns the
    lowest way id given out."""
    for feature in features:
        if feature.kind == "node":
            geom = feature.geometry
            feature.id = nodes.point(geom.x, geom.y)
    way_id = first_way
    for n, feature in enumerate(features):
        if feature.kind == "node":
            continue
        xy = feature.vertices()
        ends = {}
        for end in (0, len(xy) - 1):
            c = snapped.get((n, 0 if end == 0 else -1))
            if c is not None:
                ends[end] = int(candidates[0][c])
                xy[end] = (float(candidates[1][c]), float(candidates[2][c]))
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
    return way_id


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


def _elements_frame(frames, features, vertices):
    """The frame of the new elements: a row per feature with its columns other than the
    replaced ones (see ``_REPLACED_COLS``), then a Point row per new vertex node."""
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
    count = len(vertices)
    tables.append(
        _field_columns([v[0] for v in vertices], ["node"] * count, [None] * count)
    )
    geometry = [f.geometry for f in features]
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
    closest node of ``snap_to`` (see :func:`_snap_candidates`) within ``snap_distance``
    metres, when there is one and the vertex is not at a new Point, an inner vertex of a
    new line or a vertex of a new area. The frame has a row per feature, whose ``nodes``
    are the node ids of a way, and then a Point row per new vertex node. The arguments
    are checked by :func:`validate_inputs`.
    """
    frames = _reproject_to_wgs84(_as_frames(data))
    features = _features(frames)
    first_node, first_way = _first_ids(node_coordinates, way_records, used_ids)
    snapped, candidates = {}, None
    if snap_to is not None:
        way_by_id = {int(w["id"]): w for w in way_records}
        candidates = _snap_candidates(snap_to, way_by_id, node_coordinates)
        snapped = _snapped_ends(features, candidates, snap_distance)
    nodes = _NewNodes(first_node)
    low_way = _assign_ids(features, nodes, first_way, snapped, candidates)
    frame = _elements_frame(frames, features, nodes.vertices)
    return frame, (nodes.next_id + 1, low_way)
