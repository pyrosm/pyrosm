"""Assemble OSM records from (possibly modified) GeoDataFrames and write them to
a PBF (issue #285).

``pyrosm.OSM.write_pbf`` is a thin wrapper over :func:`write_geodataframe_to_pbf`
here; the per-row tag extraction, the whole-dataset record assembly, and the
vertex-synthesis of new geometries (approach B) all live in this module so
``pyrosm.py`` stays at a high level of abstraction. The low-level PBF block
serialization lives in the Cython ``pyrosm.pbf_export`` module.
"""

import json
import time

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from pyrosm.pbf_export import write_pbf_from_records

# Columns that are structural attributes / pyrosm-computed, never OSM tags.
_NON_TAG_COLS = {
    "id",
    "osm_type",
    "version",
    "timestamp",
    "changeset",
    "visible",
    "length",
    "tags",
    "nodes",
    "members",
    "lon",
    "lat",
    "geometry",
}


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------
def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _tag_str(value):
    """Render a tag value as the string OSM stores."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _tag_key(col):
    """Coerce a column/tag name to an OSM tag key, or None to skip it.

    OSM tag keys are non-empty strings; arbitrary GeoDataFrame columns can be
    non-string or empty, which would corrupt the string-table / keys_vals output.
    """
    key = col if isinstance(col, str) else str(col)
    return key if key != "" else None


def _row_tags(row, geom_col):
    """Tags for a GeoDataFrame row: non-structural columns + the JSON tags column."""
    tags = {}
    for col, val in row.items():
        if col == geom_col or col in _NON_TAG_COLS:
            continue
        if _is_missing(val):
            continue
        key = _tag_key(col)
        if key is None or key in _NON_TAG_COLS:
            continue
        tags[key] = _tag_str(val)
    extra = row.get("tags")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (ValueError, TypeError):
            extra = None
    if isinstance(extra, dict):
        for k, v in extra.items():
            key = _tag_key(k)
            if key is None or key in _NON_TAG_COLS or key in tags or _is_missing(v):
                continue
            tags[key] = _tag_str(v)
    return tags


def _record_tags(record):
    """Tags for a cached way record: its tag columns + any leftover tags dict."""
    tags = {}
    for col, val in record.items():
        if col in _NON_TAG_COLS:
            continue
        if _is_missing(val):
            continue
        key = _tag_key(col)
        if key is None or key in _NON_TAG_COLS:
            continue
        tags[key] = _tag_str(val)
    extra = record.get("tags")
    if isinstance(extra, dict):
        for k, v in extra.items():
            key = _tag_key(k)
            if key is None or key in _NON_TAG_COLS or key in tags or _is_missing(v):
                continue
            tags[key] = _tag_str(v)
    return tags


# ---------------------------------------------------------------------------
# Frame normalization + edit/new-row collection
# ---------------------------------------------------------------------------
def _as_frames(data):
    if isinstance(data, GeoDataFrame):
        return [data]
    if isinstance(data, (list, tuple)):
        return list(data)
    raise ValueError("'data' should be a GeoDataFrame or a list of GeoDataFrames.")


def _reproject_to_wgs84(frames):
    """Reproject CRS-tagged frames to EPSG:4326 (new geometries are lon/lat).

    ``CRS.to_epsg()`` returns ``None`` (not 4326) for a CRS without an EPSG code,
    so such a frame is reprojected too; reprojecting an already-WGS84 frame is a
    harmless no-op.
    """
    out = []
    for gdf in frames:
        crs = getattr(gdf, "crs", None)
        if crs is not None and crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        out.append(gdf)
    return out


def _normalize_osm_type(otype):
    if isinstance(otype, bytes):
        otype = otype.decode("utf-8", "replace")
    if isinstance(otype, str):
        otype = otype.lower()
    return otype


def _normalize_id(oid):
    try:
        return int(oid)
    except (TypeError, ValueError):
        return None


def _infer_osm_type(geom):
    """Best-effort OSM type for a row that lacks an ``osm_type`` column.

    Some returned frames omit ``osm_type`` -- notably the node frame from
    ``get_network(nodes=True)`` -- so without this their existing elements would not
    match the cache and would be re-synthesized as duplicate new nodes. Inferred from
    the geometry kind: ``Point`` -> node, ``LineString``/``Polygon`` -> way.
    """
    gt = getattr(geom, "geom_type", None)
    if gt == "Point":
        return "node"
    if gt in ("LineString", "Polygon"):
        return "way"
    return None


def _collect_edits(frames, way_by_id, node_coordinates, rel_ids):
    """Split frame rows into tag edits (matched by osm_type+id) and new rows."""
    node_edits, way_edits, rel_edits = {}, {}, {}
    new_rows = []
    for gdf in frames:
        geom_col = gdf.geometry.name
        for _, row in gdf.iterrows():
            otype = _normalize_osm_type(row.get("osm_type"))
            oid = _normalize_id(row.get("id"))
            if otype is None:
                otype = _infer_osm_type(row[geom_col])
            tags = _row_tags(row, geom_col)
            if oid is not None and otype == "way" and oid in way_by_id:
                way_edits[oid] = tags
            elif oid is not None and otype == "node" and oid in node_coordinates:
                node_edits[oid] = tags
            elif oid is not None and otype == "relation" and oid in rel_ids:
                rel_edits[oid] = tags
            else:
                new_rows.append((oid, row[geom_col], tags))
    return node_edits, way_edits, rel_edits, new_rows


def _subset_keep_sets(node_edits, way_edits, rel_edits, way_by_id, relations):
    """Closure of the matched elements over the cache, for ``subset_only``.

    Starts from the elements matched in the frame(s) and pulls in their references so
    the written PBF stays valid: kept relations add their member ways/nodes (recursing
    into sub-relations), then kept ways add their node refs. Only cache-present members
    are added (a member absent from the cache has no record to emit); relations are
    still written with their full original member list by ``_add_base_relations``.
    Returns ``(keep_nodes, keep_ways, keep_rels)`` as sets of ids.
    """
    keep_nodes = set(node_edits)
    keep_ways = set(way_edits)
    keep_rels = set(rel_edits)

    members_by_rid = {}
    if "id" in relations:
        rid_arr, mem_arr = relations["id"], relations["members"]
        for i in range(len(rid_arr)):
            members_by_rid[int(rid_arr[i])] = mem_arr[i]

    # Relations -> members, to a fixed point so super-relations resolve.
    pending = list(keep_rels)
    while pending:
        mem = members_by_rid.get(pending.pop())
        if mem is None:
            continue
        mtypes, mids = mem["member_type"], mem["member_id"]
        for j in range(len(mids)):
            mtype = mtypes[j]
            if isinstance(mtype, (bytes, bytearray)):
                mtype = mtype.decode()
            mid = int(mids[j])
            if mtype == "way":
                keep_ways.add(mid)
            elif mtype == "node":
                keep_nodes.add(mid)
            elif mtype == "relation" and mid in members_by_rid and mid not in keep_rels:
                keep_rels.add(mid)
                pending.append(mid)

    # Ways -> node refs (direct + pulled in via relations).
    for wid in keep_ways:
        w = way_by_id.get(wid)
        if w is not None:
            keep_nodes.update(int(n) for n in w["nodes"])

    return keep_nodes, keep_ways, keep_rels


# ---------------------------------------------------------------------------
# Record builder (base records + approach-B synthesis)
# ---------------------------------------------------------------------------
def _check_lonlat(x, y, oid):
    if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
        raise ValueError(
            "write_pbf: row id %r has coordinates (%s, %s) outside valid lon/lat "
            "ranges; new geometries must be in EPSG:4326." % (oid, x, y)
        )


class _RecordBuilder:
    """Accumulates node/way/relation records and synthesizes new geometries.

    New (approach-B) elements get decreasing negative ids starting below the
    minimum existing id in each namespace (so re-writing a file that already holds
    synthesized ids does not collide); coincident new vertices share one node.
    """

    def __init__(self):
        self.node_ids, self.lats, self.lons = [], [], []
        self.vers, self.tss, self.css, self.ntags = [], [], [], []
        self.ways = []
        self.rels = []
        self._now = int(time.time())
        self._coord_to_node = {}
        self._coord_to_index = {}
        self._node_counter = -1
        self._way_counter = -1

    def add_node(self, nid, lat, lon, version, timestamp, changeset, tags):
        self.node_ids.append(nid)
        self.lats.append(lat)
        self.lons.append(lon)
        self.vers.append(version)
        self.tss.append(timestamp)
        self.css.append(changeset)
        self.ntags.append(tags)

    def begin_synthesis(self):
        min_node = min(self.node_ids) if self.node_ids else 0
        min_way = min((w["id"] for w in self.ways), default=0)
        self._node_counter = min(-1, min_node - 1)
        self._way_counter = min(-1, min_way - 1)

    def _node_for(self, x, y, oid, tags=None):
        _check_lonlat(x, y, oid)
        key = (round(y * 1e7), round(x * 1e7))
        nid = self._coord_to_node.get(key)
        if nid is None:
            nid = self._node_counter
            self._node_counter -= 1
            self._coord_to_node[key] = nid
            self._coord_to_index[key] = len(self.node_ids)
            self.add_node(nid, y, x, 1, self._now, 0, tags)
        elif tags is not None:
            # A tagged Point coincides with an already-synthesized node; attach
            # its tags to that shared node (last-wins).
            self.ntags[self._coord_to_index[key]] = tags
        return nid

    def _add_way(self, refs, tags):
        wid = self._way_counter
        self._way_counter -= 1
        self.ways.append(
            {
                "id": wid,
                "refs": refs,
                "version": 1,
                "timestamp": self._now,
                "tags": tags,
            }
        )

    def add_geometry(self, oid, geom, tags):
        if geom is None or geom.is_empty:
            raise ValueError(
                "write_pbf: row id %r has no (or empty) geometry to synthesize a "
                "new element from." % oid
            )
        # A non-empty shapely LineString has >= 2 coords and a Polygon ring >= 4,
        # so the empty/None check above is the only degeneracy guard needed.
        gtype = geom.geom_type
        if gtype == "Point":
            self._node_for(geom.x, geom.y, oid, tags)
        elif gtype == "LineString":
            refs = [self._node_for(c[0], c[1], oid) for c in geom.coords]
            self._add_way(refs, tags)
        elif gtype == "Polygon" and len(geom.interiors) == 0:
            refs = [self._node_for(c[0], c[1], oid) for c in geom.exterior.coords]
            self._add_way(refs, tags)
        else:
            raise ValueError(
                "write_pbf cannot synthesize a new element from geometry type "
                "'%s' (row id %r). Only Point, LineString and hole-less Polygon are "
                "supported for new features in this version." % (gtype, oid)
            )

    def node_payload(self):
        return {
            "id": np.asarray(self.node_ids, dtype=np.int64),
            "lat": np.asarray(self.lats, dtype=np.float64),
            "lon": np.asarray(self.lons, dtype=np.float64),
            "version": np.asarray(self.vers, dtype=np.int64),
            "timestamp": np.asarray(self.tss, dtype=np.int64),
            "changeset": np.asarray(self.css, dtype=np.int64),
            "tags": self.ntags,
        }

    def bounds(self):
        if not self.lons:  # empty subset -> degenerate header bbox
            return (0.0, 0.0, 0.0, 0.0)
        lons = np.asarray(self.lons, dtype=np.float64)
        lats = np.asarray(self.lats, dtype=np.float64)
        return (
            float(lons.min()),
            float(lats.min()),
            float(lons.max()),
            float(lats.max()),
        )


def _node_tag_records(nodes_cache):
    """Index standalone/POI node tags from the node records by node id."""
    out = {}
    nd = nodes_cache or {}
    if "id" in nd and "tags" in nd:
        nd_ids, nd_tags = nd["id"], nd["tags"]
        for i in range(len(nd_ids)):
            t = nd_tags[i]
            if isinstance(t, dict):
                out[int(nd_ids[i])] = t
    return out


def _add_base_nodes(builder, node_coordinates, nodes_cache, node_edits, keep=None):
    poi_tags = _node_tag_records(nodes_cache)
    for nid, rec in node_coordinates.items():
        if keep is not None and nid not in keep:
            continue
        if nid in node_edits:
            tags = node_edits[nid]
        else:
            tags = poi_tags.get(nid)
            if tags is None:
                tags = rec.get("tags")
            tags = tags if isinstance(tags, dict) else None
        builder.add_node(
            nid,
            rec["lat"],
            rec["lon"],
            int(rec.get("version") or 1),
            int(rec.get("timestamp") or 0),
            int(rec.get("changeset") or 0),
            tags,
        )


def _add_base_ways(builder, way_records, way_edits, keep=None):
    for w in way_records:
        wid = w["id"]
        if keep is not None and wid not in keep:
            continue
        builder.ways.append(
            {
                "id": wid,
                "refs": list(w["nodes"]),
                "version": w.get("version") or 1,
                "timestamp": w.get("timestamp"),
                "tags": way_edits.get(wid, _record_tags(w)),
            }
        )


def _add_base_relations(builder, relations, rel_edits, keep=None):
    if "id" not in relations:
        return
    for i in range(len(relations["id"])):
        rid = int(relations["id"][i])
        if keep is not None and rid not in keep:
            continue
        mem = relations["members"][i]
        members = [
            (mem["member_type"][j], int(mem["member_id"][j]), mem["member_role"][j])
            for j in range(len(mem["member_id"]))
        ]
        rtags = relations["tags"][i]
        builder.rels.append(
            {
                "id": rid,
                "members": members,
                "version": (
                    int(relations["version"][i]) if "version" in relations else 1
                ),
                "timestamp": (
                    int(relations["timestamp"][i]) if "timestamp" in relations else None
                ),
                "changeset": (
                    int(relations["changeset"][i]) if "changeset" in relations else None
                ),
                "tags": rel_edits.get(rid, rtags if isinstance(rtags, dict) else {}),
            }
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def write_geodataframe_to_pbf(
    data,
    output_path,
    node_coordinates,
    way_records,
    relations,
    nodes,
    subset_only=False,
):
    """Write `data` (+ the cached dataset) to a valid PBF at `output_path`.

    `node_coordinates` / `way_records` / `relations` / `nodes` are the caches the
    `OSM` object holds after reading; `data` is a GeoDataFrame or list of them
    whose tag edits are applied by ``osm_type``+``id`` (new rows are synthesized).

    When ``subset_only`` is True, only the elements present in `data` are written
    (matched by ``osm_type``+``id``), together with the references they need to stay
    valid -- kept ways pull in their nodes and kept relations pull in their member
    ways/nodes (the union across all frames). New rows are still synthesized.
    """
    relations = relations or {}
    frames = _reproject_to_wgs84(_as_frames(data))

    way_by_id = {w["id"]: w for w in way_records}
    rel_ids = set(int(i) for i in relations["id"]) if "id" in relations else set()

    node_edits, way_edits, rel_edits, new_rows = _collect_edits(
        frames, way_by_id, node_coordinates, rel_ids
    )

    keep_nodes = keep_ways = keep_rels = None
    if subset_only:
        keep_nodes, keep_ways, keep_rels = _subset_keep_sets(
            node_edits, way_edits, rel_edits, way_by_id, relations
        )

    builder = _RecordBuilder()
    _add_base_nodes(builder, node_coordinates, nodes, node_edits, keep_nodes)
    _add_base_ways(builder, way_records, way_edits, keep_ways)
    _add_base_relations(builder, relations, rel_edits, keep_rels)

    builder.begin_synthesis()
    for oid, geom, tags in new_rows:
        builder.add_geometry(oid, geom, tags)

    return write_pbf_from_records(
        builder.node_payload(),
        builder.ways,
        builder.rels,
        output_path,
        builder.bounds(),
    )
