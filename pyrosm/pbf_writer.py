"""Assemble OSM records from (possibly modified) GeoDataFrames and write them to
a PBF (issue #285).

``pyrosm.OSM.write_pbf`` is a thin wrapper over :func:`write_geodataframe_to_pbf`
here; the per-row tag extraction, the whole-dataset record assembly, and the
vertex-synthesis of new geometries (approach B) all live in this module so
``pyrosm.py`` stays at a high level of abstraction. The low-level PBF block
serialization lives in the Cython ``pyrosm.pbf_export`` module.

Geometry edits follow the OSM model: coordinates live on nodes, and a way is an
ordered list of member node ids. Moving a node therefore moves every way that
references it, and an existing way is reshaped through its ``nodes`` member list
rather than through its (derived) LineString.
"""

import json
import time
import warnings

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


def _row_refs(row):
    """Ordered member node ids from a row's ``nodes`` column, or None when absent."""
    refs = row.get("nodes")
    if refs is None or not isinstance(refs, (list, tuple, np.ndarray)):
        return None
    if len(refs) == 0:
        return None
    return [int(r) for r in refs]


def _set_geometry_edit(mapping, oid, value, what):
    """Record a geometry edit, rejecting two conflicting edits of the same element.

    Tag edits are last-wins (frames legitimately carry different tag columns for the
    same element), but two different coordinates or member lists for one element are
    a topology conflict with no safe resolution. This also catches the segment-level
    edge frame from ``get_network(nodes=True)``, where each way spans several rows.
    """
    previous = mapping.get(oid)
    if previous is not None and previous != value:
        raise ValueError(
            "write_pbf: conflicting %s for id %d (%r and %r). Every row of an "
            "element must carry the same geometry." % (what, oid, previous, value)
        )
    mapping[oid] = value


class _Edits:
    """Per-element edits and additions collected from the input frame(s)."""

    def __init__(self):
        self.node_tags, self.node_coords = {}, {}
        self.way_tags, self.way_refs = {}, {}
        self.rel_tags = {}
        self.new_nodes = []  # (id, lon, lat, tags)
        self.new_ways = []  # (id, refs, tags)
        self.auto_rows = []  # (row id, geometry, tags, explicit id or None)


def _claim_new_id(seen, kind, oid):
    if (kind, oid) in seen:
        raise ValueError(
            "write_pbf: new %s id %d appears more than once." % (kind, oid)
        )
    seen.add((kind, oid))


def _add_new_element(edits, oid, otype, geom, tags, refs, seen):
    """Queue a row with a caller-chosen (negative) id as a new element.

    A ``nodes`` member list makes the row a way over existing/added nodes; otherwise
    a ``Point`` becomes a node and a line/ring becomes a way whose vertices are
    synthesized like an id-less row's. Ids are unique per element type, as in OSM, so
    a new node and a new way may both be -1.
    """
    gtype = getattr(geom, "geom_type", None)
    if otype == "node" or (otype is None and gtype == "Point"):
        _claim_new_id(seen, "node", oid)
        if gtype != "Point":
            raise ValueError(
                "write_pbf: new node id %d needs a Point geometry, got %r."
                % (oid, gtype)
            )
        _check_lonlat(geom.x, geom.y, oid)
        edits.new_nodes.append((oid, geom.x, geom.y, tags))
    elif refs is not None:
        _claim_new_id(seen, "way", oid)
        edits.new_ways.append((oid, refs, tags))
    else:
        _claim_new_id(seen, "way", oid)
        edits.auto_rows.append((oid, geom, tags, oid))


def _collect_edits(frames, way_by_id, node_coordinates, rel_ids, apply_geometry):
    """Split frame rows into edits of existing elements and new elements.

    Existing elements are matched by ``osm_type`` + ``id``; their tags are always
    taken from the row. With ``apply_geometry`` a node row's ``Point`` also moves the
    node, a way row's ``nodes`` column also reshapes the way, and a row with a
    negative unmatched id is added under exactly that id (so a way can reference a
    vertex added in the same call). Otherwise unmatched rows are synthesized with
    generated ids, as they always have been.
    """
    edits = _Edits()
    seen_new = set()
    for gdf in frames:
        geom_col = gdf.geometry.name
        for _, row in gdf.iterrows():
            otype = _normalize_osm_type(row.get("osm_type"))
            oid = _normalize_id(row.get("id"))
            geom = row[geom_col]
            if otype is None:
                otype = _infer_osm_type(geom)
            tags = _row_tags(row, geom_col)
            refs = _row_refs(row) if apply_geometry else None
            if oid is not None and otype == "way" and oid in way_by_id:
                edits.way_tags[oid] = tags
                if refs is not None:
                    _set_geometry_edit(edits.way_refs, oid, refs, "member node ids")
            elif oid is not None and otype == "node" and oid in node_coordinates:
                edits.node_tags[oid] = tags
                if apply_geometry and getattr(geom, "geom_type", None) == "Point":
                    _check_lonlat(geom.x, geom.y, oid)
                    _set_geometry_edit(
                        edits.node_coords, oid, (geom.x, geom.y), "coordinates"
                    )
            elif oid is not None and otype == "relation" and oid in rel_ids:
                edits.rel_tags[oid] = tags
            elif apply_geometry and oid is not None and oid < 0:
                _add_new_element(edits, oid, otype, geom, tags, refs, seen_new)
            else:
                edits.auto_rows.append((oid, geom, tags, None))
    return edits


def _subset_keep_sets(node_edits, way_edits, rel_edits, way_refs, relations):
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

    # Ways -> node refs (direct + pulled in via relations), as reshaped.
    for wid in keep_ways:
        refs = way_refs.get(wid)
        if refs is not None:
            keep_nodes.update(refs)

    return keep_nodes, keep_ways, keep_rels


# ---------------------------------------------------------------------------
# Deletions, reshaping and referential integrity
# ---------------------------------------------------------------------------
_ORPHAN_NODE_POLICIES = ("remove", "keep", "error")
_DELETED_MEMBER_POLICIES = ("drop", "error")


def _validate_policy(name, value, allowed):
    if value not in allowed:
        raise ValueError(
            "write_pbf: '%s' should be one of %s, got %r."
            % (name, ", ".join(repr(a) for a in allowed), value)
        )


def _parse_deletions(delete, node_coordinates, way_by_id, rel_ids):
    """Split ``delete`` into per-type id sets, rejecting ids not in the source."""
    del_nodes, del_ways, del_rels = set(), set(), set()
    for entry in delete or ():
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                "write_pbf: 'delete' entries should be (osm_type, id) pairs, got %r."
                % (entry,)
            )
        otype = _normalize_osm_type(entry[0])
        oid = _normalize_id(entry[1])
        if oid is not None and otype == "node" and oid in node_coordinates:
            del_nodes.add(oid)
        elif oid is not None and otype == "way" and oid in way_by_id:
            del_ways.add(oid)
        elif oid is not None and otype == "relation" and oid in rel_ids:
            del_rels.add(oid)
        else:
            raise ValueError(
                "write_pbf: cannot delete (%r, %r); no such element in the source data."
                % (entry[0], entry[1])
            )
    return del_nodes, del_ways, del_rels


class _KnownNodes:
    """Membership test over the source nodes plus the nodes this call adds."""

    def __init__(self, node_coordinates, added_ids):
        self._source = node_coordinates
        self._added = added_ids

    def __contains__(self, nid):
        return nid in self._added or nid in self._source


def _check_new_refs(refs, original, known_nodes, owner):
    """Reject member ids an edit introduces that no node in the output will have.

    Only ids the edit adds are checked: a source way whose refs already point outside
    a clipped extract is re-emitted as it was read.
    """
    unchanged = set(original)
    missing = sorted({r for r in refs if r not in unchanged and r not in known_nodes})
    if missing:
        raise ValueError(
            "write_pbf: %s references node id(s) %s that are neither in the source "
            "data nor added by this call." % (owner, missing[:5])
        )


def _drop_deleted_nodes(refs, del_nodes, policy, owner):
    """Member ids with the deleted nodes removed, per ``on_deleted_member``."""
    if not del_nodes:
        return refs
    hits = sorted({r for r in refs if r in del_nodes})
    if not hits:
        return refs
    if policy == "error":
        raise ValueError(
            "write_pbf: %s still references deleted node(s) %s. Pass "
            "on_deleted_member='drop' to remove them from its member list."
            % (owner, hits[:5])
        )
    return [r for r in refs if r not in del_nodes]


class _WritePlan:
    """Which cached elements the write emits, and with what geometry.

    ``way_refs`` holds the effective member list of every cached way that is written
    and ``node_coords`` the moved node coordinates; the ``removed_*`` sets are the
    cached elements the write omits. ``rel_members`` overrides the member list of the
    relations that lost a member.
    """

    def __init__(self, edits, del_nodes):
        self.node_tags = edits.node_tags
        self.node_coords = edits.node_coords
        self.way_tags = edits.way_tags
        self.rel_tags = edits.rel_tags
        self.way_refs = {}
        self.rel_members = {}
        self.removed_nodes = set(del_nodes)
        self.removed_ways = set()
        self.removed_rels = set()
        self.referenced_nodes = set()
        self.orphan_candidates = set()


def _plan_ways(plan, way_records, edits, del_nodes, del_ways, policy, known_nodes):
    """Resolve the member list of every cached way, dropping the ones left degenerate.

    A way with fewer than two members cannot form a geometry, so it is not written and
    its original members join the orphan candidates along with the deleted ways'.
    """
    degenerate = []
    for w in way_records:
        wid = w["id"]
        original = [int(n) for n in w["nodes"]]
        if wid in del_ways:
            plan.removed_ways.add(wid)
            plan.orphan_candidates.update(original)
            continue
        owner = "way %d" % wid
        refs = edits.way_refs.get(wid)
        if refs is None:
            refs = original
        else:
            _check_new_refs(refs, original, known_nodes, owner)
        refs = _drop_deleted_nodes(refs, del_nodes, policy, owner)
        if len(refs) < 2:
            plan.removed_ways.add(wid)
            plan.orphan_candidates.update(original)
            degenerate.append(wid)
            continue
        plan.way_refs[wid] = refs
    if degenerate:
        warnings.warn(
            "write_pbf: %d way(s) were left with fewer than two member nodes and "
            "were not written: %s" % (len(degenerate), sorted(degenerate)[:5]),
            UserWarning,
            stacklevel=2,
        )


def _plan_new_ways(edits, del_nodes, policy, known_nodes):
    """Validate and deletion-filter the member lists of the newly added ways."""
    out = []
    for wid, refs, tags in edits.new_ways:
        owner = "new way %d" % wid
        _check_new_refs(refs, (), known_nodes, owner)
        refs = _drop_deleted_nodes(refs, del_nodes, policy, owner)
        if len(refs) < 2:
            raise ValueError(
                "write_pbf: new way %d needs at least two member nodes, got %d."
                % (wid, len(refs))
            )
        out.append((wid, refs, tags))
    return out


def _plan_relations(plan, relations, del_nodes, del_rels, policy):
    """Drop deleted members from the kept relations and note the nodes they hold."""
    if "id" not in relations:
        return
    ids = relations["id"]
    for i in range(len(ids)):
        rid = int(ids[i])
        if rid in del_rels:
            plan.removed_rels.add(rid)
            continue
        mem = relations["members"][i]
        kept, gone = [], []
        for j in range(len(mem["member_id"])):
            mtype = mem["member_type"][j]
            mid = int(mem["member_id"][j])
            decoded = mtype.decode() if isinstance(mtype, (bytes, bytearray)) else mtype
            if (
                (decoded == "node" and mid in del_nodes)
                or (decoded == "way" and mid in plan.removed_ways)
                or (decoded == "relation" and mid in del_rels)
            ):
                gone.append((decoded, mid))
                continue
            if decoded == "node":
                plan.referenced_nodes.add(mid)
            kept.append((mtype, mid, mem["member_role"][j]))
        if gone:
            if policy == "error":
                raise ValueError(
                    "write_pbf: relation %d still references removed member(s) %s. "
                    "Pass on_deleted_member='drop' to remove them from its member "
                    "list." % (rid, gone[:5])
                )
            plan.rel_members[rid] = kept


def _resolve_orphans(plan, node_coordinates, policy):
    """Handle the nodes left unreferenced by the ways that were dropped."""
    orphans = {
        nid
        for nid in plan.orphan_candidates
        if nid not in plan.referenced_nodes
        and nid not in plan.removed_nodes
        and nid in node_coordinates
    }
    if not orphans:
        return
    if policy == "error":
        raise ValueError(
            "write_pbf: %d node(s) %s are left unreferenced by the dropped way(s). "
            "Pass on_orphan_node='remove' to drop them too, or 'keep' to write them "
            "as bare nodes." % (len(orphans), sorted(orphans)[:5])
        )
    if policy == "remove":
        plan.removed_nodes.update(orphans)


def _build_plan(
    edits,
    way_records,
    relations,
    node_coordinates,
    deletions,
    on_orphan_node,
    on_deleted_member,
):
    """Resolve edits and deletions into a plan plus the validated new ways."""
    del_nodes, del_ways, del_rels = deletions
    plan = _WritePlan(edits, del_nodes)
    known_nodes = _KnownNodes(node_coordinates, {n[0] for n in edits.new_nodes})

    _plan_ways(
        plan, way_records, edits, del_nodes, del_ways, on_deleted_member, known_nodes
    )
    new_ways = _plan_new_ways(edits, del_nodes, on_deleted_member, known_nodes)
    for refs in plan.way_refs.values():
        plan.referenced_nodes.update(refs)
    for _, refs, _ in new_ways:
        plan.referenced_nodes.update(refs)

    _plan_relations(plan, relations, del_nodes, del_rels, on_deleted_member)
    _resolve_orphans(plan, node_coordinates, on_orphan_node)
    return plan, new_ways


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

    def add_new_node(self, nid, lon, lat, tags):
        self.add_node(nid, lat, lon, 1, self._now, 0, tags)

    def add_new_way(self, wid, refs, tags):
        self.ways.append(
            {
                "id": wid,
                "refs": refs,
                "version": 1,
                "timestamp": self._now,
                "tags": tags,
            }
        )

    def begin_synthesis(self, reserved_way_ids=()):
        """Start the generated-id counters below every id already claimed.

        ``reserved_way_ids`` are the caller-chosen ids of new ways whose vertices are
        still to be synthesized, so they are not yet among ``self.ways``.
        """
        min_node = min(self.node_ids) if self.node_ids else 0
        min_way = min((w["id"] for w in self.ways), default=0)
        min_way = min([min_way] + list(reserved_way_ids))
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

    def _add_way(self, refs, tags, wid=None):
        if wid is None:
            wid = self._way_counter
            self._way_counter -= 1
        self.add_new_way(wid, refs, tags)

    def add_geometry(self, oid, geom, tags, wid=None):
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
            self._add_way(refs, tags, wid)
        elif gtype == "Polygon" and len(geom.interiors) == 0:
            refs = [self._node_for(c[0], c[1], oid) for c in geom.exterior.coords]
            self._add_way(refs, tags, wid)
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


def _add_base_nodes(builder, node_coordinates, nodes_cache, plan, keep=None):
    poi_tags = _node_tag_records(nodes_cache)
    for nid, rec in node_coordinates.items():
        if nid in plan.removed_nodes:
            continue
        if keep is not None and nid not in keep:
            continue
        if nid in plan.node_tags:
            tags = plan.node_tags[nid]
        else:
            tags = poi_tags.get(nid)
            if tags is None:
                tags = rec.get("tags")
            tags = tags if isinstance(tags, dict) else None
        lon, lat = plan.node_coords.get(nid, (rec["lon"], rec["lat"]))
        builder.add_node(
            nid,
            lat,
            lon,
            int(rec.get("version") or 1),
            int(rec.get("timestamp") or 0),
            int(rec.get("changeset") or 0),
            tags,
        )


def _add_base_ways(builder, way_records, plan, keep=None):
    for w in way_records:
        wid = w["id"]
        if wid not in plan.way_refs:
            continue
        if keep is not None and wid not in keep:
            continue
        builder.ways.append(
            {
                "id": wid,
                "refs": plan.way_refs[wid],
                "version": w.get("version") or 1,
                "timestamp": w.get("timestamp"),
                "tags": plan.way_tags.get(wid, _record_tags(w)),
            }
        )


def _add_base_relations(builder, relations, plan, keep=None):
    if "id" not in relations:
        return
    for i in range(len(relations["id"])):
        rid = int(relations["id"][i])
        if rid in plan.removed_rels:
            continue
        if keep is not None and rid not in keep:
            continue
        mem = relations["members"][i]
        members = plan.rel_members.get(rid)
        if members is None:
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
                "tags": plan.rel_tags.get(
                    rid, rtags if isinstance(rtags, dict) else {}
                ),
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
    apply_geometry=False,
    delete=None,
    on_orphan_node="remove",
    on_deleted_member="drop",
):
    """Write `data` (+ the cached dataset) to a valid PBF at `output_path`.

    `node_coordinates` / `way_records` / `relations` / `nodes` are the caches the
    `OSM` object holds after reading; `data` is a GeoDataFrame or list of them
    whose tag edits are applied by ``osm_type``+``id`` (new rows are synthesized).

    When ``subset_only`` is True, only the elements present in `data` are written
    (matched by ``osm_type``+``id``), together with the references they need to stay
    valid -- kept ways pull in their nodes and kept relations pull in their member
    ways/nodes (the union across all frames). New rows are still synthesized.

    ``apply_geometry`` additionally applies the rows' geometry to existing elements
    and ``delete`` removes elements; ``on_orphan_node`` and ``on_deleted_member``
    decide what happens to the references those removals break. See
    ``pyrosm.OSM.write_pbf`` for the full semantics.
    """
    _validate_policy("on_orphan_node", on_orphan_node, _ORPHAN_NODE_POLICIES)
    _validate_policy("on_deleted_member", on_deleted_member, _DELETED_MEMBER_POLICIES)

    relations = relations or {}
    frames = _reproject_to_wgs84(_as_frames(data))

    way_by_id = {w["id"]: w for w in way_records}
    rel_ids = set(int(i) for i in relations["id"]) if "id" in relations else set()

    edits = _collect_edits(frames, way_by_id, node_coordinates, rel_ids, apply_geometry)
    deletions = _parse_deletions(delete, node_coordinates, way_by_id, rel_ids)
    plan, new_ways = _build_plan(
        edits,
        way_records,
        relations,
        node_coordinates,
        deletions,
        on_orphan_node,
        on_deleted_member,
    )

    keep_nodes = keep_ways = keep_rels = None
    if subset_only:
        keep_nodes, keep_ways, keep_rels = _subset_keep_sets(
            set(edits.node_tags) - plan.removed_nodes,
            set(edits.way_tags) - plan.removed_ways,
            set(edits.rel_tags) - plan.removed_rels,
            plan.way_refs,
            relations,
        )

    builder = _RecordBuilder()
    _add_base_nodes(builder, node_coordinates, nodes, plan, keep_nodes)
    for nid, lon, lat, tags in edits.new_nodes:
        builder.add_new_node(nid, lon, lat, tags)
    _add_base_ways(builder, way_records, plan, keep_ways)
    for wid, refs, tags in new_ways:
        builder.add_new_way(wid, refs, tags)
    _add_base_relations(builder, relations, plan, keep_rels)

    builder.begin_synthesis(
        reserved_way_ids=[r[3] for r in edits.auto_rows if r[3] is not None]
    )
    for oid, geom, tags, wid in edits.auto_rows:
        builder.add_geometry(oid, geom, tags, wid)

    return write_pbf_from_records(
        builder.node_payload(),
        builder.ways,
        builder.rels,
        output_path,
        builder.bounds(),
    )
