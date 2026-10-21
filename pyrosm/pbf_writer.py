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

from pyrosm.filter_compiler import CompiledFilter, Condition
from pyrosm.pbf_export import write_pbf_from_records
from pyrosm.utils import _stacklevel_outside_pyrosm
from pyrosm.utils.progress import start_step
from pyrosm.way_store import NodeTags

# The steps of a write, as its progress names them, and their shares, relative to
# decoding the file in memory, from writing the buildings of Helsinki (65 MB) back.
_APPLYING = "applying the edits"
_NODES = "preparing the nodes"
_WAYS = "preparing the ways"
_RELATIONS = "preparing the relations"
_WRITING = "writing the file"
WRITE_SHARES = {
    _APPLYING: 5.6,
    _NODES: 2.2,
    _WAYS: 3.2,
    _RELATIONS: 0.5,
    _WRITING: 5.5,
}

# Frame columns that hold an element's fields, not its tags.
_FIELD_COLS = {
    "id",
    "osm_type",
    "version",
    "timestamp",
    "changeset",
    "visible",
    "tags",
    "nodes",
    "members",
    "lon",
    "lat",
    "geometry",
}
# Numeric columns get_network computes: the end node ids of a segment and its length in
# metres. A text column of one of these names holds the OSM tag of that name.
_NETWORK_COLS = ("u", "v", "length")
# The fields of a way record, next to its tags.
_RECORD_FIELDS = {"id", "version", "timestamp", "visible", "nodes", "tags"}
# Element fields the readers put into a JSON ``tags`` dict when they are not columns.
# Their values are not text, unlike those of a tag of the same name.
_DICT_FIELDS = {"version", "timestamp", "visible", "members"}


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


def _column_name(col):
    return col if isinstance(col, str) else str(col)


def _tag_key(col):
    """The OSM tag key of a column or tag name, or None to skip it: ``id_tag``, the
    name the readers give a tag keyed ``id``, is ``id`` again.

    OSM tag keys are non-empty strings; arbitrary GeoDataFrame columns can be
    non-string or empty, which would corrupt the string-table / keys_vals output.
    """
    key = _column_name(col)
    if key == "id_tag":
        return "id"
    return key if key != "" else None


def _frame_tag_columns(gdf):
    """The columns of ``gdf`` that hold tags: all but its geometry, the element fields
    and the columns get_network computes (numeric ``u``, ``v`` and ``length``)."""
    computed = {
        col
        for col in _NETWORK_COLS
        if col in gdf.columns and pd.api.types.is_numeric_dtype(gdf[col])
    }
    return [
        col
        for col in gdf.columns
        if col != gdf.geometry.name
        and _column_name(col) not in _FIELD_COLS
        and _column_name(col) not in computed
        and _tag_key(col) is not None
    ]


def _add_dict_tags(tags, extra, columns=()):
    """Add the tags of the JSON ``tags`` dict (or string) ``extra`` to ``tags``, keeping
    the values ``tags`` already has and leaving out the keys of the tag ``columns``."""
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (ValueError, TypeError):
            return tags
    if isinstance(extra, dict):
        for k, v in extra.items():
            key = _tag_key(k)
            if (
                key is None
                or (_column_name(k) in _DICT_FIELDS and not isinstance(v, str))
                or key in tags
                or key in columns
                or _is_missing(v)
            ):
                continue
            tags[key] = _tag_str(v)
    return tags


def _frame_tags(gdf, tag_cols):
    """The tags of each row of ``gdf``: its values in the tag columns ``tag_cols`` (see
    :func:`_frame_tag_columns`) and its JSON ``tags``. A missing value removes the tag
    of its column, also when the JSON holds it."""
    keys = [_tag_key(col) for col in tag_cols]
    tags = [{} for _ in range(len(gdf))]
    for col, key in zip(tag_cols, keys):
        values = gdf[col]
        present = values.notna().to_numpy()
        for i, value in zip(np.flatnonzero(present).tolist(), values[present].tolist()):
            tags[i][key] = _tag_str(value)
    extras = gdf["tags"].tolist() if "tags" in gdf.columns else [None] * len(gdf)
    columns = set(keys)
    return [_add_dict_tags(row, extra, columns) for row, extra in zip(tags, extras)]


def _record_tags(record):
    """Tags for a cached way record: its keys other than the record's fields, and any
    leftover tags dict."""
    tags = {}
    for col, val in record.items():
        key = _tag_key(col)
        if key is None or _column_name(col) in _RECORD_FIELDS or _is_missing(val):
            continue
        tags[key] = _tag_str(val)
    return _add_dict_tags(tags, record.get("tags"))


# The tags pyrosm's readers detect features by, with the reader that finds them and the
# element types it finds by their own tags (a network's nodes come with its ways). A frame
# without the note of its read (see pyrosm.utils.records_read) is checked against these.
_ALL_TYPES = frozenset({"node", "way", "relation"})
_AREA_TYPES = frozenset({"way", "relation"})
_FEATURE_TAGS = {
    "building": ("get_buildings()", _AREA_TYPES),
    "highway": ("get_network()", frozenset({"way"})),
    "amenity": ("get_pois()", _ALL_TYPES),
    "shop": ("get_pois()", _ALL_TYPES),
    "tourism": ("get_pois()", _ALL_TYPES),
    "landuse": ("get_landuse()", _ALL_TYPES),
    "natural": ("get_natural()", _ALL_TYPES),
    "boundary": ("get_boundaries()", _AREA_TYPES),
}

# What the elements of each reader are, for the warning.
_READ_FEATURES = {
    "get_buildings": "buildings",
    "get_pois": "points of interest",
    "get_landuse": "land use areas",
    "get_natural": "natural features",
    "get_boundaries": "boundaries",
    "get_network": "network ways",
    "get_data_by_custom_criteria": "matches of its filter",
}


def _text_list(items):
    """``items`` (strings) as text: ``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _elements(count):
    return f"{count} element{'' if count == 1 else 's'}"


def _read_note(gdf):
    """``(reader, filter)`` from the note a pyrosm reader left on ``gdf`` (see
    :func:`pyrosm.utils.records_read`), or ``None`` when it has none. The filter keeps
    only which keys the read selected by: each AND-group requires its positively tested
    keys to exist, so a changed value is an edit, not a lost tag."""
    note = gdf.attrs.get("pyrosm_read")
    try:
        read_filter = CompiledFilter.from_json(note["filter"])
        reader = note["reader"]
    except (KeyError, TypeError, ValueError):
        return None
    groups = tuple(
        tuple(
            Condition(key, "exists")
            for key in sorted({c.key for c in group if c.is_positive})
        )
        for group in read_filter.groups
    )
    return reader, CompiledFilter(groups)


class _LostTags:
    """The edited elements whose written tags lose what they were selected by, counted
    once each for the warning of ``write_pbf`` (see :meth:`warn`)."""

    def __init__(self):
        self.by_reader = {}  # reader -> [elements, tags the elements lose]
        self.by_tag = {}  # feature tag -> elements, for frames without a note

    def check(self, note, element, original, written):
        """Count ``element`` (its type and id) if its ``written`` tags no longer match the
        filter of the ``note`` its frame carries (see :func:`_read_note`) while its
        ``original`` tags do, or, without a note, lose a tag of :data:`_FEATURE_TAGS` that
        its reader finds elements of this type by."""
        if note is not None:
            reader, read_filter = note
            if read_filter.matches(original) and not read_filter.matches(written):
                entry = self.by_reader.setdefault(reader, [set(), set()])
                entry[0].add(element)
                entry[1].update(
                    key
                    for key in read_filter.positive_keys
                    if key in original and key not in written
                )
            return
        for tag, (_, types) in _FEATURE_TAGS.items():
            if element[0] in types and tag in original and tag not in written:
                self.by_tag.setdefault(tag, set()).add(element)

    def warn(self):
        """Warn once, at the user's call, about the elements counted."""
        parts = []
        for reader, (elements, keys) in self.by_reader.items():
            count = len(elements)
            quoted = _text_list(f"'{key}'" for key in sorted(keys))
            verb = "tag is" if len(keys) == 1 else "tags are"
            cause = f"their {quoted} {verb} not in the frame"
            parts.append(
                f"{_elements(count)} would no longer be found by {reader}(): {cause}, so "
                "pyrosm and other OSM tools cannot detect them as "
                f"{_READ_FEATURES.get(reader, 'its features')} in the written file."
            )
        if self.by_tag:
            phrases = []
            counts = sorted(
                ((tag, len(elements)) for tag, elements in self.by_tag.items()),
                key=lambda item: -item[1],
            )
            for tag, count in counts:
                if phrases:
                    subject = str(count)
                else:
                    subject = (
                        f"{_elements(count)} {'is' if count == 1 else 'are'} written"
                    )
                their = "its" if count == 1 else "their"
                phrases.append(f"{subject} without {their} '{tag}' tag")
            readers = dict.fromkeys(_FEATURE_TAGS[tag][0] for tag, _ in counts)
            them = "it" if len(counts) == 1 and counts[0][1] == 1 else "them"
            parts.append(
                f"{_text_list(phrases)}; {_text_list(readers)} will not find {them} in the "
                "written file."
            )
        if parts:
            warnings.warn(
                " ".join(parts), UserWarning, stacklevel=_stacklevel_outside_pyrosm()
            )


class _SourceTags:
    """The tags of an element in the source, as the writer extracts a row's: a way's
    record, a tagged node's ``poi_tags`` (an untagged node has none) or a relation's
    tags."""

    def __init__(self, way_by_id, poi_tags, relations):
        self._way_by_id = way_by_id
        self._poi_tags = poi_tags
        self._relations = relations
        self._rel_position = None

    def __call__(self, otype, oid):
        if otype == "way":
            return _record_tags(self._way_by_id[oid])
        if otype == "node":
            source = self._poi_tags.get(oid)
        else:
            if self._rel_position is None:
                ids = self._relations["id"]
                self._rel_position = {int(r): i for i, r in enumerate(ids)}
            source = self._relations["tags"][self._rel_position[oid]]
        return _add_dict_tags({}, source)


def _written_and_removed(edits, plan):
    """Per element type, the tags its edited elements are written with and the ids the
    ``plan`` removes (those elements are not written)."""
    return {
        "node": (edits.node_tags, plan.removed_nodes),
        "way": (edits.way_tags, plan.removed_ways),
        "relation": (edits.rel_tags, plan.removed_rels),
    }


def _warn_lost_tags(edits, plan, source_tags):
    """Warn about the edited elements whose written tags lose what a read selected them
    by (see :class:`_LostTags`), comparing each with its tags in the source."""
    by_type = _written_and_removed(edits, plan)
    lost = _LostTags()
    for (otype, oid), notes in edits.notes.items():
        written, removed = by_type[otype]
        tags = written[oid]
        if tags is None or oid in removed:
            continue
        original = source_tags(otype, oid)
        for note in [note for note in notes.values() if note is not None] or [None]:
            lost.check(note, (otype, oid), original, tags)
    lost.warn()


def _warn_partial_edits(edits, plan):
    """Warn once about the tags changed in only some rows of an element in one frame,
    which are written for the whole element (see :func:`_merged_tags`)."""
    by_type = _written_and_removed(edits, plan)
    partial = [p for p in edits.partial if p[1] not in by_type[p[0]][1]]
    if not partial:
        return
    if len(partial) == 1:
        otype, oid, key, changed, rows = partial[0]
        message = (
            f"'{key}' of {otype} {oid} was changed in {changed} of its {rows} rows; "
            f"the change is written for the whole {otype}, as an OSM element has one "
            "value per tag."
        )
    else:
        shown = [
            f"'{key}' of {otype} {oid} ({changed} of {rows} rows)"
            for otype, oid, key, changed, rows in partial[:3]
        ]
        if len(partial) > 3:
            shown.append(f"{len(partial) - 3} more")
        message = (
            f"{len(partial)} tags were changed in only some rows of an element; each "
            "change is written for the whole element, as an OSM element has one value "
            f"per tag: {_text_list(shown)}."
        )
    warnings.warn(message, UserWarning, stacklevel=_stacklevel_outside_pyrosm())


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

    The tags of an element's rows are merged (see :func:`_merged_tags`), but two
    different coordinates or member lists for one element are a topology conflict with
    no safe resolution. This also catches the segment-level edge frame from
    ``get_network(nodes=True)``, where each way spans several rows.
    """
    previous = mapping.get(oid)
    if previous is not None and previous != value:
        raise ValueError(
            "write_pbf: conflicting %s for id %d (%r and %r). Every row of an "
            "element must carry the same geometry." % (what, oid, previous, value)
        )
    mapping[oid] = value


class _Edits:
    """Per-element edits and additions collected from the input frame(s). A tag edit of
    ``None`` keeps the element's own tags."""

    def __init__(self):
        self.node_tags, self.node_coords = {}, {}
        self.way_tags, self.way_refs = {}, {}
        self.rel_tags = {}
        # (osm_type, id) -> the read notes of the frames with a row of the element
        self.notes = {}
        # (osm_type, id, key, rows changing it, rows of the element in that frame)
        self.partial = []
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


def _add_row_tags(rows, element, frame, tags):
    """Count a row of ``element`` in ``frame`` with ``tags`` (None when the frame has no
    tag columns) in ``rows``: one ``[frame, tags, row count]`` entry per distinct tags.
    """
    entries = rows.setdefault(element, [])
    for entry in entries:
        if entry[0] == frame and entry[1] == tags:
            entry[2] += 1
            return
    entries.append([frame, tags, 1])


def _merged_tags(element, entries, frame_keys, source_tags, partial):
    """The tags written for ``element`` from the ``entries`` of its rows (see
    :func:`_add_row_tags`), or None when none of its frames has tag columns.

    A tag gets the value its rows give it. A value that rows change or remove, compared
    with the ``source_tags``, is written for the whole element, and noted in ``partial``
    when other rows of that frame keep the old value; two different new values raise.
    A row removes the tags of its frame's columns and the tags other rows of the element
    in that frame have, when it lacks them; it says nothing about the other tags. A row
    of a frame without tag columns keeps the element's own tags.
    """
    tagged = [tags for _, tags, _ in entries if tags is not None]
    if not tagged:
        return None
    if len(tagged) == len(entries) and all(tags == tagged[0] for tags in tagged):
        return tagged[0]
    original = source_tags(*element)
    given_keys = {}  # frame -> the keys its rows give a value or remove
    frame_rows = {}
    for frame, tags, count in entries:
        if tags is not None:
            given_keys.setdefault(frame, set(frame_keys[frame])).update(tags)
        frame_rows[frame] = frame_rows.get(frame, 0) + count
    rows = []  # (frame, tags, the keys the row gives a value or removes, row count)
    for frame, tags, count in entries:
        if tags is None:
            rows.append((frame, original, set(original), count))
        else:
            rows.append((frame, tags, given_keys[frame], count))
    merged = {}
    for key in dict.fromkeys(k for _, _, keys, _ in rows for k in keys):
        old = original.get(key)
        given = [
            (frame, tags.get(key), n) for frame, tags, keys, n in rows if key in keys
        ]
        new = {value for _, value, _ in given if value != old}
        if len(new) > 1:
            values = _text_list(
                sorted("removed" if v is None else f"'{v}'" for v in new)
            )
            raise ValueError(
                f"write_pbf: the rows of {element[0]} {element[1]} give '{key}' "
                f"different new values ({values}). An OSM element has one value per "
                "tag, so give its rows the same value."
            )
        value = new.pop() if new else old
        if value is not None:
            merged[key] = value
        if value == old:
            continue
        changed = {}
        for frame, given_value, n in given:
            if given_value == value:
                changed[frame] = changed.get(frame, 0) + n
        for frame, n in changed.items():
            if n < frame_rows[frame]:
                partial.append((*element, key, n, frame_rows[frame]))
    return merged


def _collect_edits(
    frames, way_by_id, node_coordinates, rel_ids, apply_geometry, source_tags
):
    """Split frame rows into edits of existing elements and new elements.

    Existing elements are matched by ``osm_type`` + ``id``; their tags are taken from the
    row, unless the frame has no tag columns, in which case they keep their own. The
    rows of an element are merged tag by tag (see :func:`_merged_tags`), and the read
    notes of their frames are kept (see :func:`_read_note`). With
    ``apply_geometry`` a node row's ``Point`` also moves the
    node, a way row's ``nodes`` column also reshapes the way, and a row with a
    negative unmatched id is added under exactly that id (so a way can reference a
    vertex added in the same call). Otherwise unmatched rows are synthesized with
    generated ids, as they always have been.
    """
    edits = _Edits()
    seen_new = set()
    rows = {}  # (osm_type, id) -> the entries of its rows (see _add_row_tags)
    frame_keys = []  # per frame, the tag keys of its tag columns
    for frame, gdf in enumerate(frames):
        geom_col = gdf.geometry.name
        note = _read_note(gdf)
        tag_cols = _frame_tag_columns(gdf)
        frame_keys.append({_tag_key(col) for col in tag_cols})
        has_tags = bool(tag_cols) or ("tags" in gdf.columns and geom_col != "tags")
        fields = [c for c in ("osm_type", "id", "nodes") if c in gdf.columns]
        records = gdf[fields + [geom_col]].to_dict("records")
        for row, tags in zip(records, _frame_tags(gdf, tag_cols)):
            otype = _normalize_osm_type(row.get("osm_type"))
            oid = _normalize_id(row.get("id"))
            geom = row[geom_col]
            if otype is None:
                otype = _infer_osm_type(geom)
            refs = _row_refs(row) if apply_geometry else None
            matched = oid is not None and (
                (otype == "way" and oid in way_by_id)
                or (otype == "node" and oid in node_coordinates)
                or (otype == "relation" and oid in rel_ids)
            )
            if matched:
                _add_row_tags(rows, (otype, oid), frame, tags if has_tags else None)
                edits.notes.setdefault((otype, oid), {})[frame] = note
                if otype == "way" and refs is not None:
                    _set_geometry_edit(edits.way_refs, oid, refs, "member node ids")
                elif (
                    otype == "node"
                    and apply_geometry
                    and getattr(geom, "geom_type", None) == "Point"
                ):
                    _check_lonlat(geom.x, geom.y, oid)
                    _set_geometry_edit(
                        edits.node_coords, oid, (geom.x, geom.y), "coordinates"
                    )
            elif apply_geometry and oid is not None and oid < 0:
                _add_new_element(edits, oid, otype, geom, tags, refs, seen_new)
            else:
                edits.auto_rows.append((oid, geom, tags, None))
    written = {
        "node": edits.node_tags,
        "way": edits.way_tags,
        "relation": edits.rel_tags,
    }
    for element, entries in rows.items():
        written[element[0]][element[1]] = _merged_tags(
            element, entries, frame_keys, source_tags, edits.partial
        )
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

    def add_nodes(self, ids, lats, lons, versions, timestamps, changesets, tags):
        """Add the nodes of aligned arrays (``tags`` a list), as :meth:`add_node` adds
        one."""
        self.node_ids.extend(ids.tolist())
        self.lats.extend(lats.tolist())
        self.lons.extend(lons.tolist())
        self.vers.extend(versions.tolist())
        self.tss.extend(timestamps.tolist())
        self.css.extend(changesets.tolist())
        self.ntags.extend(tags)

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
        # NodeTags lists its tagged nodes; an object column is checked node by node.
        rows = nd_tags.rows if isinstance(nd_tags, NodeTags) else range(len(nd_ids))
        for i in rows:
            t = nd_tags[i]
            if isinstance(t, dict):
                out[int(nd_ids[i])] = t
    return out


def _add_base_nodes(builder, node_coordinates, poi_tags, plan, keep=None):
    """Add the source nodes (those in ``keep`` when given) to ``builder``, with the tags and
    coordinates their edits in ``plan`` give, else their own (``poi_tags`` for tagged nodes).
    """
    ids = np.asarray(node_coordinates.ids, dtype=np.int64)
    kept = np.ones(len(ids), dtype=bool)
    if plan.removed_nodes:
        kept &= ~np.isin(ids, np.fromiter(plan.removed_nodes, dtype=np.int64))
    if keep is not None:
        kept &= np.isin(ids, np.fromiter(keep, dtype=np.int64))
    rows = np.flatnonzero(kept)
    ids = ids[rows]
    lon = np.array(node_coordinates.values_at("lon", rows), dtype=np.float64)
    lat = np.array(node_coordinates.values_at("lat", rows), dtype=np.float64)
    tags = node_coordinates.values_at("tags", rows)
    tags = (
        [None] * len(ids)
        if tags is None
        else [t if isinstance(t, dict) else None for t in tags]
    )
    for i, (x, y) in _rows_of(ids, plan.node_coords):
        lon[i], lat[i] = x, y
    for i, t in _rows_of(ids, poi_tags):
        tags[i] = t if isinstance(t, dict) else None
    for i, t in _rows_of(ids, plan.node_tags):
        tags[i] = t
    builder.add_nodes(
        ids,
        lat,
        lon,
        _int_values(node_coordinates.values_at("version", rows), 1, len(ids)),
        _int_values(node_coordinates.values_at("timestamp", rows), 0, len(ids)),
        _int_values(node_coordinates.values_at("changeset", rows), 0, len(ids)),
        tags,
    )


def _rows_of(ids, values):
    """``(row, value)`` for each row of ``ids`` (an int64 array) whose id has a value
    other than None in ``values`` (an id -> value dict); a repeated id gets it in each
    of its rows."""
    values = {nid: value for nid, value in values.items() if value is not None}
    if not values:
        return []
    edited = np.fromiter(values, dtype=np.int64, count=len(values))
    rows = np.flatnonzero(np.isin(ids, edited))
    return [(row, values[nid]) for row, nid in zip(rows.tolist(), ids[rows].tolist())]


def _int_values(values, default, count):
    """The ``count`` integers of the node values ``values``, each ``int(value or
    default)``, or all ``default`` when ``values`` is None."""
    if values is None:
        return np.full(count, default, dtype=np.int64)
    values = np.asarray(values)
    if values.dtype.kind not in "iu":
        return np.array([int(v or default) for v in values.tolist()], dtype=np.int64)
    values = values.astype(np.int64)
    values[values == 0] = default
    return values


def _add_base_ways(builder, way_records, plan, keep=None):
    for w in way_records:
        wid = w["id"]
        if wid not in plan.way_refs:
            continue
        if keep is not None and wid not in keep:
            continue
        tags = plan.way_tags.get(wid)
        builder.ways.append(
            {
                "id": wid,
                "refs": plan.way_refs[wid],
                "version": w.get("version") or 1,
                "timestamp": w.get("timestamp"),
                "tags": _record_tags(w) if tags is None else tags,
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
        rtags = plan.rel_tags.get(rid)
        if rtags is None:
            rtags = relations["tags"][i]
            rtags = rtags if isinstance(rtags, dict) else {}
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
                "tags": rtags,
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
    steps=None,
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
    ``pyrosm.OSM.write_pbf`` for the full semantics. The write goes through the steps
    of :data:`WRITE_SHARES` in ``steps`` (a :class:`~pyrosm.utils.progress.Steps`, or
    None).
    """
    _validate_policy("on_orphan_node", on_orphan_node, _ORPHAN_NODE_POLICIES)
    _validate_policy("on_deleted_member", on_deleted_member, _DELETED_MEMBER_POLICIES)

    start_step(steps, _APPLYING)

    relations = relations or {}
    frames = _reproject_to_wgs84(_as_frames(data))

    way_by_id = {w["id"]: w for w in way_records}
    rel_ids = set(int(i) for i in relations["id"]) if "id" in relations else set()
    poi_tags = _node_tag_records(nodes)
    source_tags = _SourceTags(way_by_id, poi_tags, relations)

    edits = _collect_edits(
        frames, way_by_id, node_coordinates, rel_ids, apply_geometry, source_tags
    )
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
    _warn_partial_edits(edits, plan)
    _warn_lost_tags(edits, plan, source_tags)

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
    start_step(steps, _NODES)
    _add_base_nodes(builder, node_coordinates, poi_tags, plan, keep_nodes)
    for nid, lon, lat, tags in edits.new_nodes:
        builder.add_new_node(nid, lon, lat, tags)
    start_step(steps, _WAYS)
    _add_base_ways(builder, way_records, plan, keep_ways)
    for wid, refs, tags in new_ways:
        builder.add_new_way(wid, refs, tags)
    start_step(steps, _RELATIONS)
    _add_base_relations(builder, relations, plan, keep_rels)

    builder.begin_synthesis(
        reserved_way_ids=[r[3] for r in edits.auto_rows if r[3] is not None]
    )
    for oid, geom, tags, wid in edits.auto_rows:
        builder.add_geometry(oid, geom, tags, wid)

    payload = builder.node_payload()
    return write_pbf_from_records(
        payload,
        builder.ways,
        builder.rels,
        output_path,
        builder.bounds(),
        progress=start_step(steps, _WRITING),
    )
