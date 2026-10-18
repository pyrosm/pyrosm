"""Main-process collection.

Read the spilled shards back into the layer's node, way and relation records, their
member ways and the referenced node coordinates -- pulling only what the kept features
need off disk, and refining the worker's key-presence candidates by pyrosm's exact value
filter so peak memory stays bounded by the working set.
"""

import os

import numpy as np
from rapidjson import dumps

from pyrosm._arrays import columns_to_arrays
from pyrosm.tagparser import explode_node_tag_array
from pyrosm.data_filter import element_should_be_kept
from pyrosm.geometry import AREA_BY_TAGS, tags_make_area
from pyrosm.engine.decode import (
    _is_threaded,
    _object_array,
    _open_shard,
    _shard_tags,
)
from pyrosm.engine.bounding_box import _in_box_nodes
from pyrosm.engine.ways import (
    ElementStore,
    node_coordinates,
    node_record_blocks,
    node_features,
    relation_ways,
    result_tag_columns,
    tag_lists,
)
from pyrosm.utils import _compat

# Relation.MemberType -> the byte labels pyrosm's relation assembly expects.
_MEMBER_TYPE = {0: b"node", 1: b"way", 2: b"relation"}
# Only way members carry ring geometry. Member ids are matched against the way store, but
# OSM ids are unique only per element type, so a node/relation member id can collide with a
# way id -- mixing types in the lookup would attach or drop the wrong way.
_WAY_MEMBER = 1


# The standalone-way columns are carried as a dict of parallel arrays (id / nodes / tags /
# metadata) rather than a list of per-way dicts: the shards already hold them columnar, so
# this skips building millions of intermediate dicts (and the per-way ``.tolist()`` on the
# node refs) on the way to the assembly's column arrays. ``nodes`` is an object array whose
# entries are each way's int64 node-ref array.
def _read_way_columns(shard_paths):
    """Read the spilled matching ways back as a dict of parallel arrays (``id``, ``nodes``,
    ``tags``, ``version``, ``timestamp``, ``visible``). ``None`` if no shard holds a way.
    """
    ids, nodes_list, tags, vers, tss, viss = [], [], [], [], [], []
    for path in shard_paths:
        z = _open_shard(path)
        wid = z["way_id"]
        if len(wid) == 0:
            continue
        off, refs = z["refs_off"], z["refs"]
        ids.append(wid)
        # Per-way node-ref arrays (views into the shard's flat refs); no per-way list build.
        nodes_list.extend(np.split(refs, off[1:-1]) if len(wid) > 1 else [refs])
        tags.append(_shard_tags(z, "way"))
        vers.append(z["way_version"])
        tss.append(z["way_timestamp"])
        viss.append(z["way_visible"])
    if not ids:
        return None
    all_ids = np.concatenate(ids)
    nodes = np.empty(len(all_ids), dtype=object)
    nodes[:] = nodes_list
    return {
        "id": all_ids,
        "nodes": nodes,
        "tags": np.concatenate(tags),
        "version": np.concatenate(vers),
        "timestamp": np.concatenate(tss),
        # bool to match the in-memory reader and the node features' visible column (the shard
        # spills it as an int; a bool ``visible`` column keeps the GeoParquet schema union
        # consistent across the node and way chunks).
        "visible": np.concatenate(viss).astype(bool),
    }


def _num_ways(cols):
    """Number of standalone ways in a way-column dict or an :class:`ElementStore`."""
    return len(cols) if isinstance(cols, ElementStore) else len(cols["id"])


def _slice_way_columns(cols, start, stop):
    """A row slice ``[start:stop]`` of every column -- the chunking the GeoParquet streamer
    uses in place of slicing a list of records."""
    if isinstance(cols, ElementStore):
        return cols.take(np.arange(start, min(stop, len(cols))))
    return {k: v[start:stop] for k, v in cols.items()}


def _collect_node_features(shard_paths, tags_as_columns, keep_metadata, keep):
    """Read the spilled node features back, refine them by the exact value filter ``keep``
    and explode their tags into the same columns the in-memory reader builds (``id`` /
    ``lon`` / ``lat`` / ``visible`` + metadata when ``keep_metadata``, plus the layer tag
    columns and a JSON ``tags`` column). ``None`` if no node passes. The node features of
    a threaded decode come from their string ids (see :func:`ways.node_features`)."""
    if _is_threaded(shard_paths):
        return node_features(
            shard_paths, tags_as_columns, keep_metadata, keep.filter_spec
        )
    ids, lon, lat, tags, meta = [], [], [], [], []
    for path in shard_paths:
        z = _open_shard(path)
        nid = z["nfeat_id"]
        if len(nid) == 0:
            continue
        ids.append(nid)
        lon.append(z["nfeat_lon"])
        lat.append(z["nfeat_lat"])
        tags.append(_shard_tags(z, "nfeat"))
        meta.append(z["nfeat_meta"])
    if not ids:
        return None
    tag_array = np.concatenate(tags)
    mask = np.fromiter((keep(t) for t in tag_array), dtype=bool, count=len(tag_array))
    if not mask.any():
        return None
    node_arrays = _node_arrays(
        np.concatenate(ids)[mask],
        np.concatenate(lon)[mask],
        np.concatenate(lat)[mask],
        np.concatenate(meta)[mask],
        keep_metadata,
    )
    return _explode_node_tags(node_arrays, tag_array[mask], tags_as_columns)


def _node_arrays(ids, lon, lat, meta, keep_metadata):
    """The columns of node features that are not tags, ``meta`` holding each node's
    version, timestamp, changeset and visible. ``tags`` holds its place after ``lat``.
    """
    node_arrays = {
        "id": ids,
        "lon": lon,
        "lat": lat,
        "tags": None,
        "visible": meta[:, 3].astype(bool),
    }
    if keep_metadata:
        node_arrays["version"] = meta[:, 0]
        node_arrays["timestamp"] = meta[:, 1]
        node_arrays["changeset"] = meta[:, 2]
    return node_arrays


def _explode_node_tags(node_arrays, tags, tags_as_columns):
    """``node_arrays`` with the tag columns of the nodes' tag dicts ``tags`` merged in, as
    ``get_osm_nodes`` merges them: their JSON ``tags`` column takes the place of ``tags``
    (None for each node when none had other tags)."""
    exploded = explode_node_tag_array(tags, list(tags_as_columns))
    node_arrays["tags"] = exploded.pop("tags", [None] * len(tags))
    node_arrays.update(exploded)
    return node_arrays


def _gather_node_shards(shard_paths, needed):
    """Gather the coordinates of the ``needed`` (sorted) node ids that the shards hold.
    Returns ``(pos, lon, lat)``: ``pos`` indexes into ``needed`` for the ids found, and
    ``lon`` / ``lat`` their coordinates. A shard whose ids increase (a PBF sorted by type and
    id) searches only the needed ids within its id range; any other shard searches each of
    its ids in ``needed``.
    """
    n = len(needed)
    pos_parts, lon_parts, lat_parts = [], [], []
    for path in shard_paths:
        z = _open_shard(path)
        nid = z["node_id"]
        if len(nid) == 0:
            continue
        if np.all(nid[1:] > nid[:-1]):
            first = np.searchsorted(needed, nid[0], side="left")
            last = np.searchsorted(needed, nid[-1], side="right")
            wanted = needed[first:last]
            rows = np.searchsorted(nid, wanted)
            hit = nid[rows] == wanted
            pos, rows = first + np.flatnonzero(hit), rows[hit]
        else:
            pos = np.clip(np.searchsorted(needed, nid), 0, n - 1)
            rows = np.flatnonzero(needed[pos] == nid)
            pos = pos[rows]
        if len(pos):
            pos_parts.append(pos)
            lon_parts.append(z["node_lon"][rows])
            lat_parts.append(z["node_lat"][rows])
    if not pos_parts:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    return (
        np.concatenate(pos_parts),
        np.concatenate(lon_parts),
        np.concatenate(lat_parts),
    )


def _node_lookup(shard_paths, needed):
    """Gather only the coordinates of ``needed`` node ids from the shards (bounded memory) and
    wrap them in a ``NodeLocations`` for geometry assembly, without the ids no shard holds.
    """
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    lon = np.full(len(needed), np.nan)
    lat = np.full(len(needed), np.nan)
    # Node-only results (the filter matched no ways/relations) need no coordinates.
    if len(needed):
        pos, lon_found, lat_found = _gather_node_shards(shard_paths, needed)
        lon[pos] = lon_found
        lat[pos] = lat_found
    present = ~np.isnan(lon)
    return NodeLocations(
        pd.DataFrame({"id": needed[present], "lon": lon[present], "lat": lat[present]})
    )


def _coordinates(shard_paths, kept, relation_ways):
    """The coordinate store of the nodes the ways ``kept`` and ``relation_ways`` refer to
    (see :func:`_node_lookup`); for a threaded decode gathered on threads."""
    if not _is_threaded(shard_paths):
        return _node_lookup(shard_paths, _needed_node_ids(kept, relation_ways))
    return node_coordinates(shard_paths, _node_refs(kept, relation_ways))


# Node-record column dtypes for the graph-export gather (id/coords + element metadata).
_NODE_RECORD_DTYPES = {
    "id": np.int64,
    "lon": np.float64,
    "lat": np.float64,
    "version": np.int64,
    "timestamp": np.int64,
    "changeset": np.int64,
    "visible": bool,
}


def _gather_node_records(
    filepath, node_ids, keep_metadata, progress=None, threaded=False
):
    """Second pass over the file gathering the full records (coordinates + tags + metadata)
    of ``node_ids``, returned as a rich ``NodeLocations`` -- the coordinate store the
    graph-export node frame is built from (its records carry the node tags and metadata the
    lean coordinate lookup omits). Only the requested (network) nodes are materialised, so
    peak memory stays bounded by the graph's node set rather than the whole file.
    ``progress(done, file_size)``, when given, gets the bytes of the file read: 0 first, then
    after each data blob, and the file size at the end. With ``threaded`` (a threaded
    decode's read) the blocks are decoded without the GIL on threads (see
    :func:`~pyrosm.engine.ways.node_record_blocks`)."""
    from pyrosm.int64_hash import Int64Set
    from pyrosm.node_lookup import NodeLocations
    from pyrosm.primitive_block_decoder import decode_primitive_block
    from pyrosm.way_store import NodeTagsBuilder
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.engine.decode import _node_records_by_id

    wanted = Int64Set(node_ids)
    cols = ["id", "lon", "lat", "visible"]
    if keep_metadata:
        cols += ["version", "timestamp", "changeset"]
    arrays = {c: [] for c in cols}
    tags = NodeTagsBuilder()

    def python_path(data):
        st, header, nodes, _, _ = decode_primitive_block(data)
        return _node_records_by_id(st, header, nodes, wanted, keep_metadata, tags)

    with open(filepath, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        if progress is not None:
            progress(0, file_size)
        if threaded:
            blocks = node_record_blocks(filepath, wanted, keep_metadata)
        else:
            blocks = (
                (offset + size, _read_block(f, offset, size))
                for blob_type, offset, size in _index_blobs(filepath)
                if blob_type == "OSMData"
            )
        # ``item``: None (no wanted node), a block's bytes for the Python decoder, or the
        # wanted nodes' records and tags decoded on threads.
        for end, item in blocks:
            if isinstance(item, bytes):
                rec = python_path(item)
            elif item is not None:
                rec, counts, keys, vals, strings = item
                tags.add_pairs(counts, keys, vals, strings)
            else:
                rec = None
            if rec is not None:
                for c in cols:
                    arrays[c].append(rec[c])
            if progress is not None:
                progress(end, file_size)
        if progress is not None:
            progress(file_size, file_size)
    frame = {
        c: (
            np.concatenate(arrays[c])
            if arrays[c]
            else np.empty(0, _NODE_RECORD_DTYPES[c])
        )
        for c in cols
    }
    frame["tags"] = tags.build()
    # The in-memory node store's column order: the tags follow the coordinates.
    return NodeLocations({c: frame[c] for c in cols[:3] + ["tags"] + cols[3:]})


def _filter_way_columns(cols, exclude_ids, keep, in_box):
    """Refine read way columns down to the standalone ways to output: the candidates passing
    the exact value filter ``keep``, then -- when reading a bounding box -- restricted to ways
    with at least one node inside it (kept whole, so geometry is not cut at the edge), and
    finally with the relation member ways (``exclude_ids``) dropped (pyrosm assigns those to
    the relation, so they are not standalone way rows). Filtering is a single ordered mask, so
    the surviving rows keep their input order. ``None`` if nothing survives."""
    if cols is None:
        return None
    tags = cols["tags"]
    mask = np.fromiter((keep(t) for t in tags), dtype=bool, count=len(tags))
    if in_box is not None:
        inside = set(in_box.tolist())
        nodes = cols["nodes"]
        for i in np.nonzero(mask)[0]:
            if inside.isdisjoint(nodes[i].tolist()):
                mask[i] = False
    if len(exclude_ids):
        mask &= ~np.isin(cols["id"], exclude_ids)
    if not mask.any():
        return None
    return {k: v[mask] for k, v in cols.items()}


def _keep_fn(filter_spec):
    """The exact value-filter predicate for a layer: keep an element whose resolved tags pass
    ``filter_spec`` = ``(osm_keys, data_filter, filter_type)``. A ``None`` ``data_filter``
    (network ``all`` / ``driving_psv``) keeps every candidate."""
    osm_keys, data_filter, filter_type = filter_spec

    def keep(tag):
        return data_filter is None or element_should_be_kept(
            tag, osm_keys, data_filter, filter_type
        )

    # The element store of a threaded read applies ``filter_spec`` to its string ids itself.
    keep.filter_spec = filter_spec
    return keep


def _collect_kept_ways(shard_paths, exclude_ids, keep, in_box=None):
    """The standalone way columns to output: read the spilled candidates and refine them by
    the value filter, the bounding box, and the relation-member exclusion (see
    :func:`_filter_way_columns`). ``None`` if nothing survives. The ways of a threaded
    decode are selected in its :class:`~pyrosm.engine.ways.ElementStore`."""
    if _is_threaded(shard_paths):
        from pyrosm.engine.ways import ElementStore, kept_ways

        store = ElementStore.from_shards(shard_paths)
        return kept_ways(store, exclude_ids, keep.filter_spec, in_box)
    return _filter_way_columns(
        _read_way_columns(shard_paths), exclude_ids, keep, in_box
    )


def _collect_relations(shard_paths, keep):
    """Reassemble the spilled relations into the ``relations`` struct pyrosm's assembly
    expects (``id`` / ``members`` / ``tags`` / metadata), refined by the exact value filter
    ``keep``, and return it with the unique set of their way-member ids. ``(None, empty)``
    when no relation passes."""
    ids, members, tags, meta, way_member_ids = [], [], [], [], []
    for path in shard_paths:
        z = _open_shard(path)
        rid = z["rel_id"]
        if len(rid) == 0:
            continue
        memid, memoff, memtype, memrole, rtags, rmeta = (
            z["rel_memid"],
            z["rel_memoff"],
            z["rel_memtype"],
            z["rel_memrole"],
            z["rel_tags"],
            z["rel_meta"],
        )
        for k in range(len(rid)):
            if not keep(rtags[k]):
                continue
            s, e = memoff[k], memoff[k + 1]
            mids, mtypes = memid[s:e], memtype[s:e]
            ids.append(int(rid[k]))
            members.append(
                {
                    "member_id": mids,
                    "member_type": np.array(
                        [_MEMBER_TYPE[int(t)] for t in mtypes], dtype=object
                    ),
                    "member_role": memrole[s:e],
                }
            )
            way_member_ids.append(mids[mtypes == _WAY_MEMBER])
            tags.append(rtags[k])
            meta.append(rmeta[k])
    if not ids:
        return None, np.empty(0, np.int64)
    meta = np.array(meta, dtype=np.int64).reshape(-1, 4)
    relations = {
        "id": np.array(ids, dtype=np.int64),
        "members": _object_array(members),
        "tags": _object_array(tags),
        "version": meta[:, 0],
        "timestamp": meta[:, 1],
        "changeset": meta[:, 2],
        "visible": meta[:, 3].astype(bool),
    }
    member_ids = (
        np.unique(np.concatenate(way_member_ids))
        if way_member_ids
        else np.empty(0, np.int64)
    )
    return relations, member_ids


def _collect_relation_ways(shard_paths, member_ids, in_box=None):
    """Look up the member ways (id -> node refs) of the kept relations from the spilled
    all-ways store. Returned as a ``{id, nodes}`` dict sorted by id ascending -- pyrosm
    aligns member roles to the sorted member ids, so the ways must match that order. When
    ``in_box`` is given (a bounding-box read without ``complete_relations``), only member
    ways with a node in the box are kept, so the relation geometry is the in-box partial --
    matching the in-memory reader. ``None`` if there are no way members, or none present.
    """
    if len(member_ids) == 0:
        return None
    if _is_threaded(shard_paths):
        return relation_ways(shard_paths, member_ids, in_box)
    found = {}
    inside = None if in_box is None else set(in_box.tolist())
    for path in shard_paths:
        z = _open_shard(path)
        wid, off, refs = z["all_id"], z["all_refs_off"], z["all_refs"]
        if len(wid) == 0:
            continue
        pos = np.clip(np.searchsorted(member_ids, wid), 0, len(member_ids) - 1)
        for k in np.nonzero(member_ids[pos] == wid)[0]:
            way_refs = refs[off[k] : off[k + 1]]
            if inside is None or not inside.isdisjoint(way_refs.tolist()):
                found[int(wid[k])] = way_refs
    if not found:
        return None
    ids = np.array(sorted(found), dtype=np.int64)
    nodes = np.empty(len(ids), dtype=object)
    nodes[:] = [found[int(i)] for i in ids]
    return {"id": ids, "nodes": nodes}


def _restrict_relations_to_box(relations, present_ids):
    """Keep only the relations with at least one member way inside the box (``present_ids``)
    -- out-of-box relations are dropped by the final spatial filter anyway, and excluding
    them here keeps their tags from leaking as spurious all-None columns. Returns the
    filtered struct and its way-member ids (``(None, empty)`` if none qualify)."""
    mask = np.array(
        [
            not present_ids.isdisjoint(
                m["member_id"][m["member_type"] == b"way"].tolist()
            )
            for m in relations["members"]
        ],
        dtype=bool,
    )
    if not mask.any():
        return None, np.empty(0, np.int64)
    kept = {k: v[mask] for k, v in relations.items()}
    way_ids = [m["member_id"][m["member_type"] == b"way"] for m in kept["members"]]
    member_ids = (
        np.unique(np.concatenate(way_ids)) if way_ids else np.empty(0, np.int64)
    )
    return kept, member_ids


# The way fields that are not tags: the id, the node refs and the metadata.
_WAY_FIELDS = ("id", "nodes", "version", "timestamp", "visible")


def _way_column_keys(tags_as_columns, keep_metadata):
    """The way columns in the in-memory reader's order -- the requested tags, ``id``,
    ``nodes`` and, with ``keep_metadata``, ``timestamp`` and ``version`` -- and the tag
    columns among them."""
    keys = list(tags_as_columns) + ["id", "nodes"]
    if keep_metadata:
        keys += ["timestamp", "version"]
    keys = list(dict.fromkeys(keys))
    return keys, [k for k in keys if k not in _WAY_FIELDS]


def _route_tags(
    tags,
    version,
    timestamp,
    visible,
    tag_cols,
    keep_other_tags,
    json_version,
    json_timestamp,
    json_visible,
):
    """The tag columns and the JSON ``tags`` values of ways with the tag dicts ``tags``, as
    the in-memory reader's ``convert_way_records_to_lists`` routes them: ``(col_lists,
    leftover, has_leftover)``. A tag in ``tag_cols`` (one keyed ``id`` as ``id_tag``) goes
    to its column, None where a way lacks it. With ``keep_other_tags`` every other tag goes
    to the JSON, after the metadata fields the ``json_*`` flags name; ``leftover`` is None
    for a way with nothing there (and for every way without ``keep_other_tags``)."""
    n = len(tags)
    tag_col_set = set(tag_cols)
    col_lists = {k: [None] * n for k in tag_cols}
    leftover = [None] * n
    has_leftover = False
    for i in range(n):
        tag = tags[i]
        # ``other is None``: the JSON is not kept.
        other = {} if keep_other_tags else None
        if other is not None:
            if json_version:
                other["version"] = int(version[i])
            if json_timestamp:
                other["timestamp"] = int(timestamp[i])
            if json_visible:
                other["visible"] = bool(visible[i])
        for k, v in tag.items():
            key = "id_tag" if k == "id" else k
            if key in tag_col_set:
                col_lists[key][i] = v
            elif other is not None:
                other[key] = v
        if other:
            leftover[i] = dumps(other)
            has_leftover = True
    return col_lists, leftover, has_leftover


def _ways_arrays(
    cols,
    tags_as_columns,
    keep_metadata,
    keep_other_tags=True,
    columns=None,
    area_rule=True,
):
    """Convert the standalone way columns (from :func:`_collect_kept_ways`) into the
    ``way_elements`` dict (all occurring tag columns + the JSON ``tags`` column + metadata)
    pyrosm's assembly expects, matching the in-memory reader column for column.

    Routing mirrors the in-memory reader's ``convert_way_records_to_lists`` on the columns:
    the requested ``tags_as_columns`` become their own columns; every other tag -- and the
    metadata fields not promoted to a column (``version``/``timestamp`` without
    ``keep_metadata``, ``visible`` unless requested) -- goes into the JSON ``tags`` column in
    the same field order; a tag literally keyed ``id`` is surfaced as ``id_tag``. The tag
    columns plus that JSON column run through pyrosm's own ``columns_to_arrays`` (per-key
    dtypes, all-None drop), so they are byte-identical to the in-memory reader; ``id`` and the
    kept metadata go through the same conversion, and ``nodes`` is attached directly as the
    per-way node-ref arrays (it only feeds geometry and is dropped before output).

    With ``keep_other_tags=False`` the leftover JSON ``tags`` column is dropped by the caller,
    so it is not built at all here -- only the requested tag columns are filled. This skips a
    per-way ``rapidjson.dumps`` (the metadata/leftover serialisation) that would otherwise be
    discarded, which is a noticeable saving on country-scale minimal-tags reads.

    The columns come in the in-memory reader's order (see :func:`_way_column_keys`), then
    the JSON ``tags`` column. ``columns``, when given, are the tag columns of the whole
    result (:func:`_result_tag_columns`): one these ways lack is added as Nones, so every
    chunk of the result carries the same columns. With ``area_rule`` the ways also carry,
    under ``AREA_BY_TAGS``, whether their tags make them areas when closed (see
    :func:`pyrosm.geometry.tags_make_area`); a network's ways are lines and go without.
    """
    column_keys, tag_cols = _way_column_keys(tags_as_columns, keep_metadata)
    column_set = set(column_keys)
    version_col = "version" in column_set
    timestamp_col = "timestamp" in column_set
    visible_col = "visible" in column_set
    # The area rule of closed ways reads all of a way's tags, also those not output.
    area_by_tags = None
    if area_rule:
        if isinstance(cols, ElementStore):
            area_by_tags = cols.area_by_tags()
        else:
            area_by_tags = tags_make_area(cols["tags"])
    # ``tag_lists`` builds Arrow arrays; without pyarrow the store goes through tag dicts.
    if isinstance(cols, ElementStore) and not _compat.HAS_PYARROW:
        cols = cols.way_columns(None if keep_other_tags else set(tags_as_columns))
    if isinstance(cols, ElementStore):
        # The tag columns and the JSON come from the store's string buffers.
        store = cols
        col_lists, leftover = tag_lists(
            store, tags_as_columns, keep_metadata, keep_other_tags
        )
        has_leftover = any(value is not None for value in leftover)
        cols = {
            "id": store.id,
            "nodes": store.node_lists(),
            "version": store.version,
            "timestamp": store.timestamp,
            "visible": store.visible.astype(bool),
        }
    else:
        col_lists, leftover, has_leftover = _route_tags(
            cols["tags"],
            cols["version"],
            cols["timestamp"],
            cols["visible"],
            tag_cols,
            keep_other_tags,
            not version_col,
            not timestamp_col,
            not visible_col,
        )
    ids, nodes = cols["id"], cols["nodes"]
    version, timestamp, visible = cols["version"], cols["timestamp"], cols["visible"]
    n = len(ids)

    # Tag columns, id, and kept metadata go through pyrosm's own conversion (per-key dtypes,
    # all-None drop) so they match the in-memory reader; nodes are attached as arrays after.
    data = dict(col_lists)
    data["id"] = ids
    if version_col:
        data["version"] = version
    if timestamp_col:
        data["timestamp"] = timestamp
    if visible_col:
        data["visible"] = visible
    if has_leftover:
        data["tags"] = leftover
    ways = columns_to_arrays(data)
    ways["nodes"] = nodes
    for key in columns or ():
        if key not in ways:
            ways[key] = np.full(n, None, dtype=object)
    way_elements = {key: ways[key] for key in column_keys + ["tags"] if key in ways}
    if area_by_tags is not None:
        way_elements[AREA_BY_TAGS] = area_by_tags
    return way_elements


def _result_tag_columns(kept, tags_as_columns, keep_metadata, keep_other_tags=True):
    """The tag columns :func:`_ways_arrays` gives all the ``kept`` ways together: the
    requested tags that occur in any of them (a tag keyed ``id`` as ``id_tag``) and, when
    any way has leftover tags or metadata, the JSON ``tags`` column."""
    if isinstance(kept, ElementStore):
        return result_tag_columns(kept, tags_as_columns, keep_metadata, keep_other_tags)
    column_keys, tag_cols = _way_column_keys(tags_as_columns, keep_metadata)
    tag_col_set = set(tag_cols)
    tags = kept["tags"]
    columns = set()
    leftover = not {"version", "timestamp", "visible"}.issubset(column_keys)
    # The keys of a batch of ways at a time, so that only the requested ones are kept.
    batch = 10_000
    for start in range(0, len(tags), batch):
        keys = set().union(*tags[start : start + batch])
        if "id" in keys:
            keys.discard("id")
            keys.add("id_tag")
        columns |= keys & tag_col_set
        leftover = leftover or not keys <= tag_col_set
    if keep_other_tags and leftover and len(tags) > 0:
        columns.add("tags")
    return columns


def _node_refs(kept, relation_ways):
    """The node-ref arrays of the kept ways and of the relation member ways."""
    refs = []
    if isinstance(kept, ElementStore):
        refs.append(kept.refs)
    elif kept is not None:
        refs.extend(kept["nodes"])
    if relation_ways is not None:
        refs.extend(relation_ways["nodes"])
    return refs


def _needed_node_ids(kept, relation_ways):
    """The unique node ids referenced by the kept standalone ways and the relation
    member ways -- the only coordinates the gather has to pull off disk. Returned sorted
    (the node gather's searchsorted lookup needs it ordered); the unique set is built with
    pandas' hashtable rather than ``np.unique``'s sort, which is markedly faster on the
    tens of millions of refs a country-scale read produces."""
    import pandas as pd

    refs = _node_refs(kept, relation_ways)
    if not refs:
        return np.empty(0, np.int64)
    unique = pd.unique(np.concatenate(refs))
    unique.sort()
    return unique


def _collect_layer(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    keep_ways=True,
    keep_relations=True,
    bounding_box=None,
    complete_relations=False,
):
    """Shared collection for both output modes: node features, standalone ways, relations,
    their member ways and the node coordinates the ways/relations reference. ``filter_spec``
    is ``(osm_keys, data_filter, filter_type)``; the spilled key-presence candidates are
    refined here by pyrosm's exact value filter. ``keep_ways`` / ``keep_relations`` drop
    those element kinds from the output (``get_data_by_custom_criteria``). With a
    ``bounding_box`` only in-box ways/nodes are kept; relation member ways are likewise
    restricted to in-box (partial geometry) unless ``complete_relations``. ``None`` if there
    is nothing to assemble."""
    keep = _keep_fn(filter_spec)

    in_box = _in_box_nodes(shard_paths) if bounding_box is not None else None
    if keep_relations:
        relations, member_ids = _collect_relations(shard_paths, keep)
        if relations is not None and in_box is not None:
            # Restrict to relations with >=1 member way in the box (the rest are dropped by
            # the final spatial filter anyway; excluding them here keeps their tags from
            # creating spurious all-None columns and matches the in-memory completion scope).
            present = _collect_relation_ways(shard_paths, member_ids, in_box)
            present_ids = set() if present is None else set(present["id"].tolist())
            relations, member_ids = _restrict_relations_to_box(relations, present_ids)
        member_box = None if complete_relations else in_box
        relation_ways = (
            _collect_relation_ways(shard_paths, member_ids, member_box)
            if relations is not None
            else None
        )
        # No member way present (all outside the data/box) -> the relation can't be built.
        if relation_ways is None:
            relations = None
    else:
        # Without relations there is no member-way set, so matching ways that would have
        # been members stay as standalone ways (matching get_data_by_custom_criteria).
        relations, relation_ways, member_ids = None, None, np.empty(0, np.int64)
    kept = (
        _collect_kept_ways(shard_paths, member_ids, keep, in_box) if keep_ways else None
    )
    node_features = _collect_node_features(
        shard_paths, tags_as_columns, keep_metadata, keep
    )
    if kept is None and relations is None and node_features is None:
        return None
    node_coordinates = _coordinates(shard_paths, kept, relation_ways)
    return node_features, kept, relations, relation_ways, node_coordinates
