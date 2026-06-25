"""Main-process collection.

Read the spilled shards back into the layer's node, way and relation records, their
member ways and the referenced node coordinates -- pulling only what the kept features
need off disk, and refining the worker's key-presence candidates by pyrosm's exact value
filter so peak memory stays bounded by the working set.
"""

import numpy as np
from rapidjson import dumps

from pyrosm._arrays import columns_to_arrays
from pyrosm.tagparser import explode_node_tag_array
from pyrosm.data_filter import element_should_be_kept
from pyrosm.engine.decode import _object_array
from pyrosm.engine.bounding_box import _in_box_nodes

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
_WAY_COLUMNS = ("id", "nodes", "tags", "version", "timestamp", "visible")


def _read_way_columns(shard_paths):
    """Read the spilled matching ways back as a dict of parallel arrays (the columns in
    ``_WAY_COLUMNS``). ``None`` if no shard holds a way."""
    ids, nodes_list, tags, vers, tss, viss = [], [], [], [], [], []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid = z["way_id"]
        if len(wid) == 0:
            continue
        off, refs = z["refs_off"], z["refs"]
        ids.append(wid)
        # Per-way node-ref arrays (views into the shard's flat refs); no per-way list build.
        nodes_list.extend(np.split(refs, off[1:-1]) if len(wid) > 1 else [refs])
        tags.append(z["way_tags"])
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
    """Number of standalone ways in a way-column dict."""
    return len(cols["id"])


def _slice_way_columns(cols, start, stop):
    """A row slice ``[start:stop]`` of every column -- the chunking the GeoParquet streamer
    uses in place of slicing a list of records."""
    return {k: v[start:stop] for k, v in cols.items()}


def _concat_way_columns(parts):
    """Concatenate per-partition way-column dicts (from :func:`_read_way_columns`) back into
    one, preserving order -- the parallel way read partitions the shards into contiguous
    ordered ranges, so concatenating the parts in order reproduces the serial shard order.
    ``None`` if every part is empty."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts]) for k in _WAY_COLUMNS}


def _collect_node_features(shard_paths, tags_as_columns, keep_metadata, keep):
    """Read the spilled node features back, refine them by the exact value filter ``keep``
    and explode their tags into the same columns the in-memory reader builds (``id`` /
    ``lon`` / ``lat`` / ``visible`` + metadata when ``keep_metadata``, plus the layer tag
    columns and a JSON ``tags`` column). ``None`` if no node passes."""
    ids, lon, lat, tags, meta = [], [], [], [], []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        nid = z["nfeat_id"]
        if len(nid) == 0:
            continue
        ids.append(nid)
        lon.append(z["nfeat_lon"])
        lat.append(z["nfeat_lat"])
        tags.append(z["nfeat_tags"])
        meta.append(z["nfeat_meta"])
    if not ids:
        return None
    tag_array = np.concatenate(tags)
    mask = np.fromiter((keep(t) for t in tag_array), dtype=bool, count=len(tag_array))
    if not mask.any():
        return None
    meta = np.concatenate(meta)[mask]
    node_arrays = {
        "id": np.concatenate(ids)[mask],
        "lon": np.concatenate(lon)[mask],
        "lat": np.concatenate(lat)[mask],
        "tags": tag_array[mask],
        "visible": meta[:, 3].astype(bool),
    }
    if keep_metadata:
        node_arrays["version"] = meta[:, 0]
        node_arrays["timestamp"] = meta[:, 1]
        node_arrays["changeset"] = meta[:, 2]
    # Mirror get_osm_nodes: merge the exploded tag columns in (the original 'tags' dict
    # array stays only when no node had leftover tags, exactly as the in-memory reader).
    node_arrays.update(
        explode_node_tag_array(node_arrays["tags"], list(tags_as_columns))
    )
    return node_arrays


def _gather_node_shards(shard_subset, needed):
    """Gather the coordinates of the ``needed`` (sorted) node ids that a subset of shards
    holds. Returns ``(pos, lon, lat)``: ``pos`` indexes into ``needed`` for the ids found, and
    ``lon`` / ``lat`` their coordinates. The per-shard searchsorted lookup is the same work
    the serial gather does; splitting the shards across workers is what parallelises it.
    """
    n = len(needed)
    pos_parts, lon_parts, lat_parts = [], [], []
    for path in shard_subset:
        z = np.load(path, allow_pickle=True)
        nid = z["node_id"]
        if len(nid) == 0:
            continue
        pos = np.clip(np.searchsorted(needed, nid), 0, n - 1)
        hit = needed[pos] == nid
        if hit.any():
            pos_parts.append(pos[hit])
            lon_parts.append(z["node_lon"][hit])
            lat_parts.append(z["node_lat"][hit])
    if not pos_parts:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    return (
        np.concatenate(pos_parts),
        np.concatenate(lon_parts),
        np.concatenate(lat_parts),
    )


def _scatter_node_coords(parts, needed):
    """Scatter the per-partition ``(pos, lon, lat)`` gathers into one coordinate store for
    ``needed``, dropping ids no shard held (left ``NaN``). Order-independent: each hit is
    written at its own ``needed`` position, so the partitions may arrive in any order.
    """
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    lon = np.full(len(needed), np.nan)
    lat = np.full(len(needed), np.nan)
    for pos, lon_part, lat_part in parts:
        lon[pos] = lon_part
        lat[pos] = lat_part
    present = ~np.isnan(lon)
    return NodeLocations(
        pd.DataFrame({"id": needed[present], "lon": lon[present], "lat": lat[present]})
    )


# Worker-side state for the parallel node gather: the sorted ``needed`` array, shared
# read-only via a memory-mapped ``.npy`` so the OS page cache backs it across workers rather
# than pickling it to each.
_GATHER_NEEDED = None


def _init_node_gather(needed_path):
    global _GATHER_NEEDED
    _GATHER_NEEDED = np.load(needed_path, mmap_mode="r")


def _node_gather_worker(shard_subset):
    return _gather_node_shards(shard_subset, _GATHER_NEEDED)


def _node_lookup(shard_paths, needed, workers=1):
    """Gather only the coordinates of ``needed`` node ids from the shards (bounded memory) and
    wrap them in a ``NodeLocations`` for geometry assembly. With ``workers > 1`` the shards are
    split across a process pool -- each worker gathers its shards against ``needed`` (shared
    read-only via a memory-mapped ``.npy``) and the hits are scattered back -- which
    parallelises the dominant per-shard decompress + searchsorted work; ``workers == 1`` runs
    the same gather in this process."""
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    if len(needed) == 0:
        # Node-only result (the filter matched no ways/relations): no coordinates needed.
        empty = np.empty(0, np.int64)
        return NodeLocations(
            pd.DataFrame({"id": empty, "lon": np.empty(0), "lat": np.empty(0)})
        )

    if workers > 1:
        from pathlib import Path

        from pyrosm.engine.pool import _run_pool

        needed_path = str(Path(shard_paths[0]).parent / "needed.npy")
        np.save(needed_path, needed)
        partitions = [shard_paths[i::workers] for i in range(workers)]
        parts, _ = _run_pool(
            _node_gather_worker, partitions, workers, _init_node_gather, (needed_path,)
        )
        return _scatter_node_coords(parts, needed)

    return _scatter_node_coords([_gather_node_shards(shard_paths, needed)], needed)


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


def _gather_node_records(filepath, node_ids, keep_metadata):
    """Second pass over the file gathering the full records (coordinates + tags + metadata)
    of ``node_ids``, returned as a rich ``NodeLocations`` -- the coordinate store the
    graph-export node frame is built from (its records carry the node tags and metadata the
    lean coordinate lookup omits). Only the requested (network) nodes are materialised, so
    peak memory stays bounded by the graph's node set rather than the whole file."""
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations
    from pyrosm.primitive_block_decoder import decode_primitive_block
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.engine.decode import _node_records_by_id

    wanted = set(node_ids.tolist())
    cols = ["id", "lon", "lat", "visible"]
    if keep_metadata:
        cols += ["version", "timestamp", "changeset"]
    arrays = {c: [] for c in cols}
    tags = []
    with open(filepath, "rb") as f:
        for blob_type, offset, size in _index_blobs(filepath):
            if blob_type != "OSMData":
                continue
            st, header, nodes, _, _ = decode_primitive_block(
                _read_block(f, offset, size)
            )
            rec = _node_records_by_id(st, header, nodes, wanted, keep_metadata)
            if rec is not None:
                for c in cols:
                    arrays[c].append(rec[c])
                tags.extend(rec["tags"])
    frame = {
        c: (
            np.concatenate(arrays[c])
            if arrays[c]
            else np.empty(0, _NODE_RECORD_DTYPES[c])
        )
        for c in cols
    }
    frame["tags"] = tags
    return NodeLocations(pd.DataFrame(frame))


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
    (network ``all`` / ``driving_psv``) keeps every candidate. Rebuilt from the picklable
    ``filter_spec`` inside each parallel worker, so the serial and parallel reads filter
    identically."""
    osm_keys, data_filter, filter_type = filter_spec

    def keep(tag):
        return data_filter is None or element_should_be_kept(
            tag, osm_keys, data_filter, filter_type
        )

    return keep


def _contiguous_partitions(items, workers):
    """Split ``items`` into ``workers`` contiguous (non-strided) blocks, dropping empties --
    so concatenating the per-block results in order reproduces the serial order."""
    per = (len(items) + workers - 1) // workers
    blocks = [items[i * per : (i + 1) * per] for i in range(workers)]
    return [b for b in blocks if b]


# Worker-side state for the parallel standalone-way read: the value-filter predicate plus the
# relation-member exclusion set and the bounding-box node set, set once per worker.
_WAY_COLLECT_STATE = None


def _init_way_collect(filter_spec, exclude_ids, in_box):
    global _WAY_COLLECT_STATE
    _WAY_COLLECT_STATE = (_keep_fn(filter_spec), exclude_ids, in_box)


def _way_collect_worker(shard_subset):
    keep, exclude_ids, in_box = _WAY_COLLECT_STATE
    return _filter_way_columns(
        _read_way_columns(shard_subset), exclude_ids, keep, in_box
    )


def _collect_kept_ways(
    shard_paths, exclude_ids, keep, in_box=None, workers=1, filter_spec=None
):
    """The standalone way columns to output: read the spilled candidates and refine them by
    the value filter, the bounding box, and the relation-member exclusion (see
    :func:`_filter_way_columns`). With ``workers > 1`` (and a ``filter_spec`` to rebuild the
    predicate worker-side) the shards are split into contiguous ordered ranges across a process
    pool and the per-range columns concatenated back in order -- identical rows in identical
    order to the serial read. ``None`` if nothing survives."""
    if workers > 1 and filter_spec is not None:
        from pyrosm.engine.pool import _run_pool

        partitions = _contiguous_partitions(shard_paths, workers)
        parts, _ = _run_pool(
            _way_collect_worker,
            partitions,
            workers,
            _init_way_collect,
            (filter_spec, exclude_ids, in_box),
        )
        return _concat_way_columns(parts)
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
        z = np.load(path, allow_pickle=True)
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
    meta = np.array(meta, dtype=np.int64).reshape(-1, 3)
    relations = {
        "id": np.array(ids, dtype=np.int64),
        "members": _object_array(members),
        "tags": _object_array(tags),
        "version": meta[:, 0],
        "timestamp": meta[:, 1],
        "changeset": meta[:, 2],
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
    found = {}
    inside = None if in_box is None else set(in_box.tolist())
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
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


def _ways_arrays(cols, tags_as_columns, keep_metadata, keep_other_tags=True):
    """Convert the standalone way columns (from :func:`_collect_kept_ways`) into the
    ``way_elements`` dict (all occurring tag columns + the JSON ``tags`` column + metadata)
    pyrosm's assembly expects, matching the in-memory reader column for column.

    Routing mirrors ``explode_way_tags`` + ``way_records_to_arrays`` directly on the columns:
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
    discarded, which is a noticeable saving on country-scale minimal-tags reads."""
    ids, nodes, tags = cols["id"], cols["nodes"], cols["tags"]
    version, timestamp, visible = cols["version"], cols["timestamp"], cols["visible"]
    n = len(ids)
    column_keys = list(dict.fromkeys(list(tags_as_columns) + ["id", "nodes"]))
    if keep_metadata:
        column_keys += ["timestamp", "version"]
    column_set = set(column_keys)
    structural = ("id", "nodes", "version", "timestamp", "visible")
    tag_cols = [k for k in column_keys if k not in structural]
    tag_col_set = set(tag_cols)
    version_col = "version" in column_set
    timestamp_col = "timestamp" in column_set
    visible_col = "visible" in column_set

    # One pass over the tag dicts: fill the requested tag columns (None where absent) and --
    # only when the leftover JSON is kept -- the ``tags`` column, replicating explode_way_tags
    # + way_records_to_arrays field ordering. ``other is None`` is the skip-leftovers sentinel.
    col_lists = {k: [None] * n for k in tag_cols}
    leftover = [None] * n
    has_leftover = False
    for i in range(n):
        tag = tags[i]
        other = {} if keep_other_tags else None
        if other is not None:
            if not version_col:
                other["version"] = int(version[i])
            if not timestamp_col:
                other["timestamp"] = int(timestamp[i])
            if not visible_col:
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
    return ways


def _needed_node_ids(kept, relation_ways):
    """The unique node ids referenced by the kept standalone ways and the relation
    member ways -- the only coordinates the gather has to pull off disk. Returned sorted
    (the node gather's searchsorted lookup needs it ordered); the unique set is built with
    pandas' hashtable rather than ``np.unique``'s sort, which is markedly faster on the
    tens of millions of refs a country-scale read produces."""
    import pandas as pd

    refs = []
    if kept is not None:
        refs.extend(kept["nodes"])
    if relation_ways is not None:
        refs.extend(relation_ways["nodes"])
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
    workers=1,
):
    """Shared collection for both output modes: node features, standalone ways, relations,
    their member ways and the node coordinates the ways/relations reference. ``filter_spec``
    is ``(osm_keys, data_filter, filter_type)``; the spilled key-presence candidates are
    refined here by pyrosm's exact value filter. ``keep_ways`` / ``keep_relations`` drop
    those element kinds from the output (``get_data_by_custom_criteria``). With a
    ``bounding_box`` only in-box ways/nodes are kept; relation member ways are likewise
    restricted to in-box (partial geometry) unless ``complete_relations``. ``workers > 1``
    splits the standalone-way read and the node-coordinate gather (the dominant costs) across a
    process pool; the comparatively small relation and node-feature reads stay serial.
    ``None`` if there is nothing to assemble."""
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
        _collect_kept_ways(
            shard_paths,
            member_ids,
            keep,
            in_box,
            workers=workers,
            filter_spec=filter_spec,
        )
        if keep_ways
        else None
    )
    node_features = _collect_node_features(
        shard_paths, tags_as_columns, keep_metadata, keep
    )
    if kept is None and relations is None and node_features is None:
        return None
    node_coordinates = _node_lookup(
        shard_paths, _needed_node_ids(kept, relation_ways), workers
    )
    return node_features, kept, relations, relation_ways, node_coordinates
