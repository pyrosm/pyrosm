"""Worker-side decode and spill.

A worker decodes a contiguous run of blobs and spills each block to its own shard as it is
decoded (so only one block's arrays are held in memory at a time). Each shard holds the node
coordinates, the matching layer features (ways, point nodes and relations -- selected by
filter-key presence) and *every* way (id + refs, for relation-member lookup).
"""

from pathlib import Path

import numpy as np

from pyrosm.primitive_block_decoder import decode_primitive_block
from pyrosm.engine.blobs import _read_block
from pyrosm.engine.bounding_box import _in_box_mask, _filter_features_to_box

# Per-worker globals, set by the pool initializer (or directly for the in-process path).
# ``_OSM_KEYS`` holds the layer's filter keys (utf-8 bytes) used to pre-select elements;
# ``_INCLUDE_NODES`` is False for layers that emit no node features (buildings, boundary);
# ``_BBOX_BOUNDS`` is ``(xmin, ymin, xmax, ymax)`` when reading a bounding box, else None.
_FILEPATH = None
_SHARD_DIR = None
_OSM_KEYS = None
_INCLUDE_NODES = True
_BBOX_BOUNDS = None
# When not None, the only tag keys (utf-8 bytes) to resolve into the element tag dicts
# (the ``keep_other_tags=False`` minimal-tags mode); None resolves every tag.
_REQUESTED_TAG_KEYS = None


def _init_worker(
    filepath, shard_dir, osm_keys, include_nodes, bbox_bounds, requested_tag_keys=None
):
    global _FILEPATH, _SHARD_DIR, _OSM_KEYS, _INCLUDE_NODES, _BBOX_BOUNDS
    global _REQUESTED_TAG_KEYS
    _FILEPATH = filepath
    _SHARD_DIR = shard_dir
    _OSM_KEYS = osm_keys
    _INCLUDE_NODES = include_nodes
    _BBOX_BOUNDS = bbox_bounds
    _REQUESTED_TAG_KEYS = requested_tag_keys


def _key_indices(string_table, osm_keys):
    """Positions of the filter keys in this block's string table (skipping keys this block
    never uses), so element selection is a fast integer ``isin`` on the key indices."""
    return [string_table.index(k) for k in osm_keys if k in string_table]


def _resolve_tags(string_table, keys, vals, start, end, keep_indices=None):
    """The ``{key: value}`` tag dict for one element, resolved through the string table
    (decoded to str, as pyrosm's protobuf path produces). When ``keep_indices`` is given,
    only the pairs whose key is one of those string-table indices are resolved -- the
    ``keep_other_tags=False`` minimal-tags mode that skips the unwanted tags entirely.
    """
    return {
        string_table[keys[p]]
        .decode("utf-8", "replace"): string_table[vals[p]]
        .decode("utf-8", "replace")
        for p in range(start, end)
        if keep_indices is None or keys[p] in keep_indices
    }


def _requested_keep_indices(string_table):
    """The string-table indices for ``_REQUESTED_TAG_KEYS`` (the minimal-tags keep set) in
    this block, or ``None`` when every tag should be resolved."""
    if _REQUESTED_TAG_KEYS is None:
        return None
    return set(_key_indices(string_table, _REQUESTED_TAG_KEYS))


def _matching_ways(string_table, ways, osm_keys):
    """Select the ways carrying any of ``osm_keys`` and return, per matching way, its
    node-ref slice, full (resolved) tag dict and ``version``/``timestamp``/``visible``
    metadata -- everything pyrosm's way record carries. Returns a dict of parallel
    arrays/lists, or ``None``."""
    if ways is None:
        return None
    key_indices = _key_indices(string_table, osm_keys)
    if not key_indices:
        return None
    keys, vals, tags_off = ways["keys"], ways["vals"], ways["tags_off"]
    key_positions = np.nonzero(np.isin(keys, key_indices))[0]
    if len(key_positions) == 0:
        return None
    # A tag key belongs to the way whose [tags_off[i], tags_off[i+1]) slice contains it;
    # a way may carry several filter keys, so de-duplicate.
    way_index = np.unique(np.searchsorted(tags_off, key_positions, side="right") - 1)
    refs, refs_off = ways["refs"], ways["refs_off"]
    keep_indices = _requested_keep_indices(string_table)
    return {
        "id": ways["id"][way_index],
        "refs": [refs[refs_off[i] : refs_off[i + 1]] for i in way_index],
        "tags": [
            _resolve_tags(
                string_table, keys, vals, tags_off[i], tags_off[i + 1], keep_indices
            )
            for i in way_index
        ],
        "version": ways["version"][way_index],
        "timestamp": ways["timestamp"][way_index],
        "visible": ways["visible"][way_index],
    }


def _matching_nodes(string_table, nodes, osm_keys, node_lon, node_lat):
    """Dense nodes carrying any of ``osm_keys`` -> a dict of parallel arrays/lists
    (``id`` / ``lon`` / ``lat`` / ``tags`` + ``version`` / ``timestamp`` / ``changeset`` /
    ``visible`` metadata) for the matching nodes, or ``None``. Tags are parsed from the
    block's dense ``keys_vals`` stream; untagged nodes (the vast majority) cost nothing.
    """
    if nodes is None:
        return None
    key_indices = set(_key_indices(string_table, osm_keys))
    keys_vals = nodes["keys_vals"]
    if not key_indices or len(keys_vals) == 0:
        return None
    keep_indices = _requested_keep_indices(string_table)
    ids = nodes["id"]
    idx, tags = [], []
    p, end = 0, len(keys_vals)
    for node_i in range(len(ids)):
        pairs = []
        matched = False
        while p < end and keys_vals[p] != 0:
            k, v = keys_vals[p], keys_vals[p + 1]
            p += 2
            pairs.append((k, v))
            if k in key_indices:
                matched = True
        p += 1  # skip the per-node 0 terminator
        if matched:
            idx.append(node_i)
            tags.append(
                {
                    string_table[k]
                    .decode("utf-8", "replace"): string_table[v]
                    .decode("utf-8", "replace")
                    for k, v in pairs
                    if keep_indices is None or k in keep_indices
                }
            )
    if not idx:
        return None
    idx = np.array(idx, dtype=np.int64)

    # DenseInfo metadata is optional; when a field is absent the decoder returns an empty
    # array, so default it like pyrosm's parse_dense (visible -> False, the rest -> 0).
    def meta(name):
        arr = nodes[name]
        return arr[idx] if len(arr) == len(ids) else np.zeros(len(idx), dtype=np.int64)

    return {
        "id": ids[idx],
        "lon": node_lon[idx],
        "lat": node_lat[idx],
        "tags": tags,
        "version": meta("version"),
        "timestamp": meta("timestamp"),
        "changeset": meta("changeset"),
        "visible": meta("visible"),
    }


def _node_records_by_id(string_table, header, nodes, wanted, keep_metadata):
    """Full records (id / lon / lat / tags + metadata) for the dense nodes whose id is in
    ``wanted`` -- the second-pass gather that builds the graph-export node frame's coordinate
    store. Tags are resolved through the string table; an untagged node gets ``tags=None`` and
    metadata is included only when ``keep_metadata`` (matching the in-memory reader). Returns a
    dict of parallel arrays/lists, or ``None`` when no wanted node is in this block."""
    if nodes is None:
        return None
    ids = nodes["id"]
    gran = header["granularity"]
    lat = (nodes["lat"] * gran + header["lat_offset"]) / 1e9
    lon = (nodes["lon"] * gran + header["lon_offset"]) / 1e9
    keys_vals = nodes["keys_vals"]
    idx, tags = [], []
    p, end = 0, len(keys_vals)
    for node_i in range(len(ids)):
        pairs = []
        while p < end and keys_vals[p] != 0:
            pairs.append((keys_vals[p], keys_vals[p + 1]))
            p += 2
        p += 1  # skip the per-node 0 terminator
        if ids[node_i] in wanted:
            idx.append(node_i)
            tags.append(
                {
                    string_table[k]
                    .decode("utf-8", "replace"): string_table[v]
                    .decode("utf-8", "replace")
                    for k, v in pairs
                }
                if pairs
                else None
            )
    if not idx:
        return None
    idx = np.array(idx, dtype=np.int64)

    def meta(name):
        arr = nodes[name]
        return arr[idx] if len(arr) == len(ids) else np.zeros(len(idx), dtype=np.int64)

    # 'visible' is kept regardless of keep_metadata (the in-memory node frame retains it);
    # version/timestamp/changeset are the metadata dropped when keep_metadata is False.
    record = {
        "id": ids[idx],
        "lon": lon[idx],
        "lat": lat[idx],
        "tags": tags,
        "visible": meta("visible").astype(bool),
    }
    if keep_metadata:
        record["version"] = meta("version")
        record["timestamp"] = meta("timestamp")
        record["changeset"] = meta("changeset")
    return record


def _layer_relations(string_table, relations, osm_keys):
    """The relations carrying any of ``osm_keys`` in this block. Yields, per relation, its
    id, member id/type/role arrays, full tag dict and ``version``/``timestamp``/
    ``changeset`` metadata -- everything pyrosm needs to assemble the (multi)polygon and
    its columns. Member roles and tags are resolved through the block's string table."""
    if relations is None:
        return
    key_indices = _key_indices(string_table, osm_keys)
    if not key_indices:
        return
    key_positions = np.nonzero(np.isin(relations["keys"], key_indices))[0]
    if len(key_positions) == 0:
        return
    tags_off = relations["tags_off"]
    rel_index = np.unique(np.searchsorted(tags_off, key_positions, side="right") - 1)
    ids, keys, vals = relations["id"], relations["keys"], relations["vals"]
    version, timestamp, changeset = (
        relations["version"],
        relations["timestamp"],
        relations["changeset"],
    )
    memids, types, roles, moff = (
        relations["memids"],
        relations["types"],
        relations["roles"],
        relations["members_off"],
    )
    keep_indices = _requested_keep_indices(string_table)
    for i in rel_index:
        s, e = moff[i], moff[i + 1]
        member_role = np.array(
            [string_table[r].decode("utf-8", "replace") for r in roles[s:e]],
            dtype=object,
        )
        tags = _resolve_tags(
            string_table, keys, vals, tags_off[i], tags_off[i + 1], keep_indices
        )
        yield {
            "id": ids[i],
            "memid": memids[s:e],
            "memtype": types[s:e],
            "memrole": member_role,
            "tags": tags,
            "version": version[i],
            "timestamp": timestamp[i],
            "changeset": changeset[i],
        }


def _offsets_from_lengths(lengths):
    """CSR offsets array for variable-length rows: ``[0, l0, l0+l1, ...]``."""
    off = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        np.cumsum(lengths, out=off[1:])
    return off


def _object_array(items):
    """1-D object array of ``items`` (dicts/arrays), avoiding numpy's 2-D coercion."""
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _decode_one_block(string_table, header, nodes, ways, relations):
    """Decode one primitive block into the per-shard arrays: node coordinates, the matching
    layer point nodes, the matching layer ways (refs + resolved tags + metadata), *all* ways
    (id + refs, for relation-member lookup) and the matching layer relations. Returns a dict
    of the npz arrays ``_write_shard`` spills (every key present, empty when absent)."""
    node_id, node_lon, node_lat, in_box = [], [], [], []
    nf = {k: [] for k in ("id", "lon", "lat", "tags", "meta")}
    way_match_id, way_match_refs, way_match_tags = [], [], []
    way_version, way_timestamp, way_visible = [], [], []
    all_id, all_refs, all_count = [], [], []
    rel = {k: [] for k in ("id", "memid", "memtype", "memrole", "tags", "meta")}

    if nodes is not None:
        gran = header["granularity"]
        lat = (nodes["lat"] * gran + header["lat_offset"]) / 1e9
        lon = (nodes["lon"] * gran + header["lon_offset"]) / 1e9
        node_id.append(nodes["id"])
        node_lat.append(lat)
        node_lon.append(lon)
        if _BBOX_BOUNDS is not None:
            in_box.append(nodes["id"][_in_box_mask(lon, lat, _BBOX_BOUNDS)])
        found = (
            _matching_nodes(string_table, nodes, _OSM_KEYS, lon, lat)
            if _INCLUDE_NODES
            else None
        )
        # A node feature is kept only if it falls inside the bounding box.
        if found is not None and _BBOX_BOUNDS is not None:
            found = _filter_features_to_box(found, _BBOX_BOUNDS)
        if found is not None:
            nf["id"].append(found["id"])
            nf["lon"].append(found["lon"])
            nf["lat"].append(found["lat"])
            nf["tags"].extend(found["tags"])
            nf["meta"].append(
                np.stack(
                    [
                        found["version"],
                        found["timestamp"],
                        found["changeset"],
                        found["visible"],
                    ],
                    axis=1,
                )
            )
    if ways is not None:
        all_id.append(ways["id"])
        all_refs.append(ways["refs"])
        all_count.append(np.diff(ways["refs_off"]))
        found = _matching_ways(string_table, ways, _OSM_KEYS)
        if found is not None:
            way_match_id.append(found["id"])
            way_match_refs.extend(found["refs"])
            way_match_tags.extend(found["tags"])
            way_version.append(found["version"])
            way_timestamp.append(found["timestamp"])
            way_visible.append(found["visible"])
    for r in _layer_relations(string_table, relations, _OSM_KEYS):
        rel["id"].append(r["id"])
        rel["memid"].append(r["memid"])
        rel["memtype"].append(r["memtype"])
        rel["memrole"].append(r["memrole"])
        rel["tags"].append(r["tags"])
        rel["meta"].append((r["version"], r["timestamp"], r["changeset"]))

    return {
        "node_id": np.concatenate(node_id) if node_id else np.empty(0, np.int64),
        "node_lon": np.concatenate(node_lon) if node_lon else np.empty(0),
        "node_lat": np.concatenate(node_lat) if node_lat else np.empty(0),
        "in_box_id": np.concatenate(in_box) if in_box else np.empty(0, np.int64),
        "nfeat_id": np.concatenate(nf["id"]) if nf["id"] else np.empty(0, np.int64),
        "nfeat_lon": np.concatenate(nf["lon"]) if nf["lon"] else np.empty(0),
        "nfeat_lat": np.concatenate(nf["lat"]) if nf["lat"] else np.empty(0),
        "nfeat_tags": _object_array(nf["tags"]),
        "nfeat_meta": (
            np.concatenate(nf["meta"]) if nf["meta"] else np.empty((0, 4), np.int64)
        ),
        "way_id": (
            np.concatenate(way_match_id) if way_match_id else np.empty(0, np.int64)
        ),
        "refs": (
            np.concatenate(way_match_refs) if way_match_refs else np.empty(0, np.int64)
        ),
        "refs_off": _offsets_from_lengths([len(r) for r in way_match_refs]),
        "way_tags": _object_array(way_match_tags),
        "way_version": (
            np.concatenate(way_version) if way_version else np.empty(0, np.int64)
        ),
        "way_timestamp": (
            np.concatenate(way_timestamp) if way_timestamp else np.empty(0, np.int64)
        ),
        "way_visible": (
            np.concatenate(way_visible) if way_visible else np.empty(0, np.int64)
        ),
        "all_id": np.concatenate(all_id) if all_id else np.empty(0, np.int64),
        "all_refs": np.concatenate(all_refs) if all_refs else np.empty(0, np.int64),
        "all_refs_off": _offsets_from_lengths(
            np.concatenate(all_count).tolist() if all_count else []
        ),
        "rel_id": np.array(rel["id"], dtype=np.int64),
        "rel_memid": (
            np.concatenate(rel["memid"]) if rel["memid"] else np.empty(0, np.int64)
        ),
        "rel_memoff": _offsets_from_lengths([len(m) for m in rel["memid"]]),
        "rel_memtype": (
            np.concatenate(rel["memtype"]) if rel["memtype"] else np.empty(0, np.int64)
        ),
        "rel_memrole": (
            np.concatenate(rel["memrole"])
            if rel["memrole"]
            else np.empty(0, dtype=object)
        ),
        "rel_tags": _object_array(rel["tags"]),
        "rel_meta": np.array(rel["meta"], dtype=np.int64).reshape(-1, 3),
    }


def _decode_batch(task):
    """Worker: decode a contiguous run of blobs, spilling each block to its own shard as it
    is decoded (so only one block's arrays are held in memory at a time, rather than the
    whole batch), and return the list of shard paths."""
    worker_id, blobs = task
    paths = []
    with open(_FILEPATH, "rb") as f:
        for seq, (offset, size) in enumerate(blobs):
            data = _read_block(f, offset, size)
            arrays = _decode_one_block(*decode_primitive_block(data))
            path = Path(_SHARD_DIR) / ("shard_%d_%d.npz" % (worker_id, seq))
            np.savez(path, **arrays)
            paths.append(path)
    return paths
