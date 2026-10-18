"""Worker-side decode and spill.

A worker decodes a contiguous run of blobs and accumulates the decoded blocks into a shard,
spilling it once it reaches ``_SHARD_TARGET_BYTES`` (so a country file produces a few hundred
shards rather than one per ~8k-element block -- far less per-file open/read overhead when
collect re-reads them -- while peak memory stays bounded by one shard's worth of blocks).
Each shard holds the node coordinates, the matching layer features (ways, point nodes and
relations -- selected by filter-key presence) and, for reads that look up relation members,
*every* way (id + refs).
"""

from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import numpy as np

from pyrosm._arrays import _offsets_from_lengths
from pyrosm.primitive_block_decoder import (
    BLOCK_COUNTS,
    decode_batch_nogil,
    decode_primitive_block,
)
from pyrosm.way_columns import take
from pyrosm.engine.blobs import _inflated_batch, _read_batches, _read_block, _read_span
from pyrosm.engine.bounding_box import _in_box_mask, _filter_features_to_box

# Accumulate decoded blocks into a shard until it reaches roughly this many bytes, then spill.
# Coarser shards amortise the per-file open/read overhead collect pays across thousands of
# tiny per-block files; ~8 MB keeps the spill near the point past which that overhead stops
# shrinking, while bounding a worker's in-flight memory to about one shard.
_SHARD_TARGET_BYTES = 8_000_000

# The CSR offset arrays in a shard dict: when several blocks' shard dicts are merged into one,
# these are rebuilt from the concatenated per-row lengths (the flat arrays just concatenate).
_OFFSET_KEYS = (
    "refs_off",
    "all_refs_off",
    "rel_memoff",
    "way_pairs_off",
    "nfeat_pairs_off",
    "str_off",
)
# The string-id arrays in a threaded decode's shard dict (ids of the strings ``str_off`` /
# ``str_data`` hold): merged shards renumber them past the strings of the blocks before.
_STRING_ID_KEYS = ("way_pair_key", "way_pair_val", "nfeat_pair_key", "nfeat_pair_val")


@dataclass(frozen=True)
class _DecodeSettings:
    """What the decode of one read needs, passed to each of its tasks: the file, the folder
    its shards are spilled to, the layer's filter keys (utf-8 bytes) used to pre-select
    elements (``None`` selects every tagged element), whether point nodes are kept (False
    for layers that emit none: buildings, boundaries) and the bounding box ``(xmin, ymin,
    xmax, ymax)`` (or None). When ``requested_tag_keys`` (utf-8 bytes) is not None, only
    those tag keys are resolved into the element tag dicts (the ``keep_other_tags=False``
    minimal-tags mode). ``spill_all_ways`` spills every way's id and node refs, which only
    the lookup of relation members reads."""

    filepath: str
    shard_dir: str
    osm_keys: list = None
    include_nodes: bool = True
    bbox_bounds: tuple = None
    requested_tag_keys: list = None
    spill_all_ways: bool = True


def _key_indices(string_table, osm_keys):
    """Positions of the filter keys in this block's string table (skipping keys this block
    never uses), so element selection is a fast integer ``isin`` on the key indices."""
    return [string_table.index(k) for k in osm_keys if k in string_table]


def _key_bytes(key):
    """The bytes of the requested tag key ``key`` that a block's string table holds for it. A
    key UTF-8 cannot encode (a lone surrogate) is encoded with ``surrogatepass``: only a
    string that is not valid UTF-8 can hold those bytes, and the tags of an element with one
    come from its strings decoded with ``"replace"``, which never give the key."""
    return key.encode("utf-8", "surrogatepass")


def _lazy_string_table(string_table):
    """A function returning the block's string table as str (decoded as pyrosm's protobuf
    path decodes it). It decodes the table on its first call only, so the block's way and
    relation tags decode each string once and share their string objects."""
    table = []

    def strings():
        if not table:
            table.append([s.decode("utf-8", "replace") for s in string_table])
        return table[0]

    return strings


def _resolve_tags(strings, keys, vals, start, end, keep_indices=None):
    """The ``{key: value}`` tag dict for one element, resolved through the block's decoded
    string table ``strings``; ``keys``/``vals`` are lists of string ids. When
    ``keep_indices`` is given, only the pairs whose key is one of those string-table
    indices are resolved -- the ``keep_other_tags=False`` minimal-tags mode that skips the
    unwanted tags entirely.
    """
    if keep_indices is None:
        return {strings[keys[p]]: strings[vals[p]] for p in range(start, end)}
    return {
        strings[keys[p]]: strings[vals[p]]
        for p in range(start, end)
        if keys[p] in keep_indices
    }


def _requested_keep_indices(string_table, requested_tag_keys):
    """The string-table indices of ``requested_tag_keys`` (the minimal-tags keep set) in
    this block, or ``None`` when every tag should be resolved (``requested_tag_keys`` is
    None)."""
    if requested_tag_keys is None:
        return None
    return set(_key_indices(string_table, requested_tag_keys))


def _tagged_elements(string_table, elements, osm_keys):
    """Indices of the ways or relations in ``elements`` carrying any of ``osm_keys``, or
    any tag at all when ``osm_keys`` is ``None`` (a read with no key filter)."""
    tags_off = elements["tags_off"]
    if osm_keys is None:
        return np.flatnonzero(np.diff(tags_off) > 0)
    key_indices = _key_indices(string_table, osm_keys)
    if not key_indices:
        return np.empty(0, dtype=np.int64)
    key_positions = np.nonzero(np.isin(elements["keys"], key_indices))[0]
    # A tag key belongs to the element whose [tags_off[i], tags_off[i+1]) slice contains
    # it; an element may carry several filter keys, so de-duplicate.
    return np.unique(np.searchsorted(tags_off, key_positions, side="right") - 1)


def _matching_ways(string_table, ways, osm_keys, strings=None, requested_tag_keys=None):
    """Select the ways carrying any of ``osm_keys`` (any tag when ``None``) and return,
    per matching way, its node-ref slice, full (resolved) tag dict and
    ``version``/``timestamp``/``visible`` metadata -- everything pyrosm's way record
    carries. ``strings`` is the block's :func:`_lazy_string_table` (made here when not
    given); with ``requested_tag_keys`` only those tags are resolved. Returns a dict of
    parallel arrays/lists, or ``None``."""
    if ways is None:
        return None
    way_index = _tagged_elements(string_table, ways, osm_keys)
    if len(way_index) == 0:
        return None
    keys, vals, tags_off = ways["keys"], ways["vals"], ways["tags_off"]
    refs, refs_off = ways["refs"], ways["refs_off"]
    keep_indices = _requested_keep_indices(string_table, requested_tag_keys)
    strings = (strings or _lazy_string_table(string_table))()
    keys, vals, tags_off = keys.tolist(), vals.tolist(), tags_off.tolist()
    return {
        "id": ways["id"][way_index],
        "refs": [refs[refs_off[i] : refs_off[i + 1]] for i in way_index],
        "tags": [
            _resolve_tags(
                strings, keys, vals, tags_off[i], tags_off[i + 1], keep_indices
            )
            for i in way_index.tolist()
        ],
        "version": ways["version"][way_index],
        "timestamp": ways["timestamp"][way_index],
        "visible": ways["visible"][way_index],
    }


def _matching_nodes(
    string_table, nodes, osm_keys, node_lon, node_lat, requested_tag_keys=None
):
    """Dense nodes carrying any of ``osm_keys`` (any tag when ``None``) -> a dict of
    parallel arrays/lists (``id`` / ``lon`` / ``lat`` / ``tags`` + ``version`` /
    ``timestamp`` / ``changeset`` / ``visible`` metadata) for the matching nodes, or
    ``None``; with ``requested_tag_keys`` only those tags are resolved. Tags are parsed
    from the block's dense ``keys_vals`` stream; untagged nodes (the vast majority) cost
    nothing.
    """
    if nodes is None:
        return None
    key_indices = (
        None if osm_keys is None else set(_key_indices(string_table, osm_keys))
    )
    keys_vals = nodes["keys_vals"]
    if key_indices == set() or len(keys_vals) == 0:
        return None
    keep_indices = _requested_keep_indices(string_table, requested_tag_keys)
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
            if key_indices is None or k in key_indices:
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

    return {
        "id": ids[idx],
        "lon": node_lon[idx],
        "lat": node_lat[idx],
        "tags": tags,
        "version": _dense_meta(nodes, "version", idx),
        "timestamp": _dense_meta(nodes, "timestamp", idx),
        "changeset": _dense_meta(nodes, "changeset", idx),
        "visible": _dense_meta(nodes, "visible", idx),
    }


def _dense_meta(nodes, name, idx):
    """The DenseInfo column ``name`` of the dense nodes ``idx``. The decoder returns an
    empty array for a column the block lacks, read like the in-memory parse: ``visible``
    as 1 (as the PBF format reads it), the rest as 0."""
    arr = nodes[name]
    if len(arr) == len(nodes["id"]):
        return arr[idx]
    return np.full(len(idx), 1 if name == "visible" else 0, dtype=np.int64)


def _node_records_by_id(string_table, header, nodes, wanted, keep_metadata, tags):
    """Full records (id / lon / lat + metadata) for the dense nodes whose id is in
    ``wanted`` (an :class:`~pyrosm.int64_hash.Int64Set`) -- the second-pass gather that
    builds the graph-export node frame's coordinate store. Their tags, resolved through the
    string table, are added to ``tags`` (a :class:`~pyrosm.way_store.NodeTagsBuilder`);
    metadata is included only when ``keep_metadata`` (matching the in-memory reader).
    Returns a dict of parallel arrays, or ``None`` when no wanted node is in this block.
    """
    if nodes is None:
        return None
    ids = nodes["id"]
    keep = wanted.isin(ids)
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return None
    gran = header["granularity"]
    lat = (nodes["lat"][idx] * gran + header["lat_offset"]) / 1e9
    lon = (nodes["lon"][idx] * gran + header["lon_offset"]) / 1e9
    tags.add(nodes["keys_vals"], len(ids), _lazy_string_table(string_table)(), keep)

    # 'visible' is kept regardless of keep_metadata (the in-memory node frame retains it);
    # version/timestamp/changeset are the metadata dropped when keep_metadata is False.
    record = {
        "id": ids[idx],
        "lon": lon,
        "lat": lat,
        "visible": _dense_meta(nodes, "visible", idx).astype(bool),
    }
    if keep_metadata:
        record["version"] = _dense_meta(nodes, "version", idx)
        record["timestamp"] = _dense_meta(nodes, "timestamp", idx)
        record["changeset"] = _dense_meta(nodes, "changeset", idx)
    return record


def _layer_relations(
    string_table, relations, osm_keys, strings=None, requested_tag_keys=None
):
    """The relations carrying any of ``osm_keys`` (any tag when ``None``) in this block.
    Yields, per relation, its id, member id/type/role arrays, full tag dict and
    ``version``/``timestamp``/``changeset``/``visible`` metadata -- everything pyrosm needs to
    assemble the (multi)polygon and its columns. Member roles and tags are resolved
    through the block's string table, ``strings`` being its :func:`_lazy_string_table`
    (made here when not given); with ``requested_tag_keys`` only those tags are
    resolved."""
    if relations is None:
        return
    rel_index = _tagged_elements(string_table, relations, osm_keys)
    if len(rel_index) == 0:
        return
    tags_off = relations["tags_off"]
    ids, keys, vals = relations["id"], relations["keys"], relations["vals"]
    version, timestamp, changeset, visible = (
        relations["version"],
        relations["timestamp"],
        relations["changeset"],
        relations["visible"],
    )
    memids, types, roles, moff = (
        relations["memids"],
        relations["types"],
        relations["roles"],
        relations["members_off"],
    )
    keep_indices = _requested_keep_indices(string_table, requested_tag_keys)
    strings = (strings or _lazy_string_table(string_table))()
    keys, vals = keys.tolist(), vals.tolist()
    for i in rel_index:
        s, e = moff[i], moff[i + 1]
        member_role = np.array([strings[r] for r in roles[s:e]], dtype=object)
        tags = _resolve_tags(
            strings, keys, vals, tags_off[i], tags_off[i + 1], keep_indices
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
            "visible": visible[i],
        }


def _object_array(items):
    """1-D object array of ``items`` (dicts/arrays), avoiding numpy's 2-D coercion."""
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _decode_one_block(settings, string_table, header, nodes, ways, relations):
    """Decode one primitive block into the per-shard arrays of a read with the
    :class:`_DecodeSettings` ``settings``: node coordinates, the matching layer point
    nodes, the matching layer ways (refs + resolved tags + metadata), *all* ways (id + refs,
    for relation-member lookup; empty unless ``settings.spill_all_ways``) and the matching
    layer relations. Returns a dict of the npz arrays ``_write_shard`` spills (every key
    present, empty when absent)."""
    node_id, node_lon, node_lat, in_box = [], [], [], []
    nf = {k: [] for k in ("id", "lon", "lat", "tags", "meta")}
    way_match_id, way_match_refs, way_match_tags = [], [], []
    way_version, way_timestamp, way_visible = [], [], []
    all_id, all_refs, all_count = [], [], []
    rel = {k: [] for k in ("id", "memid", "memtype", "memrole", "tags", "meta")}
    strings = _lazy_string_table(string_table)
    osm_keys, bbox_bounds = settings.osm_keys, settings.bbox_bounds
    requested = settings.requested_tag_keys

    if nodes is not None:
        gran = header["granularity"]
        lat = (nodes["lat"] * gran + header["lat_offset"]) / 1e9
        lon = (nodes["lon"] * gran + header["lon_offset"]) / 1e9
        node_id.append(nodes["id"])
        node_lat.append(lat)
        node_lon.append(lon)
        if bbox_bounds is not None:
            in_box.append(nodes["id"][_in_box_mask(lon, lat, bbox_bounds)])
        found = (
            _matching_nodes(string_table, nodes, osm_keys, lon, lat, requested)
            if settings.include_nodes
            else None
        )
        # A node feature is kept only if it falls inside the bounding box.
        if found is not None and bbox_bounds is not None:
            found = _filter_features_to_box(found, bbox_bounds)
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
        if settings.spill_all_ways:
            all_id.append(ways["id"])
            all_refs.append(ways["refs"])
            all_count.append(np.diff(ways["refs_off"]))
        found = _matching_ways(string_table, ways, osm_keys, strings, requested)
        if found is not None:
            way_match_id.append(found["id"])
            way_match_refs.extend(found["refs"])
            way_match_tags.extend(found["tags"])
            way_version.append(found["version"])
            way_timestamp.append(found["timestamp"])
            way_visible.append(found["visible"])
    for r in _layer_relations(string_table, relations, osm_keys, strings, requested):
        rel["id"].append(r["id"])
        rel["memid"].append(r["memid"])
        rel["memtype"].append(r["memtype"])
        rel["memrole"].append(r["memrole"])
        rel["tags"].append(r["tags"])
        rel["meta"].append((r["version"], r["timestamp"], r["changeset"], r["visible"]))

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
        "rel_meta": np.array(rel["meta"], dtype=np.int64).reshape(-1, 4),
    }


def _merge_shards(blocks):
    """Merge a batch of per-block shard dicts into the single shard dict a shard spanning those
    blocks would hold: flat arrays concatenate; the CSR offset arrays are rebuilt from the
    concatenated per-row lengths so the combined offsets stay correct. A single-block batch is
    returned unchanged."""
    if len(blocks) == 1:
        return blocks[0]
    merged = {}
    for key in blocks[0]:
        if key in _OFFSET_KEYS:
            lengths = np.concatenate([np.diff(b[key]) for b in blocks])
            merged[key] = _offsets_from_lengths(lengths)
        elif key in _STRING_ID_KEYS:
            bases = _offsets_from_lengths([len(b["str_off"]) - 1 for b in blocks])
            merged[key] = np.concatenate(
                [b[key] + base for b, base in zip(blocks, bases[:-1].tolist())]
            )
        else:
            merged[key] = np.concatenate([b[key] for b in blocks])
    return merged


def _decode_batch(settings, task, advance=None):
    """Worker: decode a contiguous run of blobs of the read with the :class:`_DecodeSettings`
    ``settings``, accumulating decoded blocks into a shard and spilling it once it reaches
    ``_SHARD_TARGET_BYTES`` (so peak memory stays bounded by about one shard's worth of
    blocks rather than the whole batch, while the file count -- and the per-file overhead
    collect pays re-reading them -- drops by one to two orders of magnitude). ``advance``,
    when given, is called with each decoded blob's length in the file. Returns the list of
    shard paths."""
    task_id, blobs = task
    paths = []
    pending = []
    pending_bytes = 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        path = Path(settings.shard_dir) / ("shard_%d_%d.npz" % (task_id, len(paths)))
        np.savez(path, **_merge_shards(pending))
        paths.append(path)
        pending, pending_bytes = [], 0

    with open(settings.filepath, "rb") as f:
        for offset, size, span in blobs:
            data = _read_block(f, offset, size)
            arrays = _decode_one_block(settings, *decode_primitive_block(data))
            pending.append(arrays)
            pending_bytes += sum(v.nbytes for v in arrays.values())
            if pending_bytes >= _SHARD_TARGET_BYTES:
                flush()
            if advance is not None:
                advance(span)
    flush()
    return paths


# --- the threaded decode ----------------------------------------------------------------

# A threaded decode reads, decompresses and decodes this many blobs at a time.
_THREAD_BATCH = 64
# The status column of ``decode_batch_nogil``'s counts.
_STATUS = BLOCK_COUNTS.index("status")
# The shard keys that hold Python objects; a threaded decode keeps them in memory.
_OBJECT_KEYS = ("rel_memrole", "rel_tags")


class _Shard:
    """A shard of the threaded decode: its numeric arrays one after another in the file at
    ``path`` (``layout`` gives each one's offset, dtype and shape), its object arrays (the
    tag dicts and member roles of the matched elements) in memory."""

    def __init__(self, path, layout, objects):
        self.path = path
        self.layout = layout
        self.objects = objects

    @classmethod
    def write(cls, path, arrays, objects):
        """Write ``arrays`` to ``path`` and return the shard."""
        layout = {}
        with open(path, "wb") as f:
            for key, array in arrays.items():
                array = np.ascontiguousarray(array)
                layout[key] = (f.tell(), array.dtype, array.shape)
                array.tofile(f)
        return cls(path, layout, objects)

    def __getitem__(self, key):
        """The array ``key``, as ``np.load`` gives a spilled shard's: an object array from
        memory, a numeric one read from the file again at each access."""
        if key in self.objects:
            return self.objects[key]
        offset, dtype, shape = self.layout[key]
        count = int(np.prod(shape))
        return np.fromfile(self.path, dtype=dtype, count=count, offset=offset).reshape(
            shape
        )


def _open_shard(shard):
    """The arrays of a shard by key: a spilled ``.npz`` path, or a :class:`_Shard`."""
    if isinstance(shard, _Shard):
        return shard
    return np.load(shard, allow_pickle=True)


def _is_threaded(shard_paths):
    """Whether ``shard_paths`` are the shards of a threaded decode (:class:`_Shard`)."""
    return bool(shard_paths) and isinstance(shard_paths[0], _Shard)


def _block_shard(decoded, at, b, bbox_bounds):
    """The shard dict of block ``b`` of a batch decoded by ``decode_batch_nogil`` (with
    ``way_strings``): :func:`_decode_one_block`'s arrays, the tags as string ids (see
    :func:`_columnar_tags`). ``at[name][b]`` is where block ``b``'s entries of the count
    column ``name`` (``BLOCK_COUNTS``) start in the batch's arrays. With ``bbox_bounds``
    only the node features inside the box are kept.

    The batch's arrays hold all its blocks one after another, so a block's slices are
    rebased to the block: rows by the block's first node, string ids by its first string,
    offsets by their first value."""

    def span(name):
        return at[name][b], at[name][b + 1]

    shard = _block_nodes(decoded, span, bbox_bounds)
    shard.update(_block_ways(decoded, span))
    shard.update(_NO_RELATIONS)
    return shard


def _block_nodes(decoded, span, bbox_bounds):
    """The node coordinates and node features of a block (see :func:`_block_shard`), only
    those inside ``bbox_bounds`` when it is not None."""
    n0, n1 = span("nodes")
    node_id = decoded["node_id"][n0:n1]
    node_lon = decoded["node_lon"][n0:n1]
    node_lat = decoded["node_lat"][n0:n1]
    s0 = span("strings")[0]
    f0, f1 = span("features")
    rows = decoded["feature_row"][f0:f1] - n0
    pairs_off = decoded["feature_pairs_off"][f0 : f1 + 1]
    p0, p1 = pairs_off[0], pairs_off[-1]
    pairs_off = pairs_off - p0
    keys = decoded["feature_keys"][p0:p1] - s0
    vals = decoded["feature_vals"][p0:p1] - s0
    meta = decoded["feature_meta"][f0:f1]
    if bbox_bounds is not None and len(rows):
        inside = np.flatnonzero(
            _in_box_mask(node_lon[rows], node_lat[rows], bbox_bounds)
        )
        _, keys = take(pairs_off, keys, inside)
        pairs_off, vals = take(pairs_off, vals, inside)
        rows, meta = rows[inside], meta[inside]
    return {
        "node_id": node_id,
        "node_lon": node_lon,
        "node_lat": node_lat,
        "in_box_id": (
            node_id[decoded["in_box"][n0:n1]]
            if bbox_bounds is not None
            else np.empty(0, np.int64)
        ),
        "nfeat_id": node_id[rows],
        "nfeat_lon": node_lon[rows],
        "nfeat_lat": node_lat[rows],
        "nfeat_meta": meta,  # version, timestamp, changeset, visible
        "nfeat_pairs_off": pairs_off,
        "nfeat_pair_key": keys,
        "nfeat_pair_val": vals,
    }


def _block_ways(decoded, span):
    """The ways with a wanted key, every way's refs and the block's strings (see
    :func:`_block_shard`)."""
    w0, w1 = span("ways")
    r0, r1 = span("way_refs")
    t0, t1 = span("way_tags")
    s0, s1 = span("strings")
    b0, b1 = span("string_bytes")
    a0, a1 = span("all_ways")
    ar0, ar1 = span("all_refs")
    way_meta = decoded["way_meta"][w0:w1]  # version, timestamp, visible
    return {
        "way_id": decoded["way_id"][w0:w1],
        "refs": decoded["way_refs"][r0:r1],
        "refs_off": decoded["way_refs_off"][w0 : w1 + 1] - r0,
        "way_pairs_off": decoded["way_tags_off"][w0 : w1 + 1] - t0,
        "way_pair_key": decoded["way_keys"][t0:t1] - s0,
        "way_pair_val": decoded["way_vals"][t0:t1] - s0,
        "str_off": decoded["string_off"][s0 : s1 + 1] - b0,
        "str_data": decoded["string_data"][b0:b1],
        "way_version": way_meta[:, 0],
        "way_timestamp": way_meta[:, 1],
        "way_visible": way_meta[:, 2],
        "all_id": decoded["all_id"][a0:a1],
        "all_refs": decoded["all_refs"][ar0:ar1],
        "all_refs_off": decoded["all_refs_off"][a0 : a1 + 1] - ar0,
    }


# The relation arrays of a block the GIL-free decoder decodes: it hands a block with
# relations back (status 1) to :func:`_decode_one_block`, so they are empty.
_NO_RELATIONS = {
    "rel_id": np.empty(0, np.int64),
    "rel_memid": np.empty(0, np.int64),
    "rel_memoff": _offsets_from_lengths([]),
    "rel_memtype": np.empty(0, np.int64),
    "rel_memrole": np.empty(0, dtype=object),
    "rel_tags": _object_array([]),
    "rel_meta": np.empty((0, 4), np.int64),
}


def _columnar_tags(shard):
    """``shard`` (a :func:`_decode_one_block` dict) with its node features' and ways' tag
    dicts as the threaded decode stores them: ``nfeat_pairs_off`` / ``way_pairs_off`` and
    the ids ``*_pair_key`` / ``*_pair_val`` of the strings ``str_off`` / ``str_data`` hold,
    each string once."""
    ids = {}
    for kind in ("nfeat", "way"):
        keys, vals, off = [], [], [0]
        for tags in shard.pop(kind + "_tags"):
            for k, v in tags.items():
                keys.append(ids.setdefault(k, len(ids)))
                vals.append(ids.setdefault(v, len(ids)))
            off.append(len(keys))
        shard[kind + "_pairs_off"] = np.array(off, dtype=np.int64)
        shard[kind + "_pair_key"] = np.array(keys, dtype=np.int64)
        shard[kind + "_pair_val"] = np.array(vals, dtype=np.int64)
    data = [string.encode("utf-8") for string in ids]
    shard["str_off"] = _offsets_from_lengths([len(d) for d in data])
    shard["str_data"] = np.frombuffer(b"".join(data), dtype=np.uint8)
    return shard


def _shard_tags(shard, kind):
    """The tag dicts of the node features (``kind`` ``"nfeat"``) or ways (``"way"``) of a
    shard: its object array for a shard of the process decode, built from the string ids
    for one of the threaded decode."""
    if not isinstance(shard, _Shard):
        return shard[kind + "_tags"]
    str_off, data = shard["str_off"].tolist(), shard["str_data"].tobytes()
    strings = [
        data[a:b].decode("utf-8", "replace") for a, b in zip(str_off[:-1], str_off[1:])
    ]
    off = shard[kind + "_pairs_off"].tolist()
    keys = shard[kind + "_pair_key"].tolist()
    vals = shard[kind + "_pair_val"].tolist()
    return _object_array(
        [
            {strings[keys[p]]: strings[vals[p]] for p in range(off[i], off[i + 1])}
            for i in range(len(off) - 1)
        ]
    )


def _decoded_parts(f, blobs):
    """The decompressed blocks of ``blobs`` (``(offset, size, span)``, in file order) of the
    file ``f``, read in one go and decompressed without the GIL in parts (see
    :func:`~pyrosm.engine.blobs._inflated_batch`): yields ``(part, buffer, starts)``,
    ``part`` the blobs of the part. A blob that is not zlib data with its size, or each
    blob when the read comes short, is read by :func:`_read_block`."""
    raw = _read_span(f, blobs[0][0], blobs[-1][0] + blobs[-1][1])
    return _inflated_batch(
        raw,
        blobs,
        itemgetter(0),
        itemgetter(1),
        lambda blob: _read_block(f, blob[0], blob[1]),
    )


def _copied_views(arrays):
    """A block's shard dict ``arrays`` with every view of another array copied."""
    return {
        key: (
            value.copy()
            if isinstance(value, np.ndarray) and value.base is not None
            else value
        )
        for key, value in arrays.items()
    }


def _decode_batch_threads(settings, task, advance=None):
    """Thread worker: :func:`_decode_batch` with the blocks read, decompressed and decoded
    without the GIL, ``_THREAD_BATCH`` at a time (a block the GIL-free decoder does not
    handle is decoded by :func:`_decode_one_block`). Each shard keeps its object arrays in
    memory and spills only its numeric arrays (see :class:`_Shard`). Returns the shards.
    """
    task_id, blobs = task
    shards = []
    pending = []
    pending_bytes = 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        merged = _merge_shards(pending)
        path = Path(settings.shard_dir) / ("shard_%d_%d.bin" % (task_id, len(shards)))
        objects = {key: merged.pop(key) for key in _OBJECT_KEYS}
        shards.append(_Shard.write(path, merged, objects))
        pending, pending_bytes = [], 0

    with open(settings.filepath, "rb") as f:
        for whole in _read_batches(blobs, _THREAD_BATCH):
            for batch, buffer, starts in _decoded_parts(f, whole):
                decoded = decode_batch_nogil(
                    buffer,
                    starts,
                    keys=settings.osm_keys,
                    features=settings.include_nodes,
                    all_ways=settings.spill_all_ways,
                    bbox=settings.bbox_bounds,
                    way_strings=True,
                    requested=settings.requested_tag_keys,
                )
                counts = decoded["counts"]
                at = {
                    name: _offsets_from_lengths(counts[:, i]).tolist()
                    for i, name in enumerate(BLOCK_COUNTS)
                }
                for b in range(len(batch)):
                    data = buffer[starts[b] : starts[b + 1]]
                    if counts[b, _STATUS]:
                        # A block the GIL-free decoder hands back.
                        arrays = _columnar_tags(
                            _decode_one_block(settings, *decode_primitive_block(data))
                        )
                    else:
                        arrays = _block_shard(decoded, at, b, settings.bbox_bounds)
                    pending.append(arrays)
                    pending_bytes += sum(
                        v.nbytes for k, v in arrays.items() if k not in _OBJECT_KEYS
                    )
                    if pending_bytes >= _SHARD_TARGET_BYTES:
                        flush()
                if advance is not None:
                    advance(sum(span for _, _, span in batch))
                # The part is released before the next one is decompressed: the pending
                # shards' slices of its arrays are copied out of them.
                pending = [_copied_views(arrays) for arrays in pending]
                del buffer, starts, decoded, data, arrays
    flush()
    return shards
