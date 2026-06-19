"""Worker-side decode and spill.

A worker decodes a contiguous run of blobs and writes one shard holding the node
coordinates, the matching building ways (refs + resolved tags + metadata), *every* way
(id + refs, for relation-member lookup) and the building relations.
"""

import os

import numpy as np

from pyrosm.primitive_block_decoder import decode_primitive_block
from pyrosm.engine.blobs import _read_block

_BUILDING = b"building"

# Per-worker globals, set by the pool initializer (or directly for the in-process path).
_FILEPATH = None
_SHARD_DIR = None


def _init_worker(filepath, shard_dir):
    global _FILEPATH, _SHARD_DIR
    _FILEPATH = filepath
    _SHARD_DIR = shard_dir


def _resolve_tags(string_table, keys, vals, start, end):
    """The ``{key: value}`` tag dict for one element, resolved through the string table
    (decoded to str, as pyrosm's protobuf path produces)."""
    return {
        string_table[keys[p]]
        .decode("utf-8", "replace"): string_table[vals[p]]
        .decode("utf-8", "replace")
        for p in range(start, end)
    }


def _building_ways(string_table, ways):
    """Select the ways tagged ``building=*`` and return, per matching way, its node-ref
    slice, full (resolved) tag dict and ``version``/``timestamp``/``visible`` metadata --
    everything pyrosm's way record carries. Returns a dict of parallel arrays/lists, or
    ``None``."""
    if ways is None or _BUILDING not in string_table:
        return None
    building_idx = string_table.index(_BUILDING)
    keys, vals, tags_off = ways["keys"], ways["vals"], ways["tags_off"]
    key_positions = np.nonzero(keys == building_idx)[0]
    if len(key_positions) == 0:
        return None
    # A tag key belongs to the way whose [tags_off[i], tags_off[i+1]) slice contains it.
    way_index = np.searchsorted(tags_off, key_positions, side="right") - 1
    refs, refs_off = ways["refs"], ways["refs_off"]
    return {
        "id": ways["id"][way_index],
        "refs": [refs[refs_off[i] : refs_off[i + 1]] for i in way_index],
        "tags": [
            _resolve_tags(string_table, keys, vals, tags_off[i], tags_off[i + 1])
            for i in way_index
        ],
        "version": ways["version"][way_index],
        "timestamp": ways["timestamp"][way_index],
        "visible": ways["visible"][way_index],
    }


def _building_relations(string_table, relations):
    """The ``building=*`` relations in this block. Yields, per relation, its id, member
    id/type/role arrays, full tag dict and ``version``/``timestamp``/``changeset``
    metadata -- everything pyrosm needs to assemble the multipolygon and its columns.
    Member roles and tags are resolved through the block's string table."""
    if relations is None or _BUILDING not in string_table:
        return
    building_idx = string_table.index(_BUILDING)
    key_positions = np.nonzero(relations["keys"] == building_idx)[0]
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
    for i in rel_index:
        s, e = moff[i], moff[i + 1]
        member_role = np.array(
            [string_table[r].decode("utf-8", "replace") for r in roles[s:e]],
            dtype=object,
        )
        tags = _resolve_tags(string_table, keys, vals, tags_off[i], tags_off[i + 1])
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


def _decode_batch(task):
    """Worker: decode a contiguous run of blobs and spill one shard with the node
    coordinates, the matching building ways (refs + resolved tags + metadata), *all* ways
    (id + refs, for relation-member lookup) and the building relations, then return the
    shard path."""
    worker_id, blobs = task
    node_id, node_lon, node_lat = [], [], []
    bld_id, bld_refs, bld_tags = [], [], []
    bld_version, bld_timestamp, bld_visible = [], [], []
    all_id, all_refs, all_count = [], [], []
    rel = {k: [] for k in ("id", "memid", "memtype", "memrole", "tags", "meta")}
    with open(_FILEPATH, "rb") as f:
        for offset, size in blobs:
            data = _read_block(f, offset, size)
            string_table, header, nodes, ways, relations = decode_primitive_block(data)
            if nodes is not None:
                gran = header["granularity"]
                node_id.append(nodes["id"])
                node_lat.append((nodes["lat"] * gran + header["lat_offset"]) / 1e9)
                node_lon.append((nodes["lon"] * gran + header["lon_offset"]) / 1e9)
            if ways is not None:
                all_id.append(ways["id"])
                all_refs.append(ways["refs"])
                all_count.append(np.diff(ways["refs_off"]))
                found = _building_ways(string_table, ways)
                if found is not None:
                    bld_id.append(found["id"])
                    bld_refs.extend(found["refs"])
                    bld_tags.extend(found["tags"])
                    bld_version.append(found["version"])
                    bld_timestamp.append(found["timestamp"])
                    bld_visible.append(found["visible"])
            for r in _building_relations(string_table, relations):
                rel["id"].append(r["id"])
                rel["memid"].append(r["memid"])
                rel["memtype"].append(r["memtype"])
                rel["memrole"].append(r["memrole"])
                rel["tags"].append(r["tags"])
                rel["meta"].append((r["version"], r["timestamp"], r["changeset"]))

    path = os.path.join(_SHARD_DIR, "shard_%d.npz" % worker_id)
    np.savez(
        path,
        node_id=np.concatenate(node_id) if node_id else np.empty(0, np.int64),
        node_lon=np.concatenate(node_lon) if node_lon else np.empty(0),
        node_lat=np.concatenate(node_lat) if node_lat else np.empty(0),
        way_id=np.concatenate(bld_id) if bld_id else np.empty(0, np.int64),
        refs=np.concatenate(bld_refs) if bld_refs else np.empty(0, np.int64),
        refs_off=_offsets_from_lengths([len(r) for r in bld_refs]),
        way_tags=_object_array(bld_tags),
        way_version=(
            np.concatenate(bld_version) if bld_version else np.empty(0, np.int64)
        ),
        way_timestamp=(
            np.concatenate(bld_timestamp) if bld_timestamp else np.empty(0, np.int64)
        ),
        way_visible=(
            np.concatenate(bld_visible) if bld_visible else np.empty(0, np.int64)
        ),
        all_id=np.concatenate(all_id) if all_id else np.empty(0, np.int64),
        all_refs=np.concatenate(all_refs) if all_refs else np.empty(0, np.int64),
        all_refs_off=_offsets_from_lengths(
            np.concatenate(all_count).tolist() if all_count else []
        ),
        rel_id=np.array(rel["id"], dtype=np.int64),
        rel_memid=(
            np.concatenate(rel["memid"]) if rel["memid"] else np.empty(0, np.int64)
        ),
        rel_memoff=_offsets_from_lengths([len(m) for m in rel["memid"]]),
        rel_memtype=(
            np.concatenate(rel["memtype"]) if rel["memtype"] else np.empty(0, np.int64)
        ),
        rel_memrole=(
            np.concatenate(rel["memrole"])
            if rel["memrole"]
            else np.empty(0, dtype=object)
        ),
        rel_tags=_object_array(rel["tags"]),
        rel_meta=np.array(rel["meta"], dtype=np.int64).reshape(-1, 3),
    )
    return path
