"""Main-process collection.

Read the spilled shards back into the building way and relation records, their member
ways and the referenced node coordinates -- pulling only what the kept features need off
disk, so peak memory stays bounded by the working set.
"""

import numpy as np

from pyrosm._arrays import way_records_to_arrays
from pyrosm.tagparser import explode_way_tags
from pyrosm.engine.decode import _object_array

# Relation.MemberType -> the byte labels pyrosm's relation assembly expects.
_MEMBER_TYPE = {0: b"node", 1: b"way", 2: b"relation"}
# Only way members carry ring geometry. Member ids are matched against the way store, but
# OSM ids are unique only per element type, so a node/relation member id can collide with a
# building way id -- mixing types in the lookup would attach or drop the wrong way.
_WAY_MEMBER = 1


def _collect_building_ways(shard_paths):
    """Read the spilled building ways back as pyrosm-shaped way records (``id``,
    ``version``, ``timestamp``, ``visible``, ``nodes``, ``tags``)."""
    records = []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid = z["way_id"]
        if len(wid) == 0:
            continue
        off, refs, tags = z["refs_off"], z["refs"], z["way_tags"]
        ver, ts, vis = z["way_version"], z["way_timestamp"], z["way_visible"]
        for i in range(len(wid)):
            records.append(
                {
                    "id": int(wid[i]),
                    "version": int(ver[i]),
                    "timestamp": int(ts[i]),
                    "visible": bool(vis[i]),
                    "nodes": refs[off[i] : off[i + 1]].tolist(),
                    "tags": tags[i],
                }
            )
    return records


def _node_lookup(shard_paths, needed):
    """Gather only the coordinates of ``needed`` node ids from the shards (bounded
    memory) and wrap them in a ``NodeLocations`` for geometry assembly."""
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    lon = np.full(len(needed), np.nan)
    lat = np.full(len(needed), np.nan)
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        nid = z["node_id"]
        if len(nid) == 0:
            continue
        pos = np.clip(np.searchsorted(needed, nid), 0, len(needed) - 1)
        hit = needed[pos] == nid
        lon[pos[hit]] = z["node_lon"][hit]
        lat[pos[hit]] = z["node_lat"][hit]
    present = ~np.isnan(lon)
    coords = pd.DataFrame(
        {"id": needed[present], "lon": lon[present], "lat": lat[present]}
    )
    return NodeLocations(coords)


def _collect_kept_ways(shard_paths, exclude_ids):
    """The standalone building way records to output, after dropping the ones that are
    members of a building relation (pyrosm assigns those to the relation, so they are not
    standalone way rows)."""
    records = _collect_building_ways(shard_paths)
    if not records:
        return None
    if len(exclude_ids):
        exclude = set(exclude_ids.tolist())
        records = [r for r in records if r["id"] not in exclude]
        if not records:
            return None
    return records


def _load_building_relations(shard_paths):
    """Reassemble the building relations spilled across shards into the ``relations``
    struct pyrosm's assembly expects (``id`` / ``members`` / ``tags``), and return it
    together with the unique set of all their member ids. ``(None, empty)`` if there are
    no building relations."""
    ids, members, tags, meta, way_member_ids = [], [], [], [], []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        rid = z["rel_id"]
        if len(rid) == 0:
            continue
        memid, memoff, memtype, memrole, rtags = (
            z["rel_memid"],
            z["rel_memoff"],
            z["rel_memtype"],
            z["rel_memrole"],
            z["rel_tags"],
        )
        for k in range(len(rid)):
            s, e = memoff[k], memoff[k + 1]
            mids, mtypes = memid[s:e], memtype[s:e]
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
        ids.append(rid)
        meta.append(z["rel_meta"])
    if not ids:
        return None, np.empty(0, np.int64)
    meta = np.concatenate(meta)
    relations = {
        "id": np.concatenate(ids),
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


def _collect_relation_ways(shard_paths, member_ids):
    """Look up the member ways (id -> node refs) of the building relations from the
    spilled all-ways store. Returned as a ``{id, nodes}`` dict sorted by id ascending --
    pyrosm aligns member roles to the sorted member ids, so the ways must match that
    order. ``None`` if there are no way members, or none of them are present."""
    if len(member_ids) == 0:
        return None
    found = {}
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid, off, refs = z["all_id"], z["all_refs_off"], z["all_refs"]
        if len(wid) == 0:
            continue
        pos = np.clip(np.searchsorted(member_ids, wid), 0, len(member_ids) - 1)
        for k in np.nonzero(member_ids[pos] == wid)[0]:
            found[int(wid[k])] = refs[off[k] : off[k + 1]]
    if not found:
        return None
    ids = np.array(sorted(found), dtype=np.int64)
    nodes = np.empty(len(ids), dtype=object)
    nodes[:] = [found[int(i)] for i in ids]
    return {"id": ids, "nodes": nodes}


def _ways_arrays(records, keep_metadata):
    """Convert way records into the ``way_elements`` dict (all occurring tag columns + the
    JSON ``tags`` column + metadata) pyrosm's assembly expects, reusing pyrosm's own
    tag-explosion so the columns match the in-memory reader. Consumes (explodes) the
    records. The augmentation mirrors ``get_osm_ways_and_relations``."""
    from pyrosm.config import Conf

    tags_as_columns = list(Conf.tags.building) + ["id", "nodes"]
    if keep_metadata:
        tags_as_columns += ["timestamp", "version"]
    return way_records_to_arrays(explode_way_tags(records), tags_as_columns)


def _needed_node_ids(kept, relation_ways):
    """The unique node ids referenced by the kept standalone ways and the relation
    member ways -- the only coordinates the gather has to pull off disk."""
    refs = []
    if kept is not None:
        refs.extend(r["nodes"] for r in kept)
    if relation_ways is not None:
        refs.extend(relation_ways["nodes"])
    if not refs:
        return np.empty(0, np.int64)
    return np.unique(np.concatenate(refs))


def _collect_buildings(shard_paths):
    """Shared collection for both output modes: standalone building ways, building
    relations, their member ways and the node coordinates all of them reference. ``None``
    if there is nothing to assemble."""
    relations, member_ids = _load_building_relations(shard_paths)
    relation_ways = (
        _collect_relation_ways(shard_paths, member_ids)
        if relations is not None
        else None
    )
    # No member way is present (all outside the data) -> the relation cannot be assembled.
    if relation_ways is None:
        relations = None
    kept = _collect_kept_ways(shard_paths, member_ids)
    if kept is None and relations is None:
        return None
    node_coordinates = _node_lookup(shard_paths, _needed_node_ids(kept, relation_ways))
    return kept, relations, relation_ways, node_coordinates
