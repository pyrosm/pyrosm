"""Main-process collection.

Read the spilled shards back into the layer's node, way and relation records, their
member ways and the referenced node coordinates -- pulling only what the kept features
need off disk, and refining the worker's key-presence candidates by pyrosm's exact value
filter so peak memory stays bounded by the working set.
"""

import numpy as np

from pyrosm._arrays import way_records_to_arrays
from pyrosm.tagparser import explode_way_tags, explode_node_tag_array
from pyrosm.data_filter import element_should_be_kept
from pyrosm.engine.decode import _object_array

# Relation.MemberType -> the byte labels pyrosm's relation assembly expects.
_MEMBER_TYPE = {0: b"node", 1: b"way", 2: b"relation"}
# Only way members carry ring geometry. Member ids are matched against the way store, but
# OSM ids are unique only per element type, so a node/relation member id can collide with a
# way id -- mixing types in the lookup would attach or drop the wrong way.
_WAY_MEMBER = 1


def _collect_matching_ways(shard_paths):
    """Read the spilled matching ways back as pyrosm-shaped way records (``id``,
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


def _node_lookup(shard_paths, needed):
    """Gather only the coordinates of ``needed`` node ids from the shards (bounded
    memory) and wrap them in a ``NodeLocations`` for geometry assembly."""
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    if len(needed) == 0:
        # Node-only result (the filter matched no ways/relations): no coordinates needed.
        empty = np.empty(0, np.int64)
        return NodeLocations(
            pd.DataFrame({"id": empty, "lon": np.empty(0), "lat": np.empty(0)})
        )

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


def _collect_kept_ways(shard_paths, exclude_ids, keep):
    """The standalone way records to output: the spilled candidates refined by the exact
    value filter ``keep``, then with the relation member ways dropped (pyrosm assigns those
    to the relation, so they are not standalone way rows)."""
    records = [r for r in _collect_matching_ways(shard_paths) if keep(r["tags"])]
    if not records:
        return None
    if len(exclude_ids):
        exclude = set(exclude_ids.tolist())
        records = [r for r in records if r["id"] not in exclude]
        if not records:
            return None
    return records


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


def _collect_relation_ways(shard_paths, member_ids):
    """Look up the member ways (id -> node refs) of the kept relations from the spilled
    all-ways store. Returned as a ``{id, nodes}`` dict sorted by id ascending -- pyrosm
    aligns member roles to the sorted member ids, so the ways must match that order.
    ``None`` if there are no way members, or none of them are present."""
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


def _ways_arrays(records, tags_as_columns, keep_metadata):
    """Convert way records into the ``way_elements`` dict (all occurring tag columns + the
    JSON ``tags`` column + metadata) pyrosm's assembly expects, reusing pyrosm's own
    tag-explosion so the columns match the in-memory reader. Consumes (explodes) the
    records. The augmentation mirrors ``get_osm_ways_and_relations``."""
    augmented = list(tags_as_columns) + ["id", "nodes"]
    if keep_metadata:
        augmented += ["timestamp", "version"]
    return way_records_to_arrays(explode_way_tags(records), augmented)


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


def _collect_layer(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    keep_ways=True,
    keep_relations=True,
):
    """Shared collection for both output modes: node features, standalone ways, relations,
    their member ways and the node coordinates the ways/relations reference. ``filter_spec``
    is ``(osm_keys, data_filter, filter_type)``; the spilled key-presence candidates are
    refined here by pyrosm's exact value filter. ``keep_ways`` / ``keep_relations`` drop
    those element kinds from the output (``get_data_by_custom_criteria``). ``None`` if there
    is nothing to assemble."""
    osm_keys, data_filter, filter_type = filter_spec

    def keep(tag):
        return element_should_be_kept(tag, osm_keys, data_filter, filter_type)

    if keep_relations:
        relations, member_ids = _collect_relations(shard_paths, keep)
        relation_ways = (
            _collect_relation_ways(shard_paths, member_ids)
            if relations is not None
            else None
        )
        # No member way present (all outside the data) -> the relation can't be assembled.
        if relation_ways is None:
            relations = None
    else:
        # Without relations there is no member-way set, so matching ways that would have
        # been members stay as standalone ways (matching get_data_by_custom_criteria).
        relations, relation_ways, member_ids = None, None, np.empty(0, np.int64)
    kept = _collect_kept_ways(shard_paths, member_ids, keep) if keep_ways else None
    node_features = _collect_node_features(
        shard_paths, tags_as_columns, keep_metadata, keep
    )
    if kept is None and relations is None and node_features is None:
        return None
    node_coordinates = _node_lookup(shard_paths, _needed_node_ids(kept, relation_ways))
    return node_features, kept, relations, relation_ways, node_coordinates
