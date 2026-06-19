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
from pyrosm.engine.bounding_box import _in_box_nodes

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


def _collect_kept_ways(shard_paths, exclude_ids, keep, in_box=None):
    """The standalone way records to output: the spilled candidates refined by the exact
    value filter ``keep``, then -- when reading a bounding box -- restricted to ways with at
    least one node inside it (kept whole, so geometry is not cut at the edge), and finally
    with the relation member ways dropped (pyrosm assigns those to the relation, so they are
    not standalone way rows)."""
    records = [r for r in _collect_matching_ways(shard_paths) if keep(r["tags"])]
    if in_box is not None:
        inside = set(in_box.tolist())
        records = [r for r in records if not inside.isdisjoint(r["nodes"])]
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
    osm_keys, data_filter, filter_type = filter_spec

    def keep(tag):
        return element_should_be_kept(tag, osm_keys, data_filter, filter_type)

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
    node_coordinates = _node_lookup(shard_paths, _needed_node_ids(kept, relation_ways))
    return node_features, kept, relations, relation_ways, node_coordinates
