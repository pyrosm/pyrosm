import os
import warnings
from libc.stdint cimport int64_t
from pyrosm.tagparser cimport tounicode
from pyrosm._arrays cimport concatenate_dicts_of_arrays
from pyrosm.data_filter cimport (
    get_nodeid_lookup_khash,
    nodes_for_way_exist_khash,
    get_latest_version,
    latest_version_rows,
)
from pyrosm.utils import valid_header_block
from pyrosm.utils.progress import (
    COMPLETING_RELATIONS,
    COMPLETING_WAYS,
    DECODING,
    SELECTING_VERSIONS,
    start_step,
)
from pyrosm.frames import create_df
from pyrosm.node_lookup import NodeLocations
from pyrosm.way_store import (
    NodeTags, NodeTagsBuilder, WayStore, WayStoreBuilder, any_per_row,
)
import numpy as np
import pandas as pd

# Relation member type codes -> the labels of the members' ``member_type``.
_MEMBER_TYPES = {0: b"node", 1: b"way", 2: b"relation"}


def iter_decoded_blocks(filepath, progress=None):
    """Yield each data block of ``filepath`` decoded by the raw block decoder, as
    ``(string_table, header, nodes, ways, relations)`` (see
    :func:`~pyrosm.primitive_block_decoder.decode_primitive_block`), with the string
    table decoded to ``str``. The header block is checked first. ``progress(done, size)``
    gets the bytes of the file read: 0 first, then after each block, and the file size at
    the end."""
    from pyrosm.engine.blobs import _index_blobs, _read_block
    from pyrosm.pbf_export import read_header_block
    from pyrosm.primitive_block_decoder import decode_primitive_block

    valid_header_block(read_header_block(filepath))
    with open(filepath, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if progress is not None:
            progress(0, size)
        for kind, offset, length in _index_blobs(filepath):
            if kind != "OSMData":
                continue
            strings, header, nodes, ways, relations = decode_primitive_block(
                _read_block(f, offset, length)
            )
            if progress is not None:
                progress(offset + length, size)
            yield [tounicode(s) for s in strings], header, nodes, ways, relations
        if progress is not None:
            progress(size, size)


cdef _node_tags(const int64_t[::1] keys_vals, Py_ssize_t n, list strings):
    """The tags of a block's nodes from their ``keys_vals`` stream (each node's key and
    value ids followed by a 0): a dict per node with tags, in stream order, and None for a
    node without (every node when the stream is empty)."""
    tags = np.empty(n, dtype=object)
    cdef Py_ssize_t node = 0, pos = 0, end = keys_vals.shape[0]
    while pos < end:
        if keys_vals[pos] != 0:
            node_tags = {}
            while keys_vals[pos] != 0:
                node_tags[strings[keys_vals[pos]]] = strings[keys_vals[pos + 1]]
                pos += 2
            tags[node] = node_tags
        pos += 1
        node += 1
    return tags


cdef _or_zeros(values, Py_ssize_t n):
    """``values``, or ``n`` zeros when the block has none (no metadata)."""
    return values if len(values) else np.zeros(n, dtype=np.int64)


cdef dict _node_columns(
        dict header,
        dict nodes,
        list strings,
        bounding_box,
        unix_time_filter,
        node_id_filter=None,
        bint keep_metadata=True,
        tags_builder=None,
):
    """The node columns of a decoded block: ``id``, ``lon``, ``lat``, ``tags`` and
    ``visible`` (True when the block has no visibility, as the PBF format reads it), and
    with metadata ``version``,
    ``timestamp`` (seconds) and ``changeset``. Nodes outside ``bounding_box``, or not in
    ``node_id_filter`` when given (a pass fetching the nodes of kept ways), and nodes
    newer than ``unix_time_filter`` are left out. With ``tags_builder`` (a
    ``NodeTagsBuilder``) the tags of the nodes kept go there instead of the ``tags``
    column."""
    # History reads filter on the timestamp and visibility, so they always keep the
    # metadata.
    cdef bint keep_meta = keep_metadata or unix_time_filter is not None
    ids = nodes["id"]
    cdef Py_ssize_t n = len(ids)
    granularity = header["granularity"]
    lon = (nodes["lon"] * granularity + header["lon_offset"]) / 1000000000
    lat = (nodes["lat"] * granularity + header["lat_offset"]) / 1000000000
    visible = nodes["visible"].astype(bool)
    if len(visible) == 0:
        visible = np.ones(n, dtype=bool)
    columns = {"id": ids, "lon": lon, "lat": lat}
    if tags_builder is None:
        columns["tags"] = _node_tags(nodes["keys_vals"], n, strings)
    columns["visible"] = visible
    if keep_meta:
        timestamp = nodes["timestamp"] * header["date_granularity"] / 1000
        columns["version"] = _or_zeros(nodes["version"], n)
        columns["timestamp"] = _or_zeros(timestamp.astype(int).astype(np.uint32), n)
        columns["changeset"] = _or_zeros(nodes["changeset"].astype(np.int32), n)

    mask = None
    if node_id_filter is not None:
        mask = np.isin(ids, node_id_filter)
    elif bounding_box is not None:
        xmin, ymin, xmax, ymax = bounding_box
        mask = (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax)
    if unix_time_filter is not None:
        in_time = columns["timestamp"] <= unix_time_filter
        mask = in_time if mask is None else mask & in_time
    if mask is not None:
        columns = {name: values[mask] for name, values in columns.items()}
    if tags_builder is not None:
        tags_builder.add(nodes["keys_vals"], n, strings, mask)
    return columns


cdef dict _with_tags(dict columns, tags):
    """``columns`` with ``tags`` as their ``tags`` column, after ``lat`` as in a block's
    node columns."""
    out = {}
    for name, values in columns.items():
        out[name] = values
        if name == "lat":
            out["tags"] = tags
    return out


cdef dict _tags_as_dicts(dict columns):
    """``columns`` with a ``NodeTags`` column turned into an object array of dicts."""
    return {
        name: values[:] if isinstance(values, NodeTags) else values
        for name, values in columns.items()
    }


cdef _way_mask(dict ways, node_lookup, unix_time_filter, way_ids=None):
    """Which ways of a decoded block to keep: those not newer than ``unix_time_filter``,
    whose id is in ``way_ids`` (an array) when given, and with a node in ``node_lookup``
    when given. None keeps them all."""
    keep = None
    if unix_time_filter is not None:
        keep = ways["timestamp"] <= unix_time_filter
    if way_ids is not None:
        listed = np.isin(ways["id"], way_ids)
        keep = listed if keep is None else keep & listed
    if node_lookup is not None:
        refs = ways["refs"]
        hits = node_lookup.isin(refs)
        in_box = any_per_row(hits, ways["refs_off"])
        keep = in_box if keep is None else keep & in_box
    return keep


cdef _way_ids(ways):
    """The ids of the parsed ways (a ``WayStore``, or a list of way records)."""
    if isinstance(ways, WayStore):
        return ways.ids
    return np.array([way["id"] for way in ways], dtype=np.int64)


cdef _way_refs(ways):
    """The node ids of the parsed ways (a ``WayStore``, or a list of way records),
    joined."""
    if isinstance(ways, WayStore):
        return ways.refs
    return np.fromiter((node for way in ways for node in way["nodes"]), dtype=np.int64)


cdef dict _relation_columns(dict relations, list strings, unix_time_filter):
    """The relation columns of a decoded block: ``id``, ``version``, ``changeset``,
    ``timestamp``, ``members`` (a dict per relation of ``member_id``, ``member_type`` and
    ``member_role`` arrays), ``tags`` (a dict per relation) and ``visible``. Relations
    newer than ``unix_time_filter`` are left out."""
    cdef Py_ssize_t i, j, start, end
    cdef Py_ssize_t n = len(relations["id"])
    cdef const int64_t[::1] members_off = relations["members_off"]
    cdef const int64_t[::1] tags_off = relations["tags_off"]
    member_ids = relations["memids"]
    types = relations["types"].tolist()
    roles = relations["roles"].tolist()
    keys = relations["keys"].tolist()
    vals = relations["vals"].tolist()
    members = np.empty(n, dtype=object)
    tags = np.empty(n, dtype=object)
    for i in range(n):
        start = members_off[i]
        end = members_off[i + 1]
        members[i] = {
            "member_id": member_ids[start:end].copy(),
            "member_type": np.array(
                [_MEMBER_TYPES[t] for t in types[start:end]], dtype=object
            ),
            "member_role": np.array([strings[r] for r in roles[start:end]], dtype=object),
        }
        relation_tags = {}
        for j in range(tags_off[i], tags_off[i + 1]):
            relation_tags[strings[keys[j]]] = strings[vals[j]]
        tags[i] = relation_tags
    columns = {
        "id": relations["id"],
        "version": relations["version"],
        "changeset": relations["changeset"],
        "timestamp": relations["timestamp"],
        "members": members,
        "tags": tags,
        "visible": relations["visible"].astype(bool),
    }
    if unix_time_filter is not None:
        mask = columns["timestamp"] <= unix_time_filter
        columns = {name: values[mask] for name, values in columns.items()}
    return columns


cpdef parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
        bint complete_relations=False,
        steps=None,
):
    """The nodes, ways, relations, node coordinates and completed relation member ways
    of ``filepath``, read in memory, in the steps of ``steps`` (a
    :class:`~pyrosm.utils.progress.Steps`, or None): :data:`DECODING`, then for a
    history read (``unix_time_filter``) :data:`SELECTING_VERSIONS`, and with a
    ``bounding_box`` the passes :data:`COMPLETING_RELATIONS` (with
    ``complete_relations``) and :data:`COMPLETING_WAYS`."""
    return _parse_osm_data(filepath, bounding_box, exclude_relations,
                           unix_time_filter, keep_metadata, complete_relations,
                           steps)


cdef _parse_osm_data(
        filepath,
        bounding_box,
        exclude_relations,
        unix_time_filter,
        bint keep_metadata=True,
        bint complete_relations=False,
        steps=None,
):
    """The read of :func:`parse_osm_data`, with its arguments: one pass over the blocks
    for the elements, then each element's version at ``unix_time_filter``, and with a
    ``bounding_box`` the passes for the member ways and the nodes the box cut off."""
    ways_builder = WayStoreBuilder()
    # History reads select each node's state in a DataFrame, so they keep a dict per node.
    tags_builder = NodeTagsBuilder() if unix_time_filter is None else None
    all_nodes = []
    all_relations = []
    node_lookup = None
    node_blocks_in_lookup = 0
    # A history read applies the bounding box to each element's state at the timestamp
    # (below), not to every version: a deletion has no location, and an element can
    # move into or out of the box between versions.
    box = None if unix_time_filter is not None else bounding_box

    for strings, header, nodes, ways, relations in iter_decoded_blocks(
            filepath, start_step(steps, DECODING)):
        if nodes is not None:
            all_nodes.append(_node_columns(header, nodes, strings, box,
                                           unix_time_filter, None, keep_metadata,
                                           tags_builder))
        if ways is not None:
            if box is None:
                ways_builder.add(ways, strings, _way_mask(ways, None, unix_time_filter))
            elif len(all_nodes) > 0:
                # A way is kept when one of its nodes read so far is in the box.
                if node_blocks_in_lookup != len(all_nodes):
                    node_lookup = get_nodeid_lookup_khash(all_nodes)
                    node_blocks_in_lookup = len(all_nodes)
                ways_builder.add(
                    ways, strings, _way_mask(ways, node_lookup, unix_time_filter)
                )
        if relations is not None and not exclude_relations:
            all_relations.append(_relation_columns(relations, strings, unix_time_filter))

    all_ways = ways_builder.build()

    # Concatenate the node columns (history reads below select from them in a DataFrame)
    all_nodes = concatenate_dicts_of_arrays(all_nodes)
    relations_df = create_df(concatenate_dicts_of_arrays(all_relations))

    # If no nodes were parsed there is no 'id' column. Without any nodes the ways cannot
    # be georeferenced anyway, so return an empty result. (A swapped/inverted bounding box
    # is rejected earlier with a ValueError at OSM() construction; reaching here means the
    # bounding box is well-formed but does not overlap this PBF's data extent.)
    if "id" not in all_nodes:
        warnings.warn(
            "The given bounding box did not contain any OSM nodes, so no data could be "
            "parsed. It likely does not overlap the data extent of this PBF file.",
            UserWarning,
            stacklevel=2,
        )
        return {}, [], {}, {}, []
    if tags_builder is not None:
        all_nodes = _with_tags(all_nodes, tags_builder.build())

    # Keep each element's state at the timestamp if filter is used
    if unix_time_filter is not None:
        start_step(steps, SELECTING_VERSIONS)
        nodes_df = get_latest_version(create_df(all_nodes))
        all_ways = all_ways.records(latest_version_rows(all_ways.ids, all_ways.visible))
        relations_df = get_latest_version(relations_df)
        if bounding_box is not None:
            xmin, ymin, xmax, ymax = bounding_box
            inside = nodes_df["lon"].between(xmin, xmax) & nodes_df["lat"].between(ymin, ymax)
            nodes_df = nodes_df.loc[inside]
            if len(all_ways) > 0:
                node_lookup = get_nodeid_lookup_khash([{"id": nodes_df["id"].values}])
                all_ways = [
                    way for way in all_ways
                    if nodes_for_way_exist_khash(way["nodes"], node_lookup)
                ]
        all_nodes = {col: nodes_df[col].values for col in nodes_df.columns}

    # Keys with numpy arrays
    all_relations = {col: relations_df[col].values for col in relations_df.columns}

    # Complete the member ways of relations that straddle the bounding box (the
    # relation analogue of the #236 node completion below). A relation is assembled
    # from only the member ways present in the box, so a relation straddling the box
    # gets a partial geometry. When `complete_relations` is on, fetch the missing
    # member ways (id-filtered second pass, no bbox filter) of relations that already
    # have >=1 member way in the box -- i.e. relations that appear in results -- and
    # keep them in a SEPARATE set, never appended to `all_ways`. That keeps out-of-box
    # member ways out of normal way-feature results (e.g. networks); they are routed
    # only into relation assembly (see data_manager.get_way_arrays). Their missing node
    # coordinates are picked up by the #236 pass below. bbox-only and opt-in.
    completed_relation_ways = []
    if complete_relations and bounding_box is not None and len(all_ways) > 0:
        present_way_ids = set(_way_ids(all_ways).tolist())

        missing_way_ids = set()
        for members in all_relations.get("members", []):
            member_ids = members["member_id"]
            member_types = members["member_type"]
            # Only way members can be completed (node/relation members are not ways).
            way_member_ids = [
                int(member_ids[j])
                for j in range(len(member_ids))
                if member_types[j] == b"way"
            ]
            # Complete only relations that already have a member way in the box; a
            # relation entirely outside the box does not appear in results, and one
            # fully inside has no missing members.
            if not any(wid in present_way_ids for wid in way_member_ids):
                continue
            for wid in way_member_ids:
                if wid not in present_way_ids:
                    missing_way_ids.add(wid)

        if len(missing_way_ids) > 0:
            missing = np.fromiter(missing_way_ids, dtype=np.int64)
            completed = WayStoreBuilder()
            for strings, header, nodes, ways, relations in iter_decoded_blocks(
                    filepath, start_step(steps, COMPLETING_RELATIONS)):
                if ways is not None:
                    completed.add(
                        ways, strings, _way_mask(ways, None, unix_time_filter, missing)
                    )
            completed = completed.build()
            rows = None
            if unix_time_filter is not None:
                # The same latest-version selection as for all_ways above.
                rows = latest_version_rows(completed.ids, completed.visible)
            completed_relation_ways = completed.records(rows)

    # Complete the geometries of ways that straddle the bounding box (#236). A way is
    # kept when >=1 of its nodes is inside the box, but the box-filtered node parse
    # dropped the vertices that lie just outside it, so its polygon/line would otherwise
    # be cut. Fetch ONLY those missing node coordinates with a second streaming pass
    # over the blocks (the blocks are not retained, so this re-reads the file) and add
    # them to the coordinate lookup used for geometry building. They are deliberately
    # NOT added to `all_nodes` (the standalone node features), so out-of-box nodes never
    # leak into node-feature results (e.g. POIs). bbox-only.
    coords = all_nodes
    if bounding_box is not None and len(all_ways) > 0:
        # Completed relation member ways also need their out-of-box vertices fetched
        # into the coordinate store so the relation geometries are complete.
        refs = np.concatenate([_way_refs(all_ways), _way_refs(completed_relation_ways)])
        node_id_filter = np.setdiff1d(refs, all_nodes["id"])
        if len(node_id_filter) > 0:
            boundary_nodes = []
            for strings, header, nodes, ways, relations in iter_decoded_blocks(
                    filepath, start_step(steps, COMPLETING_WAYS)):
                if nodes is not None:
                    boundary_nodes.append(_node_columns(
                        header, nodes, strings, None, unix_time_filter,
                        node_id_filter, keep_metadata,
                    ))
            if len(boundary_nodes) > 0:
                boundary_df = create_df(concatenate_dicts_of_arrays(boundary_nodes))
                if "id" in boundary_df.columns:
                    # Apply the same OSH latest-version selection to the boundary
                    # nodes as the nodes received above, so timestamped geometries are
                    # built from each vertex's state at the timestamp rather than
                    # whichever historical version drop_duplicates happens to keep.
                    if unix_time_filter is not None:
                        boundary_df = get_latest_version(boundary_df)
                    coords = pd.concat(
                        [create_df(_tags_as_dicts(all_nodes)), boundary_df],
                        ignore_index=True,
                    ).drop_duplicates(subset="id")

    # Compact node-coordinate store (id->index hash map + column arrays),
    # replacing the per-node dict-of-dicts; geometry/graph code reads through it.
    node_coordinates_lookup = NodeLocations(coords)

    return (all_nodes, all_ways, all_relations, node_coordinates_lookup,
            completed_relation_ways)
