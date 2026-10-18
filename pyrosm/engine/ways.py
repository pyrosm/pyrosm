"""The collect step of a threaded read, column by column (see :mod:`pyrosm.way_columns`).

The threaded decode stores each element's tags as ids of strings, not as a dict. An
:class:`ElementStore` gathers those columns from the shards for the candidate ways, and
:func:`kept_ways` selects the ways a read outputs (the value filter in C, see
:func:`keep_mask`); :meth:`ElementStore.way_columns` gives the column dict the rest of the
collect step reads. The module also gathers the node coordinates, the relation member
ways and the network's node records on the read's threads.
"""

import numpy as np

from pyrosm import way_columns as wc
from pyrosm._arrays import _offsets_from_lengths
from pyrosm.engine.decode import _merge_shards, _open_shard
from pyrosm.engine.pool import thread_map

# The shard key of each column of a store of ways, in the store's constructor's order.
_WAY_KEYS = {
    "id": "way_id",
    "refs_off": "refs_off",
    "refs": "refs",
    "version": "way_version",
    "timestamp": "way_timestamp",
    "visible": "way_visible",
    "pairs_off": "way_pairs_off",
    "pair_key": "way_pair_key",
    "pair_val": "way_pair_val",
    "str_off": "str_off",
    "str_data": "str_data",
}


def _joined_shards(shard_paths, present, keys):
    """The arrays ``keys`` of the shards whose array ``present`` is not empty, joined into
    one shard dict as :func:`~pyrosm.engine.decode._merge_shards` joins blocks (offsets
    rebuilt, string ids renumbered); ``None`` when no shard has one."""
    parts = []
    for path in shard_paths:
        z = _open_shard(path)
        if len(z[present]):
            parts.append({key: z[key] for key in keys})
    return _merge_shards(parts) if parts else None


def _no_nodes():
    """An empty coordinate store."""
    from pyrosm.node_lookup import NodeLocations

    return NodeLocations(
        {"id": np.empty(0, np.int64), "lon": np.empty(0), "lat": np.empty(0)}
    )


class ElementStore:
    """The ways of a read as flat columns (see :mod:`pyrosm.way_columns`): ``id``, the node
    refs (``refs``, way ``i`` owning ``refs[refs_off[i]:refs_off[i + 1]]``), ``version``,
    ``timestamp``, ``visible``, and the tags as pairs of string ids (``pair_key`` /
    ``pair_val``, way ``i`` owning ``pairs_off[i]:pairs_off[i + 1]``) of the strings
    ``str_off`` / ``str_data`` hold."""

    def __init__(
        self,
        id,
        refs_off,
        refs,
        version,
        timestamp,
        visible,
        pairs_off,
        pair_key,
        pair_val,
        str_off,
        str_data,
    ):
        self.id = id
        self.refs_off = refs_off
        self.refs = refs
        self.version = version
        self.timestamp = timestamp
        self.visible = visible
        self.pairs_off = pairs_off
        self.pair_key = pair_key
        self.pair_val = pair_val
        self.str_off = str_off
        self.str_data = str_data

    def __len__(self):
        return len(self.id)

    @classmethod
    def from_shards(cls, shard_paths):
        """The ways of the threaded decode's shards, or ``None`` when they hold none."""
        joined = _joined_shards(shard_paths, "way_id", _WAY_KEYS.values())
        if joined is None:
            return None
        return cls(**{key: joined[shard_key] for key, shard_key in _WAY_KEYS.items()})

    def take(self, rows):
        """The store of the ways ``rows``, sharing this store's strings."""
        rows = np.asarray(rows, dtype=np.int64)
        refs_off, refs = wc.take(self.refs_off, self.refs, rows)
        pairs_off, pair_key = wc.take(self.pairs_off, self.pair_key, rows)
        _, pair_val = wc.take(self.pairs_off, self.pair_val, rows)
        return ElementStore(
            id=self.id[rows],
            refs_off=refs_off,
            refs=refs,
            version=self.version[rows],
            timestamp=self.timestamp[rows],
            visible=self.visible[rows],
            pairs_off=pairs_off,
            pair_key=pair_key,
            pair_val=pair_val,
            str_off=self.str_off,
            str_data=self.str_data,
        )

    def string(self, i):
        """String ``i`` as the Python path decodes it."""
        return bytes(self.str_data[self.str_off[i] : self.str_off[i + 1]]).decode(
            "utf-8", "replace"
        )

    def tags(self, w, keys=None):
        """The tag dict of way ``w`` as the Python path builds it, with only the keys
        ``keys`` when given."""
        tags = {}
        for p in range(self.pairs_off[w], self.pairs_off[w + 1]):
            key = self.string(self.pair_key[p])
            if keys is None or key in keys:
                tags[key] = self.string(self.pair_val[p])
        return tags

    def node_lists(self):
        """Each way's node refs, as an object array of arrays."""
        nodes = np.empty(len(self), dtype=object)
        nodes[:] = np.split(self.refs, self.refs_off[1:-1])
        return nodes

    def way_columns(self, keys=None):
        """The ways as :func:`~pyrosm.engine.collect._read_way_columns` gives them, with
        only the tag keys ``keys`` when given."""
        tags = np.empty(len(self), dtype=object)
        tags[:] = [self.tags(w, keys) for w in range(len(self))]
        return {
            "id": self.id,
            "nodes": self.node_lists(),
            "tags": tags,
            "version": self.version,
            "timestamp": self.timestamp,
            "visible": self.visible.astype(bool),
        }


def _filter_entries(data_filter):
    """The entries of a dict value filter for :func:`pyrosm.way_columns.value_filter`, or
    ``None`` for a filter it does not cover (a compiled filter, an unknown value type, a
    string UTF-8 cannot encode)."""
    if not isinstance(data_filter, dict):
        return None
    try:
        return _encoded_entries(data_filter)
    except UnicodeEncodeError:
        return None  # a string no PBF holds, such as a lone surrogate


def _encoded_entries(data_filter):
    """The entries of :func:`_filter_entries` for a dict filter, its strings encoded; raises
    ``UnicodeEncodeError`` for a string UTF-8 cannot encode."""
    from pyrosm.config.osm_filters import _CLOSED, AccessRule, MultiValue

    entries = []
    for key, values in data_filter.items():
        if isinstance(values, AccessRule):
            entries.append(
                (
                    wc.FILTER_ACCESS,
                    [k.encode("utf-8") for k in values.keys],
                    [v.encode("utf-8") for v in _CLOSED],
                )
            )
            continue
        if values is True or (isinstance(values, list) and values == [True]):
            entries.append((wc.FILTER_ANY, [key.encode("utf-8")], []))
            continue
        if not isinstance(values, (list, tuple, set, frozenset)):
            return None
        strings = [v.encode("utf-8") for v in values if isinstance(v, str)]
        kind = wc.FILTER_MULTI if isinstance(values, MultiValue) else wc.FILTER_VALUES
        entries.append((kind, [key.encode("utf-8")], strings))
    return entries


def keep_mask(store, filter_spec):
    """Whether each way of ``store`` passes the value filter ``filter_spec`` (see
    :func:`~pyrosm.engine.collect._keep_fn`). The C filter decides the ways it can; the
    others (``wc.CHECK``) go through the Python filter on their tag dicts."""
    from pyrosm.engine.collect import _keep_fn

    osm_keys, data_filter, filter_type = filter_spec
    n = len(store)
    if data_filter is None:
        return np.ones(n, dtype=bool)
    keep = _keep_fn(filter_spec)
    entries = _filter_entries(data_filter)
    if entries is None:
        return np.fromiter((keep(store.tags(w)) for w in range(n)), dtype=bool, count=n)
    valid, _ = wc.string_columns(store.str_off, store.str_data, [])
    decided = wc.value_filter(
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid.view(np.uint8),
        np.arange(n, dtype=np.int64),
        [k.encode("utf-8") for k in osm_keys],
        entries,
        filter_type == "keep",
    )
    mask = decided == wc.KEEP
    for w in np.flatnonzero(decided == wc.CHECK):
        mask[w] = keep(store.tags(w))
    return mask


def _string_columns(store, keys):
    """``(valid, column, special)`` for the tag columns ``keys`` of ``store``: whether each
    string is valid UTF-8 and the column it names as a key (see
    :func:`pyrosm.way_columns.string_columns`), and the special elements, whose tags only
    their tag dicts give (see :func:`pyrosm.way_columns.special_ways`)."""
    valid, column = wc.string_columns(
        store.str_off, store.str_data, [k.encode("utf-8") for k in keys]
    )
    special = wc.special_ways(
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid,
        column,
    )
    return valid, column, special


def result_tag_columns(store, tags_as_columns, keep_metadata, keep_other_tags):
    """The tag columns of all the ways of ``store``, as
    :func:`~pyrosm.engine.collect._result_tag_columns` gives them."""
    from pyrosm.engine.collect import _way_column_keys

    column_keys, tag_cols = _way_column_keys(tags_as_columns, keep_metadata)
    tag_col_set = set(tag_cols)
    _, column, special = _string_columns(store, tag_cols)
    present, leftover = wc.column_presence(
        store.pairs_off,
        store.pair_key,
        column,
        np.flatnonzero(~special),
        len(tag_cols),
    )
    columns = {tag_cols[i] for i in np.flatnonzero(present)}
    leftover = leftover or not {"version", "timestamp", "visible"}.issubset(column_keys)
    requested = None if keep_other_tags else set(tags_as_columns)
    for w in np.flatnonzero(special):
        keys = set(store.tags(w, requested))
        if "id" in keys:
            keys.discard("id")
            keys.add("id_tag")
        columns |= keys & tag_col_set
        leftover = leftover or not keys <= tag_col_set
    if keep_other_tags and leftover and len(store) > 0:
        columns.add("tags")
    return columns


def kept_ways(store, exclude_ids, filter_spec, in_box):
    """The ways of ``store`` a read outputs, as
    :func:`~pyrosm.engine.collect._filter_way_columns` selects them: those passing the value
    filter ``filter_spec`` (see :func:`keep_mask`), with a node in ``in_box`` (the ids of the
    nodes in the bounding box, or None), and not in ``exclude_ids``. ``None`` when none is
    left."""
    from pyrosm.int64_hash import Int64Set

    if store is None:
        return None
    mask = keep_mask(store, filter_spec)
    if in_box is not None:
        mask &= wc.with_node_in(store.refs_off, store.refs, Int64Set(in_box))
    if len(exclude_ids):
        mask &= ~wc.in_set(store.id, Int64Set(np.asarray(exclude_ids, dtype=np.int64)))
    if not mask.any():
        return None
    return store.take(np.flatnonzero(mask))


def node_coordinates(shard_paths, refs):
    """The coordinate store of the nodes ``refs`` (a list of id arrays) refer to, as
    :func:`~pyrosm.engine.collect._node_lookup` builds it: their ids in ascending order
    (the coordinates of the last shard holding an id) and coordinates, gathered from the
    shards on the read's threads."""
    import pandas as pd

    from pyrosm.int64_hash import Int64Set
    from pyrosm.node_lookup import NodeLocations

    refs = [r for r in refs if len(r)]
    if not refs:
        return _no_nodes()
    wanted = Int64Set(np.concatenate(refs))

    def gather(shard):
        z = _open_shard(shard)
        ids = z["node_id"]
        if len(ids) == 0:
            return None
        mask = wc.in_set(ids, wanted)
        return ids[mask], z["node_lon"][mask], z["node_lat"][mask]

    parts = [p for p in thread_map(gather, shard_paths) if p is not None]
    if not parts:
        return _no_nodes()
    ids, lon, lat = (np.concatenate(column) for column in zip(*parts))
    # Sorted by id with each id's last copy, unless the ids already ascend strictly.
    if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
        order = np.argsort(ids, kind="stable")
        ids, lon, lat = ids[order], lon[order], lat[order]
        last = np.append(ids[1:] != ids[:-1], True)
        ids, lon, lat = ids[last], lon[last], lat[last]
    return NodeLocations(pd.DataFrame({"id": ids, "lon": lon, "lat": lat}))


def relation_ways(shard_paths, member_ids, in_box=None):
    """The member ways of the kept relations, as
    :func:`~pyrosm.engine.collect._collect_relation_ways` gives them (the last shard
    holding a way wins), gathered from a threaded decode's shards on the read's threads,
    their node refs held in one array."""
    from pyrosm.int64_hash import Int64Set

    members = Int64Set(member_ids)
    box = None if in_box is None else Int64Set(in_box)

    def gather(shard):
        z = _open_shard(shard)
        ids = z["all_id"]
        if len(ids) == 0:
            return None
        rows = np.flatnonzero(wc.in_set(ids, members))
        off, refs = wc.take(z["all_refs_off"], z["all_refs"], rows)
        if box is not None:
            inside = np.flatnonzero(wc.with_node_in(off, refs, box))
            rows = rows[inside]
            off, refs = wc.take(off, refs, inside)
        return (ids[rows], np.diff(off), refs) if len(rows) else None

    parts = [part for part in thread_map(gather, shard_paths) if part is not None]
    if not parts:
        return None
    ids, counts, refs = (np.concatenate(column) for column in zip(*parts))
    starts = _offsets_from_lengths(counts)[:-1]
    # Each id once, ascending, at its last copy.
    unique, last_reversed = np.unique(ids[::-1], return_index=True)
    last = len(ids) - 1 - last_reversed
    nodes = np.empty(len(unique), dtype=object)
    nodes[:] = [
        refs[a : a + c] for a, c in zip(starts[last].tolist(), counts[last].tolist())
    ]
    return {"id": unique, "nodes": nodes}


def node_record_blocks(filepath, wanted, keep_metadata):
    """The records of the nodes ``wanted`` (an :class:`~pyrosm.int64_hash.Int64Set`) in
    each data block of ``filepath``, for
    :func:`~pyrosm.engine.collect._gather_node_records`, decoded without the GIL on the
    read's threads: yields ``(end, item)`` for each data blob in file order,
    ``end`` the blob's end in the file and ``item`` None (no wanted node), the block's
    bytes (for the Python path), or ``(record, counts, keys, vals, strings)``: the
    arrays of the wanted nodes and their tags, ``counts`` per node as ids of
    ``strings``."""
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs, _read_batches
    from pyrosm.engine.decode import _THREAD_BATCH, _decoded_parts

    blobs, _ = _data_blob_spans(_index_blobs(filepath))

    def decode(batch):
        out = []
        with open(filepath, "rb") as f:
            for part, buffer, starts in _decoded_parts(f, batch):
                out.extend(_part_records(part, buffer, starts, wanted, keep_metadata))
                del buffer, starts
        return out

    for items in thread_map(decode, _read_batches(blobs, _THREAD_BATCH)):
        yield from items


def _part_records(part, buffer, starts, wanted, keep_metadata):
    """The ``(end, item)`` of :func:`node_record_blocks` for the blocks of a part."""
    from pyrosm.primitive_block_decoder import BLOCK_COUNTS, decode_batch_nogil

    decoded = decode_batch_nogil(
        buffer, starts, keys=[], features=True, way_strings=True, node_ids=wanted
    )
    counts = decoded["counts"]
    status = counts[:, BLOCK_COUNTS.index("status")]
    # Block b's features are at[b]:at[b + 1]; feature_meta holds version, timestamp,
    # changeset, visible. The tags are ids of the part's strings, so ``strings`` (decoded
    # once) serves all its blocks.
    at = _offsets_from_lengths(counts[:, BLOCK_COUNTS.index("features")])
    rows, meta = decoded["feature_row"], decoded["feature_meta"]
    pairs_off = decoded["feature_pairs_off"]
    strings = None
    out = []
    for b, (offset, size, _) in enumerate(part):
        end = offset + size
        f0, f1 = at[b], at[b + 1]
        if status[b]:
            out.append((end, bytes(buffer[starts[b] : starts[b + 1]])))
            continue
        if f0 == f1:
            out.append((end, None))
            continue
        r = rows[f0:f1]
        record = {
            "id": decoded["node_id"][r],
            "lon": decoded["node_lon"][r],
            "lat": decoded["node_lat"][r],
            "visible": meta[f0:f1, 3].astype(bool),
        }
        if keep_metadata:
            for c, name in enumerate(("version", "timestamp", "changeset")):
                record[name] = meta[f0:f1, c].copy()
        if strings is None:
            off, data = decoded["string_off"], decoded["string_data"]
            strings = [
                data[a:b].tobytes().decode("utf-8", "replace")
                for a, b in zip(off[:-1].tolist(), off[1:].tolist())
            ]
        p0, p1 = pairs_off[f0], pairs_off[f1]
        out.append(
            (
                end,
                (
                    record,
                    np.diff(pairs_off[f0 : f1 + 1]),
                    decoded["feature_keys"][p0:p1],
                    decoded["feature_vals"][p0:p1],
                    strings,
                ),
            )
        )
    return out
