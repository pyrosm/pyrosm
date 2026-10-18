"""The collect step of a threaded read, column by column (see :mod:`pyrosm.way_columns`).

The threaded decode stores each element's tags as ids of strings, not as a dict. An
:class:`ElementStore` gathers those columns from the shards for the candidate ways, and
:func:`kept_ways` selects the ways a read outputs; :meth:`ElementStore.way_columns` gives
the column dict the rest of the collect step reads.
"""

import numpy as np

from pyrosm import way_columns as wc
from pyrosm.engine.decode import _merge_shards, _open_shard

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

    def tags(self, w):
        """The tag dict of way ``w`` as the Python path builds it."""
        return {
            self.string(self.pair_key[p]): self.string(self.pair_val[p])
            for p in range(self.pairs_off[w], self.pairs_off[w + 1])
        }

    def node_lists(self):
        """Each way's node refs, as an object array of arrays."""
        nodes = np.empty(len(self), dtype=object)
        nodes[:] = np.split(self.refs, self.refs_off[1:-1])
        return nodes

    def way_columns(self):
        """The ways as :func:`~pyrosm.engine.collect._read_way_columns` gives them."""
        tags = np.empty(len(self), dtype=object)
        tags[:] = [self.tags(w) for w in range(len(self))]
        return {
            "id": self.id,
            "nodes": self.node_lists(),
            "tags": tags,
            "version": self.version,
            "timestamp": self.timestamp,
            "visible": self.visible.astype(bool),
        }


def kept_ways(store, exclude_ids, keep, in_box):
    """The ways of ``store`` a read outputs, as
    :func:`~pyrosm.engine.collect._filter_way_columns` selects them: those whose tags pass
    ``keep``, with a node in ``in_box`` (the ids of the nodes in the bounding box, or
    None), and not in ``exclude_ids``. ``None`` when none is left."""
    from pyrosm.int64_hash import Int64Set

    if store is None:
        return None
    n = len(store)
    mask = np.fromiter((keep(store.tags(w)) for w in range(n)), dtype=bool, count=n)
    if in_box is not None:
        mask &= wc.with_node_in(store.refs_off, store.refs, Int64Set(in_box))
    if len(exclude_ids):
        mask &= ~wc.in_set(store.id, Int64Set(np.asarray(exclude_ids, dtype=np.int64)))
    if not mask.any():
        return None
    return store.take(np.flatnonzero(mask))
