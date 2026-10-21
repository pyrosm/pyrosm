"""Bounding-box helpers: validate/normalise a box, reduce it to its bounds, flag in-box
coordinates, and read back the in-box node ids spilled per shard."""

import numpy as np
from pyrosm.utils import validate_bounding_box


def _normalize_bounding_box(bounding_box):
    """``bounding_box`` checked and normalised as ``OSM.__init__`` does (see
    :func:`~pyrosm.utils.validate_bounding_box`); ``None`` stays ``None``."""
    return None if bounding_box is None else validate_bounding_box(bounding_box)


def _bbox_bounds(bounding_box):
    """The ``(xmin, ymin, xmax, ymax)`` rectangle of a bounding box (a list/tuple or a
    shapely geometry), used to flag in-box nodes; ``None`` when no box is given."""
    if bounding_box is None:
        return None
    if isinstance(bounding_box, (list, tuple)):
        return tuple(bounding_box)
    return tuple(bounding_box.bounds)


def _in_box_mask(lon, lat, bounds):
    """Boolean mask of the coordinates inside ``bounds`` (``xmin, ymin, xmax, ymax``)."""
    xmin, ymin, xmax, ymax = bounds
    return (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)


def _filter_features_to_box(found, bounds):
    """Keep only the node features whose coordinate is inside ``bounds``, or ``None``."""
    mask = _in_box_mask(found["lon"], found["lat"], bounds)
    if not mask.any():
        return None
    return {
        k: ([t for t, m in zip(v, mask) if m] if k == "tags" else np.asarray(v)[mask])
        for k, v in found.items()
    }


def _in_box_nodes(shard_paths):
    """The unique ids of all nodes that fell inside the bounding box (spilled per shard).
    A way is kept when at least one of its nodes is in this set (complete-ways semantics).
    """
    from pyrosm.engine.decode import _open_shard

    ids = [z for z in (_open_shard(p)["in_box_id"] for p in shard_paths) if len(z)]
    return np.unique(np.concatenate(ids)) if ids else np.empty(0, np.int64)
