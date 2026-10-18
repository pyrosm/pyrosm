"""The relation records of :func:`pyrosm.relations.relation_records`, built with vectorized
shapely calls over all relations instead of calls per relation.

Each step of the relation assembly in ``relations.pyx`` and ``geometry.pyx`` (the member
ways' lines, ``line_merge``, the closed rings, their polygons, the even-odd overlay of the
rings, the validity fixes) runs once for all relations, split across the read's threads
where GEOS does the work (shapely releases the GIL for it). The rings of a relation are
overlaid in rounds, the ``k``-th ring of every relation in round ``k``, so each relation's
operations are those of the per-relation code, in its order. A vectorized call that raises
is repeated relation by relation, with the error handling of that code. The validity
checks run on the calling thread (``shapely.is_valid`` changes the warning filters).
"""

import numpy as np
import shapely
from shapely import GEOSException

from pyrosm._arrays import _offsets_from_lengths
from pyrosm.engine.pool import read_threads, thread_map
from pyrosm.geometry import (
    _polygonal_only,
    fix_geometry,
    ring_polygon,
    valid_coordinates,
)
from pyrosm.relations import members_to_list, relation_kind

# Shapely type ids.
_LINESTRING, _MULTILINESTRING, _COLLECTION = 1, 5, 7


def _on_thread_slices(function, *arrays):
    """``function(*arrays)``, run on the read's threads, each on a slice of the arrays."""
    n = len(arrays[0])
    threads = read_threads()
    if threads <= 1 or n < 2 * threads:
        return function(*arrays)
    bounds = np.linspace(0, n, threads + 1).astype(int)
    parts = thread_map(
        lambda k: function(*(a[bounds[k] : bounds[k + 1]] for a in arrays)),
        range(threads),
    )
    return np.concatenate(list(parts))


def _each_or_none(function, *arrays):
    """``function`` of each element of ``arrays``, ``None`` where it raises
    ``GEOSException``; ``(results, failed)``."""
    out = np.empty(len(arrays[0]), dtype=object)
    failed = np.zeros(len(out), dtype=bool)
    for i, values in enumerate(zip(*arrays)):
        try:
            out[i] = function(*values)
        except GEOSException:
            failed[i] = True
    return out, failed


def _members(relations, way_ids):
    """The member ways of each relation, as ``get_ways_for_relation`` finds them: for each
    found (relation, way row) pair in order, the relation's index and the way's row in
    ``way_ids``; ``way_ids`` must be unique."""
    n = len(relations["id"])
    member_ids = [m["member_id"] for m in relations["members"]]
    counts = np.array([len(m) for m in member_ids], dtype=np.int64)
    if counts.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ids = np.concatenate(member_ids).astype(np.int64)
    owner = np.repeat(np.arange(n), counts)
    # Each relation's distinct member ids (``np.unique`` per relation).
    order = np.lexsort((ids, owner))
    ids, owner = ids[order], owner[order]
    first = np.ones(len(ids), dtype=bool)
    first[1:] = (ids[1:] != ids[:-1]) | (owner[1:] != owner[:-1])
    ids, owner = ids[first], owner[first]
    # Each member's row in ``way_ids``; the members not found are dropped.
    way_order = np.argsort(way_ids, kind="stable")
    sorted_ids = np.asarray(way_ids)[way_order]
    pos = np.searchsorted(sorted_ids, ids)
    found = pos < len(sorted_ids)
    found[found] = sorted_ids[pos[found]] == ids[found]
    owner, rows = owner[found], way_order[pos[found]]
    # Each relation's ways in the order of their rows (np.sort of the found rows).
    order = np.lexsort((rows, owner))
    return owner[order], rows[order]


def _member_lines(owner, rows, relation_ways, node_coordinates):
    """The member lines: for the members (``owner``, ``rows``) with at least two nodes
    present with valid coordinates, a LineString through those nodes. Returns the lines'
    relation indices and lines."""
    nodes = relation_ways["nodes"]
    refs = [np.asarray(nodes[r], dtype=np.int64) for r in rows]
    counts = np.array([len(r) for r in refs], dtype=np.int64)
    if counts.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, dtype=object)
    flat = np.concatenate(refs)
    idx, lon, lat = node_coordinates.gather(flat)
    valid = valid_coordinates(idx, lon, lat)
    # The member of each valid coordinate; a member with fewer than two gets no line.
    member = np.repeat(np.arange(len(rows)), counts)[valid]
    n_valid = np.bincount(member, minlength=len(rows))
    has_line = n_valid >= 2
    on_line = has_line[member]
    coords = np.column_stack([lon[valid][on_line], lat[valid][on_line]])
    index = np.repeat(np.arange(has_line.sum()), n_valid[has_line])
    return owner[has_line], shapely.linestrings(coords, indices=index)


def _reduce_overlay(ring_owner, rings, n):
    """The even-odd overlay of each relation's rings: their ``symmetric_difference`` taken
    in pairs, round by round, an odd last part carried to the next round, as
    ``geometry._even_odd_overlay`` takes it; ``None`` for a relation without rings or whose
    overlay raises."""
    area = np.empty(n, dtype=object)
    failed = np.zeros(n, dtype=bool)
    order = np.argsort(ring_owner, kind="stable")
    owner, parts = ring_owner[order], rings[order]
    while True:
        # ``owner`` is sorted: ``position`` is each part's index among its relation's
        # parts and ``count`` their number; ``left`` the first part of each pair.
        position = np.arange(len(owner)) - np.searchsorted(owner, owner)
        count = np.bincount(owner, minlength=n)[owner]
        left = np.flatnonzero((position % 2 == 0) & (position + 1 < count))
        if len(left) == 0:
            break
        try:
            merged = _on_thread_slices(
                shapely.symmetric_difference, parts[left], parts[left + 1]
            )
        except GEOSException:
            merged, error = _each_or_none(
                shapely.symmetric_difference, parts[left], parts[left + 1]
            )
            failed[owner[left[error]]] = True
        # An unpaired last part goes to the next round as it is.
        carried = np.flatnonzero((position == count - 1) & (position % 2 == 0))
        owner = np.concatenate([owner[left], owner[carried]])
        parts = np.concatenate([merged, parts[carried]])
        # The relations whose overlay raised are dropped; the parts are regrouped by
        # relation, in order.
        keep = ~failed[owner]
        order = np.argsort(owner[keep], kind="stable")
        owner, parts = owner[keep][order], parts[keep][order]
    area[owner] = parts
    return area


def _rings(owner, merged):
    """The ring polygons of the merged member lines: for each LineString component of a
    relation's merged lines that is a closed ring of at least four points, its polygon
    (fixed with ``buffer(0)`` when invalid), unless empty or GEOS raises. Returns the
    rings' relation indices and polygons, in each relation's order."""
    types = shapely.get_type_id(merged)
    single = types == _LINESTRING
    multi = np.flatnonzero((types == _MULTILINESTRING) | (types == _COLLECTION))
    parts, part_of = shapely.get_parts(merged[multi], return_index=True)
    part_lines = shapely.get_type_id(parts) == _LINESTRING
    lines = parts[part_lines]
    line_owner = owner[multi[part_of[part_lines]]]
    lines = np.concatenate([merged[single], lines])
    line_owner = np.concatenate([owner[single], line_owner])
    # Each relation's components in their order (a relation is a single LineString or has
    # parts, which come in their order).
    order = np.argsort(line_owner, kind="stable")
    lines, line_owner = lines[order], line_owner[order]
    if len(lines) == 0:
        return np.empty(0, np.int64), np.empty(0, dtype=object)
    coords, coord_of = shapely.get_coordinates(lines, return_index=True)
    counts = np.bincount(coord_of, minlength=len(lines))
    starts = _offsets_from_lengths(counts)[:-1]
    ends = starts + counts - 1
    # A closed ring: at least four points, the first equal to the last.
    closed = counts >= 4
    closed[closed] = (coords[starts[closed]] == coords[ends[closed]]).all(axis=1)
    coords = coords[np.repeat(closed, counts)]
    lines, line_owner = lines[closed], line_owner[closed]
    if len(lines) == 0:
        return np.empty(0, np.int64), np.empty(0, dtype=object)

    try:
        rings = shapely.polygons(
            shapely.linearrings(
                coords, indices=np.repeat(np.arange(len(lines)), counts[closed])
            )
        )
        valid = shapely.is_valid(rings)
        if not valid.all():
            rings[~valid] = _on_thread_slices(_repaired_polygons, rings[~valid])
    except GEOSException:
        rings = np.array(
            [ring_polygon(shapely.get_coordinates(line)) for line in lines],
            dtype=object,
        )
    keep = shapely.is_geometry(rings)
    keep[keep] = ~shapely.is_empty(rings[keep])
    return line_owner[keep], rings[keep]


def _repaired_polygons(geometries):
    """``geometries`` buffered by 0, which repairs invalid polygons."""
    return shapely.buffer(geometries, 0)


def relation_records(relations, relation_ways, node_coordinates, keep_metadata):
    """The records of ``relations`` as :func:`pyrosm.relations.relation_records` gives
    them for all of them, in order."""
    n = len(relations["id"])
    tags = relations["tags"]
    kinds = [relation_kind(tag) for tag in tags]
    force = np.array([k[0] for k in kinds], dtype=bool)
    owner, rows = _members(relations, relation_ways["id"])
    has_ways = np.bincount(owner, minlength=n) > 0
    line_owner, lines = _member_lines(owner, rows, relation_ways, node_coordinates)
    line_counts = np.bincount(line_owner, minlength=n)
    geometry = np.empty(n, dtype=object)

    # Lines: one member line is the geometry, more are merged.
    linear = np.flatnonzero(force & (line_counts > 0))
    single = linear[line_counts[linear] == 1]
    if len(single):
        first = np.searchsorted(line_owner, single)
        geometry[single] = lines[first]
    several = linear[line_counts[linear] > 1]
    if len(several):
        mask = np.isin(line_owner, several)
        multis = shapely.multilinestrings(
            lines[mask], indices=np.searchsorted(several, line_owner[mask])
        )
        geometry[several] = _on_thread_slices(shapely.line_merge, multis)

    # Areas: merge the member lines, take the closed rings and overlay them.
    areal = np.flatnonzero(~force & (line_counts > 0))
    if len(areal):
        mask = np.isin(line_owner, areal)
        local = np.searchsorted(areal, line_owner[mask])
        multis = shapely.multilinestrings(lines[mask], indices=local)
        merged = _on_thread_slices(shapely.line_merge, multis)
        ring_owner, rings = _rings(np.arange(len(areal)), merged)
        area = _reduce_overlay(ring_owner, rings, len(areal))
        area = np.array([_polygonal_only(a) for a in area], dtype=object)
        present = np.flatnonzero(shapely.is_geometry(area))
        if len(present):
            valid = shapely.is_valid(area[present])
            for i in present[~valid]:
                area[i] = _polygonal_only(fix_geometry(area[i]))
        geometry[areal] = area

    built = np.flatnonzero(has_ways & shapely.is_geometry(geometry))
    if len(built):
        valid = shapely.is_valid(geometry[built])
        for i in built[~valid & ~force[built]]:
            geometry[i] = fix_geometry(geometry[i])

    records = []
    for i in built:
        record = dict(id=relations["id"][i], geometry=geometry[i])
        if keep_metadata:
            record["version"] = relations["version"][i]
            record["changeset"] = relations["changeset"][i]
            record["timestamp"] = relations["timestamp"][i]
        for k, v in tags[i].items():
            record[k] = v
        record["members"] = members_to_list(relations["members"][i])
        records.append(record)
    return records
