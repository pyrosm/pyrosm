"""Relations that group their members, such as stop areas, routes and sites, as
:meth:`pyrosm.OSM.get_data_by_custom_criteria` returns them.

A relation tagged ``type=multipolygon`` or ``type=boundary`` forms an area of its member
ways, and those ways are not returned as rows of their own, except the inner ways of a
multipolygon whose own tags match the filter: a hole tagged as a feature of its own, such
as a lift in a corridor (see :func:`area_member_way_ids`). Any other relation groups its
members: its row holds a GeometryCollection of the members, and a member way whose own
tags match the filter is also returned as a row of its own. The layer reads, such as
:meth:`pyrosm.OSM.get_buildings`, take every relation as an area: there, only the inner
ways of multipolygons whose own tags match are rows of their own.
"""

import numpy as np
import shapely

# The relation types whose member ways make up the relation's area.
AREA_TYPES = ("multipolygon", "boundary")


def is_grouping(tags):
    """Whether a relation with the OSM tags ``tags`` groups its members rather than
    forming an area of them (see :data:`AREA_TYPES`)."""
    return tags.get("type") not in AREA_TYPES


def _member_kind(value):
    """A member type as text: ``"node"``, ``"way"`` or ``"relation"``."""
    return value.decode() if isinstance(value, bytes) else str(value)


def member_ids(relations, kind, grouping):
    """The ids of the members of type ``kind`` (``"node"`` or ``"way"``) of the
    ``relations`` (a dict of arrays with ``tags`` and ``members``) that group their
    members, when ``grouping``, or of those that form areas, when not; as an int64
    array (unsorted, possibly repeated)."""
    found = []
    for tags, members in zip(relations["tags"], relations["members"]):
        if is_grouping(tags) != grouping:
            continue
        kinds = np.array(
            [_member_kind(t) for t in members["member_type"]], dtype=object
        )
        found.append(np.asarray(members["member_id"], dtype=np.int64)[kinds == kind])
    return np.concatenate(found) if found else np.empty(0, np.int64)


def area_member_way_ids(relations, grouping=True):
    """The ids of the member ways that the ``relations`` (a dict of arrays with ``tags``
    and ``members``) that form areas take in, so that they are not rows of their own: all
    their member ways but the inner ways of multipolygons, holes such as a lift in a
    corridor that are rows of their own when their tags match the filter. Without
    ``grouping`` (a layer read), every relation forms an area. A way that is also an
    outer way of another of the relations stays taken in. An int64 array (unsorted,
    possibly repeated)."""
    found = []
    for tags, members in zip(relations["tags"], relations["members"]):
        if grouping and is_grouping(tags):
            continue
        ways = np.array(
            [_member_kind(t) == "way" for t in members["member_type"]], dtype=bool
        )
        if tags.get("type") == "multipolygon":
            ways &= np.array(
                [_member_kind(r) != "inner" for r in members["member_role"]],
                dtype=bool,
            )
        found.append(np.asarray(members["member_id"], dtype=np.int64)[ways])
    return np.concatenate(found) if found else np.empty(0, np.int64)


def grouping_records(
    relations, relation_ways, node_coordinates, keep_metadata, box=None
):
    """The records of the grouping ``relations``, as
    :func:`pyrosm.relations.relation_records` gives the records of other relations: each
    with a GeometryCollection of its members in their order, a LineString for each member
    way with at least two nodes in ``node_coordinates`` and a Point for each member node
    there (with ``box``, a shapely geometry, only those it covers). A relation with
    neither is left out; member relations are not included."""
    from pyrosm.relations import members_to_list

    way_nodes = {}
    if relation_ways is not None:
        way_nodes = dict(
            zip(np.asarray(relation_ways["id"]).tolist(), relation_ways["nodes"])
        )
    records = []
    for i, rid in enumerate(relations["id"]):
        members = relations["members"][i]
        parts = []
        for mid, kind in zip(members["member_id"], members["member_type"]):
            kind = _member_kind(kind)
            if kind == "way" and int(mid) in way_nodes:
                ids = np.asarray(way_nodes[int(mid)], dtype=np.int64)
                found, lon, lat = node_coordinates.gather(ids)
                present = found >= 0
                if present.sum() >= 2:
                    parts.append(shapely.linestrings(lon[present], y=lat[present]))
            elif kind == "node":
                found, lon, lat = node_coordinates.gather(
                    np.array([mid], dtype=np.int64)
                )
                point = shapely.Point(lon[0], lat[0])
                if found[0] >= 0 and (box is None or box.covers(point)):
                    parts.append(point)
        if not parts:
            continue
        record = dict(id=rid, geometry=shapely.GeometryCollection(parts))
        if keep_metadata:
            record["version"] = relations["version"][i]
            record["changeset"] = relations["changeset"][i]
            record["timestamp"] = relations["timestamp"][i]
        record["visible"] = bool(relations["visible"][i])
        for key, value in relations["tags"][i].items():
            record["id_tag" if key == "id" else key] = value
        record["members"] = members_to_list(members)
        records.append(record)
    return records
