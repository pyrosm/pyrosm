from pyrosm.data_filter cimport filter_array_dict_by_indices_or_mask
from pyrosm.geometry cimport pygeos_to_shapely, create_relation_geometry
from pyrosm._arrays cimport convert_to_arrays_and_drop_empty, convert_way_records_to_lists
from pygeos import multipolygons
import numpy as np

cdef get_ways_for_relation(member_ids, building_relation_ways):
    mask = np.isin(building_relation_ways["id"], member_ids)
    # If data for relation is not available, skip
    if len(member_ids) != np.sum(mask):
        return None
    return filter_array_dict_by_indices_or_mask(building_relation_ways, mask)

cdef get_relations(relations, relation_ways, node_coordinates):
    cdef int i, j, n2, m_cnt, n = len(relations["id"])

    prepared_relations = []

    for i in range(0, n):
        rel = filter_array_dict_by_indices_or_mask(relations, [i])
        geometries = []
        tag_keys = list(rel["tags"][0].keys())

        boundary = False
        if "boundary" in tag_keys:
            boundary = True

        # There might be multiple features part of the same relation
        n2 = len(rel["members"])
        for j in range(0, n2):
            member_ids = rel["members"][j]["member_id"]
            member_roles = rel["members"][j]["member_role"]

            # Get ways for given relation
            ways = get_ways_for_relation(member_ids, relation_ways)

            if ways is None:
                continue

            geometry = create_relation_geometry(node_coordinates,
                                                ways,
                                                member_roles,
                                                boundary
                                                )

            if geometry is not None:
                geometries.append(geometry)

        if len(geometries) == 0:
            continue

        elif len(geometries) == 1:
            # Create MultiPolygon if there were multiple geometries for given relation
            if isinstance(geometries[0], np.ndarray):
                geometry = pygeos_to_shapely(multipolygons(geometries[0]))
            else:
                geometry = pygeos_to_shapely(geometries[0])
        else:
            raise ValueError("Invalid relation geometry.")

        relation = dict(
            id=rel["id"][0],
            version=rel["version"][0],
            changeset=rel["changeset"][0],
            timestamp=rel["timestamp"][0],
            geometry=geometry
        )

        # Add tags
        for k, v in rel["tags"][0].items():
            relation[k] = v

        prepared_relations.append(relation)
    return prepared_relations

cdef _prepare_relations(relations, relation_ways, node_coordinates, tags_to_keep):
    # Tags to keep as separate columns
    tags_to_keep += ["id", "nodes", "timestamp", "changeset", "version", "geometry"]

    # Also geometries are parsed in this step
    relation_records = get_relations(relations, relation_ways, node_coordinates)
    data = convert_way_records_to_lists(relation_records, tags_to_keep)
    arrays = convert_to_arrays_and_drop_empty(data)
    return arrays

cpdef prepare_relations(relations, relation_ways, node_coordinates, tags_to_keep):
    return _prepare_relations(relations, relation_ways, node_coordinates, tags_to_keep)
