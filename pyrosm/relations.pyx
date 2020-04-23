from pyrosm.data_filter cimport filter_array_dict_by_indices_or_mask
from pyrosm.geometry cimport pygeos_to_shapely, create_relation_geometry
from pyrosm._arrays cimport convert_to_arrays_and_drop_empty, convert_way_records_to_lists
from pygeos import multipolygons
import numpy as np
import geopandas as gpd

cdef get_ways_for_relation(member_ids, member_roles, building_relation_ways):
    # Ensure there are no duplicate member_ids/roles
    member_ids, idx = np.unique(member_ids, return_index=True)
    member_roles = member_roles[idx]

    mask = np.isin(building_relation_ways["id"], member_ids)

    # If data for relation is not available, skip
    if np.sum(mask) == 0:
        return None, None, None

    # If some of the ways were missing, use the ones that are available.
    # Might cause incorrectly shaped geometry but it's better than not having
    # the data for relation at all
    elif len(member_ids) != np.sum(mask):
        # Filter member_ids and roles accordingly
        member_mask = np.isin(member_ids, building_relation_ways["id"])
        member_ids = member_ids[member_mask]
        member_roles = member_roles[member_mask]

    ways = filter_array_dict_by_indices_or_mask(building_relation_ways, mask)
    return member_ids, member_roles, ways

cdef get_relations(relations, relation_ways, node_coordinates):
    cdef int i, j, n2, m_cnt, n = len(relations["id"])

    prepared_relations = []

    for i in range(0, n):
        rel = filter_array_dict_by_indices_or_mask(relations, [i])
        geometries = []
        tag_keys = list(rel["tags"][0].keys())

        # There might be multiple features part of the same relation
        n2 = len(rel["members"])
        for j in range(0, n2):
            member_ids = rel["members"][j]["member_id"]
            member_roles = rel["members"][j]["member_role"]

            boundary = False
            if "boundary" in tag_keys \
                    or "route" in tag_keys \
                        or member_ids[0] == member_ids[-1]:
                boundary = True

            # Get ways for given relation
            member_ids, member_roles, ways = get_ways_for_relation(member_ids,
                                                                   member_roles,
                                                                   relation_ways)

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

    # Return empty frame if no relation records were successfully parsed
    if len(relation_records) == 0:
        return gpd.GeoDataFrame()

    data = convert_way_records_to_lists(relation_records, tags_to_keep)
    arrays = convert_to_arrays_and_drop_empty(data)
    return arrays

cpdef prepare_relations(relations, relation_ways, node_coordinates, tags_to_keep):
    return _prepare_relations(relations, relation_ways, node_coordinates, tags_to_keep)
