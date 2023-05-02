from pyrosm.data_filter cimport filter_array_dict_by_indices_or_mask
from pyrosm.geometry cimport create_relation_geometry, fix_geometry
from pyrosm._arrays cimport convert_to_arrays_and_drop_empty, convert_way_records_to_lists
from shapely import multipolygons, multilinestrings
from shapely.predicates import is_valid
import numpy as np
import geopandas as gpd

cpdef _get_ways_for_relation(member_ids, member_roles, building_relation_ways):
    return get_ways_for_relation(member_ids, member_roles, building_relation_ways)

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
        tag_keys = list(rel["tags"][0].keys())
        tag = rel["tags"][0]

        # Check if geometry should NOT be polygon
        force_linestring = False

        # ==========================================
        # Determine if relation should be LineString
        # ==========================================
        # OSM Relation keys that are typically LineStrings
        linestring_keys = ["barrier", "route",
                           "railway", "highway",
                           "waterway"]
        for lsk in linestring_keys:
            if lsk in tag_keys:
                # Railway
                # -------
                if lsk == "railway":
                    # https://wiki.openstreetmap.org/wiki/Key:railway
                    if tag["railway"] not in ["platform", "station", "turntable",
                                              "roundhouse", "traverser", "wash"]:
                        force_linestring = True
                        break
                # Highway
                # -------
                elif lsk == "highway":
                    # https://wiki.openstreetmap.org/wiki/Key:highway
                    if tag["highway"] == "pedestrian":
                        if "area" in tag_keys:
                            # If highway is pedestrian area, it should be a Polygon
                            if tag["area"] != "yes":
                                force_linestring = True
                            break
                    elif tag["highway"] not in ["platform", "rest_area", "services"]:
                        force_linestring = True
                        break
                # Waterway
                # --------
                elif lsk == "waterway":
                    # https://wiki.openstreetmap.org/wiki/Key:waterway
                    if tag["waterway"] not in ["riverbank", "dock", "boatyard",
                                               "dam", "fuel"]:
                        force_linestring = True
                        break
                else:
                    force_linestring = True
                    break

        if "area" in tag_keys:
            if tag["area"] == "no":
                force_linestring = True

        # =========================================================
        # Determine if element should be made as MultiPolygon
        # Notice: If 'force_linestring = True', this doesn't apply
        # =========================================================
        make_multipolygon = False
        if "type" in tag_keys:
            if tag["type"] in ["multipolygon", "public_transport",
                               "site", "cluster"]:
                make_multipolygon = True

        member_ids = rel["members"][0]["member_id"]
        member_roles = rel["members"][0]["member_role"]

        # Get ways for given relation
        member_ids, member_roles, ways = get_ways_for_relation(member_ids,
                                                               member_roles,
                                                               relation_ways)

        if ways is None:
            continue

        geometry = create_relation_geometry(node_coordinates,
                                            ways,
                                            member_roles,
                                            force_linestring,
                                            make_multipolygon
                                            )
        if geometry is None:
            continue

        # Check for multigeometries
        if isinstance(geometry[0], list):
            if len(geometry[0]) > 1:
                if not force_linestring:
                    geometry = multipolygons(geometry)
                else:
                    geometry = multilinestrings(geometry)
        else:
            geometry = geometry[0]

        # Check if geometry is valid
        is_valid_geom = is_valid(geometry)

        # If geometry was invalid polygon try to fix
        if not force_linestring:
            if not is_valid_geom:
                geometry = fix_geometry(geometry)

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
