import pandas as pd
import geopandas as gpd
import numpy as np
from pyrosm._arrays cimport concatenate_dicts_of_arrays
from pyrosm.geometry cimport _create_point_geometries
from pyrosm.geometry cimport create_way_geometries
from pyrosm.relations import prepare_relations
from shapely.geometry import box
from pyrosm.data_filter import get_mask_by_osmid, _filter_array_dict_by_indices_or_mask
from shapely import multilinestrings
from pyrosm.distance import calculate_geom_length, calculate_geom_array_length

cpdef create_nodes_gdf(nodes, osmids_to_keep=None):
    cdef str k
    if isinstance(nodes, list):
        nodes = concatenate_dicts_of_arrays(nodes)

    # Check if nodes should be filtered
    if osmids_to_keep is not None:
        if isinstance(osmids_to_keep, np.ndarray):
            # Get mask for nodeid array
            mask = get_mask_by_osmid(nodes["id"], osmids_to_keep)
            nodes = _filter_array_dict_by_indices_or_mask(nodes, mask)
        else:
            raise ValueError("'indices_to_keep' should be a numpy array.")

    df = pd.DataFrame()
    for k, v in nodes.items():
        df[k] = v
    df['geometry'] = _create_point_geometries(nodes['lon'], nodes['lat'])
    return gpd.GeoDataFrame(df, crs='epsg:4326')

cpdef create_gdf(data_arrays, geometry_array):
    df = create_df(data_arrays)
    df['geometry'] = geometry_array
    return gpd.GeoDataFrame(df, crs='epsg:4326')

cpdef create_df(data_arrays):
    cdef str key
    df = pd.DataFrame()
    for key, data in data_arrays.items():
        # When inserting nodes,
        # those should be converted
        # to lists to avoid block error
        if key == "nodes":
            df[key] = data.tolist()
        else:
            df[key] = data

    return df

cpdef prepare_way_gdf(node_coordinates, ways, parse_network, calculate_seg_lengths):
    if ways is not None:
        ways, geometries, from_ids, to_ids, node_attributes = create_way_geometries(
            node_coordinates,
            ways,
            parse_network
        )

        # Convert to DataFrame
        way_gdf = create_df(ways)
        way_gdf['osm_type'] = "way"
        way_gdf["geometry"] = geometries

        # In case network is parsed, include way-level length info
        if parse_network and not calculate_seg_lengths:
            # Drop rows without geometry
            way_gdf = way_gdf.dropna(subset=['geometry']).reset_index(drop=True)

            # Create MultiLineStrings and calculate the length
            way_gdf["geometry"] = [multilinestrings(geom) for geom in way_gdf["geometry"]]
            way_gdf["length"] = [calculate_geom_length(geom) for geom in way_gdf["geometry"]]
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")

            # If only edges are requested, clean node_attributes
            node_attributes = None

        # In case network is parsed and requested for graph export,
        # include segment-level length info
        elif parse_network and calculate_seg_lengths:
            # Insert way-level from/to ids
            way_gdf["u"] = from_ids
            way_gdf["v"] = to_ids

            # Drop rows without geometry
            way_gdf = way_gdf.dropna(subset=['geometry']).reset_index(drop=True)

            # Parse segment level from/to-ids
            u = np.concatenate(way_gdf["u"].to_list())
            v = np.concatenate(way_gdf["v"].to_list())

            # Explode multi-geometries
            way_gdf = way_gdf.explode("geometry").reset_index(drop=True)
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")

            # Update from/to-ids
            way_gdf["u"] = u
            way_gdf["v"] = v

            # Calculate the length of the geometries
            way_gdf["length"] = calculate_geom_array_length(way_gdf.geometry.values.to_numpy())

        # For cases not related to networks
        else:
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")
            node_attributes = None

    else:
        way_gdf = gpd.GeoDataFrame()
        node_attributes = None

    return way_gdf, node_attributes

cpdef prepare_node_gdf(nodes):
    if nodes is not None:
        # Create GeoDataFrame from nodes
        node_gdf = create_nodes_gdf(nodes)
        node_gdf['osm_type'] = "node"
    else:
        node_gdf = gpd.GeoDataFrame()
    return node_gdf

cpdef prepare_relation_gdf(node_coordinates, relations, relation_ways, tags_as_columns):
    if relations is not None:
        relations = prepare_relations(relations, relation_ways,
                                      node_coordinates,
                                      tags_as_columns)

        relation_gdf = gpd.GeoDataFrame(relations, crs="epsg:4326")
        relation_gdf['osm_type'] = "relation"

    else:
        relation_gdf = gpd.GeoDataFrame()
    return relation_gdf

cpdef prepare_geodataframe(nodes, node_coordinates, ways,
                           relations, relation_ways,
                           tags_as_columns, bounding_box,
                           parse_network=False,
                           calculate_seg_lengths=False):

    # Prepare ways
    way_gdf, node_attr = prepare_way_gdf(node_coordinates,
                                         ways,
                                         parse_network,
                                         calculate_seg_lengths)

    # Prepare relation data
    relation_gdf = prepare_relation_gdf(node_coordinates, relations, relation_ways, tags_as_columns)

    # When not parsing the network,
    # nodes should be kept as part of the main output
    if not parse_network:
        # Prepare nodes
        node_gdf = prepare_node_gdf(nodes)
    else:
        node_gdf = gpd.GeoDataFrame()

    # Merge all
    gdf = pd.concat([node_gdf, way_gdf, relation_gdf])

    if len(gdf) == 0:
        if parse_network:
            return None, None
        return None

    gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

    # When parsing the network with nodes, prepare the nodes frame
    if node_attr is not None:
        node_attr = pd.DataFrame(node_attr)
        node_attr = gpd.GeoDataFrame(node_attr,
                                     crs="epsg:4326",
                                     geometry=gpd.points_from_xy(node_attr["lon"],
                                                                 node_attr["lat"])
                                     ).drop_duplicates("id").reset_index(drop=True)

    # Filter by bounding box if it was used
    if bounding_box is not None:
        if isinstance(bounding_box, list):
            bounding_box = box(*bounding_box)
        # Filter data spatially
        orig_cols = list(gdf.columns)
        filter_gdf = gpd.GeoDataFrame({"geometry": [bounding_box]},
                                      crs="epsg:4326",
                                      index=[0])
        gdf = gpd.sjoin(gdf, filter_gdf, how="inner")
        gdf = gdf[orig_cols].reset_index(drop=True)

        if node_attr is not None:
            orig_node_cols = list(node_attr.columns)
            node_attr = gpd.sjoin(node_attr, filter_gdf, how="inner")
            node_attr = node_attr[orig_node_cols]

    if len(gdf) == 0:
        if parse_network:
            return None, None
        return None

    if parse_network:
        return gdf, node_attr
    return gdf
