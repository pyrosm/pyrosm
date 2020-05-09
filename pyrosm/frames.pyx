import pandas as pd
import geopandas as gpd
from pyrosm._arrays cimport concatenate_dicts_of_arrays
from pyrosm.geometry cimport _create_point_geometries
from pyrosm.geometry cimport create_way_geometries
from pyrosm.relations import prepare_relations
from shapely.geometry import box


cpdef create_nodes_gdf(nodes):
    cdef str k
    if isinstance(nodes, list):
        nodes = concatenate_dicts_of_arrays(nodes)
    df = pd.DataFrame()
    for k, v in nodes.items():
        df[k] = v
    df['geometry'] = _create_point_geometries(nodes['lon'], nodes['lat'])
    return gpd.GeoDataFrame(df, crs='epsg:4326')

cpdef create_gdf(data_arrays, geometry_array):
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

    df['geometry'] = geometry_array
    return gpd.GeoDataFrame(df, crs='epsg:4326')

cpdef prepare_way_gdf(node_coordinates, ways):
    if ways is not None:
        geometries = create_way_geometries(node_coordinates,
                                           ways)
        # Convert to GeoDataFrame
        way_gdf = create_gdf(ways, geometries)
        way_gdf['osm_type'] = "way"
    else:
        way_gdf = gpd.GeoDataFrame()
    return way_gdf

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

        relation_gdf = gpd.GeoDataFrame(relations)
        relation_gdf['osm_type'] = "relation"

    else:
        relation_gdf = gpd.GeoDataFrame()
    return relation_gdf

cpdef prepare_geodataframe(nodes, node_coordinates, ways,
                           relations, relation_ways,
                           tags_as_columns, bounding_box):
    # Prepare nodes
    node_gdf = prepare_node_gdf(nodes)

    # Prepare ways
    way_gdf = prepare_way_gdf(node_coordinates, ways)

    # Prepare relation data
    relation_gdf = prepare_relation_gdf(node_coordinates, relations, relation_ways, tags_as_columns)

    # Merge all
    gdf = pd.concat([node_gdf, way_gdf, relation_gdf])

    if len(gdf) == 0:
        return None

    gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

    # Filter by bounding box if it was used
    # This would be faster using Pygeos,
    # but as GeoPandas 0.8.0 should start supporting it natively,
    # let's keep things simple
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

    if len(gdf) == 0:
        return None

    return gdf
