from pyrosm.data_manager import get_osm_data
from pyrosm.geometry import create_polygon_geometries
from pyrosm.frames import create_gdf, create_nodes_gdf
from pyrosm.relations import prepare_relations
import geopandas as gpd
import pandas as pd
import warnings


def get_poi_data(nodes, node_coordinates, way_records, relations, tags_as_columns, custom_filter):

    # Call signature for fetching POIs
    nodes, ways, relation_ways, relations = get_osm_data(node_arrays=nodes,
                                                         way_records=way_records,
                                                         relations=relations,
                                                         tags_as_columns=tags_as_columns,
                                                         data_filter=custom_filter,
                                                         filter_type="keep",
                                                         osm_keys=None,
                                                         )

    # If there weren't any data, return empty GeoDataFrame
    if nodes is None and ways is None and relations is None:
        warnings.warn("Could not find any POIs for given area.",
                      UserWarning,
                      stacklevel=2)
        return gpd.GeoDataFrame()

    if nodes is not None:
        # Create GeoDataFrame from nodes
        node_gdf = create_nodes_gdf(nodes)
        node_gdf['osm_type'] = "node"
    else:
        node_gdf = gpd.GeoDataFrame()

    if ways is not None:
        # Create geometries for normal ways
        geometries = create_polygon_geometries(node_coordinates,
                                               ways)
        # Convert to GeoDataFrame
        way_gdf = create_gdf(ways, geometries)
        node_gdf['osm_type'] = "way"
    else:
        way_gdf = gpd.GeoDataFrame()

    # Prepare relation data if it is available
    if relations is not None:
        relations = prepare_relations(relations, relation_ways,
                                      node_coordinates,
                                      tags_as_columns)
        relation_gdf = gpd.GeoDataFrame(relations)
        node_gdf['osm_type'] = "relation"

    else:
        relation_gdf = gpd.GeoDataFrame()

    # Merge all
    gdf = pd.concat([node_gdf, way_gdf, relation_gdf])
    gdf = gdf.dropna(subset=['geometry'])
    return gdf
