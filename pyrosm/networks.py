from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
import geopandas as gpd
import warnings


def get_network_data(node_coordinates, way_records, tags_as_columns, network_filter):
    # Tags to keep as separate columns
    tags_as_columns += ["id", "nodes", "timestamp", "changeset", "version"]

    # Call signature for fetching network data
    nodes, ways, relation_ways, relations = get_osm_data(node_arrays=None,
                                                         way_records=way_records,
                                                         relations=None,
                                                         tags_as_columns=tags_as_columns,
                                                         data_filter=network_filter,
                                                         filter_type="exclude",
                                                         # Keep only records having 'highway' tag
                                                         osm_keys="highway",
                                                         )

    # If there weren't any data, return empty GeoDataFrame
    if ways is None:
        warnings.warn("Could not find any buildings for given area.",
                      UserWarning,
                      stacklevel=2)
        return gpd.GeoDataFrame()

    # Prepare GeoDataFrame
    gdf = prepare_geodataframe(nodes, node_coordinates, ways,
                               relations, relation_ways, tags_as_columns)
    return gdf


