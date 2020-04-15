from pyrosm.data_manager import get_osm_data
from pyrosm.frames import create_gdf
from pyrosm.geometry import create_way_geometries
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

    geometries = create_way_geometries(node_coordinates,
                                       ways)

    # Convert to GeoDataFrame
    gdf = create_gdf(ways, geometries)
    gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

    return gdf
