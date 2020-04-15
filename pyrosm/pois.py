from pyrosm.data_manager import get_osm_data
from pyrosm.geometry import create_polygon_geometries
from pyrosm.frames import create_gdf
from pyrosm.relations import prepare_relations
import geopandas as gpd
import warnings

def get_poi_data(node_coordinates, way_records, relations, tags_as_columns, custom_filter):
    # If custom_filter has not been defined, initialize with default
    if custom_filter is None:
        custom_filter = {"amenity": True,
                         "craft": True,
                         "historic": True,
                         "leisure": True,
                         "shop": True,
                         "tourism": True
                         }
    else:
        # Check that the custom filter is in correct format
        if not isinstance(custom_filter, dict):
            raise ValueError(f"'custom_filter' should be a Python dictionary. "
                             f"Got {custom_filter} with type {type(custom_filter)}.")

    # Call signature for fetching buildings
    ways, relation_ways, relations = get_osm_data(way_records=way_records,
                                                  relations=relations,
                                                  tags_as_columns=tags_as_columns,
                                                  custom_filter=custom_filter,
                                                  filter_type="keep"
                                                  )

    # If there weren't any data, return empty GeoDataFrame
    if ways is None:
        warnings.warn("Could not find any POIs for given area.",
                      UserWarning,
                      stacklevel=2)
        return gpd.GeoDataFrame()

    # Create geometries for normal ways
    geometries = create_polygon_geometries(node_coordinates,
                                           ways)

    # Convert to GeoDataFrame
    way_gdf = create_gdf(ways, geometries)

    # Prepare relation data if it is available
    if relations is not None:
        relations = prepare_relations(relations, relation_ways,
                                      node_coordinates,
                                      tags_as_columns)
        relation_gdf = gpd.GeoDataFrame(relations)
        gdf = way_gdf.append(relation_gdf, ignore_index=True)
    else:
        gdf = way_gdf
    return gdf
