import pandas as pd
import geopandas as gpd
from pyrosm._arrays cimport concatenate_dicts_of_arrays
from pyrosm.geometry cimport _create_point_geometries


cpdef create_nodes_gdf(node_dict_list):
    nodes = concatenate_dicts_of_arrays(node_dict_list)
    df = pd.DataFrame()
    for k, v in nodes.items():
        df[k] = v
    df['geometry'] = _create_point_geometries(nodes['lon'], nodes['lat'])
    return gpd.GeoDataFrame(df, crs='epsg:4326')


cpdef create_gdf(data_arrays, geometry_array):
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
