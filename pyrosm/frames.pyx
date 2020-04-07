import pandas as pd
import geopandas as gpd

cdef merge_nodes(node_dict_list):
    nodes = [pd.DataFrame(node_dict) for node_dict in node_dict_list]
    return pd.concat(nodes).reset_index(drop=True)

cpdef create_nodes_gdf(node_dict_list):
    nodes = merge_nodes(node_dict_list)
    nodes['geometry'] = gpd.points_from_xy(nodes['lon'], nodes['lat'])
    nodes = gpd.GeoDataFrame(nodes, crs='epsg:4326')
    return nodes

cpdef create_way_gdf(data_records, geometry_array):
    datasets = [v for v in data_records.values()]
    keys = list(data_records.keys())
    data = pd.DataFrame()
    for i, key in enumerate(keys):
        data[key] = datasets[i]
    data['geometry'] = geometry_array
    return gpd.GeoDataFrame(data, crs='epsg:4326')