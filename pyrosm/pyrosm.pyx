import geopandas as gpd
import pandas as pd
from pyrosm.config import highway_tags_to_keep
from pyrosm.pbfreader cimport parse_osm_data, get_way_data
from pyrosm.geometry cimport create_way_geometries

class Osm:
    from pyrosm.utils._compat import PYGEOS_SHAPELY_COMPAT

    def __init__(self, filepath,
                 bounding_box=None,
                 verbose=False):
        """
        Reads data from OSM file and parses street networks
        for walking, driving, and cycling.

        Parameters
        ----------

        filepath : str
            Filepath to input OSM dataset ( .osm.pbf | .osm | .osm.bz2 | .osm.gz )

        bounding_box : shapely.Polygon (optional)
            Bounding box (shapely.geometry.Polygon) that can be used to filter OSM data spatially.

        verbose : bool
            If True, will print parsing-related information to the screen.
        """
        if not isinstance(filepath, str):
            raise ValueError("'filepath' should be a string.")
        if not filepath.endswith(".pbf"):
            raise ValueError(f"Input data should be in Protobuf format (*.osm.pbf). "
                             f"Found: {filepath.split('.')[-1]}")

        self.filepath = filepath
        self.bounding_box = bounding_box
        self._verbose = verbose

        self._all_tag_keys = None

    def merge_nodes(self, node_dict_list):
        nodes = [pd.DataFrame(node_dict) for node_dict in node_dict_list]
        return pd.concat(nodes).reset_index(drop=True)

    def create_nodes_gdf(self, node_dict_list):
        nodes = [pd.DataFrame(node_dict) for node_dict in node_dict_list]
        nodes = pd.concat(nodes)
        nodes['geometry'] = gpd.points_from_xy(nodes['lon'], nodes['lat'])
        nodes = gpd.GeoDataFrame(nodes, crs='epsg:4326')
        return nodes

    def create_way_gdf(self, data_records, geometry_array):
        cdef int i
        cdef str key

        datasets = [v for v in data_records.values()]
        keys = list(data_records.keys())
        data = pd.DataFrame()
        for i, key in enumerate(keys):
            data[key] = datasets[i]
        data['geometry'] = geometry_array
        return gpd.GeoDataFrame(data, crs='epsg:4326')

    def parse_osm(self):
        """For testing purposes"""
        import time
        t = time.time()
        # Parse records
        nodes, ways = parse_osm_data(self.filepath, self.bounding_box)

        # TODO: Enable custom configuration of the way tags to keep
        data_records, self._all_tag_keys = get_way_data(ways, highway_tags_to_keep)

        # Create Geometries
        geometries = create_way_geometries(nodes, ways)

        # Convert to GeoDataFrame
        gdf = self.create_way_gdf(data_records, geometries)
        gdf = gdf.dropna(subset=['geometry'])
        print("Lasted", round(time.time()-t, 1), "seconds.")
        return gdf

    def get_network(filepath=None, bounding_box=None):
        pass
