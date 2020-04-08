from pyrosm.config import Conf
from pyrosm.pbfreader import parse_osm_data
from pyrosm.networks import get_way_data
from pyrosm.buildings import get_building_data
from pyrosm.geometry import create_way_geometries, create_polygon_geometries
from pyrosm.frames import create_gdf
from shapely.geometry import Polygon, MultiPolygon


class OSM:
    from pyrosm.utils._compat import PYGEOS_SHAPELY_COMPAT

    def __init__(self, filepath,
                 bounding_box=None):
        """
        Parameters
        ----------

        filepath : str
            Filepath to input OSM dataset ( .osm.pbf | .osm | .osm.bz2 | .osm.gz )

        bounding_box : list | shapely.Polygon (optional)
            Filtering OSM data spatially is allowed by passing a
            bounding box either as a list `[minx, miny, maxx, maxy]` or
            as a `shapely.geometry.Polygon`.

            Note: if using Polygon, the tool will use its bounds.
            Filtering based on complex shapes is not currently supported.
        """
        if not isinstance(filepath, str):
            raise ValueError("'filepath' should be a string.")
        if not filepath.endswith(".pbf"):
            raise ValueError(f"Input data should be in Protobuf format (*.osm.pbf). "
                             f"Found: {filepath.split('.')[-1]}")

        self.filepath = filepath

        if bounding_box is None:
            self.bounding_box = None
        elif type(bounding_box) in [Polygon, MultiPolygon]:
            self.bounding_box = bounding_box.bounds
        elif isinstance(bounding_box, list):
            self.bounding_box = bounding_box
        else:
            raise ValueError("bounding_box should be a list or a shapely Polygon.")

        self.conf = Conf
        # TODO: Add as a parameter
        self._verbose = False

        self._all_way_tags = None
        self._nodes = None
        self._way_records = None
        self._relations = None

    def _read_pbf(self):
        nodes, ways, relations, way_tags = parse_osm_data(self.filepath,
                                                          self.bounding_box,
                                                          exclude_relations=False)
        self._nodes, self._way_records, \
        self._relations, self._all_way_tags = nodes, ways, relations, way_tags

    def _get_filter(self, net_type):
        possible_filters = [a for a in self.conf.network_filters.__dir__()
                            if "__" not in a]
        possible_filters += ["all"]
        possible_values = ", ".join(possible_filters)
        msg = "'net_type' should be one of the following: " + possible_values
        if not isinstance(net_type, str):
            raise ValueError(msg)

        net_type = net_type.lower()

        if net_type not in possible_filters:
            raise ValueError(msg)

        # Get filter
        if net_type == "walking":
            return self.conf.network_filters.walking
        elif net_type == "driving":
            return self.conf.network_filters.driving
        elif net_type == "cycling":
            return self.conf.network_filters.cycling
        elif net_type == "all":
            return None

    def get_network(self, network_type="walking"):
        """
        Reads data from OSM file and parses street networks
        for walking, driving, and cycling.

        Parameters
        ----------

        network_type : str
            What kind of network to parse. Possible values are: 'walking' | 'cycling' | 'driving' | 'all'.

        """
        # Get filter
        network_filter = self._get_filter(network_type)
        tags_to_keep = self.conf.tag_filters.networks

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Filter ways
        ways = get_way_data(self._way_records,
                            tags_to_keep,
                            network_filter
                            )

        geometries = create_way_geometries(self._nodes,
                                           ways)

        # Convert to GeoDataFrame
        gdf = create_gdf(ways, geometries)
        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

        # Do not keep node information
        # (they are in a list, and causes issues saving the files)
        if "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)

        return gdf

    def get_buildings(self, tag_filters=None):
        # Default tags to keep
        tags_to_keep = self.conf.tag_filters.buildings

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        self.buildings = get_building_data(self._way_records,
                                      tags_to_keep,
                                      tag_filters)

        geometries = create_polygon_geometries(self._nodes,
                                               self.buildings)

        # Convert to GeoDataFrame
        gdf = create_gdf(self.buildings, geometries)
        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

        # Do not keep node information
        # (they are in a list, and causes issues saving the files)
        if "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)

        return gdf

    def get_pois(self):
        raise NotImplementedError()
