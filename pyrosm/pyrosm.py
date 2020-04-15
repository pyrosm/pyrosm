from pyrosm.config import Conf
from pyrosm.pbfreader import parse_osm_data
from pyrosm.networks import get_way_data
from pyrosm.buildings import get_building_data
from pyrosm.pois import get_poi_data
from pyrosm.geometry import create_way_geometries, \
    create_polygon_geometries, create_node_coordinates_lookup
from pyrosm.frames import create_gdf, create_nodes_gdf
from pyrosm.relations import prepare_relations
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import warnings


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
        self._nodes_gdf = None
        self._node_coordinates = None
        self._way_records = None
        self._relations = None

    def _read_pbf(self):
        nodes, ways, relations, way_tags = parse_osm_data(self.filepath,
                                                          self.bounding_box,
                                                          exclude_relations=False)
        self._nodes, self._way_records, \
        self._relations, self._all_way_tags = nodes, ways, relations, way_tags

        # Prepare node coordinates lookup table
        self._node_coordinates = create_node_coordinates_lookup(self._nodes)

    def _get_network_filter(self, net_type):
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
        network_filter = self._get_network_filter(network_type)
        tags_as_columns = self.conf.tags.highway

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Filter ways
        ways = get_way_data(self._way_records,
                            tags_as_columns,
                            network_filter
                            )

        # If there weren't any data, return empty GeoDataFrame
        if ways is None:
            warnings.warn("Could not find any network data for given area.",
                          UserWarning,
                          stacklevel=2)
            return gpd.GeoDataFrame()

        geometries = create_way_geometries(self._node_coordinates,
                                           ways)

        # Convert to GeoDataFrame
        gdf = create_gdf(ways, geometries)
        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

        # Do not keep node information
        # (they are in a list, and causes issues saving the files)
        if "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)

        return gdf

    def get_buildings(self, custom_filter=None):
        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.building

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        ways, relation_ways, relations = get_building_data(self._way_records,
                                                           self._relations,
                                                           tags_as_columns,
                                                           custom_filter)

        # If there weren't any data, return empty GeoDataFrame
        if ways is None:
            warnings.warn("Could not find any buildings for given area.",
                          UserWarning,
                          stacklevel=2)
            return gpd.GeoDataFrame()

        # Create geometries for normal ways
        geometries = create_polygon_geometries(self._node_coordinates,
                                               ways)

        # Convert to GeoDataFrame
        way_gdf = create_gdf(ways, geometries)
        way_gdf["osm_type"] = "way"

        # Prepare relation data if it is available
        if relations is not None:
            relations = prepare_relations(relations, relation_ways,
                                          self._node_coordinates,
                                          tags_as_columns)
            relation_gdf = gpd.GeoDataFrame(relations)
            relation_gdf["osm_type"] = "relation"

            gdf = way_gdf.append(relation_gdf, ignore_index=True)
        else:
            gdf = way_gdf

        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

        # Do not keep node information
        # (they are in a list, and causes issues saving the files)
        if "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_pois(self, custom_filter=None):
        """
        Parse Point of Interest (POI) from OSM.

        Parameters
        ----------

        custom_filter : dict
            An optional custom filter to filter only specific POIs from OpenStreetMap
            (see details below).


        How to use?
        -----------

        By default, will parse all OSM elements (points, lines and polygons)
        that are associated with following keys:
          - amenity
          - craft
          - historic
          - leisure
          - shop
          - tourism

        You can opt-out / opt-in specific elements by using 'custom_filter'.
        To parse elements associated with only specific tags, such as amenities,
        you can specify:
          `custom_filter={"amenity": True}`

        You can also combine multiple filters at the same time.
        For instance, you can parse all 'amenity' elements AND specific 'shop' elements,
        such as supermarkets and book stores by specifying:
          `custom_filter={"amenity": True, "shop": ["supermarket", "books"]}`

        You can check the most typical OSM tags for different map features from OSM Wiki
        https://wiki.openstreetmap.org/wiki/Map_Features . It is also possible to get a quick
        look at the most typical OSM tags from Pyrosm configuration:

        >>> from pyrosm.config import Conf
        >>> print("All available OSM keys", Conf.tags.available)
        All available OSM keys ['aerialway', 'aeroway', 'amenity', 'building', 'craft',
        'emergency', 'geological', 'highway', 'historic', 'landuse', 'leisure',
        'natural', 'office', 'power', 'public_transport', 'railway', 'route',
        'place', 'shop', 'tourism', 'waterway']

        >>> print("Typical tags associated with tourism:", Conf.tags.tourism)
        ['alpine_hut', 'apartment', 'aquarium', 'artwork', 'attraction', 'camp_pitch',
        'camp_site', 'caravan_site', 'chalet', 'gallery', 'guest_house', 'hostel',
        'hotel', 'information', 'motel', 'museum', 'picnic_site', 'theme_park',
        'tourism', 'viewpoint', 'wilderness_hut', 'zoo']

        """

        # Default tags to keep as columns
        tags_as_columns = []
        for k in custom_filter.keys():
            tags_as_columns += getattr(self.conf.tags, k)


        ways, relation_ways, relations = get_poi_data(self._way_records,
                                                      self._relations,
                                                      list(custom_filter.keys()),
                                                      tags_as_columns,
                                                      custom_filter)

        # If there weren't any data, return empty GeoDataFrame
        if ways is None:
            warnings.warn("Could not find any POIs for given area.",
                          UserWarning,
                          stacklevel=2)
            return gpd.GeoDataFrame()

        # Create geometries for normal ways
        geometries = create_polygon_geometries(self._node_coordinates,
                                               ways)

        # Convert to GeoDataFrame
        way_gdf = create_gdf(ways, geometries)

        # Prepare relation data if it is available
        if relations is not None:
            relations = prepare_relations(relations, relation_ways,
                                          self._node_coordinates,
                                          tags_as_columns)
            relation_gdf = gpd.GeoDataFrame(relations)
            gdf = way_gdf.append(relation_gdf, ignore_index=True)
        else:
            gdf = way_gdf

        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

        # Do not keep node information
        # (they are in a list, and causes issues saving the files)
        if "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)
        return gdf


    def get_landuse(self, custom_filter=None):
        raise NotImplementedError()

    def __getattribute__(self, name):
        # If node-gdf is requested convert to gdf before returning
        if name == "_nodes_gdf":
            return create_nodes_gdf(super(OSM, self).__getattribute__("_nodes"))
        return super(OSM, self).__getattribute__(name)
