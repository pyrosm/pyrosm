from pyrosm.config import Conf
from pyrosm.pbfreader import parse_osm_data
from pyrosm._arrays import concatenate_dicts_of_arrays
from pyrosm.geometry import create_node_coordinates_lookup
from pyrosm.frames import create_nodes_gdf
from shapely.geometry import Polygon, MultiPolygon

from pyrosm.buildings import get_building_data
from pyrosm.landuse import get_landuse_data
from pyrosm.natural import get_natural_data
from pyrosm.networks import get_network_data
from pyrosm.pois import get_poi_data


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
        self.keep_node_info = False

        # TODO: Add logging as a parameter?
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
        Parses street networks from OSM
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

        # Filter network data with given filter
        gdf = get_network_data(self._node_coordinates,
                               self._way_records,
                               tags_as_columns,
                               network_filter
                               )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)

        return gdf

    def get_buildings(self, custom_filter=None):
        """
        Parses buildings from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of buildings to parse, see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific buildings such as 'residential' and 'retail', you can apply
            a custom filter which is a Python dictionary with following format:
              `custom_filter={'building': ['residential', 'retail']}`

        Further info
        ------------

        See OSM documentation for details about the data:
        https://wiki.openstreetmap.org/wiki/Key:building

        """
        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.building

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        gdf = get_building_data(self._node_coordinates,
                                self._way_records,
                                self._relations,
                                tags_as_columns,
                                custom_filter)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_landuse(self, custom_filter=None):
        """
        Parses landuse from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of landuse to parse, see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific landuse such as 'construction' and 'industrial', you can apply
            a custom filter which is a Python dictionary with following format:
              `custom_filter={'landuse': ['construction', 'industrial']}`

        Further info
        ------------

        See OSM documentation for details about the data:
        https://wiki.openstreetmap.org/wiki/Key:landuse
        """

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.landuse

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_landuse_data(self._nodes,
                               self._node_coordinates,
                               self._way_records,
                               self._relations,
                               tags_as_columns,
                               custom_filter)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf


    def get_natural(self, custom_filter=None):
        """
        Parses natural from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of natural to parse, see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific natural such as 'wood' and 'tree', you can apply
            a custom filter which is a Python dictionary with following format:
              `custom_filter={'natural': ['wood', 'tree']}`

        Further info
        ------------

        See OSM documentation for details about the data:
        https://wiki.openstreetmap.org/wiki/Key:natural
        """

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.natural

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_natural_data(self._nodes,
                               self._node_coordinates,
                               self._way_records,
                               self._relations,
                               tags_as_columns,
                               custom_filter)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info:
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


        Filtering by tags
        -----------------

        By default, Pyrosm will parse all OSM elements (points, lines and polygons)
        that are associated with following keys:
          - amenity
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

        Filtering by OSM Element type
        -----------------------------

        A specific column called `osm_type` is added to the the resulting GeoDataFrame that informs
        about the OSM Element type. Possible values are: 'node', 'way' and 'relation'.
        These values can be used to filter out specific type of elements
        from the results. If you for example want to keep only POIs that are Points, you can select
        them with a simple Pandas query:
        >>> my_poi_gdf = my_poi_gdf.loc[my_poi_gdf['osm_type']=='node']

        Further info
        ------------

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
        # If custom_filter has not been defined, initialize with default
        if custom_filter is None:
            custom_filter = {"amenity": True,
                             "shop": True,
                             "tourism": True
                             }

        else:
            # Check that the custom filter is in correct format
            if not isinstance(custom_filter, dict):
                raise ValueError(f"'custom_filter' should be a Python dictionary. "
                                 f"Got {custom_filter} with type {type(custom_filter)}.")

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = []
        for k in custom_filter.keys():
            tags_as_columns += getattr(self.conf.tags, k)

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_poi_data(self._nodes,
                           self._node_coordinates,
                           self._way_records,
                           self._relations,
                           tags_as_columns,
                           custom_filter)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def __getattribute__(self, name):
        # If node-gdf is requested convert to gdf before returning
        if name == "_nodes_gdf":
            return create_nodes_gdf(super(OSM, self).__getattribute__("_nodes"))
        return super(OSM, self).__getattribute__(name)
