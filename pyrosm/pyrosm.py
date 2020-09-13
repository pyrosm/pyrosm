from pyrosm.config import Conf
from pyrosm.pbfreader import parse_osm_data
from pyrosm._arrays import concatenate_dicts_of_arrays
from pyrosm.geometry import create_node_coordinates_lookup
from pyrosm.frames import create_nodes_gdf
from pyrosm.utils import validate_custom_filter, validate_osm_keys, \
    validate_tags_as_columns, validate_booleans, validate_boundary_type, \
    validate_bounding_box, validate_input_file, get_bounding_box
from pyrosm.utils.download import get_file_size
from shapely.geometry import Polygon, MultiPolygon, \
    LineString, LinearRing, MultiLineString
from pyrosm.boundary import get_boundary_data
from pyrosm.buildings import get_building_data
from pyrosm.landuse import get_landuse_data
from pyrosm.natural import get_natural_data
from pyrosm.networks import get_network_data
from pyrosm.pois import get_poi_data
from pyrosm.user_defined import get_user_defined_data


class OSM:
    from pyrosm.utils._compat import PYGEOS_SHAPELY_COMPAT
    allowed_bbox_types = [Polygon, MultiPolygon, MultiLineString,
                          LineString, LinearRing]

    def __init__(self, filepath,
                 bounding_box=None):
        """
        Parameters
        ----------

        filepath : str
            Filepath to input OSM dataset ( *.osm.pbf )

        bounding_box : list | shapely geometry
            Filtering OSM data spatially is allowed by passing a
            bounding box either as a list `[minx, miny, maxx, maxy]` or
            as a Shapely Polygon/MultiPolygon or closed LineString/LinearRing.

        """

        # Check input file
        self.filepath = validate_input_file(filepath)

        if bounding_box is None:
            self.bounding_box = None
        elif type(bounding_box) in self.allowed_bbox_types:
            # Ensures bounding box is a closed geometry
            # (+ attempts to close MultiLineStrings)
            self.bounding_box = validate_bounding_box(bounding_box)
        elif isinstance(bounding_box, list):
            if not len(bounding_box) == 4:
                raise ValueError("When passing bounding box as a list it should contain 4 coordinates: "
                                 "[minx, miny, maxx, maxy].")
            self.bounding_box = bounding_box
        else:
            raise ValueError("bounding_box should be a list, Shapely Polygon or a Shapely LinearRing.")

        self.conf = Conf
        self.keep_node_info = False

        # Update file size
        self.file_size = get_file_size(self.filepath)

        # Get bounding box
        self._data_bounding_box = get_bounding_box(self.filepath)

        # TODO: Add logging as a parameter?
        self._verbose = False

        self._all_way_tags = None
        self._nodes = None
        self._nodes_gdf = None
        self._node_coordinates = None
        self._way_records = None
        self._relations = None

    def _read_pbf(self):
        # PBF reading requires a list of bounding box coordinates
        if type(self.bounding_box) in self.allowed_bbox_types:
            bounding_box = self.bounding_box.bounds
        else:
            bounding_box = self.bounding_box

        nodes, ways, relations, way_tags = parse_osm_data(self.filepath,
                                             bounding_box,
                                             exclude_relations=False)

        self._nodes = nodes
        self._way_records = ways
        self._relations = relations
        self._all_way_tags = way_tags

        # Prepare node coordinates lookup table
        self._node_coordinates = create_node_coordinates_lookup(self._nodes)

    def _get_network_filter(self, net_type):
        possible_filters = [a for a in self.conf.network_filters.__dir__()
                            if "__" not in a]
        possible_filters += ["all", "driving+service"]
        possible_values = ", ".join(possible_filters)
        msg = "'network_type' should be one of the following: " + possible_values
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
        elif net_type == "driving+service":
            return self.conf.network_filters.driving_psv
        elif net_type == "cycling":
            return self.conf.network_filters.cycling
        elif net_type == "all":
            return None

    def get_network(self, network_type="walking", extra_attributes=None):
        """
        Parses street networks from OSM
        for walking, driving, and cycling.

        Parameters
        ----------

        network_type : str
            What kind of network to parse.
            Possible values are:
              - `'walking'`
              - `'cycling'`
              - `'driving'`
              - `'driving+service'`
              - `'all'`.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        See Also
        --------

        Take a look at the OSM documentation for further details about the data:
        `https://wiki.openstreetmap.org/wiki/Key:highway <https://wiki.openstreetmap.org/wiki/Key:highway>`__

        """
        # Get filter
        network_filter = self._get_network_filter(network_type)
        tags_as_columns = self.conf.tags.highway

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Filter network data with given filter
        gdf = get_network_data(self._node_coordinates,
                               self._way_records,
                               tags_as_columns,
                               network_filter,
                               self.bounding_box
                               )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)

        return gdf

    def get_buildings(self, custom_filter=None, extra_attributes=None):
        """
        Parses buildings from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of buildings to parse,
            see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific buildings such as 'residential' and 'retail', you can apply
            a custom filter which is a Python dictionary with following format:
              - `custom_filter={'building': ['residential', 'retail']}`

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        See Also
        --------

        Take a look at the OSM documentation for further details about the data:
        `https://wiki.openstreetmap.org/wiki/Key:building <https://wiki.openstreetmap.org/wiki/Key:building>`__

        """
        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.building

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        gdf = get_building_data(self._node_coordinates,
                                self._way_records,
                                self._relations,
                                tags_as_columns,
                                custom_filter,
                                self.bounding_box
                                )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_landuse(self, custom_filter=None, extra_attributes=None):
        """
        Parses landuse from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of landuse to parse,
            see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific landuse such as 'construction' and 'industrial', you can apply
            a custom filter which is a Python dictionary with following format:
              `custom_filter={'landuse': ['construction', 'industrial']}`

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:landuse <https://wiki.openstreetmap.org/wiki/Key:landuse>`__

        """

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.landuse

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_landuse_data(self._nodes,
                               self._node_coordinates,
                               self._way_records,
                               self._relations,
                               tags_as_columns,
                               custom_filter,
                               self.bounding_box
                               )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_natural(self, custom_filter=None, extra_attributes=None):
        """
        Parses natural from OSM.

        Parameters
        ----------

        custom_filter : dict
            What kind of natural to parse,
            see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific natural such as 'wood' and 'tree', you can apply
            a custom filter which is a Python dictionary with following format:
              `custom_filter={'natural': ['wood', 'tree']}`

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:natural <https://wiki.openstreetmap.org/wiki/Key:natural>`__

        """

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.natural

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_natural_data(self._nodes,
                               self._node_coordinates,
                               self._way_records,
                               self._relations,
                               tags_as_columns,
                               custom_filter,
                               self.bounding_box)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_boundaries(self, boundary_type="administrative", name=None, custom_filter=None, extra_attributes=None):
        """
        Parses boundaries from OSM.

        Parameters
        ----------

        boundary_type : str
            The type of boundaries to parse. Possible values:
              - `"administrative"` (default)
              - `"national_park"`
              - `"political"`
              - `"postal_code"`
              - `"protected_area"`
              - `"aboriginal_lands"`
              - `"maritime"`
              - `"lot"`
              - `"parcel"`
              - `"tract"`
              - `"marker"`
              - `"all"`

        name : str (optional)
            Name of the administrative area that will be searched for.

        custom_filter : dict (optional)
            Additional filter for what kind of boundary to parse.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.


        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:boundary <https://wiki.openstreetmap.org/wiki/Key:bondary>`__

        """

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # Default tags to keep as columns
        tags_as_columns = self.conf.tags.boundary

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        # Check boundary type
        boundary_type = validate_boundary_type(boundary_type)

        if name is not None:
            if not isinstance(name, str):
                raise ValueError(f"'name' should be text."
                                 f"Got '{name}' of type {type(name)}.")

        gdf = get_boundary_data(self._node_coordinates,
                                self._way_records,
                                self._relations,
                                tags_as_columns,
                                custom_filter,
                                boundary_type,
                                name,
                                self.bounding_box)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_pois(self, custom_filter=None, extra_attributes=None):
        """
        Parse Point of Interest (POI) from OSM.

        Parameters
        ----------

        custom_filter : dict
            An optional custom filter to filter only specific POIs from OpenStreetMap,
            see details below.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.


        Notes
        -----

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


        See Also
        --------

        You can check the most typical OSM tags for different map features from OSM Wiki
        `https://wiki.openstreetmap.org/wiki/Map_Features <https://wiki.openstreetmap.org/wiki/Map_Features>`__.
        It is also possible to get a quick look at the most typical OSM tags from Pyrosm configuration:

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

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_poi_data(self._nodes,
                           self._node_coordinates,
                           self._way_records,
                           self._relations,
                           tags_as_columns,
                           custom_filter,
                           self.bounding_box)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_data_by_custom_criteria(self,
                                    custom_filter,
                                    osm_keys_to_keep=None,
                                    filter_type="keep",
                                    tags_as_columns=None,
                                    keep_nodes=True,
                                    keep_ways=True,
                                    keep_relations=True,
                                    extra_attributes=None):
        """
        Parse OSM data based on custom criteria.

        Parameters
        ----------

        custom_filter : dict (required)
            A custom filter to filter only specific POIs from OpenStreetMap.

        osm_keys_to_keep : str | list
            A filter to specify which OSM keys should be kept.

        filter_type : str
            "keep" | "exclude"
            Whether the filters should be used to keep or exclude the data from OSM.

        tags_as_columns : list
            Which tags should be kept as columns in the resulting GeoDataFrame.

        keep_nodes : bool
            Whether or not the nodes should be kept in the resulting GeoDataFrame if they are found.

        keep_ways : bool
            Whether or not the ways should be kept in the resulting GeoDataFrame if they are found.

        keep_relations : bool
            Whether or not the relations should be kept in the resulting GeoDataFrame if they are found.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        """

        # Check that the custom filter is in correct format
        custom_filter = validate_custom_filter(custom_filter)

        if not isinstance(filter_type, str):
            raise ValueError("'filter_type' -parameter should be either 'keep' or 'exclude'. ")

        # Validate osm keys
        validate_osm_keys(osm_keys_to_keep)
        if isinstance(osm_keys_to_keep, str):
            osm_keys_to_keep = [osm_keys_to_keep]

        # Validate filter
        filter_type = filter_type.lower()
        if filter_type not in ["keep", "exclude"]:
            raise ValueError("'filter_type' -parameter should be either 'keep' or 'exclude'. ")

        # Tags to keep as columns
        if tags_as_columns is None:
            tags_as_columns = []
            for k in custom_filter.keys():
                try:
                    tags_as_columns += getattr(self.conf.tags, k)
                except Exception as e:
                    pass
            # If tags weren't available in conf, store keys as columns by default
            # (all other tags in such cases will be stored in 'tags' column as JSON)
            if len(tags_as_columns) == 0:
                tags_as_columns = list(custom_filter.keys())

        else:
            # Validate tags
            validate_tags_as_columns(tags_as_columns)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # Validate booleans
        validate_booleans(keep_nodes, keep_ways, keep_relations)

        if self._nodes is None or self._way_records is None:
            self._read_pbf()

        # If nodes are still in chunks, merge before passing forward
        if isinstance(self._nodes, list):
            self._nodes = concatenate_dicts_of_arrays(self._nodes)

        gdf = get_user_defined_data(self._nodes,
                                    self._node_coordinates,
                                    self._way_records,
                                    self._relations,
                                    tags_as_columns,
                                    custom_filter,
                                    osm_keys_to_keep,
                                    filter_type,
                                    keep_nodes,
                                    keep_ways,
                                    keep_relations,
                                    self.bounding_box
                                    )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def __getattribute__(self, name):
        # If node-gdf is requested convert to gdf before returning
        if name == "_nodes_gdf":
            return create_nodes_gdf(super(OSM, self).__getattribute__("_nodes"))
        return super(OSM, self).__getattribute__(name)
