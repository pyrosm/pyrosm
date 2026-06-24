import warnings

import pandas as pd
from pyrosm.config import Conf
from pyrosm.pbfreader import parse_osm_data
from pyrosm.frames import create_nodes_gdf
from pyrosm.utils import (
    validate_custom_filter,
    validate_osm_keys,
    validate_tags_as_columns,
    validate_booleans,
    validate_boundary_type,
    validate_bounding_box,
    validate_input_file,
    validate_graph_type,
    validate_engine,
    validate_workers,
    get_bounding_box,
    get_unix_time,
    warn_about_timestamp_not_set,
    warn_about_single_core,
)
from pyrosm.utils.download import get_file_size
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    LinearRing,
    MultiLineString,
)
from pyrosm.boundary import get_boundary_data
from pyrosm.buildings import get_building_data
from pyrosm.landuse import get_landuse_data
from pyrosm.natural import get_natural_data
from pyrosm.networks import get_network_data
from pyrosm.pois import get_poi_data
from pyrosm.user_defined import get_user_defined_data
from pyrosm.graphs import to_networkx, to_igraph, to_pandana, to_pandarm

# Aliased so the module does not collide with the ``engine`` constructor
# parameter / ``self.engine`` attribute of the same name.
from pyrosm import engine as engine_backend


class OSM:
    """
    OpenStreetMap PBF reader object.

    Parameters
    ----------

    filepath : str
        Filepath to input OSM dataset ( *.osm.pbf )

    bounding_box : list | shapely geometry
        Filtering OSM data spatially is allowed by passing a
        bounding box either as a list `[minx, miny, maxx, maxy]` or
        as a Shapely Polygon/MultiPolygon or closed LineString/LinearRing.

    keep_metadata : bool (default: True)
        Whether to keep the OSM element metadata columns (`timestamp`,
        `version`, `changeset`) in the returned GeoDataFrames. Set to `False`
        to drop them and reduce memory use when the metadata is not needed;
        the per-node metadata is then also skipped while parsing, which lowers
        peak memory on node-heavy files. History (`.osh.pbf`) parsing keeps the
        metadata it requires regardless of this flag.

    complete_relations : bool (default: False)
        When reading with a `bounding_box`, a relation (e.g. a multipolygon or
        boundary) is normally assembled from only the member ways that fall
        inside the box, so a relation straddling the edge of the box comes out
        with a partial geometry. Set this to `True` to fetch each such relation's
        full member set (member ways and their nodes, even outside the box) so
        the geometry is complete. This adds two extra streaming passes over the
        file (only when a relation actually has missing members), so it is
        opt-in. It has no effect on a whole-file read (no `bounding_box`), which
        already holds every member. Only member ways are completed; relations
        whose members are themselves relations (super-relations) are not.

    engine : str (default: 'in_memory')
        Which reader backend to use. `'in_memory'` (the default) parses the whole
        file into memory. `'out_of_core'` decodes the file in a single streaming
        pass with bounded peak memory, spilling intermediate data to disk.

        The out-of-core backend reads on a single core by default. To decode in
        parallel pass `workers="auto"` (or an explicit `workers=N`); see below. The
        worker processes start with `spawn` on macOS and Windows, which re-imports
        the program's entry point, so a parallel `OSM(...)` read must run under an
        `if __name__ == "__main__":` guard::

            if __name__ == "__main__":
                osm = OSM(fp, engine="out_of_core", workers="auto")
                buildings = osm.get_buildings()

        Without the guard a parallel read still completes -- it falls back to a
        single process and warns -- but it is not parallel. On Linux (`fork`) no
        guard is needed, and the default single-core read needs no guard anywhere.

        History reads -- an `.osh.pbf` file, or any feature call with a
        `timestamp` -- are served by the in-memory reader even when
        `engine='out_of_core'`: selecting the latest version of each element
        at/before the timestamp uses pyrosm's `get_latest_version`, which pandas
        evaluates eagerly over the whole multi-version frame, so history is read
        in memory.

    workers : int | str (default: None)
        Number of worker processes the `'out_of_core'` engine uses to decode the
        file. By default (`None`) the engine reads on a single core, and the first
        out-of-core read reports how many CPU cores are available and how to opt
        into parallelism. Pass `workers="auto"` to let pyrosm choose the count
        automatically -- a single core for small files and one worker per CPU core
        for larger files -- or `workers=N` for an explicit count (a count above the
        available CPU cores is reduced to the core count, with a warning). Parallel
        reads need the `if __name__ == "__main__":` guard on macOS/Windows; pass
        `workers=1` to read on a single core silently. Has no effect on the
        `'in_memory'` engine.
    """

    allowed_bbox_types = [
        Polygon,
        MultiPolygon,
        MultiLineString,
        LineString,
        LinearRing,
    ]

    def __init__(
        self,
        filepath,
        bounding_box=None,
        keep_metadata=True,
        complete_relations=False,
        engine="in_memory",
        workers=None,
    ):
        # Check input file
        self.filepath = validate_input_file(filepath)

        if not isinstance(keep_metadata, bool):
            raise ValueError("'keep_metadata' should be a boolean.")
        self.keep_metadata = keep_metadata

        if not isinstance(complete_relations, bool):
            raise ValueError("'complete_relations' should be a boolean.")
        self.complete_relations = complete_relations

        self.engine = validate_engine(engine)
        self.workers = validate_workers(workers)
        self._single_core_notice_emitted = False

        # Check if file contains history
        self._osh_file = False
        if "osh.pbf" in self.filepath.lower():
            self._osh_file = True

        if bounding_box is None:
            self.bounding_box = None
        elif type(bounding_box) in self.allowed_bbox_types:
            # Ensures bounding box is a closed geometry
            # (+ attempts to close MultiLineStrings)
            self.bounding_box = validate_bounding_box(bounding_box)
        elif isinstance(bounding_box, list):
            if not len(bounding_box) == 4:
                raise ValueError(
                    "When passing bounding box as a list it should contain 4 coordinates: "
                    "[minx, miny, maxx, maxy]."
                )
            minx, miny, maxx, maxy = bounding_box
            if minx >= maxx or miny >= maxy:
                raise ValueError(
                    "Invalid bounding box {bbox}: expected [minx, miny, maxx, maxy] with "
                    "minx < maxx and miny < maxy. Please double-check the order of the "
                    "coordinates (they may be swapped/inverted).".format(
                        bbox=bounding_box
                    )
                )
            self.bounding_box = bounding_box
        else:
            raise ValueError(
                "bounding_box should be a list, Shapely Polygon or a Shapely LinearRing."
            )

        self.conf = Conf
        self.keep_node_info = False

        # Update file size
        self.file_size = get_file_size(self.filepath)

        # Get bounding box
        self._data_bounding_box = get_bounding_box(self.filepath)

        # TODO: Add logging as a parameter?
        self._verbose = False
        self._nodes = None
        self._nodes_gdf = None
        self._node_coordinates = None
        self._way_records = None
        self._relations = None
        self._relation_member_ways = None

        # Timestamp
        self._current_timestamp = None
        self._timestamp_changed = False

    def _use_engine(self, timestamp):
        """Whether the out-of-core engine handles this read. History reads -- an ``.osh.pbf``
        file, or an explicit ``timestamp`` -- route to the in-memory reader instead: selecting
        the latest version of each element at/before the timestamp is pyrosm's per-id
        ``get_latest_version`` merge (``df.groupby("id").last()``, each column's last non-null
        value across an element's versions), which pandas evaluates eagerly over the whole
        materialised multi-version frame, so history is read in memory."""
        return self.engine == "out_of_core" and timestamp is None and not self._osh_file

    def _read_engine(self, reader, with_relations=True, **kwargs):
        """Route a feature read to the given out-of-core engine reader, threading the
        constructor-level ``bounding_box`` / ``keep_metadata`` / ``workers`` (and
        ``complete_relations`` for the layer readers). Only non-history reads reach here;
        history reads use the in-memory path (see :meth:`_use_engine`)."""
        workers = self.workers
        if workers is None:
            workers = 1
            if not self._single_core_notice_emitted:
                self._single_core_notice_emitted = True
                warn_about_single_core()
        kwargs["bounding_box"] = self.bounding_box
        kwargs["keep_metadata"] = self.keep_metadata
        kwargs["workers"] = workers
        if with_relations:
            kwargs["complete_relations"] = self.complete_relations
        return reader(self.filepath, **kwargs)

    def _get_pbf_elements(self, bounding_box):
        (
            nodes,
            ways,
            relations,
            node_coordinates,
            relation_member_ways,
        ) = parse_osm_data(
            self.filepath,
            bounding_box,
            exclude_relations=False,
            unix_time_filter=self._current_timestamp,
            keep_metadata=self.keep_metadata,
            complete_relations=self.complete_relations,
        )

        self._nodes = nodes
        self._way_records = ways
        self._relations = relations
        self._node_coordinates = node_coordinates
        self._relation_member_ways = relation_member_ways

    def _read_pbf(self, timestamp=None):
        # PBF reading requires a list of bounding box coordinates
        if type(self.bounding_box) in self.allowed_bbox_types:
            bounding_box = self.bounding_box.bounds
        else:
            bounding_box = self.bounding_box

        # Update current timestamp
        self._set_current_time(timestamp)

        if self._nodes is None or self._way_records is None:
            self._get_pbf_elements(bounding_box)
        elif self._timestamp_changed:
            self._get_pbf_elements(bounding_box)

        # Once the data has been read update the flag
        self._timestamp_changed = False

    def _get_network_filter(self, net_type):
        possible_filters = self.conf._possible_network_filters
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

    def _set_current_time(self, timestamp):
        unix_time = None
        if timestamp is not None:
            unix_time = get_unix_time(timestamp, self._osh_file)

        if unix_time is not None:
            # Keeps track if timestamp changed
            if self._current_timestamp != unix_time:
                self._current_timestamp = unix_time
                self._timestamp_changed = True

        # In case OSH file is used but no timestamp is provided, use current UTC time
        elif self._osh_file:
            unix_time = get_unix_time(pd.Timestamp.now(tz="UTC"), True)
            self._set_current_time(unix_time)
            warn_about_timestamp_not_set(unix_time)

    def get_network(
        self,
        network_type="walking",
        extra_attributes=None,
        nodes=False,
        timestamp=None,
        custom_filter=None,
        filter_type="exclude",
        tags_to_keep=None,
    ):
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

            When `custom_filter` is given, `network_type` no longer selects the
            ways to keep; it only determines the graph semantics used by
            `OSM.to_graph` (directionality / connectivity), e.g. pass
            `network_type='driving'` for a directed custom network or keep the
            default `'walking'` for a bidirectional one.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        nodes : bool (default: False)
            If True, 1) the nodes associated with the network will be returned in addition to edges,
            and 2) every segment of a road constituting a way is parsed as a separate row
            (to enable full connectivity in the graph). Works together with
            `custom_filter` so that custom-filtered networks can also be exported
            to a graph.

        custom_filter : dict (optional)
            A custom filter for selecting which highway ways to keep, e.g.
            `{'highway': ['footway', 'residential'], 'bicycle': ['yes']}`. When
            given, it replaces the predefined `network_type` filter (the two are
            not combined). Only ways having a `highway` tag are considered; the
            filter then keeps or excludes among those according to `filter_type`.
            The filter keys are also added as columns to the result.

        filter_type : str (default: 'exclude')
            Whether `custom_filter` should `'keep'` or `'exclude'` the matching
            ways. Only consulted when `custom_filter` is given; the predefined
            `network_type` filters are always applied as `'exclude'`.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        Returns
        -------

        gdf_edges or (gdf_nodes, gdf_edges)

        Return type
        -----------

        geopandas.GeoDataFrame or tuple

        See Also
        --------

        Take a look at the OSM documentation for further details about the data:
        `https://wiki.openstreetmap.org/wiki/Key:highway <https://wiki.openstreetmap.org/wiki/Key:highway>`__
        """
        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_network,
                with_relations=False,
                network_type=network_type,
                extra_attributes=extra_attributes,
                nodes=nodes,
                custom_filter=custom_filter,
                filter_type=filter_type,
                tags_to_keep=tags_to_keep,
            )

        # Get filter (also validates network_type, which still drives the graph
        # semantics even when a custom_filter is provided)
        network_filter = self._get_network_filter(network_type)
        tags_as_columns = list(self.conf.tags.highway)

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        # A custom_filter replaces the predefined network filter and may use any
        # 'keep'/'exclude' semantics; the predefined filters are always 'exclude'.
        if custom_filter is not None:
            custom_filter = validate_custom_filter(custom_filter)

            if not isinstance(filter_type, str) or filter_type.lower() not in [
                "keep",
                "exclude",
            ]:
                raise ValueError(
                    "'filter_type' -parameter should be either 'keep' or 'exclude'. "
                )
            filter_type = filter_type.lower()

            network_filter = custom_filter
            # Expose the filter keys as columns too (e.g. 'bicycle', 'service').
            for key in custom_filter.keys():
                if key not in tags_as_columns:
                    tags_as_columns.append(key)
        else:
            # Predefined networks are always exclude filters.
            filter_type = "exclude"

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # Read pbf
        self._read_pbf(timestamp)

        # Filter network data with given filter
        edges, node_gdf = get_network_data(
            self._node_coordinates,
            self._way_records,
            tags_as_columns,
            network_filter,
            self.bounding_box,
            slice_to_segments=nodes,
            filter_type=filter_type,
            keep_metadata=self.keep_metadata,
        )

        if edges is not None:
            # Add metadata
            edges._metadata.append(network_type)

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and edges is not None:
            if "nodes" in edges.columns:
                edges = edges.drop("nodes", axis=1)

        # In case both edges and nodes are requested
        if nodes:
            return (node_gdf, edges)
        return edges

    def get_buildings(
        self,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
    ):
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

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".


        See Also
        --------

        Take a look at the OSM documentation for further details about the data:
        `https://wiki.openstreetmap.org/wiki/Key:building <https://wiki.openstreetmap.org/wiki/Key:building>`__

        """
        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_buildings,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )

        # Default tags to keep as columns
        tags_as_columns = list(self.conf.tags.building)

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        self._read_pbf(timestamp)

        gdf = get_building_data(
            self._node_coordinates,
            self._way_records,
            self._relations,
            tags_as_columns,
            custom_filter,
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_landuse(
        self,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
    ):
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

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:landuse <https://wiki.openstreetmap.org/wiki/Key:landuse>`__

        """

        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_landuse,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )

        self._read_pbf(timestamp)

        # Default tags to keep as columns
        tags_as_columns = list(self.conf.tags.landuse)

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        gdf = get_landuse_data(
            self._nodes,
            self._node_coordinates,
            self._way_records,
            self._relations,
            tags_as_columns,
            custom_filter,
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_natural(
        self,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
    ):
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

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:natural <https://wiki.openstreetmap.org/wiki/Key:natural>`__

        """

        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_natural,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )

        self._read_pbf(timestamp)

        # Default tags to keep as columns
        tags_as_columns = list(self.conf.tags.natural)

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        gdf = get_natural_data(
            self._nodes,
            self._node_coordinates,
            self._way_records,
            self._relations,
            tags_as_columns,
            custom_filter,
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_boundaries(
        self,
        boundary_type="administrative",
        name=None,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
    ):
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

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        See Also
        --------

        Take a look at OSM documentation for further details about the data:

        `https://wiki.openstreetmap.org/wiki/Key:boundary <https://wiki.openstreetmap.org/wiki/Key:boundary>`__

        """

        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_boundaries,
                boundary_type=boundary_type,
                name=name,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )

        self._read_pbf(timestamp)

        # Default tags to keep as columns
        tags_as_columns = list(self.conf.tags.boundary)

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        # Check boundary type
        boundary_type = validate_boundary_type(boundary_type)

        if name is not None:
            if not isinstance(name, str):
                raise ValueError(
                    f"'name' should be text." f"Got '{name}' of type {type(name)}."
                )

        gdf = get_boundary_data(
            self._node_coordinates,
            self._way_records,
            self._relations,
            tags_as_columns,
            custom_filter,
            boundary_type,
            name,
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_pois(
        self,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
    ):
        """
        Parse Point of Interest (POI) from OSM.

        Parameters
        ----------

        custom_filter : dict
            An optional custom filter to filter only specific POIs from OpenStreetMap,
            see details below.

        extra_attributes : list (optional)
            Additional OSM tag keys that will be converted into columns in the resulting GeoDataFrame.

        tags_to_keep : list (optional)
            When given, only these OSM tag keys are kept as columns, replacing the
            default set of tag columns (reduces memory). Structural columns and
            filtering are unaffected; `extra_attributes` still apply.

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

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
        if self._use_engine(timestamp):
            return self._read_engine(
                engine_backend.get_pois,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )

        # If custom_filter has not been defined, initialize with default
        if custom_filter is None:
            custom_filter = {"amenity": True, "shop": True, "tourism": True}

        else:
            # Check that the custom filter is in correct format
            if not isinstance(custom_filter, dict):
                raise ValueError(
                    f"'custom_filter' should be a Python dictionary. "
                    f"Got {custom_filter} with type {type(custom_filter)}."
                )

        self._read_pbf(timestamp)

        # Default tags to keep as columns
        tags_as_columns = []
        for k in custom_filter.keys():
            try:
                tags_as_columns += getattr(self.conf.tags, k)
            except AttributeError:
                tags_as_columns += self.conf.tags._basic_tags
            except Exception as e:
                raise e

        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep)
            tags_as_columns = list(tags_to_keep)

        if extra_attributes is not None:
            validate_tags_as_columns(extra_attributes)
            tags_as_columns += extra_attributes

        gdf = get_poi_data(
            self._nodes,
            self._node_coordinates,
            self._way_records,
            self._relations,
            tags_as_columns,
            custom_filter,
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def get_data_by_custom_criteria(
        self,
        custom_filter=None,
        osm_keys_to_keep=None,
        filter_type="keep",
        tags_as_columns=None,
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True,
        extra_attributes=None,
        keep_other_tags=True,
        timestamp=None,
    ):
        """
        `
        Parse OSM data based on custom criteria.

        Parameters
        ----------

        custom_filter : dict (optional)
            A custom filter to filter only specific elements from OpenStreetMap.
            If ``None`` (the default), every tagged element is returned without
            key/value filtering (tagged nodes, ways, and relations); standalone
            ways with no tags are dropped. Reading everything is memory-heavy on
            large extracts, so prefer a pre-filtered PBF and/or a bounding box.
            ``filter_type`` is ignored in this mode.

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

        keep_other_tags : bool
            By default (``True``) every tag is parsed: ``tags_as_columns`` become their own
            columns and the rest are kept in a JSON ``tags`` column. ``False`` resolves only
            the requested tags (``tags_as_columns`` plus the filter keys) and drops the JSON
            ``tags`` column, so the read does minimal tag work (a stray tag literally keyed
            ``id`` is not surfaced as ``id_tag`` in this mode). Only supported by the
            out-of-core engine (``OSM(..., engine='out_of_core')``).

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        """

        if self._use_engine(timestamp):
            if custom_filter is None:
                raise NotImplementedError(
                    "get_data_by_custom_criteria(custom_filter=None) ('return everything') "
                    "is not yet supported by the out-of-core engine; pass an explicit "
                    "custom_filter."
                )
            return self._read_engine(
                engine_backend.get_data_by_custom_criteria,
                custom_filter=custom_filter,
                osm_keys_to_keep=osm_keys_to_keep,
                filter_type=filter_type,
                tags_as_columns=tags_as_columns,
                keep_nodes=keep_nodes,
                keep_ways=keep_ways,
                keep_relations=keep_relations,
                extra_attributes=extra_attributes,
                keep_other_tags=keep_other_tags,
            )

        # keep_other_tags=False only skips tag work in the out-of-core decode; the in-memory
        # reader resolves every tag once and caches it for reuse across layers, so it cannot
        # honour the minimal-tags mode.
        if keep_other_tags is False:
            raise ValueError(
                "keep_other_tags=False is only supported by the out-of-core engine; "
                "construct OSM(..., engine='out_of_core')."
            )

        # custom_filter=None means "return everything": keep every tagged element
        # with no key/value filtering (issue #113).
        keep_all = custom_filter is None
        if keep_all:
            custom_filter = {}
            # There is nothing to keep/exclude by value, so filter_type is ignored;
            # pin it to a valid no-op so a caller-supplied value cannot reject the
            # request or reach Solver() with an invalid direction.
            filter_type = "keep"

        # Check that the custom filter is in correct format
        custom_filter = validate_custom_filter(custom_filter)

        if not isinstance(filter_type, str):
            raise ValueError(
                "'filter_type' -parameter should be either 'keep' or 'exclude'. "
            )

        # Validate osm keys
        validate_osm_keys(osm_keys_to_keep)
        if isinstance(osm_keys_to_keep, str):
            osm_keys_to_keep = [osm_keys_to_keep]

        # Validate filter
        filter_type = filter_type.lower()
        if filter_type not in ["keep", "exclude"]:
            raise ValueError(
                "'filter_type' -parameter should be either 'keep' or 'exclude'. "
            )

        # Tags to keep as columns
        if tags_as_columns is None:
            if keep_all:
                # No key filter to derive columns from: expose the default columns
                # of every known feature, so common tags become columns and the
                # rest land in the JSON 'tags' column.
                tags_as_columns = []
                for k in self.conf.tags.available:
                    tags_as_columns += getattr(self.conf.tags, k)
                tags_as_columns = list(dict.fromkeys(tags_as_columns))
            else:
                tags_as_columns = []
                for k in custom_filter.keys():
                    try:
                        tags_as_columns += getattr(self.conf.tags, k)
                    except Exception:
                        pass
                # If tags weren't available in conf, store keys as columns by
                # default (all other tags will be stored in 'tags' column as JSON)
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

        self._read_pbf(timestamp)

        gdf = get_user_defined_data(
            self._nodes,
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
            self.bounding_box,
            keep_metadata=self.keep_metadata,
            relation_member_ways=self._relation_member_ways,
            complete_relations=self.complete_relations,
            keep_all=keep_all,
        )

        # Do not keep node information unless specifically asked for
        # (they are in a list, and can cause issues when saving the files)
        if not self.keep_node_info and gdf is not None:
            if "nodes" in gdf.columns:
                gdf = gdf.drop("nodes", axis=1)
        return gdf

    def to_pbf(
        self,
        output_path=None,
        keep_relations=True,
        workers=1,
        compact=False,
        repack=False,
    ):
        """
        Crop the source PBF by this object's ``bounding_box`` and write a valid,
        re-readable ``*.osm.pbf`` to disk.

        Cropping is "complete ways": a way is kept when at least one of its nodes
        falls inside the bounding box, and the kept way retains its full node list
        so geometries are not cut at the box edge. The crop streams the source
        file blob-by-blob and never loads it fully into memory.

        A ``Polygon``/``MultiPolygon`` ``bounding_box`` is cropped by its envelope
        (bounding rectangle), matching how ``OSM()`` itself filters by a polygon
        bounding box, so a cropped-and-reread file matches reading the source with
        the same ``bounding_box``.

        Parameters
        ----------

        output_path : str, optional
            Where to write the cropped PBF. When ``None`` (default) a temporary
            file is created in the system temp directory and its path returned.

        keep_relations : bool
            When ``True`` (default) relations referencing a kept node or way are
            written; when ``False`` no relations are written.

        workers : int
            Number of worker processes for the CPU-heavy per-block work. ``1``
            (default) runs sequentially; ``>1`` uses a multiprocessing pool and
            produces a byte-identical output.

        compact : bool
            When ``False`` (default) each output block keeps its source block's
            full string table, which is the fastest crop but leaves strings used
            only by dropped elements in the file. When ``True`` each output block's
            string table is pruned to only the strings its kept elements reference,
            producing a smaller file at the cost of some extra per-block work. The
            written OSM data is identical either way.

        repack : bool
            When ``True`` the kept elements are re-chunked into canonical, densely
            packed blocks (as ``osmium``/Osmosis produce), giving the smallest output
            at the cost of speed; the re-pack write is sequential, though ``workers``
            still parallelizes the selection. Re-packed blocks already have minimal
            string tables, so ``compact`` is ignored when ``repack=True``. The written
            OSM data is identical to ``repack=False``. Default ``False`` keeps the
            current (faster, slightly larger) in-place crop.

        Returns
        -------
        str
            The path of the written PBF file.
        """
        from pyrosm.pbf_export import crop_pbf

        if self.bounding_box is None:
            raise ValueError(
                "Cropping a PBF requires a bounding box. Construct the OSM "
                "object with `OSM(filepath, bounding_box=...)` before calling "
                "`to_pbf()`."
            )
        return crop_pbf(
            self.filepath,
            output_path,
            self.bounding_box,
            keep_relations=keep_relations,
            workers=workers,
            compact=compact,
            repack=repack,
        )

    def write_pbf(self, data, output_path):
        """
        Write the OSM data this object holds back to a valid, re-readable
        ``*.osm.pbf``, applying attribute/tag edits from a (modified)
        GeoDataFrame.

        The whole cached dataset (all nodes/ways/relations read from the source)
        is written. Each row of ``data`` updates the tags of the matching element
        (by ``osm_type`` + ``id``); rows whose ``id`` is not in the source are
        added as new elements synthesized from their geometry (with negative ids).
        Topology and coordinates come from the data pyrosm read, so the output is
        faithful and re-readable (e.g. by pyrosm, osmium, GDAL and r5py/R5).

        Typical use is to modify attributes in pandas and save them back::

            osm = OSM("data.osm.pbf")
            edges = osm.get_network("driving")
            edges["maxspeed"] = edges["maxspeed"].fillna(50)
            edges["travel_time"] = edges["length"] / (edges["maxspeed"] / 3.6)
            osm.write_pbf(edges, "modified.osm.pbf")

        Parameters
        ----------

        data : GeoDataFrame or list of GeoDataFrame
            The (possibly modified) feature frame(s) whose tag columns are written
            onto the matching elements. New rows (ids not in the source) are added
            from their geometry: ``Point`` -> node, ``LineString`` -> way, hole-less
            ``Polygon`` -> closed way. Polygons with holes, MultiPolygon and
            MultiLineString geometries are not supported and raise ``ValueError``.

        output_path : str
            Where to write the PBF.

        Returns
        -------
        str
            The path of the written PBF file.

        Notes
        -----
        v1 applies edits and additions, not deletions: dropping rows from ``data``
        does not remove elements (the whole cached dataset is the base set).
        """
        from pyrosm.pbf_writer import write_geodataframe_to_pbf

        if self._way_records is None or self._node_coordinates is None:
            self._read_pbf()

        return write_geodataframe_to_pbf(
            data,
            output_path,
            node_coordinates=self._node_coordinates,
            way_records=self._way_records,
            relations=self._relations,
            nodes=self._nodes,
        )

    @staticmethod
    def to_graph(
        nodes,
        edges,
        graph_type="igraph",
        direction="oneway",
        from_id_col="u",
        to_id_col="v",
        edge_id_col="id",
        node_id_col="id",
        force_bidirectional=False,
        network_type=None,
        retain_all=False,
        osmnx_compatible=True,
        pandana_weights=["length"],
    ):
        """
        `
        Export OSM network to routable graph. Supported output graph types are:
          - "igraph" (default),
          - "networkx",
          - "pandarm",
          - "pandana" (deprecated; use "pandarm")

        For walking, the output graph will be bidirectional by default
        (i.e. travel along the street is allowed to both directions). For driving
        and cycling, one-way streets are taken into account by default and the
        travel is restricted based on the rules in OSM data (the "oneway"
        attribute; cycling additionally honours "oneway:bicycle" so that
        contraflow cycling on one-way streets is modelled correctly).

        Parameters
        ----------

        nodes : GeoDataFrame
            GeoDataFrame containing nodes of the road network.
            Note: Use `osm.get_network(nodes=True)` to retrieve both the nodes and edges.

        edges : GeoDataFrame
            GeoDataFrame containing the edges of the road network.

        graph_type : str
            Type of the output graph. Available graphs are:
              - "igraph" --> returns an igraph.Graph -object.
              - "networkx" --> returns a networkx.MultiDiGraph -object.
              - "pandarm" --> returns a pandarm.Network -object.
              - "pandana" --> returns a pandana.Network -object.
                (deprecated: pandana is unmaintained and incompatible with
                NumPy 2 on Windows; use "pandarm" instead. Will be removed in
                a future release.)

        direction : str
            Name for the column containing information about the allowed driving directions

        from_id_col : str
            Name for the column having the from-node-ids of edges.

        to_id_col : str
            Name for the column having the to-node-ids of edges.

        edge_id_col : str
            Name for the column having the unique id for edges.

        node_id_col : str
            Name for the column having the unique id for nodes.

        force_bidirectional : bool
            If True, all edges will be created as bidirectional (allow travel to both directions).

        network_type : str (optional)
            Network type for the given data. Determines how the graph will be constructed.
            The network type is typically extracted automatically from the metadata of
            the edges/nodes GeoDataFrames. This parameter can be used if this metadata is not
            available for a reason or another. By default, a bidirectional graph is created for walking and all,
            and a directed graph for driving and cycling (oneway streets are taken into account;
            cycling additionally honours oneway:bicycle for contraflow).
            Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'.

        retain_all : bool
            if True, return the entire graph even if it is not connected.
            otherwise, retain only the connected edges.

        osmnx_compatible : bool (default True)
            if True, modifies the edge and node-attribute naming to be compatible with OSMnx
            (allows utilizing all OSMnx functionalities).
            NOTE: Only applicable with "networkx" graph type.

        pandana_weights : list
            Columns that are used as weights when exporting to Pandana graph. By default uses "length" column.
        """
        graph_type = validate_graph_type(graph_type)

        if graph_type == "igraph":
            return to_igraph(
                nodes,
                edges,
                direction,
                from_id_col,
                to_id_col,
                node_id_col,
                force_bidirectional,
                network_type,
                retain_all,
            )
        elif graph_type == "networkx":
            return to_networkx(
                nodes,
                edges,
                direction,
                from_id_col,
                to_id_col,
                edge_id_col,
                node_id_col,
                force_bidirectional,
                network_type,
                retain_all,
                osmnx_compatible,
            )
        elif graph_type == "pandarm":
            return to_pandarm(
                nodes,
                edges,
                direction,
                from_id_col,
                to_id_col,
                node_id_col,
                force_bidirectional,
                network_type,
                retain_all,
                pandana_weights,
            )
        elif graph_type == "pandana":
            warnings.warn(
                "graph_type='pandana' is deprecated because pandana is "
                "unmaintained and incompatible with NumPy 2 on Windows; use "
                "graph_type='pandarm' instead. The 'pandana' backend will be "
                "removed in a future pyrosm release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return to_pandana(
                nodes,
                edges,
                direction,
                from_id_col,
                to_id_col,
                node_id_col,
                force_bidirectional,
                network_type,
                retain_all,
                pandana_weights,
            )

    @staticmethod
    def list_cache(filepath=None):
        """List the out-of-core engine's cached layer files -- the GeoParquet files that
        ``engine="out_of_core"`` reads write under ``<tempdir>/pyrosm/cache``.

        Parameters
        ----------

        filepath : str | os.PathLike (optional)
            When given, only the cached layers for that source PBF are listed; when ``None``
            (default) every cached file is listed.

        Returns
        -------
        list of str
            The cached files' paths.
        """
        from pyrosm.engine import cache

        return cache.list_files(filepath)

    @staticmethod
    def list_downloads():
        """List the PBF files downloaded by ``get_data`` in the default download directory
        (``<tempdir>/pyrosm``).

        Returns
        -------
        list of str
            The downloaded files' paths.
        """
        from pyrosm.utils.download import list_downloads

        return list_downloads()

    @staticmethod
    def clear_cache(filepath=None):
        """Remove the out-of-core engine's result cache -- the GeoParquet files that
        ``engine="out_of_core"`` reads write under ``<tempdir>/pyrosm/cache``.

        Parameters
        ----------

        filepath : str | os.PathLike (optional)
            When given, only the cached layers for that source PBF are removed; when ``None``
            (default) the whole cache is cleared.

        Returns
        -------
        int
            The number of cache files removed.
        """
        from pyrosm.engine import cache

        return cache.clear(filepath)

    @staticmethod
    def clear_downloads(filepath=None):
        """Remove PBF files downloaded by ``get_data`` from the default download directory
        (``<tempdir>/pyrosm``). The result cache and the bundled package datasets are left
        untouched.

        Parameters
        ----------

        filepath : str | os.PathLike (optional)
            A downloaded file's path or bare filename to remove just that file; when ``None``
            (default) every downloaded ``*.pbf`` in the directory is removed.

        Returns
        -------
        int
            The number of files removed.
        """
        from pyrosm.utils.download import clear_downloads

        return clear_downloads(filepath)

    def __getattribute__(self, name):
        # If node-gdf is requested convert to gdf before returning
        if name == "_nodes_gdf":
            return create_nodes_gdf(super(OSM, self).__getattribute__("_nodes"))
        return super(OSM, self).__getattribute__(name)
