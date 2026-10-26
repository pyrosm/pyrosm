import contextlib
import logging
from pathlib import Path

import pandas as pd
from pyrosm._log import log_event, logged, timed
from pyrosm.config import Conf
from pyrosm.config.osm_filters import get_osm_filter
from pyrosm.filter_compiler import CompiledFilter
from pyrosm.pbfreader import parse_osm_data
from pyrosm.frames import create_nodes_gdf
from pyrosm.utils import (
    custom_read_columns,
    feature_columns,
    keys_as_filter,
    read_tag_columns,
    validate_custom_filter,
    validate_osm_keys,
    validate_tags_as_columns,
    validate_bool,
    validate_booleans,
    validate_boundary_type,
    validate_bounding_box,
    validate_input_file,
    validate_graph_type,
    validate_engine,
    validate_workers,
    records_read,
    get_bounding_box,
    get_unix_time,
    warn_about_timestamp_not_set,
    warn_empty_result,
)
from pyrosm.utils.download import get_file_size
from pyrosm.utils.progress import (
    BUILDING_EDGES,
    BUILDING_RELATIONS,
    BUILDING_WAYS,
    COLLECTING,
    COMPLETING_RELATIONS,
    COMPLETING_WAYS,
    DECODING,
    SELECTING_VERSIONS,
    read_steps,
    stepping,
    validate_progress,
)
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    LinearRing,
    MultiLineString,
)
from pyrosm.boundary import get_boundary_data
from pyrosm.layers import get_layer_data, without_building_parts
from pyrosm.networks import get_network_data
from pyrosm.pois import get_poi_data
from pyrosm.user_defined import get_user_defined_data
from pyrosm.graphs import (
    graph_tags,
    record_network_type,
    step_shares,
    to_networkx,
    to_igraph,
    to_pandana,
    to_pandarm,
)

# Aliased so the module does not collide with the ``engine`` constructor
# parameter / ``self.engine`` attribute of the same name.
from pyrosm import engine as engine_backend
from pyrosm.engine.pool import GRAPH_SHARES, NETWORK_NODES_SHARES
from pyrosm.engine.readers import (
    _boundary_filter,
    _network_filter_type,
    _poi_filter,
    _require_layer_key,
    graph_network,
)

logger = logging.getLogger(__name__)

# The shares of the steps of an in-memory read, relative to decoding the file, from the
# times of reads of Paris (242 MB) and of the Helsinki history sample.
_PARSE_SHARES = {
    DECODING: 1.0,
    SELECTING_VERSIONS: 0.1,
    COMPLETING_RELATIONS: 1.0,
    COMPLETING_WAYS: 1.0,
}
_LAYER_SHARES = {COLLECTING: 1.0, BUILDING_WAYS: 1.4, BUILDING_RELATIONS: 0.8}
_NETWORK_SHARES = {COLLECTING: 0.35, BUILDING_EDGES: 0.12}
_NETWORK_NODES_SHARES = {COLLECTING: 0.35, BUILDING_EDGES: 0.3}
# The seconds of the reads of Paris that the shares of an out-of-core graph read and of
# an out-of-core network read with nodes stand for, and of decoding Paris in memory (the
# unit of the in-memory shares): get_graph weighs its read against the graph steps,
# whose shares are in seconds too.
_GRAPH_READ_SECONDS = 2.25
_NETWORK_NODES_READ_SECONDS = 4.2
_DECODING_SECONDS = 3.5


def _network_read_filter(args):
    """The filter a ``get_network`` call selects its ways by (see
    :func:`pyrosm.utils.records_read`): its ``custom_filter`` when that keeps, otherwise
    the ways with a ``highway`` tag."""
    custom_filter, filter_type = _network_filter_type(
        args["custom_filter"], args["filter_type"]
    )
    return custom_filter if filter_type == "keep" else {"highway": [True]}


def _custom_read_filter(args):
    """The filter a ``get_data_by_custom_criteria`` call selects its elements by (see
    :func:`pyrosm.utils.records_read`): its ``custom_filter`` with ``filter_type="keep"``,
    or the keys of ``osm_keys_to_keep`` alone (see :func:`pyrosm.utils.keys_as_filter`);
    ``None`` for an exclude filter or a read of every element."""
    custom_filter, filter_type = keys_as_filter(
        args["custom_filter"], args["osm_keys_to_keep"], args["filter_type"]
    )
    if custom_filter is None or filter_type.lower() != "keep":
        return None
    return validate_custom_filter(custom_filter)


class OSM:
    """
    OpenStreetMap PBF reader object.

    Parameters
    ----------

    filepath : str
        Filepath to input OSM dataset (``*.osm.pbf``)

    bounding_box : list | tuple | numpy.ndarray | shapely geometry | GeoDataFrame | GeoSeries
        Filtering OSM data spatially is allowed by passing a bounding box in lon/lat:
        four numbers `[minx, miny, maxx, maxy]` (a list, tuple or array, such as a
        GeoDataFrame's `total_bounds`), a Shapely Polygon/MultiPolygon or closed
        LineString/LinearRing, or a GeoDataFrame/GeoSeries (its geometries together,
        reprojected to lon/lat). Every function that takes a bounding box takes these.

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

    engine : str (default: 'out_of_core')
        Which reader backend to use. `'out_of_core'` (the default) decodes the file
        in a single streaming pass with bounded peak memory, spilling intermediate
        data to disk, and -- when the optional `pyarrow` is installed -- caches each
        read's result in the temporary folder, so an identical read returns it
        without decoding the file again (`OSM.clear_cache()` removes the cached
        results). `'in_memory'` parses the
        whole file into memory once and serves every read of the `OSM` object from
        it.

        The out-of-core backend chooses by the file's size how many threads decode
        it (`workers="auto"`, the default; see below). A parallel read runs on
        threads and needs no `if __name__ == "__main__":` guard. With the
        environment variable `PYROSM_DECODE_BACKEND=processes` (deprecated: it warns,
        and it will be removed in a future release) it runs on one core by default
        and in worker processes for `workers="auto"` or `workers=N`; those start with
        `spawn` on macOS and Windows, which re-imports the program's entry point, so
        a parallel read then must run under the guard::

            if __name__ == "__main__":
                osm = OSM(fp, engine="out_of_core", workers="auto")
                buildings = osm.get_buildings()

        Without the guard such a read still completes -- it falls back to a single
        process and warns -- but it is not parallel.

        History reads -- an `.osh.pbf` file, or any feature call with a
        `timestamp` -- choose the latest version of each element at/before the
        timestamp from all its versions, with either engine; the out-of-core engine
        caches them as its other reads, by that time. `write_pbf` reads the file in
        memory with either engine.

    keep_node_info : bool (default: False)
        Whether to keep the `nodes` column -- the ordered member node ids of each
        way -- in the returned GeoDataFrames. It is dropped by default because a
        list-valued column cannot be written to most file formats. Set to `True`
        to reshape ways with `write_pbf(..., apply_geometry=True)`, which takes a
        way's topology from this column.

    workers : int | str (default: None)
        Number of threads the `'out_of_core'` engine uses to decode the file. By
        default (`None`, as `"auto"`) pyrosm chooses the count: a single core for
        files below about 20 MB; for larger files the CPU cores this process may use
        less two, at most 4 below about 70 MB and at most one per data blob. Pass
        `workers=N` for an explicit count, e.g. `workers=1` for one core (a count
        above the cores this process may use is reduced to that count, with a
        warning). With `PYROSM_DECODE_BACKEND=processes` they are worker processes,
        one by default and at most eight with `"auto"`, and a parallel read needs the
        `if __name__ == "__main__":` guard on macOS/Windows. Has no effect on the
        `'in_memory'` engine.

    progress : bool | callable | None (default: None)
        `None` follows `pyrosm.enable_logging(progress=...)`, which is `True`
        unless set. `True` shows a bar on stderr while the file is read, once the read
        has run 2 seconds, from 0 to 100 % over the whole read with its current step in
        the label (e.g. "Reading Paris.osm.pbf: building the geometries"), and clears it
        when the read finishes; in Jupyter it is a widget when ipywidgets is installed
        (`pip install "pyrosm[notebook]"`). Without a terminal, a read that took 2
        seconds or more prints one line with the time it took instead. `False` shows
        nothing. A callable is called as `progress(done, 100)` with the percentage of
        the read done, from 0 to 100 and never back.
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
        keep_node_info=False,
        engine="out_of_core",
        workers=None,
        progress=None,
    ):
        # Check input file
        self.filepath = validate_input_file(filepath)

        if not isinstance(keep_metadata, bool):
            raise ValueError("'keep_metadata' should be a boolean.")
        self.keep_metadata = keep_metadata

        if not isinstance(complete_relations, bool):
            raise ValueError("'complete_relations' should be a boolean.")
        self.complete_relations = complete_relations

        if not isinstance(keep_node_info, bool):
            raise ValueError("'keep_node_info' should be a boolean.")

        self.engine = validate_engine(engine)
        self.workers = validate_workers(workers)
        validate_progress(progress)
        self.progress = progress

        # Check if file contains history
        self._osh_file = False
        if "osh.pbf" in self.filepath.lower():
            self._osh_file = True

        self.bounding_box = (
            None if bounding_box is None else validate_bounding_box(bounding_box)
        )

        self.conf = Conf
        self.keep_node_info = keep_node_info

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
        self._way_changesets = None

        # Timestamp
        self._current_timestamp = None
        self._timestamp_changed = False
        # The current time a history read without a timestamp uses, chosen once
        self._implicit_now = None

        bbox = self.bounding_box
        if isinstance(bbox, list):
            bbox = ",".join(str(v) for v in bbox)
        elif bbox is not None:
            bbox = bbox.geom_type
        log_event(
            logger,
            logging.INFO,
            "OSM",
            file=Path(self.filepath).name,
            engine=self.engine,
            workers=self.workers,
            bounding_box=bbox,
            keep_metadata=self.keep_metadata,
            complete_relations=self.complete_relations,
            progress="callable" if callable(progress) else progress,
        )

    def _use_engine(self):
        """Whether the out-of-core engine reads the file, history reads included (see
        :meth:`_read_engine`), rather than the in-memory reader."""
        return self.engine == "out_of_core"

    def _history_time(self, timestamp):
        """The UNIX time (seconds) at which a read with ``timestamp`` chooses each element's
        version, as :meth:`_set_current_time` sets it, or None for a file without history.
        """
        self._set_current_time(timestamp)
        return self._current_timestamp

    def _read_engine(self, reader, timestamp, with_relations=True, **kwargs):
        """Route a feature read to the given out-of-core engine reader, threading the
        constructor-level ``bounding_box`` / ``keep_metadata`` / ``keep_node_info`` /
        ``workers`` (and ``complete_relations`` for the layer readers), and the time of a
        history read with ``timestamp`` as ``unix_time`` (see :meth:`_history_time`)."""
        kwargs["unix_time"] = self._history_time(timestamp)
        kwargs["bounding_box"] = self.bounding_box
        kwargs["keep_metadata"] = self.keep_metadata
        kwargs["workers"] = self.workers
        kwargs["progress"] = self.progress
        kwargs["keep_node_info"] = self.keep_node_info
        if with_relations:
            kwargs["complete_relations"] = self.complete_relations
        return reader(self.filepath, **kwargs)

    def _get_pbf_elements(self, bounding_box, steps):
        """Parse the file within ``bounding_box`` in memory, in the steps of ``steps``
        (see :func:`~pyrosm.pbfreader.parse_osm_data`), and keep its elements."""
        name = Path(self.filepath).name
        size = Path(self.filepath).stat().st_size
        with timed(logger, "read_pbf", file=name, bytes=size):
            (
                nodes,
                ways,
                relations,
                node_coordinates,
                relation_member_ways,
                way_changesets,
            ) = parse_osm_data(
                self.filepath,
                bounding_box,
                exclude_relations=False,
                unix_time_filter=self._current_timestamp,
                keep_metadata=self.keep_metadata,
                complete_relations=self.complete_relations,
                steps=steps,
            )

        self._nodes = nodes
        self._way_records = ways
        self._relations = relations
        self._node_coordinates = node_coordinates
        self._relation_member_ways = relation_member_ways
        self._way_changesets = way_changesets

    def _read_pbf(self, timestamp=None, steps=None):
        """Parse the file for ``timestamp`` unless it is parsed for it already, in the
        steps of ``steps``, or without them as a read of its own (see
        :meth:`_read_steps`)."""
        # PBF reading requires a list of bounding box coordinates
        if type(self.bounding_box) in self.allowed_bbox_types:
            bounding_box = self.bounding_box.bounds
        else:
            bounding_box = self.bounding_box

        # Update current timestamp
        self._set_current_time(timestamp)

        if self._parse_needed():
            if steps is None:
                with self._read_steps(timestamp, {}) as steps:
                    self._get_pbf_elements(bounding_box, steps)
            else:
                self._get_pbf_elements(bounding_box, steps)

        # Once the data has been read update the flag
        self._timestamp_changed = False

    def _parse_needed(self):
        """Whether the file is not parsed yet, or not for the current timestamp."""
        return (
            self._nodes is None or self._way_records is None or self._timestamp_changed
        )

    def _read_steps(self, timestamp, shares):
        """The :func:`~pyrosm.utils.progress.read_steps` of an in-memory read for
        ``timestamp`` whose own steps have the ``shares``, after the steps of parsing
        the file when it is not parsed yet: decoding it, selecting the versions of a
        history read, and the passes that complete the elements the bounding box cuts.
        """
        self._set_current_time(timestamp)
        planned = self._parse_shares() if self._parse_needed() else {}
        planned.update(shares)
        return read_steps(self.filepath, self.progress, planned)

    def _parse_shares(self):
        """The shares of the steps of parsing the file for the current timestamp:
        decoding it, selecting the versions of a history read, and the passes that
        complete the elements the bounding box cuts."""
        names = [DECODING]
        if self._current_timestamp is not None:
            names.append(SELECTING_VERSIONS)
        if self.bounding_box is not None:
            if self.complete_relations:
                names.append(COMPLETING_RELATIONS)
            names.append(COMPLETING_WAYS)
        return {name: _PARSE_SHARES[name] for name in names}

    def _drop_node_lists(self, gdf):
        """``gdf`` without its ``nodes`` column (each row's list of node ids, which file
        formats cannot store) unless ``keep_node_info`` is set."""
        if not self.keep_node_info and gdf is not None and "nodes" in gdf.columns:
            gdf = gdf.drop("nodes", axis=1)
        return gdf

    def _read_layer(
        self,
        layer,
        result_name,
        engine_reader,
        custom_filter,
        extra_attributes,
        tags_to_keep,
        timestamp,
        with_nodes=True,
    ):
        """The elements of the layer ``layer`` (an OSM key, such as ``"building"``) for
        :meth:`get_buildings`, :meth:`get_landuse` and :meth:`get_natural`: read by
        ``engine_reader`` with the out-of-core engine, else in memory, in both without
        nodes unless ``with_nodes``, with a warning naming ``result_name`` when nothing
        matches."""
        if self._use_engine():
            gdf = self._read_engine(
                engine_reader,
                timestamp,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
                keep_nodes=with_nodes,
            )
            if gdf is None:
                warn_empty_result(result_name)
            return gdf

        read_filter = _require_layer_key(custom_filter, layer)
        tags_as_columns = read_tag_columns(
            getattr(self.conf.tags, layer),
            extra_attributes,
            tags_to_keep,
            read_filter.keys(),
        )
        with self._read_steps(timestamp, _LAYER_SHARES) as steps:
            self._read_pbf(timestamp, steps)
            gdf = get_layer_data(
                layer,
                result_name,
                self._nodes if with_nodes else None,
                self._node_coordinates,
                self._way_records,
                self._relations,
                tags_as_columns,
                custom_filter,
                self.bounding_box,
                keep_metadata=self.keep_metadata,
                relation_member_ways=self._relation_member_ways,
                complete_relations=self.complete_relations,
                steps=steps,
            )
        return self._drop_node_lists(gdf)

    def _get_network_filter(self, net_type):
        """Resolve a predefined network type to its filter, raising for an unknown one."""
        return get_osm_filter(net_type)

    def _set_current_time(self, timestamp):
        """Set the time a history read selects each element's version at: ``timestamp``,
        or, for a history file read without one, the current UTC time chosen at the first
        such read (with a warning). A changed time makes :meth:`_read_pbf` parse again.
        """
        unix_time = None
        if timestamp is not None:
            unix_time = get_unix_time(timestamp, self._osh_file)

        if unix_time is not None:
            # Keeps track if timestamp changed
            if self._current_timestamp != unix_time:
                self._current_timestamp = unix_time
                self._timestamp_changed = True

        # In case OSH file is used but no timestamp is provided, use the current UTC
        # time, chosen once for this object so that later reads reuse the parsed data
        elif self._osh_file:
            if self._implicit_now is None:
                self._implicit_now = get_unix_time(pd.Timestamp.now(tz="UTC"), True)
                warn_about_timestamp_not_set(self._implicit_now)
            self._set_current_time(self._implicit_now)

    @logged("network_type", "nodes")
    @records_read(_network_read_filter)
    def get_network(
        self,
        network_type="walking",
        extra_attributes=None,
        nodes=False,
        timestamp=None,
        custom_filter=None,
        filter_type=None,
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

              - `'walking'` (OSMnx `'walk'`)
              - `'cycling'` (OSMnx `'bike'`)
              - `'driving'` (OSMnx `'drive'`)
              - `'driving+service'` (OSMnx `'drive_service'`)
              - `'all'`
              - `'all_public'`.

            The filters follow the way filters of OSMnx closely and in most cases
            select the same ways. Where the two differ, pyrosm follows OSM's
            access rules: a way tagged `access=no` is left out unless a tag of the
            network's mode of transport, such as `foot=yes`, allows the use. The
            OSMnx name of a network type is accepted as well.

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

        custom_filter : dict | str | list (optional)
            A custom filter for selecting which ways to keep. When given, it replaces the
            predefined `network_type` filter (the two are not combined), and the filter keys
            are added as columns to the result. Three forms are supported:

              - a dict like `{'highway': ['footway', 'residential'], 'bicycle': ['yes']}`
                (exact values, combined with OR; only ways having a `highway` tag are
                considered);
              - a dict with compiled regex values, e.g. `{'ref': [re.compile(r'I[ -]?20')]}`;
              - an Overpass-style bracket string, or a list of them, e.g.
                `['["highway"~"cycleway"]', '["highway"~"path"]["bicycle"~"designated"]']`
                (each string is the AND of its brackets, a list is their OR). These select
                ways by the filter's own keys, so non-highway networks (e.g. `railway`,
                `cycleway`) are supported.

        filter_type : str (optional)
            Whether `custom_filter` should `'keep'` or `'exclude'` the matching ways. Only
            consulted when `custom_filter` is given. When omitted it defaults to `'keep'`,
            as in every other reader and for every form of the filter (a plain dict, a
            regex or a bracket filter); the predefined `network_type` filters are always
            applied as `'exclude'`.

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
        # Resolve the custom filter and filter_type once, so the engine and in-memory paths
        # agree.
        custom_filter, filter_type = _network_filter_type(custom_filter, filter_type)
        advanced_filter = isinstance(custom_filter, CompiledFilter)

        if self._use_engine():
            result = self._read_engine(
                engine_backend.get_network,
                timestamp,
                with_relations=False,
                network_type=network_type,
                extra_attributes=extra_attributes,
                nodes=nodes,
                custom_filter=custom_filter,
                filter_type=filter_type,
                tags_to_keep=tags_to_keep,
            )
            if (result[1] if nodes else result) is None:
                warn_empty_result("network")
            return result

        # Get filter (also validates network_type, which still drives the graph
        # semantics even when a custom_filter is provided)
        network_filter = self._get_network_filter(network_type)
        # Predefined networks select ways by 'highway'; an advanced custom filter selects by
        # its own positive keys (so railway/cycleway networks work).
        network_osm_keys = "highway"
        column_keys = ["highway"]

        if custom_filter is not None:
            network_filter = custom_filter
            # The filter keys are columns too (e.g. 'bicycle', 'service').
            column_keys = custom_filter.keys()
            if advanced_filter:
                network_osm_keys = list(custom_filter.positive_keys)

        tags_as_columns = read_tag_columns(
            self.conf.tags.highway, extra_attributes, tags_to_keep, column_keys
        )

        # Read pbf and filter network data with given filter
        shares = _NETWORK_NODES_SHARES if nodes else _NETWORK_SHARES
        with self._read_steps(timestamp, shares) as steps:
            self._read_pbf(timestamp, steps)
            edges, node_gdf = get_network_data(
                self._node_coordinates,
                self._way_records,
                tags_as_columns,
                network_filter,
                self.bounding_box,
                slice_to_segments=nodes,
                filter_type=filter_type,
                keep_metadata=self.keep_metadata,
                osm_keys=network_osm_keys,
                steps=steps,
            )

        edges = self._drop_node_lists(edges)

        # In case both edges and nodes are requested
        if nodes:
            return record_network_type((node_gdf, edges), network_type)
        return record_network_type(edges, network_type)

    @logged()
    @records_read(lambda args: _require_layer_key(args["custom_filter"], "building"))
    def get_buildings(
        self,
        custom_filter=None,
        extra_attributes=None,
        timestamp=None,
        tags_to_keep=None,
        keep_nodes=False,
    ):
        """
        Parses buildings from OSM: the building ways and relations, and with ``keep_nodes``
        the buildings mapped as single nodes.

        Parameters
        ----------

        custom_filter : dict
            What kind of buildings to parse,
            see details below.

            You can opt-in specific elements by using 'custom_filter'.
            To keep only specific buildings such as 'residential' and 'retail', you can apply
            a custom filter which is a Python dictionary with the following format::

                custom_filter={'building': ['residential', 'retail']}

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

        keep_nodes : bool (default: False)
            When True, the buildings mapped as single nodes are returned too, as Points.
            Nodes tagged ``building=entrance`` or ``building=window`` mark a part of a
            building and ``building=no`` says that a feature is not one, so they are left
            out: the result holds only features that represent buildings.
            ``get_data_by_custom_criteria({"building": True})`` returns every node tagged
            ``building``, these included.


        See Also
        --------

        Take a look at the OSM documentation for further details about the data:
        `https://wiki.openstreetmap.org/wiki/Key:building <https://wiki.openstreetmap.org/wiki/Key:building>`__

        """
        validate_bool(keep_nodes, "keep_nodes")
        gdf = self._read_layer(
            "building",
            "buildings",
            engine_backend.get_buildings,
            custom_filter,
            extra_attributes,
            tags_to_keep,
            timestamp,
            with_nodes=keep_nodes,
        )
        return without_building_parts(gdf) if keep_nodes else gdf

    @logged()
    @records_read(lambda args: _require_layer_key(args["custom_filter"], "landuse"))
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
            a custom filter which is a Python dictionary with the following format::

                custom_filter={'landuse': ['construction', 'industrial']}

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
        return self._read_layer(
            "landuse",
            "landuse",
            engine_backend.get_landuse,
            custom_filter,
            extra_attributes,
            tags_to_keep,
            timestamp,
        )

    @logged()
    @records_read(lambda args: _require_layer_key(args["custom_filter"], "natural"))
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
            a custom filter which is a Python dictionary with the following format::

                custom_filter={'natural': ['wood', 'tree']}

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
        return self._read_layer(
            "natural",
            "natural",
            engine_backend.get_natural,
            custom_filter,
            extra_attributes,
            tags_to_keep,
            timestamp,
        )

    @logged()
    @records_read(
        lambda args: _boundary_filter(
            args["custom_filter"], validate_boundary_type(args["boundary_type"])
        )
    )
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
            Name of the administrative area that will be searched for: the boundaries
            whose name contains this text, ignoring case (``"vuori"`` finds
            ``"Punavuori"``), or None with a warning when no name does.

        custom_filter : dict (optional)
            Additional filter for what kind of boundary to parse: it narrows the boundaries
            of ``boundary_type``, e.g. ``{"admin_level": ["10"]}``. A filter on ``boundary``
            itself chooses the boundary values in place of ``boundary_type``.

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

        # Check boundary type and name (with either engine)
        boundary_type = validate_boundary_type(boundary_type)
        if name is not None:
            if not isinstance(name, str):
                raise ValueError(
                    f"'name' should be text." f"Got '{name}' of type {type(name)}."
                )

        if self._use_engine():
            gdf = self._read_engine(
                engine_backend.get_boundaries,
                timestamp,
                boundary_type=boundary_type,
                name=name,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )
            if gdf is None:
                warn_empty_result("boundaries")
            return gdf

        read_filter = _boundary_filter(custom_filter, boundary_type)
        tags_as_columns = read_tag_columns(
            self.conf.tags.boundary, extra_attributes, tags_to_keep, read_filter.keys()
        )

        with self._read_steps(timestamp, _LAYER_SHARES) as steps:
            self._read_pbf(timestamp, steps)
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
                steps=steps,
            )

        return self._drop_node_lists(gdf)

    @logged()
    @records_read(lambda args: _poi_filter(args["custom_filter"]))
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
        you can specify::

            custom_filter={"amenity": True}

        You can also combine multiple filters at the same time.
        For instance, you can parse all 'amenity' elements AND specific 'shop' elements,
        such as supermarkets and book stores by specifying::

            custom_filter={"amenity": True, "shop": ["supermarket", "books"]}

        See Also
        --------

        You can check the most typical OSM tags for different map features from OSM Wiki
        `https://wiki.openstreetmap.org/wiki/Map_Features <https://wiki.openstreetmap.org/wiki/Map_Features>`__.
        It is also possible to get a quick look at the most typical OSM tags from Pyrosm configuration:

        >>> from pyrosm.config import Conf
        >>> print("All available OSM keys", Conf.tags.available)
        All available OSM keys ['aerialway', 'aeroway', 'amenity', 'boundary', 'building',
        'craft', 'emergency', 'geological', 'highway', 'historic', 'landuse', 'leisure',
        'natural', 'office', 'power', 'public_transport', 'railway', 'route', 'place',
        'shop', 'tourism', 'waterway']

        >>> print("Typical tags associated with tourism:", Conf.tags.tourism)
        ['alpine_hut', 'apartment', 'aquarium', 'artwork', 'attraction', 'camp_pitch',
        'camp_site', 'caravan_site', 'chalet', 'gallery', 'guest_house', 'hostel',
        'hotel', 'information', 'motel', 'museum', 'picnic_site', 'theme_park',
        'tourism', 'viewpoint', 'wilderness_hut', 'zoo']

        """
        if self._use_engine():
            gdf = self._read_engine(
                engine_backend.get_pois,
                timestamp,
                custom_filter=custom_filter,
                extra_attributes=extra_attributes,
                tags_to_keep=tags_to_keep,
            )
            if gdf is None:
                warn_empty_result("pois")
            return gdf

        custom_filter = _poi_filter(custom_filter)

        tags_as_columns = read_tag_columns(
            feature_columns(custom_filter.keys(), self.conf.tags._basic_tags),
            extra_attributes,
            tags_to_keep,
            custom_filter.keys(),
        )

        with self._read_steps(timestamp, _LAYER_SHARES) as steps:
            self._read_pbf(timestamp, steps)
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
                steps=steps,
            )

        return self._drop_node_lists(gdf)

    @logged()
    @records_read(_custom_read_filter)
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
        tags_to_keep=None,
    ):
        """
        Parse OSM data based on custom criteria.

        Parameters
        ----------

        custom_filter : dict (optional)
            A custom filter to filter only specific elements from OpenStreetMap.
            If ``None`` (the default), every tagged element is returned without
            key/value filtering (tagged nodes, ways, and relations); standalone
            ways with no tags are dropped. Reading everything is memory-heavy on
            large extracts, so prefer a pre-filtered PBF and/or a bounding box.
            ``filter_type`` is ignored in this mode. With ``osm_keys_to_keep``, the
            elements with one of those keys are returned. A tag keyed ``id`` is filtered
            as ``id_tag``, the column it is returned in; ``id`` is the element's OSM id
            and not a filter key.

        osm_keys_to_keep : str | list
            The keys an element must have one of to be returned: the first level of
            filtering, before ``custom_filter``.

        filter_type : str
            "keep" | "exclude"
            Whether the filters should be used to keep or exclude the data from OSM.

        tags_as_columns : list
            Which tags should be kept as columns in the resulting GeoDataFrame. The keys
            of the filter (and of ``osm_keys_to_keep``) are always columns too, so each
            row shows what it was selected by.

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
            ``tags`` column, so the read does minimal tag work. Only supported by the
            out-of-core engine (``OSM(..., engine='out_of_core')``).

        timestamp: str | datetime | int
            If provided, the data from given moment of time will be returned. The time should be provided in UTC.
            Note: This functionality only works with OSH.PBF files that can be downloaded manually e.g. from Geofabrik
            (requires login with OSM account).

            The logic: the closest version of each element up to given timestamp will be selected to the result.
            This means that elements can be older than the given timestamp (the most up-to-date version is selected),
            but not newer (records having exactly the selected timestamp will be kept). In case only a date is given,
            the time will represent midnight of the given day, such as "2021-01-01 00:00:00".

        tags_to_keep : list (optional)
            The name the other readers give ``tags_as_columns``: the tags kept as columns.
            Give one of the two.

        """

        # custom_filter=None means "return everything": keep every tagged element
        # with no key/value filtering (issue #113). There is nothing to keep or exclude
        # by value then, so filter_type is ignored; it is pinned to a valid no-op so a
        # caller-supplied value cannot reject the request.
        validate_osm_keys(osm_keys_to_keep)
        custom_filter, filter_type = keys_as_filter(
            custom_filter, osm_keys_to_keep, filter_type
        )
        keep_all = custom_filter is None
        if keep_all:
            filter_type = "keep"
        if not isinstance(filter_type, str) or filter_type.lower() not in (
            "keep",
            "exclude",
        ):
            raise ValueError(
                "'filter_type' -parameter should be either 'keep' or 'exclude'. "
            )
        filter_type = filter_type.lower()
        validate_booleans(keep_nodes, keep_ways, keep_relations)

        if self._use_engine():
            gdf = self._read_engine(
                engine_backend.get_data_by_custom_criteria,
                timestamp,
                custom_filter=custom_filter,
                osm_keys_to_keep=osm_keys_to_keep,
                filter_type=filter_type,
                tags_as_columns=tags_as_columns,
                keep_nodes=keep_nodes,
                keep_ways=keep_ways,
                keep_relations=keep_relations,
                extra_attributes=extra_attributes,
                keep_other_tags=keep_other_tags,
                tags_to_keep=tags_to_keep,
            )
            if gdf is None:
                warn_empty_result("custom")
            return gdf

        # keep_other_tags=False only skips tag work in the out-of-core decode; the in-memory
        # reader resolves every tag once and caches it for reuse across layers, so it cannot
        # honour the minimal-tags mode.
        if keep_other_tags is False:
            raise ValueError(
                "keep_other_tags=False is only supported by the out-of-core engine; "
                "construct OSM(..., engine='out_of_core')."
            )

        # Check that the custom filter is in correct format
        custom_filter = validate_custom_filter({} if keep_all else custom_filter)

        # Validate osm keys
        validate_osm_keys(osm_keys_to_keep)
        if isinstance(osm_keys_to_keep, str):
            osm_keys_to_keep = [osm_keys_to_keep]

        tags_as_columns = read_tag_columns(
            custom_read_columns(
                None if keep_all else custom_filter,
                osm_keys_to_keep,
                tags_as_columns,
                tags_to_keep,
            ),
            extra_attributes,
        )

        with self._read_steps(timestamp, _LAYER_SHARES) as steps:
            self._read_pbf(timestamp, steps)
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
                steps=steps,
            )

        return self._drop_node_lists(gdf)

    def to_pbf(
        self,
        output_path=None,
        keep_relations=True,
        workers=None,
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

        A ``Polygon``/``MultiPolygon`` ``bounding_box`` crops to the polygon itself,
        as ``OSM()`` reads by it, so reading the cropped file returns the features
        that reading the source with the same ``bounding_box`` does.

        The crop shows its progress as this object's ``progress`` says: one bar, or
        ``progress(done, 100)`` calls, from 0 to 100 % over the whole crop.

        Parameters
        ----------

        output_path : str, optional
            Where to write the cropped PBF. When ``None`` (default) a temporary
            file is created in the system temp directory and its path returned.

        keep_relations : bool
            When ``True`` (default) relations referencing a kept node or way are
            written; when ``False`` no relations are written. With
            ``complete_relations=True`` (see :class:`OSM`), a relation with a way member
            in the area is written with all its way members and their nodes, so the
            cropped file reads back with its complete geometry.

        workers : int | str
            Number of threads for the CPU-heavy per-block work. By default (``None``)
            the ``workers`` of this ``OSM`` object; when that is not set either, the
            count is chosen as an out-of-core read does for a file of this size (as
            ``"auto"``); ``1`` runs sequentially. Every count produces a byte-identical
            output. With ``PYROSM_DECODE_BACKEND=processes`` they are worker processes,
            one by default; when those cannot run (on macOS and Windows: a script
            without an ``if __name__ == "__main__":`` guard, or one read from stdin), the
            crop runs in one process and warns.

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
            at the cost of speed. ``workers`` threads select the elements, read and
            decode the source blocks of the write and encode the re-packed blocks, which
            are written in order (the whole write runs in one process with
            ``PYROSM_DECODE_BACKEND=processes``). Re-packed blocks already have minimal
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
        region = {"bounding_box": self.bounding_box}
        if isinstance(self.bounding_box, (Polygon, MultiPolygon)):
            region = {"polygon": self.bounding_box}
        return crop_pbf(
            self.filepath,
            output_path,
            keep_relations=keep_relations,
            workers=self.workers if workers is None else workers,
            compact=compact,
            repack=repack,
            progress=self.progress,
            complete_relations=self.complete_relations,
            **region,
        )

    def write_pbf(
        self,
        data,
        output_path,
        subset_only=False,
        *,
        apply_geometry=False,
        delete=None,
        on_orphan_node="remove",
        on_deleted_member="drop",
        progress=None,
    ):
        """
        Write the OSM data this object holds back to a valid, re-readable
        ``*.osm.pbf``, applying attribute/tag edits from a (modified)
        GeoDataFrame.

        The whole cached dataset (all nodes/ways/relations read from the source)
        is written; for a read with a ``bounding_box``, its nodes and ways and the
        relations with a member among them, and with ``complete_relations=True`` also the
        member ways outside the box and their nodes. The tag columns of a row (and its JSON ``tags``) are the tags written
        for the matching element (by ``osm_type`` + ``id``), so a tag whose column is
        left out of the frame is not written; rows whose ``id`` is not in the source are
        added as new elements synthesized from their geometry (with negative ids).
        A frame without tag columns (only ``id``, ``osm_type``, the geometry and other
        structural columns) changes no tags: the elements it matches keep their own.
        The columns ``get_network`` computes (``u``, ``v`` and the numeric ``length``)
        are not tags, and ``id_tag``, the column of a tag keyed ``id``, is written as
        ``id``. An element in several rows, such as a way split into segments by
        ``get_network(nodes=True)`` or an element in two frames, gets each tag from its
        rows: a value changed or removed in only some of them is written for the whole
        element (with a ``UserWarning`` when other rows of the same frame keep the old
        value), and rows giving a tag different new values raise ``ValueError``.
        When elements lose the tag their read selected them by, e.g. buildings written
        without a ``building`` column, a ``UserWarning`` says how many and which reader
        will no longer find them in the written file.
        Topology and coordinates come from the data pyrosm read, so the output is
        faithful and re-readable (e.g. by pyrosm, osmium, GDAL and r5py/R5). The
        elements are written sorted by type then id, new ones included, with the version,
        timestamp and changeset they were read with (none for nodes read with
        ``keep_metadata=False``); the file is written next to ``output_path`` and
        replaces it only when it is complete.

        Pass ``subset_only=True`` to instead write a PBF containing **only** the
        elements in ``data`` (matched by ``osm_type`` + ``id``) plus the references
        they need to stay valid -- e.g. export just the buildings or just the road
        network to a new file. Multiple layers can be combined by passing a list
        (``[buildings, network]``); the union of their elements is written.

        Typical use is to modify attributes in pandas and save them back (tags are
        text, so a number is read from them first)::

            import pandas as pd

            osm = OSM("data.osm.pbf")
            edges = osm.get_network("driving")
            edges["maxspeed"] = edges["maxspeed"].fillna("50")
            speed = pd.to_numeric(edges["maxspeed"], errors="coerce")  # km/h
            edges["travel_time"] = edges["length"] / (speed / 3.6)
            osm.write_pbf(edges, "modified.osm.pbf")

        Pass ``apply_geometry=True`` and/or ``delete=`` to edit the geometry itself.
        Geometry follows the OSM model: coordinates live on nodes, and a way is an
        ordered list of member node ids. So moving a node moves every way that
        references it, and an existing way is reshaped through its ``nodes`` column
        (construct the ``OSM`` object with ``keep_node_info=True`` to get it) rather
        than through its LineString, which is derived::

            osm = OSM("data.osm.pbf", keep_node_info=True)
            nodes, edges = osm.get_network("driving", nodes=True)
            # Move an existing node; the ways through it follow.
            nodes.loc[nodes["id"] == 12345, "geometry"] = Point(24.94, 60.17)
            # Insert a new vertex into a way, referencing a node added in the same call.
            new_node = gpd.GeoDataFrame(
                {"id": [-1], "osm_type": ["node"]},
                geometry=[Point(24.95, 60.18)], crs="EPSG:4326",
            )
            row = edges["id"] == 4732994
            edges.loc[row, "nodes"] = edges.loc[row, "nodes"].apply(
                lambda ids: ids[:2] + [-1] + ids[2:]
            )
            osm.write_pbf(
                [nodes, edges, new_node], "edited.osm.pbf",
                apply_geometry=True, delete=[("way", 4732995)],
            )

        Parameters
        ----------

        data : GeoDataFrame or list of GeoDataFrame
            The (possibly modified) feature frame(s) whose tag columns are written
            onto the matching elements; a frame with no tag columns leaves their tags
            as they are. New rows (ids not in the source) are added
            from their geometry: ``Point`` -> node, ``LineString`` -> way, hole-less
            ``Polygon`` -> closed way. Polygons with holes, MultiPolygon and
            MultiLineString geometries are not supported and raise ``ValueError``.

        output_path : str
            Where to write the PBF.

        subset_only : bool (default False)
            If True, write only the elements present in ``data`` (and the nodes and
            relation members they reference) instead of the whole cached dataset.

        apply_geometry : bool (default False)
            If True, also apply the rows' geometry to the elements that already exist
            in the source, and honour caller-chosen ids for new ones:

            - a node row's ``Point`` moves that node to those coordinates;
            - a way row's ``nodes`` column replaces that way's member node ids, which
              covers inserting, removing and reordering vertices (the way's own
              LineString is ignored -- it is derived from the nodes, and under
              ``get_network(nodes=True)`` it is a single segment of the way);
            - a row with a **negative** id that is not in the source is added under
              exactly that id, instead of a generated one, so a way's ``nodes`` list
              can reference a vertex added in the same call.

            Removing a vertex from a way's ``nodes`` does not delete the node itself;
            use ``delete`` for that. When False (default), rows only edit tags and add
            new elements, exactly as before.

        delete : iterable of (osm_type, id), optional
            Elements to omit from the output, e.g.
            ``[("way", 4732994), ("node", 277446336)]``. An id that is not in the
            source raises ``ValueError``; an element that is both deleted and present
            as a row in ``data`` is deleted. Deleting a relation drops only the
            relation -- its members are independent elements and survive.

        on_orphan_node : {'remove', 'keep', 'error'} (default 'remove')
            What to do with a node that was a member of a dropped way and is left
            referenced by no kept way or relation: ``'remove'`` drops it too (tagged
            nodes included), ``'keep'`` writes it as a bare node, ``'error'`` raises.
            Nodes that no dropped way referenced -- standalone POIs, for instance --
            are never touched.

        on_deleted_member : {'drop', 'error'} (default 'drop')
            What to do when a kept way or relation still references a deleted element:
            ``'drop'`` removes it from that member list, ``'error'`` raises. A way left
            with fewer than two members cannot form a geometry, so it is not written
            either (with a ``UserWarning``) and its own members go through
            ``on_orphan_node``; a relation left with no members is still written.

        progress : bool | callable | None (default: None)
            How the write shows its progress, as ``progress`` of :class:`OSM` does
            (``None`` follows the ``progress`` of this object): one bar "Writing
            <file>" from 0 to 100 % over the whole write, its current step in the
            label, including the read of the file when it is not read in memory yet,
            or ``progress(done, 100)`` calls.

        Returns
        -------
        str
            The path of the written PBF file.

        Notes
        -----
        Elements not present in ``data`` keep their ids, tags, coordinates and member
        order, so the output re-reads identically in pyrosm, osmium/pyosmium, GDAL and
        r5py/R5. Edits never introduce a reference to a node that is not in the output;
        a source way whose members already point outside a clipped extract is re-emitted
        as it was read.
        """
        from pyrosm.pbf_writer import WRITE_SHARES, write_geodataframe_to_pbf

        validate_progress(progress)
        parse = self._way_records is None or self._node_coordinates is None
        shares = {}
        if parse:
            self._set_current_time(None)
            shares = self._parse_shares()
        shares.update(WRITE_SHARES)
        with timed(
            logger,
            "write_pbf",
            logging.INFO,
            output=Path(output_path).name,
            subset_only=subset_only,
        ), stepping(
            self.progress if progress is None else progress,
            "Writing %s" % Path(output_path).name,
            shares,
            delay=2.0,
            leave=False,
            timed=True,
        ) as steps:
            if parse:
                self._read_pbf(steps=steps)
            return write_geodataframe_to_pbf(
                data,
                output_path,
                node_coordinates=self._node_coordinates,
                way_records=self._way_records,
                relations=self._relations,
                nodes=self._nodes,
                subset_only=subset_only,
                apply_geometry=apply_geometry,
                delete=delete,
                on_orphan_node=on_orphan_node,
                on_deleted_member=on_deleted_member,
                steps=steps,
                relation_member_ways=self._relation_member_ways,
                bounded=self.bounding_box is not None,
                way_changesets=self._way_changesets,
            )

    @logged("network_type", "graph_type", "simplify", count=False)
    def get_graph(
        self,
        network_type="walking",
        graph_type="igraph",
        direction="oneway",
        force_bidirectional=False,
        retain_all=False,
        osmnx_compatible=True,
        pandana_weights=["length"],
        simplify=True,
        simplify_kwargs=None,
        extra_attributes=None,
        custom_filter=None,
        filter_type=None,
        tags_to_keep=None,
        timestamp=None,
    ):
        """
        Read the street network of ``network_type`` as a routable graph: the graph of
        ``OSM.to_graph(*osm.get_network(network_type, nodes=True), graph_type)``.

        With the out-of-core engine the simplified graph is built straight from the
        ways and nodes read from the file: the tables of every street segment and of
        every node that ``get_network(nodes=True)`` returns are not built, so the read
        takes less memory. With ``simplify=False`` or the in-memory engine, the tables are
        read with ``get_network`` and exported with ``to_graph``.

        The read and the export show one progress as the ``progress`` of this object
        says: a bar "Building the graph of <file>" from 0 to 100 % with the current step
        in its label, or ``progress(done, 100)`` calls.

        Parameters
        ----------

        network_type, extra_attributes, custom_filter, filter_type, tags_to_keep, timestamp
            As in :meth:`get_network`. With ``tags_to_keep``, the tags the graph is built
            from are read too: those of its edges' directions (``oneway`` and
            ``junction``, and ``oneway:bicycle`` for cycling) and, with ``simplify``, the
            ``edge_attrs_differ`` tags (by default ``maxspeed`` and ``highway``), so the
            graph is the same and only carries fewer attributes.

        graph_type, direction, force_bidirectional, retain_all, osmnx_compatible, pandana_weights, simplify, simplify_kwargs
            As in :meth:`to_graph`.

        Returns
        -------

        igraph.Graph, networkx.MultiDiGraph, pandarm.Network or pandana.Network, or
        None when the network is empty.
        """
        validate_graph_type(graph_type)
        if tags_to_keep is not None:
            validate_tags_as_columns(tags_to_keep, "tags_to_keep")
            tags_to_keep = list(tags_to_keep) + [
                tag
                for tag in graph_tags(
                    direction,
                    network_type,
                    force_bidirectional,
                    simplify,
                    simplify_kwargs,
                )
                if tag not in tags_to_keep
            ]
        shares = self._graph_read_shares(simplify, timestamp)
        shares.update(
            step_shares(simplify, retain_all, validate_graph_type(graph_type))
        )
        with stepping(
            self.progress,
            "Building the graph of %s" % Path(self.filepath).name,
            shares,
            delay=2.0,
            leave=False,
            timed=True,
        ) as steps, self._progress_as(steps):
            if simplify and self._use_engine():
                nodes, edges = self._read_engine(
                    graph_network,
                    timestamp,
                    with_relations=False,
                    network_type=network_type,
                    extra_attributes=extra_attributes,
                    custom_filter=custom_filter,
                    filter_type=filter_type,
                    tags_to_keep=tags_to_keep,
                )
                if edges is None:
                    warn_empty_result("network")
            else:
                nodes, edges = self.get_network(
                    network_type,
                    extra_attributes,
                    nodes=True,
                    timestamp=timestamp,
                    custom_filter=custom_filter,
                    filter_type=filter_type,
                    tags_to_keep=tags_to_keep,
                )
            if edges is None:
                return None
            return self.to_graph(
                nodes,
                edges,
                graph_type,
                direction,
                force_bidirectional=force_bidirectional,
                network_type=network_type,
                retain_all=retain_all,
                osmnx_compatible=osmnx_compatible,
                pandana_weights=pandana_weights,
                simplify=simplify,
                simplify_kwargs=simplify_kwargs,
                progress=steps,
            )

    def _graph_read_shares(self, simplify, timestamp):
        """The shares of the steps of the read of :meth:`get_graph`, in seconds of reads of
        Paris, as the shares of the graph steps are (see
        :func:`pyrosm.graphs.step_shares`)."""
        if self._use_engine():
            if simplify:
                shares, seconds = GRAPH_SHARES, _GRAPH_READ_SECONDS
            else:
                shares, seconds = NETWORK_NODES_SHARES, _NETWORK_NODES_READ_SECONDS
        else:
            self._set_current_time(timestamp)
            shares = self._parse_shares() if self._parse_needed() else {}
            shares.update(_NETWORK_NODES_SHARES)
            seconds = _DECODING_SECONDS
        return {name: share * seconds for name, share in shares.items()}

    @contextlib.contextmanager
    def _progress_as(self, progress):
        """Use ``progress`` as the ``progress`` of this object inside the block."""
        kept = self.progress
        self.progress = progress
        try:
            yield
        finally:
            self.progress = kept

    @staticmethod
    @logged("graph_type", "network_type", "simplify", count=False)
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
        simplify=True,
        simplify_kwargs=None,
        progress=None,
    ):
        """
        Export OSM network to routable graph. Supported output graph types are:

          - "igraph" (default),
          - "networkx",
          - "pandarm",
          - "pandana" (deprecated, removed in pyrosm 1.0; use "pandarm")

        For walking, the output graph will be bidirectional by default
        (i.e. travel along the street is allowed to both directions). For driving
        and cycling, one-way streets are taken into account by default and the
        travel is restricted based on the rules in OSM data (the "oneway"
        attribute; cycling additionally honours "oneway:bicycle" so that
        contraflow cycling on one-way streets is modelled correctly).

        The graph is topologically simplified before export: interstitial degree-2
        nodes are removed and each intersection-to-intersection chain is collapsed into
        a single edge that keeps the full geometry, except where ``maxspeed`` or
        ``highway`` changes along it. This reduces the graph size while preserving
        routing distances and is equivalent to ``osmnx.simplification.simplify_graph``
        with ``edge_attrs_differ=["maxspeed", "highway"]``. Pass ``simplify=False`` to
        keep every segment of every way as its own edge.

        The export shows its progress as ``progress`` says: by default a bar on stderr
        once it has run 2 seconds, from 0 to 100 % with the step it is in (directing the
        edges, keeping the largest connected part, simplifying, building the graph) in
        its label, cleared at the end, or without a terminal one line with the time it
        took.

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
                NumPy 2 on Windows; use "pandarm" instead. Removed in pyrosm
                1.0.)

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
            If not given, the type that ``get_network`` recorded on the frames
            (``edges.attrs["network_type"]``) is used; pass it for frames that do not
            carry one, such as frames read from a file. By default, a bidirectional graph
            is created for walking and all, and a directed graph for driving and cycling
            (oneway streets are taken into account; cycling additionally honours
            oneway:bicycle for contraflow).
            Possible values are: 'walking', 'cycling', 'driving', 'driving+service', 'all'
            and 'all_public', plus the OSMnx names 'walk', 'bike', 'drive' and
            'drive_service'.

        retain_all : bool
            if True, return the entire graph even if it is not connected.
            otherwise, retain only the connected edges.

        osmnx_compatible : bool (default True)
            if True, modifies the edge and node-attribute naming to be compatible with OSMnx
            (allows utilizing all OSMnx functionalities): the node/edge id is renamed to
            ``osmid``, the nodes get ``x``/``y`` from their geometry, in the CRS of the
            frames, in place of ``lon``/``lat``, and node ``tags`` are stored as a JSON
            string. The node ``osmid`` is the key of the node in the graph and is not
            repeated as a node attribute. The edges get OSMnx's ``oneway`` and
            ``reversed`` booleans, and the OSM text of the ``oneway`` tag is
            ``oneway_tag`` (see :func:`pyrosm.graphs.to_networkx`).
            NOTE: Only applicable with "networkx" graph type.

        pandana_weights : list
            Columns that are used as weights when exporting to Pandana graph. By default uses "length" column.

        simplify : bool (default True)
            If True, topologically simplify the graph before export: interstitial
            nodes that only carry geometry (degree-2 pass-throughs) are removed and
            each chain between intersections/endpoints is collapsed into a single edge
            that keeps the full geometry. A chain is not merged across a change of
            ``maxspeed`` or ``highway``, so each edge keeps one speed limit and one road
            class. Edge ``length`` (and, for pandana and pandarm graphs, the
            ``pandana_weights``) is summed along the chain, and so is a ``travel_time``
            column, as in OSMnx; the merged geometry runs from ``u`` to ``v``. Any other
            column that differs along the chain becomes a list, unless
            ``simplify_kwargs["edge_attr_aggs"]`` aggregates it (e.g.
            ``{"travel_time_s": sum}``, as OSMnx's ``edge_attr_aggs``) or
            ``simplify_kwargs["length_cols"]`` lists it to be summed. See
            ``pyrosm.graph_simplify.simplify_graph`` for details. If False, every segment
            of every way is its own edge.

        simplify_kwargs : dict (optional)
            Keyword arguments forwarded to ``pyrosm.graph_simplify.simplify_graph``
            when ``simplify=True``, e.g. ``edge_attrs_differ`` (the columns whose change
            stops a merge; ``None`` merges across every change, as OSMnx does by
            default), ``node_attrs_include`` (keep the nodes with a value for one of the listed node
            columns or tags as endpoints),
            ``remove_rings``, ``track_merged``, ``length_cols`` or ``edge_attr_aggs``.

        progress : bool | callable | None (default: None)
            `None` follows `pyrosm.enable_logging(progress=...)`, which is `True`
            unless set. `True` shows the bar, `False` nothing, and a callable is called
            as `progress(done, 100)` with the percentage of the export done, from 0 to
            100 and never back.
        """
        graph_type = validate_graph_type(graph_type)
        validate_progress(progress)

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
                simplify=simplify,
                simplify_kwargs=simplify_kwargs,
                progress=progress,
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
                simplify=simplify,
                simplify_kwargs=simplify_kwargs,
                progress=progress,
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
                simplify=simplify,
                simplify_kwargs=simplify_kwargs,
                progress=progress,
            )
        elif graph_type == "pandana":
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
                simplify=simplify,
                simplify_kwargs=simplify_kwargs,
                progress=progress,
            )

    @staticmethod
    def list_cache(filepath=None):
        """List the out-of-core engine's cached layer files -- the GeoParquet files that
        ``engine="out_of_core"`` reads write under ``<tempdir>/pyrosm/cache``.

        Parameters
        ----------

        filepath : str | os.PathLike (optional)
            A source PBF's path. A bare file name names both the file of that name in the
            working directory and the one ``get_data`` downloaded.
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
            A source PBF's path. A bare file name names both the file of that name in the
            working directory and the one ``get_data`` downloaded.
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
        """Remove PBF files downloaded or cropped into the default download directory
        (``<tempdir>/pyrosm``), with the partial files that interrupted downloads leave and
        the records kept beside crops. The result cache, the provider indexes and the
        bundled package datasets are left untouched.

        Parameters
        ----------

        filepath : str | os.PathLike (optional)
            A ``.osm.pbf`` or ``.osh.pbf`` file to remove, with its partial files: a path
            removes exactly that file, a bare file name the file of that name in the download
            directory. When ``None`` (default) every ``*.pbf`` file in the download directory
            is removed.

        Returns
        -------
        int
            The number of PBF and partial files removed.

        Raises
        ------
        ValueError
            If ``filepath`` does not name a ``.osm.pbf`` or ``.osh.pbf`` file.
        """
        from pyrosm.utils.download import clear_downloads

        return clear_downloads(filepath)

    def __getattribute__(self, name):
        # If node-gdf is requested convert to gdf before returning
        if name == "_nodes_gdf":
            return create_nodes_gdf(super(OSM, self).__getattribute__("_nodes"))
        return super(OSM, self).__getattribute__(name)
