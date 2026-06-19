"""Public out-of-core readers (one per layer). Each indexes the file's blobs, decodes them
into per-worker shards selecting the layer's elements (and, for point layers, the matching
nodes), then collects and assembles (or streams to GeoParquet) the requested layer. A
``bounding_box`` restricts the read to that area."""

from pyrosm.data_manager import parse_custom_filter
from pyrosm.utils._compat import require_pyarrow
from pyrosm.engine.pool import _decode_and_run
from pyrosm.engine.bounding_box import _bbox_bounds, _normalize_bounding_box
from pyrosm.engine.assemble import _assemble_layer, _assemble_network
from pyrosm.engine import geoparquet


def _get_layer(
    filepath,
    custom_filter,
    filter_type,
    tags_as_columns,
    workers,
    output,
    keep_metadata,
    include_nodes=True,
    keep_ways=True,
    keep_relations=True,
    osm_keys=None,
    bounding_box=None,
    complete_relations=False,
):
    """Read a layer: decode the file in parallel selecting the elements that carry any of
    the filter keys (``osm_keys`` if given, else ``custom_filter``'s keys; and, when
    ``include_nodes``, the matching nodes as point features), refine them by the exact value
    filter (``filter_type`` keep/exclude), then assemble with the full ``tags_as_columns``
    schema (every occurring tag as its own column, the rest in a JSON ``tags`` column, and
    -- when ``keep_metadata`` -- the element metadata), matching the in-memory reader.
    ``keep_ways`` / ``keep_relations`` drop those element kinds from the output. A
    ``bounding_box`` restricts the read to that area (relations are partial unless
    ``complete_relations``).

    Returns an in-memory GeoDataFrame, or -- when ``output`` is a path -- streams the layer
    to a chunked GeoParquet there and returns the path (needs the optional ``pyarrow``).
    ``workers`` defaults to one for small files and otherwise to a worker per CPU."""
    if output is not None:
        require_pyarrow()
    data_filter, derived_keys = parse_custom_filter(custom_filter)
    if osm_keys is None:
        osm_keys = derived_keys
    filter_spec = (osm_keys, data_filter, filter_type)
    osm_key_bytes = [k.encode("utf-8") for k in osm_keys]
    bounding_box = _normalize_bounding_box(bounding_box)
    bounds = _bbox_bounds(bounding_box)

    def run(shard_paths):
        if output is None:
            return _assemble_layer(
                shard_paths,
                tags_as_columns,
                keep_metadata,
                filter_spec,
                keep_ways,
                keep_relations,
                bounding_box,
                complete_relations,
            )
        return geoparquet._stream_layer_to_parquet(
            shard_paths,
            output,
            geoparquet._OUTPUT_CHUNK_SIZE,
            tags_as_columns,
            keep_metadata,
            filter_spec,
            keep_ways,
            keep_relations,
            bounding_box,
            complete_relations,
        )

    return _decode_and_run(
        filepath, osm_key_bytes, include_nodes, workers, run, bbox_bounds=bounds
    )


def get_buildings(
    filepath,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read building geometries (ways + relations) from ``filepath`` with the out-of-core
    engine, with the same columns as ``OSM(...).get_buildings()``. See :func:`_get_layer`
    for ``bounding_box`` / ``complete_relations`` / ``output`` / ``workers`` /
    ``keep_metadata``."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        {"building": [True]},
        "keep",
        Conf.tags.building,
        workers,
        output,
        keep_metadata,
        include_nodes=False,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )


def get_landuse(
    filepath,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read landuse geometries (ways + relations) from ``filepath`` with the out-of-core
    engine, with the same columns as ``OSM(...).get_landuse()``. See :func:`_get_layer` for
    the keyword arguments."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        {"landuse": [True]},
        "keep",
        Conf.tags.landuse,
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )


def get_natural(
    filepath,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read natural features (nodes + ways + relations) from ``filepath`` with the
    out-of-core engine, with the same columns as ``OSM(...).get_natural()``. See
    :func:`_get_layer` for the keyword arguments."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        {"natural": [True]},
        "keep",
        Conf.tags.natural,
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )


def get_pois(
    filepath,
    custom_filter=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read points of interest (nodes + ways + relations) from ``filepath`` with the
    out-of-core engine, with the same columns as ``OSM(...).get_pois(custom_filter=...)``.
    ``custom_filter`` defaults to ``{"amenity": True, "shop": True, "tourism": True}``. See
    :func:`_get_layer` for the other keyword arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        custom_filter = {"amenity": True, "shop": True, "tourism": True}
    # Per-key tag columns, exactly as OSM.get_pois builds them (Conf.tags.<key>, or the
    # basic tags for keys without a dedicated column set).
    tags_as_columns = []
    for k in custom_filter.keys():
        tags_as_columns += getattr(Conf.tags, k, list(Conf.tags._basic_tags))
    return _get_layer(
        filepath,
        validate_custom_filter(custom_filter),
        "keep",
        tags_as_columns,
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )


def get_boundaries(
    filepath,
    boundary_type="administrative",
    name=None,
    custom_filter=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read boundaries (ways + relations) from ``filepath`` with the out-of-core engine,
    with the same columns as ``OSM(...).get_boundaries()``. ``boundary_type`` selects the
    ``boundary=*`` value (``"all"`` for any); ``name`` keeps only boundaries whose name
    contains that text. See :func:`_get_layer` for the other keyword arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter, validate_boundary_type

    boundary_type = validate_boundary_type(boundary_type)
    if name is not None and output is not None:
        raise ValueError(
            "get_boundaries(name=...) cannot be combined with output= -- the streamed "
            "GeoParquet is written before the name filter is applied. Omit output= to "
            "filter by name, or omit name to stream all boundaries."
        )
    value = True if boundary_type == "all" else [boundary_type]
    if custom_filter is None:
        custom_filter = {"boundary": value}
    if "boundary" not in custom_filter:
        custom_filter["boundary"] = True
    gdf = _get_layer(
        filepath,
        validate_custom_filter(custom_filter),
        "keep",
        list(Conf.tags.boundary),
        workers,
        output,
        keep_metadata,
        include_nodes=False,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )
    # Name post-filter (substring match), as OSM.get_boundaries does. The output= + name
    # combination is rejected above, so reaching here means an in-memory frame.
    if name is not None and gdf is not None:
        if "name" not in gdf.columns:
            raise ValueError(
                "Could not filter by name from given area. "
                "Any of the OSM elements did not have a name tag."
            )
        gdf = gdf.dropna(subset=["name"])
        gdf = gdf.loc[gdf["name"].str.contains(name)].reset_index(drop=True).copy()
    return gdf


def get_data_by_custom_criteria(
    filepath,
    custom_filter,
    osm_keys_to_keep=None,
    filter_type="keep",
    tags_as_columns=None,
    keep_nodes=True,
    keep_ways=True,
    keep_relations=True,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read OSM elements matching an arbitrary ``custom_filter`` from ``filepath`` with the
    out-of-core engine, with the same columns as
    ``OSM(...).get_data_by_custom_criteria(...)``. ``osm_keys_to_keep`` (if given) is the
    set of keys filtered on; ``keep_nodes`` / ``keep_ways`` / ``keep_relations`` select
    which element kinds are returned. See :func:`_get_layer` for the other keyword
    arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter, validate_osm_keys

    custom_filter = validate_custom_filter(custom_filter)
    filter_type = filter_type.lower()
    validate_osm_keys(osm_keys_to_keep)
    if isinstance(osm_keys_to_keep, str):
        osm_keys_to_keep = [osm_keys_to_keep]
    if tags_as_columns is None:
        tags_as_columns = []
        for k in custom_filter.keys():
            try:
                tags_as_columns += getattr(Conf.tags, k)
            except Exception:
                pass
        # Keys without a dedicated column set become columns themselves.
        if len(tags_as_columns) == 0:
            tags_as_columns = list(custom_filter.keys())
    return _get_layer(
        filepath,
        custom_filter,
        filter_type,
        tags_as_columns,
        workers,
        output,
        keep_metadata,
        include_nodes=keep_nodes,
        keep_ways=keep_ways,
        keep_relations=keep_relations,
        osm_keys=osm_keys_to_keep,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
    )


def _network_filter(network_type):
    """Resolve a predefined ``network_type`` to its filter dict (or ``None`` for the
    unrestricted ``all`` / ``driving_psv``), mirroring ``OSM._get_network_filter``."""
    from pyrosm.config import Conf

    possible = Conf._possible_network_filters
    msg = "'network_type' should be one of: " + ", ".join(possible)
    if not isinstance(network_type, str):
        raise ValueError(msg)
    network_type = network_type.lower()
    if network_type not in possible:
        raise ValueError(msg)
    if network_type == "walking":
        return Conf.network_filters.walking
    if network_type == "driving":
        return Conf.network_filters.driving
    if network_type == "driving+service":
        return Conf.network_filters.driving_psv
    if network_type == "cycling":
        return Conf.network_filters.cycling
    return None  # "all" and "driving_psv" -> every highway


def get_network(
    filepath,
    network_type="walking",
    nodes=False,
    custom_filter=None,
    filter_type="exclude",
    bounding_box=None,
    workers=None,
    keep_metadata=True,
):
    """Read a street network (``highway=*`` ways as LineString edges + a ``length`` column)
    from ``filepath`` with the out-of-core engine, with the same columns as
    ``OSM(...).get_network()``. ``network_type`` selects a predefined filter (``walking`` /
    ``driving`` / ``cycling`` / ``all`` / ...); a ``custom_filter`` replaces it
    (``filter_type`` keep/exclude). ``bounding_box`` (a ``[minx, miny, maxx, maxy]`` list or
    a shapely polygon) restricts the read to that area.

    ``nodes=True`` (the graph-export node frame) is not yet supported by this backend: the
    coordinate store carries only id/lon/lat, so the node frame would lack the element
    metadata the in-memory reader produces."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter

    if nodes:
        raise NotImplementedError(
            "get_network(nodes=True) (graph-export node frame) is not yet supported by "
            "the out-of-core engine; only the edges are available."
        )

    tags_as_columns = list(Conf.tags.highway)
    if custom_filter is not None:
        custom_filter = validate_custom_filter(custom_filter)
        filter_type = filter_type.lower()
        if filter_type not in ("keep", "exclude"):
            raise ValueError(
                "'filter_type' -parameter should be either 'keep' or 'exclude'."
            )
        network_filter = custom_filter
        # Expose the filter keys as columns too (e.g. 'bicycle', 'service').
        for key in custom_filter.keys():
            if key not in tags_as_columns:
                tags_as_columns.append(key)
    else:
        network_filter = _network_filter(network_type)
        # Predefined networks are always exclude filters keyed on 'highway'.
        filter_type = "exclude"

    data_filter = (
        None if network_filter is None else parse_custom_filter(network_filter)[0]
    )
    # Networks always select highway ways (the filter values may reference other keys).
    filter_spec = (["highway"], data_filter, filter_type)
    bounding_box = _normalize_bounding_box(bounding_box)
    bounds = _bbox_bounds(bounding_box)

    def run(shard_paths):
        edges, _ = _assemble_network(
            shard_paths,
            tags_as_columns,
            keep_metadata,
            filter_spec,
            False,
            bounding_box,
        )
        return edges

    return _decode_and_run(
        filepath, [b"highway"], False, workers, run, bbox_bounds=bounds
    )
