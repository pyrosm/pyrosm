"""Public out-of-core readers (one per layer). Each indexes the file's blobs, decodes them
into per-worker shards selecting the layer's elements (and, for point layers, the matching
nodes), then collects and assembles (or streams to GeoParquet) the requested layer."""

import tempfile
import shutil

from pyrosm.data_manager import parse_custom_filter
from pyrosm.utils._compat import require_pyarrow
from pyrosm.engine.blobs import _index_blobs
from pyrosm.engine.pool import _auto_workers, _decode_all
from pyrosm.engine.assemble import _assemble_layer
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
):
    """Read a layer: decode the file in parallel selecting the elements that carry any of
    the filter keys (``osm_keys`` if given, else ``custom_filter``'s keys; and, when
    ``include_nodes``, the matching nodes as point features), refine them by the exact value
    filter (``filter_type`` keep/exclude), then assemble with the full ``tags_as_columns``
    schema (every occurring tag as its own column, the rest in a JSON ``tags`` column, and
    -- when ``keep_metadata`` -- the element metadata), matching the in-memory reader.
    ``keep_ways`` / ``keep_relations`` drop those element kinds from the output.

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

    data_blobs = [
        (offset, size)
        for (blob_type, offset, size) in _index_blobs(filepath)
        if blob_type == "OSMData"
    ]
    if workers is None:
        workers = _auto_workers(filepath, len(data_blobs))

    shard_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_")
    try:
        shard_paths = _decode_all(
            filepath, data_blobs, workers, shard_dir, osm_key_bytes, include_nodes
        )
        if output is None:
            return _assemble_layer(
                shard_paths,
                tags_as_columns,
                keep_metadata,
                filter_spec,
                keep_ways,
                keep_relations,
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
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


def get_buildings(filepath, workers=None, output=None, keep_metadata=True):
    """Read building geometries (ways + relations) from ``filepath`` with the out-of-core
    engine, with the same columns as ``OSM(...).get_buildings()``. See :func:`_get_layer`
    for ``output`` / ``workers`` / ``keep_metadata``."""
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
    )


def get_landuse(filepath, workers=None, output=None, keep_metadata=True):
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
    )


def get_natural(filepath, workers=None, output=None, keep_metadata=True):
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
    )


def get_pois(
    filepath, custom_filter=None, workers=None, output=None, keep_metadata=True
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
    )


def get_boundaries(
    filepath,
    boundary_type="administrative",
    name=None,
    custom_filter=None,
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
    )
