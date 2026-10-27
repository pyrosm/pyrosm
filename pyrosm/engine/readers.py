"""The out-of-core readers (one per layer), internals of the engine behind ``OSM``
(see :mod:`pyrosm.engine`). Each indexes the file's blobs, decodes them
into shards selecting the layer's elements (and, for point layers, the matching
nodes), then collects and assembles (or streams to GeoParquet) the requested layer. A
``bounding_box`` restricts the read to that area."""

from pathlib import Path

import numpy as np
from rapidjson import dumps, loads

from pyrosm.data_manager import parse_custom_filter
from pyrosm.filter_compiler import rename_filter_key
from pyrosm.utils import _compat, default_workers
from pyrosm.engine.decode import _key_bytes
from pyrosm.engine.pool import (
    GRAPH_SHARES,
    LAYER_SHARES,
    NETWORK_NODES_SHARES,
    NETWORK_SHARES,
    _decode_and_run,
)
from pyrosm.engine.bounding_box import _bbox_bounds, _normalize_bounding_box
from pyrosm.engine.assemble import (
    _assemble_graph_network,
    _assemble_layer,
    _assemble_network,
)
from pyrosm.engine import cache, geoparquet
from pyrosm.engine.ways import ArrowFrame
from pyrosm.utils.progress import WRITING, read_steps, start_step, validate_progress

# Tags the geometry assembly reads straight from an element's tag dict (not from the exploded
# columns): ``relations.pyx`` consults ``type`` and ``area`` plus the linestring keys
# (``barrier``/``route``/``railway``/``highway``/``waterway``) to decide whether a relation is an
# area or a LineString. All must be resolved under ``keep_other_tags=False`` so relation
# geometries match the full read (e.g. a ``type=route`` relation stays a LineString and is not
# dropped, #355), then they are dropped with the other leftovers. Keep in sync with
# ``relations.pyx`` ``linestring_keys``.
_GEOMETRY_TAG_KEYS = (
    "type",
    "area",
    "barrier",
    "route",
    "railway",
    "highway",
    "waterway",
)


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
    keep_other_tags=True,
    progress=None,
    keep_node_info=False,
    unix_time=None,
):
    """Read a layer: decode the file in parallel selecting the elements that carry any of
    the filter keys (``osm_keys`` if given, else ``custom_filter``'s keys, or every tagged
    element when ``custom_filter`` is ``None``; and, when ``include_nodes``, the matching
    nodes as point features), refine them by the exact value filter (``filter_type``
    keep/exclude), then assemble with the full ``tags_as_columns``
    schema (every occurring tag as its own column, the rest in a JSON ``tags`` column, and
    -- when ``keep_metadata`` -- the element metadata), matching the in-memory reader.
    ``keep_ways`` / ``keep_relations`` drop those element kinds from the output. A
    ``bounding_box`` restricts the read to that area (relations are partial unless
    ``complete_relations``).

    Returns an in-memory GeoDataFrame, or -- when ``output`` is a path -- streams the layer
    to a chunked GeoParquet there and returns the path (needs the optional ``pyarrow``).
    ``workers`` defaults to ``"auto"``, a thread count chosen by file size; pass
    ``workers=N`` for N threads (with ``PYROSM_DECODE_BACKEND=processes`` they are
    processes, one by default, and on macOS/Windows a parallel read must then run under an
    ``if __name__ == "__main__":`` guard) -- see the package docstring. ``progress``
    (``None``, ``True``, ``False`` or a callable) shows the read as for
    ``OSM(..., progress=...)``; a cached read has nothing to show. ``keep_node_info`` keeps each way's ``nodes`` column (a list of node ids), as
    ``OSM(..., keep_node_info=True)`` does. ``unix_time`` (seconds), for a history file
    (``.osh.pbf``), reads each element's version at that time, as
    ``OSM(...).get_buildings(timestamp=...)`` and the other layer reads do.
    """
    validate_progress(progress)
    workers = default_workers(workers)
    if output is not None:
        _compat.require_pyarrow()
    if custom_filter is None:
        # Every tagged element: no filter keys to select by, nothing to filter by value.
        data_filter, osm_keys, osm_key_bytes = None, None, None
    else:
        data_filter, derived_keys = parse_custom_filter(custom_filter)
        if osm_keys is None:
            osm_keys = derived_keys
        # The tag pyrosm returns as "id_tag" is keyed "id" in the elements' tags.
        data_filter, osm_keys = rename_filter_key(data_filter, osm_keys, "id_tag", "id")
        osm_key_bytes = [_key_bytes(k) for k in osm_keys]
    filter_spec = (osm_keys, data_filter, filter_type)
    # keep_other_tags=False: the workers resolve only the requested tag keys -- the output
    # columns (tags_as_columns) plus the filter keys (so the value filter still has what it
    # checks). None lets them resolve every tag (the default).
    requested_tag_keys = None
    if not keep_other_tags:
        wanted = list(tags_as_columns) + list(osm_keys or []) + list(_GEOMETRY_TAG_KEYS)
        if "id_tag" in tags_as_columns:
            wanted.append("id")
        requested_tag_keys = [_key_bytes(k) for k in dict.fromkeys(wanted)]
    bounding_box = _normalize_bounding_box(bounding_box)
    bounds = _bbox_bounds(bounding_box)

    with read_steps(filepath, progress, LAYER_SHARES) as steps:
        # Per-layer result cache: when returning an in-memory frame (not streaming to a
        # user file) and pyarrow is available, assemble the layer once via the bounded
        # per-call path, write its result to a deterministic GeoParquet keyed by the
        # read, and reuse that file on any identical later read instead of re-decoding
        # the PBF. Each layer is cached separately, so memory stays bounded by one
        # layer. With pyarrow absent the engine returns the in-memory frame (no cache).
        if output is None and _compat.HAS_PYARROW:
            key_params = {
                "filter_spec": filter_spec,
                "tags_as_columns": tags_as_columns,
                "keep_metadata": keep_metadata,
                "include_nodes": include_nodes,
                "keep_ways": keep_ways,
                "keep_relations": keep_relations,
                "bounding_box": bounding_box,
                "complete_relations": complete_relations,
                "keep_other_tags": keep_other_tags,
                "keep_node_info": keep_node_info,
            }
            if unix_time is not None:
                key_params["unix_time"] = unix_time
            cache_path = cache.result_path(filepath, key_params)
            result = cache.materialize(
                cache_path,
                lambda tmp_path: _decode_and_run(
                    filepath,
                    osm_key_bytes,
                    include_nodes,
                    workers,
                    lambda shard_paths, steps: geoparquet._stream_layer_to_parquet(
                        shard_paths,
                        tmp_path,
                        geoparquet._OUTPUT_CHUNK_SIZE,
                        tags_as_columns,
                        keep_metadata,
                        filter_spec,
                        keep_ways,
                        keep_relations,
                        bounding_box,
                        complete_relations,
                        keep_other_tags=keep_other_tags,
                        keep_node_info=keep_node_info,
                        steps=steps,
                    ),
                    bbox_bounds=bounds,
                    requested_tag_keys=requested_tag_keys,
                    steps=steps,
                    spill_all_ways=keep_relations,
                    unix_time=unix_time,
                )
                is not None,
                steps=steps,
            )
            return _node_lists(result)

        def run(shard_paths, steps):
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
                    keep_other_tags=keep_other_tags,
                    keep_node_info=keep_node_info,
                    steps=steps,
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
                keep_other_tags=keep_other_tags,
                keep_node_info=keep_node_info,
                steps=steps,
            )

        result = _decode_and_run(
            filepath,
            osm_key_bytes,
            include_nodes,
            workers,
            run,
            bbox_bounds=bounds,
            requested_tag_keys=requested_tag_keys,
            steps=steps,
            spill_all_ways=keep_relations,
            unix_time=unix_time,
        )
        return result if output is not None else _node_lists(result)


def _node_lists(result):
    """``result`` (a frame, or the ``(nodes, edges)`` pair of a network read) with each
    way's ``nodes`` -- kept with ``keep_node_info`` -- as a list of ints, as the in-memory
    reader returns it, instead of the array the assembly or the cached GeoParquet gives.
    """
    frames = result if isinstance(result, tuple) else (result,)
    for frame in frames:
        if frame is not None and "nodes" in frame.columns:
            frame["nodes"] = [
                v.tolist() if isinstance(v, np.ndarray) else v for v in frame["nodes"]
            ]
    return result


def _require_layer_key(custom_filter, key):
    """A ``None`` ``custom_filter`` becomes ``{key: [True]}``; otherwise it is narrowed to
    the layer of ``key`` (see :func:`pyrosm.utils.require_layer_key`), as the in-memory
    ``get_<layer>_data`` readers do. Handles both a plain dict and an advanced filter.
    """
    from pyrosm.utils import validate_custom_filter, require_layer_key

    if custom_filter is None:
        return {key: [True]}
    return require_layer_key(validate_custom_filter(custom_filter), key)


def _poi_filter(custom_filter):
    """The validated filter of a POI read: ``custom_filter``, by default the elements with an
    ``amenity``, ``shop`` or ``tourism`` tag."""
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        custom_filter = {"amenity": True, "shop": True, "tourism": True}
    return validate_custom_filter(custom_filter)


def _boundary_filter(custom_filter, boundary_type):
    """The validated filter of a boundary read: ``custom_filter``, by default every boundary,
    narrowed to the boundaries of the validated ``boundary_type`` (``"all"`` for any).
    """
    from pyrosm.utils import validate_custom_filter, require_layer_key

    value = True if boundary_type == "all" else [boundary_type]
    if custom_filter is None:
        custom_filter = {"boundary": value}
    return require_layer_key(validate_custom_filter(custom_filter), "boundary", value)


def get_buildings(
    filepath,
    custom_filter=None,
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    keep_nodes=False,
    unix_time=None,
):
    """Read building geometries (ways + relations, and with ``keep_nodes`` the building
    nodes as points) from ``filepath`` with the out-of-core engine, with the same columns as
    ``OSM(...).get_buildings()``. ``custom_filter`` refines which buildings to keep (the
    ``building`` key is always ensured); ``extra_attributes`` / ``tags_to_keep`` adjust the
    tag columns. See :func:`_get_layer` for ``bounding_box`` / ``complete_relations`` /
    ``output`` / ``workers`` / ``keep_metadata`` / ``progress`` / ``unix_time``.
    """
    from pyrosm.config import Conf
    from pyrosm.utils import read_tag_columns

    read_filter = _require_layer_key(custom_filter, "building")
    return _get_layer(
        filepath,
        read_filter,
        "keep",
        read_tag_columns(
            Conf.tags.building, extra_attributes, tags_to_keep, read_filter.keys()
        ),
        workers,
        output,
        keep_metadata,
        include_nodes=keep_nodes,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )


def get_landuse(
    filepath,
    custom_filter=None,
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    keep_nodes=True,
    unix_time=None,
):
    """Read landuse geometries (nodes as points unless ``keep_nodes`` is False, ways and
    relations) from ``filepath`` with the out-of-core engine, with the same columns as
    ``OSM(...).get_landuse()``. ``custom_filter`` refines
    which landuse to keep (the ``landuse`` key is always ensured); ``extra_attributes`` /
    ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for the others."""
    from pyrosm.config import Conf
    from pyrosm.utils import read_tag_columns

    read_filter = _require_layer_key(custom_filter, "landuse")
    return _get_layer(
        filepath,
        read_filter,
        "keep",
        read_tag_columns(
            Conf.tags.landuse, extra_attributes, tags_to_keep, read_filter.keys()
        ),
        workers,
        output,
        keep_metadata,
        include_nodes=keep_nodes,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )


def get_natural(
    filepath,
    custom_filter=None,
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    keep_nodes=True,
    unix_time=None,
):
    """Read natural features (nodes as points unless ``keep_nodes`` is False, ways and
    relations) from ``filepath`` with the out-of-core engine, with the same columns as ``OSM(...).get_natural()``. ``custom_filter``
    refines which natural features to keep (the ``natural`` key is always ensured);
    ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for
    the others."""
    from pyrosm.config import Conf
    from pyrosm.utils import read_tag_columns

    read_filter = _require_layer_key(custom_filter, "natural")
    return _get_layer(
        filepath,
        read_filter,
        "keep",
        read_tag_columns(
            Conf.tags.natural, extra_attributes, tags_to_keep, read_filter.keys()
        ),
        workers,
        output,
        keep_metadata,
        include_nodes=keep_nodes,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )


def get_pois(
    filepath,
    custom_filter=None,
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    unix_time=None,
):
    """Read points of interest (nodes + ways + relations) from ``filepath`` with the
    out-of-core engine, with the same columns as ``OSM(...).get_pois(custom_filter=...)``.
    ``custom_filter`` defaults to ``{"amenity": True, "shop": True, "tourism": True}``;
    ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for
    the other keyword arguments."""
    from pyrosm.config import Conf

    from pyrosm.utils import feature_columns, read_tag_columns

    custom_filter = _poi_filter(custom_filter)
    base_tags = feature_columns(custom_filter.keys(), Conf.tags._basic_tags)
    return _get_layer(
        filepath,
        custom_filter,
        "keep",
        read_tag_columns(
            base_tags, extra_attributes, tags_to_keep, custom_filter.keys()
        ),
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )


def get_boundaries(
    filepath,
    boundary_type="administrative",
    name=None,
    custom_filter=None,
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    unix_time=None,
):
    """Read boundaries (ways + relations) from ``filepath`` with the out-of-core engine,
    with the same columns as ``OSM(...).get_boundaries()``. ``boundary_type`` selects the
    ``boundary=*`` value (``"all"`` for any); ``name`` keeps only boundaries whose name
    contains that text; ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See
    :func:`_get_layer` for the other keyword arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import read_tag_columns, validate_boundary_type

    boundary_type = validate_boundary_type(boundary_type)
    if name is not None and output is not None:
        raise ValueError(
            "get_boundaries(name=...) cannot be combined with output= -- the streamed "
            "GeoParquet is written before the name filter is applied. Omit output= to "
            "filter by name, or omit name to stream all boundaries."
        )
    custom_filter = _boundary_filter(custom_filter, boundary_type)
    gdf = _get_layer(
        filepath,
        custom_filter,
        "keep",
        read_tag_columns(
            Conf.tags.boundary, extra_attributes, tags_to_keep, custom_filter.keys()
        ),
        workers,
        output,
        keep_metadata,
        include_nodes=False,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )
    # The name post-filter, as OSM.get_boundaries does. The output= + name combination is
    # rejected above, so reaching here means an in-memory frame.
    if name is not None and gdf is not None:
        from pyrosm.boundary import boundaries_named

        gdf = boundaries_named(gdf, name)
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
    extra_attributes=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
    keep_other_tags=True,
    progress=None,
    keep_node_info=False,
    tags_to_keep=None,
    unix_time=None,
):
    """Read OSM elements matching an arbitrary ``custom_filter`` from ``filepath`` with the
    out-of-core engine, with the same columns as
    ``OSM(...).get_data_by_custom_criteria(...)``. ``osm_keys_to_keep`` (if given) is the
    set of keys filtered on; ``keep_nodes`` / ``keep_ways`` / ``keep_relations`` select
    which element kinds are returned; ``tags_to_keep`` (another name of
    ``tags_as_columns``) and ``extra_attributes`` set the tag columns, to which the filter
    keys are added (see :func:`pyrosm.utils.custom_read_columns`).
    ``keep_other_tags=False`` resolves only the requested tags (``tags_as_columns`` plus the
    filter keys) and drops the JSON ``tags`` column of leftovers, so the read does minimal
    tag work. ``custom_filter=None`` reads every tagged element (``filter_type`` is then
    not used), with the default columns of every known feature as tag columns, or, with
    ``osm_keys_to_keep``, every element with one of those keys. See :func:`_get_layer` for
    the other keyword arguments.
    """
    from pyrosm.utils import (
        custom_read_columns,
        keys_as_filter,
        read_tag_columns,
        validate_custom_filter,
        validate_osm_keys,
    )

    validate_osm_keys(osm_keys_to_keep)
    custom_filter, filter_type = keys_as_filter(
        custom_filter, osm_keys_to_keep, filter_type
    )
    keep_all = custom_filter is None
    if keep_all:
        # Every tagged element: there is nothing to keep or exclude by value, and the
        # filter keys are not used.
        filter_type = "keep"
        osm_keys_to_keep = None
    else:
        custom_filter = validate_custom_filter(custom_filter)
    filter_type = filter_type.lower()
    if isinstance(osm_keys_to_keep, str):
        osm_keys_to_keep = [osm_keys_to_keep]
    tags_as_columns = read_tag_columns(
        custom_read_columns(
            custom_filter, osm_keys_to_keep, tags_as_columns, tags_to_keep
        ),
        extra_attributes,
    )
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
        keep_other_tags=keep_other_tags,
        progress=progress,
        keep_node_info=keep_node_info,
        unix_time=unix_time,
    )


def _network_filter_type(custom_filter, filter_type):
    """``(custom_filter, filter_type)`` of a network read, validated: with a filter,
    ``filter_type`` defaults to ``"keep"``, as in every other reader; without one it is
    ``"exclude"`` (the network type's own filter applies)."""
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        return None, "exclude"
    custom_filter = validate_custom_filter(custom_filter)
    if filter_type is None:
        return custom_filter, "keep"
    if isinstance(filter_type, str) and filter_type.lower() in ("keep", "exclude"):
        return custom_filter, filter_type.lower()
    raise ValueError("'filter_type' -parameter should be either 'keep' or 'exclude'.")


def _network_filter(network_type):
    """Resolve a predefined ``network_type`` to its filter dict, mirroring
    ``OSM._get_network_filter``."""
    from pyrosm.config.osm_filters import get_osm_filter

    return get_osm_filter(network_type)


def _write_result_parquet(gdf, path, steps=None):
    """Write an assembled frame to ``path`` (a result-cache temp file or an ``output=``
    path), the step ``WRITING`` of ``steps`` (a :class:`~pyrosm.utils.progress.Steps`,
    or None); a ``None`` (empty read) writes nothing and reports the empty result."""
    if gdf is None:
        return False
    start_step(steps, WRITING)
    if isinstance(gdf, ArrowFrame):
        gdf.write(path)
    else:
        gdf.to_parquet(path)
    return True


def _write_nodes_parquet(node_gdf, path):
    """Write the graph-export node frame to ``path``, serialising its ``tags`` dict column to
    JSON strings first -- a column of heterogeneous dicts has no faithful GeoParquet schema
    (pyarrow infers a struct and drops keys), whereas JSON strings round-trip exactly.
    """
    node_gdf = node_gdf.copy()
    node_gdf["tags"] = node_gdf["tags"].map(
        lambda t: dumps(t) if isinstance(t, dict) else None
    )
    node_gdf.to_parquet(path)


def _read_nodes_parquet(path):
    """Read a cached node frame back, parsing the JSON ``tags`` column written by
    :func:`_write_nodes_parquet` back into dicts (``None`` for missing), matching the in-memory
    reader's representation."""
    gdf = cache.read_result(path)
    gdf["tags"] = gdf["tags"].map(lambda s: loads(s) if isinstance(s, str) else None)
    return gdf


def _write_network_pair(result, edges_path, nodes_path, steps=None):
    """Write ``get_network(nodes=True)``'s ``(nodes, edges)`` tuple to the two cache
    files, the step ``WRITING`` of ``steps``; a ``(None, None)`` (empty read) writes
    nothing and reports the empty result."""
    node_gdf, edges = result
    if edges is None:
        return False
    start_step(steps, WRITING)
    edges.to_parquet(edges_path)
    _write_nodes_parquet(node_gdf, nodes_path)
    return True


def _write_network_dir(result, dirpath, steps=None):
    """Write ``get_network(nodes=True, output=dirpath)``'s ``(nodes, edges)`` tuple into
    ``dirpath`` as ``edges.parquet`` + ``nodes.parquet`` and return ``dirpath``, the
    step ``WRITING`` of ``steps``; an empty read writes nothing and returns ``None``."""
    node_gdf, edges = result
    if edges is None:
        return None
    start_step(steps, WRITING)
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out / "edges.parquet")
    _write_nodes_parquet(node_gdf, out / "nodes.parquet")
    return dirpath


def get_network(
    filepath,
    network_type="walking",
    extra_attributes=None,
    nodes=False,
    custom_filter=None,
    filter_type=None,
    tags_to_keep=None,
    bounding_box=None,
    workers=None,
    output=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    unix_time=None,
):
    """Read a street network (``highway=*`` ways as LineString edges + a ``length`` column)
    from ``filepath`` with the out-of-core engine, with the same columns as
    ``OSM(...).get_network()``. ``network_type`` selects a predefined filter (``walking`` /
    ``driving`` / ``cycling`` / ``all`` / ...); a ``custom_filter`` replaces it
    (``filter_type`` keep/exclude). ``extra_attributes`` / ``tags_to_keep`` adjust the tag
    columns. ``bounding_box`` (a ``[minx, miny, maxx, maxy]`` list or a shapely polygon)
    restricts the read to that area.

    ``nodes=True`` returns ``(nodes, edges)``: the ways are sliced into per-segment edges
    and the graph-export node frame is built (its node tags + metadata are gathered with a
    second pass over the file), matching ``OSM(...).get_network(nodes=True)``.

    With ``output=None`` (default) the result is cached to a deterministic GeoParquet under a
    temp dir and reused on identical later reads, like the area/point layers; ``nodes=True``
    caches the ``(nodes, edges)`` tuple as two files. ``output="path"`` writes the edges to that
    GeoParquet and returns the path; with ``nodes=True`` it writes ``edges.parquet`` +
    ``nodes.parquet`` into the ``path`` directory and returns the directory (both require
    ``pyarrow``). With ``pyarrow`` absent the default read returns the in-memory result with no
    cache. ``progress`` shows the read as for ``OSM(..., progress=...)``, including the second
    pass of ``nodes=True``; a cached read has nothing to show. ``unix_time`` (seconds), for
    a history file (``.osh.pbf``), reads each element's version at that time, as
    ``OSM(...).get_network(timestamp=...)`` does."""
    from pyrosm.graphs import record_network_type

    validate_progress(progress)
    workers = default_workers(workers)
    tags_as_columns, filter_spec = _network_spec(
        network_type, extra_attributes, custom_filter, filter_type, tags_to_keep
    )
    bounding_box = _normalize_bounding_box(bounding_box)

    if output is not None:
        _compat.require_pyarrow()

    shares = NETWORK_NODES_SHARES if nodes else NETWORK_SHARES
    with read_steps(filepath, progress, shares) as steps:

        def assemble(shard_paths, steps):
            edges, node_gdf = _assemble_network(
                shard_paths,
                tags_as_columns,
                keep_metadata,
                filter_spec,
                nodes,
                bounding_box,
                filepath=filepath,
                steps=steps,
                keep_node_info=keep_node_info,
                unix_time=unix_time,
            )
            return (node_gdf, edges) if nodes else edges

        def decode():
            return _decode_network(
                filepath, filter_spec, bounding_box, workers, steps, assemble, unix_time
            )

        # A user-supplied output writes the result there and returns it: a GeoParquet
        # file for the edges (nodes=False), or a directory holding edges.parquet +
        # nodes.parquet (nodes=True).
        if output is not None:
            if nodes:
                return _write_network_dir(decode(), output, steps)
            return output if _write_result_parquet(decode(), output, steps) else None

        # With pyarrow absent there is nowhere to cache, so return the direct in-memory
        # result (the edges frame, or the (nodes, edges) tuple for nodes=True).
        if not _compat.HAS_PYARROW:
            return record_network_type(_node_lists(decode()), network_type)

        # Cache the result to / serve it from a per-read GeoParquet, keyed apart from
        # the area/point layers via "network". nodes=True is a (nodes, edges) tuple,
        # cached as two files.
        key_params = {
            "network": True,
            "nodes": nodes,
            "filter_spec": filter_spec,
            "tags_as_columns": tags_as_columns,
            "keep_metadata": keep_metadata,
            "bounding_box": bounding_box,
            "keep_node_info": keep_node_info,
        }
        if unix_time is not None:
            key_params["unix_time"] = unix_time
        if nodes:
            edges_path = cache.result_path(filepath, {**key_params, "part": "edges"})
            nodes_path = cache.result_path(filepath, {**key_params, "part": "nodes"})
            result = cache.materialize_pair(
                edges_path,
                nodes_path,
                lambda e_tmp, n_tmp: _write_network_pair(decode(), e_tmp, n_tmp, steps),
                read_nodes=_read_nodes_parquet,
                steps=steps,
            )
            return record_network_type(_node_lists(result), network_type)

        cache_path = cache.result_path(filepath, key_params)
        result = cache.materialize(
            cache_path,
            lambda tmp_path: _write_result_parquet(decode(), tmp_path, steps),
            steps=steps,
        )
        return record_network_type(_node_lists(result), network_type)


def _network_spec(
    network_type, extra_attributes, custom_filter, filter_type, tags_to_keep
):
    """The tag columns and the filter of a network read: ``(tags_as_columns,
    filter_spec)``, the filter ``(network_keys, data_filter, filter_type)`` with the keys
    whose ways are read."""
    from pyrosm.config import Conf
    from pyrosm.utils import read_tag_columns
    from pyrosm.filter_compiler import CompiledFilter

    # The network type drives the graph semantics even with a custom_filter, so it is
    # validated for every read, as the in-memory reader does.
    _network_filter(network_type)

    # Predefined networks select 'highway' ways; an advanced custom filter selects ways by its
    # own positive keys (so railway/cycleway networks work).
    network_keys = ["highway"]
    column_keys = ["highway"]
    if custom_filter is not None:
        custom_filter, filter_type = _network_filter_type(custom_filter, filter_type)
        advanced_filter = isinstance(custom_filter, CompiledFilter)
        network_filter = custom_filter
        # The filter keys are columns too (e.g. 'bicycle', 'service').
        column_keys = custom_filter.keys()
        if advanced_filter:
            network_keys = list(custom_filter.positive_keys)
    else:
        network_filter = _network_filter(network_type)
        # Predefined networks are always exclude filters keyed on 'highway'.
        filter_type = "exclude"
    tags_as_columns = read_tag_columns(
        Conf.tags.highway, extra_attributes, tags_to_keep, column_keys
    )

    data_filter = (
        None if network_filter is None else parse_custom_filter(network_filter)[0]
    )
    # The tag pyrosm returns as "id_tag" is keyed "id" in the ways' tags.
    data_filter, network_keys = rename_filter_key(
        data_filter, network_keys, "id_tag", "id"
    )
    return tags_as_columns, (network_keys, data_filter, filter_type)


def _decode_network(
    filepath, filter_spec, bounding_box, workers, steps, assemble, unix_time=None
):
    """``assemble(shard_paths, steps)`` over the ways of ``filepath`` with the keys of
    ``filter_spec`` (see :func:`_network_spec`), decoded as
    :func:`~pyrosm.engine.pool._decode_and_run` does with ``workers``, ``steps`` and
    ``unix_time``.
    """
    return _decode_and_run(
        filepath,
        [_key_bytes(k) for k in filter_spec[0]],
        False,
        workers,
        assemble,
        bbox_bounds=_bbox_bounds(bounding_box),
        steps=steps,
        spill_all_ways=False,
        unix_time=unix_time,
    )


def graph_network(
    filepath,
    network_type="walking",
    extra_attributes=None,
    custom_filter=None,
    filter_type=None,
    tags_to_keep=None,
    bounding_box=None,
    workers=None,
    keep_metadata=True,
    progress=None,
    keep_node_info=False,
    unix_time=None,
):
    """The network that :func:`get_network` with ``nodes=True`` reads, for graph export:
    ``(nodes, edges)``, readers of its node and edge tables that do not build them (see
    :func:`~pyrosm.engine.assemble._assemble_graph_network`), or ``(None, None)`` when no
    edge is read. The parameters are those of :func:`get_network`; nothing is cached."""
    validate_progress(progress)
    workers = default_workers(workers)
    tags_as_columns, filter_spec = _network_spec(
        network_type, extra_attributes, custom_filter, filter_type, tags_to_keep
    )
    bounding_box = _normalize_bounding_box(bounding_box)

    def assemble(shard_paths, steps):
        return _assemble_graph_network(
            shard_paths,
            tags_as_columns,
            keep_metadata,
            filter_spec,
            bounding_box,
            filepath,
            steps,
            keep_node_info,
            unix_time,
        )

    with read_steps(filepath, progress, GRAPH_SHARES) as steps:
        return _decode_network(
            filepath, filter_spec, bounding_box, workers, steps, assemble, unix_time
        )
