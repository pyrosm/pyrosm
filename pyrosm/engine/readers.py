"""Public out-of-core readers (one per layer). Each indexes the file's blobs, decodes them
into per-worker shards selecting the layer's elements (and, for point layers, the matching
nodes), then collects and assembles (or streams to GeoParquet) the requested layer. A
``bounding_box`` restricts the read to that area."""

import os
import tempfile

from pyrosm.data_manager import parse_custom_filter
from pyrosm.utils import _compat
from pyrosm.engine.pool import _decode_and_run
from pyrosm.engine.bounding_box import _bbox_bounds, _normalize_bounding_box
from pyrosm.engine.assemble import _assemble_layer, _assemble_network
from pyrosm.engine import cache, geoparquet


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
    ``workers`` defaults to one for small files and otherwise to a worker per CPU; on
    macOS/Windows a parallel read must run under an ``if __name__ == "__main__":`` guard
    (otherwise it falls back to one process with a warning) -- see the package docstring.
    """
    if output is not None:
        _compat.require_pyarrow()
    data_filter, derived_keys = parse_custom_filter(custom_filter)
    if osm_keys is None:
        osm_keys = derived_keys
    filter_spec = (osm_keys, data_filter, filter_type)
    osm_key_bytes = [k.encode("utf-8") for k in osm_keys]
    bounding_box = _normalize_bounding_box(bounding_box)
    bounds = _bbox_bounds(bounding_box)

    # Per-layer result cache: when returning an in-memory frame (not streaming to a user file)
    # and pyarrow is available, assemble the layer once via the bounded per-call path, write its
    # result to a deterministic GeoParquet keyed by the read, and reuse that file on any identical
    # later read instead of re-decoding the PBF. Each layer is cached separately, so memory stays
    # bounded by one layer. With pyarrow absent the engine returns the in-memory frame (no cache).
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
        }
        cache_path = cache.result_path(filepath, key_params)
        # An empty result is recorded with a side marker, so an identical later read returns None
        # without re-decoding the whole file.
        empty_marker = cache_path + ".empty"
        if os.path.exists(empty_marker):
            return None
        if not os.path.exists(cache_path):
            # Stream to a unique temp file in the cache dir, then atomically move it into place,
            # so concurrent identical first-reads never share a temp file or see a half-written
            # cache. The temp file is removed unless it was moved into place.
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(cache_path),
                prefix=os.path.basename(cache_path) + ".",
                suffix=".tmp",
            )
            os.close(fd)
            try:
                written = _decode_and_run(
                    filepath,
                    osm_key_bytes,
                    include_nodes,
                    workers,
                    lambda shard_paths: geoparquet._stream_layer_to_parquet(
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
                    ),
                    bbox_bounds=bounds,
                )
                if written is None:
                    with open(empty_marker, "w"):
                        pass
                    return None
                os.replace(tmp_path, cache_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return cache.read_result(cache_path)

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


def _resolve_tags_as_columns(base_tags, extra_attributes, tags_to_keep):
    """Build the tag-as-columns list the way the in-memory feature methods do: ``tags_to_keep``
    replaces the layer default, ``extra_attributes`` appends (both validated)."""
    from pyrosm.utils import validate_tags_as_columns

    tags_as_columns = list(base_tags)
    if tags_to_keep is not None:
        validate_tags_as_columns(tags_to_keep)
        tags_as_columns = list(tags_to_keep)
    if extra_attributes is not None:
        validate_tags_as_columns(extra_attributes)
        tags_as_columns = tags_as_columns + list(extra_attributes)
    return tags_as_columns


def _ensure_layer_key(custom_filter, key):
    """A ``None`` ``custom_filter`` becomes ``{key: [True]}``; otherwise the layer key is
    ensured present -- mirroring the in-memory ``get_<layer>_data`` default merge."""
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        return {key: [True]}
    custom_filter = dict(validate_custom_filter(custom_filter))
    if key not in custom_filter:
        custom_filter[key] = [True]
    return custom_filter


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
):
    """Read building geometries (ways + relations) from ``filepath`` with the out-of-core
    engine, with the same columns as ``OSM(...).get_buildings()``. ``custom_filter`` refines
    which buildings to keep (the ``building`` key is always ensured); ``extra_attributes`` /
    ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for ``bounding_box`` /
    ``complete_relations`` / ``output`` / ``workers`` / ``keep_metadata``."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        _ensure_layer_key(custom_filter, "building"),
        "keep",
        _resolve_tags_as_columns(Conf.tags.building, extra_attributes, tags_to_keep),
        workers,
        output,
        keep_metadata,
        include_nodes=False,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
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
):
    """Read landuse geometries (ways + relations) from ``filepath`` with the out-of-core
    engine, with the same columns as ``OSM(...).get_landuse()``. ``custom_filter`` refines
    which landuse to keep (the ``landuse`` key is always ensured); ``extra_attributes`` /
    ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for the others."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        _ensure_layer_key(custom_filter, "landuse"),
        "keep",
        _resolve_tags_as_columns(Conf.tags.landuse, extra_attributes, tags_to_keep),
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
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
):
    """Read natural features (nodes + ways + relations) from ``filepath`` with the
    out-of-core engine, with the same columns as ``OSM(...).get_natural()``. ``custom_filter``
    refines which natural features to keep (the ``natural`` key is always ensured);
    ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for
    the others."""
    from pyrosm.config import Conf

    return _get_layer(
        filepath,
        _ensure_layer_key(custom_filter, "natural"),
        "keep",
        _resolve_tags_as_columns(Conf.tags.natural, extra_attributes, tags_to_keep),
        workers,
        output,
        keep_metadata,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
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
):
    """Read points of interest (nodes + ways + relations) from ``filepath`` with the
    out-of-core engine, with the same columns as ``OSM(...).get_pois(custom_filter=...)``.
    ``custom_filter`` defaults to ``{"amenity": True, "shop": True, "tourism": True}``;
    ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See :func:`_get_layer` for
    the other keyword arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter

    if custom_filter is None:
        custom_filter = {"amenity": True, "shop": True, "tourism": True}
    # Per-key tag columns, exactly as OSM.get_pois builds them (Conf.tags.<key>, or the
    # basic tags for keys without a dedicated column set).
    base_tags = []
    for k in custom_filter.keys():
        base_tags += getattr(Conf.tags, k, list(Conf.tags._basic_tags))
    return _get_layer(
        filepath,
        validate_custom_filter(custom_filter),
        "keep",
        _resolve_tags_as_columns(base_tags, extra_attributes, tags_to_keep),
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
    extra_attributes=None,
    tags_to_keep=None,
    bounding_box=None,
    complete_relations=False,
    workers=None,
    output=None,
    keep_metadata=True,
):
    """Read boundaries (ways + relations) from ``filepath`` with the out-of-core engine,
    with the same columns as ``OSM(...).get_boundaries()``. ``boundary_type`` selects the
    ``boundary=*`` value (``"all"`` for any); ``name`` keeps only boundaries whose name
    contains that text; ``extra_attributes`` / ``tags_to_keep`` adjust the tag columns. See
    :func:`_get_layer` for the other keyword arguments."""
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
        _resolve_tags_as_columns(Conf.tags.boundary, extra_attributes, tags_to_keep),
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
    extra_attributes=None,
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
    which element kinds are returned; ``extra_attributes`` adds further tag columns. See
    :func:`_get_layer` for the other keyword arguments."""
    from pyrosm.config import Conf
    from pyrosm.utils import (
        validate_custom_filter,
        validate_osm_keys,
        validate_tags_as_columns,
    )

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
    else:
        validate_tags_as_columns(tags_as_columns)
    if extra_attributes is not None:
        validate_tags_as_columns(extra_attributes)
        tags_as_columns = list(tags_as_columns) + list(extra_attributes)
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
    extra_attributes=None,
    nodes=False,
    custom_filter=None,
    filter_type="exclude",
    tags_to_keep=None,
    bounding_box=None,
    workers=None,
    keep_metadata=True,
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
    second pass over the file), matching ``OSM(...).get_network(nodes=True)``."""
    from pyrosm.config import Conf
    from pyrosm.utils import validate_custom_filter, validate_tags_as_columns

    tags_as_columns = list(Conf.tags.highway)
    if tags_to_keep is not None:
        validate_tags_as_columns(tags_to_keep)
        tags_as_columns = list(tags_to_keep)
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
    if extra_attributes is not None:
        validate_tags_as_columns(extra_attributes)
        tags_as_columns = tags_as_columns + list(extra_attributes)

    data_filter = (
        None if network_filter is None else parse_custom_filter(network_filter)[0]
    )
    # Networks always select highway ways (the filter values may reference other keys).
    filter_spec = (["highway"], data_filter, filter_type)
    bounding_box = _normalize_bounding_box(bounding_box)
    bounds = _bbox_bounds(bounding_box)

    def run(shard_paths):
        edges, node_gdf = _assemble_network(
            shard_paths,
            tags_as_columns,
            keep_metadata,
            filter_spec,
            nodes,
            bounding_box,
            filepath=filepath,
        )
        return (node_gdf, edges) if nodes else edges

    return _decode_and_run(
        filepath, [b"highway"], False, workers, run, bbox_bounds=bounds
    )
