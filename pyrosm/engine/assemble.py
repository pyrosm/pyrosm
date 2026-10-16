"""Assemble collected node/way/relation records into a GeoDataFrame through pyrosm's own
tag and geometry pipeline, so the output matches the in-memory reader column-for-column.
"""

import numpy as np

from pyrosm.engine.bounding_box import _in_box_nodes
from pyrosm.engine.collect import (
    _ways_arrays,
    _collect_layer,
    _collect_kept_ways,
    _node_lookup,
    _gather_node_records,
    _needed_node_ids,
)


def _assemble_chunk(
    node_coordinates,
    way_records,
    relations,
    relation_ways,
    tags_as_columns,
    keep_metadata,
    nodes=None,
    bounding_box=None,
    complete_relations=False,
    keep_other_tags=True,
    keep_node_info=False,
    way_columns=None,
):
    """Build one GeoDataFrame from node, way and/or relation elements using pyrosm's own
    tag + geometry pipeline (full columns, missing-node handling, polygon/linestring
    typing, ring assembly, dropna, orientation, the ``bounding_box`` spatial filter), so the
    result matches the in-memory reader exactly. The ways' ``nodes`` column is kept only
    with ``keep_node_info``; ``way_columns`` are passed to :func:`_ways_arrays`."""
    from pyrosm.frames import prepare_geodataframe

    ways = (
        _ways_arrays(
            way_records, tags_as_columns, keep_metadata, keep_other_tags, way_columns
        )
        if way_records
        else None
    )
    gdf = prepare_geodataframe(
        nodes,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        list(tags_as_columns),
        bounding_box,
        keep_metadata=keep_metadata,
        complete_relations=complete_relations,
    )
    if not keep_node_info and gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
    # keep_other_tags=False (minimal-tags mode): drop the JSON 'tags' column of leftover tags
    # so the result holds only the requested tags_as_columns. _ways_arrays already skips
    # building it for ways; this also covers a 'tags' column that a relation chunk produced.
    if not keep_other_tags and gdf is not None and "tags" in gdf.columns:
        gdf = gdf.drop(columns=["tags"])
    return gdf


def _assemble_layer(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    keep_ways=True,
    keep_relations=True,
    bounding_box=None,
    complete_relations=False,
    keep_other_tags=True,
    keep_node_info=False,
):
    """Assemble all matching nodes, ways and relations into one in-memory GeoDataFrame."""
    collected = _collect_layer(
        shard_paths,
        tags_as_columns,
        keep_metadata,
        filter_spec,
        keep_ways,
        keep_relations,
        bounding_box,
        complete_relations,
    )
    if collected is None:
        return None
    node_features, kept, relations, relation_ways, node_coordinates = collected
    return _assemble_chunk(
        node_coordinates,
        kept,
        relations,
        relation_ways,
        tags_as_columns,
        keep_metadata,
        nodes=node_features,
        bounding_box=bounding_box,
        complete_relations=complete_relations,
        keep_other_tags=keep_other_tags,
        keep_node_info=keep_node_info,
    )


def _network_ways(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    segments,
    bounding_box,
    filepath,
    progress,
):
    """The highway ways a network read keeps, as way arrays, and the coordinate store of
    their nodes: ``(ways, node_coordinates)``, or ``None`` when no way is kept. With
    ``segments`` the store holds the node records (tags + metadata), gathered with a
    second pass over ``filepath`` that ``progress`` reports (see
    :func:`~pyrosm.engine.collect._gather_node_records`); otherwise only the coordinates.
    A ``None`` ``data_filter`` (network types ``all`` / ``driving_psv``) keeps every
    highway way."""
    from pyrosm.engine.collect import _keep_fn

    keep = _keep_fn(filter_spec)
    in_box = _in_box_nodes(shard_paths) if bounding_box is not None else None
    kept = _collect_kept_ways(shard_paths, np.empty(0, np.int64), keep, in_box)
    if kept is None:
        return None
    needed = _needed_node_ids(kept, None)
    if segments:
        node_coordinates = _gather_node_records(
            filepath, needed, keep_metadata, progress
        )
    else:
        node_coordinates = _node_lookup(shard_paths, needed)
    return _ways_arrays(kept, tags_as_columns, keep_metadata), node_coordinates


def _assemble_network(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    segments,
    bounding_box,
    filepath=None,
    progress=None,
    keep_node_info=False,
):
    """Assemble the matching highway ways as a network (LineString edges + a ``length``
    column) through pyrosm's ``parse_network`` path. Returns ``(edges, nodes)``; ``nodes``
    is the graph-export node frame, only built when ``segments`` is True
    (``get_network(nodes=True)``), from the node records of :func:`_network_ways`."""
    from pyrosm.frames import prepare_geodataframe

    read = _network_ways(
        shard_paths,
        tags_as_columns,
        keep_metadata,
        filter_spec,
        segments,
        bounding_box,
        filepath,
        progress,
    )
    if read is None:
        return None, None
    ways, node_coordinates = read
    edges, node_gdf = prepare_geodataframe(
        None,
        node_coordinates,
        ways,
        None,
        None,
        list(tags_as_columns),
        bounding_box,
        parse_network=True,
        calculate_seg_lengths=segments,
        keep_metadata=keep_metadata,
    )
    # The per-way 'nodes' list is dropped by default (it breaks file export), as
    # OSM.get_network does unless keep_node_info=True.
    if not keep_node_info and edges is not None and "nodes" in edges.columns:
        edges = edges.drop(columns=["nodes"])
    return edges, node_gdf


def _assemble_graph_network(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    bounding_box,
    filepath,
    progress,
    keep_node_info,
):
    """The network of :func:`_assemble_network` with ``segments`` as the readers of graph
    simplification, ``(nodes, edges)``: a :class:`~pyrosm.graph_simplify._StoreNodes` and
    a :class:`~pyrosm.graph_simplify._WayEdges` that read the node and edge tables of
    ``get_network(nodes=True)`` without building them; ``(None, None)`` when there are no
    edges."""
    from shapely import box, intersects
    from pyrosm.engine.cache import missing_as_nan
    from pyrosm.engine.readers import _node_lists
    from pyrosm.frames import create_df
    from pyrosm.geometry import create_network_segments, segment_lines
    from pyrosm.graph_simplify import _StoreNodes, _WayEdges

    read = _network_ways(
        shard_paths,
        tags_as_columns,
        keep_metadata,
        filter_spec,
        True,
        bounding_box,
        filepath,
        progress,
    )
    if read is None:
        return None, None
    ways, store = read
    ways, segment_counts, start, end, u, v, node_ids, node_rows = (
        create_network_segments(store, ways)
    )
    way_of = np.repeat(np.arange(len(segment_counts)), segment_counts)
    if bounding_box is not None:
        if isinstance(bounding_box, list):
            bounding_box = box(*bounding_box)
        inside = intersects(segment_lines(start, end), bounding_box)
        way_of, u, v, start, end = (a[inside] for a in (way_of, u, v, start, end))
        referenced = np.isin(node_ids, np.concatenate([u, v]))
        node_ids, node_rows = node_ids[referenced], node_rows[referenced]
    if len(way_of) == 0:
        return None, None
    ways = create_df(ways).assign(osm_type="way")
    if not keep_node_info and "nodes" in ways.columns:
        ways = ways.drop(columns=["nodes"])
    ways = _node_lists(missing_as_nan(ways))
    return _StoreNodes(store, node_ids, node_rows), _WayEdges(
        ways, way_of, u, v, start, end
    )
