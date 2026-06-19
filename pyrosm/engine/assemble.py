"""Assemble collected node/way/relation records into a GeoDataFrame through pyrosm's own
tag and geometry pipeline, so the output matches the in-memory reader column-for-column.
"""

import numpy as np

from pyrosm.data_filter import element_should_be_kept
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
):
    """Build one GeoDataFrame from node, way and/or relation elements using pyrosm's own
    tag + geometry pipeline (full columns, missing-node handling, polygon/linestring
    typing, ring assembly, dropna, orientation, the ``bounding_box`` spatial filter), so the
    result matches the in-memory reader exactly."""
    from pyrosm.frames import prepare_geodataframe

    ways = (
        _ways_arrays(way_records, tags_as_columns, keep_metadata)
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
    if gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
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
    )


def _assemble_network(
    shard_paths,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    segments,
    bounding_box,
    filepath=None,
):
    """Assemble the matching highway ways as a network (LineString edges + a ``length``
    column) through pyrosm's ``parse_network`` path. Returns ``(edges, nodes)``; ``nodes``
    is the graph-export node frame, only built when ``segments`` is True
    (``get_network(nodes=True)``), which needs the node tags + metadata, so the coordinate
    store is gathered with a second pass over ``filepath`` rather than the lean shard lookup.
    A ``None`` ``data_filter`` (network types ``all`` / ``driving_psv``) keeps every highway
    way."""
    from pyrosm.frames import prepare_geodataframe

    osm_keys, data_filter, filter_type = filter_spec

    def keep(tag):
        return data_filter is None or element_should_be_kept(
            tag, osm_keys, data_filter, filter_type
        )

    in_box = _in_box_nodes(shard_paths) if bounding_box is not None else None
    kept = _collect_kept_ways(shard_paths, np.empty(0, np.int64), keep, in_box)
    if kept is None:
        return None, None
    needed = _needed_node_ids(kept, None)
    if segments:
        node_coordinates = _gather_node_records(filepath, needed, keep_metadata)
    else:
        node_coordinates = _node_lookup(shard_paths, needed)
    ways = _ways_arrays(kept, tags_as_columns, keep_metadata)
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
    # The per-way 'nodes' list is dropped by default (it breaks file export), matching
    # OSM.get_network with the default keep_node_info=False.
    if edges is not None and "nodes" in edges.columns:
        edges = edges.drop(columns=["nodes"])
    return edges, node_gdf
