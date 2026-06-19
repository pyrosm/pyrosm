"""Assemble collected node/way/relation records into a GeoDataFrame through pyrosm's own
tag and geometry pipeline, so the output matches the in-memory reader column-for-column.
"""

from pyrosm.engine.collect import _ways_arrays, _collect_layer


def _assemble_chunk(
    node_coordinates,
    way_records,
    relations,
    relation_ways,
    tags_as_columns,
    keep_metadata,
    nodes=None,
):
    """Build one GeoDataFrame from node, way and/or relation elements using pyrosm's own
    tag + geometry pipeline (full columns, missing-node handling, polygon/linestring
    typing, ring assembly, dropna, orientation), so the result matches the in-memory
    reader exactly."""
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
        None,
        keep_metadata=keep_metadata,
    )
    if gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
    return gdf


def _assemble_layer(
    shard_paths, tags_as_columns, keep_metadata, filter_spec, keep_ways, keep_relations
):
    """Assemble all matching nodes, ways and relations into one in-memory GeoDataFrame."""
    collected = _collect_layer(
        shard_paths,
        tags_as_columns,
        keep_metadata,
        filter_spec,
        keep_ways,
        keep_relations,
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
    )
