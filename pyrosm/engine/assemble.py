"""Assemble collected way/relation records into a GeoDataFrame through pyrosm's own tag
and geometry pipeline, so the output matches the in-memory reader column-for-column."""

from pyrosm.engine.collect import _ways_arrays, _collect_buildings


def _assemble_chunk(
    node_coordinates, way_records, relations, relation_ways, keep_metadata
):
    """Build one GeoDataFrame from way and/or relation elements using pyrosm's own tag +
    geometry pipeline (full columns, missing-node handling, polygon/linestring typing,
    ring assembly, dropna, orientation), so the result matches the in-memory reader
    exactly."""
    from pyrosm.config import Conf
    from pyrosm.frames import prepare_geodataframe

    ways = _ways_arrays(way_records, keep_metadata) if way_records else None
    gdf = prepare_geodataframe(
        None,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        list(Conf.tags.building),
        None,
        keep_metadata=keep_metadata,
    )
    if gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
    return gdf


def _assemble_buildings(shard_paths, keep_metadata):
    """Assemble all building ways and relations into a single in-memory GeoDataFrame."""
    collected = _collect_buildings(shard_paths)
    if collected is None:
        return None
    kept, relations, relation_ways, node_coordinates = collected
    return _assemble_chunk(
        node_coordinates, kept, relations, relation_ways, keep_metadata
    )
