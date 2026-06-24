"""Stream a layer to a GeoParquet file without materialising the whole output frame.

Each chunk (the point nodes, then ways in batches, then the relations) is assembled and
written to its own temporary parquet file, so only one chunk is in memory at a time.
Because pyrosm builds a tag column only when that tag occurs in a chunk, different chunks
carry different columns; the temporary files are then combined under one schema that is the
union (by name) of every chunk's columns, so a tag column that first appears in a later
chunk is not dropped. Needs the optional pyarrow dependency.
"""

import shutil
import tempfile
from pathlib import Path

from pyrosm.engine.collect import _collect_layer, _num_ways, _slice_way_columns
from pyrosm.engine.assemble import _assemble_chunk

# Assemble and write this many ways per chunk, so the output frame is never fully
# materialised.
_OUTPUT_CHUNK_SIZE = 250_000


def _align_table(table, schema):
    """Reorder ``table`` to ``schema``'s field order, adding any columns it lacks as typed
    nulls -- so a chunk that materialised only a subset of the union's columns is widened
    to the full schema (rather than its extra-or-missing columns being dropped)."""
    import pyarrow as pa

    arrays = [
        (
            table.column(i)
            if (i := table.schema.get_field_index(field.name)) >= 0
            else pa.nulls(table.num_rows, type=field.type)
        )
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _unify_schemas(schemas):
    """One arrow schema holding the union (by name) of these schemas' fields, with the
    GeoParquet ``geo`` metadata preserved -- so the per-chunk parquet files, which can
    each carry a different subset of the tag columns, combine without dropping any."""
    import pyarrow as pa

    unified = pa.unify_schemas(schemas, promote_options="permissive")
    for schema in schemas:
        if schema.metadata and b"geo" in schema.metadata:
            return unified.with_metadata(schema.metadata)
    return unified


def _stream_layer_to_parquet(
    shard_paths,
    output,
    chunk_size,
    tags_as_columns,
    keep_metadata,
    filter_spec,
    keep_ways,
    keep_relations,
    bounding_box=None,
    complete_relations=False,
    keep_other_tags=True,
    workers=1,
):
    """Stream the layer (point nodes, then ways in chunks, then relations) to a GeoParquet
    at ``output``, spilling each chunk to its own temporary parquet file and then combining
    the files under the union of their schemas. Returns the path, or ``None`` if there was
    nothing to write. ``workers > 1`` runs the collect phase across a process pool."""
    import pyarrow.parquet as pq
    from geopandas.io.arrow import _geopandas_to_arrow

    collected = _collect_layer(
        shard_paths,
        tags_as_columns,
        keep_metadata,
        filter_spec,
        keep_ways,
        keep_relations,
        bounding_box,
        complete_relations,
        workers=workers,
    )
    if collected is None:
        return None
    node_features, kept, relations, relation_ways, node_coordinates = collected

    def to_table(way_records=None, relations=None, relation_ways=None, nodes=None):
        gdf = _assemble_chunk(
            node_coordinates,
            way_records,
            relations,
            relation_ways,
            tags_as_columns,
            keep_metadata,
            nodes=nodes,
            bounding_box=bounding_box,
            complete_relations=complete_relations,
            keep_other_tags=keep_other_tags,
        )
        if gdf is None or len(gdf) == 0:
            return None
        return _geopandas_to_arrow(gdf, index=False, geometry_encoding="WKB")

    def chunk_tables():
        if node_features is not None:
            table = to_table(nodes=node_features)
            if table is not None:
                yield table
        if kept is not None:
            for start in range(0, _num_ways(kept), chunk_size):
                table = to_table(
                    way_records=_slice_way_columns(kept, start, start + chunk_size)
                )
                if table is not None:
                    yield table
        if relations is not None:
            table = to_table(relations=relations, relation_ways=relation_ways)
            if table is not None:
                yield table

    part_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_parquet_")
    try:
        # Spill each chunk to its own parquet file (heterogeneous columns allowed).
        part_paths = []
        for table in chunk_tables():
            part_path = Path(part_dir) / ("part_%d.parquet" % len(part_paths))
            pq.write_table(table, part_path)
            part_paths.append(part_path)
        if not part_paths:
            return None
        # Combine the parts into the single output under the union of their schemas, one
        # part in memory at a time so the full frame is never materialised.
        schema = _unify_schemas([pq.read_schema(p) for p in part_paths])
        writer = pq.ParquetWriter(output, schema)
        try:
            for part_path in part_paths:
                writer.write_table(_align_table(pq.read_table(part_path), schema))
        finally:
            writer.close()
        return output
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)
