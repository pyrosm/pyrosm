"""Stream a layer to a GeoParquet file without materialising the whole output frame.

The chunks (the point nodes, then ways in batches, then the relations) are assembled one
at a time and written in that order by one writer, under the columns of the in-memory
reader's result in its order. Every way chunk carries the tag columns of all the ways, so a
tag that first occurs in a later chunk keeps its place. Needs the optional pyarrow
dependency.
"""

import tempfile
from pathlib import Path

from pyrosm.engine.collect import (
    _collect_layer,
    _num_ways,
    _result_tag_columns,
    _slice_way_columns,
)
from pyrosm.engine.assemble import _assemble_chunk
from pyrosm.engine.ways import ChunkContext, ElementStore, chunk_tables

# Assemble and write this many ways per chunk, so the output frame is never fully
# materialised.
_OUTPUT_CHUNK_SIZE = 250_000
# The point nodes and the relations are set aside on disk, and read back, in row groups of
# this many rows, so that neither is held in memory whole with a chunk of ways.
_SET_ASIDE_ROWS = 50_000


def _align_table(table, schema):
    """``table`` with ``schema``'s columns in its order: a column the table lacks as typed
    nulls, and a column of another type cast to the schema's (a column of Nones, which
    arrow types as null, to a string)."""
    import pyarrow as pa

    arrays = []
    for field in schema:
        i = table.schema.get_field_index(field.name)
        if i < 0:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        column = table.column(i)
        arrays.append(column if column.type == field.type else column.cast(field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _unify_schemas(schemas):
    """One arrow schema holding the union (by name) of these schemas' fields in order of
    appearance, a field of nulls (a tag column of Nones) typed as a string, with the
    GeoParquet ``geo`` metadata preserved."""
    import pyarrow as pa

    unified = pa.unify_schemas(schemas, promote_options="permissive")
    unified = pa.schema(
        [f.with_type(pa.large_string()) if f.type == pa.null() else f for f in unified],
        metadata=unified.metadata,
    )
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
    keep_node_info=False,
):
    """Stream the layer (point nodes, then ways in chunks, then relations) to a GeoParquet
    at ``output``, a few chunks in memory at a time (see
    :func:`~pyrosm.engine.ways.chunk_tables`). Returns the path, or ``None`` if there
    was nothing to write. ``keep_node_info`` keeps the ways' ``nodes`` column."""
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
    )
    if collected is None:
        return None
    node_features, kept, relations, relation_ways, node_coordinates = collected
    way_columns = (
        _result_tag_columns(kept, tags_as_columns, keep_metadata, keep_other_tags)
        if kept is not None
        else None
    )

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
            keep_node_info=keep_node_info,
            way_columns=way_columns,
        )
        if gdf is None or len(gdf) == 0:
            return None
        return _geopandas_to_arrow(gdf, index=False, geometry_encoding="WKB")

    def way_table(start):
        return to_table(way_records=_slice_way_columns(kept, start, start + chunk_size))

    with tempfile.TemporaryDirectory(
        prefix="pyrosm_ooc_parquet_", ignore_cleanup_errors=True
    ) as tmp:

        def set_aside(table, name):
            """Write ``table``, if any, to the temporary folder and return its path."""
            if table is None:
                return None
            path = Path(tmp) / (name + ".parquet")
            pq.write_table(table, path, row_group_size=_SET_ASIDE_ROWS)
            return path

        # The point nodes are written first and the relations last, but the columns of both
        # are needed before any way is written: assemble them now and set them aside.
        node_path = relation_path = None
        if node_features is not None:
            node_path = set_aside(to_table(nodes=node_features), "nodes")
        if relations is not None:
            relation_path = set_aside(
                to_table(relations=relations, relation_ways=relation_ways), "relations"
            )
        # The first way chunk that has rows.
        starts = iter(range(0, _num_ways(kept), chunk_size) if kept is not None else ())
        first_ways = None
        for start in starts:
            first_ways = way_table(start)
            if first_ways is not None:
                break
        # The columns of the nodes, the ways and the relations in order of appearance, as
        # the in-memory reader's concatenation of the three frames gives them.
        schemas = []
        if node_path is not None:
            schemas.append(pq.read_schema(node_path))
        if first_ways is not None:
            schemas.append(first_ways.schema)
        if relation_path is not None:
            schemas.append(pq.read_schema(relation_path))
        if not schemas:
            return None
        schema = _unify_schemas(schemas)
        writer = pq.ParquetWriter(output, schema)

        def write(table):
            if table is not None:
                writer.write_table(_align_table(table, schema))

        def write_set_aside(path):
            if path is not None:
                with pq.ParquetFile(path) as parquet_file:
                    for i in range(parquet_file.num_row_groups):
                        write(parquet_file.read_row_group(i))

        try:
            with writer:
                write_set_aside(node_path)
                write(first_ways)
                # A further way chunk is referenced only while it is written.
                first_ways = None
                if isinstance(kept, ElementStore):
                    ctx = ChunkContext(
                        kept,
                        tags_as_columns,
                        keep_metadata,
                        keep_other_tags,
                        way_columns,
                        node_coordinates,
                        keep_node_info,
                        bounding_box,
                    )
                    for table in chunk_tables(ctx, starts, chunk_size):
                        write(table)
                else:
                    for start in starts:
                        write(way_table(start))
                write_set_aside(relation_path)
        except BaseException:
            # A read that fails leaves no partial file.
            Path(output).unlink(missing_ok=True)
            raise
        return output
