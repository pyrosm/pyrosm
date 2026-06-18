"""Streaming, single-pass PBF reader (the ``engine="streaming"`` backend).

The file is read in one pass: blobs are decoded in parallel with the raw Cython
``primitive_block_decoder`` (protobuf is used only for the small ``BlobHeader`` /
``Blob`` framing), each worker spills the node coordinates and the building ways it
finds to a per-worker shard on disk, and the main process then gathers only the
coordinates the kept ways reference and assembles the geometries vectorised. Peak
memory is bounded by the working set rather than the whole file.

Phase 1 covers way-based buildings; Phase 2 adds an optional ``output`` path: when given,
the buildings are assembled in chunks and appended to a single GeoParquet file (one row
group per chunk) instead of one in-memory frame, so the output is never fully
materialised. ``pyarrow`` is an optional dependency, required only for ``output``.

Phase 3 adds building *relations* (multipolygons). Because a relation's member ways can
live in different blocks than the relation, each worker also spills *every* way it decodes
(id + node refs) to its shard; the main process then looks up the member ways of the
building relations by id from those shards -- the same disk-lookup the node coordinates
use -- so relations need no PBF re-read. The relations and their member ways are handed to
pyrosm's own ``prepare_geodataframe`` so the assembled multipolygons (roles, ring logic)
match the in-memory reader exactly. Later phases add the other layers, tag columns,
history and the disk-backed coordinate join.
"""

import os
import struct
import zlib
import tempfile
import shutil
from multiprocessing import Pool

import numpy as np

from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.primitive_block_decoder import decode_primitive_block

# Below this many data blobs the multiprocessing overhead is not worth it, so the read
# runs in-process (1 worker). Small bundled extracts therefore decode without a pool.
_SMALL_FILE_BLOBS = 16

_BUILDING = b"building"
# Relation.MemberType -> the byte labels pyrosm's relation assembly expects.
_MEMBER_TYPE = {0: b"node", 1: b"way", 2: b"relation"}

# When streaming to GeoParquet, assemble and write this many ways per chunk so the
# output frame is never fully materialised.
_OUTPUT_CHUNK_SIZE = 250_000

# Per-worker globals, set by the pool initializer (or directly for the in-process path).
_FILEPATH = None
_SHARD_DIR = None


def _index_blobs(filepath):
    """One cheap sequential pass: ``(type, data_offset, data_size)`` per blob, reading
    only the ``BlobHeader``s and skipping the payloads (no decompression)."""
    blobs = []
    with open(filepath, "rb") as f:
        while True:
            header_len_bytes = f.read(4)
            if len(header_len_bytes) < 4:
                break
            (header_len,) = struct.unpack("!L", header_len_bytes)
            header = BlobHeader()
            header.ParseFromString(f.read(header_len))
            offset = f.tell()
            blobs.append((header.type, offset, header.datasize))
            f.seek(header.datasize, os.SEEK_CUR)
    return blobs


def _read_block(f, offset, size):
    """Read and decompress one ``Blob`` payload into the raw ``PrimitiveBlock`` bytes."""
    f.seek(offset)
    blob = Blob()
    blob.ParseFromString(f.read(size))
    if blob.HasField("zlib_data"):
        return zlib.decompress(blob.zlib_data)
    if blob.HasField("raw"):
        return blob.raw
    if blob.HasField("lzma_data"):
        import lzma

        return lzma.decompress(blob.lzma_data)
    raise ValueError("Unsupported Blob compression in '%s'." % _FILEPATH)


def _building_ways(string_table, ways):
    """Select the ways tagged ``building=*``: their ids, node-ref slices and the
    building value. Returns ``(ids, refs_list, values)`` or ``None``."""
    if ways is None or _BUILDING not in string_table:
        return None
    building_idx = string_table.index(_BUILDING)
    keys = ways["keys"]
    key_positions = np.nonzero(keys == building_idx)[0]
    if len(key_positions) == 0:
        return None
    # A tag key belongs to the way whose [tags_off[i], tags_off[i+1]) slice contains it.
    way_index = np.searchsorted(ways["tags_off"], key_positions, side="right") - 1
    refs, refs_off, vals, ids = (
        ways["refs"],
        ways["refs_off"],
        ways["vals"],
        ways["id"],
    )
    ref_slices = [refs[refs_off[i] : refs_off[i + 1]] for i in way_index]
    values = [string_table[vals[p]].decode("utf-8", "replace") for p in key_positions]
    return ids[way_index], ref_slices, values


def _resolve_tags(string_table, keys, vals, start, end):
    """The ``{key: value}`` tag dict for one element, resolved through the string table
    (decoded to str, as pyrosm's protobuf path produces)."""
    return {
        string_table[keys[p]]
        .decode("utf-8", "replace"): string_table[vals[p]]
        .decode("utf-8", "replace")
        for p in range(start, end)
    }


def _building_relations(string_table, relations):
    """The ``building=*`` relations in this block. Yields, per relation, its id, member
    id/type/role arrays and full tag dict -- everything pyrosm needs to assemble the
    multipolygon. Member roles and tags are resolved through the block's string table.
    """
    if relations is None or _BUILDING not in string_table:
        return
    building_idx = string_table.index(_BUILDING)
    key_positions = np.nonzero(relations["keys"] == building_idx)[0]
    if len(key_positions) == 0:
        return
    tags_off = relations["tags_off"]
    rel_index = np.unique(np.searchsorted(tags_off, key_positions, side="right") - 1)
    ids, keys, vals = relations["id"], relations["keys"], relations["vals"]
    memids, types, roles, moff = (
        relations["memids"],
        relations["types"],
        relations["roles"],
        relations["members_off"],
    )
    for i in rel_index:
        s, e = moff[i], moff[i + 1]
        member_role = np.array(
            [string_table[r].decode("utf-8", "replace") for r in roles[s:e]],
            dtype=object,
        )
        tags = _resolve_tags(string_table, keys, vals, tags_off[i], tags_off[i + 1])
        yield ids[i], memids[s:e], types[s:e], member_role, tags


def _init_worker(filepath, shard_dir):
    global _FILEPATH, _SHARD_DIR
    _FILEPATH = filepath
    _SHARD_DIR = shard_dir


def _offsets_from_lengths(lengths):
    """CSR offsets array for variable-length rows: ``[0, l0, l0+l1, ...]``."""
    off = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        np.cumsum(lengths, out=off[1:])
    return off


def _decode_batch(task):
    """Worker: decode a contiguous run of blobs and spill one shard with the node
    coordinates, the building ways, *all* ways (id + refs, for relation-member lookup)
    and the building relations, then return the shard path."""
    worker_id, blobs = task
    node_id, node_lon, node_lat = [], [], []
    bld_id, bld_refs, bld_value = [], [], []
    all_id, all_refs, all_count = [], [], []
    rel_id, rel_memid, rel_memtype, rel_memrole, rel_tags = [], [], [], [], []
    with open(_FILEPATH, "rb") as f:
        for offset, size in blobs:
            data = _read_block(f, offset, size)
            string_table, header, nodes, ways, relations = decode_primitive_block(data)
            if nodes is not None:
                gran = header["granularity"]
                node_id.append(nodes["id"])
                node_lat.append((nodes["lat"] * gran + header["lat_offset"]) / 1e9)
                node_lon.append((nodes["lon"] * gran + header["lon_offset"]) / 1e9)
            if ways is not None:
                all_id.append(ways["id"])
                all_refs.append(ways["refs"])
                all_count.append(np.diff(ways["refs_off"]))
                found = _building_ways(string_table, ways)
                if found is not None:
                    ids, ref_slices, values = found
                    bld_id.append(ids)
                    bld_refs.extend(ref_slices)
                    bld_value.extend(values)
            for rid, memid, memtype, memrole, tags in _building_relations(
                string_table, relations
            ):
                rel_id.append(rid)
                rel_memid.append(memid)
                rel_memtype.append(memtype)
                rel_memrole.append(memrole)
                rel_tags.append(tags)

    path = os.path.join(_SHARD_DIR, "shard_%d.npz" % worker_id)
    np.savez(
        path,
        node_id=np.concatenate(node_id) if node_id else np.empty(0, np.int64),
        node_lon=np.concatenate(node_lon) if node_lon else np.empty(0),
        node_lat=np.concatenate(node_lat) if node_lat else np.empty(0),
        way_id=np.concatenate(bld_id) if bld_id else np.empty(0, np.int64),
        refs=np.concatenate(bld_refs) if bld_refs else np.empty(0, np.int64),
        refs_off=_offsets_from_lengths([len(r) for r in bld_refs]),
        value=np.array(bld_value, dtype=object),
        all_id=np.concatenate(all_id) if all_id else np.empty(0, np.int64),
        all_refs=np.concatenate(all_refs) if all_refs else np.empty(0, np.int64),
        all_refs_off=_offsets_from_lengths(
            np.concatenate(all_count).tolist() if all_count else []
        ),
        rel_id=np.array(rel_id, dtype=np.int64),
        rel_memid=np.concatenate(rel_memid) if rel_memid else np.empty(0, np.int64),
        rel_memoff=_offsets_from_lengths([len(m) for m in rel_memid]),
        rel_memtype=(
            np.concatenate(rel_memtype) if rel_memtype else np.empty(0, np.int64)
        ),
        rel_memrole=(
            np.concatenate(rel_memrole) if rel_memrole else np.empty(0, dtype=object)
        ),
        rel_tags=np.array(rel_tags, dtype=object),
    )
    return path


def _auto_workers(n_blobs):
    if n_blobs < _SMALL_FILE_BLOBS:
        return 1
    return min(os.cpu_count() or 1, n_blobs)


def _decode_all(filepath, blobs, workers, shard_dir):
    """Decode every data blob into per-worker shards; return the shard paths."""
    n = len(blobs)
    per = (n + workers - 1) // workers
    tasks = [
        (i, blobs[i * per : (i + 1) * per])
        for i in range(workers)
        if blobs[i * per : (i + 1) * per]
    ]
    if workers == 1:
        _init_worker(filepath, shard_dir)
        return [_decode_batch(tasks[0])] if tasks else []
    with Pool(
        workers, initializer=_init_worker, initargs=(filepath, shard_dir)
    ) as pool:
        return pool.map(_decode_batch, tasks)


def _collect_building_ways(shard_paths):
    """Read the spilled building ways back: ids, values and a node-ref array per way."""
    way_id, value, ref_slices = [], [], []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid, off, refs, val = z["way_id"], z["refs_off"], z["refs"], z["value"]
        for i in range(len(wid)):
            ref_slices.append(refs[off[i] : off[i + 1]])
        way_id.append(wid)
        value.append(val)
    way_id = np.concatenate(way_id) if way_id else np.empty(0, np.int64)
    value = np.concatenate(value) if value else np.empty(0, dtype=object)
    return way_id, value, ref_slices


def _node_lookup(shard_paths, needed):
    """Gather only the coordinates of ``needed`` node ids from the shards (bounded
    memory) and wrap them in a ``NodeLocations`` for geometry assembly."""
    import pandas as pd

    from pyrosm.node_lookup import NodeLocations

    lon = np.full(len(needed), np.nan)
    lat = np.full(len(needed), np.nan)
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        nid = z["node_id"]
        if len(nid) == 0:
            continue
        pos = np.clip(np.searchsorted(needed, nid), 0, len(needed) - 1)
        hit = needed[pos] == nid
        lon[pos[hit]] = z["node_lon"][hit]
        lat[pos[hit]] = z["node_lat"][hit]
    present = ~np.isnan(lon)
    coords = pd.DataFrame(
        {"id": needed[present], "lon": lon[present], "lat": lat[present]}
    )
    return NodeLocations(coords)


def _collect_kept_ways(shard_paths, exclude_ids):
    """The standalone building ways to output: ids, values and node-ref arrays, after
    dropping the ones that are members of a building relation (pyrosm assigns those to
    the relation, so they are not standalone way rows)."""
    way_id, value, ref_slices = _collect_building_ways(shard_paths)
    if len(way_id) == 0:
        return None
    if len(exclude_ids):
        keep = ~np.isin(way_id, exclude_ids)
        way_id = way_id[keep]
        value = value[keep]
        ref_slices = [r for r, k in zip(ref_slices, keep) if k]
        if len(way_id) == 0:
            return None
    return way_id, value, ref_slices


def _load_building_relations(shard_paths):
    """Reassemble the building relations spilled across shards into the ``relations``
    struct pyrosm's assembly expects (``id`` / ``members`` / ``tags``), and return it
    together with the unique set of all their member ids. ``(None, empty)`` if there are
    no building relations."""
    ids, members, tags = [], [], []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        rid = z["rel_id"]
        if len(rid) == 0:
            continue
        memid, memoff, memtype, memrole, rtags = (
            z["rel_memid"],
            z["rel_memoff"],
            z["rel_memtype"],
            z["rel_memrole"],
            z["rel_tags"],
        )
        for k in range(len(rid)):
            s, e = memoff[k], memoff[k + 1]
            members.append(
                {
                    "member_id": memid[s:e],
                    "member_type": np.array(
                        [_MEMBER_TYPE[int(t)] for t in memtype[s:e]], dtype=object
                    ),
                    "member_role": memrole[s:e],
                }
            )
            tags.append(rtags[k])
        ids.append(rid)
    if not ids:
        return None, np.empty(0, np.int64)
    members_arr = np.empty(len(members), dtype=object)
    members_arr[:] = members
    tags_arr = np.empty(len(tags), dtype=object)
    tags_arr[:] = tags
    relations = {"id": np.concatenate(ids), "members": members_arr, "tags": tags_arr}
    member_ids = np.unique(np.concatenate([m["member_id"] for m in members]))
    return relations, member_ids


def _gather_relation_ways(shard_paths, member_ids):
    """Look up the member ways (id -> node refs) of the building relations from the
    spilled all-ways store. Returned as a ``{id, nodes}`` dict sorted by id ascending --
    pyrosm aligns member roles to the sorted member ids, so the ways must match that
    order. ``None`` if none of the member ways are present."""
    found = {}
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid, off, refs = z["all_id"], z["all_refs_off"], z["all_refs"]
        if len(wid) == 0:
            continue
        pos = np.clip(np.searchsorted(member_ids, wid), 0, len(member_ids) - 1)
        for k in np.nonzero(member_ids[pos] == wid)[0]:
            found[int(wid[k])] = refs[off[k] : off[k + 1]]
    if not found:
        return None
    ids = np.array(sorted(found), dtype=np.int64)
    nodes = np.empty(len(ids), dtype=object)
    nodes[:] = [found[int(i)] for i in ids]
    return {"id": ids, "nodes": nodes}


def _ways_dict(way_id, value, ref_slices):
    """Pack way arrays into the ``way_elements`` dict pyrosm's assembly expects."""
    nodes = np.empty(len(ref_slices), dtype=object)
    nodes[:] = ref_slices
    return {"id": way_id, "nodes": nodes, "building": value}


def _needed_node_ids(kept, relation_ways):
    """The unique node ids referenced by the kept standalone ways and the relation
    member ways -- the only coordinates the gather has to pull off disk."""
    refs = []
    if kept is not None:
        refs.extend(kept[2])
    if relation_ways is not None:
        refs.extend(relation_ways["nodes"])
    if not refs:
        return np.empty(0, np.int64)
    return np.unique(np.concatenate(refs))


def _gather_buildings(shard_paths):
    """Shared gather for both output modes: standalone building ways, building relations,
    their member ways and the node coordinates all of them reference. ``None`` if there
    is nothing to assemble."""
    relations, member_ids = _load_building_relations(shard_paths)
    relation_ways = (
        _gather_relation_ways(shard_paths, member_ids)
        if relations is not None
        else None
    )
    # No member way is present (all outside the data) -> the relation cannot be assembled.
    if relation_ways is None:
        relations = None
    kept = _collect_kept_ways(shard_paths, member_ids)
    if kept is None and relations is None:
        return None
    node_coordinates = _node_lookup(shard_paths, _needed_node_ids(kept, relation_ways))
    return kept, relations, relation_ways, node_coordinates


def _assemble_chunk(node_coordinates, ways, relations=None, relation_ways=None):
    """Build one GeoDataFrame from way and/or relation elements using pyrosm's own
    geometry pipeline (missing-node handling, polygon/linestring typing, ring assembly,
    dropna, orientation), so the result matches the in-memory reader exactly."""
    from pyrosm.frames import prepare_geodataframe

    gdf = prepare_geodataframe(
        None,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        ["building"],
        None,
        keep_metadata=False,
    )
    if gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
    return gdf


def _assemble_buildings(shard_paths):
    """Assemble all building ways and relations into a single in-memory GeoDataFrame."""
    gathered = _gather_buildings(shard_paths)
    if gathered is None:
        return None
    kept, relations, relation_ways, node_coordinates = gathered
    ways = _ways_dict(*kept) if kept is not None else None
    return _assemble_chunk(node_coordinates, ways, relations, relation_ways)


def _require_pyarrow():
    """``output`` writes GeoParquet, which needs the optional ``pyarrow`` dependency."""
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Writing to GeoParquet (output=...) requires the optional 'pyarrow' "
            "dependency. Install it with `pip install pyarrow`."
        ) from e


def _align_table(table, schema):
    """Reorder ``table`` to ``schema``'s field order, adding any columns it lacks as typed
    nulls -- so heterogeneous chunks (way rows lack the relation-only ``tags`` column)
    share one parquet schema."""
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


def _write_geoparquet(output, gdfs, schema_gdf=None):
    """Append each non-empty GeoDataFrame in ``gdfs`` as a row group to a single
    GeoParquet at ``output``, so the output frame is never fully materialised. All row
    groups share one schema: when ``schema_gdf`` is given (the column superset -- the
    relation rows, which carry the ``tags`` column) it defines the schema and the way
    chunks are aligned to it. Returns the path, or ``None`` if nothing was written."""
    import pyarrow.parquet as pq
    from geopandas.io.arrow import _geopandas_to_arrow

    schema = None
    if schema_gdf is not None and len(schema_gdf) > 0:
        schema = _geopandas_to_arrow(
            schema_gdf, index=False, geometry_encoding="WKB"
        ).schema

    writer = None
    try:
        for gdf in gdfs:
            if gdf is None or len(gdf) == 0:
                continue
            table = _geopandas_to_arrow(gdf, index=False, geometry_encoding="WKB")
            if schema is None:
                schema = table.schema
            table = _align_table(table, schema)
            if writer is None:
                writer = pq.ParquetWriter(output, schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return output if writer is not None else None


def _stream_buildings_to_parquet(shard_paths, output, chunk_size):
    """Stream the buildings (ways then relations) to a chunked GeoParquet at ``output``.
    Returns the path, or ``None`` if there were no buildings to write."""
    gathered = _gather_buildings(shard_paths)
    if gathered is None:
        return None
    kept, relations, relation_ways, node_coordinates = gathered

    # Relation rows carry the column superset (the 'tags' column); assemble them once (they
    # are few) so their schema can pin every row group, and the way chunks align to it.
    relation_gdf = (
        _assemble_chunk(node_coordinates, None, relations, relation_ways)
        if relations is not None
        else None
    )

    def chunks():
        if kept is not None:
            way_id, value, ref_slices = kept
            for start in range(0, len(way_id), chunk_size):
                sl = slice(start, start + chunk_size)
                yield _assemble_chunk(
                    node_coordinates, _ways_dict(way_id[sl], value[sl], ref_slices[sl])
                )
        if relation_gdf is not None and len(relation_gdf) > 0:
            yield relation_gdf

    return _write_geoparquet(output, chunks(), schema_gdf=relation_gdf)


def get_buildings(filepath, workers=None, output=None):
    """Read building geometries from ``filepath``.

    With ``output=None`` (the default) returns an in-memory GeoDataFrame of the building
    ways and relations. With ``output`` set to a path, the buildings are streamed to a
    GeoParquet file in chunks (never fully materialised) and the path is returned; this
    needs the optional ``pyarrow`` dependency.

    ``workers`` defaults to one for small files (no multiprocessing overhead) and
    otherwise to a worker per CPU, bounded by the blob count.
    """
    if output is not None:
        _require_pyarrow()

    data_blobs = [
        (offset, size)
        for (blob_type, offset, size) in _index_blobs(filepath)
        if blob_type == "OSMData"
    ]
    if workers is None:
        workers = _auto_workers(len(data_blobs))

    shard_dir = tempfile.mkdtemp(prefix="pyrosm_stream_")
    try:
        shard_paths = _decode_all(filepath, data_blobs, workers, shard_dir)
        if output is None:
            return _assemble_buildings(shard_paths)
        return _stream_buildings_to_parquet(shard_paths, output, _OUTPUT_CHUNK_SIZE)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
