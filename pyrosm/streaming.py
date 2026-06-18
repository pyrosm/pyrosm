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
match the in-memory reader exactly.

Phase 4a adds full tag + metadata column parity: each matching way is spilled as a
pyrosm-shaped way record (its resolved tag dict + ``version`` / ``timestamp`` / ``visible``
metadata), and assembly runs those records through pyrosm's own tag-explosion converters
(``explode_way_tags`` + ``way_records_to_arrays``), so every occurring tag becomes its own
column, the rest land in the JSON ``tags`` column, and ``keep_metadata`` is honoured --
column-for-column the in-memory reader's output. Later phases add the other layers /
filters, the network path, bounding boxes, history and the disk-backed coordinate join.
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
from pyrosm._arrays import way_records_to_arrays
from pyrosm.tagparser import explode_way_tags

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
    """Select the ways tagged ``building=*`` and return, per matching way, its node-ref
    slice, full (resolved) tag dict and ``version``/``timestamp``/``visible`` metadata --
    everything pyrosm's way record carries. Returns a dict of parallel arrays/lists, or
    ``None``."""
    if ways is None or _BUILDING not in string_table:
        return None
    building_idx = string_table.index(_BUILDING)
    keys, vals, tags_off = ways["keys"], ways["vals"], ways["tags_off"]
    key_positions = np.nonzero(keys == building_idx)[0]
    if len(key_positions) == 0:
        return None
    # A tag key belongs to the way whose [tags_off[i], tags_off[i+1]) slice contains it.
    way_index = np.searchsorted(tags_off, key_positions, side="right") - 1
    refs, refs_off = ways["refs"], ways["refs_off"]
    return {
        "id": ways["id"][way_index],
        "refs": [refs[refs_off[i] : refs_off[i + 1]] for i in way_index],
        "tags": [
            _resolve_tags(string_table, keys, vals, tags_off[i], tags_off[i + 1])
            for i in way_index
        ],
        "version": ways["version"][way_index],
        "timestamp": ways["timestamp"][way_index],
        "visible": ways["visible"][way_index],
    }


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
    id/type/role arrays, full tag dict and ``version``/``timestamp``/``changeset``
    metadata -- everything pyrosm needs to assemble the multipolygon and its columns.
    Member roles and tags are resolved through the block's string table."""
    if relations is None or _BUILDING not in string_table:
        return
    building_idx = string_table.index(_BUILDING)
    key_positions = np.nonzero(relations["keys"] == building_idx)[0]
    if len(key_positions) == 0:
        return
    tags_off = relations["tags_off"]
    rel_index = np.unique(np.searchsorted(tags_off, key_positions, side="right") - 1)
    ids, keys, vals = relations["id"], relations["keys"], relations["vals"]
    version, timestamp, changeset = (
        relations["version"],
        relations["timestamp"],
        relations["changeset"],
    )
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
        yield {
            "id": ids[i],
            "memid": memids[s:e],
            "memtype": types[s:e],
            "memrole": member_role,
            "tags": tags,
            "version": version[i],
            "timestamp": timestamp[i],
            "changeset": changeset[i],
        }


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


def _object_array(items):
    """1-D object array of ``items`` (dicts/arrays), avoiding numpy's 2-D coercion."""
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _decode_batch(task):
    """Worker: decode a contiguous run of blobs and spill one shard with the node
    coordinates, the matching building ways (refs + resolved tags + metadata), *all* ways
    (id + refs, for relation-member lookup) and the building relations, then return the
    shard path."""
    worker_id, blobs = task
    node_id, node_lon, node_lat = [], [], []
    bld_id, bld_refs, bld_tags = [], [], []
    bld_version, bld_timestamp, bld_visible = [], [], []
    all_id, all_refs, all_count = [], [], []
    rel = {k: [] for k in ("id", "memid", "memtype", "memrole", "tags", "meta")}
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
                    bld_id.append(found["id"])
                    bld_refs.extend(found["refs"])
                    bld_tags.extend(found["tags"])
                    bld_version.append(found["version"])
                    bld_timestamp.append(found["timestamp"])
                    bld_visible.append(found["visible"])
            for r in _building_relations(string_table, relations):
                rel["id"].append(r["id"])
                rel["memid"].append(r["memid"])
                rel["memtype"].append(r["memtype"])
                rel["memrole"].append(r["memrole"])
                rel["tags"].append(r["tags"])
                rel["meta"].append((r["version"], r["timestamp"], r["changeset"]))

    path = os.path.join(_SHARD_DIR, "shard_%d.npz" % worker_id)
    np.savez(
        path,
        node_id=np.concatenate(node_id) if node_id else np.empty(0, np.int64),
        node_lon=np.concatenate(node_lon) if node_lon else np.empty(0),
        node_lat=np.concatenate(node_lat) if node_lat else np.empty(0),
        way_id=np.concatenate(bld_id) if bld_id else np.empty(0, np.int64),
        refs=np.concatenate(bld_refs) if bld_refs else np.empty(0, np.int64),
        refs_off=_offsets_from_lengths([len(r) for r in bld_refs]),
        way_tags=_object_array(bld_tags),
        way_version=(
            np.concatenate(bld_version) if bld_version else np.empty(0, np.int64)
        ),
        way_timestamp=(
            np.concatenate(bld_timestamp) if bld_timestamp else np.empty(0, np.int64)
        ),
        way_visible=(
            np.concatenate(bld_visible) if bld_visible else np.empty(0, np.int64)
        ),
        all_id=np.concatenate(all_id) if all_id else np.empty(0, np.int64),
        all_refs=np.concatenate(all_refs) if all_refs else np.empty(0, np.int64),
        all_refs_off=_offsets_from_lengths(
            np.concatenate(all_count).tolist() if all_count else []
        ),
        rel_id=np.array(rel["id"], dtype=np.int64),
        rel_memid=(
            np.concatenate(rel["memid"]) if rel["memid"] else np.empty(0, np.int64)
        ),
        rel_memoff=_offsets_from_lengths([len(m) for m in rel["memid"]]),
        rel_memtype=(
            np.concatenate(rel["memtype"]) if rel["memtype"] else np.empty(0, np.int64)
        ),
        rel_memrole=(
            np.concatenate(rel["memrole"])
            if rel["memrole"]
            else np.empty(0, dtype=object)
        ),
        rel_tags=_object_array(rel["tags"]),
        rel_meta=np.array(rel["meta"], dtype=np.int64).reshape(-1, 3),
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
    """Read the spilled building ways back as pyrosm-shaped way records (``id``,
    ``version``, ``timestamp``, ``visible``, ``nodes``, ``tags``)."""
    records = []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        wid = z["way_id"]
        if len(wid) == 0:
            continue
        off, refs, tags = z["refs_off"], z["refs"], z["way_tags"]
        ver, ts, vis = z["way_version"], z["way_timestamp"], z["way_visible"]
        for i in range(len(wid)):
            records.append(
                {
                    "id": int(wid[i]),
                    "version": int(ver[i]),
                    "timestamp": int(ts[i]),
                    "visible": bool(vis[i]),
                    "nodes": refs[off[i] : off[i + 1]].tolist(),
                    "tags": tags[i],
                }
            )
    return records


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
    """The standalone building way records to output, after dropping the ones that are
    members of a building relation (pyrosm assigns those to the relation, so they are not
    standalone way rows)."""
    records = _collect_building_ways(shard_paths)
    if not records:
        return None
    if len(exclude_ids):
        exclude = set(exclude_ids.tolist())
        records = [r for r in records if r["id"] not in exclude]
        if not records:
            return None
    return records


def _load_building_relations(shard_paths):
    """Reassemble the building relations spilled across shards into the ``relations``
    struct pyrosm's assembly expects (``id`` / ``members`` / ``tags``), and return it
    together with the unique set of all their member ids. ``(None, empty)`` if there are
    no building relations."""
    ids, members, tags, meta = [], [], [], []
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
        meta.append(z["rel_meta"])
    if not ids:
        return None, np.empty(0, np.int64)
    meta = np.concatenate(meta)
    relations = {
        "id": np.concatenate(ids),
        "members": _object_array(members),
        "tags": _object_array(tags),
        "version": meta[:, 0],
        "timestamp": meta[:, 1],
        "changeset": meta[:, 2],
    }
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


def _ways_arrays(records, keep_metadata):
    """Convert way records into the ``way_elements`` dict (all occurring tag columns + the
    JSON ``tags`` column + metadata) pyrosm's assembly expects, reusing pyrosm's own
    tag-explosion so the columns match the in-memory reader. Consumes (explodes) the
    records. The augmentation mirrors ``get_osm_ways_and_relations``."""
    from pyrosm.config import Conf

    tags_as_columns = list(Conf.tags.building) + ["id", "nodes"]
    if keep_metadata:
        tags_as_columns += ["timestamp", "version"]
    return way_records_to_arrays(explode_way_tags(records), tags_as_columns)


def _needed_node_ids(kept, relation_ways):
    """The unique node ids referenced by the kept standalone ways and the relation
    member ways -- the only coordinates the gather has to pull off disk."""
    refs = []
    if kept is not None:
        refs.extend(r["nodes"] for r in kept)
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
    gathered = _gather_buildings(shard_paths)
    if gathered is None:
        return None
    kept, relations, relation_ways, node_coordinates = gathered
    return _assemble_chunk(
        node_coordinates, kept, relations, relation_ways, keep_metadata
    )


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


def _union_schema(tables):
    """One arrow schema holding the union of these tables' fields, with the GeoParquet
    ``geo`` metadata preserved -- so chunks with different optional columns (way rows have
    ``visible``, relation rows have ``changeset``) can share one parquet schema."""
    import pyarrow as pa

    schema = pa.unify_schemas([t.schema for t in tables], promote_options="permissive")
    for t in tables:
        if t.schema.metadata and b"geo" in t.schema.metadata:
            return schema.with_metadata(t.schema.metadata)
    return schema


def _write_tables(output, tables, schema):
    """Append each arrow table as a row group to a single GeoParquet at ``output``, each
    aligned to ``schema``. Returns the path, or ``None`` if nothing was written."""
    import pyarrow.parquet as pq

    writer = None
    try:
        for table in tables:
            if table is None or table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(output, schema)
            writer.write_table(_align_table(table, schema))
    finally:
        if writer is not None:
            writer.close()
    return output if writer is not None else None


def _stream_buildings_to_parquet(shard_paths, output, chunk_size, keep_metadata):
    """Stream the buildings (ways in chunks, then relations) to a chunked GeoParquet at
    ``output``, so the frame is never fully materialised. Returns the path, or ``None`` if
    there were no buildings to write."""
    from geopandas.io.arrow import _geopandas_to_arrow

    gathered = _gather_buildings(shard_paths)
    if gathered is None:
        return None
    kept, relations, relation_ways, node_coordinates = gathered

    def to_table(gdf):
        return _geopandas_to_arrow(gdf, index=False, geometry_encoding="WKB")

    def way_tables():
        if kept is None:
            return
        for start in range(0, len(kept), chunk_size):
            gdf = _assemble_chunk(
                node_coordinates,
                kept[start : start + chunk_size],
                None,
                None,
                keep_metadata,
            )
            if gdf is not None and len(gdf) > 0:
                yield to_table(gdf)

    relation_table = None
    if relations is not None:
        rgdf = _assemble_chunk(
            node_coordinates, None, relations, relation_ways, keep_metadata
        )
        if rgdf is not None and len(rgdf) > 0:
            relation_table = to_table(rgdf)

    # The union schema must cover both the way and relation row shapes; peek the first way
    # chunk so it can be unified with the relation rows before the first write.
    way_gen = way_tables()
    first_way = next(way_gen, None)
    samples = [t for t in (first_way, relation_table) if t is not None]
    if not samples:
        return None
    schema = _union_schema(samples)

    def all_tables():
        if first_way is not None:
            yield first_way
        yield from way_gen
        if relation_table is not None:
            yield relation_table

    return _write_tables(output, all_tables(), schema)


def get_buildings(filepath, workers=None, output=None, keep_metadata=True):
    """Read building geometries from ``filepath``.

    Returns the building ways and relations with the same columns as the in-memory reader:
    every occurring ``Conf.tags.building`` tag as its own column, the remaining tags as a
    JSON ``tags`` column, and -- when ``keep_metadata`` -- the ``version`` / ``timestamp``
    / ``changeset`` / ``visible`` element metadata.

    With ``output=None`` (the default) returns an in-memory GeoDataFrame. With ``output``
    set to a path, the buildings are streamed to a GeoParquet file in chunks (never fully
    materialised) and the path is returned; this needs the optional ``pyarrow`` dependency.

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
            return _assemble_buildings(shard_paths, keep_metadata)
        return _stream_buildings_to_parquet(
            shard_paths, output, _OUTPUT_CHUNK_SIZE, keep_metadata
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
