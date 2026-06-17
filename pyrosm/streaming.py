"""Streaming, single-pass PBF reader (the ``engine="streaming"`` backend).

The file is read in one pass: blobs are decoded in parallel with the raw Cython
``primitive_block_decoder`` (protobuf is used only for the small ``BlobHeader`` /
``Blob`` framing), each worker spills the node coordinates and the building ways it
finds to a per-worker shard on disk, and the main process then gathers only the
coordinates the kept ways reference and assembles the geometries vectorised. Peak
memory is bounded by the working set rather than the whole file.

Phase 1 covers way-based buildings; later phases add the other layers, relations, tag
columns, history and the disk-backed coordinate join.
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
_MEMBER_WAY = 1  # Relation.MemberType WAY (proto/osmformat.proto)

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


def _building_relation_member_ways(string_table, relations):
    """Member way ids of the ``building`` relations in this block. pyrosm assigns these
    ways to the relation geometry, so they must be dropped from the standalone way
    output to match ``get_buildings``' way rows."""
    if relations is None or _BUILDING not in string_table:
        return None
    building_idx = string_table.index(_BUILDING)
    key_positions = np.nonzero(relations["keys"] == building_idx)[0]
    if len(key_positions) == 0:
        return None
    rel_index = np.searchsorted(relations["tags_off"], key_positions, side="right") - 1
    memids, types, off = (
        relations["memids"],
        relations["types"],
        relations["members_off"],
    )
    member_ways = [
        memids[off[i] : off[i + 1]][types[off[i] : off[i + 1]] == _MEMBER_WAY]
        for i in rel_index
    ]
    return np.concatenate(member_ways) if member_ways else None


def _init_worker(filepath, shard_dir):
    global _FILEPATH, _SHARD_DIR
    _FILEPATH = filepath
    _SHARD_DIR = shard_dir


def _decode_batch(task):
    """Worker: decode a contiguous run of blobs, scaling node coordinates per block and
    collecting building ways, then spill one shard to disk and return its path."""
    worker_id, blobs = task
    node_id, node_lon, node_lat = [], [], []
    way_id, way_refs, way_value = [], [], []
    relation_ways = []
    with open(_FILEPATH, "rb") as f:
        for offset, size in blobs:
            data = _read_block(f, offset, size)
            string_table, header, nodes, ways, relations = decode_primitive_block(data)
            if nodes is not None:
                gran = header["granularity"]
                node_id.append(nodes["id"])
                node_lat.append((nodes["lat"] * gran + header["lat_offset"]) / 1e9)
                node_lon.append((nodes["lon"] * gran + header["lon_offset"]) / 1e9)
            found = _building_ways(string_table, ways)
            if found is not None:
                ids, ref_slices, values = found
                way_id.append(ids)
                way_refs.extend(ref_slices)
                way_value.extend(values)
            member_ways = _building_relation_member_ways(string_table, relations)
            if member_ways is not None:
                relation_ways.append(member_ways)

    refs_off = np.zeros(len(way_refs) + 1, dtype=np.int64)
    if way_refs:
        np.cumsum([len(r) for r in way_refs], out=refs_off[1:])
    path = os.path.join(_SHARD_DIR, "shard_%d.npz" % worker_id)
    np.savez(
        path,
        node_id=np.concatenate(node_id) if node_id else np.empty(0, np.int64),
        node_lon=np.concatenate(node_lon) if node_lon else np.empty(0),
        node_lat=np.concatenate(node_lat) if node_lat else np.empty(0),
        way_id=np.concatenate(way_id) if way_id else np.empty(0, np.int64),
        refs=np.concatenate(way_refs) if way_refs else np.empty(0, np.int64),
        refs_off=refs_off,
        value=np.array(way_value, dtype=object),
        relation_ways=(
            np.concatenate(relation_ways) if relation_ways else np.empty(0, np.int64)
        ),
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


def _assemble_buildings(shard_paths):
    """Assemble the building ways into a GeoDataFrame using pyrosm's own geometry
    pipeline (missing-node handling, polygon/linestring typing, dropna, orientation),
    so the result matches the in-memory reader exactly."""
    from pyrosm.frames import prepare_geodataframe

    way_id, value, ref_slices = _collect_building_ways(shard_paths)
    if len(way_id) == 0:
        return None

    # Drop ways that belong to a building relation (pyrosm assigns them to the relation).
    relation_ways = np.unique(
        np.concatenate(
            [np.load(p, allow_pickle=True)["relation_ways"] for p in shard_paths]
        )
    )
    if len(relation_ways):
        keep = ~np.isin(way_id, relation_ways)
        way_id = way_id[keep]
        value = value[keep]
        ref_slices = [r for r, k in zip(ref_slices, keep) if k]
        if len(way_id) == 0:
            return None

    node_coordinates = _node_lookup(shard_paths, np.unique(np.concatenate(ref_slices)))
    nodes = np.empty(len(ref_slices), dtype=object)
    nodes[:] = ref_slices
    ways = {"id": way_id, "nodes": nodes, "building": value}
    gdf = prepare_geodataframe(
        None,
        node_coordinates,
        ways,
        None,
        None,
        ["building"],
        None,
        keep_metadata=False,
    )
    if gdf is not None and "nodes" in gdf.columns:
        gdf = gdf.drop(columns=["nodes"])
    return gdf


def get_buildings(filepath, workers=None):
    """Read building geometries from ``filepath`` and return a GeoDataFrame.

    Phase 1: way-based buildings only (relations come later). ``workers`` defaults to
    one for small files (no multiprocessing overhead) and otherwise to a worker per
    CPU, bounded by the blob count.
    """
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
        return _assemble_buildings(shard_paths)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
