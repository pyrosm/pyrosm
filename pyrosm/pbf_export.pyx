"""Memory-efficient cropping of an ``*.osm.pbf`` by a bounding box (issue #6).

``crop_pbf`` streams the source file blob-by-blob and writes a valid, re-readable
OSM PBF holding only the data that falls inside (or completes) the crop box. It
never materializes the whole file: only compact id sets are held in memory.

Selection is "complete ways" (like osmconvert ``--complete-ways``): a way is kept
when at least one of its nodes is inside the box, and the kept way keeps its full
node list so geometries are not cut at the box edge. Relations are kept when they
reference a kept node or way.

The id/coordinate re-encoding works in the raw integer (delta) space of the PBF,
so coordinates round-trip exactly (no rounding loss).
"""

import os
import shutil
import tempfile
import zlib
from struct import pack, unpack

import numpy as np

from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.proto.osmformat_pb2 import (
    HeaderBlock,
    PrimitiveBlock,
    DenseNodes,
    DenseInfo,
)
from pyrosm.delta_compression cimport delta_encode

from cykhash import Int64Set
from cykhash.khashsets cimport isin_int64, Int64Set_from_buffer

DIV = 1000000000

# Relation member types (osmformat.proto Relation.MemberType): 0=node, 1=way.
_MEMBER_NODE = 0
_MEMBER_WAY = 1


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------
cdef _bounds_from_bbox(bounding_box):
    """Return (xmin, ymin, xmax, ymax) from a list or shapely (Multi)Polygon."""
    if bounding_box is None:
        raise ValueError(
            "Cropping a PBF requires a bounding box. Construct the OSM object "
            "with `OSM(filepath, bounding_box=...)` before calling `to_pbf()`."
        )
    if isinstance(bounding_box, (list, tuple)):
        xmin, ymin, xmax, ymax = bounding_box
        return float(xmin), float(ymin), float(xmax), float(ymax)
    # shapely geometry -> use its envelope (matches how OSM() filters by bbox)
    xmin, ymin, xmax, ymax = bounding_box.bounds
    return float(xmin), float(ymin), float(xmax), float(ymax)


# ---------------------------------------------------------------------------
# Blob-level I/O
# ---------------------------------------------------------------------------
cdef _read_next_blob(f):
    """Read one (BlobHeader, decompressed_bytes) from `f`; (None, None) at EOF."""
    buf = f.read(4)
    if len(buf) == 0:
        return None, None
    msg_len = unpack("!L", buf)[0]
    blob_header = BlobHeader()
    blob_header.ParseFromString(f.read(msg_len))
    blob = Blob()
    blob.ParseFromString(f.read(blob_header.datasize))
    if blob.HasField("raw"):
        data = blob.raw
    elif blob.HasField("zlib_data"):
        data = zlib.decompress(blob.zlib_data)
    else:
        raise ValueError(
            "Unsupported Blob compression in source PBF (only raw and zlib are "
            "handled by pyrosm)."
        )
    return blob_header, data


def _iter_primitive_blocks(filepath):
    """Yield each parsed OSMData `PrimitiveBlock` (skips the leading OSMHeader)."""
    with open(filepath, "rb") as f:
        _read_next_blob(f)  # header blob, validated separately in _read_header
        while True:
            blob_header, data = _read_next_blob(f)
            if blob_header is None:
                break
            if blob_header.type != "OSMData":
                continue
            pblock = PrimitiveBlock()
            pblock.ParseFromString(data)
            yield pblock


cdef _read_header(filepath):
    """Parse + validate the leading HeaderBlock; reject unsupported features."""
    with open(filepath, "rb") as f:
        blob_header, data = _read_next_blob(f)
    if blob_header is None or blob_header.type != "OSMHeader":
        raise ValueError(
            "File does not start with an OSMHeader block; it is not a valid "
            "OSM PBF file."
        )
    header = HeaderBlock()
    header.ParseFromString(data)
    for feature in header.required_features:
        if feature in ("OsmSchema-V0.6", "DenseNodes"):
            continue
        if feature == "HistoricalInformation":
            raise ValueError(
                "Cropping history files (.osh.pbf / 'HistoricalInformation') is "
                "not supported."
            )
        if feature == "LocationsOnWays":
            raise ValueError(
                "Cropping PBF files that store node locations on ways "
                "('LocationsOnWays') is not supported."
            )
        raise ValueError(
            "Source PBF requires unsupported feature '%s'; cannot crop it." % feature
        )
    return header


# ---------------------------------------------------------------------------
# Id-set helpers (cykhash int64 sets for memory-efficient membership)
# ---------------------------------------------------------------------------
cdef _to_set(id_array):
    arr = np.ascontiguousarray(id_array, dtype=np.int64)
    if len(arr) == 0:
        return Int64Set()
    return Int64Set_from_buffer(memoryview(arr))


cdef _isin(values, lookup):
    cdef int n = len(values)
    arr = np.ascontiguousarray(values, dtype=np.int64)
    result = np.empty(n, dtype=bool)
    if n > 0:
        isin_int64(arr, lookup, result)
    return result


cdef _unique_concat(arrays):
    if len(arrays) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


# ---------------------------------------------------------------------------
# Selection stages (each re-streams the whole file, inspecting one element type)
# ---------------------------------------------------------------------------
cdef _node_coords(pblock, dense):
    """Absolute (ids, lons, lats) of a dense group in degrees."""
    cdef long granularity = pblock.granularity
    cdef long lat_offset = pblock.lat_offset
    cdef long lon_offset = pblock.lon_offset
    ids = np.cumsum(np.fromiter(dense.id, dtype=np.int64, count=len(dense.id)))
    lat_raw = np.cumsum(np.fromiter(dense.lat, dtype=np.int64, count=len(dense.lat)))
    lon_raw = np.cumsum(np.fromiter(dense.lon, dtype=np.int64, count=len(dense.lon)))
    lats = (lat_raw * granularity + lat_offset) / DIV
    lons = (lon_raw * granularity + lon_offset) / DIV
    return ids, lons, lats


cdef _stage1_nodes_in_bbox(filepath, bounds):
    xmin, ymin, xmax, ymax = bounds
    selected = []
    for pblock in _iter_primitive_blocks(filepath):
        granularity = pblock.granularity
        lat_offset = pblock.lat_offset
        lon_offset = pblock.lon_offset
        for g in pblock.primitivegroup:
            if len(g.dense.id) > 0:
                ids, lons, lats = _node_coords(pblock, g.dense)
                mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
                if mask.any():
                    selected.append(ids[mask])
            elif len(g.nodes) > 0:
                n = len(g.nodes)
                ids = np.fromiter((node.id for node in g.nodes), dtype=np.int64, count=n)
                lat_raw = np.fromiter((node.lat for node in g.nodes), dtype=np.int64, count=n)
                lon_raw = np.fromiter((node.lon for node in g.nodes), dtype=np.int64, count=n)
                lats = (lat_raw * granularity + lat_offset) / DIV
                lons = (lon_raw * granularity + lon_offset) / DIV
                mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
                if mask.any():
                    selected.append(ids[mask])
    return _unique_concat(selected)


cdef _stage2_ways(filepath, nodes_in_bbox_set):
    kept_way_ids = []
    extra_nodes = []
    for pblock in _iter_primitive_blocks(filepath):
        for g in pblock.primitivegroup:
            if len(g.ways) == 0:
                continue
            for way in g.ways:
                refs = np.cumsum(
                    np.fromiter(way.refs, dtype=np.int64, count=len(way.refs))
                )
                if len(refs) == 0:
                    continue
                if _isin(refs, nodes_in_bbox_set).any():
                    kept_way_ids.append(way.id)
                    extra_nodes.append(refs)
    return (
        np.array(kept_way_ids, dtype=np.int64),
        _unique_concat(extra_nodes),
    )


cdef _stage3_relations(filepath, kept_nodes_set, kept_ways_set):
    kept_rel_ids = []
    for pblock in _iter_primitive_blocks(filepath):
        for g in pblock.primitivegroup:
            if len(g.relations) == 0:
                continue
            for rel in g.relations:
                memids = np.cumsum(
                    np.fromiter(rel.memids, dtype=np.int64, count=len(rel.memids))
                )
                if len(memids) == 0:
                    continue
                types = np.fromiter(rel.types, dtype=np.int64, count=len(rel.types))
                node_members = memids[types == _MEMBER_NODE]
                way_members = memids[types == _MEMBER_WAY]
                keep = False
                if len(node_members) > 0 and _isin(node_members, kept_nodes_set).any():
                    keep = True
                if not keep and len(way_members) > 0 and \
                        _isin(way_members, kept_ways_set).any():
                    keep = True
                if keep:
                    kept_rel_ids.append(rel.id)
    return np.array(kept_rel_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Write pass
# ---------------------------------------------------------------------------
cdef _split_keys_vals(keys_vals, int n_nodes):
    """Split a dense `keys_vals` array into one (key,val,...) segment per node.

    Layout per node is ``(<keyid> <valid>)* 0``; the trailing 0 delimits nodes.
    """
    segments = [[] for _ in range(n_nodes)]
    cdef int node_i = 0
    cdef int i = 0
    cdef int m = len(keys_vals)
    while i < m and node_i < n_nodes:
        v = keys_vals[i]
        if v == 0:
            node_i += 1
            i += 1
            continue
        segments[node_i].append(v)
        segments[node_i].append(keys_vals[i + 1])
        i += 2
    return segments


cdef _build_denseinfo(di, mask):
    """Rebuild a DenseInfo for the masked subset, re-delta-encoding delta fields."""
    new_di = DenseInfo()
    any_set = False
    if len(di.version) > 0:
        v = np.fromiter(di.version, dtype=np.int64, count=len(di.version))[mask]
        new_di.version.extend(v.tolist())
        any_set = True
    if len(di.timestamp) > 0:
        t = np.cumsum(
            np.fromiter(di.timestamp, dtype=np.int64, count=len(di.timestamp))
        )[mask]
        new_di.timestamp.extend(delta_encode(t).tolist())
        any_set = True
    if len(di.changeset) > 0:
        c = np.cumsum(
            np.fromiter(di.changeset, dtype=np.int64, count=len(di.changeset))
        )[mask]
        new_di.changeset.extend(delta_encode(c).tolist())
        any_set = True
    if len(di.uid) > 0:
        u = np.cumsum(np.fromiter(di.uid, dtype=np.int64, count=len(di.uid)))[mask]
        new_di.uid.extend(delta_encode(u).tolist())
        any_set = True
    if len(di.user_sid) > 0:
        s = np.cumsum(
            np.fromiter(di.user_sid, dtype=np.int64, count=len(di.user_sid))
        )[mask]
        new_di.user_sid.extend(delta_encode(s).tolist())
        any_set = True
    if len(di.visible) > 0:
        vis = np.fromiter(di.visible, dtype=bool, count=len(di.visible))[mask]
        new_di.visible.extend(vis.tolist())
        any_set = True
    return new_di if any_set else None


cdef _build_dense_group(dense, kept_nodes_set):
    """Filter a dense group to `kept_nodes_set`; return a new DenseNodes or None."""
    ids = np.cumsum(np.fromiter(dense.id, dtype=np.int64, count=len(dense.id)))
    mask = _isin(ids, kept_nodes_set)
    if not mask.any():
        return None
    lat_raw = np.cumsum(np.fromiter(dense.lat, dtype=np.int64, count=len(dense.lat)))
    lon_raw = np.cumsum(np.fromiter(dense.lon, dtype=np.int64, count=len(dense.lon)))

    new_dense = DenseNodes()
    new_dense.id.extend(delta_encode(ids[mask]).tolist())
    new_dense.lat.extend(delta_encode(lat_raw[mask]).tolist())
    new_dense.lon.extend(delta_encode(lon_raw[mask]).tolist())

    di = _build_denseinfo(dense.denseinfo, mask)
    if di is not None:
        new_dense.denseinfo.CopyFrom(di)

    if len(dense.keys_vals) > 0:
        segments = _split_keys_vals(dense.keys_vals, len(ids))
        keys_vals = []
        for i in range(len(ids)):
            if mask[i]:
                keys_vals.extend(segments[i])
                keys_vals.append(0)
        new_dense.keys_vals.extend(keys_vals)
    return new_dense


cdef _build_output_block(pblock, kept_nodes_set, kept_ways_set, kept_rel_set,
                         compact=False):
    """Build a cropped copy of `pblock`, or None if nothing is kept.

    With `compact=True` the copied string table is pruned to only the strings the
    kept elements reference (smaller output); otherwise it is copied verbatim.
    """
    out_block = PrimitiveBlock()
    out_block.stringtable.CopyFrom(pblock.stringtable)
    out_block.granularity = pblock.granularity
    out_block.lat_offset = pblock.lat_offset
    out_block.lon_offset = pblock.lon_offset
    out_block.date_granularity = pblock.date_granularity

    has_data = False
    for g in pblock.primitivegroup:
        if len(g.dense.id) > 0:
            new_dense = _build_dense_group(g.dense, kept_nodes_set)
            if new_dense is not None:
                out_block.primitivegroup.add().dense.CopyFrom(new_dense)
                has_data = True
        elif len(g.nodes) > 0:
            kept = [node for node in g.nodes if node.id in kept_nodes_set]
            if kept:
                out_block.primitivegroup.add().nodes.extend(kept)
                has_data = True
        elif len(g.ways) > 0:
            kept = [way for way in g.ways if way.id in kept_ways_set]
            if kept:
                out_block.primitivegroup.add().ways.extend(kept)
                has_data = True
        elif len(g.relations) > 0:
            kept = [rel for rel in g.relations if rel.id in kept_rel_set]
            if kept:
                out_block.primitivegroup.add().relations.extend(kept)
                has_data = True
    if not has_data:
        return None
    if compact:
        _compact_string_table(out_block)
    return out_block


cdef _remap_repeated(field, new_index):
    """Remap a repeated string-index field in place through `new_index`."""
    cdef int n = len(field)
    if n == 0:
        return
    arr = new_index[np.fromiter(field, dtype=np.int64, count=n)]
    del field[:]
    field.extend(arr.tolist())


cdef _compact_string_table(out_block):
    """Prune `out_block`'s string table to only the strings its kept elements use,
    remapping every string-index field accordingly. The kept strings are emitted in
    ascending original order (index 0, the blank entry, stays first), so the result
    is deterministic and the parallel write stays byte-identical to the sequential
    one. A no-op when every string is still referenced.
    """
    s = out_block.stringtable.s
    cdef int n_old = len(s)

    # Pass 1: mark referenced string indices (0 = blank/dense-delimiter, always kept).
    used = np.zeros(n_old, dtype=bool)
    used[0] = True
    for g in out_block.primitivegroup:
        if len(g.dense.id) > 0:
            kv = g.dense.keys_vals
            if len(kv) > 0:
                used[np.fromiter(kv, dtype=np.int64, count=len(kv))] = True
            di = g.dense.denseinfo
            if len(di.user_sid) > 0:
                used[np.cumsum(np.fromiter(di.user_sid, dtype=np.int64,
                                           count=len(di.user_sid)))] = True
        elif len(g.nodes) > 0:
            for node in g.nodes:
                if len(node.keys) > 0:
                    used[np.fromiter(node.keys, dtype=np.int64, count=len(node.keys))] = True
                if len(node.vals) > 0:
                    used[np.fromiter(node.vals, dtype=np.int64, count=len(node.vals))] = True
                if node.HasField("info"):
                    used[node.info.user_sid] = True
        elif len(g.ways) > 0:
            for way in g.ways:
                if len(way.keys) > 0:
                    used[np.fromiter(way.keys, dtype=np.int64, count=len(way.keys))] = True
                if len(way.vals) > 0:
                    used[np.fromiter(way.vals, dtype=np.int64, count=len(way.vals))] = True
                if way.HasField("info"):
                    used[way.info.user_sid] = True
        elif len(g.relations) > 0:
            for rel in g.relations:
                if len(rel.keys) > 0:
                    used[np.fromiter(rel.keys, dtype=np.int64, count=len(rel.keys))] = True
                if len(rel.vals) > 0:
                    used[np.fromiter(rel.vals, dtype=np.int64, count=len(rel.vals))] = True
                if len(rel.roles_sid) > 0:
                    used[np.fromiter(rel.roles_sid, dtype=np.int64,
                                     count=len(rel.roles_sid))] = True
                if rel.HasField("info"):
                    used[rel.info.user_sid] = True

    kept = np.flatnonzero(used)
    if len(kept) == n_old:
        return  # every string still referenced -> nothing to prune

    new_index = np.full(n_old, -1, dtype=np.int64)
    new_index[kept] = np.arange(len(kept), dtype=np.int64)
    new_strings = [s[i] for i in kept.tolist()]

    # Pass 2: remap every string-index field through new_index.
    for g in out_block.primitivegroup:
        if len(g.dense.id) > 0:
            _remap_repeated(g.dense.keys_vals, new_index)
            di = g.dense.denseinfo
            if len(di.user_sid) > 0:
                abs_sid = np.cumsum(np.fromiter(di.user_sid, dtype=np.int64,
                                                count=len(di.user_sid)))
                del di.user_sid[:]
                di.user_sid.extend(delta_encode(new_index[abs_sid]).tolist())
        elif len(g.nodes) > 0:
            for node in g.nodes:
                _remap_repeated(node.keys, new_index)
                _remap_repeated(node.vals, new_index)
                if node.HasField("info"):
                    node.info.user_sid = int(new_index[node.info.user_sid])
        elif len(g.ways) > 0:
            for way in g.ways:
                _remap_repeated(way.keys, new_index)
                _remap_repeated(way.vals, new_index)
                if way.HasField("info"):
                    way.info.user_sid = int(new_index[way.info.user_sid])
        elif len(g.relations) > 0:
            for rel in g.relations:
                _remap_repeated(rel.keys, new_index)
                _remap_repeated(rel.vals, new_index)
                _remap_repeated(rel.roles_sid, new_index)
                if rel.HasField("info"):
                    rel.info.user_sid = int(new_index[rel.info.user_sid])

    del out_block.stringtable.s[:]
    out_block.stringtable.s.extend(new_strings)


cdef _frame_blob(blob_type, message):
    """Serialize+compress `message` into the on-disk blob framing bytes."""
    data = message.SerializeToString()
    blob = Blob()
    blob.raw_size = len(data)
    blob.zlib_data = zlib.compress(data)
    blob_bytes = blob.SerializeToString()
    blob_header = BlobHeader()
    blob_header.type = blob_type
    blob_header.datasize = len(blob_bytes)
    header_bytes = blob_header.SerializeToString()
    return pack("!L", len(header_bytes)) + header_bytes + blob_bytes


cdef _write_blob(out, blob_type, message):
    out.write(_frame_blob(blob_type, message))


cdef _write_header(out, bounds):
    xmin, ymin, xmax, ymax = bounds
    header = HeaderBlock()
    header.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
    header.writingprogram = "pyrosm"
    header.bbox.left = int(round(xmin * DIV))
    header.bbox.right = int(round(xmax * DIV))
    header.bbox.top = int(round(ymax * DIV))
    header.bbox.bottom = int(round(ymin * DIV))
    _write_blob(out, "OSMHeader", header)


cdef _write_pbf(filepath, output_path, kept_nodes_set, kept_ways_set, kept_rel_set,
                bounds, compact=False):
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        for pblock in _iter_primitive_blocks(filepath):
            out_block = _build_output_block(
                pblock, kept_nodes_set, kept_ways_set, kept_rel_set, compact
            )
            if out_block is not None:
                _write_blob(out, "OSMData", out_block)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
cdef _count_data_blocks(filepath):
    """Count OSMData blocks by reading only blob headers (seeks past blob data)."""
    cdef int n = 0
    cdef int msg_len
    with open(filepath, "rb") as f:
        buf = f.read(4)
        if len(buf) == 4:  # skip the leading OSMHeader blob
            msg_len = unpack("!L", buf)[0]
            blob_header = BlobHeader()
            blob_header.ParseFromString(f.read(msg_len))
            f.seek(blob_header.datasize, 1)
        while True:
            buf = f.read(4)
            if len(buf) == 0:
                break
            msg_len = unpack("!L", buf)[0]
            blob_header = BlobHeader()
            blob_header.ParseFromString(f.read(msg_len))
            if blob_header.type == "OSMData":
                n += 1
            f.seek(blob_header.datasize, 1)
    return n


cpdef crop_pbf(source_path, output_path, bounding_box, keep_relations=True,
               workers=1, compact=False, repack=False):
    """Crop `source_path` by `bounding_box`, writing a valid PBF to `output_path`.

    Returns the output path. When ``workers > 1`` and the file has enough blocks
    to amortize pool startup (>= ``2 * workers`` OSMData blocks), the parallel
    path is used; otherwise the (faster, for small files) sequential path runs.

    When ``compact`` is True each output block's string table is pruned to only the
    strings its kept elements reference (smaller output, slightly slower); when
    False (default) the source block's full string table is copied verbatim.

    When ``repack`` is True the kept elements are re-chunked into canonical, densely
    packed blocks (smallest output, slowest); the re-pack write is sequential, but
    ``workers`` still parallelizes the selection. ``repack=True`` produces minimal
    string tables, so ``compact`` is irrelevant and ignored.
    """
    bounds = _bounds_from_bbox(bounding_box)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_crop_")
        os.close(fd)

    # Stage 0: header pre-flight (rejects unsupported inputs before any streaming).
    _read_header(source_path)

    if workers is not None and workers > 1:
        if _count_data_blocks(source_path) >= 2 * int(workers):
            return _crop_pbf_parallel(
                source_path, output_path, bounds, keep_relations, int(workers),
                compact, repack
            )
        # Too few blocks for parallelism to pay off -> run sequentially.

    # Stage 1: nodes inside the bbox.
    nodes_in_bbox = _stage1_nodes_in_bbox(source_path, bounds)
    nib_set = _to_set(nodes_in_bbox)

    # Stage 2: ways with >=1 node in the bbox (+ all their refs -> complete ways).
    kept_ways, extra_nodes = _stage2_ways(source_path, nib_set)
    kept_nodes = _unique_concat([nodes_in_bbox, extra_nodes])
    kept_nodes_set = _to_set(kept_nodes)
    kept_ways_set = _to_set(kept_ways)

    # Stage 3: relations referencing a kept node/way.
    if keep_relations:
        kept_rel = _stage3_relations(source_path, kept_nodes_set, kept_ways_set)
    else:
        kept_rel = np.empty(0, dtype=np.int64)
    kept_rel_set = _to_set(kept_rel)

    # Write pass.
    if repack:
        _repack_write(
            source_path, output_path, kept_nodes_set, kept_ways_set, kept_rel_set,
            bounds
        )
    else:
        _write_pbf(
            source_path, output_path, kept_nodes_set, kept_ways_set, kept_rel_set,
            bounds, compact
        )
    return output_path


# ---------------------------------------------------------------------------
# Parallel path (workers > 1)
# ---------------------------------------------------------------------------
# The selection is staged (node -> way -> relation -> write); each stage is
# internally parallel across blobs but the stages run in sequence because each
# depends on the previous stage's *complete* result. A SINGLE pool is reused
# across all four stages (re-spawning a pool per stage would re-pay the worker
# startup cost four times). The main process reads raw (still-compressed) blob
# payloads sequentially (cheap I/O) and hands them to the pool; workers do the
# heavy decompress + protobuf parse + (re-)encode. The growing kept-id arrays a
# stage needs are broadcast to the persistent workers via small `.npy` files in
# a temp dir (written by the main process between stages, memory-mapped + cached
# per worker on first use) rather than re-pickled per task. Output blobs come
# back in input order (Pool.imap preserves order) so the written bytes are
# identical to the sequential path.

# Per-worker globals populated by `_winit`; `_W_CACHE` memoizes the khash sets
# built from the broadcast `.npy` files so each worker loads each set only once.
_W_BOUNDS = None
_W_TMPDIR = None
_W_CACHE = {}
_W_COMPACT = False


cdef _read_next_payload(f):
    """Read one (blob_type, (is_raw, payload_bytes)); (None, None) at EOF."""
    buf = f.read(4)
    if len(buf) == 0:
        return None, None
    msg_len = unpack("!L", buf)[0]
    blob_header = BlobHeader()
    blob_header.ParseFromString(f.read(msg_len))
    blob = Blob()
    blob.ParseFromString(f.read(blob_header.datasize))
    if blob.HasField("raw"):
        return blob_header.type, (True, blob.raw)
    elif blob.HasField("zlib_data"):
        return blob_header.type, (False, blob.zlib_data)
    else:
        raise ValueError(
            "Unsupported Blob compression in source PBF (only raw and zlib are "
            "handled by pyrosm)."
        )


def _iter_payloads(filepath):
    """Yield each OSMData blob's raw (still-compressed) payload, skipping header."""
    with open(filepath, "rb") as f:
        _read_next_payload(f)  # header
        while True:
            btype, payload = _read_next_payload(f)
            if btype is None:
                break
            if btype != "OSMData":
                continue
            yield payload


cdef _payload_to_block(payload):
    is_raw, data = payload
    if not is_raw:
        data = zlib.decompress(data)
    pblock = PrimitiveBlock()
    pblock.ParseFromString(data)
    return pblock


def _winit(bounds, tmpdir, compact):
    global _W_BOUNDS, _W_TMPDIR, _W_CACHE, _W_COMPACT
    _W_BOUNDS = bounds
    _W_TMPDIR = tmpdir
    _W_CACHE = {}
    _W_COMPACT = compact


def _w_get_set(name):
    """Lazily load + cache the broadcast id set `name` from the temp dir."""
    s = _W_CACHE.get(name)
    if s is None:
        arr = np.load(os.path.join(_W_TMPDIR, name + ".npy"))
        s = _to_set(arr)
        _W_CACHE[name] = s
    return s


def _w_stage1(payload):
    pblock = _payload_to_block(payload)
    xmin, ymin, xmax, ymax = _W_BOUNDS
    granularity = pblock.granularity
    lat_offset = pblock.lat_offset
    lon_offset = pblock.lon_offset
    selected = []
    for g in pblock.primitivegroup:
        if len(g.dense.id) > 0:
            ids, lons, lats = _node_coords(pblock, g.dense)
            mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
            if mask.any():
                selected.append(ids[mask])
        elif len(g.nodes) > 0:
            n = len(g.nodes)
            ids = np.fromiter((node.id for node in g.nodes), dtype=np.int64, count=n)
            lat_raw = np.fromiter((node.lat for node in g.nodes), dtype=np.int64, count=n)
            lon_raw = np.fromiter((node.lon for node in g.nodes), dtype=np.int64, count=n)
            lats = (lat_raw * granularity + lat_offset) / DIV
            lons = (lon_raw * granularity + lon_offset) / DIV
            mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
            if mask.any():
                selected.append(ids[mask])
    return _unique_concat(selected)


def _w_stage2(payload):
    pblock = _payload_to_block(payload)
    nodes_set = _w_get_set("nodes_in_bbox")
    kept_way_ids = []
    extra_nodes = []
    for g in pblock.primitivegroup:
        if len(g.ways) == 0:
            continue
        for way in g.ways:
            refs = np.cumsum(np.fromiter(way.refs, dtype=np.int64, count=len(way.refs)))
            if len(refs) == 0:
                continue
            if _isin(refs, nodes_set).any():
                kept_way_ids.append(way.id)
                extra_nodes.append(refs)
    return (np.array(kept_way_ids, dtype=np.int64), _unique_concat(extra_nodes))


def _w_stage3(payload):
    pblock = _payload_to_block(payload)
    nodes_set = _w_get_set("kept_nodes")
    ways_set = _w_get_set("kept_ways")
    kept_rel_ids = []
    for g in pblock.primitivegroup:
        if len(g.relations) == 0:
            continue
        for rel in g.relations:
            memids = np.cumsum(
                np.fromiter(rel.memids, dtype=np.int64, count=len(rel.memids))
            )
            if len(memids) == 0:
                continue
            types = np.fromiter(rel.types, dtype=np.int64, count=len(rel.types))
            node_members = memids[types == _MEMBER_NODE]
            way_members = memids[types == _MEMBER_WAY]
            keep = False
            if len(node_members) > 0 and _isin(node_members, nodes_set).any():
                keep = True
            if not keep and len(way_members) > 0 and \
                    _isin(way_members, ways_set).any():
                keep = True
            if keep:
                kept_rel_ids.append(rel.id)
    return np.array(kept_rel_ids, dtype=np.int64)


def _w_write(payload):
    pblock = _payload_to_block(payload)
    out_block = _build_output_block(
        pblock,
        _w_get_set("kept_nodes"),
        _w_get_set("kept_ways"),
        _w_get_set("kept_rel"),
        _W_COMPACT,
    )
    if out_block is None:
        return None
    return _frame_blob("OSMData", out_block)


cdef _broadcast(tmpdir, name, arr):
    """Write a kept-id array to the temp dir for the persistent workers to load."""
    np.save(os.path.join(tmpdir, name + ".npy"), np.ascontiguousarray(arr, dtype=np.int64))


cdef _crop_pbf_parallel(source_path, output_path, bounds, keep_relations, workers,
                        compact=False, repack=False):
    import multiprocessing as mp

    tmpdir = tempfile.mkdtemp(prefix="pyrosm_crop_par_")
    pool = mp.Pool(workers, initializer=_winit, initargs=(bounds, tmpdir, compact))
    try:
        # Stage 1: nodes inside the bbox.
        results = pool.map(_w_stage1, _iter_payloads(source_path))
        nodes_in_bbox = _unique_concat(results)
        _broadcast(tmpdir, "nodes_in_bbox", nodes_in_bbox)

        # Stage 2: complete ways.
        results = pool.map(_w_stage2, _iter_payloads(source_path))
        kept_ways = _unique_concat([w for w, _ in results])
        extra_nodes = _unique_concat([e for _, e in results])
        kept_nodes = _unique_concat([nodes_in_bbox, extra_nodes])
        _broadcast(tmpdir, "kept_nodes", kept_nodes)
        _broadcast(tmpdir, "kept_ways", kept_ways)

        # Stage 3: relations.
        if keep_relations:
            results = pool.map(_w_stage3, _iter_payloads(source_path))
            kept_rel = _unique_concat(results)
        else:
            kept_rel = np.empty(0, dtype=np.int64)
        _broadcast(tmpdir, "kept_rel", kept_rel)

        if repack:
            # Re-pack needs a global re-chunk, so the write is sequential (selection
            # above stays parallel). Build the kept-id sets in the main process.
            _repack_write(
                source_path, output_path, _to_set(kept_nodes), _to_set(kept_ways),
                _to_set(kept_rel), bounds
            )
        else:
            # Write pass (imap preserves input order -> deterministic output bytes).
            with open(output_path, "wb") as out:
                _write_header(out, bounds)
                for blob_bytes in pool.imap(_w_write, _iter_payloads(source_path)):
                    if blob_bytes is not None:
                        out.write(blob_bytes)
    finally:
        pool.close()
        pool.join()
        shutil.rmtree(tmpdir, ignore_errors=True)
    return output_path


# ---------------------------------------------------------------------------
# Build a PBF from records (issue #285): the write-side of OSM.write_pbf
# ---------------------------------------------------------------------------
# Unlike the crop path (which copies source elements verbatim), this constructs
# fresh PBF blocks from node/way/relation records + their (possibly edited) tags.
# Coordinates use granularity 100 / offset 0; ids/coords/timestamps are encoded in
# raw integer (delta) space via `delta_encode`. The OSM `visible` flag is omitted
# (current-data PBF: absent visible means "visible"); each block carries only the
# strings it uses.

_MAX_GROUP = 8000


cdef _coord_to_raw(values):
    # degrees -> raw integer grid: lat = p * granularity / 1e9 with granularity 100
    # and offset 0, so p = round(deg * 1e7).
    return np.rint(np.ascontiguousarray(values, dtype=np.float64) * 1e7).astype(np.int64)


cdef class _StringTable:
    """Per-block string interner; index 0 is the reserved blank entry."""
    cdef dict index
    cdef list strings

    def __cinit__(self):
        self.index = {"": 0}
        self.strings = [b""]

    cdef int intern(self, s):
        cdef object i = self.index.get(s)
        if i is None:
            i = len(self.strings)
            self.index[s] = i
            self.strings.append(s.encode("utf-8") if isinstance(s, str) else s)
        return i


cdef _new_block():
    block = PrimitiveBlock()
    block.granularity = 100
    block.lat_offset = 0
    block.lon_offset = 0
    block.date_granularity = 1000
    return block


cdef _emit_node_block(out, ids, lat_raw, lon_raw, versions, timestamps, changesets,
                      tags_list):
    block = _new_block()
    st = _StringTable()

    has_tags = False
    keys_vals = []
    for t in tags_list:
        if t:
            has_tags = True
            for k, v in t.items():
                keys_vals.append(st.intern(k))
                keys_vals.append(st.intern(v))
        keys_vals.append(0)

    group = block.primitivegroup.add()
    dense = group.dense
    dense.id.extend(delta_encode(ids).tolist())
    dense.lat.extend(delta_encode(lat_raw).tolist())
    dense.lon.extend(delta_encode(lon_raw).tolist())
    if has_tags:
        dense.keys_vals.extend(keys_vals)

    di = dense.denseinfo
    di.version.extend([int(v) for v in versions])
    di.timestamp.extend(delta_encode(timestamps).tolist())
    di.changeset.extend(delta_encode(changesets).tolist())

    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_way_block(out, way_batch):
    block = _new_block()
    st = _StringTable()
    group = block.primitivegroup.add()
    for w in way_batch:
        way = group.ways.add()
        way.id = w["id"]
        tags = w["tags"]
        if tags:
            for k, v in tags.items():
                way.keys.append(st.intern(k))
                way.vals.append(st.intern(v))
        way.info.version = int(w.get("version") or 1)
        if w.get("timestamp") is not None:
            way.info.timestamp = int(w["timestamp"])
        refs = np.ascontiguousarray(w["refs"], dtype=np.int64)
        way.refs.extend(delta_encode(refs).tolist())
    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_relation_block(out, rel_batch):
    type_map = {
        b"node": 0, "node": 0, 0: 0,
        b"way": 1, "way": 1, 1: 1,
        b"relation": 2, "relation": 2, 2: 2,
    }
    block = _new_block()
    st = _StringTable()
    group = block.primitivegroup.add()
    for r in rel_batch:
        rel = group.relations.add()
        rel.id = r["id"]
        tags = r["tags"]
        if tags:
            for k, v in tags.items():
                rel.keys.append(st.intern(k))
                rel.vals.append(st.intern(v))
        rel.info.version = int(r.get("version") or 1)
        if r.get("timestamp") is not None:
            rel.info.timestamp = int(r["timestamp"])
        if r.get("changeset") is not None:
            rel.info.changeset = int(r["changeset"])
        memids = []
        for (mtype, mref, mrole) in r["members"]:
            if isinstance(mrole, bytes):
                mrole = mrole.decode("utf-8")
            rel.roles_sid.append(st.intern(mrole if mrole is not None else ""))
            rel.types.append(type_map[mtype])
            memids.append(mref)
        rel.memids.extend(
            delta_encode(np.ascontiguousarray(memids, dtype=np.int64)).tolist()
        )
    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cpdef write_pbf_from_records(nodes, ways, relations, output_path, bounds):
    """Serialize node/way/relation records to a valid PBF at `output_path`.

    `nodes` is a dict of aligned arrays (id/lat/lon/version/timestamp/changeset and
    an object array `tags`); `ways`/`relations` are lists of record dicts. `bounds`
    is (xmin, ymin, xmax, ymax) for the header bbox.
    """
    ids = np.ascontiguousarray(nodes["id"], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    lat_raw = _coord_to_raw(nodes["lat"])[order]
    lon_raw = _coord_to_raw(nodes["lon"])[order]
    versions = np.ascontiguousarray(nodes["version"], dtype=np.int64)[order]
    timestamps = np.ascontiguousarray(nodes["timestamp"], dtype=np.int64)[order]
    changesets = np.ascontiguousarray(nodes["changeset"], dtype=np.int64)[order]
    tags_arr = nodes["tags"]
    tags_ordered = [tags_arr[i] for i in order]

    cdef int n = len(ids)
    cdef int i = 0
    cdef int j
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        while i < n:
            j = min(i + _MAX_GROUP, n)
            _emit_node_block(
                out, ids[i:j], lat_raw[i:j], lon_raw[i:j], versions[i:j],
                timestamps[i:j], changesets[i:j], tags_ordered[i:j],
            )
            i = j
        i = 0
        while i < len(ways):
            j = min(i + _MAX_GROUP, len(ways))
            _emit_way_block(out, ways[i:j])
            i = j
        i = 0
        while i < len(relations):
            j = min(i + _MAX_GROUP, len(relations))
            _emit_relation_block(out, relations[i:j])
            i = j
    return output_path


# ---------------------------------------------------------------------------
# Re-pack write (issue #6): rewrite the crop as canonical, densely-packed blocks
# ---------------------------------------------------------------------------
# Unlike the default crop (which filters each source block in place, leaving
# partially-full blocks), `repack` decodes the kept elements and re-chunks them into
# full _MAX_GROUP blocks with fresh minimal string tables -- the canonical form
# osmium/Osmosis produce, which is smaller. The source is already id-sorted and the
# selection preserves that order, so this is a streaming re-chunk (bounded to ~one
# output block per element type), not a global sort. Source raw integer coordinates,
# timestamps and ids are passed straight through (exact on the granularity-100 grid;
# see the guard in `_repack_write`). These emitters are separate from the from-records
# `_emit_*` so the `write_pbf` path is untouched.

cdef _emit_repack_node_block(out, ids, lat_raw, lon_raw, tags_list, meta):
    block = _new_block()
    st = _StringTable()
    has_tags = False
    keys_vals = []
    for t in tags_list:
        if t:
            has_tags = True
            for k, v in t.items():
                keys_vals.append(st.intern(k))
                keys_vals.append(st.intern(v))
        keys_vals.append(0)

    group = block.primitivegroup.add()
    dense = group.dense
    dense.id.extend(delta_encode(ids).tolist())
    dense.lat.extend(delta_encode(lat_raw).tolist())
    dense.lon.extend(delta_encode(lon_raw).tolist())
    if has_tags:
        dense.keys_vals.extend(keys_vals)
    if meta is not None:
        di = dense.denseinfo
        if "version" in meta:
            di.version.extend([int(v) for v in meta["version"]])
        if "timestamp" in meta:
            di.timestamp.extend(delta_encode(meta["timestamp"]).tolist())
        if "changeset" in meta:
            di.changeset.extend(delta_encode(meta["changeset"]).tolist())
        if "uid" in meta:
            di.uid.extend(delta_encode(meta["uid"]).tolist())
        if "user" in meta:
            sids = np.asarray([st.intern(u) for u in meta["user"]], dtype=np.int64)
            di.user_sid.extend(delta_encode(sids).tolist())
        if "visible" in meta:
            di.visible.extend([bool(x) for x in meta["visible"]])

    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_repack_info(info_msg, _StringTable st, info):
    """Set an Info submessage from a decoded-source info dict (or omit if None)."""
    if info is None:
        return
    if info.get("version") is not None:
        info_msg.version = int(info["version"])
    if info.get("timestamp") is not None:
        info_msg.timestamp = int(info["timestamp"])
    if info.get("changeset") is not None:
        info_msg.changeset = int(info["changeset"])
    if info.get("uid") is not None:
        info_msg.uid = int(info["uid"])
    if info.get("user") is not None:
        info_msg.user_sid = st.intern(info["user"])
    if info.get("visible") is not None:
        info_msg.visible = bool(info["visible"])


cdef _emit_repack_way_block(out, way_batch):
    block = _new_block()
    st = _StringTable()
    group = block.primitivegroup.add()
    for w in way_batch:
        way = group.ways.add()
        way.id = w["id"]
        tags = w["tags"]
        if tags:
            for k, v in tags.items():
                way.keys.append(st.intern(k))
                way.vals.append(st.intern(v))
        _emit_repack_info(way.info, st, w["info"])
        way.refs.extend(delta_encode(w["refs"]).tolist())
    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_repack_relation_block(out, rel_batch):
    block = _new_block()
    st = _StringTable()
    group = block.primitivegroup.add()
    for r in rel_batch:
        rel = group.relations.add()
        rel.id = r["id"]
        tags = r["tags"]
        if tags:
            for k, v in tags.items():
                rel.keys.append(st.intern(k))
                rel.vals.append(st.intern(v))
        _emit_repack_info(rel.info, st, r["info"])
        memids = []
        for (mtype, mref, mrole) in r["members"]:
            rel.roles_sid.append(st.intern(mrole))
            rel.types.append(int(mtype))
            memids.append(mref)
        rel.memids.extend(
            delta_encode(np.ascontiguousarray(memids, dtype=np.int64)).tolist()
        )
    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


# ---- decode the kept elements of a source block into re-pack records ----------

cdef _decode_info(info_msg):
    """Decode an Info submessage to a dict, or None when no metadata is present.

    Each optional field is checked independently -- a legal PBF Info may carry any
    subset (e.g. timestamp without version) and all present fields are preserved.
    """
    info = {}
    if info_msg.HasField("version"):
        info["version"] = info_msg.version
    if info_msg.HasField("timestamp"):
        info["timestamp"] = info_msg.timestamp
    if info_msg.HasField("changeset"):
        info["changeset"] = info_msg.changeset
    if info_msg.HasField("uid"):
        info["uid"] = info_msg.uid
    if info_msg.HasField("user_sid"):
        info["_user_sid"] = info_msg.user_sid  # resolved to a string by the caller
    if info_msg.HasField("visible"):
        info["visible"] = info_msg.visible
    return info if info else None


cdef _decode_kept_dense_nodes(dense, stringtable, kept_nodes_set):
    cdef int n = len(dense.id)
    ids = np.cumsum(np.fromiter(dense.id, dtype=np.int64, count=n))
    mask = _isin(ids, kept_nodes_set)
    if not mask.any():
        return None
    lat = np.cumsum(np.fromiter(dense.lat, dtype=np.int64, count=n))[mask]
    lon = np.cumsum(np.fromiter(dense.lon, dtype=np.int64, count=n))[mask]
    kept_ids = ids[mask]
    keep_pos = np.flatnonzero(mask)

    tags_list = []
    if len(dense.keys_vals) > 0:
        segments = _split_keys_vals(dense.keys_vals, n)
        for i in keep_pos:
            seg = segments[i]
            if seg:
                d = {}
                for j in range(0, len(seg), 2):
                    d[stringtable[seg[j]]] = stringtable[seg[j + 1]]
                tags_list.append(d)
            else:
                tags_list.append(None)
    else:
        tags_list = [None] * len(kept_ids)

    di = dense.denseinfo
    meta = {}
    if len(di.version) > 0:
        meta["version"] = np.fromiter(di.version, dtype=np.int64, count=len(di.version))[mask]
    if len(di.timestamp) > 0:
        meta["timestamp"] = np.cumsum(
            np.fromiter(di.timestamp, dtype=np.int64, count=len(di.timestamp)))[mask]
    if len(di.changeset) > 0:
        meta["changeset"] = np.cumsum(
            np.fromiter(di.changeset, dtype=np.int64, count=len(di.changeset)))[mask]
    if len(di.uid) > 0:
        meta["uid"] = np.cumsum(
            np.fromiter(di.uid, dtype=np.int64, count=len(di.uid)))[mask]
    if len(di.user_sid) > 0:
        sids = np.cumsum(
            np.fromiter(di.user_sid, dtype=np.int64, count=len(di.user_sid)))[mask]
        meta["user"] = [stringtable[s] for s in sids.tolist()]
    if len(di.visible) > 0:
        meta["visible"] = np.array(list(di.visible), dtype=bool)[mask]
    if not meta:
        meta = None
    return kept_ids, lat, lon, tags_list, meta


cdef _decode_kept_plain_nodes(nodes, stringtable, kept_nodes_set):
    """Non-dense node group -> the same chunk shape (emitted as dense)."""
    kept = [node for node in nodes if _isin(
        np.asarray([node.id], dtype=np.int64), kept_nodes_set)[0]]
    if not kept:
        return None
    ids = np.asarray([node.id for node in kept], dtype=np.int64)
    lat = np.asarray([node.lat for node in kept], dtype=np.int64)
    lon = np.asarray([node.lon for node in kept], dtype=np.int64)
    tags_list = []
    metas = []
    for node in kept:
        if len(node.keys) > 0:
            tags_list.append({stringtable[k]: stringtable[v]
                              for k, v in zip(node.keys, node.vals)})
        else:
            tags_list.append(None)
        info = _decode_info(node.info)
        if info is not None and "_user_sid" in info:
            info["user"] = stringtable[info.pop("_user_sid")]
        metas.append(info)
    # A canonical dense block carries DenseInfo for every node or for none, so a
    # non-dense group whose kept nodes have mixed metadata (some with, some without,
    # or differing fields) cannot be re-packed faithfully -> reject rather than drop
    # or synthesize metadata. Real (dense) PBF never hits this.
    if len(set(_meta_schema(m) for m in metas)) > 1:
        raise ValueError(
            "to_pbf(repack=True) does not support a non-dense node group whose nodes "
            "carry mixed element metadata; use repack=False for this file."
        )
    meta = None
    if metas[0] is not None:
        meta = {}
        for key in metas[0].keys():
            if key == "user":
                meta["user"] = [m["user"] for m in metas]
            elif key == "visible":
                meta["visible"] = np.asarray([m["visible"] for m in metas], dtype=bool)
            else:
                meta[key] = np.asarray([m[key] for m in metas], dtype=np.int64)
    return ids, lat, lon, tags_list, meta


cdef _decode_kept_ways(ways, stringtable, kept_ways_set):
    records = []
    for way in ways:
        if not _isin(np.asarray([way.id], dtype=np.int64), kept_ways_set)[0]:
            continue
        tags = {stringtable[k]: stringtable[v]
                for k, v in zip(way.keys, way.vals)} if len(way.keys) > 0 else None
        info = _decode_info(way.info)
        if info is not None and "_user_sid" in info:
            info["user"] = stringtable[info.pop("_user_sid")]
        refs = np.cumsum(np.fromiter(way.refs, dtype=np.int64, count=len(way.refs)))
        records.append({"id": way.id, "tags": tags, "info": info, "refs": refs})
    return records


cdef _decode_kept_relations(relations, stringtable, kept_rel_set):
    records = []
    for rel in relations:
        if not _isin(np.asarray([rel.id], dtype=np.int64), kept_rel_set)[0]:
            continue
        tags = {stringtable[k]: stringtable[v]
                for k, v in zip(rel.keys, rel.vals)} if len(rel.keys) > 0 else None
        info = _decode_info(rel.info)
        if info is not None and "_user_sid" in info:
            info["user"] = stringtable[info.pop("_user_sid")]
        memids = np.cumsum(np.fromiter(rel.memids, dtype=np.int64, count=len(rel.memids)))
        members = [
            (rel.types[j], int(memids[j]), stringtable[rel.roles_sid[j]])
            for j in range(len(memids))
        ]
        records.append({"id": rel.id, "tags": tags, "info": info, "members": members})
    return records


# ---- re-chunk buffer (bounded: ~one output block per element type) ------------

cdef _meta_schema(meta):
    """The set of metadata fields present (or None) -- used to detect mixed schemas."""
    return None if meta is None else frozenset(meta.keys())


cdef _concat_meta(metas):
    # All chunks merged into one output block must share a metadata schema: a dense
    # block's DenseInfo is per-field all-or-nothing, so it cannot represent some nodes
    # with metadata and others without. Real PBF is uniform; reject the mixed case
    # rather than drop or misalign fields.
    if len(set(_meta_schema(m) for m in metas)) > 1:
        raise ValueError(
            "to_pbf(repack=True) requires a uniform element-metadata schema across the "
            "file; this file mixes blocks with and without (or with differing) "
            "metadata. Use repack=False for this file."
        )
    if not metas or metas[0] is None:
        return None
    out = {}
    for key in metas[0].keys():
        if key == "user":
            users = []
            for m in metas:
                users.extend(m["user"])
            out["user"] = users
        else:
            out[key] = np.concatenate([m[key] for m in metas])
    return out


cdef _slice_meta(meta, a, b):
    if meta is None:
        return None
    return {k: v[a:b] for k, v in meta.items()}


cdef class _RepackWriter:
    cdef object out
    cdef list node_chunks
    cdef long node_count
    cdef list way_buf
    cdef list rel_buf

    def __cinit__(self, out):
        self.out = out
        self.node_chunks = []
        self.node_count = 0
        self.way_buf = []
        self.rel_buf = []

    cdef add_nodes(self, chunk):
        self.node_chunks.append(chunk)
        self.node_count += len(chunk[0])
        while self.node_count >= _MAX_GROUP:
            self._emit_full_node_block()

    cdef _emit_full_node_block(self):
        ids = np.concatenate([c[0] for c in self.node_chunks])
        lat = np.concatenate([c[1] for c in self.node_chunks])
        lon = np.concatenate([c[2] for c in self.node_chunks])
        tags = []
        for c in self.node_chunks:
            tags.extend(c[3])
        meta = _concat_meta([c[4] for c in self.node_chunks])
        _emit_repack_node_block(
            self.out, ids[:_MAX_GROUP], lat[:_MAX_GROUP], lon[:_MAX_GROUP],
            tags[:_MAX_GROUP], _slice_meta(meta, 0, _MAX_GROUP))
        rem = (ids[_MAX_GROUP:], lat[_MAX_GROUP:], lon[_MAX_GROUP:],
               tags[_MAX_GROUP:], _slice_meta(meta, _MAX_GROUP, len(ids)))
        self.node_chunks = [rem]
        self.node_count = len(ids) - _MAX_GROUP

    cdef _flush_nodes(self):
        if self.node_count > 0:
            ids = np.concatenate([c[0] for c in self.node_chunks])
            lat = np.concatenate([c[1] for c in self.node_chunks])
            lon = np.concatenate([c[2] for c in self.node_chunks])
            tags = []
            for c in self.node_chunks:
                tags.extend(c[3])
            meta = _concat_meta([c[4] for c in self.node_chunks])
            _emit_repack_node_block(self.out, ids, lat, lon, tags, meta)
        self.node_chunks = []
        self.node_count = 0

    cdef add_ways(self, records):
        self._flush_nodes()
        self.way_buf.extend(records)
        while len(self.way_buf) >= _MAX_GROUP:
            _emit_repack_way_block(self.out, self.way_buf[:_MAX_GROUP])
            self.way_buf = self.way_buf[_MAX_GROUP:]

    cdef _flush_ways(self):
        if self.way_buf:
            _emit_repack_way_block(self.out, self.way_buf)
            self.way_buf = []

    cdef add_relations(self, records):
        self._flush_nodes()
        self._flush_ways()
        self.rel_buf.extend(records)
        while len(self.rel_buf) >= _MAX_GROUP:
            _emit_repack_relation_block(self.out, self.rel_buf[:_MAX_GROUP])
            self.rel_buf = self.rel_buf[_MAX_GROUP:]

    cdef close(self):
        self._flush_nodes()
        self._flush_ways()
        if self.rel_buf:
            _emit_repack_relation_block(self.out, self.rel_buf)
            self.rel_buf = []


cdef _repack_write(source_path, output_path, kept_nodes_set, kept_ways_set,
                   kept_rel_set, bounds):
    """Sequential re-pack write: re-chunk the kept crop into canonical full blocks."""
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        writer = _RepackWriter(out)
        for pblock in _iter_primitive_blocks(source_path):
            if (pblock.granularity != 100 or pblock.lat_offset != 0
                    or pblock.lon_offset != 0 or pblock.date_granularity != 1000):
                raise ValueError(
                    "to_pbf(repack=True) requires the standard PBF grid "
                    "(granularity 100, zero lat/lon offsets, date_granularity 1000); "
                    "this file uses a different grid. Use repack=False."
                )
            st = pblock.stringtable.s
            for g in pblock.primitivegroup:
                if len(g.dense.id) > 0:
                    chunk = _decode_kept_dense_nodes(g.dense, st, kept_nodes_set)
                    if chunk is not None:
                        writer.add_nodes(chunk)
                elif len(g.nodes) > 0:
                    chunk = _decode_kept_plain_nodes(g.nodes, st, kept_nodes_set)
                    if chunk is not None:
                        writer.add_nodes(chunk)
                elif len(g.ways) > 0:
                    records = _decode_kept_ways(g.ways, st, kept_ways_set)
                    if records:
                        writer.add_ways(records)
                elif len(g.relations) > 0:
                    records = _decode_kept_relations(g.relations, st, kept_rel_set)
                    if records:
                        writer.add_relations(records)
        writer.close()
    return output_path
