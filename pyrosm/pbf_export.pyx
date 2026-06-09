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


cdef _build_output_block(pblock, kept_nodes_set, kept_ways_set, kept_rel_set):
    """Build a cropped copy of `pblock`, or None if nothing is kept."""
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
    return out_block


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
                bounds):
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        for pblock in _iter_primitive_blocks(filepath):
            out_block = _build_output_block(
                pblock, kept_nodes_set, kept_ways_set, kept_rel_set
            )
            if out_block is not None:
                _write_blob(out, "OSMData", out_block)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
cpdef crop_pbf(source_path, output_path, bounding_box, keep_relations=True,
               workers=1):
    """Crop `source_path` by `bounding_box`, writing a valid PBF to `output_path`.

    Returns the output path. `workers` is accepted for API symmetry; the parallel
    path is selected when ``workers > 1``.
    """
    bounds = _bounds_from_bbox(bounding_box)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_crop_")
        os.close(fd)

    # Stage 0: header pre-flight (rejects unsupported inputs before any streaming).
    _read_header(source_path)

    if workers is not None and workers > 1:
        return _crop_pbf_parallel(
            source_path, output_path, bounds, keep_relations, int(workers)
        )

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
    _write_pbf(
        source_path, output_path, kept_nodes_set, kept_ways_set, kept_rel_set, bounds
    )
    return output_path


# ---------------------------------------------------------------------------
# Parallel path (workers > 1)
# ---------------------------------------------------------------------------
# The selection is staged (node -> way -> relation -> write); each stage is
# internally parallel across blobs but the stages run in sequence because each
# depends on the previous stage's *complete* result. The main process reads raw
# (still-compressed) blob payloads sequentially (cheap I/O) and hands them to a
# pool; workers do the heavy decompress + protobuf parse + (re-)encode. The
# kept-id arrays a stage needs are distributed to workers via the pool
# initializer. Output blobs come back in input order (Pool.imap preserves
# order) so the written bytes are identical to the sequential path.

# Per-worker globals populated by `_winit`.
_W_BOUNDS = None
_W_NODES_SET = None
_W_WAYS_SET = None
_W_REL_SET = None


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


def _winit(bounds, nodes_arr, ways_arr, rel_arr):
    global _W_BOUNDS, _W_NODES_SET, _W_WAYS_SET, _W_REL_SET
    _W_BOUNDS = bounds
    _W_NODES_SET = None if nodes_arr is None else _to_set(nodes_arr)
    _W_WAYS_SET = None if ways_arr is None else _to_set(ways_arr)
    _W_REL_SET = None if rel_arr is None else _to_set(rel_arr)


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
    kept_way_ids = []
    extra_nodes = []
    for g in pblock.primitivegroup:
        if len(g.ways) == 0:
            continue
        for way in g.ways:
            refs = np.cumsum(np.fromiter(way.refs, dtype=np.int64, count=len(way.refs)))
            if len(refs) == 0:
                continue
            if _isin(refs, _W_NODES_SET).any():
                kept_way_ids.append(way.id)
                extra_nodes.append(refs)
    return (np.array(kept_way_ids, dtype=np.int64), _unique_concat(extra_nodes))


def _w_stage3(payload):
    pblock = _payload_to_block(payload)
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
            if len(node_members) > 0 and _isin(node_members, _W_NODES_SET).any():
                keep = True
            if not keep and len(way_members) > 0 and \
                    _isin(way_members, _W_WAYS_SET).any():
                keep = True
            if keep:
                kept_rel_ids.append(rel.id)
    return np.array(kept_rel_ids, dtype=np.int64)


def _w_write(payload):
    pblock = _payload_to_block(payload)
    out_block = _build_output_block(
        pblock, _W_NODES_SET, _W_WAYS_SET, _W_REL_SET
    )
    if out_block is None:
        return None
    return _frame_blob("OSMData", out_block)


cdef _crop_pbf_parallel(source_path, output_path, bounds, keep_relations, workers):
    import multiprocessing as mp

    # Stage 1: nodes inside the bbox.
    with mp.Pool(workers, initializer=_winit,
                 initargs=(bounds, None, None, None)) as pool:
        results = pool.map(_w_stage1, _iter_payloads(source_path))
    nodes_in_bbox = _unique_concat(results)

    # Stage 2: complete ways.
    with mp.Pool(workers, initializer=_winit,
                 initargs=(bounds, nodes_in_bbox, None, None)) as pool:
        results = pool.map(_w_stage2, _iter_payloads(source_path))
    kept_ways = _unique_concat([w for w, _ in results])
    extra_nodes = _unique_concat([e for _, e in results])
    kept_nodes = _unique_concat([nodes_in_bbox, extra_nodes])

    # Stage 3: relations.
    if keep_relations:
        with mp.Pool(workers, initializer=_winit,
                     initargs=(bounds, kept_nodes, kept_ways, None)) as pool:
            results = pool.map(_w_stage3, _iter_payloads(source_path))
        kept_rel = _unique_concat(results)
    else:
        kept_rel = np.empty(0, dtype=np.int64)

    # Write pass (imap preserves input order -> deterministic output bytes).
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        with mp.Pool(workers, initializer=_winit,
                     initargs=(bounds, kept_nodes, kept_ways, kept_rel)) as pool:
            for blob_bytes in pool.imap(_w_write, _iter_payloads(source_path)):
                if blob_bytes is not None:
                    out.write(blob_bytes)
    return output_path
