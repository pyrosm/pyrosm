"""Memory-efficient cropping of an ``*.osm.pbf`` by a bounding box (issue #6).

``crop_pbf`` streams the source file blob-by-blob and writes a valid, re-readable
OSM PBF holding only the data that falls inside (or completes) the crop box. It
never materializes the whole file: only compact id sets are held in memory.

Selection is "complete ways" (like osmconvert ``--complete-ways``): a way is kept
when at least one of its nodes is inside the box, and the kept way keeps its full
node list so geometries are not cut at the box edge. Relations are kept when they
reference a kept node or way. A polygon can be given in place of the box; a node is
then kept when it lies inside the polygon or on its boundary.

The id/coordinate re-encoding works in the raw integer (delta) space of the PBF,
so coordinates round-trip exactly (no rounding loss).

``merge_pbf`` merges several overlapping extracts into one file sorted by type
then id, de-duplicating the elements they share, and applies the same crop rule
to their union when a bounding box or a polygon is given.
"""

import os
import shutil
import tempfile
import zlib
from pathlib import Path
from struct import pack, unpack

import numpy as np

from google.protobuf.message import DecodeError
from pyrosm.exceptions import InvalidOSMFileError
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

# Primitive group kinds, in the order a file sorted by type then id holds them.
_NODES = 0
_WAYS = 1
_RELATIONS = 2


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


cdef _region(bounding_box, polygon):
    """The crop region: its box ``(xmin, ymin, xmax, ymax)`` and the prepared polygon
    to crop to, or None to crop to the box. A polygon's box is its envelope."""
    if polygon is None:
        return _bounds_from_bbox(bounding_box), None
    if bounding_box is not None:
        raise ValueError("Give either a bounding box or a polygon to crop to, not both.")
    if getattr(polygon, "geom_type", None) not in ("Polygon", "MultiPolygon") or \
            polygon.is_empty:
        raise ValueError(
            "The polygon to crop to must be a non-empty Shapely Polygon or "
            "MultiPolygon; got %r." % (polygon,)
        )
    return _bounds_from_bbox(polygon), _prepared(polygon.wkb)


def _prepared(wkb):
    """The polygon in `wkb`, prepared for fast point tests."""
    import shapely

    polygon = shapely.from_wkb(wkb)
    shapely.prepare(polygon)
    return polygon


# ---------------------------------------------------------------------------
# Blob-level I/O
# ---------------------------------------------------------------------------
# The OSM PBF spec caps a BlobHeader at 64 KiB and a blob at 32 MiB (compressed
# and uncompressed); larger declared sizes mean the file is not an OSM PBF (the
# limits osmium checks).
_MAX_BLOB_HEADER_SIZE = 64 * 1024
_MAX_BLOB_SIZE = 32 * 1024 * 1024
# Re-packed blocks are split above the 16 MiB the spec recommends as a blob's size.
_MAX_PACKED_BLOCK_SIZE = 16 * 1024 * 1024


cdef _invalid(filepath, reason):
    return InvalidOSMFileError(
        "'%s' is not a valid OSM PBF file. Pyrosm reads OpenStreetMap data in the "
        "OSM PBF format (https://wiki.openstreetmap.org/wiki/PBF_Format); this file "
        "does not follow the OSM PBF schema (%s)." % (filepath, reason)
    )


cdef _read_exact(f, n):
    data = f.read(n)
    if len(data) < n:
        raise _invalid(f.name, "the file is truncated")
    return data


cdef _parse(message, data, f, reason=None):
    """Parse `data` into `message`, naming the file of `f` when it is not valid."""
    try:
        message.ParseFromString(data)
    except DecodeError as err:
        raise _invalid(f.name, reason or err) from None
    return message


cdef _read_blob_header(f):
    """Read the next BlobHeader from `f`; None at EOF."""
    buf = f.read(4)
    if len(buf) == 0:
        return None
    if len(buf) < 4:
        raise _invalid(f.name, "the file is truncated")
    msg_len = unpack("!L", buf)[0]
    if msg_len > _MAX_BLOB_HEADER_SIZE:
        raise _invalid(
            f.name,
            "declared BlobHeader size %d exceeds the %d-byte maximum"
            % (msg_len, _MAX_BLOB_HEADER_SIZE),
        )
    blob_header = _parse(
        BlobHeader(), _read_exact(f, msg_len), f, "the BlobHeader could not be parsed"
    )
    if not 0 <= blob_header.datasize <= _MAX_BLOB_SIZE:
        raise _invalid(
            f.name,
            "declared blob size %d is outside 0-%d bytes"
            % (blob_header.datasize, _MAX_BLOB_SIZE),
        )
    return blob_header


cdef _read_blob(f, blob_header):
    """Read the raw or zlib Blob that follows `blob_header`."""
    blob = _parse(Blob(), _read_exact(f, blob_header.datasize), f)
    if not (blob.HasField("raw") or blob.HasField("zlib_data")):
        raise ValueError(
            "'%s' uses a blob compression other than raw and zlib, which pyrosm "
            "does not read." % f.name
        )
    return blob


cdef _decompress(data, raw_size, filepath):
    """Decompress a zlib blob of at most 32 MiB, checking it against `raw_size`."""
    decompressor = zlib.decompressobj()
    try:
        out = decompressor.decompress(data, _MAX_BLOB_SIZE)
    except zlib.error as err:
        raise _invalid(filepath, err) from None
    if decompressor.unconsumed_tail:
        raise _invalid(filepath, "a blob is larger than 32 MiB when decompressed")
    if not decompressor.eof:
        raise _invalid(filepath, "a zlib blob is truncated")
    if raw_size is not None and raw_size != len(out):
        raise _invalid(
            filepath,
            "a blob decompresses to %d bytes but declares raw_size %d"
            % (len(out), raw_size),
        )
    return out


cdef _raw_size(blob):
    return blob.raw_size if blob.HasField("raw_size") else None


cdef _read_next_blob(f):
    """Read one (BlobHeader, decompressed_bytes) from `f`; (None, None) at EOF."""
    blob_header = _read_blob_header(f)
    if blob_header is None:
        return None, None
    blob = _read_blob(f, blob_header)
    if blob.HasField("raw"):
        return blob_header, blob.raw
    return blob_header, _decompress(blob.zlib_data, _raw_size(blob), f.name)


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
            yield _parse(PrimitiveBlock(), data, f)


cpdef read_header_block(filepath):
    """Read the leading HeaderBlock of a PBF.

    Raises InvalidOSMFileError naming the file when it does not start with a
    valid OSMHeader block.
    """
    with open(filepath, "rb") as f:
        blob_header, data = _read_next_blob(f)
        if blob_header is None:
            raise _invalid(f.name, "the file is empty")
        if blob_header.type != "OSMHeader":
            raise _invalid(
                f.name, "first block is '%s', expected 'OSMHeader'" % blob_header.type
            )
        return _parse(HeaderBlock(), data, f)


cdef _read_header(filepath):
    """Parse + validate the leading HeaderBlock; reject unsupported features."""
    header = read_header_block(filepath)
    for feature in header.required_features:
        if feature in ("OsmSchema-V0.6", "DenseNodes"):
            continue
        if feature == "HistoricalInformation":
            reason = "history files (.osh.pbf) are not supported"
        elif feature == "LocationsOnWays":
            reason = "node locations stored on ways are not supported"
        else:
            reason = "its required feature '%s' is not supported" % feature
        raise ValueError("Cannot crop or merge '%s': %s." % (filepath, reason))
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
cdef _node_coords(pblock, g):
    """Absolute (ids, lons, lats) of the nodes of group `g` in degrees."""
    granularity = pblock.granularity
    lat_offset = pblock.lat_offset
    lon_offset = pblock.lon_offset
    if len(g.dense.id) > 0:
        dense = g.dense
        ids = np.cumsum(np.fromiter(dense.id, dtype=np.int64, count=len(dense.id)))
        lat_raw = np.cumsum(np.fromiter(dense.lat, dtype=np.int64, count=len(dense.lat)))
        lon_raw = np.cumsum(np.fromiter(dense.lon, dtype=np.int64, count=len(dense.lon)))
    else:
        n = len(g.nodes)
        ids = np.fromiter((node.id for node in g.nodes), dtype=np.int64, count=n)
        lat_raw = np.fromiter((node.lat for node in g.nodes), dtype=np.int64, count=n)
        lon_raw = np.fromiter((node.lon for node in g.nodes), dtype=np.int64, count=n)
    lats = (lat_raw * granularity + lat_offset) / DIV
    lons = (lon_raw * granularity + lon_offset) / DIV
    return ids, lons, lats


cdef _block_nodes_inside(pblock, bounds, polygon):
    """Ids of the nodes in `pblock` inside the box `bounds` and, when `polygon` is
    given, inside it or on its boundary."""
    xmin, ymin, xmax, ymax = bounds
    selected = []
    for g in pblock.primitivegroup:
        if len(g.dense.id) == 0 and len(g.nodes) == 0:
            continue
        ids, lons, lats = _node_coords(pblock, g)
        mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
        if polygon is not None and mask.any():
            import shapely

            mask[mask] = shapely.intersects_xy(polygon, lons[mask], lats[mask])
        if mask.any():
            selected.append(ids[mask])
    return _unique_concat(selected)


cdef _stage1_nodes_inside(filepath, region):
    bounds, polygon = region
    return _unique_concat(
        [_block_nodes_inside(pb, bounds, polygon) for pb in _iter_primitive_blocks(filepath)]
    )


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


cdef _block_way_copies(pblock, candidates, nodes_in_bbox_set):
    """The copies in `pblock` of the ways in `candidates`, as (ids, versions,
    timestamps, touches, ref_counts, refs): versions and timestamps are -1 where not
    recorded, `touches` marks copies with a node in the box, and `refs` holds the
    node ids of the touching copies back to back (`ref_counts` per copy, 0 for the
    others)."""
    ids, versions, timestamps, touches, ref_counts, refs = [], [], [], [], [], []
    for g in pblock.primitivegroup:
        for way in g.ways:
            if way.id not in candidates:
                continue
            ids.append(way.id)
            versions.append(way.info.version if way.info.HasField("version") else -1)
            timestamps.append(
                way.info.timestamp if way.info.HasField("timestamp") else -1
            )
            way_refs = np.cumsum(
                np.fromiter(way.refs, dtype=np.int64, count=len(way.refs))
            )
            touching = bool(_isin(way_refs, nodes_in_bbox_set).any())
            touches.append(touching)
            ref_counts.append(len(way_refs) if touching else 0)
            if touching:
                refs.append(way_refs)
    return (
        np.array(ids, dtype=np.int64),
        np.array(versions, dtype=np.int64),
        np.array(timestamps, dtype=np.int64),
        np.array(touches, dtype=bool),
        np.array(ref_counts, dtype=np.int64),
        np.concatenate(refs) if refs else np.empty(0, dtype=np.int64),
    )


cdef _way_winners(sources, touching, nodes_in_bbox, pool, tmpdir):
    """Keep a way only where its winning copy is a copy that touches the box.

    `touching` holds, per file, the ways whose copy in that file has a node in the
    box. Every copy of those ways is ranked as in the merge, so a way whose winning
    copy lies outside the box is dropped, and only the winning copies' nodes are
    kept. Returns the kept way ids of each file and the kept node ids.
    """
    candidates = _unique_concat(touching)
    if pool is None:
        candidates_set = _to_set(candidates)
        nib_set = _to_set(nodes_in_bbox)
        copies = [
            [
                _block_way_copies(pb, candidates_set, nib_set)
                for pb in _iter_primitive_blocks(p)
            ]
            for p in sources
        ]
    else:
        _broadcast(tmpdir, "candidate_ways", candidates)
        copies = [list(pool.imap(_w_way_copies, _iter_payloads(p))) for p in sources]
    blocks = [block for source_blocks in copies for block in source_blocks]
    ids, versions, timestamps, touches, ref_counts, refs = [
        np.concatenate([block[k] for block in blocks]) if blocks
        else np.empty(0, dtype=np.int64)
        for k in range(6)
    ]
    src = np.repeat(
        np.arange(len(sources)),
        [sum([len(block[0]) for block in source_blocks]) for source_blocks in copies],
    )

    winners = _winning_copies(ids, versions, timestamps, src)
    passing = np.zeros(len(ids), dtype=bool)
    passing[winners] = touches[winners].astype(bool)
    kept_ways = [ids[passing & (src == i)] for i in range(len(sources))]
    kept_refs = refs[np.repeat(passing, ref_counts)]
    return kept_ways, _unique_concat([nodes_in_bbox, kept_refs])


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


cdef _write_header(out, bounds, sorted_output=False):
    """Write the OSMHeader blob; `bounds` None leaves out the header bbox."""
    header = HeaderBlock()
    header.required_features.extend(["OsmSchema-V0.6", "DenseNodes"])
    if sorted_output:
        header.optional_features.append("Sort.Type_then_ID")
    header.writingprogram = "pyrosm"
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
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
    with open(filepath, "rb") as f:
        while True:
            blob_header = _read_blob_header(f)
            if blob_header is None:
                break
            if blob_header.type == "OSMData":
                n += 1
            f.seek(blob_header.datasize, 1)
    return n


cpdef crop_pbf(source_path, output_path, bounding_box=None, keep_relations=True,
               workers=1, compact=False, repack=False, polygon=None):
    """Crop `source_path` by `bounding_box`, or by the Shapely (Multi)Polygon
    `polygon`, writing a valid PBF to `output_path`.

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
    region = _region(bounding_box, polygon)
    bounds = region[0]

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_crop_")
        os.close(fd)

    # Stage 0: header pre-flight (rejects unsupported inputs before any streaming).
    _read_header(source_path)

    pool, tmpdir = _open_pool(workers, [source_path], region, compact)
    try:
        kept_nodes, kept_ways, kept_rel = _select(
            [source_path], region, keep_relations, pool, tmpdir
        )
        kept_ways, kept_rel = kept_ways[0], kept_rel[0]
        if repack:
            _repack_write(
                source_path, output_path, _to_set(kept_nodes), _to_set(kept_ways),
                _to_set(kept_rel), bounds
            )
        elif pool is not None:
            _broadcast(tmpdir, "kept_nodes", kept_nodes)
            _broadcast(tmpdir, "kept_ways", kept_ways)
            _broadcast(tmpdir, "kept_rel", kept_rel)
            # imap preserves input order -> the same bytes as the sequential write.
            with open(output_path, "wb") as out:
                _write_header(out, bounds)
                for blob_bytes in pool.imap(_w_write, _iter_payloads(source_path)):
                    if blob_bytes is not None:
                        out.write(blob_bytes)
        else:
            _write_pbf(
                source_path, output_path, _to_set(kept_nodes), _to_set(kept_ways),
                _to_set(kept_rel), bounds, compact
            )
    finally:
        _close_pool(pool, tmpdir)
    return output_path


cdef _select(sources, region, keep_relations, pool, tmpdir):
    """Run the crop selection stages over one or more PBF files.

    Returns the kept node ids of all files together, and a list with the kept way
    ids and a list with the kept relation ids of each file. With several files, a
    way is kept only from the file holding its winning copy (see `_way_winners`).
    With a `pool`, each stage spreads a file's blocks over the workers, which read
    the id sets the stage needs from `tmpdir`.
    """
    # Stage 1: nodes inside the bbox (or polygon).
    if pool is None:
        nodes_in_bbox = _unique_concat(
            [_stage1_nodes_inside(p, region) for p in sources]
        )
        nib_set = _to_set(nodes_in_bbox)
    else:
        nodes_in_bbox = _unique_concat(
            [ids for p in sources for ids in pool.imap(_w_stage1, _iter_payloads(p))]
        )
        _broadcast(tmpdir, "nodes_in_bbox", nodes_in_bbox)

    # Stage 2: ways with >=1 node in the bbox (+ all their refs -> complete ways).
    kept_ways = []
    kept_nodes = [nodes_in_bbox]
    for p in sources:
        if pool is None:
            ways, refs = _stage2_ways(p, nib_set)
        else:
            results = list(pool.imap(_w_stage2, _iter_payloads(p)))
            ways = _unique_concat([w for w, _ in results])
            refs = _unique_concat([r for _, r in results])
        kept_ways.append(ways)
        kept_nodes.append(refs)
    if len(sources) > 1:
        kept_ways, kept_nodes = _way_winners(
            sources, kept_ways, nodes_in_bbox, pool, tmpdir
        )
    else:
        kept_nodes = _unique_concat(kept_nodes)

    # Stage 3: relations referencing a kept node/way.
    if not keep_relations:
        return kept_nodes, kept_ways, [np.empty(0, dtype=np.int64) for _ in sources]
    all_ways = _unique_concat(kept_ways)
    if pool is None:
        kept_nodes_set = _to_set(kept_nodes)
        kept_ways_set = _to_set(all_ways)
        kept_rel = [_stage3_relations(p, kept_nodes_set, kept_ways_set) for p in sources]
    else:
        _broadcast(tmpdir, "kept_nodes", kept_nodes)
        _broadcast(tmpdir, "kept_ways", all_ways)
        kept_rel = [
            _unique_concat(list(pool.imap(_w_stage3, _iter_payloads(p))))
            for p in sources
        ]
    return kept_nodes, kept_ways, kept_rel


# ---------------------------------------------------------------------------
# Parallel path (workers > 1)
# ---------------------------------------------------------------------------
# The selection is staged (node -> way -> relation -> write); each stage is
# internally parallel across blobs but the stages run in sequence because each
# depends on the previous stage's *complete* result. A SINGLE pool is reused
# across all four stages (re-spawning a pool per stage would re-pay the worker
# startup cost four times). The main process reads raw (still-compressed) blob
# payloads sequentially (cheap I/O) and feeds them to the pool through `Pool.imap`
# as they are read; workers do the heavy decompress + protobuf parse + (re-)encode.
# The growing kept-id arrays a stage needs are broadcast to the persistent workers
# via small `.npy` files in a temp dir (written by the main process between stages,
# memory-mapped + cached per worker on first use) rather than re-pickled per task.
# Output blobs come back in input order (Pool.imap preserves order) so the written
# bytes are identical to the sequential path.

# Per-worker globals populated by `_winit`; `_W_CACHE` memoizes the khash sets
# built from the broadcast `.npy` files so each worker loads each set only once.
_W_BOUNDS = None
_W_POLYGON = None
_W_TMPDIR = None
_W_CACHE = {}
_W_COMPACT = False


cdef _read_next_payload(f):
    """Read one (blob_type, (filepath, is_raw, payload_bytes, raw_size)); (None, None)
    at EOF."""
    blob_header = _read_blob_header(f)
    if blob_header is None:
        return None, None
    blob = _read_blob(f, blob_header)
    if blob.HasField("raw"):
        return blob_header.type, (f.name, True, blob.raw, None)
    return blob_header.type, (f.name, False, blob.zlib_data, _raw_size(blob))


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
    filepath, is_raw, data, raw_size = payload
    if not is_raw:
        data = _decompress(data, raw_size, filepath)
    pblock = PrimitiveBlock()
    try:
        pblock.ParseFromString(data)
    except DecodeError as err:
        raise _invalid(filepath, err) from None
    return pblock


def _winit(bounds, polygon_wkb, tmpdir, compact):
    global _W_BOUNDS, _W_POLYGON, _W_TMPDIR, _W_CACHE, _W_COMPACT
    _W_BOUNDS = bounds
    _W_POLYGON = None if polygon_wkb is None else _prepared(polygon_wkb)
    _W_TMPDIR = tmpdir
    _W_CACHE = {}
    _W_COMPACT = compact


def _w_get_set(name):
    """Lazily load + cache the broadcast id set `name` from the temp dir."""
    s = _W_CACHE.get(name)
    if s is None:
        arr = np.load(Path(_W_TMPDIR) / (name + ".npy"))
        s = _to_set(arr)
        _W_CACHE[name] = s
    return s


def _w_stage1(payload):
    return _block_nodes_inside(_payload_to_block(payload), _W_BOUNDS, _W_POLYGON)


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


def _w_way_copies(payload):
    return _block_way_copies(
        _payload_to_block(payload),
        _w_get_set("candidate_ways"),
        _w_get_set("nodes_in_bbox"),
    )


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
    np.save(Path(tmpdir) / (name + ".npy"), np.ascontiguousarray(arr, dtype=np.int64))


cdef _open_pool(workers, sources, region, compact):
    """A worker pool and its broadcast temp dir, or (None, None) to run sequentially.

    Sequential when ``workers <= 1`` or the files have fewer than ``2 * workers``
    OSMData blocks in total.
    """
    if workers is None or workers <= 1:
        return None, None
    if sum([_count_data_blocks(p) for p in sources]) < 2 * int(workers):
        return None, None
    import multiprocessing as mp

    tmpdir = tempfile.mkdtemp(prefix="pyrosm_crop_par_")
    bounds, polygon = region
    wkb = None if polygon is None else polygon.wkb
    pool = mp.Pool(
        int(workers), initializer=_winit, initargs=(bounds, wkb, tmpdir, compact)
    )
    return pool, tmpdir


cdef _close_pool(pool, tmpdir):
    if pool is not None:
        pool.close()
        pool.join()
        shutil.rmtree(tmpdir, ignore_errors=True)


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

cdef _too_large(block, n_elements):
    """True when a re-packed block of several elements should be split in two."""
    return n_elements > 1 and block.ByteSize() > _MAX_PACKED_BLOCK_SIZE


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
    if _too_large(block, len(ids)):
        half = len(ids) // 2
        _emit_repack_node_block(
            out, ids[:half], lat_raw[:half], lon_raw[:half], tags_list[:half],
            _slice_meta(meta, slice(None, half)),
        )
        _emit_repack_node_block(
            out, ids[half:], lat_raw[half:], lon_raw[half:], tags_list[half:],
            _slice_meta(meta, slice(half, None)),
        )
        return
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
    if _too_large(block, len(way_batch)):
        half = len(way_batch) // 2
        _emit_repack_way_block(out, way_batch[:half])
        _emit_repack_way_block(out, way_batch[half:])
        return
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
    if _too_large(block, len(rel_batch)):
        half = len(rel_batch) // 2
        _emit_repack_relation_block(out, rel_batch[:half])
        _emit_repack_relation_block(out, rel_batch[half:])
        return
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
    """Dense group -> (ids, lat, lon, tags, meta) of the nodes in `kept_nodes_set`.

    Every node is kept when `kept_nodes_set` is None; None when no node is kept.
    """
    cdef int n = len(dense.id)
    ids = np.cumsum(np.fromiter(dense.id, dtype=np.int64, count=n))
    if kept_nodes_set is None:
        mask = np.ones(n, dtype=bool)
    else:
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
        meta["user"] = np.array([stringtable[s] for s in sids.tolist()], dtype=object)
    if len(di.visible) > 0:
        meta["visible"] = np.array(list(di.visible), dtype=bool)[mask]
    if not meta:
        meta = None
    return kept_ids, lat, lon, tags_list, meta


cdef _plain_nodes_chunk(run):
    """(ids, lat, lon, tags, meta) of (node, tags, info) triples sharing a schema."""
    ids = np.asarray([node.id for node, _, _ in run], dtype=np.int64)
    lat = np.asarray([node.lat for node, _, _ in run], dtype=np.int64)
    lon = np.asarray([node.lon for node, _, _ in run], dtype=np.int64)
    tags_list = [tags for _, tags, _ in run]
    metas = [info for _, _, info in run]
    meta = None
    if metas[0] is not None:
        meta = {}
        for key in metas[0].keys():
            if key == "user":
                meta["user"] = np.array([m["user"] for m in metas], dtype=object)
            elif key == "visible":
                meta["visible"] = np.asarray([m["visible"] for m in metas], dtype=bool)
            else:
                meta[key] = np.asarray([m[key] for m in metas], dtype=np.int64)
    return ids, lat, lon, tags_list, meta


cdef _decode_kept_plain_nodes(nodes, stringtable, kept_nodes_set):
    """Non-dense node group -> chunks of the dense chunk shape, one per run of
    consecutive kept nodes with the same metadata schema (a dense block's DenseInfo
    is all-or-nothing per field)."""
    chunks, run = [], []
    for node in nodes:
        if kept_nodes_set is not None and node.id not in kept_nodes_set:
            continue
        tags = {stringtable[k]: stringtable[v]
                for k, v in zip(node.keys, node.vals)} if len(node.keys) > 0 else None
        info = _decode_info(node.info)
        if info is not None and "_user_sid" in info:
            info["user"] = stringtable[info.pop("_user_sid")]
        if run and _meta_schema(info) != _meta_schema(run[-1][2]):
            chunks.append(_plain_nodes_chunk(run))
            run = []
        run.append((node, tags, info))
    if run:
        chunks.append(_plain_nodes_chunk(run))
    return chunks


cdef _decode_kept_ways(ways, stringtable, kept_ways_set):
    records = []
    for way in ways:
        if kept_ways_set is not None and way.id not in kept_ways_set:
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
        if kept_rel_set is not None and rel.id not in kept_rel_set:
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
    """Concatenate the metadata of chunks that share one metadata schema."""
    if metas[0] is None:
        return None
    return {key: np.concatenate([m[key] for m in metas]) for key in metas[0]}


cdef _slice_meta(meta, sel):
    """The metadata at `sel` (a slice or an index array)."""
    if meta is None:
        return None
    return {k: v[sel] for k, v in meta.items()}


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

    cdef add(self, kind, chunk):
        if kind == _NODES:
            self.add_nodes(chunk)
        elif kind == _WAYS:
            self.add_ways(chunk)
        else:
            self.add_relations(chunk)

    cdef add_nodes(self, chunk):
        # A dense block's DenseInfo is per-field all-or-nothing, so the nodes of one
        # block share a metadata schema: a chunk with another schema starts a new block.
        if self.node_chunks and \
                _meta_schema(chunk[4]) != _meta_schema(self.node_chunks[0][4]):
            self._flush_nodes()
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
            tags[:_MAX_GROUP], _slice_meta(meta, slice(None, _MAX_GROUP)))
        rem = (ids[_MAX_GROUP:], lat[_MAX_GROUP:], lon[_MAX_GROUP:],
               tags[_MAX_GROUP:], _slice_meta(meta, slice(_MAX_GROUP, None)))
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


cdef _check_standard_grid(pblock, filepath):
    if (pblock.granularity != 100 or pblock.lat_offset != 0
            or pblock.lon_offset != 0 or pblock.date_granularity != 1000):
        raise ValueError(
            "to_pbf(repack=True) and merge_pbf() require the standard PBF grid "
            "(granularity 100, zero lat/lon offsets, date_granularity 1000), which "
            "'%s' does not use; to_pbf(repack=False) can still crop it." % filepath
        )


def _iter_groups(filepath):
    """Yield (kind, string table, group) for each primitive group of a PBF."""
    for pblock in _iter_primitive_blocks(filepath):
        _check_standard_grid(pblock, filepath)
        st = pblock.stringtable.s
        for g in pblock.primitivegroup:
            if len(g.dense.id) > 0 or len(g.nodes) > 0:
                yield _NODES, st, g
            elif len(g.ways) > 0:
                yield _WAYS, st, g
            elif len(g.relations) > 0:
                yield _RELATIONS, st, g


cdef _decode_group(kind, st, g, kept):
    """Decoded chunks of the elements of `g` in the id set `kept` (all when None);
    an empty list when none is kept."""
    if kind == _NODES:
        if len(g.dense.id) > 0:
            chunk = _decode_kept_dense_nodes(g.dense, st, kept)
            return [chunk] if chunk is not None else []
        return _decode_kept_plain_nodes(g.nodes, st, kept)
    if kind == _WAYS:
        records = _decode_kept_ways(g.ways, st, kept)
    else:
        records = _decode_kept_relations(g.relations, st, kept)
    return [records] if records else []


cdef _repack_write(source_path, output_path, kept_nodes_set, kept_ways_set,
                   kept_rel_set, bounds):
    """Sequential re-pack write: re-chunk the kept crop into canonical full blocks."""
    kept = (kept_nodes_set, kept_ways_set, kept_rel_set)
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        writer = _RepackWriter(out)
        for kind, st, g in _iter_groups(source_path):
            for chunk in _decode_group(kind, st, g, kept[kind]):
                writer.add(kind, chunk)
        writer.close()
    return output_path


# ---------------------------------------------------------------------------
# Merge several extracts into one sorted, de-duplicated PBF
# ---------------------------------------------------------------------------
# Each input is read in file order, one element type at a time. Its decoded
# chunks are merged in id order: every id up to the smallest last id of the
# inputs' current chunks is complete, because the later chunks of each input
# hold only larger ids. The copies of one id are ranked by version, timestamp
# and input order, and the winners go through the re-pack writer.

cdef _not_sorted(filepath):
    return ValueError(
        "'%s' is not sorted by type then id, which merging requires. Sort it first, "
        "e.g. with `osmium sort`." % filepath
    )


cdef _chunk_ids(kind, chunk):
    if kind == _NODES:
        return chunk[0]
    return np.array([r["id"] for r in chunk], dtype=np.int64)


cdef _chunk_ranks(kind, chunk):
    """(versions, timestamps) of a decoded chunk, -1 where the metadata is missing."""
    if kind == _NODES:
        meta = chunk[4] or {}
        missing = np.full(len(chunk[0]), -1, dtype=np.int64)
        return meta.get("version", missing), meta.get("timestamp", missing)
    infos = [r["info"] or {} for r in chunk]
    return (
        np.array([info.get("version", -1) for info in infos], dtype=np.int64),
        np.array([info.get("timestamp", -1) for info in infos], dtype=np.int64),
    )


cdef _take(kind, chunk, idx):
    """The elements of a decoded chunk at the positions `idx`."""
    if kind == _NODES:
        ids, lat, lon, tags, meta = chunk
        return ids[idx], lat[idx], lon[idx], [tags[i] for i in idx], _slice_meta(meta, idx)
    return [chunk[i] for i in idx]


cdef _group_ids(kind, g):
    """Ids of all elements of a primitive group, in file order."""
    if kind == _NODES:
        if len(g.dense.id) > 0:
            return np.cumsum(np.fromiter(g.dense.id, dtype=np.int64, count=len(g.dense.id)))
        return np.array([node.id for node in g.nodes], dtype=np.int64)
    if kind == _WAYS:
        return np.array([way.id for way in g.ways], dtype=np.int64)
    return np.array([rel.id for rel in g.relations], dtype=np.int64)


class _SortedInput:
    """One merge input, read group by group and checked to be sorted by type then id."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._groups = _iter_groups(filepath)
        self._next = next(self._groups, None)

    def chunks(self, kind, kept):
        """Yield (chunk, ids, versions, timestamps) for the elements of `kind` in `kept`."""
        last_id = None
        while self._next is not None and self._next[0] <= kind:
            group_kind, st, g = self._next
            if group_kind < kind:
                raise _not_sorted(self.filepath)
            self._next = next(self._groups, None)
            # Every element counts for the order, not only the kept ones.
            group_ids = _group_ids(kind, g)
            if (group_ids[1:] <= group_ids[:-1]).any() or (
                last_id is not None and group_ids[0] <= last_id
            ):
                raise _not_sorted(self.filepath)
            last_id = group_ids[-1]
            for chunk in _decode_group(kind, st, g, kept):
                versions, timestamps = _chunk_ranks(kind, chunk)
                yield chunk, _chunk_ids(kind, chunk), versions, timestamps


cdef _winning_copies(ids, versions, timestamps, src):
    """Positions of the winning copy of each id, in id order.

    The copy with the highest version wins, then the one with the latest timestamp,
    then the one from the earliest input (lowest `src`).
    """
    order = np.lexsort((src, -timestamps, -versions, ids))
    first = np.ones(len(order), dtype=bool)
    first[1:] = ids[order[1:]] != ids[order[:-1]]
    return order[first]


cdef _write_winners(_RepackWriter writer, kind, chunk, won, i):
    """Write winning elements of input `i`; with `won`, only those selected in it."""
    if won is not None:
        selected = won[i]
        chunk = [r for r in chunk if r["id"] in selected]
        if not chunk:
            return
    writer.add(kind, chunk)


cdef _merge_kind(inputs, kind, kept, won, _RepackWriter writer):
    """Write the elements of `kind` of all inputs, one copy per id, in id order."""
    streams = [inp.chunks(kind, kept) for inp in inputs]
    heads = [next(s, None) for s in streams]
    while True:
        live = [i for i in range(len(heads)) if heads[i] is not None]
        if not live:
            return
        if len(live) == 1:
            i = live[0]
            _write_winners(writer, kind, heads[i][0], won, i)
            heads[i] = next(streams[i], None)
            continue

        frontier = min([heads[i][1][-1] for i in live])
        counts = [int(np.searchsorted(heads[i][1], frontier, side="right")) for i in live]
        src = np.repeat(np.array(live), counts)
        pos = np.concatenate([np.arange(n) for n in counts])
        ids = np.concatenate([heads[i][1][:n] for i, n in zip(live, counts)])
        versions = np.concatenate([heads[i][2][:n] for i, n in zip(live, counts)])
        timestamps = np.concatenate([heads[i][3][:n] for i, n in zip(live, counts)])

        winners = _winning_copies(ids, versions, timestamps, src)
        cuts = np.flatnonzero(src[winners[1:]] != src[winners[:-1]]) + 1
        for run in np.split(winners, cuts):
            i = src[run[0]]
            _write_winners(writer, kind, _take(kind, heads[i][0], pos[run]), won, i)

        for i, n in zip(live, counts):
            chunk, chunk_ids, chunk_versions, chunk_timestamps = heads[i]
            if n == len(chunk_ids):
                heads[i] = next(streams[i], None)
            elif n > 0:
                rest = np.arange(n, len(chunk_ids))
                heads[i] = (
                    _take(kind, chunk, rest), chunk_ids[n:], chunk_versions[n:],
                    chunk_timestamps[n:],
                )


cdef _header_bounds(headers):
    """Union of the header bounding boxes, or None when no header has one."""
    boxes = np.array(
        [(h.bbox.left, h.bbox.bottom, h.bbox.right, h.bbox.top)
         for h in headers if h.HasField("bbox")],
        dtype=np.float64,
    ) / DIV
    if len(boxes) == 0:
        return None
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


cdef _fingerprint(path):
    """Identity of a path and of the file it resolves to, and that file's size and
    modification time.

    Change times are left out: sync clients such as OneDrive update them when they
    touch a file's metadata, which leaves its contents unchanged.
    """
    link, stat = os.lstat(path), os.stat(path)
    return (
        link.st_dev, link.st_ino,
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )


cpdef merge_pbf(inputs, output_path=None, bounding_box=None, keep_relations=True,
                workers=1, polygon=None):
    """
    Merge overlapping ``*.osm.pbf`` extracts into one PBF, optionally cropped
    to a bounding box or a polygon.

    The inputs may come from different providers, have snapshots days apart and
    carry element metadata or not. An element found in several inputs is written
    once: the copy with the highest ``version`` wins, then the one with the latest
    ``timestamp``, then the one from the earliest input. The output is sorted by
    type then id (header feature ``Sort.Type_then_ID``) and written in densely
    packed blocks. The inputs are streamed block by block; only id sets are held
    in memory.

    Without ``bounding_box`` or ``polygon`` the output holds everything the inputs
    hold. With ``bounding_box``, the crop rule of :meth:`OSM.to_pbf` is applied to the union of the
    inputs: a node is kept when a copy of it lies inside the box, a way when its
    winning copy has a node inside the box, and a relation when its winning copy
    references a kept node or way. A kept way keeps its full node list, whichever
    input holds those nodes. With ``polygon`` the same rule applies with the
    polygon in place of the box. A single input is cropped the same way.

    Parameters
    ----------

    inputs : list of str or path-like
        The PBF files to merge, each sorted by type then id (``osmium sort``
        sorts a file that is not). History files and files storing node
        locations on ways are not supported.

    output_path : str or path-like, optional
        Where to write the merged PBF. When ``None`` (default) a temporary file
        is created in the system temp directory and its path returned.

    bounding_box : list or shapely geometry, optional
        ``[minx, miny, maxx, maxy]`` in lon/lat, or a ``Polygon``/``MultiPolygon``,
        which is cropped by its envelope. The output's header bounding box is
        this box (the polygon's envelope with ``polygon``), or else the union of
        the inputs' header boxes.

    keep_relations : bool
        When ``True`` (default) relations are written (with ``bounding_box`` or
        ``polygon``, those referencing a kept node or way); when ``False`` none
        are written.

    workers : int
        Number of worker processes for selecting the elements inside
        ``bounding_box`` or ``polygon``. ``1`` (default) runs sequentially. The
        merged file is written sequentially either way.

    polygon : shapely Polygon or MultiPolygon, optional
        The area to crop to, in lon/lat, in place of ``bounding_box``: a node is
        kept when it lies inside the polygon or on its boundary. The output's
        header bounding box is the polygon's envelope.

    Returns
    -------
    str or path-like
        The path of the written PBF file.

    Raises
    ------
    ValueError
        When an input cannot be read as a PBF, is not supported, or is not sorted
        by type then id, or when an input's size or modification time changes (or
        the file is replaced) while the merge runs; the message names the file.
        When both ``bounding_box`` and ``polygon`` are given, or ``polygon`` is not
        a non-empty Polygon or MultiPolygon.

    Examples
    --------
    >>> import pyrosm
    >>> out = pyrosm.merge_pbf(
    ...     ["switzerland-latest.osm.pbf", "france-latest.osm.pbf"],
    ...     "basel.osm.pbf",
    ...     bounding_box=[7.52, 47.51, 7.66, 47.60],
    ... )
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    sources = [os.fspath(p) for p in inputs]
    if not sources:
        raise ValueError("merge_pbf() needs at least one input file.")
    fingerprints = [_fingerprint(p) for p in sources]
    headers = [_read_header(p) for p in sources]

    if output_path is not None and Path(output_path).exists():
        clash = [p for p in sources if os.path.samefile(output_path, p)]
        if clash:
            raise ValueError(
                "The output path '%s' is the input '%s'." % (output_path, clash[0])
            )

    if bounding_box is None and polygon is None:
        bounds = _header_bounds(headers)
        kept = (None, None, None if keep_relations else Int64Set())
        won = (None, None, None)
    else:
        region = _region(bounding_box, polygon)
        bounds = region[0]
        pool, tmpdir = _open_pool(workers, sources, region, False)
        try:
            kept_nodes, kept_ways, kept_rel = _select(
                sources, region, keep_relations, pool, tmpdir
            )
        finally:
            _close_pool(pool, tmpdir)
        kept = (
            _to_set(kept_nodes),
            _to_set(_unique_concat(kept_ways)),
            _to_set(_unique_concat(kept_rel)),
        )
        won = (None, [_to_set(w) for w in kept_ways], [_to_set(r) for r in kept_rel])

    # Written next to the output and moved into place when complete, so a failed
    # merge leaves no partial file at `output_path`.
    if output_path is None:
        out_dir = tempfile.gettempdir()
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
    fd, partial_path = tempfile.mkstemp(
        suffix=".osm.pbf", prefix=".pyrosm_merge_", dir=out_dir
    )
    try:
        with os.fdopen(fd, "wb") as out:
            inputs = [_SortedInput(p) for p in sources]
            _write_header(out, bounds, sorted_output=True)
            writer = _RepackWriter(out)
            for kind in (_NODES, _WAYS, _RELATIONS):
                _merge_kind(inputs, kind, kept[kind], won[kind], writer)
            writer.close()
        changed = [
            p for p, before in zip(sources, fingerprints) if _fingerprint(p) != before
        ]
        if changed:
            raise ValueError(
                "'%s' changed while it was being merged; the merge was discarded."
                % changed[0]
            )
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_merge_")
            os.close(fd)
        os.replace(partial_path, output_path)
    except BaseException:
        Path(partial_path).unlink(missing_ok=True)
        raise
    return output_path
