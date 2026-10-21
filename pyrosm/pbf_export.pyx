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

import contextlib
import logging
import os
import shutil
import tempfile
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from itertools import chain
from operator import itemgetter
from pathlib import Path
from struct import pack, unpack

import numpy as np

cimport cython
from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t, uint64_t, uintptr_t
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdlib cimport free, realloc
from libc.string cimport memcpy, memset

from google.protobuf.message import DecodeError
from pyrosm.exceptions import InvalidOSMFileError
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob
from pyrosm.proto.osmformat_pb2 import (
    HeaderBlock, PrimitiveBlock, PrimitiveGroup, StringTable,
)
from pyrosm.delta_compression cimport delta_encode
from pyrosm.primitive_block_decoder cimport (
    DENSE_DENSEINFO, DENSE_ID, DENSE_KEYS_VALS, DENSE_LAT, DENSE_LON, DI_CHANGESET,
    DI_TIMESTAMP, DI_UID, DI_USER_SID, DI_VERSION, DI_VISIBLE, GROUP_DENSE, GROUP_NODES,
    GROUP_RELATIONS, GROUP_WAYS, INFO_CHANGESET, INFO_TIMESTAMP, INFO_UID, INFO_USER_SID,
    INFO_VERSION, INFO_VISIBLE, PB_DATE_GRANULARITY, PB_GRANULARITY, PB_LAT_OFFSET,
    PB_LON_OFFSET, PB_PRIMITIVEGROUP, PB_STRINGTABLE, REL_MEMIDS, REL_ROLES_SID,
    REL_TYPES, ST_S, WAY_ID, WAY_INFO, WAY_KEYS, WAY_REFS, WAY_VALS, NODE_ID, NODE_INFO,
    WIRE_LENGTH, WIRE_VARINT, _Runs, _add_span, _field_end, _fields_end,
    _read_element_id, _read_tag, _read_varint, _runs_count, _runs_fill, _skip_field,
)

from pyrosm.engine.blobs import (
    _MAX_BLOB_SIZE, _decompress_blob, _inflated_batch, _invalid_pbf, _read_batches,
    _read_span,
)
from pyrosm._arrays import _offsets_from_lengths
from pyrosm.engine.pool import _PoolStartError, _ordered_map
from pyrosm.int64_hash cimport Int64Set, SortedInt64Set, _set_has, kh_i64set_t
from pyrosm._log import timed
from pyrosm.utils import _file_identity, default_workers, validate_workers
from pyrosm.utils.progress import stepping, validate_progress
from pyrosm.zlib_blob import zlib_compress
from pyrosm.primitive_block_decoder import (
    BLOCK_COUNTS, check_batch_starts, decode_batch_nogil, decode_elements,
    decode_primitive_block, element_layout,
)

logger = logging.getLogger(__name__)

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
# Re-packed blocks are split above the 16 MiB the spec recommends as a blob's size.
_MAX_PACKED_BLOCK_SIZE = 16 * 1024 * 1024


cdef _read_exact(f, n):
    """The next ``n`` bytes of the file ``f``; a file that ends sooner is truncated."""
    data = f.read(n)
    if len(data) < n:
        raise _invalid_pbf(f.name, "the file is truncated")
    return data


cdef _parse(message, data, filepath, reason=None):
    """Parse `data` into `message`, naming `filepath` when it is not valid."""
    try:
        message.ParseFromString(data)
    except DecodeError as err:
        raise _invalid_pbf(filepath, reason or err) from None
    return message


cdef _read_blob_header(f):
    """Read the next BlobHeader from `f`; None at EOF."""
    buf = f.read(4)
    if len(buf) == 0:
        return None
    if len(buf) < 4:
        raise _invalid_pbf(f.name, "the file is truncated")
    msg_len = unpack("!L", buf)[0]
    if msg_len > _MAX_BLOB_HEADER_SIZE:
        raise _invalid_pbf(
            f.name,
            "declared BlobHeader size %d exceeds the %d-byte maximum"
            % (msg_len, _MAX_BLOB_HEADER_SIZE),
        )
    blob_header = _parse(
        BlobHeader(), _read_exact(f, msg_len), f.name, "the BlobHeader could not be parsed"
    )
    if not 0 <= blob_header.datasize <= _MAX_BLOB_SIZE:
        raise _invalid_pbf(
            f.name,
            "declared blob size %d is outside 0-%d bytes"
            % (blob_header.datasize, _MAX_BLOB_SIZE),
        )
    return blob_header


cdef _read_blob(f, datasize):
    """Read the raw or zlib Blob of `datasize` bytes at the position of `f`."""
    blob = _parse(Blob(), _read_exact(f, datasize), f.name)
    if not (blob.HasField("raw") or blob.HasField("zlib_data")):
        raise ValueError(
            "'%s' uses a blob compression other than raw and zlib, which pyrosm "
            "does not read." % f.name
        )
    return blob


cdef _decompress(data, raw_size, filepath):
    """Decompress a zlib blob of at most 32 MiB, checking it against `raw_size` (see
    `_decompress_blob`)."""
    return _decompress_blob(data, raw_size, filepath)


cdef _raw_size(blob):
    return blob.raw_size if blob.HasField("raw_size") else None


cdef _read_next_blob(f):
    """Read one (BlobHeader, decompressed_bytes) from `f`; (None, None) at EOF."""
    blob_header = _read_blob_header(f)
    if blob_header is None:
        return None, None
    blob = _read_blob(f, blob_header.datasize)
    if blob.HasField("raw"):
        return blob_header, blob.raw
    return blob_header, _decompress(blob.zlib_data, _raw_size(blob), f.name)


# The passes of a crop or merge over its input files, the steps of its progress, and
# their shares of it, from their times in crops of Finland to a city and in merges of
# two overlapping extracts of Helsinki with a box.
_SELECTING = "selecting the elements"
_FINDING_NODES = "finding the nodes in the area"
_FINDING_WAYS = "finding the ways"
_CHOOSING_WAYS = "choosing each way's copy"
_FINDING_RELATIONS = "finding the relations"
_WRITING = "writing the file"
_CROP_SHARES = {
    _SELECTING: 1.1,
    _FINDING_NODES: 1.1,
    _FINDING_WAYS: 1.4,
    _FINDING_RELATIONS: 0.4,
    _WRITING: 1.0,
}
_MERGE_SHARES = {
    _FINDING_NODES: 0.5,
    _FINDING_WAYS: 0.6,
    _CHOOSING_WAYS: 1.0,
    _FINDING_RELATIONS: 0.3,
    _WRITING: 3.5,
}


class _ReadProgress:
    """The passes of a crop or merge over its input files as steps of `steps` (a
    `pyrosm.utils.progress.Steps`, or None): each pass is a named step, moved through
    its share by the bytes of the files read in it (out of the files' sizes at the
    start)."""

    def __init__(self, steps, paths):
        self.steps = steps
        self.total = sum(Path(path).stat().st_size for path in paths)
        self.done = {}
        self.advance_step = None

    def plan(self, names):
        """The passes still to come, by name (see `Steps.plan`)."""
        if self.steps is not None:
            self.steps.plan(names)

    @contextlib.contextmanager
    def one_pass(self, name):
        """A pass over the files, the step `name`."""
        self.done = {}
        if self.steps is not None:
            self.advance_step = self.steps.start_step(name)
        try:
            yield
        finally:
            self._report_progress(self.total)

    def advance(self, reader, position):
        """The file `reader` (any object that stands for one reading of a file, as a
        file object does) has been read up to `position` in this pass."""
        self.done[reader] = position
        self._report_progress(sum(self.done.values()))

    def _report_progress(self, done):
        if self.advance_step is not None:
            self.advance_step(min(done, self.total), self.total)


def _iter_block_data(filepath, progress=None):
    """Yield the decompressed bytes of each OSMData block (skips the leading OSMHeader),
    reporting the bytes read to `progress` (a `_ReadProgress`)."""
    with open(filepath, "rb") as f:
        _read_next_blob(f)  # header blob, validated separately in _read_header
        while True:
            if progress is not None:
                progress.advance(f, f.tell())
            blob_header, data = _read_next_blob(f)
            if blob_header is None:
                break
            if blob_header.type == "OSMData":
                yield data


cdef _raw_decoded(decode, data, filepath, args=()):
    """`decode(block, *args)`, a function of the raw block decoder, applied to the
    decompressed block `data` of `filepath`; a corrupt block raises the error naming the
    file."""
    try:
        return decode(np.frombuffer(data, dtype=np.uint8), *args)
    except InvalidOSMFileError as err:
        raise _invalid_pbf(filepath, err) from None


cdef _decode_block(data, filepath):
    """The header, nodes, ways and relations of the decompressed block `data` of
    `filepath`, decoded without the elements' metadata (see `decode_primitive_block`)."""
    _, header, nodes, ways, relations = _raw_decoded(
        decode_primitive_block, data, filepath, (False,)
    )
    return header, nodes, ways, relations


def _iter_primitive_blocks(filepath, progress=None):
    """Yield each parsed OSMData `PrimitiveBlock` (skips the leading OSMHeader)."""
    for data in _iter_block_data(filepath, progress):
        yield _parse(PrimitiveBlock(), data, filepath)


def _iter_decoded_blocks(filepath, progress=None):
    """Yield each OSMData block decoded by `_decode_block`."""
    for data in _iter_block_data(filepath, progress):
        yield _decode_block(data, filepath)


cpdef read_header_block(filepath):
    """Read the leading HeaderBlock of a PBF.

    Raises InvalidOSMFileError naming the file when it does not start with a
    valid OSMHeader block.
    """
    with open(filepath, "rb") as f:
        blob_header, data = _read_next_blob(f)
        if blob_header is None:
            raise _invalid_pbf(f.name, "the file is empty")
        if blob_header.type != "OSMHeader":
            raise _invalid_pbf(
                f.name, "first block is '%s', expected 'OSMHeader'" % blob_header.type
            )
        return _parse(HeaderBlock(), data, f.name)


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
# Id-set helpers (int64 hash sets for memory-efficient membership)
# ---------------------------------------------------------------------------
cdef _to_set(id_array):
    return Int64Set(id_array)


cdef _to_sorted_set(id_array):
    """The kept node ids `id_array` as the set the write passes test each block's
    ascending node ids against."""
    return SortedInt64Set(id_array)


cdef _isin(values, lookup):
    return lookup.isin(values)


cdef _unique_concat(arrays):
    """The distinct ids of the id arrays `arrays`, in no particular order."""
    if len(arrays) == 0:
        return np.empty(0, dtype=np.int64)
    return Int64Set(np.concatenate(arrays)).values()


# ---------------------------------------------------------------------------
# Selection stages (each re-streams the whole file, inspecting one element type)
# ---------------------------------------------------------------------------
cdef _nodes_inside(header, nodes, bounds, polygon):
    """Ids of the decoded `nodes` of a block inside the box `bounds` and, when `polygon`
    is given, inside it or on its boundary."""
    if nodes is None or len(nodes["id"]) == 0:
        return np.empty(0, dtype=np.int64)
    xmin, ymin, xmax, ymax = bounds
    lats = (nodes["lat"] * header["granularity"] + header["lat_offset"]) / DIV
    lons = (nodes["lon"] * header["granularity"] + header["lon_offset"]) / DIV
    mask = (xmin <= lons) & (lons <= xmax) & (ymin <= lats) & (lats <= ymax)
    if polygon is not None and mask.any():
        import shapely

        mask[mask] = shapely.intersects_xy(polygon, lons[mask], lats[mask])
    return nodes["id"][mask]


cdef _any_member(hits, offsets):
    """Whether each element has a hit among its members: `hits` holds one value per
    member, laid out by the elements' CSR `offsets`."""
    counts = np.diff(offsets)
    out = np.zeros(len(counts), dtype=bool)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.logical_or.reduceat(hits, offsets[:-1][nonempty])
    return out


cdef _ways_touching(ways, nodes_set):
    """Ids of the decoded `ways` of a block with a node in `nodes_set`, and the nodes of
    those ways."""
    if ways is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    refs, offsets = ways["refs"], ways["refs_off"]
    touching = _any_member(_isin(refs, nodes_set), offsets)
    return ways["id"][touching], refs[np.repeat(touching, np.diff(offsets))]


cdef _relations_touching(relations, nodes_set, ways_set):
    """Ids of the decoded `relations` of a block with a node member in `nodes_set` or a
    way member in `ways_set`."""
    if relations is None:
        return np.empty(0, dtype=np.int64)
    memids, types = relations["memids"], relations["types"]
    hits = np.zeros(len(memids), dtype=bool)
    is_node = types == _MEMBER_NODE
    is_way = types == _MEMBER_WAY
    hits[is_node] = _isin(memids[is_node], nodes_set)
    hits[is_way] = _isin(memids[is_way], ways_set)
    return relations["id"][_any_member(hits, relations["members_off"])]


cdef _nodes_inside_file(filepath, region, progress):
    bounds, polygon = region
    return _unique_concat([
        _nodes_inside(header, nodes, bounds, polygon)
        for header, nodes, _, _ in _iter_decoded_blocks(filepath, progress)
    ])


cdef _ways_touching_file(filepath, nodes_in_bbox_set, progress):
    kept_way_ids = []
    extra_nodes = []
    for _, _, ways, _ in _iter_decoded_blocks(filepath, progress):
        ids, refs = _ways_touching(ways, nodes_in_bbox_set)
        kept_way_ids.append(ids)
        extra_nodes.append(refs)
    return np.concatenate(kept_way_ids or [np.empty(0, dtype=np.int64)]), _unique_concat(
        extra_nodes
    )


cdef _relations_touching_file(filepath, kept_nodes_set, kept_ways_set, progress):
    return np.concatenate([
        _relations_touching(relations, kept_nodes_set, kept_ways_set)
        for _, _, _, relations in _iter_decoded_blocks(filepath, progress)
    ] or [np.empty(0, dtype=np.int64)])


cdef _select_one_pass(filepath, region, keep_relations, progress):
    """The crop selection of `filepath` in one pass, as (kept node ids, [kept way ids],
    [kept relation ids]) like `_select`, for a file whose blocks hold every node before
    every way and every way before every relation, as a file sorted by type then id
    does. None when a block breaks that order."""
    bounds, polygon = region
    inside, way_ids, way_refs, rel_ids = [], [], [], []
    nodes_set = kept_nodes_set = kept_ways_set = None
    seen_ways = seen_relations = False
    for header, nodes, ways, relations in _iter_decoded_blocks(filepath, progress):
        has_nodes = nodes is not None and len(nodes["id"]) > 0
        if (has_nodes and seen_ways) or (
            (has_nodes or ways is not None) and seen_relations
        ):
            return None
        if has_nodes:
            inside.append(_nodes_inside(header, nodes, bounds, polygon))
        if ways is not None:
            if nodes_set is None:
                nodes_set = _to_set(_unique_concat(inside))
            ids, refs = _ways_touching(ways, nodes_set)
            way_ids.append(ids)
            way_refs.append(refs)
            seen_ways = True
        if relations is not None:
            seen_relations = True
            if keep_relations:
                if kept_nodes_set is None:
                    kept_nodes_set = _to_set(_unique_concat(inside + way_refs))
                    kept_ways_set = _to_set(np.concatenate(
                        way_ids or [np.empty(0, dtype=np.int64)]))
                rel_ids.append(
                    _relations_touching(relations, kept_nodes_set, kept_ways_set)
                )
    empty = np.empty(0, dtype=np.int64)
    return (
        _unique_concat(inside + way_refs),
        [np.concatenate(way_ids or [empty])],
        [np.concatenate(rel_ids or [empty])],
    )


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


cdef _way_winners(sources, touching, nodes_in_bbox, pool, progress):
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
                for pb in _iter_primitive_blocks(p, progress)
            ]
            for p in sources
        ]
    else:
        pool.broadcast("candidate_ways", candidates)
        copies = [
            list(pool.imap(_worker_way_copies, _iter_payloads(p, progress)))
            for p in sources
        ]
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
# The DenseInfo fields as (field number, name, delta coded); the delta-coded ones are
# also zig-zag encoded (``sint``).
_DENSE_INFO = (
    (DI_VERSION, "version", False),
    (DI_TIMESTAMP, "timestamp", True),
    (DI_CHANGESET, "changeset", True),
    (DI_UID, "uid", True),
    (DI_USER_SID, "user_sid", True),
    (DI_VISIBLE, "visible", False),
)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _write_varint(unsigned char* out, uint64_t x) noexcept nogil:
    """Write `x` as a protobuf varint at `out` (room for 10 bytes); return its length."""
    cdef Py_ssize_t n = 0
    while x >= 0x80:
        out[n] = <unsigned char>((x & 0x7F) | 0x80)
        x >>= 7
        n += 1
    out[n] = <unsigned char>x
    return n + 1


cdef bytes _varint(uint64_t x):
    """`x` as a protobuf varint."""
    cdef unsigned char buf[10]
    return buf[:_write_varint(buf, x)]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bytes _packed(values, bint zigzag):
    """The int64 `values` as protobuf varints, one after another, zig-zag encoded first
    when `zigzag` (as `sint` fields are)."""
    cdef const int64_t[::1] v = np.ascontiguousarray(values, dtype=np.int64)
    out = np.empty(10 * v.shape[0], dtype=np.uint8)  # a varint takes at most 10 bytes
    cdef unsigned char[::1] o = out
    cdef Py_ssize_t i, n = 0
    cdef uint64_t x
    for i in range(v.shape[0]):
        if zigzag:
            x = (<uint64_t>v[i] << 1) ^ <uint64_t>(v[i] >> 63)
        else:
            x = <uint64_t>v[i]
        n += _write_varint(&o[n], x)
    return out[:n].tobytes()


cdef bytes _field(int number, payload):
    """A length-delimited protobuf field: its tag and length, then `payload`."""
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


cdef bytes _varint_field(int number, int64_t value):
    """A varint protobuf field holding `value`."""
    return _varint(number << 3) + _varint(<uint64_t>value)


cdef bytes _string_table_bytes(list strings):
    """The bytes of a StringTable holding `strings` in order."""
    cdef bytes tag = _varint((ST_S << 3) | 2)
    return b"".join([tag + _varint(len(s)) + s for s in strings])


# The block header of the blocks the re-pack writer builds: the standard grid.
_STANDARD_GRID = {"granularity": 100, "date_granularity": 1000, "lat_offset": 0,
                  "lon_offset": 0}


cdef bytes _block_bytes(table, list groups, header):
    """A serialized PrimitiveBlock: the string table bytes `table`, the serialized
    groups `groups` and the grid of `header`, in protobuf's field order."""
    return b"".join([_field(PB_STRINGTABLE, table)] + [
        _field(PB_PRIMITIVEGROUP, group) for group in groups
    ] + [
        _varint_field(PB_GRANULARITY, header["granularity"]),
        _varint_field(PB_DATE_GRANULARITY, header["date_granularity"]),
        _varint_field(PB_LAT_OFFSET, header["lat_offset"]),
        _varint_field(PB_LON_OFFSET, header["lon_offset"]),
    ])


cdef _kept_keys_vals(const int64_t[::1] keys_vals, keep):
    """The tag stream (each node's key and value ids followed by a 0) of the nodes that
    `keep` marks, out of `keys_vals`, the stream of all of them."""
    cdef const uint8_t[::1] kept = keep.view(np.uint8)
    out = np.empty(keys_vals.shape[0], dtype=np.int64)
    cdef int64_t[::1] o = out
    cdef Py_ssize_t i = 0, n = 0, node = 0, end = keys_vals.shape[0]
    while i < end:
        if keys_vals[i] == 0:
            if kept[node]:
                o[n] = 0
                n += 1
            node += 1
            i += 1
        else:
            if kept[node]:
                o[n] = keys_vals[i]
                o[n + 1] = keys_vals[i + 1]
                n += 2
            i += 2
    return out[:n]


cdef bytes _dense_bytes(dense, keep):
    """The DenseNodes bytes of the nodes of the decoded group `dense` that `keep` marks."""
    parts = [_field(DENSE_ID, _packed(delta_encode(dense["id"][keep]), True))]
    info = [
        _field(number, _packed(delta_encode(dense[name][keep]), True) if delta
               else _packed(dense[name][keep], False))
        for number, name, delta in _DENSE_INFO
        if len(dense[name]) > 0
    ]
    if info:
        parts.append(_field(DENSE_DENSEINFO, b"".join(info)))
    parts.append(_field(DENSE_LAT, _packed(delta_encode(dense["lat"][keep]), True)))
    parts.append(_field(DENSE_LON, _packed(delta_encode(dense["lon"][keep]), True)))
    if len(dense["keys_vals"]) > 0:
        parts.append(_field(
            DENSE_KEYS_VALS, _packed(_kept_keys_vals(dense["keys_vals"], keep), False)
        ))
    return b"".join(parts)


cdef list _tag_dicts(const int64_t[::1] keys_vals, strings):
    """The tags of each node of the tag stream `keys_vals` (each node's key and value ids
    followed by a 0) as a dict of the `strings` they index, or None for a node without."""
    cdef Py_ssize_t i = 0, end = keys_vals.shape[0]
    tags_list = []
    tags = None
    while i < end:
        if keys_vals[i] == 0:
            tags_list.append(tags)
            tags = None
            i += 1
        else:
            if tags is None:
                tags = {}
            tags[strings[keys_vals[i]]] = strings[keys_vals[i + 1]]
            i += 2
    return tags_list


# The group field of each kind of element, indexed by `_NODES`, `_WAYS`, `_RELATIONS`.
_GROUP_FIELD = (GROUP_NODES, GROUP_WAYS, GROUP_RELATIONS)


cdef _dense_nodes(group):
    """The decoded dense nodes of a group of `element_layout`, or None when it has none."""
    dense = group["dense"]
    return dense if dense is not None and len(dense["id"]) > 0 else None


cdef _group_kind(group):
    """The kind of elements (`_NODES`, `_WAYS` or `_RELATIONS`) a group of
    `element_layout` holds: nodes when it has dense or plain nodes, else ways, else
    relations; None when it holds no element."""
    if _dense_nodes(group) is not None:
        return _NODES
    for kind in (_NODES, _WAYS, _RELATIONS):
        if (group["kind"] == _GROUP_FIELD[kind]).any():
            return kind
    return None


cdef _kept_rows(group, int kind, kept_set):
    """The positions in `group` (see `element_layout`) of its elements of `kind` (a group
    field number) that are in `kept_set` (all when None)."""
    rows = np.flatnonzero(group["kind"] == kind)
    if kept_set is not None:
        rows = rows[_isin(group["id"][rows], kept_set)]
    return rows


cdef bytes _kept_elements(data, group, int kind, kept_set):
    """The fields of the elements of `kind` (a group field number) in `group` (see
    `element_layout`) that are in `kept_set` (all when None), copied from `data` as they
    are."""
    rows = _kept_rows(group, kind, kept_set)
    return b"".join([
        data[start:end]
        for start, end in zip(group["start"][rows].tolist(), group["end"][rows].tolist())
    ])


cdef _crop_block(data, kept_nodes_set, kept_ways_set, kept_rel_set, compact, filepath):
    """The cropped copy of the decompressed block `data` of `filepath`, serialized, or
    None if nothing in it is kept.

    Kept plain nodes, ways and relations are copied as they are stored, and kept dense
    nodes encoded again. With `compact=True` the string table is pruned to only the
    strings the kept elements reference (smaller output); otherwise it is copied.
    """
    header, string_table, groups = _raw_decoded(element_layout, data, filepath)
    kept = (kept_nodes_set, kept_ways_set, kept_rel_set)
    parts = []
    for group in groups:
        kind = _group_kind(group)
        dense = _dense_nodes(group)
        if kind is None:
            continue
        if dense is not None:
            keep = _isin(dense["id"], kept_nodes_set)
            body = _field(GROUP_DENSE, _dense_bytes(dense, keep)) if keep.any() else b""
        else:
            body = _kept_elements(data, group, _GROUP_FIELD[kind], kept[kind])
        if body:
            parts.append(body)
    if not parts:
        return None
    table = b"" if string_table is None else data[string_table[0]:string_table[1]]
    block = _block_bytes(table, parts, header)
    if compact:
        out_block = _parse(PrimitiveBlock(), block, filepath)
        _compact_string_table(out_block)
        block = out_block.SerializeToString()
    return block


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


cdef _frame_blob(blob_type, data):
    """Compress the serialized message `data` into the on-disk blob framing bytes."""
    blob = Blob()
    blob.raw_size = len(data)
    blob.zlib_data = zlib_compress(data)
    blob_bytes = blob.SerializeToString()
    blob_header = BlobHeader()
    blob_header.type = blob_type
    blob_header.datasize = len(blob_bytes)
    header_bytes = blob_header.SerializeToString()
    return pack("!L", len(header_bytes)) + header_bytes + blob_bytes


cdef _write_blob(out, blob_type, message):
    out.write(_frame_blob(blob_type, message.SerializeToString()))


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
                bounds, compact, progress):
    """Sequential write of the crop: the kept elements of each block of `filepath` to
    `output_path`, in the pass `_WRITING` of `progress` (a `_ReadProgress`)."""
    with open(output_path, "wb") as out, progress.one_pass(_WRITING):
        _write_header(out, bounds)
        for data in _iter_block_data(filepath, progress):
            block = _crop_block(
                data, kept_nodes_set, kept_ways_set, kept_rel_set, compact, filepath
            )
            if block is not None:
                out.write(_frame_blob("OSMData", block))


# ---------------------------------------------------------------------------
# Write pass without the GIL (thread pools)
# ---------------------------------------------------------------------------
# `_crop_blocks` builds the cropped copy of each block of a batch as `_crop_block` does,
# byte for byte, without the GIL. A block it does not handle is left to `_crop_block`:
# data `element_layout` refuses, a packed field in more than `_MAX_RUNS` runs, or dense
# nodes whose coordinates, metadata or tags do not match their ids in number.
#
# The functions, from the outside in, and the Python code each one matches:
#   `_append_cropped_block`  one block: its string table copied, its kept groups, its
#                  header values (`_crop_block`, `_block_bytes`);
#   `_crop_group`  one PrimitiveGroup: the kept plain nodes, ways or relations copied as
#                  they are stored, or the kept dense nodes encoded again (`_group_kind`,
#                  `_kept_elements`);
#   `_dense_runs`, `_dense_values`, `_dense_payload`  one DenseNodes group: find its
#                  packed fields, decode them, write the kept nodes' fields
#                  (`_decode_dense_group`, `_dense_bytes`).
# Each level writes its output into its own `_Buffer` of the `_CropState`, which the level
# above wraps in a field of its own.
#
# The second part of this section is the scans the selection stages run on threads:
# `_relation_blocks` (which blocks hold relations) and `_way_copies_blocks` (the copies
# of the candidate ways, `_block_way_copies`).

# What an empty batch buffer points to, so that offsets are never added to NULL.
cdef unsigned char _NO_BYTES[1]


# What `_append_cropped_block` did with a block.
cdef enum:
    _CROPPED = 0                   # its cropped copy is in the output
    _BY_PYTHON = 1                 # left to `_crop_block`
    _NOTHING_KEPT = 2              # nothing in it is kept

cdef inline bint _info_delta(int k) noexcept nogil:
    """Whether DenseInfo field ``k + 1`` (``_DENSE_INFO``'s order) is delta coded."""
    return 1 <= k <= 4


# A growable byte buffer; its owner frees ``data``.
cdef struct _Buffer:
    unsigned char* data
    Py_ssize_t n                   # bytes written
    Py_ssize_t cap                 # bytes allocated


cdef int _grow(_Buffer* b, Py_ssize_t extra) noexcept nogil:
    """Room for ``extra`` more bytes in ``b``; -1 when memory runs out."""
    cdef Py_ssize_t cap = b.cap if b.cap else 4096
    cdef void* p
    if extra > PY_SSIZE_T_MAX - b.n:
        return -1
    if b.n + extra <= b.cap:
        return 0
    while cap < b.n + extra:
        if cap > PY_SSIZE_T_MAX // 2:
            cap = b.n + extra
            break
        cap *= 2
    p = realloc(b.data, cap)
    if p == NULL:
        return -1
    b.data = <unsigned char*>p
    b.cap = cap
    return 0


cdef int _append(_Buffer* b, const unsigned char* src, Py_ssize_t n) noexcept nogil:
    if _grow(b, n) < 0:
        return -1
    if n > 0:
        memcpy(b.data + b.n, src, n)
        b.n += n
    return 0


cdef int _append_varint(_Buffer* b, uint64_t x) noexcept nogil:
    if _grow(b, 10) < 0:
        return -1
    b.n += _write_varint(b.data + b.n, x)
    return 0


cdef int _append_field(_Buffer* b, int number, const unsigned char* payload,
                       Py_ssize_t n) noexcept nogil:
    """A length-delimited field, as `_field` writes it."""
    if _append_varint(b, (number << 3) | 2) < 0 or _append_varint(b, n) < 0:
        return -1
    return _append(b, payload, n)


cdef int _append_packed(_Buffer* b, const int64_t* values, const uint8_t* keep,
                        Py_ssize_t n, bint delta) noexcept nogil:
    """The kept ``values`` as varints, as `_packed` writes them: with ``delta`` each the
    difference from the previous kept one (as `delta_encode`), zig-zag encoded."""
    cdef Py_ssize_t i
    cdef int64_t v, previous = 0
    cdef uint64_t x
    if n > PY_SSIZE_T_MAX // 10 or _grow(b, 10 * n) < 0:
        return -1
    # The difference is taken in uint64, so it wraps around as numpy's int64 does. Zig-zag
    # maps 0, -1, 1, -2, ... to 0, 1, 2, 3, ..., so small negative values stay short.
    for i in range(n):
        if not keep[i]:
            continue
        v = values[i]
        if delta:
            v = <int64_t>(<uint64_t>values[i] - <uint64_t>previous)
            previous = values[i]
            x = (<uint64_t>v << 1) ^ <uint64_t>(v >> 63)
        else:
            x = <uint64_t>v
        b.n += _write_varint(b.data + b.n, x)
    return 0


cdef int _append_kept_tags(_Buffer* b, const int64_t* keys_vals, Py_ssize_t n,
                           const uint8_t* keep) noexcept nogil:
    """The tag stream of the kept nodes, as `_kept_keys_vals` gives it."""
    # The stream is ``k v k v ... 0`` per node; ``node`` counts the 0s passed.
    cdef Py_ssize_t i = 0, node = 0
    if n > PY_SSIZE_T_MAX // 10 or _grow(b, 10 * n) < 0:
        return -1
    while i < n:
        if keys_vals[i] == 0:
            if keep[node]:
                b.n += _write_varint(b.data + b.n, 0)
            node += 1
            i += 1
        else:
            if keep[node]:
                b.n += _write_varint(b.data + b.n, <uint64_t>keys_vals[i])
                b.n += _write_varint(b.data + b.n, <uint64_t>keys_vals[i + 1])
            i += 2
    return 0


cdef bint _tags_end_per_node(const int64_t* keys_vals, Py_ssize_t n_kv,
                             Py_ssize_t n) noexcept nogil:
    """Whether the tag stream of ``n`` dense nodes is empty or ends once per node, as
    `_check_keys_vals` requires."""
    cdef Py_ssize_t i, ends = 0
    cdef bint at_key = True, ended = False
    if n_kv == 0:
        return True
    for i in range(n_kv):
        ended = at_key and keys_vals[i] == 0
        if ended:
            ends += 1
        else:
            at_key = not at_key
    return ends == n and ended


cdef Py_ssize_t _gallop(const int64_t* members, Py_ssize_t lo, Py_ssize_t m,
                        int64_t key) noexcept nogil:
    """The first position at or after ``lo`` of the ascending ``members`` whose value is
    not below ``key``."""
    # A galloping search: steps of 1, 2, 4, ... from ``lo`` until a step passes ``key``,
    # then a binary search inside the last step. In a sorted file a block's node ids
    # ascend, so the next one is usually close to the last and few steps are needed.
    cdef Py_ssize_t step = 1, hi, mid
    if lo >= m or members[lo] >= key:
        return lo
    while lo + step < m and members[lo + step] < key:
        lo += step
        step *= 2
    hi = lo + step if lo + step < m else m
    lo += 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if members[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef struct _Sorted:
    const int64_t* members         # the ascending kept node ids
    Py_ssize_t m
    Py_ssize_t at                  # where the last search ended
    int64_t last                   # the last id searched


cdef inline bint _sorted_has(_Sorted* s, int64_t key) noexcept nogil:
    """Whether ``key`` is a kept node id (`SortedInt64Set.isin`); ascending keys are found
    from where the last search ended."""
    if key < s.last:
        s.at = 0
    s.at = _gallop(s.members, s.at, s.m, key)
    s.last = key
    return s.at < s.m and s.members[s.at] == key


# Where a DenseNodes group's packed fields are, and how many values each holds.
cdef struct _Dense:
    _Runs id, lat, lon, keys_vals
    _Runs info[6]                  # DenseInfo field k + 1 (DI_VERSION ... DI_VISIBLE)
    Py_ssize_t n, n_keys_vals      # nodes, and values in the tag stream
    Py_ssize_t n_info[6]           # n, or 0 when the group has no such field


# The cropper's buffers and the kept ids, reused from block to block of a batch.
cdef struct _CropState:
    _Buffer out                    # the cropped blocks of the batch, one after another
    _Buffer parts                  # the current block's kept groups, as fields
    _Buffer group                  # the current group's kept elements
    _Buffer dense                  # the current dense group's fields, re-encoded
    _Buffer info                   # its DenseInfo fields
    _Buffer packed                 # the packed field being written
    int64_t* values                # the decoded values of the current dense group
    Py_ssize_t values_cap
    uint8_t* keep                  # whether each node of the current dense group is kept
    Py_ssize_t keep_cap
    _Sorted nodes                  # the kept node ids
    const kh_i64set_t* ways        # the kept way ids
    const kh_i64set_t* relations   # the kept relation ids


cdef int _dense_runs(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                     _Dense* d) noexcept nogil:
    """The packed runs of a DenseNodes group and their counts, as `_decode_dense_group`
    reads them (with metadata); 1 for data it refuses or more runs than kept."""
    cdef Py_ssize_t pos = start, f_start, info_start = -1, info_end = -1
    cdef int field, wire, k
    cdef uint64_t length
    d.id.n = d.lat.n = d.lon.n = d.keys_vals.n = 0
    for k in range(6):
        d.info[k].n = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return 1
        if field == DENSE_ID:
            if _add_span(&d.id, f_start, pos) < 0:
                return 1
        elif field == DENSE_LAT:
            if _add_span(&d.lat, f_start, pos) < 0:
                return 1
        elif field == DENSE_LON:
            if _add_span(&d.lon, f_start, pos) < 0:
                return 1
        elif field == DENSE_KEYS_VALS:
            if _add_span(&d.keys_vals, f_start, pos) < 0:
                return 1
        elif field == DENSE_DENSEINFO:
            if info_start >= 0:
                return 1  # several DenseInfo messages are merged by the Python path
            info_start = f_start
            info_end = pos
    if pos != end:
        return 1
    if info_start >= 0:
        pos = info_start
        while pos < info_end:
            _read_tag(buf, &pos, info_end, &field, &wire)
            if wire != WIRE_LENGTH:
                _skip_field(buf, &pos, info_end, wire)
                continue
            length = _read_varint(buf, &pos, info_end)
            f_start = pos
            pos = _field_end(pos, length, info_end)
            if pos > info_end:
                return 1
            if DI_VERSION <= field <= DI_VISIBLE:
                if _add_span(&d.info[field - 1], f_start, pos) < 0:
                    return 1
        if pos != info_end:
            return 1
    d.n = _runs_count(buf, &d.id)
    if d.n < 0 or _runs_count(buf, &d.lat) != d.n or _runs_count(buf, &d.lon) != d.n:
        return 1
    d.n_keys_vals = _runs_count(buf, &d.keys_vals)
    if d.n_keys_vals < 0:
        return 1
    for k in range(6):
        d.n_info[k] = _runs_count(buf, &d.info[k])
        if d.n_info[k] < 0 or (d.n_info[k] != 0 and d.n_info[k] != d.n):
            return 1
    return 0


cdef int _dense_values(const unsigned char* buf, _Dense* d, _CropState* st) noexcept nogil:
    """Decode the group's values into ``st.values``: ids, lats, lons, then each DenseInfo
    field's ``n`` values (whether present or not), then the tag stream. 1 when the tags do
    not end once per node, -1 when memory runs out."""
    # The layout of ``st.values``: ids at v[0:n], lats at v[n:2n], lons at v[2n:3n],
    # DenseInfo field k + 1 at v[(3 + k)n:(4 + k)n] for k = 0 ... 5, the tag stream at
    # v[9n:]. A field the group lacks leaves its slot unused.
    cdef Py_ssize_t n = d.n, need, k
    cdef int64_t* v
    cdef void* p
    # The values, and their bytes, fit in a Py_ssize_t.
    if d.n > PY_SSIZE_T_MAX // 80 or d.n_keys_vals > PY_SSIZE_T_MAX // 80:
        return -1
    need = 9 * d.n + d.n_keys_vals
    if need == 0:
        return 0  # an empty group: no values, and no pointer into ``st.values``
    if need > st.values_cap:
        p = realloc(st.values, need * sizeof(int64_t))
        if p == NULL:
            return -1
        st.values = <int64_t*>p
        st.values_cap = need
    v = st.values
    if (_runs_fill(buf, &d.id, True, True, v) != n
            or _runs_fill(buf, &d.lat, True, True, v + n) != n
            or _runs_fill(buf, &d.lon, True, True, v + 2 * n) != n):
        return 1
    for k in range(6):
        if d.n_info[k] and _runs_fill(buf, &d.info[k], _info_delta(k), _info_delta(k),
                                      v + (3 + k) * n) != n:
            return 1
    if _runs_fill(buf, &d.keys_vals, False, False, v + 9 * n) != d.n_keys_vals:
        return 1
    if not _tags_end_per_node(v + 9 * n, d.n_keys_vals, n):
        return 1
    return 0


cdef int _dense_payload(_Dense* d, _CropState* st) noexcept nogil:
    """The DenseNodes bytes of the kept nodes into ``st.dense``, as `_dense_bytes` writes
    them; 1 when no node is kept, -1 when memory runs out."""
    cdef Py_ssize_t n = d.n, i, k
    cdef const int64_t* v = st.values
    cdef bint kept = False, has_info = False
    cdef void* p
    if n > st.keep_cap:
        p = realloc(st.keep, n)
        if p == NULL:
            return -1
        st.keep = <uint8_t*>p
        st.keep_cap = n
    for i in range(n):
        st.keep[i] = _sorted_has(&st.nodes, v[i])
        kept = kept or st.keep[i]
    if not kept:
        return 1
    # The fields in `_dense_bytes`'s order: ids, DenseInfo, lats, lons, tags. Each packed
    # field is written to ``st.packed`` first, so its length is known for the field header.
    st.dense.n = 0
    st.packed.n = 0
    if (_append_packed(&st.packed, v, st.keep, n, True) < 0
            or _append_field(&st.dense, DENSE_ID, st.packed.data, st.packed.n) < 0):
        return -1
    st.info.n = 0
    for k in range(6):
        if d.n_info[k]:
            has_info = True
            st.packed.n = 0
            if (_append_packed(&st.packed, v + (3 + k) * n, st.keep, n, _info_delta(k)) < 0
                    or _append_field(&st.info, DI_VERSION + k, st.packed.data,
                                     st.packed.n) < 0):
                return -1
    if has_info and _append_field(&st.dense, DENSE_DENSEINFO, st.info.data, st.info.n) < 0:
        return -1
    st.packed.n = 0
    if (_append_packed(&st.packed, v + n, st.keep, n, True) < 0
            or _append_field(&st.dense, DENSE_LAT, st.packed.data, st.packed.n) < 0):
        return -1
    st.packed.n = 0
    if (_append_packed(&st.packed, v + 2 * n, st.keep, n, True) < 0
            or _append_field(&st.dense, DENSE_LON, st.packed.data, st.packed.n) < 0):
        return -1
    if d.n_keys_vals:
        st.packed.n = 0
        if (_append_kept_tags(&st.packed, v + 9 * n, d.n_keys_vals, st.keep) < 0
                or _append_field(&st.dense, DENSE_KEYS_VALS, st.packed.data,
                                 st.packed.n) < 0):
            return -1
    return 0


cdef inline bint _is_element(int field, int wire) noexcept nogil:
    return wire == WIRE_LENGTH and (
        field == GROUP_NODES or field == GROUP_WAYS or field == GROUP_RELATIONS)


cdef int _crop_group(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                     _CropState* st) noexcept nogil:
    """The kept part of one PrimitiveGroup into ``st.group`` (empty when nothing is kept),
    as `_group_layout` and `_crop_block` read and write it; 1 for data they refuse, -1 when
    memory runs out."""
    cdef Py_ssize_t pos = start, f_start, tag_start
    cdef int field, wire, kind = 0, status
    cdef uint64_t length
    cdef int64_t element_id
    cdef bint has_nodes = False, has_ways = False, has_relations = False, dense_seen = False
    cdef bint kept
    cdef _Dense d
    st.group.n = 0
    # Three scans of the group's fields. 1: they are well formed.
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        _skip_field(buf, &pos, end, wire)
    if pos != end:
        return 1
    # 2: every dense group and element is checked, as `_group_layout` decodes them all;
    # the last dense group is the one kept. Also notes which kinds of element it holds.
    pos = start
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return 1
        if field == GROUP_DENSE:
            if _dense_runs(buf, f_start, pos, &d) != 0:
                return 1
            status = _dense_values(buf, &d, st)
            if status != 0:
                return status
            dense_seen = True
        elif _is_element(field, wire):
            if _read_element_id(buf, f_start, pos, field == GROUP_NODES, &element_id) != 0:
                return 1
            has_nodes = has_nodes or field == GROUP_NODES
            has_ways = has_ways or field == GROUP_WAYS
            has_relations = has_relations or field == GROUP_RELATIONS
    if dense_seen and d.n > 0:
        status = _dense_payload(&d, st)
        if status < 0:
            return -1
        if status == 0 and _append_field(&st.group, GROUP_DENSE, st.dense.data,
                                         st.dense.n) < 0:
            return -1
        return 0
    # 3: (no dense nodes) a group keeps one kind of element, as `_group_kind` picks it:
    # plain nodes, else ways, else relations. The kept ones are copied with their field
    # header.
    if has_nodes:
        kind = GROUP_NODES
    elif has_ways:
        kind = GROUP_WAYS
    elif has_relations:
        kind = GROUP_RELATIONS
    else:
        return 0
    pos = start
    while pos < end:
        tag_start = pos
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if field != kind:
            continue
        _read_element_id(buf, f_start, pos, kind == GROUP_NODES, &element_id)
        if kind == GROUP_NODES:
            kept = _sorted_has(&st.nodes, element_id)
        elif kind == GROUP_WAYS:
            kept = _set_has(st.ways, element_id)
        else:
            kept = _set_has(st.relations, element_id)
        if kept and _append(&st.group, buf + tag_start, pos - tag_start) < 0:
            return -1
    return 0


cdef int _append_cropped_block(const unsigned char* buf, Py_ssize_t end,
                               _CropState* st) noexcept nogil:
    """The cropped copy of one block appended to ``st.out``, as `_crop_block` builds it:
    `_CROPPED`, `_NOTHING_KEPT` (nothing appended), `_BY_PYTHON` for data it leaves to
    `_crop_block`, or -1 when memory runs out."""
    cdef Py_ssize_t pos = 0, f_start, table_start = -1, table_end = -1
    cdef int field, wire, status
    cdef uint64_t length, value
    cdef int64_t granularity = 100, date_granularity = 1000, lat_offset = 0, lon_offset = 0
    st.parts.n = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            value = _read_varint(buf, &pos, end)
            if field == PB_GRANULARITY:
                granularity = <int64_t>value
            elif field == PB_DATE_GRANULARITY:
                date_granularity = <int64_t>value
            elif field == PB_LAT_OFFSET:
                lat_offset = <int64_t>value
            elif field == PB_LON_OFFSET:
                lon_offset = <int64_t>value
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                return _BY_PYTHON
            if field == PB_STRINGTABLE:
                table_start = f_start
                table_end = pos
            elif field == PB_PRIMITIVEGROUP:
                status = _crop_group(buf, f_start, pos, st)
                if status != 0:
                    return _BY_PYTHON if status > 0 else -1
                if st.group.n and _append_field(&st.parts, PB_PRIMITIVEGROUP,
                                                st.group.data, st.group.n) < 0:
                    return -1
        else:
            _skip_field(buf, &pos, end, wire)
    if pos != end:
        return _BY_PYTHON
    if table_start >= 0 and _fields_end(buf, table_start, table_end) != table_end:
        return _BY_PYTHON  # `_crop_block` raises for the table
    if st.parts.n == 0:
        return _NOTHING_KEPT
    # The block in `_block_bytes`'s field order: the string table as it is, the kept
    # groups, then the four header values.
    if table_start < 0:
        table_start = table_end = 0
    if (_append_field(&st.out, PB_STRINGTABLE, buf + table_start,
                      table_end - table_start) < 0
            or _append(&st.out, st.parts.data, st.parts.n) < 0
            or _append_varint(&st.out, PB_GRANULARITY << 3) < 0
            or _append_varint(&st.out, <uint64_t>granularity) < 0
            or _append_varint(&st.out, PB_DATE_GRANULARITY << 3) < 0
            or _append_varint(&st.out, <uint64_t>date_granularity) < 0
            or _append_varint(&st.out, PB_LAT_OFFSET << 3) < 0
            or _append_varint(&st.out, <uint64_t>lat_offset) < 0
            or _append_varint(&st.out, PB_LON_OFFSET << 3) < 0
            or _append_varint(&st.out, <uint64_t>lon_offset) < 0):
        return -1
    return _CROPPED


cdef bint _has_relations(const unsigned char* buf, Py_ssize_t end) noexcept nogil:
    """Whether a block has a group holding relations, or data this scan cannot read."""
    cdef Py_ssize_t pos = 0, g_pos, g_end
    cdef int field, wire
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        g_pos = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return True
        if field != PB_PRIMITIVEGROUP:
            continue
        g_end = pos
        while g_pos < g_end:
            _read_tag(buf, &g_pos, g_end, &field, &wire)
            if field == GROUP_RELATIONS and wire == WIRE_LENGTH:
                return True
            _skip_field(buf, &g_pos, g_end, wire)
        if g_pos != g_end:
            return True
    return pos != end


@cython.boundscheck(False)
@cython.wraparound(False)
def _relation_blocks(const unsigned char[::1] buffer, const int64_t[::1] starts):
    """For each block ``buffer[starts[i]:starts[i + 1]]``, whether it holds relations or
    data the scan cannot read (so that `_decode_block` decodes it or raises its error)."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b
    check_batch_starts(starts, buffer.shape[0])
    out = np.zeros(n_blocks, dtype=np.uint8)
    cdef uint8_t[::1] marked = out
    cdef const unsigned char* buf = &buffer[0] if buffer.shape[0] else _NO_BYTES
    with nogil:
        for b in range(n_blocks):
            marked[b] = _has_relations(buf + starts[b], starts[b + 1] - starts[b])
    return out.view(bool)


# A growable int64 array; its owner frees ``data``.
cdef struct _Int64s:
    int64_t* data
    Py_ssize_t n
    Py_ssize_t cap


cdef int _push(_Int64s* a, int64_t value) noexcept nogil:
    """Append ``value``; -1 when memory runs out."""
    cdef Py_ssize_t cap
    cdef void* p
    if a.n == a.cap:
        if a.cap > PY_SSIZE_T_MAX // (2 * <Py_ssize_t>sizeof(int64_t)):
            return -1
        cap = a.cap * 2 if a.cap else 1024
        p = realloc(a.data, cap * sizeof(int64_t))
        if p == NULL:
            return -1
        a.data = <int64_t*>p
        a.cap = cap
    a.data[a.n] = value
    a.n += 1
    return 0


# The copies of candidate ways found so far, in `_block_way_copies`'s six arrays: one entry
# per copy in the first five, and the refs of the copies that touch the box in ``refs``.
cdef struct _Copies:
    _Int64s ids, versions, timestamps, touches, ref_counts, refs
    const kh_i64set_t* candidates  # the ways to look for
    const kh_i64set_t* nodes_in_box


cdef int _info_version_timestamp(const unsigned char* buf, Py_ssize_t start,
                                 Py_ssize_t end, int64_t* version,
                                 int64_t* timestamp) noexcept nogil:
    """Read an Info's version and timestamp as protobuf does (the last of each; a version
    as an int32) into ``version`` / ``timestamp``, which keep their values when absent; 1
    when a field has another wire type than its own."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t value
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if field == INFO_VERSION or field == INFO_TIMESTAMP:
            if wire != WIRE_VARINT:
                return 1
            value = _read_varint(buf, &pos, end)
            if field == INFO_VERSION:
                version[0] = <int32_t><uint32_t>value
            else:
                timestamp[0] = <int64_t>value
        else:
            _skip_field(buf, &pos, end, wire)
    return 0 if pos == end else 1


cdef int _way_copy(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                   _Copies* c) noexcept nogil:
    """Add the way in ``buf[start:end]`` to ``c`` when it is a candidate, as
    `_block_way_copies` reads it; 1 for data it leaves to protobuf, -1 when memory runs
    out."""
    cdef Py_ssize_t pos = start, f_start, n, first
    cdef int field, wire
    cdef uint64_t length
    cdef int64_t way_id = 0, version = -1, timestamp = -1
    cdef bint touching = False
    cdef _Runs refs
    cdef Py_ssize_t i
    refs.n = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if field == WAY_ID:
            if wire != WIRE_VARINT:
                return 1
            way_id = <int64_t>_read_varint(buf, &pos, end)
        elif field == WAY_INFO or field == WAY_REFS:
            if wire != WIRE_LENGTH:
                return 1
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                return 1
            if field == WAY_INFO:
                if _info_version_timestamp(buf, f_start, pos, &version, &timestamp) != 0:
                    return 1
            elif _add_span(&refs, f_start, pos) < 0:
                return 1
        else:
            _skip_field(buf, &pos, end, wire)
    if pos != end:
        return 1
    if not _set_has(c.candidates, way_id):
        return 0
    # The refs are decoded onto the end of ``c.refs`` and kept there only when one of
    # them is in the box.
    n = _runs_count(buf, &refs)
    if n < 0:
        return 1
    first = c.refs.n
    for i in range(n):
        if _push(&c.refs, 0) < 0:
            return -1
    if n and _runs_fill(buf, &refs, True, True, c.refs.data + first) != n:
        return 1
    for i in range(n):
        if _set_has(c.nodes_in_box, c.refs.data[first + i]):
            touching = True
            break
    if not touching:
        c.refs.n = first
    if (_push(&c.ids, way_id) < 0 or _push(&c.versions, version) < 0
            or _push(&c.timestamps, timestamp) < 0 or _push(&c.touches, touching) < 0
            or _push(&c.ref_counts, n if touching else 0) < 0):
        return -1
    return 0


cdef int _block_copies(const unsigned char* buf, Py_ssize_t end, _Copies* c) noexcept nogil:
    """The candidate ways' copies of one block into ``c``; 1 for data left to protobuf, -1
    when memory runs out."""
    cdef Py_ssize_t pos = 0, g_pos, g_end, e_start
    cdef int field, wire, status
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        g_pos = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return 1
        if field != PB_PRIMITIVEGROUP:
            continue
        g_end = pos
        while g_pos < g_end:
            _read_tag(buf, &g_pos, g_end, &field, &wire)
            if field != GROUP_WAYS:
                _skip_field(buf, &g_pos, g_end, wire)
                continue
            if wire != WIRE_LENGTH:
                return 1
            length = _read_varint(buf, &g_pos, g_end)
            e_start = g_pos
            g_pos = _field_end(g_pos, length, g_end)
            if g_pos > g_end:
                return 1
            status = _way_copy(buf, e_start, g_pos, c)
            if status != 0:
                return status
        if g_pos != g_end:
            return 1
    return 0 if pos == end else 1


@cython.boundscheck(False)
@cython.wraparound(False)
def _way_copies_blocks(const unsigned char[::1] buffer, const int64_t[::1] starts,
                       Int64Set candidates, Int64Set nodes_in_box):
    """`_block_way_copies` of each block ``buffer[starts[i]:starts[i + 1]]``, read without
    the GIL: ``(status, copies, refs, (ids, versions, timestamps, touches, ref_counts,
    refs))``, block ``i`` holding ``copies[i]`` of the copies and ``refs[i]`` of the refs;
    status 1 for a block left to `_block_way_copies`."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b, k
    cdef Py_ssize_t marks[6]
    cdef int status
    cdef bint out_of_memory = False
    check_batch_starts(starts, buffer.shape[0])
    result = np.zeros(n_blocks, dtype=np.uint8)
    copies = np.zeros(n_blocks, dtype=np.int64)
    ref_total = np.zeros(n_blocks, dtype=np.int64)
    cdef uint8_t[::1] status_out = result
    cdef int64_t[::1] copies_out = copies, refs_out = ref_total
    cdef const unsigned char* buf = &buffer[0] if buffer.shape[0] else _NO_BYTES
    cdef _Copies c
    memset(&c, 0, sizeof(_Copies))
    c.candidates = candidates.table
    c.nodes_in_box = nodes_in_box.table
    cdef _Int64s* arrays[6]
    arrays[0] = &c.ids
    arrays[1] = &c.versions
    arrays[2] = &c.timestamps
    arrays[3] = &c.touches
    arrays[4] = &c.ref_counts
    arrays[5] = &c.refs
    try:
        # A block left to `_block_way_copies` takes back what it added (``marks``).
        with nogil:
            for b in range(n_blocks):
                for k in range(6):
                    marks[k] = arrays[k].n
                status = _block_copies(buf + starts[b], starts[b + 1] - starts[b], &c)
                if status < 0:
                    out_of_memory = True
                    break
                if status:
                    for k in range(6):
                        arrays[k].n = marks[k]
                status_out[b] = status
                copies_out[b] = c.ids.n - marks[0]
                refs_out[b] = c.refs.n - marks[5]
        if out_of_memory:
            raise MemoryError()
        out = []
        for k in range(6):
            values = np.empty(arrays[k].n, dtype=np.int64)
            if arrays[k].n:
                memcpy(<void*><uintptr_t>values.ctypes.data, arrays[k].data,
                       arrays[k].n * sizeof(int64_t))
            out.append(values)
    finally:
        for k in range(6):
            free(arrays[k].data)
    out[3] = out[3].astype(bool)
    return result, copies, ref_total, out


@cython.boundscheck(False)
@cython.wraparound(False)
def _crop_blocks(const unsigned char[::1] buffer, const int64_t[::1] starts,
                 SortedInt64Set kept_nodes, Int64Set kept_ways, Int64Set kept_rel):
    """The cropped copies of the blocks ``buffer[starts[i]:starts[i + 1]]``, built without
    the GIL as `_crop_block` builds them (``compact=False``). Returns ``(status, offsets,
    data)``: block ``i`` is ``data[offsets[i]:offsets[i + 1]]`` when its status is
    `_CROPPED`; `_NOTHING_KEPT` when nothing in it is kept, `_BY_PYTHON` when it is left to
    `_crop_block`."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b, mark
    cdef int status
    check_batch_starts(starts, buffer.shape[0])
    members = np.ascontiguousarray(kept_nodes._sorted, dtype=np.int64)
    cdef const int64_t[::1] m = members
    result = np.zeros(n_blocks, dtype=np.uint8)
    offsets = np.zeros(n_blocks + 1, dtype=np.int64)
    cdef uint8_t[::1] status_out = result
    cdef int64_t[::1] at = offsets
    cdef const unsigned char* buf = &buffer[0] if buffer.shape[0] else _NO_BYTES
    cdef bint out_of_memory = False
    cdef _CropState st
    memset(&st, 0, sizeof(_CropState))
    st.nodes.members = &m[0] if m.shape[0] else NULL
    st.nodes.m = m.shape[0]
    st.ways = kept_ways.table
    st.relations = kept_rel.table
    try:
        # Each block's copy is appended to ``st.out``; one not cropped here is taken back.
        with nogil:
            for b in range(n_blocks):
                mark = st.out.n
                st.nodes.at = 0
                st.nodes.last = 0
                status = _append_cropped_block(
                    buf + starts[b], starts[b + 1] - starts[b], &st
                )
                if status < 0:
                    out_of_memory = True
                    break
                if status != _CROPPED:
                    st.out.n = mark
                status_out[b] = status
                at[b + 1] = st.out.n
        if out_of_memory:
            raise MemoryError()
        data = (<char*>st.out.data)[:st.out.n] if st.out.n else b""
    finally:
        free(st.out.data)
        free(st.parts.data)
        free(st.group.data)
        free(st.dense.data)
        free(st.info.data)
        free(st.packed.data)
        free(st.values)
        free(st.keep)
    return result, offsets, data


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


cdef _refuse_input_as_output(output_path, sources):
    """Raise ValueError when `output_path` is one of the input paths `sources`, under the
    same name or another (a link to it)."""
    if output_path is not None and Path(output_path).exists():
        clash = [p for p in sources if os.path.samefile(output_path, p)]
        if clash:
            raise ValueError(
                "The output path '%s' is the input '%s'." % (output_path, clash[0])
            )


cpdef crop_pbf(source_path, output_path, bounding_box=None, keep_relations=True,
               workers=None, compact=False, repack=False, polygon=None, progress=None):
    """Crop `source_path` by `bounding_box`, or by the Shapely (Multi)Polygon
    `polygon`, writing a valid PBF to `output_path`.

    Returns the output path. When ``workers > 1`` and the file has enough blocks
    to amortize pool startup (>= ``2 * workers`` OSMData blocks), the parallel
    path is used; otherwise the (faster, for small files) sequential path runs.
    By default (``workers=None``, as ``"auto"``) the count is chosen as an out-of-core read
    does for a file of this size (one process with ``PYROSM_DECODE_BACKEND=processes``).

    When ``compact`` is True each output block's string table is pruned to only the
    strings its kept elements reference (smaller output, slightly slower); when
    False (default) the source block's full string table is copied verbatim.

    When ``repack`` is True the kept elements are re-chunked into canonical, densely
    packed blocks (smallest output, slowest); the re-pack write is sequential, but
    ``workers`` still parallelizes the selection. ``repack=True`` produces minimal
    string tables, so ``compact`` is irrelevant and ignored.

    ``progress``: ``None`` (default) follows :func:`pyrosm.enable_logging`, ``True``
    shows a bar on stderr once the crop has run 2 seconds, from 0 to 100 % over the
    whole crop with its current pass in the label, ``False`` shows nothing, and a
    callable is called as ``progress(done, 100)`` with the percentage of the crop done,
    from 0 to 100 and never back.
    """
    validate_progress(progress)
    workers = default_workers(validate_workers(workers))
    region = _region(bounding_box, polygon)
    bounds = region[0]
    _refuse_input_as_output(output_path, [source_path])

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_crop_")
        os.close(fd)

    # Stage 0: header pre-flight (rejects unsupported inputs before any streaming).
    _read_header(source_path)

    name = Path(source_path).name
    with timed(
        logger, "crop_pbf", logging.INFO, file=name, output=Path(output_path).name
    ) as fields, stepping(
        progress, "Cropping %s" % name, _CROP_SHARES, delay=2.0, leave=False, timed=True
    ) as steps:
        _with_pool(
            _select_and_write,
            (source_path, output_path, region, keep_relations, compact, repack,
             _ReadProgress(steps, [source_path])),
            workers, [source_path], region, compact,
        )
        fields["megabytes"] = os.path.getsize(output_path) / 1e6
    return output_path


def _select_and_write(source_path, output_path, region, keep_relations, compact, repack,
              progress, pool):
    """Select the elements to keep and write `output_path`; in parallel with `pool`.
    Each pass over the file is reported to `progress` (a `_ReadProgress`)."""
    bounds = region[0]
    kept_nodes, kept_ways, kept_rel = _select(
        [source_path], region, keep_relations, pool, progress
    )
    kept_ways, kept_rel = kept_ways[0], kept_rel[0]
    if repack:
        _repack_write(
            source_path, output_path, _to_sorted_set(kept_nodes), _to_set(kept_ways),
            _to_set(kept_rel), bounds, progress
        )
    elif pool is not None:
        pool.broadcast("kept_nodes", kept_nodes)
        pool.broadcast("kept_ways", kept_ways)
        pool.broadcast("kept_rel", kept_rel)
        # imap preserves input order -> the same bytes as the sequential write.
        with open(output_path, "wb") as out, progress.one_pass(_WRITING):
            _write_header(out, bounds)
            payloads = _iter_payloads(source_path, progress)
            for blob_bytes in pool.imap(_worker_write, payloads):
                if blob_bytes is not None:
                    out.write(blob_bytes)
    else:
        _write_pbf(
            source_path, output_path, _to_sorted_set(kept_nodes), _to_set(kept_ways),
            _to_set(kept_rel), bounds, compact, progress
        )


cdef _select(sources, region, keep_relations, pool, progress):
    """Run the crop selection stages over one or more PBF files.

    Returns the kept node ids of all files together, and a list with the kept way
    ids and a list with the kept relation ids of each file. With several files, a
    way is kept only from the file holding its winning copy (see `_way_winners`).
    With a `pool`, each stage spreads a file's blocks over the workers, which get the
    id sets the stage needs from `pool.broadcast`. Each pass over the files is reported
    to `progress` (a `_ReadProgress`).
    """
    if pool is None and len(sources) == 1:
        progress.plan([_SELECTING, _WRITING])
        with progress.one_pass(_SELECTING):
            selection = _select_one_pass(sources[0], region, keep_relations, progress)
        if selection is not None:
            return selection

    progress.plan(
        [_FINDING_NODES, _FINDING_WAYS]
        + ([_CHOOSING_WAYS] if len(sources) > 1 else [])
        + ([_FINDING_RELATIONS] if keep_relations else [])
        + [_WRITING]
    )
    # Stage 1: nodes inside the bbox (or polygon).
    with progress.one_pass(_FINDING_NODES):
        if pool is None:
            nodes_in_bbox = _unique_concat(
                [_nodes_inside_file(p, region, progress) for p in sources]
            )
            nib_set = _to_set(nodes_in_bbox)
        else:
            nodes_in_bbox = _unique_concat([
                ids for p in sources
                for ids in pool.imap(_worker_nodes_inside, _iter_payloads(p, progress))
            ])
            pool.broadcast("nodes_in_bbox", nodes_in_bbox)

    # Stage 2: ways with >=1 node in the bbox (+ all their refs -> complete ways).
    kept_ways = []
    kept_nodes = [nodes_in_bbox]
    with progress.one_pass(_FINDING_WAYS):
        for p in sources:
            if pool is None:
                ways, refs = _ways_touching_file(p, nib_set, progress)
            else:
                results = list(
                    pool.imap(_worker_ways_touching, _iter_payloads(p, progress))
                )
                ways = _unique_concat([w for w, _ in results])
                refs = _unique_concat([r for _, r in results])
            kept_ways.append(ways)
            kept_nodes.append(refs)
    if len(sources) > 1:
        with progress.one_pass(_CHOOSING_WAYS):
            kept_ways, kept_nodes = _way_winners(
                sources, kept_ways, nodes_in_bbox, pool, progress
            )
    else:
        kept_nodes = _unique_concat(kept_nodes)

    # Stage 3: relations referencing a kept node/way.
    if not keep_relations:
        return kept_nodes, kept_ways, [np.empty(0, dtype=np.int64) for _ in sources]
    all_ways = _unique_concat(kept_ways)
    with progress.one_pass(_FINDING_RELATIONS):
        if pool is None:
            kept_nodes_set = _to_set(kept_nodes)
            kept_ways_set = _to_set(all_ways)
            kept_rel = [
                _relations_touching_file(p, kept_nodes_set, kept_ways_set, progress)
                for p in sources
            ]
        else:
            pool.broadcast("kept_nodes", kept_nodes)
            pool.broadcast("kept_ways", all_ways)
            kept_rel = []
            for p in sources:
                payloads = _iter_payloads(p, progress)
                parts = pool.imap(_worker_relations_touching, payloads)
                kept_rel.append(_unique_concat(list(parts)))
    return kept_nodes, kept_ways, kept_rel


# ---------------------------------------------------------------------------
# Parallel path (workers > 1)
# ---------------------------------------------------------------------------
# The selection is staged (node -> way -> relation -> write); each stage is
# internally parallel across blobs but the stages run in sequence because each
# depends on the previous stage's *complete* result. A SINGLE pool is reused
# across all four stages (re-spawning a pool per stage would re-pay the worker
# startup cost four times). The main process reads the blob headers and feeds each
# OSMData blob's position to the pool through `_Pool.imap`; workers read the blob from
# the file and do the heavy decompress + protobuf parse + (re-)encode.
# The growing kept-id arrays a stage needs are broadcast to the persistent workers
# via small `.npy` files in a temp dir (written by the main process between stages,
# memory-mapped + cached per worker on first use) rather than re-pickled per task.
# Output blobs come back in input order (`_Pool.imap` preserves order) so the written
# bytes are identical to the sequential path.
#
# With threads (the default `PYROSM_DECODE_BACKEND`), `_ThreadPool.imap` hands its
# threads batches of `_THREAD_BATCH` blobs, read in one go and decompressed without the
# GIL in parts (`_batch_parts`). The batch version of each stage function (`_BATCHED`)
# decodes them with `decode_batch_nogil`, or crops them with `_crop_blocks`, and leaves
# the blocks those do not handle to the per-block function the worker processes use.
# Each pool keeps what its workers share in its own `_Context`, the kept-id arrays in
# memory with threads.

# The context of a worker process, made by `_init_worker`.
_WORKER_CONTEXT = None
_THREAD_BATCH = 32
_BC_STATUS = BLOCK_COUNTS.index("status")
_BC_NODES = BLOCK_COUNTS.index("nodes")
_BC_ALL_WAYS = BLOCK_COUNTS.index("all_ways")


def _iter_payloads(filepath, progress=None):
    """Yield ``(filepath, identity, offset, size)`` of each OSMData blob; other blobs are
    read and checked here. The bytes passed are reported to `progress` (a
    `_ReadProgress`).

    A worker reads the blob itself, so only these small tuples go through the pool's task
    queue: a large write to it can hang a Python without the fix of CPython gh-94777 when a
    worker dies.
    """
    with open(filepath, "rb") as f:
        identity = _file_identity(f)
        while True:
            if progress is not None:
                progress.advance(f, f.tell())
            blob_header = _read_blob_header(f)
            if blob_header is None:
                break
            if blob_header.type != "OSMData":
                _read_blob(f, blob_header.datasize)
                continue
            offset = f.tell()
            f.seek(blob_header.datasize, 1)
            yield filepath, identity, offset, blob_header.datasize


cdef _payload_data(payload):
    """The decompressed bytes of the blob a payload of `_iter_payloads` points at."""
    filepath, identity, offset, size = payload
    with open(filepath, "rb") as f:
        if _file_identity(f) != identity:
            raise ValueError("'%s' changed while it was being read." % filepath)
        f.seek(offset)
        blob = _read_blob(f, size)
    if blob.HasField("raw"):
        return blob.raw
    return _decompress(blob.zlib_data, _raw_size(blob), filepath)


class _Context:
    """What the workers of one pool share: the crop box, the polygon to crop to (or None),
    the ``compact`` flag and the broadcast kept-id sets. With ``in_memory`` (threads) the
    sets' arrays are held here, else read from ``tmpdir``; each set is built once."""

    def __init__(self, bounds, polygon_wkb, tmpdir, compact, in_memory=False):
        self.bounds = bounds
        self.polygon_wkb = polygon_wkb
        self.shared_polygon = None if polygon_wkb is None else _prepared(polygon_wkb)
        self.tmpdir = tmpdir
        self.compact = compact
        self.arrays = {} if in_memory else None
        self.sets = {}
        self.lock = threading.Lock()
        self.local = threading.local()

    def put(self, name, arr):
        """Hold the kept-id array ``name`` for the threads, replacing its sets."""
        with self.lock:
            self.arrays[name] = arr
            self.sets.pop((name, False), None)
            self.sets.pop((name, True), None)

    def get_set(self, name, sorted_ids=False):
        """The broadcast id set ``name``; with ``sorted_ids``, as `_to_sorted_set` builds
        it."""
        s = self.sets.get((name, sorted_ids))
        if s is None:
            with self.lock:
                s = self.sets.get((name, sorted_ids))
                if s is None:
                    if self.arrays is not None:
                        arr = self.arrays[name]
                    else:
                        arr = np.load(Path(self.tmpdir) / (name + ".npy"))
                    s = _to_sorted_set(arr) if sorted_ids else _to_set(arr)
                    self.sets[(name, sorted_ids)] = s
        return s

    def polygon(self):
        """The prepared polygon to crop to, or None: with threads, one per thread."""
        if self.arrays is None or self.polygon_wkb is None:
            return self.shared_polygon
        polygon = getattr(self.local, "polygon", None)
        if polygon is None:
            polygon = self.local.polygon = _prepared(self.polygon_wkb)
        return polygon


def _init_worker(bounds, polygon_wkb, compact, tmpdir):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _Context(bounds, polygon_wkb, tmpdir, compact)


cdef _nodes_inside_block(ctx, data, filepath):
    header, nodes, _, _ = _decode_block(data, filepath)
    return np.unique(_nodes_inside(header, nodes, ctx.bounds, ctx.polygon()))


cdef _ways_touching_block(ctx, data, filepath):
    _, _, ways, _ = _decode_block(data, filepath)
    ids, refs = _ways_touching(ways, ctx.get_set("nodes_in_bbox"))
    return ids, np.unique(refs)


cdef _relations_touching_block(ctx, data, filepath):
    _, _, _, relations = _decode_block(data, filepath)
    return _relations_touching(
        relations, ctx.get_set("kept_nodes"), ctx.get_set("kept_ways")
    )


cdef _way_copies_block(ctx, data, filepath):
    return _block_way_copies(
        _parse(PrimitiveBlock(), data, filepath),
        ctx.get_set("candidate_ways"),
        ctx.get_set("nodes_in_bbox"),
    )


cdef _write_block(ctx, data, filepath):
    block = _crop_block(
        data,
        ctx.get_set("kept_nodes", True),
        ctx.get_set("kept_ways"),
        ctx.get_set("kept_rel"),
        ctx.compact,
        filepath,
    )
    if block is None:
        return None
    return _frame_blob("OSMData", block)


def _worker_nodes_inside(payload):
    return _nodes_inside_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _worker_ways_touching(payload):
    return _ways_touching_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _worker_way_copies(payload):
    return _way_copies_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _worker_relations_touching(payload):
    return _relations_touching_block(
        _WORKER_CONTEXT, _payload_data(payload), payload[0]
    )


def _worker_write(payload):
    return _write_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _batch_parts(payloads):
    """The decompressed blocks of the payloads of one file, read in one go and
    decompressed without the GIL in parts (see `_inflated_batch`): yields ``(part,
    buffer, starts)``, ``part`` the payloads of the part. A blob that is not zlib data
    with its size, or each blob when the read comes short, is read by `_payload_data`."""
    filepath, identity, first, _ = payloads[0]
    with open(filepath, "rb") as f:
        if _file_identity(f) != identity:
            raise ValueError("'%s' changed while it was being read." % filepath)
        raw = _read_span(f, first, payloads[-1][2] + payloads[-1][3])
    return _inflated_batch(
        raw, payloads, itemgetter(2), itemgetter(3), lambda p: _payload_data(p)
    )


def _run_batched(batched, ctx, payloads):
    """``batched(ctx, part, buffer, starts)``, a batch function of `_BATCHED`, over the
    parts of a batch of payloads, each released before the next one is decompressed; its
    results in order."""
    out = []
    for part, buffer, starts in _batch_parts(payloads):
        out.extend(batched(ctx, part, buffer, starts))
        del buffer, starts
    return out


def _batch_nodes_inside(ctx, payloads, buffer, starts):
    """`_worker_nodes_inside` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``."""
    decoded = decode_batch_nogil(buffer, starts, keys=[], bbox=ctx.bounds)
    counts = decoded["counts"]
    inside = decoded["in_box"]
    polygon = ctx.polygon()
    if polygon is not None and inside.any():
        import shapely

        rows = np.flatnonzero(inside)
        inside[rows] = shapely.intersects_xy(
            polygon, decoded["node_lon"][rows], decoded["node_lat"][rows]
        )
    ids = decoded["node_id"]
    at = _offsets_from_lengths(counts[:, _BC_NODES])
    out = []
    for b, payload in enumerate(payloads):
        if counts[b, _BC_STATUS]:
            out.append(
                _nodes_inside_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0])
            )
        else:
            out.append(np.unique(ids[at[b]:at[b + 1]][inside[at[b]:at[b + 1]]]))
    return out


def _batch_ways_touching(ctx, payloads, buffer, starts):
    """`_worker_ways_touching` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``."""
    decoded = decode_batch_nogil(buffer, starts, keys=[], all_ways=True)
    counts = decoded["counts"]
    ids, refs, refs_off = decoded["all_id"], decoded["all_refs"], decoded["all_refs_off"]
    touching = _any_member(_isin(refs, ctx.get_set("nodes_in_bbox")), refs_off)
    at = _offsets_from_lengths(counts[:, _BC_ALL_WAYS])
    out = []
    for b, payload in enumerate(payloads):
        if counts[b, _BC_STATUS]:
            out.append(
                _ways_touching_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0])
            )
            continue
        w0, w1 = at[b], at[b + 1]
        kept = touching[w0:w1]
        way_refs = refs[refs_off[w0]:refs_off[w1]]
        out.append((
            ids[w0:w1][kept],
            np.unique(way_refs[np.repeat(kept, np.diff(refs_off[w0:w1 + 1]))]),
        ))
    return out


def _batch_relations_touching(ctx, payloads, buffer, starts):
    """`_worker_relations_touching` of each payload of a part of a batch (its blocks
    in ``buffer``), with the pool's context ``ctx``: only the blocks `_relation_blocks`
    marks are decoded."""
    marked = _relation_blocks(buffer, starts)
    empty = np.empty(0, dtype=np.int64)
    return [
        _relations_touching_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0])
        if marked[b] else empty
        for b, payload in enumerate(payloads)
    ]


def _batch_way_copies(ctx, payloads, buffer, starts):
    """`_worker_way_copies` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``."""
    status, copies, refs, arrays = _way_copies_blocks(
        buffer, starts, ctx.get_set("candidate_ways"), ctx.get_set("nodes_in_bbox")
    )
    copy_at, ref_at = _offsets_from_lengths(copies), _offsets_from_lengths(refs)
    out = []
    for b, payload in enumerate(payloads):
        if status[b]:
            out.append(_way_copies_block(
                ctx, bytes(buffer[starts[b]:starts[b + 1]]), payload[0]
            ))
            continue
        c0, c1, r0, r1 = copy_at[b], copy_at[b + 1], ref_at[b], ref_at[b + 1]
        out.append(tuple(a[c0:c1] for a in arrays[:5]) + (arrays[5][r0:r1],))
    return out


def _batch_write(ctx, payloads, buffer, starts):
    """`_worker_write` of each payload of a part of a batch (its blocks in ``buffer``),
    with the pool's context ``ctx``: cropped by `_crop_blocks` (by `_crop_block` with
    ``compact``)."""
    if ctx.compact:
        status = np.full(len(payloads), _BY_PYTHON, dtype=np.uint8)
    else:
        status, offsets, data = _crop_blocks(
            buffer,
            starts,
            ctx.get_set("kept_nodes", True),
            ctx.get_set("kept_ways"),
            ctx.get_set("kept_rel"),
        )
    out = []
    for b, payload in enumerate(payloads):
        if status[b] == _CROPPED:
            out.append(_frame_blob("OSMData", data[offsets[b]:offsets[b + 1]]))
        elif status[b] == _NOTHING_KEPT:
            out.append(None)
        else:
            out.append(
                _write_block(ctx, bytes(buffer[starts[b]:starts[b + 1]]), payload[0])
            )
    return out


# The batch version of each stage function with one.
_BATCHED = {
    _worker_nodes_inside: _batch_nodes_inside,
    _worker_ways_touching: _batch_ways_touching,
    _worker_relations_touching: _batch_relations_touching,
    _worker_way_copies: _batch_way_copies,
    _worker_write: _batch_write,
}


_POOL_FALLBACK_WARNING = (
    "The worker processes of the crop could not run, so it ran in one process instead. This "
    'happens when they cannot re-import the program: a script without an `if __name__ == '
    '"__main__":` guard, or one read from stdin. A worker process may also have died. Guard '
    "the entry point, or pass workers=1 to silence this."
)
_THREAD_FALLBACK_WARNING = (
    "The threads of the crop or merge could not start, so it ran in one thread instead. "
    "Pass workers=1 to silence this."
)


class _Pool:
    """Worker processes that map a function over blob payloads in order (``imap``). The
    kept-id arrays a stage needs are broadcast to them as files in a temp dir."""

    def __init__(self, workers, initargs):
        self.tmpdir = tempfile.mkdtemp(prefix="pyrosm_crop_par_")
        try:
            self.executor = ProcessPoolExecutor(
                int(workers),
                initializer=_init_worker,
                initargs=initargs + (self.tmpdir,),
            )
        except OSError as err:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            raise _PoolStartError(err) from err
        except BaseException:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            raise
        self.window = 2 * int(workers)

    def imap(self, func, items):
        return _ordered_map(self.executor, func, items, self.window)

    def broadcast(self, name, arr):
        """Write the kept-id array ``name`` for the workers to load."""
        np.save(
            Path(self.tmpdir) / (name + ".npy"), np.ascontiguousarray(arr, dtype=np.int64)
        )

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class _ThreadPool:
    """Threads that map a function over blob payloads in order (``imap``), as `_Pool`
    does: in batches through the function's batch version in `_BATCHED`, with this pool's
    own `_Context`."""

    def __init__(self, workers, initargs):
        bounds, polygon_wkb, compact = initargs
        self.context = _Context(bounds, polygon_wkb, None, compact, in_memory=True)
        self.executor = ThreadPoolExecutor(int(workers))
        self.window = 2 * int(workers)

    def imap(self, func, items):
        batches = _read_batches(items, _THREAD_BATCH, itemgetter(2), itemgetter(3))
        return chain.from_iterable(_ordered_map(
            self.executor, partial(_run_batched, _BATCHED[func], self.context), batches,
            self.window,
        ))

    def broadcast(self, name, arr):
        """Hold the kept-id array ``name`` for the threads."""
        self.context.put(name, np.ascontiguousarray(arr, dtype=np.int64))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def _with_pool(func, args, workers, sources, region, compact):
    """Return ``func(*args, pool)``, run with a worker pool when one is used (see
    `_open_pool`). When the pool cannot start or breaks, warn and run ``func`` again in one
    process."""
    try:
        pool = _open_pool(workers, sources, region, compact)
    except _PoolStartError:
        pool = None
        warnings.warn(_POOL_FALLBACK_WARNING, RuntimeWarning)
    if pool is None:
        return func(*args, None)
    try:
        return func(*args, pool)
    except (BrokenProcessPool, _PoolStartError):
        pass
    finally:
        pool.close()
    if isinstance(pool, _ThreadPool):
        warnings.warn(_THREAD_FALLBACK_WARNING, RuntimeWarning)
    else:
        warnings.warn(_POOL_FALLBACK_WARNING, RuntimeWarning)
    return func(*args, None)


def _merge_select(sources, region, keep_relations, progress, pool):
    return _select(sources, region, keep_relations, pool, progress)


cdef _open_pool(workers, sources, region, compact):
    """A worker pool, or None to run sequentially.

    ``workers="auto"`` takes the count an out-of-core read takes for the same amount of
    data (see :func:`pyrosm.engine.pool.auto_worker_count`). Sequential when
    ``workers <= 1`` or the files have fewer than ``2 * workers`` OSMData blocks in total.
    """
    if workers is None or (workers != "auto" and workers <= 1):
        return None
    n_blocks = sum([_count_data_blocks(p) for p in sources])
    if workers == "auto":
        from pyrosm.engine.pool import auto_worker_count

        workers = auto_worker_count(sum([os.path.getsize(p) for p in sources]), n_blocks)
        if workers <= 1:
            return None
    if n_blocks < 2 * int(workers):
        return None
    from pyrosm.engine.pool import _decode_backend

    pool_type = _ThreadPool if _decode_backend() == "threads" else _Pool
    bounds, polygon = region
    wkb = None if polygon is None else polygon.wkb
    return pool_type(workers, (bounds, wkb, compact))


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


# The time of writing a way, and a relation, in times of writing a node (Helsinki).
_WAY_WRITE_COST = 16
_RELATION_WRITE_COST = 50


cpdef write_pbf_from_records(nodes, ways, relations, output_path, bounds, progress=None):
    """Serialize node/way/relation records to a valid PBF at `output_path`.

    `nodes` is a dict of aligned arrays (id/lat/lon/version/timestamp/changeset and
    an object array `tags`); `ways`/`relations` are lists of record dicts. `bounds`
    is (xmin, ymin, xmax, ymax) for the header bbox. `progress(done, total)`, when
    given, gets the work done after each block (a node counting 1, a way
    `_WAY_WRITE_COST` and a relation `_RELATION_WRITE_COST`), and the total once the file
    is closed.
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
    cdef Py_ssize_t way_work = _WAY_WRITE_COST * len(ways)
    cdef Py_ssize_t total = n + way_work + _RELATION_WRITE_COST * len(relations)
    with open(output_path, "wb") as out:
        _write_header(out, bounds)
        while i < n:
            j = min(i + _MAX_GROUP, n)
            _emit_node_block(
                out, ids[i:j], lat_raw[i:j], lon_raw[i:j], versions[i:j],
                timestamps[i:j], changesets[i:j], tags_ordered[i:j],
            )
            i = j
            if progress is not None and i < total:
                progress(i, total)
        i = 0
        while i < len(ways):
            j = min(i + _MAX_GROUP, len(ways))
            _emit_way_block(out, ways[i:j])
            i = j
            if progress is not None and n + _WAY_WRITE_COST * i < total:
                progress(n + _WAY_WRITE_COST * i, total)
        i = 0
        while i < len(relations):
            j = min(i + _MAX_GROUP, len(relations))
            _emit_relation_block(out, relations[i:j])
            i = j
            if progress is not None and n + way_work + _RELATION_WRITE_COST * i < total:
                progress(n + way_work + _RELATION_WRITE_COST * i, total)
    if progress is not None:
        progress(total, total)
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

    empty = np.empty(0, dtype=np.int64)
    fields = meta or {}
    dense = {"id": ids, "lat": lat_raw, "lon": lon_raw, "user_sid": empty,
             "keys_vals": np.asarray(keys_vals, dtype=np.int64) if has_tags else empty}
    for name in ("version", "timestamp", "changeset", "uid", "visible"):
        dense[name] = fields.get(name, empty)
    if "user" in fields:
        users = fields["user"]
        for user in dict.fromkeys(users):
            st.intern(user)
        dense["user_sid"] = np.array([st.index[user] for user in users], dtype=np.int64)
    group = _field(GROUP_DENSE, _dense_bytes(dense, np.ones(len(ids), dtype=bool)))
    block = _block_bytes(_string_table_bytes(st.strings), [group], _STANDARD_GRID)
    if len(ids) > 1 and len(block) > _MAX_PACKED_BLOCK_SIZE:
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
    out.write(_frame_blob("OSMData", block))


# Per-element columns of `_ElementChunk` arrays; `tags_off` and `members_off` are CSR
# offsets into `keys`/`vals` and into `members` (and a relation's `types`/`roles`).
_ELEMENT_COLS = ("id", "info_fields", "version", "timestamp", "changeset", "uid",
                 "user_sid", "visible")
# The bit `info_fields` sets for each Info field (see `decode_elements`).
_HAS_VERSION = 1 << (INFO_VERSION - 1)
_HAS_TIMESTAMP = 1 << (INFO_TIMESTAMP - 1)
_HAS_USER = 1 << (INFO_USER_SID - 1)


cdef _csr_take(off, values, idx):
    """The rows `idx` of the CSR values `values` (arrays sharing the offsets `off`), as
    (offsets, values)."""
    starts = off[idx]
    counts = off[idx + 1] - starts
    new_off = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_off[1:])
    at = np.repeat(starts - new_off[:-1], counts) + np.arange(new_off[-1])
    return new_off, [v[at] for v in values]


cdef _csr_concat(offsets):
    """The CSR offsets `offsets` of consecutive row runs, joined into one."""
    bases = np.cumsum([0] + [off[-1] for off in offsets[:-1]])
    return np.concatenate([off[:-1] + base for off, base in zip(offsets, bases)]
                          + [[offsets[-1][-1] + bases[-1]]])


cdef dict _take_elements(dict arrays, idx):
    """The elements at positions `idx` of element arrays (see `_ElementChunk`)."""
    out = {name: arrays[name][idx] for name in _ELEMENT_COLS}
    out["tags_off"], (out["keys"], out["vals"]) = _csr_take(
        arrays["tags_off"], (arrays["keys"], arrays["vals"]), idx
    )
    member_cols = [name for name in ("members", "types", "roles") if name in arrays]
    out["members_off"], members = _csr_take(
        arrays["members_off"], [arrays[name] for name in member_cols], idx
    )
    out.update(zip(member_cols, members))
    return out


cdef dict _concat_elements(list parts):
    """Element arrays (see `_ElementChunk`) of consecutive elements, joined."""
    out = {name: np.concatenate([a[name] for a in parts]) for name in parts[0]
           if name not in ("tags_off", "members_off")}
    out["tags_off"] = _csr_concat([a["tags_off"] for a in parts])
    out["members_off"] = _csr_concat([a["members_off"] for a in parts])
    return out


cdef _dict_tags(off, keys, vals):
    """Each element's tags as a dict of them holds them: a key given again keeps its
    first place and takes the later value. Returns (offsets, keys, vals)."""
    element = np.repeat(np.arange(len(off) - 1), np.diff(off))
    order = np.lexsort((keys, element))
    pairs = np.stack((element[order], keys[order]))
    if not (pairs[:, 1:] == pairs[:, :-1]).all(axis=0).any():
        return off, keys, vals
    new_off, new_keys, new_vals = [0], [], []
    for start, end in zip(off[:-1].tolist(), off[1:].tolist()):
        tags = dict(zip(keys[start:end].tolist(), vals[start:end].tolist()))
        new_keys.extend(tags)
        new_vals.extend(tags.values())
        new_off.append(len(new_keys))
    return (np.array(new_off, dtype=np.int64), np.array(new_keys, dtype=np.int64),
            np.array(new_vals, dtype=np.int64))


cdef class _ElementChunk:
    """Kept ways or relations of one source block, decoded into arrays (see
    `decode_elements`): the per-element `_ELEMENT_COLS`, tags as `tags_off`/`keys`/
    `vals` and members as `members_off`/`members` (and `types`/`roles`), with string
    ids into `strings`, the strings they use."""
    cdef readonly dict arrays
    cdef readonly object strings
    cdef readonly bint is_relation

    def __cinit__(self, dict arrays, strings, bint is_relation):
        self.arrays = arrays
        self.strings = strings
        self.is_relation = is_relation

    def __len__(self):
        return len(self.arrays["id"])

    @property
    def ids(self):
        return self.arrays["id"]

    def ranks(self):
        """(versions, timestamps), -1 where the Info has none."""
        fields = self.arrays["info_fields"]
        return (
            np.where(fields & _HAS_VERSION, self.arrays["version"], -1),
            np.where(fields & _HAS_TIMESTAMP, self.arrays["timestamp"], -1),
        )

    def take(self, idx):
        """The elements at positions `idx`."""
        return _ElementChunk(_take_elements(self.arrays, idx), self.strings,
                             self.is_relation)


cdef _element_chunk(decoded, strings, bint is_relation):
    """An `_ElementChunk` of the arrays `decode_elements` gives for elements of a block
    with the string table `strings`, keeping only the strings they use."""
    arrays = {name: decoded[name] for name in _ELEMENT_COLS}
    arrays["tags_off"], arrays["keys"], arrays["vals"] = _dict_tags(
        decoded["tags_off"], decoded["keys"], decoded["vals"]
    )
    if is_relation:
        arrays["members_off"] = decoded["members_off"]
        arrays["members"] = decoded["memids"]
        arrays["types"] = decoded["types"]
        arrays["roles"] = decoded["roles"]
    else:
        arrays["members_off"] = decoded["refs_off"]
        arrays["members"] = decoded["refs"]
    has_user = (arrays["info_fields"] & _HAS_USER) != 0
    names = ["keys", "vals", "roles"] if is_relation else ["keys", "vals"]
    used = np.unique(np.concatenate(
        [arrays[name] for name in names] + [arrays["user_sid"][has_user]]
    ))
    for name in names:
        arrays[name] = np.searchsorted(used, arrays[name])
    arrays["user_sid"] = np.where(has_user, np.searchsorted(used, arrays["user_sid"]), 0)
    return _ElementChunk(arrays, [strings[i] for i in used.tolist()], is_relation)


cdef _string_order(dict a, bint is_relation):
    """The string ids of element arrays in the order a re-packed block interns them:
    each element's tags (key, value), its user, then its roles."""
    cdef const int64_t[::1] tags_off = a["tags_off"], keys = a["keys"], vals = a["vals"]
    cdef const int64_t[::1] fields = a["info_fields"], user = a["user_sid"]
    cdef const int64_t[::1] members_off = a["members_off"]
    cdef const int64_t[::1] roles = a["roles"] if is_relation else a["members"][:0]
    cdef int64_t has_user = _HAS_USER
    cdef Py_ssize_t n = fields.shape[0], e, j, t = 0
    order = np.empty(2 * keys.shape[0] + n + roles.shape[0], dtype=np.int64)
    cdef int64_t[::1] o = order
    for e in range(n):
        for j in range(tags_off[e], tags_off[e + 1]):
            o[t] = keys[j]
            o[t + 1] = vals[j]
            t += 2
        if fields[e] & has_user:
            o[t] = user[e]
            t += 1
        if is_relation:
            for j in range(members_off[e], members_off[e + 1]):
                o[t] = roles[j]
                t += 1
    return order[:t]


cdef dict _interned(_ElementChunk chunk, _StringTable st):
    """`chunk`'s arrays with its string ids replaced by ids into the block's string
    table `st`, interning the strings in the order `_string_order` gives."""
    a = chunk.arrays
    order = _string_order(a, chunk.is_relation)
    distinct, first = np.unique(order, return_index=True)
    used = distinct[np.argsort(first, kind="stable")]
    index = np.zeros(max(len(chunk.strings), 1), dtype=np.int64)
    index[used] = [st.intern(chunk.strings[i]) for i in used.tolist()]
    out = dict(a)
    for name in ("keys", "vals", "user_sid", "roles"):
        if name in out:
            out[name] = index[out[name]]
    return out


cdef inline Py_ssize_t _varint_size(uint64_t x) noexcept nogil:
    """The length of `x` as a protobuf varint."""
    cdef Py_ssize_t n = 1
    while x >= 0x80:
        x >>= 7
        n += 1
    return n


cdef inline Py_ssize_t _put_varint(unsigned char* out, Py_ssize_t pos,
                                   uint64_t x) noexcept nogil:
    """Write `x` as a varint at `out + pos`, or only measure it when `out` is NULL;
    return the position after it."""
    if out == NULL:
        return pos + _varint_size(x)
    return pos + _write_varint(out + pos, x)


cdef inline uint64_t _zigzag(int64_t value) noexcept nogil:
    """Protobuf's zig-zag encoding of a signed integer (`sint` fields)."""
    return (<uint64_t>value << 1) ^ <uint64_t>(value >> 63)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class _ElementWriter:
    """Writes the ways or relations of element arrays (string ids into the block's
    table) in protobuf's encoding: the fields in field-number order, repeated fields
    packed and left out when empty, `Info` with the fields present only."""
    cdef const int64_t[::1] ids, tags_off, keys, vals, fields
    cdef const int64_t[::1] version, timestamp, changeset, uid, user, visible
    cdef const int64_t[::1] members_off, members, types, roles
    cdef bint is_relation

    def __cinit__(self, dict a, bint is_relation):
        self.ids = a["id"]
        self.tags_off = a["tags_off"]
        self.keys = a["keys"]
        self.vals = a["vals"]
        self.fields = a["info_fields"]
        self.version = a["version"]
        self.timestamp = a["timestamp"]
        self.changeset = a["changeset"]
        self.uid = a["uid"]
        self.user = a["user_sid"]
        self.visible = a["visible"]
        self.members_off = a["members_off"]
        self.members = a["members"]
        self.is_relation = is_relation
        if is_relation:
            self.types = a["types"]
            self.roles = a["roles"]

    cdef Py_ssize_t packed(self, unsigned char* out, Py_ssize_t pos, int field,
                           const int64_t[::1] values, Py_ssize_t start, Py_ssize_t end,
                           bint delta) noexcept nogil:
        """Write `values[start:end]` as a packed field (delta- and zig-zag-coded when
        `delta`); nothing when empty."""
        cdef Py_ssize_t j, size = 0
        cdef int64_t previous = 0
        if start == end:
            return pos
        for j in range(start, end):
            size += _varint_size(_zigzag(values[j] - previous) if delta
                                 else <uint64_t>values[j])
            previous = values[j]
        pos = _put_varint(out, pos, (field << 3) | 2)
        pos = _put_varint(out, pos, size)
        if out == NULL:
            return pos + size
        previous = 0
        for j in range(start, end):
            pos = _put_varint(out, pos, _zigzag(values[j] - previous) if delta
                              else <uint64_t>values[j])
            previous = values[j]
        return pos

    cdef Py_ssize_t info_fields(self, unsigned char* out, Py_ssize_t pos,
                                Py_ssize_t e) noexcept nogil:
        """Write the fields of element `e`'s Info that it has."""
        cdef int64_t present = self.fields[e]
        if present & (1 << (INFO_VERSION - 1)):
            pos = _put_varint(out, pos, INFO_VERSION << 3)
            pos = _put_varint(out, pos, <uint64_t>self.version[e])
        if present & (1 << (INFO_TIMESTAMP - 1)):
            pos = _put_varint(out, pos, INFO_TIMESTAMP << 3)
            pos = _put_varint(out, pos, <uint64_t>self.timestamp[e])
        if present & (1 << (INFO_CHANGESET - 1)):
            pos = _put_varint(out, pos, INFO_CHANGESET << 3)
            pos = _put_varint(out, pos, <uint64_t>self.changeset[e])
        if present & (1 << (INFO_UID - 1)):
            pos = _put_varint(out, pos, INFO_UID << 3)
            pos = _put_varint(out, pos, <uint64_t>self.uid[e])
        if present & (1 << (INFO_USER_SID - 1)):
            pos = _put_varint(out, pos, INFO_USER_SID << 3)
            pos = _put_varint(out, pos, <uint64_t>self.user[e])
        if present & (1 << (INFO_VISIBLE - 1)):
            pos = _put_varint(out, pos, INFO_VISIBLE << 3)
            pos = _put_varint(out, pos, 1 if self.visible[e] != 0 else 0)
        return pos

    cdef Py_ssize_t element(self, unsigned char* out, Py_ssize_t pos,
                            Py_ssize_t e) noexcept nogil:
        """Write element `e`'s message, or only measure it when `out` is NULL."""
        cdef Py_ssize_t tags_start = self.tags_off[e], tags_end = self.tags_off[e + 1]
        cdef Py_ssize_t start = self.members_off[e], end = self.members_off[e + 1]
        pos = _put_varint(out, pos, WAY_ID << 3)
        pos = _put_varint(out, pos, <uint64_t>self.ids[e])
        pos = self.packed(out, pos, WAY_KEYS, self.keys, tags_start, tags_end, False)
        pos = self.packed(out, pos, WAY_VALS, self.vals, tags_start, tags_end, False)
        if self.fields[e] != 0:
            pos = _put_varint(out, pos, (WAY_INFO << 3) | 2)
            pos = _put_varint(out, pos, self.info_fields(NULL, 0, e))
            pos = self.info_fields(out, pos, e)
        if self.is_relation:
            pos = self.packed(out, pos, REL_ROLES_SID, self.roles, start, end, False)
            pos = self.packed(out, pos, REL_MEMIDS, self.members, start, end, True)
            pos = self.packed(out, pos, REL_TYPES, self.types, start, end, False)
        else:
            pos = self.packed(out, pos, WAY_REFS, self.members, start, end, True)
        return pos

    cdef bytes group(self):
        """The elements as the fields of a primitive group."""
        cdef int field = GROUP_RELATIONS if self.is_relation else GROUP_WAYS
        cdef Py_ssize_t n = self.ids.shape[0], e, size = 0, pos = 0
        lengths = np.empty(n, dtype=np.int64)
        cdef int64_t[::1] length = lengths
        for e in range(n):
            length[e] = self.element(NULL, 0, e)
            size = _put_varint(NULL, size, (field << 3) | 2)
            size = _put_varint(NULL, size, length[e]) + length[e]
        out = np.empty(size, dtype=np.uint8)
        cdef unsigned char[::1] buffer = out
        cdef unsigned char* o = &buffer[0] if size > 0 else NULL
        for e in range(n):
            pos = _put_varint(o, pos, (field << 3) | 2)
            pos = _put_varint(o, pos, length[e])
            pos = self.element(o, pos, e)
        return out.tobytes()


cdef _split_chunks(list chunks, Py_ssize_t n):
    """The first `n` elements of the element chunks `chunks`, and the rest."""
    head, tail = [], []
    cdef Py_ssize_t size
    for chunk in chunks:
        size = len(chunk)
        if n >= size:
            head.append(chunk)
            n -= size
        elif n > 0:
            head.append(chunk.take(np.arange(n)))
            tail.append(chunk.take(np.arange(n, size)))
            n = 0
        else:
            tail.append(chunk)
    return head, tail


cdef _emit_repack_elements(out, list chunks, bint is_relation):
    """Write the ways or relations of the element chunks `chunks` as one re-packed
    block, or as two halves (each split again as needed) when it is larger than
    `_MAX_PACKED_BLOCK_SIZE`."""
    cdef _StringTable st = _StringTable()
    arrays = _concat_elements([_interned(chunk, st) for chunk in chunks])
    group = _ElementWriter(arrays, is_relation).group()
    block = _block_bytes(_string_table_bytes(st.strings), [group], _STANDARD_GRID)
    n = len(arrays["id"])
    if n > 1 and len(block) > _MAX_PACKED_BLOCK_SIZE:
        head, tail = _split_chunks(chunks, n // 2)
        _emit_repack_elements(out, head, is_relation)
        _emit_repack_elements(out, tail, is_relation)
        return
    out.write(_frame_blob("OSMData", block))


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


cdef class _Block:
    """A decompressed block of a PBF, with its string table parsed on first use."""
    cdef readonly object data
    cdef readonly object filepath
    cdef object _table
    cdef object _strings

    def __cinit__(self, data, table, filepath):
        self.data = data
        self.filepath = filepath
        self._table = table
        self._strings = None

    def strings(self):
        """The block's string table (`StringTable.s`): `table` is where its bytes lie in
        `data`, None for a block without one."""
        if self._strings is None:
            table = StringTable()
            if self._table is not None:
                start, end = self._table
                _parse(table, self.data[start:end], self.filepath)
            self._strings = table.s
        return self._strings


cdef _decode_kept_dense_nodes(dense, _Block block, kept_nodes_set):
    """(ids, lat, lon, tags, meta) of the nodes of the decoded dense group `dense` of
    `block` in `kept_nodes_set`.

    Every node is kept when `kept_nodes_set` is None; None when no node is kept.
    """
    ids = dense["id"]
    if kept_nodes_set is None:
        keep = np.ones(len(ids), dtype=bool)
    else:
        keep = _isin(ids, kept_nodes_set)
    if not keep.any():
        return None
    strings = block.strings()
    if len(dense["keys_vals"]) > 0:
        tags_list = _tag_dicts(_kept_keys_vals(dense["keys_vals"], keep), strings)
    else:
        tags_list = [None] * int(keep.sum())

    meta = {}
    for name in ("version", "timestamp", "changeset", "uid"):
        if len(dense[name]) > 0:
            meta[name] = dense[name][keep]
    if len(dense["user_sid"]) > 0:
        sids = dense["user_sid"][keep]
        meta["user"] = np.array([strings[s] for s in sids.tolist()], dtype=object)
    if len(dense["visible"]) > 0:
        meta["visible"] = dense["visible"][keep].astype(bool)
    return ids[keep], dense["lat"][keep], dense["lon"][keep], tags_list, meta or None


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


cdef _decode_kept_plain_nodes(nodes, stringtable):
    """Kept plain nodes -> chunks of the dense chunk shape, one per run of consecutive
    nodes with the same metadata schema (a dense block's DenseInfo is all-or-nothing per
    field)."""
    chunks, run = [], []
    for node in nodes:
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
    cdef long way_count
    cdef list rel_buf
    cdef long rel_count

    def __cinit__(self, out):
        self.out = out
        self.node_chunks = []
        self.node_count = 0
        self.way_buf = []
        self.way_count = 0
        self.rel_buf = []
        self.rel_count = 0

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

    cdef add_ways(self, chunk):
        self._flush_nodes()
        self.way_buf.append(chunk)
        self.way_count += len(chunk)
        while self.way_count >= _MAX_GROUP:
            batch, self.way_buf = _split_chunks(self.way_buf, _MAX_GROUP)
            self.way_count -= _MAX_GROUP
            _emit_repack_elements(self.out, batch, False)

    cdef _flush_ways(self):
        if self.way_buf:
            _emit_repack_elements(self.out, self.way_buf, False)
        self.way_buf = []
        self.way_count = 0

    cdef add_relations(self, chunk):
        self._flush_nodes()
        self._flush_ways()
        self.rel_buf.append(chunk)
        self.rel_count += len(chunk)
        while self.rel_count >= _MAX_GROUP:
            batch, self.rel_buf = _split_chunks(self.rel_buf, _MAX_GROUP)
            self.rel_count -= _MAX_GROUP
            _emit_repack_elements(self.out, batch, True)

    cdef close(self):
        self._flush_nodes()
        self._flush_ways()
        if self.rel_buf:
            _emit_repack_elements(self.out, self.rel_buf, True)
        self.rel_buf = []
        self.rel_count = 0


cdef _check_standard_grid(header, filepath):
    if (header["granularity"] != 100 or header["lat_offset"] != 0
            or header["lon_offset"] != 0 or header["date_granularity"] != 1000):
        raise ValueError(
            "to_pbf(repack=True) and merge_pbf() require the standard PBF grid "
            "(granularity 100, zero lat/lon offsets, date_granularity 1000), which "
            "'%s' does not use; to_pbf(repack=False) can still crop it." % filepath
        )


def _iter_groups(filepath, progress=None):
    """Yield (kind, block, group) for each primitive group of a PBF that holds elements:
    `group` as `element_layout` gives it and `block` the `_Block` it is in."""
    for data in _iter_block_data(filepath, progress):
        header, table, groups = _raw_decoded(element_layout, data, filepath)
        _check_standard_grid(header, filepath)
        block = _Block(data, table, filepath)
        for group in groups:
            kind = _group_kind(group)
            if kind is not None:
                yield kind, block, group


cdef _decode_group(kind, _Block block, group, kept):
    """Decoded chunks of the elements of `group` in the id set `kept` (all when None); an
    empty list when none is kept. Protobuf parses only the kept plain nodes."""
    dense = _dense_nodes(group)
    if dense is not None:
        chunk = _decode_kept_dense_nodes(dense, block, kept)
        return [chunk] if chunk is not None else []
    rows = _kept_rows(group, _GROUP_FIELD[kind], kept)
    if len(rows) == 0:
        return []
    if kind == _NODES:
        elements = _kept_elements(block.data, group, GROUP_NODES, kept)
        parsed = _parse(PrimitiveGroup(), elements, block.filepath)
        return _decode_kept_plain_nodes(parsed.nodes, block.strings())
    decoded = _raw_decoded(
        decode_elements, block.data, block.filepath,
        (group["start"][rows], group["end"][rows], kind == _RELATIONS),
    )
    return [_element_chunk(decoded, block.strings(), kind == _RELATIONS)]


cdef _repack_write(source_path, output_path, kept_nodes_set, kept_ways_set,
                   kept_rel_set, bounds, progress):
    """Sequential re-pack write: re-chunk the kept crop into canonical full blocks."""
    kept = (kept_nodes_set, kept_ways_set, kept_rel_set)
    with open(output_path, "wb") as out, progress.one_pass(_WRITING):
        _write_header(out, bounds)
        writer = _RepackWriter(out)
        for kind, block, group in _iter_groups(source_path, progress):
            for chunk in _decode_group(kind, block, group, kept[kind]):
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
    return chunk.ids


cdef _chunk_ranks(kind, chunk):
    """(versions, timestamps) of a decoded chunk, -1 where the metadata is missing."""
    if kind == _NODES:
        meta = chunk[4] or {}
        missing = np.full(len(chunk[0]), -1, dtype=np.int64)
        return meta.get("version", missing), meta.get("timestamp", missing)
    return chunk.ranks()


cdef _take(kind, chunk, idx):
    """The elements of a decoded chunk at the positions `idx`."""
    if kind == _NODES:
        ids, lat, lon, tags, meta = chunk
        return ids[idx], lat[idx], lon[idx], [tags[i] for i in idx], _slice_meta(meta, idx)
    return chunk.take(idx)


cdef _group_ids(kind, group):
    """Ids of all elements of `kind` in a group of `element_layout`, in file order."""
    dense = _dense_nodes(group)
    if dense is not None:
        return dense["id"]
    return group["id"][group["kind"] == _GROUP_FIELD[kind]]


class _SortedInput:
    """One merge input, read group by group and checked to be sorted by type then id."""

    def __init__(self, filepath, progress=None):
        self.filepath = filepath
        self._groups = _iter_groups(filepath, progress)
        self._next = next(self._groups, None)

    def chunks(self, kind, kept):
        """Yield (chunk, ids, versions, timestamps) for the elements of `kind` in `kept`."""
        last_id = None
        while self._next is not None and self._next[0] <= kind:
            group_kind, block, group = self._next
            if group_kind < kind:
                raise _not_sorted(self.filepath)
            self._next = next(self._groups, None)
            # Every element counts for the order, not only the kept ones.
            group_ids = _group_ids(kind, group)
            if (group_ids[1:] <= group_ids[:-1]).any() or (
                last_id is not None and group_ids[0] <= last_id
            ):
                raise _not_sorted(self.filepath)
            last_id = group_ids[-1]
            for chunk in _decode_group(kind, block, group, kept):
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
        rows = np.flatnonzero(won[i].isin(chunk.ids))
        if len(rows) == 0:
            return
        chunk = chunk.take(rows)
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
                workers=None, polygon=None, progress=None):
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

    workers : int | str
        Number of threads for selecting the elements inside ``bounding_box`` or
        ``polygon``. By default (``None``, as ``"auto"``) the count is chosen as an
        out-of-core read does for the inputs' total size; ``1`` runs sequentially. The
        merged file is written sequentially either way. With
        ``PYROSM_DECODE_BACKEND=processes`` they are worker processes, one by default;
        when those cannot run (on macOS and Windows: a script without an
        ``if __name__ == "__main__":`` guard, or one read from stdin), the merge runs
        in one process and warns.

    polygon : shapely Polygon or MultiPolygon, optional
        The area to crop to, in lon/lat, in place of ``bounding_box``: a node is
        kept when it lies inside the polygon or on its boundary. The output's
        header bounding box is the polygon's envelope.

    progress : bool | callable | None (default: None)
        ``None`` follows :func:`pyrosm.enable_logging`, ``True`` shows a bar on stderr
        once the merge has run 2 seconds, from 0 to 100 % over the whole merge with its
        current pass in the label, ``False`` shows nothing, and a callable is called as
        ``progress(done, 100)`` with the percentage of the merge done, from 0 to 100 and
        never back.

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
    validate_progress(progress)
    workers = default_workers(validate_workers(workers))
    name = Path(sources[0]).name if len(sources) == 1 else "%d files" % len(sources)
    with timed(
        logger, "merge_pbf", logging.INFO, files=",".join([Path(p).name for p in sources])
    ) as fields, stepping(
        progress, "Merging %s" % name, _MERGE_SHARES, delay=2.0, leave=False, timed=True
    ) as steps:
        merged = _merge(
            sources, output_path, bounding_box, keep_relations, workers, polygon,
            _ReadProgress(steps, sources),
        )
        fields["output"] = Path(merged).name
        fields["megabytes"] = os.path.getsize(merged) / 1e6
    return merged


cdef _merge(sources, output_path, bounding_box, keep_relations, workers, polygon,
            progress):
    """`merge_pbf` of the input paths `sources`, reporting each pass over them to
    `progress` (a `_ReadProgress`)."""
    fingerprints = [_fingerprint(p) for p in sources]
    headers = [_read_header(p) for p in sources]

    _refuse_input_as_output(output_path, sources)

    if bounding_box is None and polygon is None:
        bounds = _header_bounds(headers)
        kept = (None, None, None if keep_relations else Int64Set())
        won = (None, None, None)
        progress.plan([_WRITING])
    else:
        region = _region(bounding_box, polygon)
        bounds = region[0]
        kept_nodes, kept_ways, kept_rel = _with_pool(
            _merge_select, (sources, region, keep_relations, progress), workers,
            sources, region, False,
        )
        kept = (
            _to_sorted_set(kept_nodes),
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
        with os.fdopen(fd, "wb") as out, progress.one_pass(_WRITING):
            inputs = [_SortedInput(p, progress) for p in sources]
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
