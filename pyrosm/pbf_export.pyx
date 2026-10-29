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
from collections import deque
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
    _zig_zag_decode,
)

from pyrosm.engine.blobs import (
    _MAX_BLOB_SIZE, _decompress_blob, _inflated_batch, _invalid_pbf, _read_batches,
    _read_span,
)
from pyrosm._arrays cimport _kept_keys_vals
from pyrosm._arrays import _offsets_from_lengths
from pyrosm.engine.pool import _PoolStartError, _ordered_map
from pyrosm.int64_hash cimport Int64Set, SortedInt64Set, _set_has, kh_i64set_t
from pyrosm._log import timed
from pyrosm.utils import (
    _file_identity,
    default_workers,
    validate_bounding_box,
    validate_workers,
)
from pyrosm.utils.download import write_atomic_path
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
cdef enum:
    _NODES = 0
    _WAYS = 1
    _RELATIONS = 2


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------
cdef _bounds(region):
    """``(xmin, ymin, xmax, ymax)`` of a checked box: four numbers, or a geometry's
    envelope (matches how OSM() filters by bbox)."""
    if isinstance(region, list):
        return tuple(region)
    return tuple(float(v) for v in region.bounds)


cdef _region(bounding_box, polygon):
    """The crop region: its box ``(xmin, ymin, xmax, ymax)`` and the prepared polygon
    to crop to, or None to crop to the box. Both take any form
    :func:`~pyrosm.utils.validate_bounding_box` takes; a polygon's box is its
    envelope."""
    if polygon is None:
        if bounding_box is None:
            raise ValueError("Give a bounding box or a polygon to crop to.")
        return _bounds(validate_bounding_box(bounding_box)), None
    if bounding_box is not None:
        raise ValueError("Give either a bounding box or a polygon to crop to, not both.")
    polygon = validate_bounding_box(polygon, "polygon")
    if isinstance(polygon, list):
        from shapely.geometry import box

        polygon = box(*polygon)
    return _bounds(polygon), _prepared(polygon.wkb)


cdef _warn_empty(sources, bounds):
    """Warn that the file written from the files `sources` holds no node: none lies in
    the area of the box `bounds`, or with `bounds` None, the files hold none."""
    names = ", ".join(Path(p).name for p in sources)
    if bounds is None:
        message = "%s hold no node; the written file holds no elements." % names
    else:
        message = "No node of %s lies in the area %s; the written file holds no elements." % (
            names, list(bounds)
        )
    warnings.warn(message)


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
_FINDING_MEMBERS = "finding the members of the relations"
_COMPLETING_RELATIONS = "completing the relations"
_WRITING = "writing the file"
_CROP_SHARES = {
    _SELECTING: 1.1,
    _FINDING_NODES: 1.1,
    _FINDING_WAYS: 1.4,
    _FINDING_RELATIONS: 0.4,
    _FINDING_MEMBERS: 0.4,
    _COMPLETING_RELATIONS: 1.4,
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


def _any_member(hits, offsets):
    """Whether each element has a hit among its members: `hits` holds one value per
    member, laid out by the elements' CSR `offsets`."""
    counts = np.diff(offsets)
    out = np.zeros(len(counts), dtype=bool)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.logical_or.reduceat(hits, offsets[:-1][nonempty])
    return out


cdef _ways_and_nodes(ways, kept):
    """Ids of the decoded `ways` of a block where `kept` is true, and the nodes of those
    ways."""
    return ways["id"][kept], ways["refs"][np.repeat(kept, np.diff(ways["refs_off"]))]


cdef _ways_touching(ways, nodes_set):
    """Ids of the decoded `ways` of a block with a node in `nodes_set`, and the nodes of
    those ways."""
    if ways is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return _ways_and_nodes(ways, _any_member(_isin(ways["refs"], nodes_set), ways["refs_off"]))


cdef _relations_touching(relations, nodes_set, ways_set):
    """Ids of the decoded `relations` of a block with a node member in `nodes_set` or a
    way member in `ways_set`, and the ids of those with a way member in `ways_set` (the
    relations to complete)."""
    empty = np.empty(0, dtype=np.int64)
    if relations is None:
        return empty, empty
    memids, types, offsets = relations["memids"], relations["types"], relations["members_off"]
    hits = np.zeros(len(memids), dtype=bool)
    is_node = types == _MEMBER_NODE
    is_way = types == _MEMBER_WAY
    hits[is_node] = _isin(memids[is_node], nodes_set)
    way_hits = np.zeros(len(memids), dtype=bool)
    way_hits[is_way] = _isin(memids[is_way], ways_set)
    hits |= way_hits
    return (
        relations["id"][_any_member(hits, offsets)],
        np.unique(relations["id"][_any_member(way_hits, offsets)]),
    )


cdef _member_ways(relations, ids_set):
    """The way members of the decoded `relations` of a block whose id is in `ids_set`,
    of every version of them."""
    if relations is None:
        return np.empty(0, dtype=np.int64)
    memids, types, offsets = relations["memids"], relations["types"], relations["members_off"]
    listed = np.repeat(_isin(relations["id"], ids_set), np.diff(offsets))
    return np.unique(memids[listed & (types == _MEMBER_WAY)])


cdef _ways_listed(ways, ids_set):
    """Ids of the decoded `ways` of a block in `ids_set`, and the nodes of those ways."""
    if ways is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return _ways_and_nodes(ways, _isin(ways["id"], ids_set))


cdef _nodes_inside_file(filepath, region, progress):
    bounds, polygon = region
    return _unique_concat([
        _nodes_inside(header, nodes, bounds, polygon)
        for header, nodes, _, _ in _iter_decoded_blocks(filepath, progress)
    ])


cdef _ways_touching_file(filepath, nodes_in_bbox_set, progress):
    """The ids of the ways of `filepath` with a node in the set `nodes_in_bbox_set`, and
    the distinct node ids of those ways."""
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
    """`_relations_touching` over the blocks of `filepath`: the relation ids and the ways
    that complete them."""
    found = [
        _relations_touching(relations, kept_nodes_set, kept_ways_set)
        for _, _, _, relations in _iter_decoded_blocks(filepath, progress)
    ]
    return _concat_pairs(found)


cdef _concat_pairs(pairs):
    """The relation ids and the ids of the relations to complete of `_relations_touching`
    results, each concatenated (the second without duplicates)."""
    empty = np.empty(0, dtype=np.int64)
    return (
        np.concatenate([ids for ids, _ in pairs] or [empty]),
        _unique_concat([ways for _, ways in pairs]),
    )


cdef _member_ways_file(filepath, ids_set, progress):
    """`_member_ways` over the blocks of `filepath` that hold relations."""
    found = []
    for data in _iter_block_data(filepath, progress):
        if _data_kinds(data) & _HOLDS_RELATIONS:
            found.append(_member_ways(_decode_block(data, filepath)[3], ids_set))
    return _unique_concat(found)


cdef _ways_listed_file(filepath, ids_set, progress):
    """`_ways_listed` over the blocks of `filepath`: the way ids and their nodes."""
    found = [
        _ways_listed(ways, ids_set)
        for _, _, ways, _ in _iter_decoded_blocks(filepath, progress)
    ]
    return _unique_concat([ids for ids, _ in found]), _unique_concat(
        [refs for _, refs in found]
    )


cdef _select_one_pass(filepath, region, keep_relations, progress):
    """The crop selection of `filepath` in one pass, as (kept node ids, [kept way ids],
    [kept relation ids]) like `_select`, and the ids of the kept relations to complete,
    for a file whose blocks hold every node before every way and every way before every
    relation, as a file sorted by type then id does. None when a block breaks that
    order."""
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
    rels, reaching = _concat_pairs(rel_ids)
    return (
        _unique_concat(inside + way_refs),
        [np.concatenate(way_ids or [empty])],
        [rels],
    ), reaching


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


cdef _way_winners(sources, touching, nodes_in_bbox, nib_set, pool, progress,
                  payload_kinds):
    """Keep a way only where its winning copy is a copy that touches the box.

    `touching` holds, per file, the ways whose copy in that file has a node in the
    box. Every copy of those ways is ranked as in the merge, so a way whose winning
    copy lies outside the box is dropped, and only the winning copies' nodes are
    kept. Returns the kept way ids of each file and the kept node ids: the nodes in the
    box `nodes_in_bbox` (`nib_set` as a set, or None to build it here) and those of the
    winning copies outside it. The copies are read by `pool` from the blocks that hold
    ways, as `payload_kinds` (the (payload, kinds) pairs of each file's blocks from the
    node pass, see `_select`) gives them.
    """
    candidates = _unique_concat(touching)
    pool.broadcast("candidate_ways", candidates)
    copies = [
        list(pool.imap(
            _worker_way_copies,
            _payloads_holding(payload_kinds[i], _HOLDS_WAYS, progress),
        ))
        for i in range(len(sources))
    ]
    blocks = [block for source_blocks in copies for block in source_blocks]
    counts = [sum([len(block[0]) for block in source]) for source in copies]
    src = np.repeat(np.arange(len(sources)), counts)
    ids, versions, timestamps, touches, ref_counts, refs = [
        np.concatenate([block[k] for block in blocks]) if blocks
        else np.empty(0, dtype=np.int64)
        for k in range(6)
    ]
    del copies, blocks

    winners, unsorted = _winning_copies(
        ids, versions, timestamps, np.cumsum([0] + counts, dtype=np.int64)
    )
    if unsorted >= 0:
        raise _not_sorted(sources[unsorted])
    passing = np.zeros(len(ids), dtype=bool)
    passing[winners] = touches[winners].astype(bool)
    kept_ways = [ids[passing & (src == i)] for i in range(len(sources))]
    kept_refs = refs[np.repeat(passing, ref_counts)]
    del refs
    if nib_set is None:
        nib_set = _to_set(nodes_in_bbox)
    outside = np.unique(kept_refs[~nib_set.isin(kept_refs)])
    return kept_ways, np.concatenate([nodes_in_bbox, outside])


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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _tag_rows(const int64_t[::1] keys_vals, Py_ssize_t n):
    """The tags of the `n` nodes of the tag stream `keys_vals` (each node's key and
    value ids followed by a 0; empty when no node has tags) as (offsets, keys, vals)."""
    cdef Py_ssize_t i = 0, end = keys_vals.shape[0], node = 0, t = 0, pairs = 0
    while i < end:
        if keys_vals[i] == 0:
            i += 1
        else:
            pairs += 1
            i += 2
    off = np.zeros(n + 1, dtype=np.int64)
    keys = np.empty(pairs, dtype=np.int64)
    vals = np.empty(pairs, dtype=np.int64)
    cdef int64_t[::1] o = off, k = keys, v = vals
    i = 0
    while i < end and node < n:
        if keys_vals[i] == 0:
            node += 1
            o[node] = t
            i += 1
        else:
            k[t] = keys_vals[i]
            v[t] = keys_vals[i + 1]
            t += 1
            i += 2
    return off, keys, vals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _tag_stream(const int64_t[::1] off, const int64_t[::1] keys,
                 const int64_t[::1] vals):
    """The tag stream (each node's key and value ids followed by a 0) of nodes with the
    tags `keys` and `vals` at the offsets `off`."""
    cdef Py_ssize_t n = off.shape[0] - 1, e, j, t = 0
    out = np.empty(2 * keys.shape[0] + n, dtype=np.int64)
    cdef int64_t[::1] o = out
    for e in range(n):
        for j in range(off[e], off[e + 1]):
            o[t] = keys[j]
            o[t + 1] = vals[j]
            t += 2
        o[t] = 0
        t += 1
    return out


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
# `_block_kinds` (which blocks hold ways and relations) and `_way_copies_blocks` (the copies
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
                if table_start >= 0:
                    return _BY_PYTHON  # `_crop_block` raises for a second table
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


# The kinds of elements `_group_kinds` finds in the groups of a block.
cdef enum:
    _HOLDS_WAYS = 1
    _HOLDS_RELATIONS = 2


cdef uint8_t _group_kinds(const unsigned char* buf, Py_ssize_t end) noexcept nogil:
    """Which kinds of elements the groups of a block hold: `_HOLDS_WAYS` and
    `_HOLDS_RELATIONS`, both for data this scan cannot read."""
    cdef Py_ssize_t pos = 0, g_pos, g_end
    cdef int field, wire
    cdef uint64_t length
    cdef uint8_t kinds = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        g_pos = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return _HOLDS_WAYS | _HOLDS_RELATIONS
        if field != PB_PRIMITIVEGROUP:
            continue
        g_end = pos
        while g_pos < g_end:
            _read_tag(buf, &g_pos, g_end, &field, &wire)
            if wire == WIRE_LENGTH:
                if field == GROUP_WAYS:
                    kinds |= _HOLDS_WAYS
                elif field == GROUP_RELATIONS:
                    kinds |= _HOLDS_RELATIONS
            _skip_field(buf, &g_pos, g_end, wire)
        if g_pos != g_end:
            return _HOLDS_WAYS | _HOLDS_RELATIONS
    if pos != end:
        return _HOLDS_WAYS | _HOLDS_RELATIONS
    return kinds


cdef uint8_t _data_kinds(data):
    """`_group_kinds` of the decompressed block `data`."""
    cdef const unsigned char[::1] view = data
    return _group_kinds(&view[0], len(data)) if len(data) else 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _block_kinds(const unsigned char[::1] buffer, const int64_t[::1] starts):
    """For each block ``buffer[starts[i]:starts[i + 1]]``, the kinds of elements its groups
    hold (see `_group_kinds`): 1 for ways, 2 for relations, both for data the scan cannot
    read (so that `_decode_block` decodes it or raises its error)."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b
    check_batch_starts(starts, buffer.shape[0])
    out = np.zeros(n_blocks, dtype=np.uint8)
    cdef uint8_t[::1] kinds = out
    cdef const unsigned char* buf = &buffer[0] if buffer.shape[0] else _NO_BYTES
    with nogil:
        for b in range(n_blocks):
            kinds[b] = _group_kinds(buf + starts[b], starts[b + 1] - starts[b])
    return out


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


# The order a merge requires of its inputs (types in the order nodes, ways, relations;
# ids rising within each), read from a block's groups without the GIL: for each group
# that holds elements, its kind (as `_group_kind` takes it), its first and last id of
# that kind, and whether those ids rise.

cdef inline void _note_id(int64_t element_id, Py_ssize_t* count, int64_t* first,
                          int64_t* last, int64_t* rising) noexcept nogil:
    """Count `element_id` as the next id of a group (see `_group_order`)."""
    if count[0] == 0:
        first[0] = element_id
    elif element_id <= last[0]:
        rising[0] = 0
    last[0] = element_id
    count[0] += 1


cdef int _group_order(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                      int64_t* kind, int64_t* first, int64_t* last,
                      int64_t* rising) noexcept nogil:
    """The order of the elements of the group `buf[start:end]`: its `kind` (`_NODES` when
    it holds dense or plain nodes, else `_WAYS`, else `_RELATIONS`; -1 when it holds no
    element), the `first` and `last` id of its elements of that kind (of its dense nodes
    when it has them) and whether those ids `rising`. Returns 1 when the group is not
    made of well-formed fields."""
    cdef Py_ssize_t pos = start, f_start, f_end, d_pos, d_end, count = 0
    cdef int field, wire, d_field, d_wire, element_field
    cdef bint has_dense = False, has_nodes = False, has_ways = False, has_rels = False
    cdef uint64_t length
    cdef int64_t element_id = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            has_dense |= field == GROUP_DENSE
            has_nodes |= field == GROUP_NODES
            has_ways |= field == GROUP_WAYS
            has_rels |= field == GROUP_RELATIONS
        _skip_field(buf, &pos, end, wire)
    if pos != end:
        return 1
    kind[0] = -1
    if has_dense or has_nodes:
        kind[0] = _NODES
    elif has_ways:
        kind[0] = _WAYS
    elif has_rels:
        kind[0] = _RELATIONS
    else:
        return 0
    element_field = (
        GROUP_DENSE if has_dense else GROUP_NODES if has_nodes
        else GROUP_WAYS if has_ways else GROUP_RELATIONS
    )
    rising[0] = 1
    pos = start
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if field != element_field or wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        f_end = _field_end(pos, length, end)
        if f_end > end:
            return 1
        pos = f_end
        if element_field != GROUP_DENSE:
            if _read_element_id(buf, f_start, f_end, element_field == GROUP_NODES,
                                &element_id):
                return 1
            _note_id(element_id, &count, first, last, rising)
            continue
        # The ids of dense nodes are delta coded, in one or more packed runs.
        d_pos = f_start
        while d_pos < f_end:
            _read_tag(buf, &d_pos, f_end, &d_field, &d_wire)
            if d_field == DENSE_ID and d_wire == WIRE_VARINT:
                element_id += _zig_zag_decode(_read_varint(buf, &d_pos, f_end))
                _note_id(element_id, &count, first, last, rising)
            elif d_field == DENSE_ID and d_wire == WIRE_LENGTH:
                length = _read_varint(buf, &d_pos, f_end)
                d_end = _field_end(d_pos, length, f_end)
                if d_end > f_end:
                    return 1
                while d_pos < d_end:
                    element_id += _zig_zag_decode(_read_varint(buf, &d_pos, d_end))
                    _note_id(element_id, &count, first, last, rising)
                if d_pos != d_end:
                    return 1
            else:
                _skip_field(buf, &d_pos, f_end, d_wire)
        if d_pos != f_end:
            return 1
    if pos != end:
        return 1
    if count == 0:
        kind[0] = -1
    return 0


cdef int _block_order(const unsigned char* buf, Py_ssize_t end,
                      _Int64s* out) noexcept nogil:
    """Append to `out` the (kind, first id, last id, rising) of each group of the block
    `buf[:end]` that holds elements, in file order (see `_group_order`). Returns 1 when
    the block is not made of well-formed fields, -1 when memory runs out."""
    cdef Py_ssize_t pos = 0, f_start
    cdef int field, wire
    cdef uint64_t length
    cdef int64_t kind = -1, first = 0, last = 0, rising = 1
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if field != PB_PRIMITIVEGROUP or wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end or _group_order(buf, f_start, pos, &kind, &first, &last, &rising):
            return 1
        if kind >= 0 and (
            _push(out, kind) < 0 or _push(out, first) < 0 or _push(out, last) < 0
            or _push(out, rising) < 0
        ):
            return -1
    return 1 if pos != end else 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _block_orders(const unsigned char[::1] buffer, const int64_t[::1] starts):
    """For each block ``buffer[starts[i]:starts[i + 1]]``, the order of its elements as an
    array of (kind, first id, last id, rising) rows, one per group that holds elements
    (see `_group_order`), or None for a block the scan cannot read (its decode raises
    the error)."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b, mark
    cdef int status
    check_batch_starts(starts, buffer.shape[0])
    cdef const unsigned char* buf = &buffer[0] if buffer.shape[0] else _NO_BYTES
    cdef _Int64s rows
    memset(&rows, 0, sizeof(_Int64s))
    at = np.zeros(n_blocks + 1, dtype=np.int64)
    readable = np.ones(n_blocks, dtype=bool)
    cdef int64_t[::1] at_view = at
    cdef uint8_t[::1] readable_view = readable.view(np.uint8)
    cdef bint out_of_memory = False
    try:
        with nogil:
            for b in range(n_blocks):
                mark = rows.n
                status = _block_order(buf + starts[b], starts[b + 1] - starts[b], &rows)
                if status < 0:
                    out_of_memory = True
                    break
                if status > 0:
                    rows.n = mark
                    readable_view[b] = 0
                at_view[b + 1] = rows.n
        if out_of_memory:
            raise MemoryError()
        table = np.array(<int64_t[:rows.n]>rows.data, dtype=np.int64) if rows.n else (
            np.empty(0, dtype=np.int64)
        )
    finally:
        free(rows.data)
    table = table.reshape(-1, 4)
    return [
        table[at[b] // 4:at[b + 1] // 4] if readable[b] else None
        for b in range(n_blocks)
    ]


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
               workers=None, compact=False, repack=False, polygon=None, progress=None,
               complete_relations=False):
    """Crop `source_path` by `bounding_box`, or by the Shapely (Multi)Polygon
    `polygon`, writing a valid PBF to `output_path`. Both take any form
    :class:`pyrosm.OSM` takes for its bounding box (a GeoDataFrame or GeoSeries is
    united and reprojected to lon/lat); an unusable box is refused as ``OSM`` refuses it,
    and a crop that keeps no node warns.

    With `complete_relations`, a relation kept for a way member in the area also keeps
    its other way members and their nodes, as ``OSM(complete_relations=True)`` reads it,
    so its geometry is complete in the cropped file (one more pass over the file).
    Members that are relations are not completed.

    The crop is written next to `output_path` and replaces it only when it is complete,
    so a failed or interrupted crop leaves the earlier file as it was.

    Returns the output path. When ``workers > 1`` and the file has enough blocks
    to amortize pool startup (>= ``2 * workers`` OSMData blocks), the parallel
    path is used; otherwise the (faster, for small files) sequential path runs.
    By default (``workers=None``, as ``"auto"``) the count is chosen as an out-of-core read
    does for a file of this size (one process with ``PYROSM_DECODE_BACKEND=processes``).

    When ``compact`` is True each output block's string table is pruned to only the
    strings its kept elements reference (smaller output, slightly slower); when
    False (default) the source block's full string table is copied verbatim.

    When ``repack`` is True the kept elements are re-chunked into canonical, densely
    packed blocks (smallest output, slowest). ``workers`` threads select the elements,
    read and decode the source blocks of the write and encode the re-packed blocks, which
    are written in order (the whole write runs in one process with
    ``PYROSM_DECODE_BACKEND=processes``).
    ``repack=True`` produces minimal string tables, so ``compact`` is irrelevant and
    ignored.

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

    # Stage 0: header pre-flight (rejects unsupported inputs before any streaming).
    _read_header(source_path)

    made = output_path is None
    if made:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_crop_")
        os.close(fd)
    name = Path(source_path).name
    try:
        with timed(
            logger, "crop_pbf", logging.INFO, file=name, output=Path(output_path).name
        ) as fields, stepping(
            progress, "Cropping %s" % name, _CROP_SHARES, delay=2.0, leave=False,
            timed=True
        ) as steps:
            write_atomic_path(output_path, partial(
                _write_crop, source_path, region, keep_relations, compact, repack,
                _ReadProgress(steps, [source_path]), complete_relations, workers,
            ))
            fields["megabytes"] = os.path.getsize(output_path) / 1e6
    except BaseException:
        if made:
            Path(output_path).unlink(missing_ok=True)
        raise
    return output_path


def _write_crop(source_path, region, keep_relations, compact, repack, progress,
                complete_relations, workers, path):
    """Write the crop of `crop_pbf` to `path`, with a worker pool when one is used (see
    `_with_pool`)."""
    _with_pool(
        _select_and_write,
        (source_path, path, region, keep_relations, compact, repack, progress,
         complete_relations),
        workers, [source_path], region, compact,
    )


def _select_and_write(source_path, output_path, region, keep_relations, compact, repack,
              progress, complete_relations, pool):
    """Select the elements to keep (completing the kept relations with
    `complete_relations`, see `_select`) and write `output_path`; in parallel with `pool`.
    Each pass over the file is reported to `progress` (a `_ReadProgress`)."""
    bounds = region[0]
    kept_nodes, kept_ways, kept_rel = _select(
        [source_path], region, keep_relations, pool, progress, complete_relations
    )
    if len(kept_nodes) == 0:
        _warn_empty([source_path], bounds)
    kept_ways, kept_rel = kept_ways[0], kept_rel[0]
    if repack:
        _repack_write(
            source_path, output_path, kept_nodes, kept_ways, kept_rel, bounds, progress,
            pool
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


cdef _select(sources, region, keep_relations, pool, progress, complete_relations=False):
    """Run the crop selection stages over one or more PBF files.

    Returns the kept node ids of all files together, and a list with the kept way
    ids and a list with the kept relation ids of each file. With several files, a
    way is kept only from the file holding its winning copy (see `_way_winners`).
    With `complete_relations` (one file only), a relation kept for a kept way member
    also keeps its other way members and their nodes (see `_complete_relations`).
    With a `pool`, each stage spreads a file's blocks over the workers, which get the
    id sets the stage needs from `pool.broadcast`; the stages after the node stage read
    only the blocks that hold ways or relations, as the node stage found them. Several
    files need a pool (an `_InlinePool` runs its stages on the calling thread). Each pass
    over the files is reported to `progress` (a `_ReadProgress`).
    """
    completing = complete_relations and keep_relations
    completion = [_FINDING_MEMBERS, _COMPLETING_RELATIONS] if completing else []
    if pool is None and len(sources) == 1:
        progress.plan([_SELECTING] + completion + [_WRITING])
        with progress.one_pass(_SELECTING):
            found = _select_one_pass(sources[0], region, keep_relations, progress)
        if found is not None:
            selection, reaching = found
            if completing:
                return _complete_relations(sources[0], selection, reaching, pool, progress)
            return selection

    progress.plan(
        [_FINDING_NODES, _FINDING_WAYS]
        + ([_CHOOSING_WAYS] if len(sources) > 1 else [])
        + ([_FINDING_RELATIONS] if keep_relations else [])
        + completion
        + [_WRITING]
    )
    # Stage 1: nodes inside the bbox (or polygon), and with a pool which blocks hold
    # ways and relations, so that the later stages read only those.
    payload_kinds = None
    with progress.one_pass(_FINDING_NODES):
        if pool is None:
            nodes_in_bbox = _unique_concat(
                [_nodes_inside_file(p, region, progress) for p in sources]
            )
            nib_set = _to_set(nodes_in_bbox)
        else:
            found_nodes, payload_kinds = [], []
            for p in sources:
                payloads, kinds = [], []
                read = _iter_and_record(_iter_payloads(p, progress), payloads)
                for ids, block_kinds in pool.imap(_worker_nodes_inside, read):
                    found_nodes.append(ids)
                    kinds.append(block_kinds)
                payload_kinds.append(list(zip(payloads, kinds)))
            nodes_in_bbox = _unique_concat(found_nodes)
            del found_nodes
            pool.broadcast("nodes_in_bbox", nodes_in_bbox)
            nib_set = (
                pool.context.get_set("nodes_in_bbox") if isinstance(pool, _ThreadPool)
                else None
            )

    # Stage 2: ways with >=1 node in the bbox (+ all their refs -> complete ways). With
    # several files, the nodes of the kept ways are those of their winning copies (see
    # `_way_winners`), so the refs found here are not kept.
    one_file = len(sources) == 1
    kept_ways = []
    kept_nodes = [nodes_in_bbox]
    with progress.one_pass(_FINDING_WAYS):
        for i, p in enumerate(sources):
            if pool is None:
                ways, refs = _ways_touching_file(p, nib_set, progress)
            else:
                payloads = _payloads_holding(payload_kinds[i], _HOLDS_WAYS, progress)
                results = list(pool.imap(_worker_ways_touching, payloads))
                ways = _unique_concat([w for w, _ in results])
                refs = _unique_concat([r for _, r in results]) if one_file else None
                del results
            kept_ways.append(ways)
            kept_nodes.append(refs)
    if one_file:
        kept_nodes = _unique_concat(kept_nodes)
    else:
        with progress.one_pass(_CHOOSING_WAYS):
            kept_ways, kept_nodes = _way_winners(
                sources, kept_ways, nodes_in_bbox, nib_set, pool, progress, payload_kinds
            )

    # Stage 3: relations referencing a kept node/way.
    if not keep_relations:
        return kept_nodes, kept_ways, [np.empty(0, dtype=np.int64) for _ in sources]
    all_ways = _unique_concat(kept_ways)
    with progress.one_pass(_FINDING_RELATIONS):
        if pool is None:
            kept_nodes_set = _to_set(kept_nodes)
            kept_ways_set = _to_set(all_ways)
            found = [
                _relations_touching_file(p, kept_nodes_set, kept_ways_set, progress)
                for p in sources
            ]
        else:
            pool.broadcast("relation_nodes", kept_nodes)
            pool.broadcast("relation_ways", all_ways)
            found = []
            for i, p in enumerate(sources):
                payloads = _payloads_holding(payload_kinds[i], _HOLDS_RELATIONS, progress)
                parts = list(pool.imap(_worker_relations_touching, payloads))
                ids, reaching = _concat_pairs(parts)
                found.append((np.unique(ids), reaching))
    selection = kept_nodes, kept_ways, [ids for ids, _ in found]
    if completing:
        return _complete_relations(
            sources[0], selection, found[0][1], pool, progress,
            None if pool is None else payload_kinds[0],
        )
    return selection


cdef _complete_relations(source, selection, reaching, pool, progress, payload_kinds=None):
    """`selection` (of the one file `source`) with the way members of the relations
    `reaching` (those it keeps for a kept way member), of every version of them, and
    their nodes, so those relations are complete in the cropped file. With a `pool`,
    `payload_kinds` are the (payload, kinds) pairs of the file's blocks from the node
    pass (see `_select`)."""
    kept_nodes, kept_ways, kept_rel = selection
    if len(reaching) == 0:
        return selection
    with progress.one_pass(_FINDING_MEMBERS):
        if pool is None:
            ways = _member_ways_file(source, _to_set(reaching), progress)
        else:
            pool.broadcast("reaching_relations", reaching)
            payloads = _payloads_holding(payload_kinds, _HOLDS_RELATIONS, progress)
            ways = _unique_concat(list(pool.imap(_worker_member_ways, payloads)))
    missing = np.setdiff1d(ways, kept_ways[0])
    if len(missing) == 0:
        return selection
    with progress.one_pass(_COMPLETING_RELATIONS):
        if pool is None:
            ids, refs = _ways_listed_file(source, _to_set(missing), progress)
        else:
            pool.broadcast("listed_ways", missing)
            payloads = _payloads_holding(payload_kinds, _HOLDS_WAYS, progress)
            results = list(pool.imap(_worker_ways_listed, payloads))
            ids = _unique_concat([found for found, _ in results])
            refs = _unique_concat([found for _, found in results])
    return (
        _unique_concat([kept_nodes, refs]),
        [np.concatenate([kept_ways[0], ids])],
        kept_rel,
    )


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


def _iter_and_record(items, seen):
    """Yield `items`, appending each to the list `seen`."""
    for item in items:
        seen.append(item)
        yield item


def _payloads_holding(payload_kinds, kind, progress):
    """Yield the payloads of one file whose blocks hold `kind` (see `_group_kinds`), from
    its (payload, kinds) pairs of the node pass, reporting to `progress` (a
    `_ReadProgress`) how far the file is read."""
    for payload, kinds in payload_kinds:
        if kinds & kind:
            if progress is not None:
                progress.advance(payload[0], payload[2] + payload[3])
            yield payload


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
    """The ids of the nodes of the block `data` inside the crop region, and the kinds of
    elements the block holds (see `_group_kinds`)."""
    header, nodes, _, _ = _decode_block(data, filepath)
    inside = np.unique(_nodes_inside(header, nodes, ctx.bounds, ctx.polygon()))
    return inside, _data_kinds(data)


cdef _ways_block(ctx, data, filepath, bint listed):
    """The kept ways of the block `data` and their nodes: with `listed` the ways in the
    set ``listed_ways`` of the pool's context `ctx`, else those with a node in
    ``nodes_in_bbox``."""
    _, _, ways, _ = _decode_block(data, filepath)
    if listed:
        ids, refs = _ways_listed(ways, ctx.get_set("listed_ways"))
    else:
        ids, refs = _ways_touching(ways, ctx.get_set("nodes_in_bbox"))
    return ids, np.unique(refs)


cdef _member_ways_block(ctx, data, filepath):
    """`_member_ways` of the block `data` with the set ``reaching_relations`` of the
    pool's context `ctx`."""
    _, _, _, relations = _decode_block(data, filepath)
    return _member_ways(relations, ctx.get_set("reaching_relations"))


cdef _relations_touching_block(ctx, data, filepath):
    """`_relations_touching` of the block `data` with the sets ``relation_nodes`` and
    ``relation_ways`` of the pool's context `ctx`."""
    _, _, _, relations = _decode_block(data, filepath)
    return _relations_touching(
        relations, ctx.get_set("relation_nodes"), ctx.get_set("relation_ways")
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
    return _ways_block(_WORKER_CONTEXT, _payload_data(payload), payload[0], False)


def _worker_way_copies(payload):
    return _way_copies_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _worker_member_ways(payload):
    return _member_ways_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


def _worker_ways_listed(payload):
    return _ways_block(_WORKER_CONTEXT, _payload_data(payload), payload[0], True)


def _worker_relations_touching(payload):
    return _relations_touching_block(
        _WORKER_CONTEXT, _payload_data(payload), payload[0]
    )


def _worker_write(payload):
    return _write_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


cdef _repack_block(ctx, data, filepath):
    """The kept elements of the block `data` (see `_kept_chunks`), by the kept id sets of
    the pool's context `ctx`."""
    return _kept_chunks(
        data,
        filepath,
        (ctx.get_set("kept_nodes", True), ctx.get_set("kept_ways"), ctx.get_set("kept_rel")),
    )


def _worker_repack(payload):
    """The kept elements of a payload's block for the re-pack write. Only the thread pool
    runs it: its chunks do not pickle."""
    return _repack_block(_WORKER_CONTEXT, _payload_data(payload), payload[0])


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
    kinds = _block_kinds(buffer, starts)
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
            out.append(
                (np.unique(ids[at[b]:at[b + 1]][inside[at[b]:at[b + 1]]]), kinds[b])
            )
    return out


def _batch_ways_touching(ctx, payloads, buffer, starts):
    """`_worker_ways_touching` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``."""
    return _batch_way_selection(ctx, payloads, buffer, starts, False)


def _batch_ways_listed(ctx, payloads, buffer, starts):
    """`_worker_ways_listed` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``."""
    return _batch_way_selection(ctx, payloads, buffer, starts, True)


cdef _batch_way_selection(ctx, payloads, buffer, starts, bint listed):
    """The kept ways of each payload of a part of a batch (its blocks in ``buffer``) and
    their nodes, as `_ways_block` with ``listed`` selects them."""
    decoded = decode_batch_nogil(buffer, starts, keys=[], all_ways=True)
    counts = decoded["counts"]
    ids, refs, refs_off = decoded["all_id"], decoded["all_refs"], decoded["all_refs_off"]
    if listed:
        kept_ways = _isin(ids, ctx.get_set("listed_ways"))
    else:
        kept_ways = _any_member(_isin(refs, ctx.get_set("nodes_in_bbox")), refs_off)
    at = _offsets_from_lengths(counts[:, _BC_ALL_WAYS])
    out = []
    for b, payload in enumerate(payloads):
        if counts[b, _BC_STATUS]:
            out.append(
                _ways_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0], listed)
            )
            continue
        w0, w1 = at[b], at[b + 1]
        kept = kept_ways[w0:w1]
        way_refs = refs[refs_off[w0]:refs_off[w1]]
        out.append((
            ids[w0:w1][kept],
            np.unique(way_refs[np.repeat(kept, np.diff(refs_off[w0:w1 + 1]))]),
        ))
    return out


def _batch_relations_touching(ctx, payloads, buffer, starts):
    """`_worker_relations_touching` of each payload of a part of a batch (its blocks
    in ``buffer``), with the pool's context ``ctx``: only the blocks that `_block_kinds`
    finds relations in are decoded."""
    marked = _block_kinds(buffer, starts) & _HOLDS_RELATIONS
    empty = np.empty(0, dtype=np.int64)
    return [
        _relations_touching_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0])
        if marked[b] else (empty, empty)
        for b, payload in enumerate(payloads)
    ]


def _batch_member_ways(ctx, payloads, buffer, starts):
    """`_worker_member_ways` of each payload of a part of a batch (its blocks in
    ``buffer``), with the pool's context ``ctx``: only the blocks that `_block_kinds`
    finds relations in are decoded."""
    marked = _block_kinds(buffer, starts) & _HOLDS_RELATIONS
    return [
        _member_ways_block(ctx, buffer[starts[b]:starts[b + 1]], payload[0])
        if marked[b] else np.empty(0, dtype=np.int64)
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


def _batch_repack(ctx, payloads, buffer, starts):
    """`_worker_repack` of each payload of a part of a batch (its blocks in ``buffer``),
    with the pool's context ``ctx``: the blocks are cropped by `_crop_blocks` without the
    GIL, and the elements of each cropped copy decoded; a block `_crop_blocks` leaves to
    `_crop_block` is copied out of ``buffer``, which is released before the next part, and
    decoded as `_worker_repack` does."""
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
            cropped = data[offsets[b]:offsets[b + 1]]
            out.append(_kept_chunks(cropped, payload[0], (None, None, None)))
        elif status[b] == _NOTHING_KEPT:
            out.append([])
        else:
            out.append(
                _repack_block(ctx, bytes(buffer[starts[b]:starts[b + 1]]), payload[0])
            )
    return out


cdef _layout_order(data, filepath):
    """The order rows of `_block_orders` for the decompressed block `data` of `filepath`,
    read from its layout as `_SortedInput` reads it; for a block the GIL-free scan cannot
    read. A block whose layout cannot be read raises its error."""
    rows = []
    for kind, block, group in _block_groups(data, filepath):
        ids = _group_ids(kind, group)
        if len(ids):
            rows.append((kind, ids[0], ids[-1], bool((ids[1:] > ids[:-1]).all())))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def _batch_orders(payloads, buffer, starts):
    """The order of the elements of each block of a part of a batch (its blocks in
    ``buffer``): `_block_orders`, or `_layout_order` for a block it cannot read."""
    orders = _block_orders(buffer, starts)
    for b, payload in enumerate(payloads):
        if orders[b] is None:
            orders[b] = _layout_order(bytes(buffer[starts[b]:starts[b + 1]]), payload[0])
    return orders


def _batch_merge_input(ctx, payloads, buffer, starts):
    """The order of the elements of each block of a part of a batch for a merge with a
    region (its blocks in ``buffer``; see `_batch_orders`) and its kept elements (see
    `_batch_repack`). Only the thread pool runs it: its chunks do not pickle."""
    return list(zip(
        _batch_orders(payloads, buffer, starts), _batch_repack(ctx, payloads, buffer, starts)
    ))


def _batch_merge_whole(ctx, payloads, buffer, starts):
    """The order of the elements of each block of a part of a batch for a merge of whole
    inputs (its blocks in ``buffer``; see `_batch_orders`) and all its elements, each
    block copied out of ``buffer``, which is released before the next part. Only the
    thread pool runs it: its chunks do not pickle."""
    orders = _batch_orders(payloads, buffer, starts)
    return [
        (orders[b], _kept_chunks(
            bytes(buffer[starts[b]:starts[b + 1]]), payload[0], (None, None, None)
        ))
        for b, payload in enumerate(payloads)
    ]


# The batch version of each stage function with one.
_BATCHED = {
    _worker_nodes_inside: _batch_nodes_inside,
    _worker_ways_touching: _batch_ways_touching,
    _worker_ways_listed: _batch_ways_listed,
    _worker_member_ways: _batch_member_ways,
    _worker_relations_touching: _batch_relations_touching,
    _worker_way_copies: _batch_way_copies,
    _worker_write: _batch_write,
    _worker_repack: _batch_repack,
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
    does: in batches through the function's batch version in `_BATCHED` (or a batch
    function itself, for the stages only threads run), with this pool's own
    `_Context`."""

    def __init__(self, workers, initargs):
        bounds, polygon_wkb, compact = initargs
        self.context = _Context(bounds, polygon_wkb, None, compact, in_memory=True)
        self.executor = ThreadPoolExecutor(int(workers))
        self.workers = int(workers)
        self.window = 2 * int(workers)

    def imap(self, func, items, batch=_THREAD_BATCH, window=None):
        """``func`` mapped over ``items`` in order, in batches of at most ``batch`` blobs
        with at most ``window`` batches in flight (twice the threads by default)."""
        batches = _read_batches(items, batch, itemgetter(2), itemgetter(3))
        return chain.from_iterable(_ordered_map(
            self.executor, partial(_run_batched, _BATCHED.get(func, func), self.context),
            batches, self.window if window is None else window,
        ))

    def broadcast(self, name, arr):
        """Hold the kept-id array ``name`` for the threads."""
        self.context.put(name, np.ascontiguousarray(arr, dtype=np.int64))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class _InlinePool(_ThreadPool):
    """The stages of a `_ThreadPool` run on the calling thread, batch by batch, with its
    `_Context` for the crop region `region` (bounds, polygon): the selection of a merge
    with one worker, whose passes after the node pass then read only the blocks that hold
    ways or relations, while nothing runs beside the calling thread."""

    def __init__(self, region):
        bounds, polygon = region
        self.context = _Context(
            bounds, None if polygon is None else polygon.wkb, None, False, in_memory=True
        )
        self.executor = None
        self.workers = 1
        self.window = 1

    def imap(self, func, items, batch=_THREAD_BATCH, window=None):
        """``func`` mapped over ``items`` in order, in batches of at most ``batch`` blobs,
        each run when the next result is asked for."""
        batched = _BATCHED.get(func, func)
        for payloads in _read_batches(items, batch, itemgetter(2), itemgetter(3)):
            yield from _run_batched(batched, self.context, payloads)

    def close(self):
        pass


def _with_pool(func, args, workers, sources, region, compact, threads_only=False):
    """Return ``func(*args, pool)``, run with a worker pool when one is used (see
    `_open_pool`). When the pool cannot start or breaks, warn and run ``func`` again in one
    process."""
    try:
        pool = _open_pool(workers, sources, region, compact, threads_only)
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


cdef _open_pool(workers, sources, region, compact, threads_only):
    """A worker pool, or None to run sequentially.

    ``workers="auto"`` takes the count an out-of-core read takes for the same amount of
    data (see :func:`pyrosm.engine.pool.auto_worker_count`). Sequential when
    ``workers <= 1``, the files have fewer than ``2 * workers`` OSMData blocks in total,
    or with ``threads_only`` when the decode backend is processes.
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
    if threads_only and pool_type is _Pool:
        return None
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


cdef _emit_node_block(out, ids, lat_raw, lon_raw, meta, tags_list):
    """Write a block of the nodes `ids`, with the DenseInfo of `meta` (their versions,
    timestamps and changesets) when it is not None."""
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
        versions, timestamps, changesets = meta
        di = dense.denseinfo
        di.version.extend([int(v) for v in versions])
        di.timestamp.extend(delta_encode(timestamps).tolist())
        di.changeset.extend(delta_encode(changesets).tolist())

    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_way_block(out, way_batch):
    """Write a block of the way records `way_batch`, each with the version, timestamp
    and changeset it has (none when 0 or missing)."""
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
        if w.get("version"):
            way.info.version = int(w["version"])
        if w.get("timestamp"):
            way.info.timestamp = int(w["timestamp"])
        if w.get("changeset"):
            way.info.changeset = int(w["changeset"])
        refs = np.ascontiguousarray(w["refs"], dtype=np.int64)
        way.refs.extend(delta_encode(refs).tolist())
    for s in st.strings:
        block.stringtable.s.append(s)
    _write_blob(out, "OSMData", block)


cdef _emit_relation_block(out, rel_batch):
    """Write a block of the relation records `rel_batch`, each with the version,
    timestamp and changeset it has (none when 0 or missing)."""
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
        if r.get("version"):
            rel.info.version = int(r["version"])
        if r.get("timestamp"):
            rel.info.timestamp = int(r["timestamp"])
        if r.get("changeset"):
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
    """Serialize node/way/relation records to a valid PBF at `output_path`, sorted by type
    then id, replacing `output_path` only once the file is complete.

    `nodes` is a dict of aligned arrays (id/lat/lon, an object array `tags` and,
    optionally, version/timestamp/changeset; a node whose three are all 0 is written
    without metadata); `ways`/`relations` are lists of record dicts, whose `version`,
    `timestamp` and `changeset` are written when given.
    `bounds` is (xmin, ymin, xmax, ymax) for the header bbox. `progress(done, total)`, when
    given, gets the work done after each block (a node counting 1, a way
    `_WAY_WRITE_COST` and a relation `_RELATION_WRITE_COST`), and the total once the file
    is closed.
    """
    write_atomic_path(
        output_path, partial(_write_records, nodes, ways, relations, bounds, progress)
    )
    return output_path


def _write_records(nodes, ways, relations, bounds, progress, path):
    """Write the records of `write_pbf_from_records` to `path`."""
    ids = np.ascontiguousarray(nodes["id"], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    lat_raw = _coord_to_raw(nodes["lat"])[order]
    lon_raw = _coord_to_raw(nodes["lon"])[order]
    meta = None
    if "version" in nodes:
        meta = [
            np.ascontiguousarray(nodes[name], dtype=np.int64)[order]
            for name in ("version", "timestamp", "changeset")
        ]
        has_meta = (meta[0] != 0) | (meta[1] != 0) | (meta[2] != 0)
    tags_arr = nodes["tags"]
    tags_ordered = [tags_arr[i] for i in order]
    ways = sorted(ways, key=itemgetter("id"))
    relations = sorted(relations, key=itemgetter("id"))

    cdef int n = len(ids)
    cdef int i = 0
    cdef int j
    cdef Py_ssize_t way_work = _WAY_WRITE_COST * len(ways)
    cdef Py_ssize_t total = n + way_work + _RELATION_WRITE_COST * len(relations)
    with open(path, "wb") as out:
        _write_header(out, bounds, sorted_output=True)
        while i < n:
            j = min(i + _MAX_GROUP, n)
            if meta is not None:
                # A block's DenseInfo is all or nothing: end the block where the nodes
                # with metadata and those without it meet.
                change = np.flatnonzero(has_meta[i:j] != has_meta[i])
                if len(change):
                    j = i + int(change[0])
            _emit_node_block(
                out, ids[i:j], lat_raw[i:j], lon_raw[i:j],
                [values[i:j] for values in meta] if meta is not None and has_meta[i]
                else None,
                tags_ordered[i:j],
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

cdef _emit_repack_nodes(out, list chunks):
    """Write the nodes of the node chunks `chunks`, which share one metadata schema, as
    one re-packed block, or as two halves (each split again as needed) when it is larger
    than `_MAX_PACKED_BLOCK_SIZE`. The block's string table holds every node's tags
    (key, value) in node order, then the users, each string at its first use."""
    cdef _StringTable st = _StringTable()
    indexes = [np.zeros(max(len(chunk.strings), 1), dtype=np.int64) for chunk in chunks]
    for chunk, index in zip(chunks, indexes):
        tags = np.empty(2 * len(chunk.arrays["keys"]), dtype=np.int64)
        tags[0::2] = chunk.arrays["keys"]
        tags[1::2] = chunk.arrays["vals"]
        _intern_first_uses(st, chunk.strings, tags, index)
    if "user_sid" in chunks[0].meta:
        for chunk, index in zip(chunks, indexes):
            _intern_first_uses(st, chunk.strings, chunk.arrays["user_sid"], index)
    parts = []
    for chunk, index in zip(chunks, indexes):
        part = dict(chunk.arrays)
        for name in ("keys", "vals", "user_sid"):
            if name in part:
                part[name] = index[part[name]]
        parts.append(part)
    a = _concat_elements(parts)
    empty = np.empty(0, dtype=np.int64)
    dense = {name: a.get(name, empty) for name in _NODE_COLS + _NODE_META}
    dense["keys_vals"] = (
        _tag_stream(a["tags_off"], a["keys"], a["vals"]) if len(a["keys"]) else empty
    )
    n = len(a["id"])
    group = _field(GROUP_DENSE, _dense_bytes(dense, np.ones(n, dtype=bool)))
    block = _block_bytes(_string_table_bytes(st.strings), [group], _STANDARD_GRID)
    if n > 1 and len(block) > _MAX_PACKED_BLOCK_SIZE:
        head, tail = _split_chunks(chunks, n // 2)
        _emit_repack_nodes(out, head)
        _emit_repack_nodes(out, tail)
        return
    out.write(_frame_blob("OSMData", block))


# Per-element columns of `_ElementChunk` arrays of ways and relations; `tags_off` and
# `members_off` are CSR offsets into `keys`/`vals` and into `members` (and a relation's
# `types`/`roles`).
_ELEMENT_COLS = ("id", "info_fields", "version", "timestamp", "changeset", "uid",
                 "user_sid", "visible")
# Per-node columns of `_ElementChunk` arrays of nodes, and the DenseInfo columns a node
# chunk can have (all of its nodes have the same ones), in `_DENSE_INFO` order.
_NODE_COLS = ("id", "lat", "lon")
_NODE_META = ("version", "timestamp", "changeset", "uid", "user_sid", "visible")
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


cdef dict _take_rows(dict arrays, idx, row_cols):
    """The elements at positions `idx` of chunk arrays (see `_ElementChunk`): their
    per-element columns `row_cols`, tags and members."""
    out = {name: arrays[name][idx] for name in row_cols}
    out["tags_off"], (out["keys"], out["vals"]) = _csr_take(
        arrays["tags_off"], (arrays["keys"], arrays["vals"]), idx
    )
    if "members_off" in arrays:
        member_cols = [name for name in ("members", "types", "roles") if name in arrays]
        out["members_off"], members = _csr_take(
            arrays["members_off"], [arrays[name] for name in member_cols], idx
        )
        out.update(zip(member_cols, members))
    return out


cdef dict _concat_elements(list parts):
    """Chunk arrays (see `_ElementChunk`) of consecutive elements, joined; the columns
    ending in ``_off`` are CSR offsets."""
    out = {}
    for name in parts[0]:
        if name.endswith("_off"):
            out[name] = _csr_concat([a[name] for a in parts])
        else:
            out[name] = np.concatenate([a[name] for a in parts])
    return out


# The most tags of an element that `_repeats_a_key` compares pairwise.
_PAIRWISE_TAGS = 32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _repeats_a_key(const int64_t[::1] off, const int64_t[::1] keys):
    """Whether the tags of an element (the keys `keys` at the offsets `off`) give a key
    twice: compared pairwise, or sorted for an element with more than `_PAIRWISE_TAGS`
    tags."""
    cdef Py_ssize_t e, i, j
    for e in range(off.shape[0] - 1):
        if off[e + 1] - off[e] > _PAIRWISE_TAGS:
            if len(np.unique(keys[off[e]:off[e + 1]])) < off[e + 1] - off[e]:
                return True
            continue
        for i in range(off[e] + 1, off[e + 1]):
            for j in range(off[e], i):
                if keys[i] == keys[j]:
                    return True
    return False


cdef _dict_tags(off, keys, vals):
    """Each element's tags as a dict of them holds them: a key given again keeps its
    first place and takes the later value. Returns (offsets, keys, vals)."""
    if not _repeats_a_key(off, keys):
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
    """Kept elements of one kind (`_NODES`, `_WAYS` or `_RELATIONS`) of one source
    block, decoded into arrays: the per-element columns (for nodes `_NODE_COLS` and the
    DenseInfo columns `meta` they have, for ways and relations `_ELEMENT_COLS`, see
    `decode_elements`), tags as `tags_off`/`keys`/`vals` and the members of ways and
    relations as `members_off`/`members` (and `types`/`roles`), with string ids into
    `strings`, the strings they use."""
    cdef readonly dict arrays
    cdef readonly object strings
    cdef readonly int kind
    cdef readonly tuple meta

    def __cinit__(self, dict arrays, strings, int kind, tuple meta=None):
        self.arrays = arrays
        self.strings = strings
        self.kind = kind
        self.meta = meta

    def __len__(self):
        return len(self.arrays["id"])

    @property
    def ids(self):
        return self.arrays["id"]

    def ranks(self):
        """(versions, timestamps), -1 where the metadata has none."""
        if self.kind == _NODES:
            missing = np.full(len(self), -1, dtype=np.int64)
            return (
                self.arrays["version"] if "version" in self.meta else missing,
                self.arrays["timestamp"] if "timestamp" in self.meta else missing,
            )
        fields = self.arrays["info_fields"]
        return (
            np.where(fields & _HAS_VERSION, self.arrays["version"], -1),
            np.where(fields & _HAS_TIMESTAMP, self.arrays["timestamp"], -1),
        )

    def take(self, idx):
        """The elements at positions `idx`."""
        cols = _NODE_COLS + self.meta if self.kind == _NODES else _ELEMENT_COLS
        return _ElementChunk(
            _take_rows(self.arrays, idx, cols), self.strings, self.kind, self.meta
        )


cdef _element_chunk(decoded, strings, int kind):
    """An `_ElementChunk` of the arrays `decode_elements` gives for ways or relations
    (`kind`) of a block with the string table `strings`, keeping only the strings they
    use."""
    arrays = {name: decoded[name] for name in _ELEMENT_COLS}
    arrays["tags_off"], arrays["keys"], arrays["vals"] = _dict_tags(
        decoded["tags_off"], decoded["keys"], decoded["vals"]
    )
    if kind == _RELATIONS:
        arrays["members_off"] = decoded["members_off"]
        arrays["members"] = decoded["memids"]
        arrays["types"] = decoded["types"]
        arrays["roles"] = decoded["roles"]
    else:
        arrays["members_off"] = decoded["refs_off"]
        arrays["members"] = decoded["refs"]
    return _with_used_strings(arrays, strings, kind)


cdef _ElementChunk _node_chunk(dict arrays, strings, tuple meta, off, keys, vals):
    """A node chunk of the node arrays `arrays` (with the DenseInfo columns `meta`) and
    the tags `keys` and `vals` at the offsets `off`, string ids into the string table
    `strings` of their block, keeping only the strings they use. Each node's tags hold
    a key once, as a dict of their texts does: a key given again keeps its first place
    and takes the later value, also when the table holds its text twice."""
    distinct = np.unique(keys)
    texts = [strings[i] for i in distinct.tolist()]
    first = {}
    same = [first.setdefault(text, distinct[i]) for i, text in enumerate(texts)]
    if len(first) < len(same):
        keys = np.array(same, dtype=np.int64)[np.searchsorted(distinct, keys)]
    arrays["tags_off"], arrays["keys"], arrays["vals"] = _dict_tags(off, keys, vals)
    return _with_used_strings(arrays, strings, _NODES, meta)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _mark_used(uint8_t[::1] mark, const int64_t[::1] ids):
    """Set `mark[i]` for each string id `i` of `ids`; an id outside `mark` raises
    IndexError."""
    cdef Py_ssize_t j, n = mark.shape[0]
    cdef int64_t i
    for j in range(ids.shape[0]):
        i = ids[j]
        if i < 0 or i >= n:
            raise IndexError(
                "string id %d is outside the block's string table of %d" % (i, n)
            )
        mark[i] = 1


cdef _ElementChunk _with_used_strings(dict arrays, strings, int kind, tuple meta=None):
    """An `_ElementChunk` of `kind` with the arrays `arrays` (and for nodes the
    DenseInfo columns `meta`), whose string ids index `strings`, keeping only the
    strings they use, in their order (`arrays` gets the new string ids)."""
    names = ["keys", "vals", "roles"] if kind == _RELATIONS else ["keys", "vals"]
    if kind == _NODES and "user_sid" in meta:
        names.append("user_sid")
    mark = np.zeros(len(strings), dtype=np.uint8)
    for name in names:
        _mark_used(mark, arrays[name])
    if kind != _NODES:
        has_user = (arrays["info_fields"] & _HAS_USER) != 0
        users = arrays["user_sid"][has_user]
        _mark_used(mark, users)
    position = np.cumsum(mark, dtype=np.int32) - 1
    for name in names:
        arrays[name] = position[arrays[name]].astype(np.int64)
    if kind != _NODES:
        arrays["user_sid"] = np.zeros(len(has_user), dtype=np.int64)
        arrays["user_sid"][has_user] = position[users]
    used = np.flatnonzero(mark)
    return _ElementChunk(arrays, [strings[i] for i in used.tolist()], kind, meta)


cdef _ElementChunk _taken(_ElementChunk chunk, rows):
    """The elements of `chunk` at the positions `rows`, keeping only the strings they
    use."""
    taken = chunk.take(rows)
    return _with_used_strings(dict(taken.arrays), taken.strings, taken.kind, taken.meta)


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


cdef _intern_first_uses(_StringTable st, strings, order, index):
    """Intern in `st` the strings `strings[i]` of the string ids `order`, each at its
    first use, and set `index[i]` to its id in `st`."""
    if len(order) == 0:
        return
    distinct, first = np.unique(order, return_index=True)
    used = distinct[np.argsort(first, kind="stable")]
    index[used] = [st.intern(strings[i]) for i in used.tolist()]


cdef dict _interned(_ElementChunk chunk, _StringTable st):
    """`chunk`'s arrays with its string ids replaced by ids into the block's string
    table `st`, interning the strings in the order `_string_order` gives."""
    a = chunk.arrays
    index = np.zeros(max(len(chunk.strings), 1), dtype=np.int64)
    _intern_first_uses(
        st, chunk.strings, _string_order(a, chunk.kind == _RELATIONS), index
    )
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
    """Decode an Info submessage to a dict of its fields in `_NODE_META` order, or None
    when no metadata is present.

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
        info["user_sid"] = info_msg.user_sid
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
    """The node chunk (see `_node_chunk`) of the nodes of the decoded dense group
    `dense` of `block` in `kept_nodes_set`: every node when it is None; None when no
    node is kept."""
    ids = dense["id"]
    if kept_nodes_set is None:
        keep = np.ones(len(ids), dtype=bool)
    else:
        keep = _isin(ids, kept_nodes_set)
    if not keep.any():
        return None
    meta = tuple(name for name in _NODE_META if len(dense[name]) > 0)
    arrays = {name: dense[name][keep] for name in _NODE_COLS + meta}
    if "visible" in meta:
        arrays["visible"] = arrays["visible"].astype(bool)
    off, keys, vals = _tag_rows(
        _kept_keys_vals(dense["keys_vals"], keep), len(arrays["id"])
    )
    return _node_chunk(arrays, block.strings(), meta, off, keys, vals)


cdef _plain_nodes_chunk(list run, stringtable):
    """The node chunk of the (node, info) pairs `run` of plain nodes, `info` as
    `_decode_info` gives it with the same fields for each, with the string table
    `stringtable` of their block."""
    meta = tuple(run[0][1] or ())
    arrays = {
        "id": np.asarray([node.id for node, _ in run], dtype=np.int64),
        "lat": np.asarray([node.lat for node, _ in run], dtype=np.int64),
        "lon": np.asarray([node.lon for node, _ in run], dtype=np.int64),
    }
    for name in meta:
        arrays[name] = np.asarray(
            [info[name] for _, info in run],
            dtype=bool if name == "visible" else np.int64,
        )
    tags = [list(zip(node.keys, node.vals)) for node, _ in run]
    off = np.zeros(len(run) + 1, dtype=np.int64)
    np.cumsum([len(pairs) for pairs in tags], out=off[1:])
    keys = np.asarray([k for pairs in tags for k, _ in pairs], dtype=np.int64)
    vals = np.asarray([v for pairs in tags for _, v in pairs], dtype=np.int64)
    return _node_chunk(arrays, stringtable, meta, off, keys, vals)


cdef _decode_kept_plain_nodes(nodes, stringtable):
    """Kept plain nodes -> node chunks, one per run of consecutive nodes with the same
    metadata fields (a dense block's DenseInfo is all-or-nothing per field)."""
    chunks, run = [], []
    for node in nodes:
        info = _decode_info(node.info)
        if run and tuple(info or ()) != tuple(run[-1][1] or ()):
            chunks.append(_plain_nodes_chunk(run, stringtable))
            run = []
        run.append((node, info))
    if run:
        chunks.append(_plain_nodes_chunk(run, stringtable))
    return chunks


# ---- re-chunk buffer (bounded: ~one output block per element type) ------------

cdef _ElementChunk _joined_elements(list chunks):
    """Chunks of one kind (and for nodes one metadata schema) joined into one, the
    strings of each after those of the chunks before it."""
    parts, strings = [], []
    for chunk in chunks:
        a = dict(chunk.arrays)
        base = len(strings)
        for name in ("keys", "vals", "roles"):
            if name in a:
                a[name] = a[name] + base
        if chunk.kind != _NODES:
            a["user_sid"] = np.where(
                a["info_fields"] & _HAS_USER, a["user_sid"] + base, 0
            )
        elif "user_sid" in a:
            a["user_sid"] = a["user_sid"] + base
        parts.append(a)
        strings.extend(chunk.strings)
    first = chunks[0]
    return _ElementChunk(_concat_elements(parts), strings, first.kind, first.meta)


class _Collected(list):
    """The blocks an emit function writes, in order."""

    def write(self, data):
        self.append(data)


cdef _emit_blocks(out, int kind, list chunks):
    """Write to `out` the blocks of the chunks `chunks` of `kind`: of nodes as
    `_emit_repack_nodes` writes them, of ways or relations as `_emit_repack_elements`
    does."""
    if kind == _NODES:
        _emit_repack_nodes(out, chunks)
    else:
        _emit_repack_elements(out, chunks, kind == _RELATIONS)


def _encoded_blocks(int kind, list chunks):
    """The framed blocks that `_emit_blocks` writes for `chunks`, in order."""
    out = _Collected()
    _emit_blocks(out, kind, chunks)
    return out


# How many groups of re-packed blocks the executor may encode ahead of the write.
_ENCODE_WINDOW = 16


cdef class _RepackWriter:
    """Re-chunks decoded element chunks, added in file order, into canonical full blocks
    and writes them in that order; with an `executor`, the blocks are encoded and
    compressed on it."""
    cdef object out
    cdef list chunks
    cdef list counts
    cdef readonly long nodes_written
    cdef object executor
    cdef object pending

    def __cinit__(self, out, executor=None):
        self.out = out
        self.chunks = [[], [], []]
        self.counts = [0, 0, 0]
        self.nodes_written = 0
        self.executor = executor
        self.pending = deque()

    cdef _emit(self, int kind, list chunks):
        """Encode and write the blocks of `chunks` (see `_emit_blocks`): here, or on the
        executor, the blocks then written in order."""
        if self.executor is None:
            _emit_blocks(self.out, kind, chunks)
            return
        while len(self.pending) >= _ENCODE_WINDOW:
            self._write_oldest()
        self.pending.append(self.executor.submit(_encoded_blocks, kind, chunks))

    cdef _write_oldest(self):
        """Write the oldest blocks encoded on the executor, once they are done."""
        for data in self.pending.popleft().result():
            self.out.write(data)

    cdef add(self, int kind, _ElementChunk chunk):
        """Buffer the decoded chunk `chunk` of elements of `kind`, writing each full
        block. The buffered elements of the kinds before it (nodes, then ways) are
        written first, and the buffered nodes before nodes with other metadata fields: a
        dense block's DenseInfo is all-or-nothing per field."""
        cdef int before
        for before in range(kind):
            self._flush(before)
        if kind == _NODES:
            if self.chunks[_NODES] and chunk.meta != self.chunks[_NODES][0].meta:
                self._flush(_NODES)
            self.nodes_written += len(chunk)
        self.chunks[kind].append(chunk)
        self.counts[kind] += len(chunk)
        while self.counts[kind] >= _MAX_GROUP:
            batch, self.chunks[kind] = _split_chunks(self.chunks[kind], _MAX_GROUP)
            self.counts[kind] -= _MAX_GROUP
            self._emit(kind, batch)

    cdef _flush(self, int kind):
        """Write the buffered elements of `kind` as one block."""
        if self.chunks[kind]:
            self._emit(kind, self.chunks[kind])
        self.chunks[kind] = []
        self.counts[kind] = 0

    cdef close(self):
        """Write everything buffered, and the blocks still being encoded."""
        for kind in (_NODES, _WAYS, _RELATIONS):
            self._flush(kind)
        while self.pending:
            self._write_oldest()


cdef _check_standard_grid(header, filepath):
    if (header["granularity"] != 100 or header["lat_offset"] != 0
            or header["lon_offset"] != 0 or header["date_granularity"] != 1000):
        raise ValueError(
            "to_pbf(repack=True) and merge_pbf() require the standard PBF grid "
            "(granularity 100, zero lat/lon offsets, date_granularity 1000), which "
            "'%s' does not use; to_pbf(repack=False) can still crop it." % filepath
        )


def _block_groups(data, filepath):
    """Yield (kind, block, group) for each primitive group of the decompressed block
    `data` of `filepath` that holds elements: `group` as `element_layout` gives it and
    `block` the `_Block` of `data`."""
    header, table, groups = _raw_decoded(element_layout, data, filepath)
    _check_standard_grid(header, filepath)
    block = _Block(data, table, filepath)
    for group in groups:
        kind = _group_kind(group)
        if kind is not None:
            yield kind, block, group


def _iter_groups(filepath, progress=None):
    """Yield (kind, block, group) for each primitive group of a PBF that holds elements
    (see `_block_groups`)."""
    for data in _iter_block_data(filepath, progress):
        yield from _block_groups(data, filepath)


cdef list _kept_chunks(data, filepath, kept):
    """The elements of the decompressed block `data` of `filepath` in the id sets `kept`
    (nodes, ways, relations), as (kind, chunk) pairs in block order (see
    `_decode_group`)."""
    return [
        (kind, chunk)
        for kind, block, group in _block_groups(data, filepath)
        for chunk in _decode_group(kind, block, group, kept[kind])
    ]


def _merge_input_blocks(source_path, n_inputs, kept_nodes, kept_ways, kept_rel, progress,
                        pool):
    """Yield (order, chunks) for each block of the merge input `source_path`, one of
    `n_inputs`, in file order: the order of its elements (see `_block_orders`) and its
    kept elements (see `_kept_chunks`), read, cropped and decoded by the threads of
    `pool`, a `_ThreadPool`."""
    pool.broadcast("kept_nodes", kept_nodes)
    pool.broadcast("kept_ways", kept_ways)
    pool.broadcast("kept_rel", kept_rel)
    yield from pool.imap(
        _batch_merge_input, _iter_payloads(source_path, progress),
        *_merge_read_ahead(pool, n_inputs, _THREAD_BATCH),
    )


# The blobs of a batch for the threads of a merge of whole inputs: every element is
# decoded, so few blocks are held ahead of the merge.
_WHOLE_MERGE_BATCH = 4


def _merge_read_ahead(pool, n_inputs, batch):
    """(blobs per batch, batches in flight) for each input of a merge of `n_inputs` inputs
    on the threads of `pool`, with batches of at most `batch` blobs: the threads shared
    out among the inputs, at least 2 batches per input so that each one is read ahead,
    and smaller batches, down to one blob, when the inputs outnumber the threads. The
    merge then holds about ``pool.workers * batch`` blobs ahead in all, and at most 2 per
    input beyond that."""
    window = max(2, -(-pool.workers // n_inputs))
    return max(1, min(batch, pool.workers * batch // (window * n_inputs))), window


def _whole_merge_input_blocks(source_path, n_inputs, progress, pool):
    """Yield (order, chunks) for each block of the merge input `source_path` read whole,
    one of `n_inputs`, in file order: the order of its elements (see `_block_orders`) and
    all its elements, read and decoded by the threads of `pool`, a `_ThreadPool`."""
    yield from pool.imap(
        _batch_merge_whole, _iter_payloads(source_path, progress),
        *_merge_read_ahead(pool, n_inputs, _WHOLE_MERGE_BATCH),
    )


def _repack_chunks(source_path, kept_nodes, kept_ways, kept_rel, progress, pool):
    """Yield the kept elements of each block of `source_path` (see `_kept_chunks`), in file
    order: read, decompressed and decoded by the threads of `pool` when it is a
    `_ThreadPool`, otherwise here."""
    if isinstance(pool, _ThreadPool):
        pool.broadcast("kept_nodes", kept_nodes)
        pool.broadcast("kept_ways", kept_ways)
        pool.broadcast("kept_rel", kept_rel)
        yield from pool.imap(_worker_repack, _iter_payloads(source_path, progress))
    else:
        kept = (_to_sorted_set(kept_nodes), _to_set(kept_ways), _to_set(kept_rel))
        for data in _iter_block_data(source_path, progress):
            yield _kept_chunks(data, source_path, kept)


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
    return [_element_chunk(decoded, block.strings(), kind)]


cdef _repack_write(source_path, output_path, kept_nodes, kept_ways, kept_rel, bounds,
                   progress, pool):
    """Re-pack write: re-chunk the kept crop (the kept id arrays) into canonical full
    blocks, decoded as `_repack_chunks` does, encoded on the threads of `pool` when it is
    a `_ThreadPool` and written in file order here."""
    with open(output_path, "wb") as out, progress.one_pass(_WRITING):
        _write_header(out, bounds)
        writer = _RepackWriter(
            out, pool.executor if isinstance(pool, _ThreadPool) else None
        )
        for chunks in _repack_chunks(
            source_path, kept_nodes, kept_ways, kept_rel, progress, pool
        ):
            for kind, chunk in chunks:
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
# and input order, and the winners of each such step go through the re-pack
# writer, in id order.

cdef _not_sorted(filepath):
    return ValueError(
        "'%s' is not sorted by type then id, which merging requires. Sort it first, "
        "e.g. with `osmium sort`." % filepath
    )


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
                versions, timestamps = chunk.ranks()
                yield chunk, chunk.ids, versions, timestamps


class _ChunkInput:
    """One merge input read as the (order, chunks) of its blocks in file order (see
    `_merge_input_blocks`): every element is checked to be sorted by type then id, as
    `_SortedInput` checks them, and the kept elements are merged."""

    def __init__(self, filepath, blocks):
        self.filepath = filepath
        self._chunks = self._checked(blocks)
        self._next = next(self._chunks, None)

    def _checked(self, blocks):
        """The (kind, chunk) pairs of `blocks`, each block checked for its order first."""
        last_kind, last_id = -1, None
        for order, chunks in blocks:
            for kind, first, last, rising in order.tolist():
                if kind < last_kind or not rising or (
                    kind == last_kind and first <= last_id
                ):
                    raise _not_sorted(self.filepath)
                last_kind, last_id = kind, last
            yield from chunks

    def chunks(self, kind, kept):
        """Yield (chunk, ids, versions, timestamps) for the elements of `kind` in the id
        set `kept` (all when None)."""
        while self._next is not None and self._next[0] <= kind:
            chunk = self._next[1]
            self._next = next(self._chunks, None)
            if kept is not None:
                rows = np.flatnonzero(kept.isin(chunk.ids))
                if len(rows) == 0:
                    continue
                chunk = _taken(chunk, rows)
            versions, timestamps = chunk.ranks()
            yield chunk, chunk.ids, versions, timestamps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _winning_copies(const int64_t[::1] ids, const int64_t[::1] versions,
                           const int64_t[::1] timestamps, const int64_t[::1] runs):
    """Positions of the winning copy of each id, in id order, among copies given as runs
    sorted by id: run r holds the rows ``runs[r]`` to ``runs[r + 1]``.

    The copy with the highest version wins, then the one with the latest timestamp,
    then the one from the earliest run. Returns (positions, r), r the first run whose
    ids do not rise, or -1.
    """
    cdef Py_ssize_t k = runs.shape[0] - 1, r, first, win, n = 0, unsorted = -1
    cdef int64_t low = 0, row
    cursors = np.array(runs[:k], dtype=np.int64)
    out = np.empty(ids.shape[0], dtype=np.int64)
    cdef int64_t[::1] at = cursors, o = out
    with nogil:
        while unsorted < 0:
            first = -1
            for r in range(k):
                if at[r] < runs[r + 1] and (first < 0 or ids[at[r]] < low):
                    first, low = r, ids[at[r]]
            if first < 0:
                break
            win = first
            for r in range(first + 1, k):
                row = at[r]
                if row < runs[r + 1] and ids[row] == low and (
                    versions[row] > versions[at[win]]
                    or (versions[row] == versions[at[win]]
                        and timestamps[row] > timestamps[at[win]])
                ):
                    win = r
            o[n] = at[win]
            n += 1
            for r in range(first, k):
                if at[r] < runs[r + 1] and ids[at[r]] == low:
                    at[r] += 1
                    if at[r] < runs[r + 1] and ids[at[r]] <= low:
                        unsorted = r
    return out[:n], unsorted


cdef _write_step(_RepackWriter writer, kind, list heads, list starts, list ends,
                 list won):
    """Write the winning copies among the rows ``starts[k]:ends[k]`` of the chunks of
    `heads` (the (chunk, ids, versions, timestamps) of each input, read in order), in id
    order; with `won`, a copy only when its id is in the set of its input in `won`. Each
    run of copies whose inputs share a node metadata schema goes to `writer` as one
    chunk."""
    bounds = np.cumsum([0] + [e - s for s, e in zip(starts, ends)], dtype=np.int64)
    ids, versions, timestamps = [
        np.concatenate([h[f][s:e] for h, s, e in zip(heads, starts, ends)])
        for f in (1, 2, 3)
    ]
    rows = _winning_copies(ids, versions, timestamps, bounds)[0]
    part = np.searchsorted(bounds, rows, side="right") - 1
    if won is not None:
        keep = np.zeros(len(rows), dtype=bool)
        for k in range(len(heads)):
            mine = part == k
            keep[mine] = won[k].isin(ids[rows[mine]])
        rows, part = rows[keep], part[keep]
    in_chunk = rows - bounds[part] + np.asarray(starts, dtype=np.int64)[part]

    schemas = [h[0].meta for h in heads]
    labels = np.array([schemas.index(s) for s in schemas], dtype=np.int64)
    label = labels[part]
    edges = [0] + (np.flatnonzero(label[1:] != label[:-1]) + 1).tolist() + [len(rows)]
    for a, b in zip(edges[:-1], edges[1:]):
        if a == b:
            continue
        members = np.unique(part[a:b])
        writer.add(kind, _gathered(
            [heads[k][0] for k in members.tolist()],
            np.searchsorted(members, part[a:b]), in_chunk[a:b],
        ))


cdef _gathered(list chunks, member, rows):
    """The elements of the decoded chunks `chunks` (of one kind and, for nodes, one
    metadata schema) at the positions `rows`, each in the chunk `member` names (each
    chunk's positions rising), in that order, as one chunk (see `_taken` for the strings
    each keeps)."""
    if len(chunks) == 1:
        if len(rows) == len(chunks[0]):
            return chunks[0]
        return _taken(chunks[0], rows)
    pieces = []
    order = np.empty(len(rows), dtype=np.int64)
    cdef Py_ssize_t placed = 0
    for k, chunk in enumerate(chunks):
        mine = np.flatnonzero(member == k)
        order[mine] = placed + np.arange(len(mine))
        placed += len(mine)
        pieces.append(_taken(chunk, rows[mine]))
    return _joined_elements(pieces).take(order)


cdef _merge_kind(inputs, kind, kept, won, _RepackWriter writer):
    """Write the elements of `kind` of all inputs, one copy per id, in id order."""
    streams = [inp.chunks(kind, kept) for inp in inputs]
    heads = [next(s, None) for s in streams]
    starts = [0] * len(heads)
    while True:
        live = [i for i in range(len(heads)) if heads[i] is not None]
        if not live:
            return
        frontier = min([heads[i][1][-1] for i in live])
        ends = [int(np.searchsorted(heads[i][1], frontier, side="right")) for i in live]
        _write_step(
            writer, kind, [heads[i] for i in live], [starts[i] for i in live], ends,
            None if won is None else [won[i] for i in live],
        )
        for i, end in zip(live, ends):
            if end == len(heads[i][1]):
                heads[i] = next(streams[i], None)
                starts[i] = 0
            else:
                starts[i] = end


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

    bounding_box : list | tuple | numpy.ndarray | shapely geometry | GeoDataFrame | GeoSeries, optional
        The box in lon/lat, in any form :class:`pyrosm.OSM` takes: four numbers
        ``[minx, miny, maxx, maxy]``, or a geometry or GeoDataFrame, which is cropped by
        its envelope. The output's header bounding box is this box (the polygon's
        envelope with ``polygon``), or else the union of the inputs' header boxes. A
        box that ``OSM`` refuses is refused, and a merge that keeps no node warns.

    keep_relations : bool
        When ``True`` (default) relations are written (with ``bounding_box`` or
        ``polygon``, those referencing a kept node or way); when ``False`` none
        are written.

    workers : int | str
        Number of threads for the merge: they select the elements inside
        ``bounding_box`` or ``polygon``, read and decode the inputs' blocks (cropped
        first to the kept elements with a region) and encode the merged blocks, which
        are written in order. By default (``None``, as ``"auto"``) the count is chosen as
        an out-of-core read does for the inputs' total size; ``1`` runs sequentially.
        With ``PYROSM_DECODE_BACKEND=processes`` they are worker processes, one by
        default, which only select the elements; when those cannot run (on macOS and
        Windows: a script without an ``if __name__ == "__main__":`` guard, or one read
        from stdin), the merge runs in one process and warns.

    polygon : shapely Polygon or MultiPolygon | GeoDataFrame | GeoSeries, optional
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

    made = output_path is None
    if made:
        fd, output_path = tempfile.mkstemp(suffix=".osm.pbf", prefix="pyrosm_merge_")
        os.close(fd)
    try:
        if bounding_box is None and polygon is None:
            kept = (None, None, None if keep_relations else np.empty(0, dtype=np.int64))
            progress.plan([_WRITING])
            _with_pool(
                _write_whole_merge,
                (sources, fingerprints, _header_bounds(headers), kept, progress,
                 output_path),
                workers, sources, (None, None), False, threads_only=True,
            )
        else:
            region = _region(bounding_box, polygon)
            _with_pool(
                _select_and_merge,
                (sources, fingerprints, region, keep_relations, progress, output_path),
                workers, sources, region, False,
            )
    except BaseException:
        if made:
            Path(output_path).unlink(missing_ok=True)
        raise
    return output_path


def _write_whole_merge(sources, fingerprints, bounds, kept, progress, output_path,
                       pool):
    """Write the merge of the whole inputs `sources` to `output_path` (see
    `_write_merge`), with `pool` when one is used."""
    write_atomic_path(output_path, partial(
        _write_merge, sources, fingerprints, bounds, False, kept, (None, None, None),
        progress, pool,
    ))


def _select_and_merge(sources, fingerprints, region, keep_relations, progress,
                      output_path, pool):
    """Select the elements a merge cropped to `region` keeps (see `_select`) and write
    the merge to `output_path`, both with `pool` when one is used; without one, the
    selection runs the stages of the threads on the calling thread (see
    `_InlinePool`)."""
    selecting = _InlinePool(region) if pool is None else pool
    kept_nodes, kept_ways, kept_rel = _select(
        sources, region, keep_relations, selecting, progress
    )
    del selecting
    kept = (kept_nodes, _unique_concat(kept_ways), _unique_concat(kept_rel))
    won = (None, [_to_set(w) for w in kept_ways], [_to_set(r) for r in kept_rel])
    write_atomic_path(output_path, partial(
        _write_merge, sources, fingerprints, region[0], True, kept, won, progress, pool
    ))


def _write_merge(sources, fingerprints, bounds, boxed, kept, won, progress, pool, path):
    """Write the merge of `_merge` to `path` with the header box `bounds`, refusing it
    when an input changed since its `fingerprints`; warn when no node is written (none
    lies in the box `bounds` when `boxed`, else the inputs hold none). `kept` holds the
    ids of the nodes, ways and relations to write (None: all of them). With a
    `_ThreadPool`, its threads read and decode the inputs' blocks, cropped first when
    `boxed` (see `_merge_input_blocks`, `_whole_merge_input_blocks`), and encode the
    merged blocks."""
    threaded = isinstance(pool, _ThreadPool)
    with open(path, "wb") as out, progress.one_pass(_WRITING):
        if threaded and boxed:
            inputs = [
                _ChunkInput(
                    p, _merge_input_blocks(
                        p, len(sources), kept[0], kept[1], kept[2], progress, pool
                    )
                )
                for p in sources
            ]
            kept_sets = (None, None, None)
        else:
            if threaded:
                inputs = [
                    _ChunkInput(
                        p, _whole_merge_input_blocks(p, len(sources), progress, pool)
                    )
                    for p in sources
                ]
            else:
                inputs = [_SortedInput(p, progress) for p in sources]
            kept_sets = (
                None if kept[0] is None else _to_sorted_set(kept[0]),
                None if kept[1] is None else _to_set(kept[1]),
                None if kept[2] is None else _to_set(kept[2]),
            )
        _write_header(out, bounds, sorted_output=True)
        writer = _RepackWriter(out, pool.executor if threaded else None)
        for kind in (_NODES, _WAYS, _RELATIONS):
            _merge_kind(inputs, kind, kept_sets[kind], won[kind], writer)
        writer.close()
    if writer.nodes_written == 0:
        _warn_empty(sources, bounds if boxed else None)
    changed = [p for p, before in zip(sources, fingerprints) if _fingerprint(p) != before]
    if changed:
        raise ValueError(
            "'%s' changed while it was being merged; the merge was discarded." % changed[0]
        )
