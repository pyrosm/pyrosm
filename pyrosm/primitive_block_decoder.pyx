# cython: boundscheck=False, wraparound=False
# Bounds checks are disabled for speed: every index below is bounded by hand
# (``i < count``, ``pos < end``, and an empty-buffer guard in decode_primitive_block).
"""Raw decoder for the PBF ``PrimitiveBlock`` -- the hot path of OSM PBF reading.

This reads the decompressed ``PrimitiveBlock`` bytes directly, in the protobuf wire
format documented in ``proto/osmformat.proto``, and returns the node / way / relation
arrays pyrosm needs -- without building the protobuf library's per-field Python
objects. Only this one message is hand-decoded; the cheap, infrequent ``BlobHeader`` /
``Blob`` / ``HeaderBlock`` structures keep using protobuf.

Wire-format primer (the field numbers live in ``primitive_block_decoder.pxd``):

* Every field is preceded by a *tag* varint = ``(field_number << 3) | wire_type``.
* Wire type 0 is a single varint value; wire type 2 is a length-delimited block (a
  sub-message, a packed repeated field, or raw bytes).
* A packed repeated number field is a length-delimited run of back-to-back varints.
* Signed numbers (``sint64``/``sint32``) are zig-zag encoded; several of them are
  additionally *delta coded* (each value is added to the running total), which we undo
  with a cumulative sum -- mirroring ``delta_compression.pyx``.

Nodes are decoded straight into flat arrays. Ways and relations are variable length, so
they are decoded in two passes: a cheap counting pass sizes the flat (CSR) arrays
exactly, then a single fill pass writes into them -- no per-element allocation.
Coordinates and ids are returned as raw cumulative integers (not scaled by
``granularity``/offset); scaling is the caller's job, which keeps the values exact and
easy to compare against the protobuf path in tests.
"""
import numpy as np

from libc.stdint cimport int64_t, uint64_t


# The per-element metadata columns (from ``Info`` / ``DenseInfo``), in output order.
_META_COLS = ("version", "timestamp", "changeset", "uid", "user_sid", "visible")


# The protobuf wire types we encounter.
cdef enum WireType:
    WIRE_VARINT = 0
    WIRE_FIXED64 = 1
    WIRE_LENGTH = 2
    WIRE_FIXED32 = 5


# --- low-level wire-format primitives -------------------------------------------------

cdef inline uint64_t _read_varint(const unsigned char* buf, Py_ssize_t* pos,
                                  Py_ssize_t end) nogil:
    """Read one base-128 varint at ``pos`` and advance ``pos`` past it."""
    cdef uint64_t result = 0
    cdef int shift = 0
    cdef unsigned char byte
    while pos[0] < end:
        byte = buf[pos[0]]
        pos[0] += 1
        result |= (<uint64_t>(byte & 0x7F)) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result


cdef inline int64_t _zig_zag_decode(uint64_t value) nogil:
    """Undo protobuf's zig-zag encoding of a signed integer."""
    return (<int64_t>(value >> 1)) ^ (-(<int64_t>(value & 1)))


cdef inline void _read_tag(const unsigned char* buf, Py_ssize_t* pos, Py_ssize_t end,
                           int* field_number, int* wire_type) nogil:
    """Read a field tag and split it into its field number and wire type."""
    cdef uint64_t tag = _read_varint(buf, pos, end)
    field_number[0] = <int>(tag >> 3)
    wire_type[0] = <int>(tag & 0x7)


cdef inline void _skip_field(const unsigned char* buf, Py_ssize_t* pos,
                             Py_ssize_t end, int wire_type) nogil:
    """Advance ``pos`` past a field whose value we don't need."""
    cdef uint64_t length
    if wire_type == WIRE_VARINT:
        _read_varint(buf, pos, end)
    elif wire_type == WIRE_LENGTH:
        length = _read_varint(buf, pos, end)
        pos[0] += <Py_ssize_t>length
    elif wire_type == WIRE_FIXED64:
        pos[0] += 8
    elif wire_type == WIRE_FIXED32:
        pos[0] += 4


cdef inline Py_ssize_t _count_packed(const unsigned char* buf, Py_ssize_t start,
                                     Py_ssize_t end) nogil:
    """Count the varints packed in ``buf[start:end]`` (one ends per continuation-clear
    byte) without decoding them -- used by the way/relation sizing pass."""
    cdef Py_ssize_t pos = start
    cdef Py_ssize_t count = 0
    while pos < end:
        if (buf[pos] & 0x80) == 0:
            count += 1
        pos += 1
    return count


cdef Py_ssize_t _fill_packed(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                             bint is_signed, int64_t[::1] out, Py_ssize_t at) nogil:
    """Decode a packed run of varints into ``out[at:]``; return how many were written.
    ``is_signed`` zig-zag-decodes each value."""
    cdef Py_ssize_t pos = start
    cdef Py_ssize_t i = at
    cdef uint64_t raw
    while pos < end:
        raw = _read_varint(buf, &pos, end)
        if is_signed:
            out[i] = _zig_zag_decode(raw)
        else:
            out[i] = <int64_t>raw
        i += 1
    return i - at


cdef Py_ssize_t _fill_packed_delta(const unsigned char* buf, Py_ssize_t start,
                                   Py_ssize_t end, int64_t[::1] out,
                                   Py_ssize_t at) nogil:
    """Like ``_fill_packed`` for a delta-coded ``sint`` field: write the running
    cumulative sum into ``out[at:]`` and return how many were written."""
    cdef Py_ssize_t pos = start
    cdef Py_ssize_t i = at
    cdef int64_t running = 0
    while pos < end:
        running += _zig_zag_decode(_read_varint(buf, &pos, end))
        out[i] = running
        i += 1
    return i - at


cdef _read_packed(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                  bint is_signed):
    """Decode a packed run of varints into a freshly allocated ``int64`` array (used by
    the dense-node path, which is naturally flat)."""
    out = np.empty(_count_packed(buf, start, end), dtype=np.int64)
    cdef int64_t[::1] values = out
    _fill_packed(buf, start, end, is_signed, values, 0)
    return out


# --- string table & dense nodes -------------------------------------------------------

cdef list _decode_string_table(const unsigned char* buf, Py_ssize_t start,
                               Py_ssize_t end):
    """Decode a ``StringTable`` into a list of ``bytes`` (one per string id)."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t s_start
    strings = []
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            s_start = pos
            pos += <Py_ssize_t>length
            if field == ST_S:
                strings.append((<char*>buf)[s_start:pos])
        else:
            _skip_field(buf, &pos, end, wire)
    return strings


cdef _decode_dense_info(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        dict node):
    """Decode a ``DenseInfo`` (per-node metadata) into ``node``. ``version`` and
    ``visible`` are plain per-node values; the rest are delta coded."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == DI_VERSION:
                node["version"] = _read_packed(buf, f_start, pos, False)
            elif field == DI_TIMESTAMP:
                node["timestamp"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DI_CHANGESET:
                node["changeset"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DI_UID:
                node["uid"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DI_USER_SID:
                node["user_sid"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DI_VISIBLE:
                node["visible"] = _read_packed(buf, f_start, pos, False)
        else:
            _skip_field(buf, &pos, end, wire)


cdef dict _decode_dense_group(const unsigned char* buf, Py_ssize_t start,
                              Py_ssize_t end, bint keep_metadata):
    """Decode one ``DenseNodes`` group into a dict of arrays (ids/lats/lons are
    cumulative; ``keys_vals`` is the raw 0-delimited tag stream)."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    node = {"id": None, "lat": None, "lon": None,
            "keys_vals": np.empty(0, dtype=np.int64)}
    # A DenseInfo field that is absent (e.g. ``visible`` outside history files) stays
    # an empty array, matching what the protobuf path returns for it.
    if keep_metadata:
        for name in _META_COLS:
            node[name] = np.empty(0, dtype=np.int64)
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == DENSE_ID:
                node["id"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DENSE_LAT:
                node["lat"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DENSE_LON:
                node["lon"] = np.cumsum(_read_packed(buf, f_start, pos, True))
            elif field == DENSE_KEYS_VALS:
                node["keys_vals"] = _read_packed(buf, f_start, pos, False)
            elif field == DENSE_DENSEINFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    if keep_metadata and info_start >= 0:
        _decode_dense_info(buf, info_start, info_end, node)
    return node


cdef dict _merge_dense(list groups, bint keep_metadata):
    """Concatenate the per-group dense-node dicts into one. (Delta coding is per
    group, so each group's arrays are already cumulative before they are joined.)"""
    if len(groups) == 0:
        return None
    out = {
        "id": np.concatenate([g["id"] for g in groups]),
        "lat": np.concatenate([g["lat"] for g in groups]),
        "lon": np.concatenate([g["lon"] for g in groups]),
        "keys_vals": np.concatenate([g["keys_vals"] for g in groups]),
    }
    if keep_metadata:
        for name in _META_COLS:
            out[name] = np.concatenate([g[name] for g in groups])
    return out


# --- ways & relations (two-pass flat decode) ------------------------------------------

cdef class _ElementArrays:
    """Flat output buffers for one block's ways or relations, sized exactly by a
    counting pass and then filled in a single pass. ``members`` is way refs or relation
    memids; ``types``/``roles`` are relation-only. ``*_off`` are CSR offsets so element
    ``i`` owns ``members[members_off[i]:members_off[i + 1]]`` (and likewise its tags)."""
    cdef int64_t[::1] ids, keys, vals, tags_off, members, members_off, types, roles
    cdef int64_t[::1] version, timestamp, changeset, uid, user_sid, visible
    cdef bint keep_metadata, is_relation
    cdef Py_ssize_t ei, ti, mi          # next element / tag / member write position
    cdef dict _arr                      # the owning numpy arrays, by name

    def __cinit__(self, Py_ssize_t n_elem, Py_ssize_t n_tags, Py_ssize_t n_members,
                  bint keep_metadata, bint is_relation):
        self.keep_metadata = keep_metadata
        self.is_relation = is_relation
        self.ei = 0
        self.ti = 0
        self.mi = 0
        self._arr = {
            "id": np.empty(n_elem, np.int64),
            "keys": np.empty(n_tags, np.int64),
            "vals": np.empty(n_tags, np.int64),
            "tags_off": np.empty(n_elem + 1, np.int64),
            "members": np.empty(n_members, np.int64),
            "members_off": np.empty(n_elem + 1, np.int64),
        }
        if is_relation:
            self._arr["types"] = np.empty(n_members, np.int64)
            self._arr["roles"] = np.empty(n_members, np.int64)
        if keep_metadata:
            for name in _META_COLS:
                self._arr[name] = np.empty(n_elem, np.int64)
        self.ids = self._arr["id"]
        self.keys = self._arr["keys"]
        self.vals = self._arr["vals"]
        self.tags_off = self._arr["tags_off"]
        self.members = self._arr["members"]
        self.members_off = self._arr["members_off"]
        self.tags_off[0] = 0
        self.members_off[0] = 0
        if is_relation:
            self.types = self._arr["types"]
            self.roles = self._arr["roles"]
        if keep_metadata:
            self.version = self._arr["version"]
            self.timestamp = self._arr["timestamp"]
            self.changeset = self._arr["changeset"]
            self.uid = self._arr["uid"]
            self.user_sid = self._arr["user_sid"]
            self.visible = self._arr["visible"]

    cdef dict result(self):
        cdef str member = "memids" if self.is_relation else "refs"
        cdef str off = "members_off" if self.is_relation else "refs_off"
        out = {
            "id": self._arr["id"],
            "keys": self._arr["keys"],
            "vals": self._arr["vals"],
            "tags_off": self._arr["tags_off"],
            member: self._arr["members"],
            off: self._arr["members_off"],
        }
        if self.is_relation:
            out["types"] = self._arr["types"]
            out["roles"] = self._arr["roles"]
        if self.keep_metadata:
            for name in _META_COLS:
                out[name] = self._arr[name]
        return out


cdef void _fill_metadata(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         _ElementArrays b, Py_ssize_t i):
    """Decode an ``Info`` sub-message (or the proto2 default when absent) into ``b``'s
    per-element metadata arrays at index ``i``. Info fields are plain absolute values."""
    cdef int64_t version = -1, timestamp = 0, changeset = 0
    cdef int64_t uid = 0, user_sid = 0, visible = 0
    cdef Py_ssize_t pos
    cdef int field, wire
    if start >= 0:
        pos = start
        while pos < end:
            _read_tag(buf, &pos, end, &field, &wire)
            if wire == WIRE_VARINT:
                if field == INFO_VERSION:
                    version = <int64_t>_read_varint(buf, &pos, end)
                elif field == INFO_TIMESTAMP:
                    timestamp = <int64_t>_read_varint(buf, &pos, end)
                elif field == INFO_CHANGESET:
                    changeset = <int64_t>_read_varint(buf, &pos, end)
                elif field == INFO_UID:
                    uid = <int64_t>_read_varint(buf, &pos, end)
                elif field == INFO_USER_SID:
                    user_sid = <int64_t>_read_varint(buf, &pos, end)
                elif field == INFO_VISIBLE:
                    visible = <int64_t>_read_varint(buf, &pos, end)
                else:
                    _read_varint(buf, &pos, end)
            else:
                _skip_field(buf, &pos, end, wire)
    b.version[i] = version
    b.timestamp[i] = timestamp
    b.changeset[i] = changeset
    b.uid[i] = uid
    b.user_sid[i] = user_sid
    b.visible[i] = visible


cdef void _count_element(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         int keys_field, int members_field, Py_ssize_t* n_tags,
                         Py_ssize_t* n_members) nogil:
    """Add one way/relation's tag and member counts to the running totals, so the flat
    arrays can be sized exactly. (keys and vals share a count; types/roles share the
    member count.)"""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == keys_field:
                n_tags[0] += _count_packed(buf, f_start, pos)
            elif field == members_field:
                n_members[0] += _count_packed(buf, f_start, pos)
        else:
            _skip_field(buf, &pos, end, wire)


cdef void _fill_way(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                    _ElementArrays b):
    """Decode one ``Way`` into the builder at its current write positions."""
    cdef Py_ssize_t pos = start, f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t tstart = b.ti, mstart = b.mi
    cdef Py_ssize_t n_keys = 0, n_refs = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            if field == WAY_ID:
                b.ids[b.ei] = <int64_t>_read_varint(buf, &pos, end)
            else:
                _read_varint(buf, &pos, end)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == WAY_KEYS:
                n_keys = _fill_packed(buf, f_start, pos, False, b.keys, tstart)
            elif field == WAY_VALS:
                _fill_packed(buf, f_start, pos, False, b.vals, tstart)
            elif field == WAY_REFS:
                n_refs = _fill_packed_delta(buf, f_start, pos, b.members, mstart)
            elif field == WAY_INFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    b.ti = tstart + n_keys
    b.mi = mstart + n_refs
    b.tags_off[b.ei + 1] = b.ti
    b.members_off[b.ei + 1] = b.mi
    if b.keep_metadata:
        _fill_metadata(buf, info_start, info_end, b, b.ei)
    b.ei += 1


cdef void _fill_relation(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         _ElementArrays b):
    """Decode one ``Relation`` into the builder. Members carry an id (delta), a type and
    a role, all the same count, sharing the member offsets."""
    cdef Py_ssize_t pos = start, f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t tstart = b.ti, mstart = b.mi
    cdef Py_ssize_t n_keys = 0, n_members = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            if field == REL_ID:
                b.ids[b.ei] = <int64_t>_read_varint(buf, &pos, end)
            else:
                _read_varint(buf, &pos, end)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == REL_KEYS:
                n_keys = _fill_packed(buf, f_start, pos, False, b.keys, tstart)
            elif field == REL_VALS:
                _fill_packed(buf, f_start, pos, False, b.vals, tstart)
            elif field == REL_MEMIDS:
                n_members = _fill_packed_delta(buf, f_start, pos, b.members, mstart)
            elif field == REL_TYPES:
                _fill_packed(buf, f_start, pos, False, b.types, mstart)
            elif field == REL_ROLES_SID:
                _fill_packed(buf, f_start, pos, False, b.roles, mstart)
            elif field == REL_INFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    b.ti = tstart + n_keys
    b.mi = mstart + n_members
    b.tags_off[b.ei + 1] = b.ti
    b.members_off[b.ei + 1] = b.mi
    if b.keep_metadata:
        _fill_metadata(buf, info_start, info_end, b, b.ei)
    b.ei += 1


cdef dict _decode_elements(const unsigned char* buf, list group_ranges,
                           int element_field, int keys_field, int members_field,
                           bint keep_metadata, bint is_relation):
    """Two-pass decode of every ``element_field`` (way or relation) across the block's
    groups: count to size the flat arrays, then fill them. Returns ``None`` if there
    are none."""
    cdef Py_ssize_t n_elem = 0, n_tags = 0, n_members = 0
    cdef Py_ssize_t gs, ge, pos, f_start
    cdef int field, wire
    cdef uint64_t length

    for gr in group_ranges:
        gs = gr[0]
        ge = gr[1]
        pos = gs
        while pos < ge:
            _read_tag(buf, &pos, ge, &field, &wire)
            if wire == WIRE_LENGTH:
                length = _read_varint(buf, &pos, ge)
                f_start = pos
                pos += <Py_ssize_t>length
                if field == element_field:
                    n_elem += 1
                    _count_element(buf, f_start, pos, keys_field, members_field,
                                   &n_tags, &n_members)
            else:
                _skip_field(buf, &pos, ge, wire)
    if n_elem == 0:
        return None

    cdef _ElementArrays b = _ElementArrays(n_elem, n_tags, n_members, keep_metadata,
                                           is_relation)
    for gr in group_ranges:
        gs = gr[0]
        ge = gr[1]
        pos = gs
        while pos < ge:
            _read_tag(buf, &pos, ge, &field, &wire)
            if wire == WIRE_LENGTH:
                length = _read_varint(buf, &pos, ge)
                f_start = pos
                pos += <Py_ssize_t>length
                if field == element_field:
                    if is_relation:
                        _fill_relation(buf, f_start, pos, b)
                    else:
                        _fill_way(buf, f_start, pos, b)
            else:
                _skip_field(buf, &pos, ge, wire)
    return b.result()


cdef void _collect_dense(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         bint keep_metadata, list dense_groups):
    """Decode every ``DenseNodes`` block in one group into ``dense_groups``."""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == GROUP_DENSE:
                dense_groups.append(
                    _decode_dense_group(buf, f_start, pos, keep_metadata))
        else:
            _skip_field(buf, &pos, end, wire)


def decode_primitive_block(const unsigned char[::1] data, bint keep_metadata=True):
    """Decode a decompressed ``PrimitiveBlock`` into ``(string_table, header, nodes,
    ways, relations)``.

    ``string_table`` is a list of ``bytes``; ``header`` carries ``granularity`` /
    ``date_granularity`` / ``lat_offset`` / ``lon_offset``. ``nodes`` / ``ways`` /
    ``relations`` are dicts of numpy arrays (or ``None`` when the block has none of
    that element). Tag and member arrays use CSR ``*_off`` offsets. When
    ``keep_metadata`` is False the per-element version/timestamp/etc. are skipped.
    """
    cdef Py_ssize_t end = data.shape[0]
    if end == 0:
        return [], {"granularity": 100, "date_granularity": 1000,
                    "lat_offset": 0, "lon_offset": 0}, None, None, None
    cdef const unsigned char* buf = &data[0]
    cdef Py_ssize_t pos = 0
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start

    string_table = []
    header = {"granularity": 100, "date_granularity": 1000,
              "lat_offset": 0, "lon_offset": 0}
    dense_groups = []
    group_ranges = []

    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            if field == PB_GRANULARITY:
                header["granularity"] = <int64_t>_read_varint(buf, &pos, end)
            elif field == PB_DATE_GRANULARITY:
                header["date_granularity"] = <int64_t>_read_varint(buf, &pos, end)
            elif field == PB_LAT_OFFSET:
                header["lat_offset"] = <int64_t>_read_varint(buf, &pos, end)
            elif field == PB_LON_OFFSET:
                header["lon_offset"] = <int64_t>_read_varint(buf, &pos, end)
            else:
                _read_varint(buf, &pos, end)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos += <Py_ssize_t>length
            if field == PB_STRINGTABLE:
                string_table = _decode_string_table(buf, f_start, pos)
            elif field == PB_PRIMITIVEGROUP:
                group_ranges.append((f_start, pos))
                _collect_dense(buf, f_start, pos, keep_metadata, dense_groups)
        else:
            _skip_field(buf, &pos, end, wire)

    nodes = _merge_dense(dense_groups, keep_metadata)
    ways = _decode_elements(buf, group_ranges, GROUP_WAYS, WAY_KEYS, WAY_REFS,
                            keep_metadata, False)
    relations = _decode_elements(buf, group_ranges, GROUP_RELATIONS, REL_KEYS,
                                 REL_MEMIDS, keep_metadata, True)
    return string_table, header, nodes, ways, relations
