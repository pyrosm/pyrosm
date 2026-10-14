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

Dense nodes are decoded straight into flat arrays. Ways, relations and plain (non-dense)
nodes are variable length, so they are decoded in two passes: a cheap counting pass sizes
the flat (CSR) arrays exactly, then a single fill pass writes into them -- no per-element
allocation. Plain nodes are returned in the same form as dense ones.
Coordinates and ids are returned as raw cumulative integers (not scaled by
``granularity``/offset); scaling is the caller's job, which keeps the values exact and
easy to compare against the protobuf path in tests.
"""
import numpy as np

from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport int64_t, uint64_t

from pyrosm.exceptions import InvalidOSMFileError


# The per-element metadata columns (from ``Info`` / ``DenseInfo``), in output order.
_META_COLS = ("version", "timestamp", "changeset", "uid", "user_sid", "visible")


# The protobuf wire types we encounter.
cdef enum WireType:
    WIRE_VARINT = 0
    WIRE_FIXED64 = 1
    WIRE_LENGTH = 2
    WIRE_FIXED32 = 5


# --- low-level wire-format primitives -------------------------------------------------

# The position a read moves to when the data does not follow the wire format: a varint
# cut off at the end of its data or longer than 10 bytes, a field that does not fit in
# the data that holds it, or an unknown wire type. It lies past the end of any data, so
# the loop over that data stops; the function that owns the loop then raises.
cdef Py_ssize_t _CORRUPT = PY_SSIZE_T_MAX


cdef int _corrupt(str reason) except -1:
    """Raise the error for a data block that does not follow the PBF schema."""
    raise InvalidOSMFileError("A data block of the PBF is corrupt: %s." % reason)


cdef int _check_end(Py_ssize_t pos, Py_ssize_t end) except -1:
    """Raise unless a loop over data that ends at ``end`` stopped exactly there."""
    if pos != end:
        _corrupt("a field or a number does not fit in the data that holds it")
    return 0


cdef inline uint64_t _read_varint(const unsigned char* buf, Py_ssize_t* pos,
                                  Py_ssize_t end) nogil:
    """Read one base-128 varint at ``pos`` and advance ``pos`` past it. A varint cut off
    at ``end`` or larger than 64 bits moves ``pos`` to ``_CORRUPT``."""
    cdef uint64_t result = 0
    cdef int shift = 0
    cdef unsigned char byte
    while pos[0] < end and shift < 64:
        byte = buf[pos[0]]
        pos[0] += 1
        result |= (<uint64_t>(byte & 0x7F)) << shift
        if (byte & 0x80) == 0:
            if shift == 63 and byte > 1:
                break
            return result
        shift += 7
    pos[0] = _CORRUPT
    return 0


cdef inline Py_ssize_t _field_end(Py_ssize_t pos, uint64_t length,
                                  Py_ssize_t end) nogil:
    """The end of a field of ``length`` bytes that starts at ``pos``, or ``_CORRUPT``
    when the field does not fit in data that ends at ``end``."""
    if pos > end or length > <uint64_t>(end - pos):
        return _CORRUPT
    return pos + <Py_ssize_t>length


cdef inline int64_t _zig_zag_decode(uint64_t value) nogil:
    """Undo protobuf's zig-zag encoding of a signed integer."""
    return (<int64_t>(value >> 1)) ^ (-(<int64_t>(value & 1)))


cdef inline void _read_tag(const unsigned char* buf, Py_ssize_t* pos, Py_ssize_t end,
                           int* field_number, int* wire_type) nogil:
    """Read a field tag and split it into its field number and wire type. A field
    number outside 1 to 2**29 - 1 moves ``pos`` to ``_CORRUPT``."""
    cdef uint64_t tag = _read_varint(buf, pos, end)
    if tag < 8 or (tag >> 32) != 0:
        pos[0] = _CORRUPT
        tag = 0
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
        pos[0] = _field_end(pos[0], length, end)
    elif wire_type == WIRE_FIXED64:
        pos[0] = _field_end(pos[0], 8, end)
    elif wire_type == WIRE_FIXED32:
        pos[0] = _field_end(pos[0], 4, end)
    else:
        pos[0] = _CORRUPT


cdef inline Py_ssize_t _count_packed(const unsigned char* buf, Py_ssize_t start,
                                     Py_ssize_t end) nogil:
    """Count the varints packed in ``buf[start:end]`` (one ends per continuation-clear
    byte, plus one cut off at ``end``) without decoding them -- used to size the arrays
    the ``_fill_packed*`` calls write."""
    cdef Py_ssize_t pos = start
    cdef Py_ssize_t count = 0
    while pos < end:
        if (buf[pos] & 0x80) == 0:
            count += 1
        pos += 1
    if end > start and (buf[end - 1] & 0x80) != 0:
        count += 1
    return count


cdef Py_ssize_t _fill_packed(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                             bint is_signed, int64_t[::1] out, Py_ssize_t at) nogil:
    """Decode a packed run of varints into ``out[at:]``; return how many were written,
    or -1 when a varint is cut off or larger than 64 bits. ``is_signed`` zig-zag-decodes
    each value. ``out`` has a slot for each varint ``_count_packed`` counts."""
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
    if pos != end:
        return -1
    return i - at


cdef Py_ssize_t _fill_packed_delta(const unsigned char* buf, Py_ssize_t start,
                                   Py_ssize_t end, int64_t[::1] out, Py_ssize_t at,
                                   int64_t running) nogil:
    """Like ``_fill_packed`` for a delta-coded ``sint`` field: write the running
    cumulative sum, which starts from ``running``, into ``out[at:]`` and return how many
    were written, or -1."""
    cdef Py_ssize_t pos = start
    cdef Py_ssize_t i = at
    while pos < end:
        running += _zig_zag_decode(_read_varint(buf, &pos, end))
        out[i] = running
        i += 1
    if pos != end:
        return -1
    return i - at


cdef inline Py_ssize_t _filled(Py_ssize_t n) except -1:
    """``n``, the count a ``_fill_packed*`` call wrote; raise when it met a varint that
    is cut off or larger than 64 bits (-1)."""
    if n < 0:
        _corrupt("a number does not fit in the data that holds it")
    return n


cdef _read_packed(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                  bint is_signed):
    """Decode a packed run of varints into a freshly allocated ``int64`` array (used by
    the dense-node path, which is naturally flat)."""
    cdef Py_ssize_t n = _count_packed(buf, start, end)
    out = np.empty(n, dtype=np.int64)
    cdef int64_t[::1] values = out
    if _fill_packed(buf, start, end, is_signed, values, 0) != n:
        _corrupt("a number does not fit in the data that holds it")
    return out


cdef int _check_keys_vals(const int64_t[::1] keys_vals, Py_ssize_t n) except -1:
    """Raise unless the tag stream of ``n`` dense nodes is empty or holds, for each node
    in turn, its key and value ids followed by a 0."""
    cdef Py_ssize_t i, ends = 0
    # Masks (0 or -1) rather than branches: whether the next id is a key, whether it is
    # 0, and whether it ended a node.
    cdef int64_t at_key = -1, zero, ended = 0
    if keys_vals.shape[0] == 0:
        return 0
    for i in range(keys_vals.shape[0]):
        zero = -<int64_t>(keys_vals[i] == 0)
        ended = at_key & zero
        ends -= ended
        at_key = ~at_key | zero
    if ends != n or ended == 0:
        _corrupt("the tags of a dense node group do not end once for each of its %d "
                 "nodes" % n)
    return 0


# --- string table & dense nodes -------------------------------------------------------

cdef _add_run(dict node, dict repeats, str name, values):
    """Store a packed run of field ``name`` in ``node``. A field that comes in several
    runs collects them in ``repeats``, for ``_join_runs`` to join once."""
    previous = node.get(name)
    if previous is None or len(previous) == 0:
        node[name] = values
    elif name in repeats:
        repeats[name].append(values)
    else:
        repeats[name] = [previous, values]


cdef _join_runs(dict node, dict repeats):
    """Replace each field of ``node`` that came in several runs by the runs joined."""
    for name, runs in repeats.items():
        node[name] = np.concatenate(runs)


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
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == ST_S:
                strings.append((<char*>buf)[s_start:pos])
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    return strings


cdef _decode_dense_info(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        dict node):
    """Decode a ``DenseInfo`` (per-node metadata) into ``node``. ``version`` and
    ``visible`` are plain per-node values; the rest are delta coded."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start
    repeats = {}
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == DI_VERSION:
                _add_run(node, repeats, "version", _read_packed(buf, f_start, pos, False))
            elif field == DI_TIMESTAMP:
                _add_run(node, repeats, "timestamp", _read_packed(buf, f_start, pos, True))
            elif field == DI_CHANGESET:
                _add_run(node, repeats, "changeset", _read_packed(buf, f_start, pos, True))
            elif field == DI_UID:
                _add_run(node, repeats, "uid", _read_packed(buf, f_start, pos, True))
            elif field == DI_USER_SID:
                _add_run(node, repeats, "user_sid", _read_packed(buf, f_start, pos, True))
            elif field == DI_VISIBLE:
                _add_run(node, repeats, "visible", _read_packed(buf, f_start, pos, False))
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    _join_runs(node, repeats)
    for name in ("timestamp", "changeset", "uid", "user_sid"):
        node[name] = np.cumsum(node[name])


cdef dict _decode_dense_group(const unsigned char* buf, Py_ssize_t start,
                              Py_ssize_t end, bint keep_metadata):
    """Decode one ``DenseNodes`` group into a dict of arrays (ids/lats/lons are
    cumulative; ``keys_vals`` is the raw 0-delimited tag stream)."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    node = {"keys_vals": np.empty(0, dtype=np.int64)}
    repeats = {}
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
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == DENSE_ID:
                _add_run(node, repeats, "id", _read_packed(buf, f_start, pos, True))
            elif field == DENSE_LAT:
                _add_run(node, repeats, "lat", _read_packed(buf, f_start, pos, True))
            elif field == DENSE_LON:
                _add_run(node, repeats, "lon", _read_packed(buf, f_start, pos, True))
            elif field == DENSE_KEYS_VALS:
                _add_run(node, repeats, "keys_vals",
                         _read_packed(buf, f_start, pos, False))
            elif field == DENSE_DENSEINFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    _join_runs(node, repeats)
    for name in ("id", "lat", "lon"):
        node[name] = np.cumsum(node[name]) if name in node else np.empty(0, dtype=np.int64)
    if keep_metadata and info_start >= 0:
        _decode_dense_info(buf, info_start, info_end, node)
    n = len(node["id"])
    if len(node["lat"]) != n or len(node["lon"]) != n:
        _corrupt("a dense node group has %d ids but %d latitudes and %d longitudes"
                 % (n, len(node["lat"]), len(node["lon"])))
    _check_keys_vals(node["keys_vals"], n)
    if keep_metadata:
        for name in _META_COLS:
            if len(node[name]) not in (0, n):
                _corrupt("a dense node group has %d nodes but %d values of %s"
                         % (n, len(node[name]), name))
    return node


cdef _join_aligned(list groups, str name):
    """Concatenate the groups' ``name`` arrays. When only some groups have one (an
    untagged group's empty ``keys_vals``, an absent ``DenseInfo`` field), each of the
    others contributes a 0 per node, so every node keeps its own values."""
    parts = [g[name] for g in groups]
    if all(len(p) for p in parts) or not any(len(p) for p in parts):
        return np.concatenate(parts)
    return np.concatenate([
        p if len(p) else np.zeros(len(g["id"]), dtype=np.int64)
        for p, g in zip(parts, groups)
    ])


cdef dict _merge_dense(list groups, bint keep_metadata):
    """Concatenate the per-group node dicts into one. (Delta coding is per group, so
    each group's arrays are already cumulative before they are joined.)"""
    if len(groups) == 0:
        return None
    out = {
        "id": np.concatenate([g["id"] for g in groups]),
        "lat": np.concatenate([g["lat"] for g in groups]),
        "lon": np.concatenate([g["lon"] for g in groups]),
        "keys_vals": _join_aligned(groups, "keys_vals"),
    }
    if keep_metadata:
        for name in _META_COLS:
            out[name] = _join_aligned(groups, name)
    return out


# --- ways & relations (two-pass flat decode) ------------------------------------------

cdef class _ElementArrays:
    """Flat output buffers for one block's ways or relations, sized exactly by a
    counting pass and then filled in a single pass. ``members`` is way refs or relation
    memids; ``types``/``roles`` are relation-only. ``*_off`` are CSR offsets so element
    ``i`` owns ``members[members_off[i]:members_off[i + 1]]`` (and likewise its tags)."""
    cdef int64_t[::1] ids, keys, vals, tags_off, members, members_off, types, roles
    cdef int64_t[::1] version, timestamp, changeset, uid, user_sid, visible, info_fields
    cdef bint keep_metadata, is_relation, track_info
    cdef Py_ssize_t ei, ti, mi          # next element / tag / member write position
    cdef dict _arr                      # the owning numpy arrays, by name

    def __cinit__(self, Py_ssize_t n_elem, Py_ssize_t n_tags, Py_ssize_t n_members,
                  bint keep_metadata, bint is_relation, bint info_fields=False):
        self.keep_metadata = keep_metadata
        self.is_relation = is_relation
        self.track_info = keep_metadata and info_fields
        self.ei = 0
        self.ti = 0
        self.mi = 0
        self._arr = {
            "id": np.zeros(n_elem, np.int64),  # 0, the proto2 default, when absent
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
        if self.track_info:
            self._arr["info_fields"] = np.zeros(n_elem, np.int64)
            self.info_fields = self._arr["info_fields"]
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
        if self.track_info:
            out["info_fields"] = self._arr["info_fields"]
        return out


cdef int _fill_metadata(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        _ElementArrays b, Py_ssize_t i) except -1:
    """Decode an ``Info`` sub-message (or the proto2 default when absent) into ``b``'s
    per-element metadata arrays at index ``i``. Info fields are plain absolute values;
    when ``b`` tracks them, ``info_fields`` gets bit ``1 << (n - 1)`` for each field
    number ``n`` the Info holds."""
    cdef int64_t version = -1, timestamp = 0, changeset = 0
    cdef int64_t uid = 0, user_sid = 0, visible = 0, present = 0
    cdef Py_ssize_t pos
    cdef int field, wire
    if start >= 0:
        pos = start
        while pos < end:
            _read_tag(buf, &pos, end, &field, &wire)
            if wire == WIRE_VARINT:
                if field == INFO_VERSION:
                    version = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_VERSION - 1)
                elif field == INFO_TIMESTAMP:
                    timestamp = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_TIMESTAMP - 1)
                elif field == INFO_CHANGESET:
                    changeset = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_CHANGESET - 1)
                elif field == INFO_UID:
                    uid = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_UID - 1)
                elif field == INFO_USER_SID:
                    user_sid = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_USER_SID - 1)
                elif field == INFO_VISIBLE:
                    visible = <int64_t>_read_varint(buf, &pos, end)
                    present |= 1 << (INFO_VISIBLE - 1)
                else:
                    _read_varint(buf, &pos, end)
            else:
                _skip_field(buf, &pos, end, wire)
        _check_end(pos, end)
    b.version[i] = version
    b.timestamp[i] = timestamp
    b.changeset[i] = changeset
    b.uid[i] = uid
    b.user_sid[i] = user_sid
    b.visible[i] = visible
    if b.track_info:
        b.info_fields[i] = present
    return 0


cdef int _count_element(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        bint is_relation, Py_ssize_t* n_tags,
                        Py_ssize_t* n_members) except -1:
    """Add one way/relation's tag and member counts to the running totals, so the flat
    arrays can be sized exactly. Keys and vals share a count, and so do a relation's
    member ids, types and roles; raise when they differ."""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t keys = 0, vals = 0, members = 0, types = 0, roles = 0
    cdef int members_field = <int>REL_MEMIDS if is_relation else <int>WAY_REFS
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == WAY_KEYS:
                keys += _count_packed(buf, f_start, pos)
            elif field == WAY_VALS:
                vals += _count_packed(buf, f_start, pos)
            elif field == members_field:
                members += _count_packed(buf, f_start, pos)
            elif is_relation and field == REL_TYPES:
                types += _count_packed(buf, f_start, pos)
            elif is_relation and field == REL_ROLES_SID:
                roles += _count_packed(buf, f_start, pos)
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    if keys != vals:
        _corrupt("a %s has %d tag keys but %d values"
                 % ("relation" if is_relation else "way", keys, vals))
    if is_relation and (types != members or roles != members):
        _corrupt("a relation has %d member ids but %d member types and %d roles"
                 % (members, types, roles))
    n_tags[0] += keys
    n_members[0] += members
    return 0


cdef int _fill_way(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                   _ElementArrays b) except -1:
    """Decode one ``Way`` into the builder at its current write positions. A packed field
    may come in several runs, which follow each other."""
    cdef Py_ssize_t pos = start, f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t kpos = b.ti, vpos = b.ti, mpos = b.mi
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
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == WAY_KEYS:
                kpos += _filled(_fill_packed(buf, f_start, pos, False, b.keys, kpos))
            elif field == WAY_VALS:
                vpos += _filled(_fill_packed(buf, f_start, pos, False, b.vals, vpos))
            elif field == WAY_REFS:
                mpos += _filled(_fill_packed_delta(
                    buf, f_start, pos, b.members, mpos,
                    b.members[mpos - 1] if mpos > b.mi else 0))
            elif field == WAY_INFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    b.ti = kpos
    b.mi = mpos
    b.tags_off[b.ei + 1] = b.ti
    b.members_off[b.ei + 1] = b.mi
    if b.keep_metadata:
        _fill_metadata(buf, info_start, info_end, b, b.ei)
    b.ei += 1
    return 0


cdef int _fill_relation(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        _ElementArrays b) except -1:
    """Decode one ``Relation`` into the builder. Members carry an id (delta), a type and
    a role, all the same count, sharing the member offsets."""
    cdef Py_ssize_t pos = start, f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t kpos = b.ti, vpos = b.ti, mpos = b.mi, tpos = b.mi, rpos = b.mi
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
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == REL_KEYS:
                kpos += _filled(_fill_packed(buf, f_start, pos, False, b.keys, kpos))
            elif field == REL_VALS:
                vpos += _filled(_fill_packed(buf, f_start, pos, False, b.vals, vpos))
            elif field == REL_MEMIDS:
                mpos += _filled(_fill_packed_delta(
                    buf, f_start, pos, b.members, mpos,
                    b.members[mpos - 1] if mpos > b.mi else 0))
            elif field == REL_TYPES:
                tpos += _filled(_fill_packed(buf, f_start, pos, False, b.types, tpos))
            elif field == REL_ROLES_SID:
                rpos += _filled(_fill_packed(buf, f_start, pos, False, b.roles, rpos))
            elif field == REL_INFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    b.ti = kpos
    b.mi = mpos
    b.tags_off[b.ei + 1] = b.ti
    b.members_off[b.ei + 1] = b.mi
    if b.keep_metadata:
        _fill_metadata(buf, info_start, info_end, b, b.ei)
    b.ei += 1
    return 0


cdef dict _decode_elements(const unsigned char* buf, list group_ranges,
                           int element_field, bint keep_metadata, bint is_relation):
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
                pos = _field_end(pos, length, ge)
                if pos > ge:
                    break
                if field == element_field:
                    n_elem += 1
                    _count_element(buf, f_start, pos, is_relation, &n_tags, &n_members)
            else:
                _skip_field(buf, &pos, ge, wire)
        _check_end(pos, ge)
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
                pos = _field_end(pos, length, ge)
                if pos > ge:
                    break
                if field == element_field:
                    if is_relation:
                        _fill_relation(buf, f_start, pos, b)
                    else:
                        _fill_way(buf, f_start, pos, b)
            else:
                _skip_field(buf, &pos, ge, wire)
        _check_end(pos, ge)
    return b.result()


cdef int _count_node_tags(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                          Py_ssize_t* n_keys, Py_ssize_t* n_vals) except -1:
    """Count one plain ``Node``'s key ids and value ids, whether packed (in one run or
    several) or not."""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == NODE_KEYS:
                n_keys[0] += _count_packed(buf, f_start, pos)
            elif field == NODE_VALS:
                n_vals[0] += _count_packed(buf, f_start, pos)
        elif wire == WIRE_VARINT:
            _read_varint(buf, &pos, end)
            if field == NODE_KEYS:
                n_keys[0] += 1
            elif field == NODE_VALS:
                n_vals[0] += 1
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    return 0


cdef int _fill_plain_node(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                          _ElementArrays b, int64_t[::1] lat,
                          int64_t[::1] lon) except -1:
    """Decode one plain ``Node`` into the builder (its tags as ``keys``/``vals``) and its
    coordinates into ``lat``/``lon``. Its id and coordinates are absolute, not delta
    coded; key and value ids may come packed (in one run or several) or not."""
    cdef Py_ssize_t pos = start, f_start
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t kpos = b.ti, vpos = b.ti
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            if field == NODE_ID:
                b.ids[b.ei] = _zig_zag_decode(_read_varint(buf, &pos, end))
            elif field == NODE_LAT:
                lat[b.ei] = _zig_zag_decode(_read_varint(buf, &pos, end))
            elif field == NODE_LON:
                lon[b.ei] = _zig_zag_decode(_read_varint(buf, &pos, end))
            elif field == NODE_KEYS:
                b.keys[kpos] = <int64_t>_read_varint(buf, &pos, end)
                kpos += 1
            elif field == NODE_VALS:
                b.vals[vpos] = <int64_t>_read_varint(buf, &pos, end)
                vpos += 1
            else:
                _read_varint(buf, &pos, end)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == NODE_KEYS:
                kpos += _filled(_fill_packed(buf, f_start, pos, False, b.keys, kpos))
            elif field == NODE_VALS:
                vpos += _filled(_fill_packed(buf, f_start, pos, False, b.vals, vpos))
            elif field == NODE_INFO:
                info_start = f_start
                info_end = pos
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    b.ti = kpos
    b.tags_off[b.ei + 1] = b.ti
    b.members_off[b.ei + 1] = 0
    if b.keep_metadata:
        _fill_metadata(buf, info_start, info_end, b, b.ei)
    b.ei += 1
    return 0


cdef dict _decode_plain_nodes(const unsigned char* buf, list spans, bint keep_metadata):
    """Decode a group's plain ``Node`` messages (the non-dense encoding) into the dict a
    ``DenseNodes`` group gives: absolute ``id``/``lat``/``lon``, ``keys_vals`` as each
    node's key and value ids followed by a 0 (empty when no node has tags), and the
    metadata from each node's ``Info`` (the proto2 defaults when it has none)."""
    cdef Py_ssize_t n = len(spans), n_tags = 0, n_keys, n_vals
    cdef Py_ssize_t start, end
    for start, end in spans:
        n_keys = 0
        n_vals = 0
        _count_node_tags(buf, start, end, &n_keys, &n_vals)
        # Each key needs its value; the fill pass relies on it to stay in its arrays.
        if n_keys != n_vals:
            raise InvalidOSMFileError(
                "A node in the PBF has %d tag keys but %d values." % (n_keys, n_vals))
        n_tags += n_keys
    cdef _ElementArrays b = _ElementArrays(n, n_tags, 0, keep_metadata, False)
    lat = np.zeros(n, dtype=np.int64)  # 0, the proto2 default, when absent
    lon = np.zeros(n, dtype=np.int64)
    cdef int64_t[::1] lat_view = lat
    cdef int64_t[::1] lon_view = lon
    for start, end in spans:
        _fill_plain_node(buf, start, end, b, lat_view, lon_view)
    arrays = b.result()
    node = {"id": arrays["id"], "lat": lat, "lon": lon,
            "keys_vals": np.empty(0, dtype=np.int64)}
    if n_tags > 0:
        # Node i's pairs start at 2 * tags_off[i] + i (each earlier node adds its pairs
        # and a 0), so its tag j goes to position 2 * j + i.
        tag_node = np.repeat(np.arange(n), np.diff(arrays["tags_off"]))
        at = 2 * np.arange(n_tags) + tag_node
        keys_vals = np.zeros(2 * n_tags + n, dtype=np.int64)
        keys_vals[at] = arrays["keys"]
        keys_vals[at + 1] = arrays["vals"]
        node["keys_vals"] = keys_vals
    if keep_metadata:
        for name in _META_COLS:
            node[name] = arrays[name]
    return node


cdef int _collect_nodes(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        bint keep_metadata, list node_groups) except -1:
    """Decode the nodes of one group into ``node_groups``: each ``DenseNodes`` block,
    and the group's plain ``Node`` messages as one more entry of the same form."""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t length
    plain = []
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == GROUP_DENSE:
                node_groups.append(
                    _decode_dense_group(buf, f_start, pos, keep_metadata))
            elif field == GROUP_NODES:
                plain.append((f_start, pos))
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    if plain:
        node_groups.append(_decode_plain_nodes(buf, plain, keep_metadata))
    return 0


cdef dict _default_header():
    """The block header values a block that leaves them out has (the proto2 defaults)."""
    return {"granularity": 100, "date_granularity": 1000, "lat_offset": 0, "lon_offset": 0}


cdef int _read_header_value(const unsigned char* buf, Py_ssize_t* pos, Py_ssize_t end,
                            int field, dict header) except -1:
    """Read the varint of block field ``field`` into ``header`` when it is one of the
    header's values, or skip it."""
    cdef int64_t value = <int64_t>_read_varint(buf, pos, end)
    if field == PB_GRANULARITY:
        header["granularity"] = value
    elif field == PB_DATE_GRANULARITY:
        header["date_granularity"] = value
    elif field == PB_LAT_OFFSET:
        header["lat_offset"] = value
    elif field == PB_LON_OFFSET:
        header["lon_offset"] = value
    return 0


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
        return [], _default_header(), None, None, None
    cdef const unsigned char* buf = &data[0]
    cdef Py_ssize_t pos = 0
    cdef int field, wire
    cdef uint64_t length
    cdef Py_ssize_t f_start

    string_table = []
    header = _default_header()
    node_groups = []
    group_ranges = []

    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            _read_header_value(buf, &pos, end, field, header)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == PB_STRINGTABLE:
                string_table = _decode_string_table(buf, f_start, pos)
            elif field == PB_PRIMITIVEGROUP:
                group_ranges.append((f_start, pos))
                _collect_nodes(buf, f_start, pos, keep_metadata, node_groups)
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)

    nodes = _merge_dense(node_groups, keep_metadata)
    ways = _decode_elements(buf, group_ranges, GROUP_WAYS, keep_metadata, False)
    relations = _decode_elements(buf, group_ranges, GROUP_RELATIONS, keep_metadata, True)
    return string_table, header, nodes, ways, relations


cdef int _check_fields(const unsigned char* buf, Py_ssize_t start,
                       Py_ssize_t end) except -1:
    """Raise unless ``buf[start:end]`` is a run of well-formed fields."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        _skip_field(buf, &pos, end, wire)
    return _check_end(pos, end)


cdef int64_t _element_id(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         bint is_signed) except? -1:
    """The id of the plain node, way or relation stored in ``buf[start:end]``, zig-zag
    decoded when ``is_signed`` (a node's): the last one, as protobuf and the decoder read
    it, or 0, the proto2 default, when it has none. Raises unless the element and its
    ``Info`` are well-formed fields; the three number the id and ``Info`` alike."""
    cdef Py_ssize_t pos = start, f_start
    cdef int field, wire
    cdef uint64_t value = 0, length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if field == NODE_ID and wire == WIRE_VARINT:
            value = _read_varint(buf, &pos, end)
        elif field == NODE_INFO and wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            _check_fields(buf, f_start, pos)
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    return _zig_zag_decode(value) if is_signed else <int64_t>value


cdef inline bint _is_element(int field, int wire) nogil:
    """Whether a group field holds a plain node, a way or a relation."""
    return wire == WIRE_LENGTH and (
        field == GROUP_NODES or field == GROUP_WAYS or field == GROUP_RELATIONS)


cdef dict _group_layout(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end):
    """The parts of one ``PrimitiveGroup``, for ``element_layout``."""
    cdef Py_ssize_t pos = start, f_start, tag_start, n = 0, i = 0
    cdef int field, wire
    cdef uint64_t length
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        _skip_field(buf, &pos, end, wire)
        n += _is_element(field, wire)
    _check_end(pos, end)

    group = {"dense": None}
    for name in ("kind", "id", "start", "end"):
        group[name] = np.empty(n, dtype=np.int64)
    cdef int64_t[::1] kinds = group["kind"], ids = group["id"]
    cdef int64_t[::1] starts = group["start"], ends = group["end"]
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
        if pos > end:
            break
        if field == GROUP_DENSE:
            group["dense"] = _decode_dense_group(buf, f_start, pos, True)
        elif _is_element(field, wire):
            kinds[i] = field
            ids[i] = _element_id(buf, f_start, pos, field == GROUP_NODES)
            starts[i] = tag_start
            ends[i] = pos
            i += 1
    return group


def element_layout(const unsigned char[::1] data):
    """Where the parts of a decompressed ``PrimitiveBlock`` lie, for copying its elements
    without decoding them: ``(header, string_table, groups)``.

    ``header`` is as in ``decode_primitive_block``; ``string_table`` is the (start, end)
    of the string table's bytes, or None when the block has none. Each group is a dict:
    ``dense`` is the dict ``decode_primitive_block`` gives for its ``DenseNodes`` (with
    metadata), or None, and ``kind``, ``id``, ``start`` and ``end`` hold, for each plain
    node, way and relation in it, its group field number, its id, and where its field
    (tag and length included) starts and ends in ``data``.
    """
    cdef Py_ssize_t end = data.shape[0]
    header = _default_header()
    if end == 0:
        return header, None, []
    cdef const unsigned char* buf = &data[0]
    cdef Py_ssize_t pos = 0, f_start
    cdef int field, wire
    cdef uint64_t length
    string_table = None
    groups = []
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            _read_header_value(buf, &pos, end, field, header)
        elif wire == WIRE_LENGTH:
            length = _read_varint(buf, &pos, end)
            f_start = pos
            pos = _field_end(pos, length, end)
            if pos > end:
                break
            if field == PB_STRINGTABLE:
                string_table = (f_start, pos)
            elif field == PB_PRIMITIVEGROUP:
                groups.append(_group_layout(buf, f_start, pos))
        else:
            _skip_field(buf, &pos, end, wire)
    _check_end(pos, end)
    return header, string_table, groups


def decode_elements(const unsigned char[::1] data, starts, ends, bint is_relation):
    """Decode the ways (or, with ``is_relation``, the relations) whose fields (tag and
    length included) lie at ``data[starts[i]:ends[i]]``, as ``element_layout`` gives
    them, into the dict ``decode_primitive_block`` gives, with metadata and
    ``info_fields``: a bit ``1 << (n - 1)`` for each ``Info`` field number ``n`` an
    element holds."""
    cdef Py_ssize_t n = len(starts), n_tags = 0, n_members = 0, i
    cdef Py_ssize_t pos, end, payload, size = data.shape[0]
    cdef int field, wire
    cdef uint64_t length
    cdef const unsigned char* buf = &data[0] if size > 0 else NULL
    spans = []
    for i in range(n):
        pos = starts[i]
        end = ends[i]
        if pos < 0 or pos > end or end > size:
            _corrupt("an element does not fit in the data that holds it")
        _read_tag(buf, &pos, end, &field, &wire)
        length = _read_varint(buf, &pos, end)
        payload = _field_end(pos, length, end)
        if wire != WIRE_LENGTH or payload != end:
            _corrupt("an element does not fit in the data that holds it")
        spans.append((pos, end))
        _count_element(buf, pos, end, is_relation, &n_tags, &n_members)
    cdef _ElementArrays b = _ElementArrays(n, n_tags, n_members, True, is_relation, True)
    for pos, end in spans:
        if is_relation:
            _fill_relation(buf, pos, end, b)
        else:
            _fill_way(buf, pos, end, b)
    return b.result()
