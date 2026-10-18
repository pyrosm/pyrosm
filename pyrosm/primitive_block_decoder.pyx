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
easy to compare against the protobuf path in tests. ``decode_batch_nogil`` (the GIL-free
batch decoder at the end of this module) gives coordinates in degrees.
"""
import numpy as np

from libc.stdint cimport int64_t, uint64_t

from pyrosm.exceptions import InvalidOSMFileError


# The per-element metadata columns (from ``Info`` / ``DenseInfo``), in output order.
_META_COLS = ("version", "timestamp", "changeset", "uid", "user_sid", "visible")


# --- low-level wire-format primitives (the readers are in the .pxd) -------------------

cdef int _corrupt(str reason) except -1:
    """Raise the error for a data block that does not follow the PBF schema."""
    raise InvalidOSMFileError("A data block of the PBF is corrupt: %s." % reason)


cdef int _check_end(Py_ssize_t pos, Py_ssize_t end) except -1:
    """Raise unless a loop over data that ends at ``end`` stopped exactly there."""
    if pos != end:
        _corrupt("a field or a number does not fit in the data that holds it")
    return 0


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
    cdef bint at_key = True, ended = False
    if keys_vals.shape[0] == 0:
        return 0
    for i in range(keys_vals.shape[0]):
        ended = at_key and keys_vals[i] == 0
        if ended:
            ends += 1
        else:
            at_key = not at_key
    if ends != n or not ended:
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


# --- GIL-free batch decoding (the threaded reader) -------------------------------------
#
# ``decode_batch_nogil`` decodes a batch of blocks without the GIL into flat arrays: every
# dense node's id and coordinates, the node features (dense nodes carrying a wanted key or a
# wanted id), all ways' ids and refs, and the ways carrying a wanted key with their tags and
# metadata. A block it does not handle (plain nodes, relations, a packed field in more runs
# than it keeps, dense groups whose tags or metadata only some groups have, several string
# tables, data that does not follow the format) gets status 1 and none of its elements; the
# caller decodes it with ``decode_primitive_block``.
#
# C code cannot grow numpy arrays without the GIL, so every block is decoded twice by the
# same functions (``_decode_block_nogil`` and the ones it calls). In the count pass
# ``_Out.fill`` is False: the functions only add to the ``n_*`` counters, and a block
# that fails is undone. The caller then allocates arrays of exactly those sizes, and in
# the fill pass (``_Out.fill`` True) the same functions write the values, using the
# counters as write positions. Both passes see the same bytes, so they take the same
# steps; a block that decodes in the first pass but not the second (status 2) is a bug
# and raises.
#
# A variable-length list per element (a way's refs, its tags) is stored CSR style: one flat
# value array plus an offsets array with one entry more than there are elements, so element
# ``i`` owns ``values[off[i]:off[i + 1]]``.
#
# Tags are string ids of their block's string table, so a block's ids mean nothing outside
# it.

from libc.string cimport memcmp, memset
from libc.stdint cimport uint8_t, uintptr_t

from pyrosm.int64_hash cimport Int64Set, _set_has, kh_i64set_t

# The columns of the per-block count table ``decode_batch_nogil`` returns.
BLOCK_COUNTS = (
    "status", "nodes", "features", "pairs", "all_ways", "all_refs", "ways", "way_refs",
    "way_tags",
)

cdef enum:
    BC_STATUS = 0
    BC_NODES = 1
    BC_FEATURES = 2
    BC_PAIRS = 3
    BC_ALL_WAYS = 4
    BC_ALL_REFS = 5
    BC_WAYS = 6
    BC_WAY_REFS = 7
    BC_WAY_TAGS = 8
    BC_COUNT = 9


# What the caller asks for, as plain C values so the decoder can read it without the GIL.
cdef struct _Request:
    const unsigned char* keys      # the wanted keys, concatenated
    const int64_t* keys_off        # key ``k`` is keys[keys_off[k]:keys_off[k + 1]]
    Py_ssize_t n_keys
    bint key_filter                # False: any tagged element is wanted
    int64_t* key_ids               # scratch: the wanted keys' string ids in a block
    uint8_t* found                 # scratch: whether key ``k`` was found in a block
    bint features                  # give the node features
    bint all_ways                  # give every way's id and refs
    bint bbox                      # flag each node inside xmin..xmax, ymin..ymax
    double xmin
    double ymin
    double xmax
    double ymax
    const kh_i64set_t* node_set    # the ids that make a dense node a feature, or NULL


# The decoder's output. The counters are the sizes in the count pass and the next write
# positions in the fill pass; the pointers are NULL in the count pass. ``*_off`` arrays are
# CSR offsets (see the section comment); their entry 0 is 0 from the allocation.
cdef struct _Out:
    bint fill                      # False: count pass, True: fill pass
    Py_ssize_t n_nodes, n_features, n_pairs, n_all_ways, n_all_refs
    Py_ssize_t n_ways, n_way_refs, n_way_tags
    Py_ssize_t max_group           # nodes in the largest dense group, to size ``scratch``
    int64_t* node_id               # every dense node, in block order
    double* node_lon
    double* node_lat
    uint8_t* in_box
    int64_t* feature_row           # a node feature's row in node_id
    int64_t* feature_pairs_off     # a node feature's tags: feature_keys/vals[off[i]:...]
    int64_t* feature_keys
    int64_t* feature_vals
    int64_t* feature_meta          # version, timestamp, changeset, visible per feature
    int64_t* all_id                # every way (with all_ways), its refs in all_refs
    int64_t* all_refs_off
    int64_t* all_refs
    int64_t* way_id                # the ways with a wanted key, their refs and tags
    int64_t* way_refs_off
    int64_t* way_refs
    int64_t* way_tags_off
    int64_t* way_keys
    int64_t* way_vals
    int64_t* way_meta              # version, timestamp, visible per way
    int64_t* scratch               # 4 values per node of the current dense group


# Which fields the block's dense groups have, so ``_decode_dense_group_nogil`` can check
# that they all have the same ones.
cdef struct _DenseState:
    bint seen                      # a dense group of the block was decoded
    bint has_tags
    int info_mask                  # bit f - 1 for each DenseInfo field f the groups have


cdef Py_ssize_t _runs_count(const unsigned char* buf, _Runs* runs) noexcept nogil:
    """The number of varints in the runs; -1 when one is cut off or larger than 64 bits
    (as ``_read_varint`` reads them)."""
    # A varint stores 7 bits per byte, low bits first; a byte with its high bit (0x80) set
    # is followed by another. So each byte without the high bit ends one value. 64 bits
    # need at most 10 bytes, and the 10th holds only the top bit, so it must be 0 or 1.
    # ``continued`` counts the bytes of the current value read so far.
    cdef Py_ssize_t r, pos, n = 0, continued
    for r in range(runs.n):
        continued = 0
        for pos in range(runs.start[r], runs.end[r]):
            if buf[pos] & 0x80:
                continued += 1
                if continued > 9:
                    return -1
            else:
                if continued == 9 and buf[pos] > 1:
                    return -1
                n += 1
                continued = 0
        if continued:
            return -1
    return n


cdef inline int64_t _wrapping_add(int64_t a, int64_t b) noexcept nogil:
    """``a + b`` wrapping around as numpy's int64 arithmetic does (a signed overflow is
    undefined in C)."""
    return <int64_t>(<uint64_t>a + <uint64_t>b)


cdef inline double _degrees(int64_t value, int64_t granularity, int64_t offset) noexcept nogil:
    """A coordinate in degrees: ``(value * granularity + offset) / 1e9`` with the int64
    arithmetic of numpy, which wraps around."""
    return (<double><int64_t>(<uint64_t>value * <uint64_t>granularity
                              + <uint64_t>offset)) / 1e9


cdef Py_ssize_t _runs_fill(const unsigned char* buf, _Runs* runs, bint delta, bint signed,
                           int64_t* out) noexcept nogil:
    """Decode the runs into ``out``: values, or a running sum of zig-zag deltas that goes on
    across the runs. Returns how many were written, or -1 for a cut-off varint."""
    cdef Py_ssize_t r, pos, end, i = 0
    cdef int64_t running = 0
    cdef uint64_t raw
    for r in range(runs.n):
        pos = runs.start[r]
        end = runs.end[r]
        while pos < end:
            raw = _read_varint(buf, &pos, end)
            if delta:
                running = _wrapping_add(running, _zig_zag_decode(raw))
                out[i] = running
            elif signed:
                out[i] = _zig_zag_decode(raw)
            else:
                out[i] = <int64_t>raw
            i += 1
        if pos != end:
            return -1
    return i


cdef inline bint _is_wanted_key(int64_t key, const int64_t* key_ids,
                                Py_ssize_t n_ids) noexcept nogil:
    """Whether ``key`` is one of the ``n_ids`` ids (a linear scan: there are few)."""
    cdef Py_ssize_t j
    for j in range(n_ids):
        if key == key_ids[j]:
            return True
    return False


cdef Py_ssize_t _key_ids(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                         const _Request* req, int64_t* key_ids,
                         Py_ssize_t* n_table) noexcept nogil:
    """The string ids of the wanted keys in the string table ``buf[start:end]``, the first
    occurrence of each (as ``list.index`` gives it), and the table's size in ``n_table``.
    -1 when the table is corrupt."""
    cdef Py_ssize_t pos = start, s_start, k, klen, n_ids = 0
    cdef int field, wire
    cdef uint64_t length
    cdef int64_t sid = 0
    cdef uint8_t* found = req.found
    for k in range(req.n_keys):
        found[k] = False
    # The table is a list of ``ST_S`` fields; ``sid`` is the index of the current one.
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        s_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return -1
        if field != ST_S:
            continue
        for k in range(req.n_keys):
            klen = req.keys_off[k + 1] - req.keys_off[k]
            if (not found[k] and klen == pos - s_start
                    and memcmp(buf + s_start, req.keys + req.keys_off[k], klen) == 0):
                found[k] = True
                key_ids[n_ids] = sid
                n_ids += 1
        sid += 1
    if pos != end:
        return -1
    n_table[0] = sid
    return n_ids


cdef int _tag_stream_ok(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                        Py_ssize_t n) noexcept nogil:
    """Whether the dense tag stream holds, for each of ``n`` nodes in turn, its key and value
    ids followed by a 0 (as ``_check_keys_vals`` requires)."""
    # The stream is ``k v k v ... 0`` per node. A 0 where a key is due ends a node; any
    # other value flips between expecting a key and expecting a value.
    cdef Py_ssize_t pos = start, ends = 0
    cdef bint at_key = True, ended = False
    cdef int64_t value
    while pos < end:
        value = <int64_t>_read_varint(buf, &pos, end)
        ended = at_key and value == 0
        if ended:
            ends += 1
        else:
            at_key = not at_key
    return pos == end and ends == n and ended


# A cursor over a delta-coded packed field, for reading its values one at a time alongside
# another stream (the node ids next to the tag stream in
# ``_decode_node_features_nogil``).
cdef struct _Deltas:
    const _Runs* runs              # a delta-coded packed field, read value after value
    Py_ssize_t run
    Py_ssize_t pos
    int64_t value


cdef inline void _deltas_start(_Deltas* d, const _Runs* runs) noexcept nogil:
    d.runs = runs
    d.run = 0
    d.pos = runs.start[0] if runs.n else 0
    d.value = 0


cdef inline int64_t _next_delta(const unsigned char* buf, _Deltas* d) noexcept nogil:
    """The next value of the field (whose values ``_runs_count`` has counted)."""
    while d.pos >= d.runs.end[d.run] and d.run + 1 < d.runs.n:
        d.run += 1
        d.pos = d.runs.start[d.run]
    d.value = _wrapping_add(
        d.value, _zig_zag_decode(_read_varint(buf, &d.pos, d.runs.end[d.run]))
    )
    return d.value


cdef int _decode_node_features_nogil(const unsigned char* buf, Py_ssize_t start,
                                     Py_ssize_t end, _Runs* info, const _Request* req,
                                     const int64_t* key_ids, Py_ssize_t n_ids,
                                     Py_ssize_t n_table, _Out* out, Py_ssize_t row0,
                                     Py_ssize_t n, const _Runs* ids) noexcept nogil:
    """Count, or write, the node features of a dense group: the nodes whose tags (the
    stream ``buf[start:end]``, empty when the group has none) carry a wanted key, or
    whose id (``ids``) is in ``req.node_set``, with their tag pairs and DenseInfo
    metadata (0 where the group has no such field; timestamp and changeset are delta
    coded). 1 when a pair's string id is outside the block's table of ``n_table``
    strings."""
    cdef int64_t* columns[4]
    cdef int fields[4]
    cdef Py_ssize_t c, i, pos = start, node_start, node = 0
    cdef int64_t key, value
    cdef bint matched
    cdef _Deltas node_ids
    if req.node_set != NULL:
        _deltas_start(&node_ids, ids)
    if out.fill:
        # The group's version, timestamp, changeset and visible columns, decoded into
        # scratch[c * n:(c + 1) * n] so a feature's metadata can be picked by its node.
        fields[0] = DI_VERSION
        fields[1] = DI_TIMESTAMP
        fields[2] = DI_CHANGESET
        fields[3] = DI_VISIBLE
        for c in range(4):
            columns[c] = out.scratch + c * n
            if info[fields[c] - 1].n:
                if _runs_fill(buf, &info[fields[c] - 1], c == 1 or c == 2, c == 1 or c == 2,
                              columns[c]) != n:
                    return 1
            else:
                for i in range(n):
                    columns[c][i] = 0
    if start >= end:
        # A group without tags: its features are the nodes with a wanted id.
        for node in range(n if req.node_set != NULL else 0):
            if _set_has(req.node_set, _next_delta(buf, &node_ids)):
                if out.fill:
                    out.feature_row[out.n_features] = row0 + node
                    for c in range(4):
                        out.feature_meta[4 * out.n_features + c] = columns[c][node]
                out.n_features += 1
                if out.fill:
                    out.feature_pairs_off[out.n_features] = out.n_pairs
        return 0
    while pos < end:
        # One node: check its keys, then copy its pairs when it is a feature.
        node_start = pos
        matched = req.node_set != NULL and _set_has(
            req.node_set, _next_delta(buf, &node_ids)
        )
        key = <int64_t>_read_varint(buf, &pos, end)
        while key != 0:
            if not req.key_filter or _is_wanted_key(key, key_ids, n_ids):
                matched = True
            _read_varint(buf, &pos, end)
            key = <int64_t>_read_varint(buf, &pos, end)
        if matched:
            if out.fill:
                out.feature_row[out.n_features] = row0 + node
                for c in range(4):
                    out.feature_meta[4 * out.n_features + c] = columns[c][node]
            pos = node_start
            key = <int64_t>_read_varint(buf, &pos, end)
            while key != 0:
                value = <int64_t>_read_varint(buf, &pos, end)
                if key < 0 or key >= n_table or value < 0 or value >= n_table:
                    return 1
                if out.fill:
                    out.feature_keys[out.n_pairs] = key
                    out.feature_vals[out.n_pairs] = value
                out.n_pairs += 1
                key = <int64_t>_read_varint(buf, &pos, end)
            out.n_features += 1
            if out.fill:
                out.feature_pairs_off[out.n_features] = out.n_pairs
        node += 1
    return 0


cdef int _decode_dense_group_nogil(const unsigned char* buf, Py_ssize_t start,
                                   Py_ssize_t end, const _Request* req, _Out* out,
                                   int64_t granularity, int64_t lat_offset,
                                   int64_t lon_offset, const int64_t* key_ids,
                                   Py_ssize_t n_ids, Py_ssize_t n_table,
                                   _DenseState* state) noexcept nogil:
    """Count or decode one ``DenseNodes`` group. 1 when the block needs the Python path."""
    # A DenseNodes group stores its nodes column by column: packed fields of ids, lats and
    # lons (each delta coded), one tag stream for all nodes, and a DenseInfo message with
    # one packed field per metadata column. The steps: find each field's runs, check that
    # every column has one value per node, then (fill pass) decode the columns.
    cdef _Runs ids, lats, lons, kv
    cdef _Runs info[6]
    cdef Py_ssize_t pos = start, f_start, i_pos, i_start, n, i, row0
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef int field, wire, f, info_mask = 0
    cdef uint64_t length
    cdef double lon, lat
    ids.n = 0
    lats.n = 0
    lons.n = 0
    kv.n = 0
    for f in range(6):
        info[f].n = 0
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
            if _add_span(&ids, f_start, pos) < 0:
                return 1
        elif field == DENSE_LAT:
            if _add_span(&lats, f_start, pos) < 0:
                return 1
        elif field == DENSE_LON:
            if _add_span(&lons, f_start, pos) < 0:
                return 1
        elif field == DENSE_KEYS_VALS:
            if pos > f_start and _add_span(&kv, f_start, pos) < 0:
                return 1
        elif field == DENSE_DENSEINFO:
            if info_start >= 0:
                return 1
            info_start = f_start
            info_end = pos
    if pos != end or kv.n > 1:
        return 1
    # DenseInfo's fields 1 to 6 (DI_VERSION ... DI_VISIBLE) go to info[0] ... info[5].
    if info_start >= 0:
        i_pos = info_start
        while i_pos < info_end:
            _read_tag(buf, &i_pos, info_end, &field, &wire)
            if wire != WIRE_LENGTH:
                _skip_field(buf, &i_pos, info_end, wire)
                continue
            length = _read_varint(buf, &i_pos, info_end)
            i_start = i_pos
            i_pos = _field_end(i_pos, length, info_end)
            if i_pos > info_end:
                return 1
            if 1 <= field <= 6 and i_pos > i_start:
                if _add_span(&info[field - 1], i_start, i_pos) < 0:
                    return 1
        if i_pos != info_end:
            return 1
    # Every column holds one value per node: ``n`` nodes.
    n = _runs_count(buf, &ids)
    if n < 0 or _runs_count(buf, &lats) != n or _runs_count(buf, &lons) != n:
        return 1
    for f in range(6):
        if info[f].n:
            if _runs_count(buf, &info[f]) != n:
                return 1
            info_mask |= 1 << f
    # The Python path fills a column only some of a block's groups have with zeros for the
    # others (``_join_aligned``). Such blocks are rare and go there, so all groups of a
    # block decoded here must have the same columns -- of the ones pyrosm reads.
    info_mask &= ((1 << (DI_VERSION - 1)) | (1 << (DI_TIMESTAMP - 1))
                  | (1 << (DI_CHANGESET - 1)) | (1 << (DI_VISIBLE - 1)))
    if state.seen and (state.has_tags != (kv.n > 0) or state.info_mask != info_mask):
        return 1
    state.seen = True
    state.has_tags = kv.n > 0
    state.info_mask = info_mask

    row0 = out.n_nodes
    if not out.fill:
        if kv.n and not _tag_stream_ok(buf, kv.start[0], kv.end[0], n):
            return 1
        if n > out.max_group:
            out.max_group = n
    else:
        if _runs_fill(buf, &ids, True, True, out.node_id + row0) != n:
            return 1
        # lat/lon go through the scratch space: running sums, then scaled to degrees
        # (nanodegrees = offset + granularity * value).
        if _runs_fill(buf, &lats, True, True, out.scratch) != n:
            return 1
        if _runs_fill(buf, &lons, True, True, out.scratch + n) != n:
            return 1
        for i in range(n):
            lat = _degrees(out.scratch[i], granularity, lat_offset)
            lon = _degrees(out.scratch[n + i], granularity, lon_offset)
            out.node_lat[row0 + i] = lat
            out.node_lon[row0 + i] = lon
            if req.bbox:
                out.in_box[row0 + i] = (lon >= req.xmin and lon <= req.xmax
                                        and lat >= req.ymin and lat <= req.ymax)
    out.n_nodes += n
    # Node features: tagged nodes with a wanted key, or with a wanted id (untagged too).
    if req.features and kv.n:
        return _decode_node_features_nogil(buf, kv.start[0], kv.end[0], info, req,
                                           key_ids, n_ids, n_table, out, row0, n, &ids)
    if req.features and req.node_set != NULL:
        return _decode_node_features_nogil(buf, 0, 0, info, req, key_ids, n_ids,
                                           n_table, out, row0, n, &ids)
    return 0


cdef bint _ids_in_table(const unsigned char* buf, const _Runs* runs,
                        Py_ssize_t n_table) noexcept nogil:
    """Whether every value of the runs (counted by ``_runs_count``) is a string id of a
    table of ``n_table`` strings."""
    cdef Py_ssize_t r, pos
    for r in range(runs.n):
        pos = runs.start[r]
        while pos < runs.end[r]:
            if _read_varint(buf, &pos, runs.end[r]) >= <uint64_t>n_table:
                return False
    return True


cdef int _decode_way_nogil(const unsigned char* buf, Py_ssize_t start, Py_ssize_t end,
                           const _Request* req, _Out* out, const int64_t* key_ids,
                           Py_ssize_t n_ids, Py_ssize_t n_table) noexcept nogil:
    """Count or decode one ``Way``. 1 when the block needs the Python path."""
    # The steps: find the way's fields (id, the packed keys, vals and refs, and Info), read
    # its Info, decide whether it is wanted, then count (count pass) or write (fill pass).
    cdef _Runs keys, vals, refs
    cdef Py_ssize_t pos = start, f_start, n_keys, n_refs, k
    cdef Py_ssize_t info_start = -1, info_end = -1
    cdef Py_ssize_t i_pos
    cdef int field, wire
    cdef uint64_t length, value
    cdef int64_t way_id = 0, version = -1, timestamp = 0, visible = 0
    cdef bint matched
    keys.n = 0
    vals.n = 0
    refs.n = 0
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            value = _read_varint(buf, &pos, end)
            if field == WAY_ID:
                way_id = <int64_t>value
            continue
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return 1
        if field == WAY_KEYS:
            if _add_span(&keys, f_start, pos) < 0:
                return 1
        elif field == WAY_VALS:
            if _add_span(&vals, f_start, pos) < 0:
                return 1
        elif field == WAY_REFS:
            if _add_span(&refs, f_start, pos) < 0:
                return 1
        elif field == WAY_INFO:
            if info_start >= 0:
                return 1
            info_start = f_start
            info_end = pos
    if pos != end:
        return 1
    n_keys = _runs_count(buf, &keys)
    if n_keys < 0 or _runs_count(buf, &vals) != n_keys:
        return 1
    n_refs = _runs_count(buf, &refs)
    if n_refs < 0:
        return 1
    if info_start >= 0:
        i_pos = info_start
        while i_pos < info_end:
            _read_tag(buf, &i_pos, info_end, &field, &wire)
            if wire == WIRE_VARINT:
                value = _read_varint(buf, &i_pos, info_end)
                if field == INFO_VERSION:
                    version = <int64_t>value
                elif field == INFO_TIMESTAMP:
                    timestamp = <int64_t>value
                elif field == INFO_VISIBLE:
                    visible = <int64_t>value
            else:
                _skip_field(buf, &i_pos, info_end, wire)
        if i_pos != info_end:
            return 1

    # Whether the way carries a wanted key (any tag without a key filter).
    matched = n_keys > 0
    if req.key_filter:
        matched = False
        for k in range(keys.n):
            i_pos = keys.start[k]
            while i_pos < keys.end[k]:
                if _is_wanted_key(<int64_t>_read_varint(buf, &i_pos, keys.end[k]),
                                  key_ids, n_ids):
                    matched = True
            if i_pos != keys.end[k]:
                return 1

    if not out.fill:
        # Count pass: only the sizes, and the check that a wanted way's tags are string
        # ids of the block's table.
        if req.all_ways:
            out.n_all_ways += 1
            out.n_all_refs += n_refs
        if matched:
            if (not _ids_in_table(buf, &keys, n_table)
                    or not _ids_in_table(buf, &vals, n_table)):
                return 1
            out.n_ways += 1
            out.n_way_refs += n_refs
            out.n_way_tags += n_keys
        return 0

    # Fill pass: write the way at the current positions, then move them on.
    if req.all_ways:
        out.all_id[out.n_all_ways] = way_id
        if _runs_fill(buf, &refs, True, True, out.all_refs + out.n_all_refs) != n_refs:
            return 1
        out.n_all_refs += n_refs
        out.n_all_ways += 1
        out.all_refs_off[out.n_all_ways] = out.n_all_refs
    if not matched:
        return 0
    out.way_id[out.n_ways] = way_id
    out.way_meta[3 * out.n_ways] = version
    out.way_meta[3 * out.n_ways + 1] = timestamp
    out.way_meta[3 * out.n_ways + 2] = visible
    if _runs_fill(buf, &refs, True, True, out.way_refs + out.n_way_refs) != n_refs:
        return 1
    if _runs_fill(buf, &keys, False, False, out.way_keys + out.n_way_tags) != n_keys:
        return 1
    if _runs_fill(buf, &vals, False, False, out.way_vals + out.n_way_tags) != n_keys:
        return 1
    out.n_way_refs += n_refs
    out.n_way_tags += n_keys
    out.n_ways += 1
    out.way_refs_off[out.n_ways] = out.n_way_refs
    out.way_tags_off[out.n_ways] = out.n_way_tags
    return 0


cdef int _decode_block_nogil(const unsigned char* buf, Py_ssize_t end,
                             const _Request* req, _Out* out) noexcept nogil:
    """Count or decode one block. 1 when it needs the Python path."""
    cdef Py_ssize_t pos = 0, f_start, g_pos, g_end, e_start, st_start = -1, st_end = -1
    cdef int field, wire
    cdef uint64_t length
    cdef int64_t granularity = 100, lat_offset = 0, lon_offset = 0
    cdef int64_t* key_ids = req.key_ids
    cdef Py_ssize_t n_ids = 0, n_table = 0
    cdef _DenseState state
    state.seen = False
    state.has_tags = False
    state.info_mask = 0
    # Two scans of the block's fields. Protobuf fields may come in any order, and the
    # groups need the coordinate scale and the string table, so the first scan reads the
    # header values and finds the table, and the second decodes the groups.
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire == WIRE_VARINT:
            length = _read_varint(buf, &pos, end)
            if field == PB_GRANULARITY:
                granularity = <int64_t>length
            elif field == PB_LAT_OFFSET:
                lat_offset = <int64_t>length
            elif field == PB_LON_OFFSET:
                lon_offset = <int64_t>length
            continue
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            return 1
        if field == PB_STRINGTABLE:
            if st_start >= 0:
                return 1  # each of several string tables is checked by the Python path
            st_start = f_start
            st_end = pos
    if pos != end:
        return 1
    if st_start >= 0:
        # Also checks the table's encoding, which decode_primitive_block always does.
        n_ids = _key_ids(buf, st_start, st_end, req, key_ids, &n_table)
        if n_ids < 0:
            return 1
    pos = 0
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
        if field != PB_PRIMITIVEGROUP:
            continue
        # A group's fields are its elements: dense groups, ways, plain nodes, relations.
        g_pos = f_start
        g_end = pos
        while g_pos < g_end:
            _read_tag(buf, &g_pos, g_end, &field, &wire)
            if wire != WIRE_LENGTH:
                return 1
            length = _read_varint(buf, &g_pos, g_end)
            e_start = g_pos
            g_pos = _field_end(g_pos, length, g_end)
            if g_pos > g_end:
                return 1
            if field == GROUP_NODES or field == GROUP_RELATIONS:
                return 1
            if field == GROUP_DENSE:
                if _decode_dense_group_nogil(buf, e_start, g_pos, req, out, granularity,
                                             lat_offset, lon_offset, key_ids, n_ids,
                                             n_table, &state) != 0:
                    return 1
            elif field == GROUP_WAYS:
                if _decode_way_nogil(buf, e_start, g_pos, req, out, key_ids, n_ids,
                                     n_table) != 0:
                    return 1
        if g_pos != g_end:
            return 1
    return 0


# What an empty batch buffer points to, so that offsets are never added to NULL.
cdef unsigned char _NO_BYTES[1]


def check_batch_starts(starts, Py_ssize_t size):
    """Raise ``ValueError`` unless the block offsets ``starts`` of a batch rise from 0 or
    more to at most ``size``, the length of the batch's buffer."""
    starts = np.asarray(starts)
    # Compared, not subtracted: a difference of int64 offsets can overflow.
    if (len(starts) == 0 or (starts < 0).any() or (starts > size).any()
            or (starts[1:] < starts[:-1]).any()):
        raise ValueError("starts must rise from 0 to at most the buffer's length")


cdef inline int64_t* _int64_ptr(array):
    return <int64_t*><uintptr_t>array.ctypes.data


def decode_batch_nogil(const unsigned char[::1] buffer, const int64_t[::1] starts,
                       keys=None, bint features=False, bint all_ways=False, bbox=None,
                       Int64Set node_ids=None):
    """Decode the blocks ``buffer[starts[i]:starts[i + 1]]`` without the GIL (it is held only
    to allocate the arrays). ``keys`` are the wanted tag keys (bytes; ``None`` wants every
    tagged element), ``features`` asks for the node features, ``all_ways`` for every way's id
    and refs, ``bbox`` (``xmin, ymin, xmax, ymax``) for the in-box flag of each node. With
    ``node_ids`` (an :class:`~pyrosm.int64_hash.Int64Set`) a dense node whose id is in it is
    a node feature too, tagged or not.

    Returns a dict: ``counts``, a table with a row per block and the columns
    ``BLOCK_COUNTS`` (status 1: the block was not decoded and has no elements), and the flat
    arrays of all blocks in order -- ``node_id``, ``node_lon``, ``node_lat``, ``in_box``,
    ``feature_row`` (into the node arrays), ``feature_pairs_off``, ``feature_keys``,
    ``feature_vals``, ``feature_meta`` (version, timestamp, changeset, visible), ``all_id``,
    ``all_refs_off``, ``all_refs``, ``way_id``, ``way_refs_off``, ``way_refs``,
    ``way_tags_off``, ``way_keys``, ``way_vals`` and ``way_meta`` (version, timestamp,
    visible). Tag keys and values are string ids of their block's string table (see
    :func:`block_string_table`)."""
    cdef Py_ssize_t n_blocks = starts.shape[0] - 1, b
    cdef _Request req
    cdef _Out out, before, fill
    check_batch_starts(starts, buffer.shape[0])
    # The wanted keys as one byte buffer plus offsets (with a trailing 0 so the buffer is
    # never empty), and the scratch arrays ``_key_ids`` fills per block.
    key_list = list(keys or ())
    key_bytes = np.frombuffer(b"".join(key_list) + b"\0", dtype=np.uint8)
    key_starts = np.zeros(len(key_list) + 1, dtype=np.int64)
    if key_list:
        np.cumsum([len(k) for k in key_list], out=key_starts[1:])
    req.keys = <const unsigned char*><uintptr_t>key_bytes.ctypes.data
    req.keys_off = _int64_ptr(key_starts)
    req.n_keys = len(key_list)
    key_ids = np.empty(len(key_list) + 1, dtype=np.int64)
    found = np.empty(len(key_list) + 1, dtype=np.uint8)
    req.key_ids = _int64_ptr(key_ids)
    req.found = <uint8_t*><uintptr_t>found.ctypes.data
    req.key_filter = keys is not None
    req.features = features
    req.all_ways = all_ways
    req.node_set = NULL if node_ids is None else node_ids.table
    req.bbox = bbox is not None
    req.xmin = req.ymin = req.xmax = req.ymax = 0
    if bbox is not None:
        req.xmin, req.ymin, req.xmax, req.ymax = bbox
    counts = np.zeros((n_blocks, BC_COUNT), dtype=np.int64)
    cdef int64_t[:, ::1] c = counts
    cdef const unsigned char* buf = _NO_BYTES
    if buffer.shape[0]:
        buf = &buffer[0]
    cdef const int64_t* at = &starts[0]

    memset(&out, 0, sizeof(_Out))
    memset(&fill, 0, sizeof(_Out))
    # Count pass: each block's sizes, in its row of ``counts``. A block that fails gets
    # status 1, and the counters go back to what they were before it.
    with nogil:
        for b in range(n_blocks):
            before = out
            if _decode_block_nogil(buf + at[b], at[b + 1] - at[b], &req, &out) != 0:
                c[b, BC_STATUS] = 1
                out = before
                continue
            c[b, BC_NODES] = out.n_nodes - before.n_nodes
            c[b, BC_FEATURES] = out.n_features - before.n_features
            c[b, BC_PAIRS] = out.n_pairs - before.n_pairs
            c[b, BC_ALL_WAYS] = out.n_all_ways - before.n_all_ways
            c[b, BC_ALL_REFS] = out.n_all_refs - before.n_all_refs
            c[b, BC_WAYS] = out.n_ways - before.n_ways
            c[b, BC_WAY_REFS] = out.n_way_refs - before.n_way_refs
            c[b, BC_WAY_TAGS] = out.n_way_tags - before.n_way_tags

    # The arrays, sized by the count pass's totals. The *_off arrays start with 0.
    arrays = {
        "node_id": np.empty(out.n_nodes, dtype=np.int64),
        "node_lon": np.empty(out.n_nodes, dtype=np.float64),
        "node_lat": np.empty(out.n_nodes, dtype=np.float64),
        "in_box": np.zeros(out.n_nodes if bbox is not None else 0, dtype=np.uint8),
        "feature_row": np.empty(out.n_features, dtype=np.int64),
        "feature_pairs_off": np.zeros(out.n_features + 1, dtype=np.int64),
        "feature_keys": np.empty(out.n_pairs, dtype=np.int64),
        "feature_vals": np.empty(out.n_pairs, dtype=np.int64),
        "feature_meta": np.empty((out.n_features, 4), dtype=np.int64),
        "all_id": np.empty(out.n_all_ways, dtype=np.int64),
        "all_refs_off": np.zeros(out.n_all_ways + 1, dtype=np.int64),
        "all_refs": np.empty(out.n_all_refs, dtype=np.int64),
        "way_id": np.empty(out.n_ways, dtype=np.int64),
        "way_refs_off": np.zeros(out.n_ways + 1, dtype=np.int64),
        "way_refs": np.empty(out.n_way_refs, dtype=np.int64),
        "way_tags_off": np.zeros(out.n_ways + 1, dtype=np.int64),
        "way_keys": np.empty(out.n_way_tags, dtype=np.int64),
        "way_vals": np.empty(out.n_way_tags, dtype=np.int64),
        "way_meta": np.empty((out.n_ways, 3), dtype=np.int64),
    }
    scratch = np.empty(4 * out.max_group + 1, dtype=np.int64)
    fill.fill = True
    fill.node_id = _int64_ptr(arrays["node_id"])
    fill.node_lon = <double*><uintptr_t>arrays["node_lon"].ctypes.data
    fill.node_lat = <double*><uintptr_t>arrays["node_lat"].ctypes.data
    fill.in_box = <uint8_t*><uintptr_t>arrays["in_box"].ctypes.data
    fill.feature_row = _int64_ptr(arrays["feature_row"])
    fill.feature_pairs_off = _int64_ptr(arrays["feature_pairs_off"])
    fill.feature_keys = _int64_ptr(arrays["feature_keys"])
    fill.feature_vals = _int64_ptr(arrays["feature_vals"])
    fill.feature_meta = _int64_ptr(arrays["feature_meta"])
    fill.all_id = _int64_ptr(arrays["all_id"])
    fill.all_refs_off = _int64_ptr(arrays["all_refs_off"])
    fill.all_refs = _int64_ptr(arrays["all_refs"])
    fill.way_id = _int64_ptr(arrays["way_id"])
    fill.way_refs_off = _int64_ptr(arrays["way_refs_off"])
    fill.way_refs = _int64_ptr(arrays["way_refs"])
    fill.way_tags_off = _int64_ptr(arrays["way_tags_off"])
    fill.way_keys = _int64_ptr(arrays["way_keys"])
    fill.way_vals = _int64_ptr(arrays["way_vals"])
    fill.way_meta = _int64_ptr(arrays["way_meta"])
    fill.scratch = _int64_ptr(scratch)
    # Fill pass: the blocks of status 0 again, now writing into the arrays.
    with nogil:
        for b in range(n_blocks):
            if c[b, BC_STATUS] == 0 and _decode_block_nogil(
                    buf + at[b], at[b + 1] - at[b], &req, &fill) != 0:
                c[b, BC_STATUS] = 2
    if (counts[:, BC_STATUS] == 2).any():
        raise RuntimeError("a block decoded differently in its two passes")
    arrays["in_box"] = arrays["in_box"].view(bool)
    arrays["counts"] = counts
    return arrays


def block_string_table(const unsigned char[::1] data):
    """The string table of a decompressed block, as ``decode_primitive_block`` gives it
    (the last one of a block that has several, each of which must decode), for resolving
    the string ids ``decode_batch_nogil`` gives. Raises ``InvalidOSMFileError`` for a
    block whose fields do not fit in it."""
    cdef Py_ssize_t end = data.shape[0], pos = 0, f_start
    cdef int field, wire
    cdef uint64_t length
    table = []
    if end == 0:
        return table
    cdef const unsigned char* buf = &data[0]
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        if wire != WIRE_LENGTH:
            _skip_field(buf, &pos, end, wire)
            continue
        length = _read_varint(buf, &pos, end)
        f_start = pos
        pos = _field_end(pos, length, end)
        if pos > end:
            break
        if field == PB_STRINGTABLE:
            table = _decode_string_table(buf, f_start, pos)
    _check_end(pos, end)
    return table
