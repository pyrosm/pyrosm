from libc.stdint cimport int64_t, uint64_t

# Field numbers of the PBF messages this module decodes, taken straight from
# proto/osmformat.proto and grouped by message. They are named so the decoder reads
# as "field DENSE_LAT" rather than a bare 8; the wire layout lives only here.

cdef enum PrimitiveBlockField:
    PB_STRINGTABLE = 1
    PB_PRIMITIVEGROUP = 2
    PB_GRANULARITY = 17
    PB_DATE_GRANULARITY = 18
    PB_LAT_OFFSET = 19
    PB_LON_OFFSET = 20

cdef enum StringTableField:
    ST_S = 1

cdef enum GroupField:
    GROUP_NODES = 1
    GROUP_DENSE = 2
    GROUP_WAYS = 3
    GROUP_RELATIONS = 4

cdef enum NodeField:
    NODE_ID = 1
    NODE_KEYS = 2
    NODE_VALS = 3
    NODE_INFO = 4
    NODE_LAT = 8
    NODE_LON = 9

cdef enum DenseField:
    DENSE_ID = 1
    DENSE_DENSEINFO = 5
    DENSE_LAT = 8
    DENSE_LON = 9
    DENSE_KEYS_VALS = 10

cdef enum DenseInfoField:
    DI_VERSION = 1
    DI_TIMESTAMP = 2
    DI_CHANGESET = 3
    DI_UID = 4
    DI_USER_SID = 5
    DI_VISIBLE = 6

cdef enum WayField:
    WAY_ID = 1
    WAY_KEYS = 2
    WAY_VALS = 3
    WAY_INFO = 4
    WAY_REFS = 8

cdef enum RelationField:
    REL_ID = 1
    REL_KEYS = 2
    REL_VALS = 3
    REL_INFO = 4
    REL_ROLES_SID = 8
    REL_MEMIDS = 9
    REL_TYPES = 10

cdef enum InfoField:
    INFO_VERSION = 1
    INFO_TIMESTAMP = 2
    INFO_CHANGESET = 3
    INFO_UID = 4
    INFO_USER_SID = 5
    INFO_VISIBLE = 6

# --- low-level wire-format primitives, shared with zlib_blob -------------------------

# The protobuf wire types we encounter.
cdef enum WireType:
    WIRE_VARINT = 0
    WIRE_FIXED64 = 1
    WIRE_LENGTH = 2
    WIRE_FIXED32 = 5


# The position a read moves to when the data does not follow the wire format: a varint
# cut off at the end of its data or longer than 10 bytes, a field that does not fit in
# the data that holds it, or an unknown wire type. It lies past the end of any data, so
# the loop over that data stops; the function that owns the loop then raises.
cdef extern from *:
    const Py_ssize_t _CORRUPT "PY_SSIZE_T_MAX"


cdef inline uint64_t _read_varint(const unsigned char* buf, Py_ssize_t* pos,
                                  Py_ssize_t end) noexcept nogil:
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
                                  Py_ssize_t end) noexcept nogil:
    """The end of a field of ``length`` bytes that starts at ``pos``, or ``_CORRUPT``
    when the field does not fit in data that ends at ``end``."""
    if pos > end or length > <uint64_t>(end - pos):
        return _CORRUPT
    return pos + <Py_ssize_t>length


cdef inline int64_t _zig_zag_decode(uint64_t value) noexcept nogil:
    """Undo protobuf's zig-zag encoding of a signed integer."""
    return (<int64_t>(value >> 1)) ^ (-(<int64_t>(value & 1)))


cdef inline void _read_tag(const unsigned char* buf, Py_ssize_t* pos, Py_ssize_t end,
                           int* field_number, int* wire_type) noexcept nogil:
    """Read a field tag and split it into its field number and wire type. A field
    number outside 1 to 2**29 - 1 moves ``pos`` to ``_CORRUPT``."""
    cdef uint64_t tag = _read_varint(buf, pos, end)
    if tag < 8 or (tag >> 32) != 0:
        pos[0] = _CORRUPT
        tag = 0
    field_number[0] = <int>(tag >> 3)
    wire_type[0] = <int>(tag & 0x7)


cdef inline void _skip_field(const unsigned char* buf, Py_ssize_t* pos,
                             Py_ssize_t end, int wire_type) noexcept nogil:
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


cdef inline Py_ssize_t _fields_end(const unsigned char* buf, Py_ssize_t start,
                                   Py_ssize_t end) noexcept nogil:
    """Where a scan of ``buf[start:end]`` as a run of fields stops: ``end`` when the
    fields are well-formed."""
    cdef Py_ssize_t pos = start
    cdef int field, wire
    while pos < end:
        _read_tag(buf, &pos, end, &field, &wire)
        _skip_field(buf, &pos, end, wire)
    return pos


cdef inline int _read_element_id(const unsigned char* buf, Py_ssize_t start,
                                 Py_ssize_t end, bint is_node,
                                 int64_t* out) noexcept nogil:
    """Write to ``out`` the id of the plain node, way or relation in ``buf[start:end]``:
    the last one, as protobuf reads it, or 0, the proto2 default, when it has none. A
    node's id is zig-zag coded. Returns 1 unless the element and its ``Info`` are
    well-formed fields, else 0. The three messages number the id and ``Info`` alike."""
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
            if pos > end or _fields_end(buf, f_start, pos) != pos:
                return 1
        else:
            _skip_field(buf, &pos, end, wire)
    if pos != end:
        return 1
    out[0] = _zig_zag_decode(value) if is_node else <int64_t>value
    return 0


# A packed repeated field (a way's refs, a dense group's ids) may come as several fields
# with the same number, which together form one list. ``_Runs`` keeps each one's byte span
# (``start[i]:end[i]``), up to ``_MAX_RUNS``; writers put a field in one run.
cdef enum:
    _MAX_RUNS = 8


cdef struct _Runs:
    Py_ssize_t n
    Py_ssize_t start[_MAX_RUNS]
    Py_ssize_t end[_MAX_RUNS]


cdef inline int _add_span(_Runs* runs, Py_ssize_t start, Py_ssize_t end) noexcept nogil:
    """Add a packed run; -1 when the field comes in more runs than are kept."""
    if runs.n == _MAX_RUNS:
        return -1
    runs.start[runs.n] = start
    runs.end[runs.n] = end
    runs.n += 1
    return 0


cdef Py_ssize_t _runs_count(const unsigned char* buf, _Runs* runs) noexcept nogil
cdef Py_ssize_t _runs_fill(const unsigned char* buf, _Runs* runs, bint delta, bint signed,
                           int64_t* out) noexcept nogil
