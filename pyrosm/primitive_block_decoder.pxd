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
    GROUP_DENSE = 2
    GROUP_WAYS = 3
    GROUP_RELATIONS = 4

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
