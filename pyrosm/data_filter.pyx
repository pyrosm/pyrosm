import numpy as np
import pandas as pd
from cykhash.khashsets cimport any_int64_from_iter, isin_int64, Int64Set_from_buffer
from cpython cimport array
from pyrosm.filter_compiler import CompiledFilter


# Structural fields the parser attaches to every (flattened) way record; any other
# top-level key is an OSM tag. Used to decide whether a way carries a real tag when
# reading all data with no key filter (custom_filter=None / keep_all).
WAY_STRUCTURAL_KEYS = frozenset(
    {"id", "version", "timestamp", "visible", "nodes", "changeset"}
)


class Solver:
    """Solver is used to toggle between exclude / keep checks applied in data filter."""
    def __init__(self, direction):

        if direction == "exclude":
            self.solver = self.isin_check
        elif direction == "keep":
            self.solver = self.notin_check
        else:
            raise ValueError("filter type should be 'keep' or 'exclude'")

    def isin_check(self, value, container):
        if True in container or value in container:
            return True
        return False

    def notin_check(self, value, container):
        return not self.isin_check(value, container)

    def check(self, value, container):
        return self.solver(value, container)


cdef has_osm_data_type(osm_data_types, tag_keys):
    cdef str key, osm_data_type
    cdef int i, n = len(tag_keys)
    for i in range(0, n):
        key = tag_keys[i]
        for osm_data_type in osm_data_types:
            if osm_data_type == key:
                return True
    return False


cdef way_is_part_of_relation(way_record, lookup_dict):
    try:
        lookup_dict[way_record["id"]]
        return True
    except KeyError as e:
        return False
    except Exception as e:
        raise e


cdef filter_osm_records(data_records,
                        data_filter,
                        osm_data_type,
                        relation_way_ids,
                        filter_type,
                        bint keep_all=False):
    """
    Filter OSM data records by given OSM tag key:value pairs.
    
    Parameters
    ----------
    data_records : list:
        A list of OSM data records. 
        A single record is a dictionary with OSM data, such as: 
           - {"highway": "primary", "maxspeed": 80, "name": "Highway-name-foo", 
              "nodes": [1111,2222,3333,4444]}
              
    data_filter : dict ( {"osm-key": ["list-of-osm-values"]} )
        A dictionary of tag-keys and associated values that will be used to
        filter the OSM data records, e.g. {"building": ["residential"]} filters
        the records where "building" key has a value "residential". 
        The records will be kept or excluded according the <filter_type> parameter.   
              
    osm_data_type : str | list
        Basic data type(s) used for filtering, such as:
         - 'highway' (for roads), 
         - 'building' (for buildings), 
         - 'landuse' (for landuse) etc.
         - ['amenity', 'shop', 'craft'] (a combination that can be useful for parsing POIs)
         
    relation_way_ids : list (optional)
        A list of way ids that belong to relations. Ways that match with these ids are always kept.
         
    filter_type : str ( 'keep' | 'exclude' ) 
        Whether the given data_filter should 'keep' or 'exclude' the records 
        where given tag:value pair is present in the record.  
    """
    cdef str rec_value
    cdef int i, N = len(data_records)

    solver = Solver(filter_type)
    filtered_data = []

    # An advanced (compiled) filter is evaluated as a predicate; the dict normalization and
    # the per-key OR loop below are skipped for it.
    cdef bint use_predicate = isinstance(data_filter, CompiledFilter)

    if not isinstance(osm_data_type, list):
        osm_data_type = [osm_data_type]

    if data_filter is not None and not use_predicate:
        if len(data_filter) == 0:
            data_filter = None

    if data_filter is not None and not use_predicate:
        # A bare True value ({osm_key: True}) means "match any value for this
        # key". Normalize it to [True] so the membership checks below and the
        # Solver can treat every filter value as an iterable, rather than
        # choking on a non-iterable bool (TypeError on `True in True`).
        data_filter = {
            k: [True] if v is True else v for k, v in data_filter.items()
        }

        # Filter keys to test each record against.
        filter_keys = data_filter.keys()

    relation_check = False
    if relation_way_ids is not None:
        relation_way_lookup = dict.fromkeys(relation_way_ids)
        relation_check = True

    for i in range(0, N):
        record = data_records[i]
        record_keys = list(record.keys())
        # If way is part of relation it should be kept
        # (ways that are part of relation might not have any tags)
        if relation_check:
            if way_is_part_of_relation(record, relation_way_lookup):
                filtered_data.append(record)
                continue

        # keep_all (custom_filter=None): keep any way carrying at least one OSM tag,
        # i.e. any key outside the parser's structural fields. Standalone untagged
        # ways are dropped (relation-member ways are already kept above).
        if keep_all:
            if len(set(record_keys) - WAY_STRUCTURAL_KEYS) == 0:
                continue
        elif not has_osm_data_type(osm_data_type, record_keys):
            continue

        # Advanced filter: the key gate above already restricted candidacy to the filter's
        # positive keys; keep/exclude the record by the predicate (issues #116, #341).
        if use_predicate:
            if data_filter.matches(record):
                if filter_type == "keep":
                    filtered_data.append(record)
            elif filter_type == "exclude":
                filtered_data.append(record)
            continue

        # Check if should be filtered based on given data_filter.
        # Evaluate ALL filter keys present in the record (OR semantics): a record
        # matches the filter if any present filter key's value is in that key's
        # list. The old code broke on the first filter key in the record, so
        # exclusions/keeps on secondary keys were skipped and depended on tag
        # order (issues #108, #112). This mirrors the relations/nodes path
        # (record_should_be_kept).
        if data_filter is not None:
            matched = False
            for k in filter_keys:
                if k in record:
                    if solver.isin_check(record[k], data_filter[k]):
                        matched = True
                        break

            # keep: retain the record only if a filter key matched.
            # exclude: retain it only if no filter key matched (a record with
            # none of the filter keys is kept under exclude).
            if filter_type == "keep":
                if matched:
                    filtered_data.append(record)
            else:
                if not matched:
                    filtered_data.append(record)

        # If data_filter has not been specified,
        # all data for specified osm-keys should be kept.
        else:
            filtered_data.append(record)
    return filtered_data


cpdef _filter_array_dict_by_indices_or_mask(array_dict, indices):
    return filter_array_dict_by_indices_or_mask(array_dict, indices)


cdef filter_array_dict_by_indices_or_mask(array_dict, indices):
    return {k: v[indices] for k, v in array_dict.items()}


cdef get_lookup_khash_for_int64(int64_id_array):
    return Int64Set_from_buffer(
        memoryview(
            int64_id_array
        )
    )


cdef get_nodeid_lookup_khash(nodes):
    return get_lookup_khash_for_int64(
        np.concatenate([group['id'].tolist()
                        for group in nodes]).astype(np.int64)
    )


cdef nodes_for_way_exist_khash(nodes, node_lookup):
    return any_int64_from_iter(nodes, node_lookup)


cpdef get_mask_by_osmid(src_array, osm_ids):
    """
    Creates a (boolean) mask for the given source array flagging True
    all items that exist in the 'osm_ids' array. Can be used to filter items
    e.g. from OSM node data arrays.
    """
    n = len(src_array)
    lookup = Int64Set_from_buffer(osm_ids)
    result = np.empty(src_array.size, dtype=np.bool)
    isin_int64(src_array, lookup, result)
    return result


cdef record_should_be_kept(tag, osm_keys, data_filter, filter_type, bint keep_all=False):
    if tag is None:
        return False

    # keep_all (custom_filter=None): keep any element that carries a tag. For nodes
    # and relations 'tag' is already the separated tags dict, so this is exact.
    if keep_all:
        return len(tag) > 0

    cdef str k, osm_key

    # Advanced (compiled) filter: gate on the candidate keys (the element must carry at least
    # one positive key), then keep/exclude by the predicate (issues #116, #341).
    if isinstance(data_filter, CompiledFilter):
        for osm_key in osm_keys:
            if osm_key in tag:
                break
        else:
            return False
        if filter_type == "keep":
            return data_filter.matches(tag)
        return not data_filter.matches(tag)

    filter_keys = list(data_filter.keys())
    tag_keys = list(tag.keys())

    # Check if OSM key exist for the given element
    osm_key_was_found = False
    for osm_key in osm_keys:
        if osm_key in tag_keys:
            osm_key_was_found = True

    # If not, the element shouldn't be kept
    if not osm_key_was_found:
        return False

    # If there is no filter but the element is correct kind
    if len(filter_keys) == 0:
        if filter_type == "keep":
            return True
        else:
            return False

    # If there is a filter, check if match is found
    for k, v in data_filter.items():
        if k in tag_keys:
            # Check match with data filter
            if tag[k] in v:
                if filter_type == "keep":
                    return True
                else:
                    return False
            # If filter is not defined, check for 'osm_key': True
            elif v == [True] or v == True:
                if filter_type == "keep":
                    return True
                else:
                    return False

    if filter_type == "keep":
        return False
    return True


cpdef element_should_be_kept(tag, osm_keys, data_filter, filter_type):
    """Python entry point to the per-element keep/exclude decision, so an alternative
    reader can refine its key-presence candidates by the exact value filter pyrosm uses."""
    return record_should_be_kept(tag, osm_keys, data_filter, filter_type)


cdef filter_relation_indices(relations, osm_keys, data_filter, filter_type, bint keep_all=False):
    cdef int i, n = len(relations.get("tags", []))
    indices = []

    # Ensure keys (an advanced compiled filter is passed straight through to the predicate).
    if isinstance(data_filter, CompiledFilter):
        relation_filter = data_filter
    elif len(data_filter) == 0:
        relation_filter = {}
    else:
        relation_filter = {key: value for key, value in data_filter.items()}

    for i in range(0, n):
        tag = relations["tags"][i]
        if record_should_be_kept(tag, osm_keys, relation_filter, filter_type, keep_all):
            indices.append(i)
    return indices


cdef filter_node_indices(node_arrays, osm_keys, data_filter, filter_type, bint keep_all=False):
    cdef int i, n = len(node_arrays["tags"])
    indices = []

    if isinstance(data_filter, CompiledFilter):
        node_filter = data_filter
    elif len(data_filter) == 0:
        node_filter = {}
    else:
        node_filter = {key: value for key, value in data_filter.items()}

    for i in range(0, n):
        tag = node_arrays["tags"][i]
        if record_should_be_kept(tag, osm_keys, node_filter, filter_type, keep_all):
            indices.append(i)

    return indices


cpdef get_latest_version(df):
    # The order of versions is always the same
    # (newest version is the last)
    return df.groupby("id").last().reset_index()


cdef inline bint _is_empty_tag_value(object v):
    """Return True if ``v`` is a missing-tag sentinel that should be dropped.

    A tag cell is "empty" only if it is ``None`` (pre-pandas-3.0), ``NaN``, or
    pandas-3.0 string/object ``pd.NA``. Array/list/tuple cells (e.g. the
    "nodes" key) are always real data here, never a missing sentinel -- and
    ``pd.isna()`` on them returns an array (calling ``bool()`` on which raises)
    -- so they are kept unconditionally. Real scalar values, including ``0``,
    ``0.0`` and ``""``, are kept.
    """
    if isinstance(v, (np.ndarray, list, tuple)):
        return False
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


cdef clean_empty_values_from_ways(ways):
    cdef int i, n = len(ways)
    cleaned = []
    for i in range(0, n):
        cleaned.append(
            {x: y for x, y in ways[i].items() if not _is_empty_tag_value(y)}
        )
    return cleaned
