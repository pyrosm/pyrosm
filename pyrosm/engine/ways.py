"""The collect step of a threaded read, column by column (see :mod:`pyrosm.way_columns`).

The threaded decode stores each element's tags as ids of strings, not as a dict. An
:class:`ElementStore` gathers those columns from the shards for the candidate ways, and
:func:`kept_ways` selects the ways a read outputs (the value filter in C, see
:func:`keep_mask`); :func:`tag_lists` builds their tag columns and JSON ``tags`` column
in C, and :func:`chunk_tables` the chunks of a result, with the geometries, on the read's
threads (:func:`network_frame` the edges of a network); :func:`node_features` and
:func:`node_table` give the point features from their string ids. The module also gathers
the node coordinates, the relation member ways and the network's node records on the
read's threads.
"""

import sys

import numpy as np
from rapidjson import dumps

from pyrosm import way_columns as wc
from pyrosm._arrays import _offsets_from_lengths
from pyrosm.engine.decode import _key_bytes, _merge_shards, _open_shard
from pyrosm.engine.pool import read_threads, thread_map

# The shard key of each column of a store of ways, in the store's constructor's order.
_WAY_KEYS = {
    "id": "way_id",
    "refs_off": "refs_off",
    "refs": "refs",
    "version": "way_version",
    "timestamp": "way_timestamp",
    "visible": "way_visible",
    "pairs_off": "way_pairs_off",
    "pair_key": "way_pair_key",
    "pair_val": "way_pair_val",
    "str_off": "str_off",
    "str_data": "str_data",
}


def _joined_shards(shard_paths, present, keys):
    """The arrays ``keys`` of the shards whose array ``present`` is not empty, joined into
    one shard dict as :func:`~pyrosm.engine.decode._merge_shards` joins blocks (offsets
    rebuilt, string ids renumbered); ``None`` when no shard has one."""
    parts = []
    for path in shard_paths:
        z = _open_shard(path)
        if len(z[present]):
            parts.append({key: z[key] for key in keys})
    return _merge_shards(parts) if parts else None


def _no_nodes():
    """An empty coordinate store."""
    from pyrosm.node_lookup import NodeLocations

    return NodeLocations(
        {"id": np.empty(0, np.int64), "lon": np.empty(0), "lat": np.empty(0)}
    )


class ElementStore:
    """The ways of a read as flat columns (see :mod:`pyrosm.way_columns`): ``id``, the node
    refs (``refs``, way ``i`` owning ``refs[refs_off[i]:refs_off[i + 1]]``), ``version``,
    ``timestamp``, ``visible``, and the tags as pairs of string ids (``pair_key`` /
    ``pair_val``, way ``i`` owning ``pairs_off[i]:pairs_off[i + 1]``) of the strings
    ``str_off`` / ``str_data`` hold. ``string_info`` keeps the
    :func:`pyrosm.way_columns.string_columns` of the strings per tuple of column keys; the
    stores :meth:`take` gives share it, as they share the strings."""

    def __init__(
        self,
        id,
        refs_off,
        refs,
        version,
        timestamp,
        visible,
        pairs_off,
        pair_key,
        pair_val,
        str_off,
        str_data,
        string_info=None,
    ):
        self.id = id
        self.refs_off = refs_off
        self.refs = refs
        self.version = version
        self.timestamp = timestamp
        self.visible = visible
        self.pairs_off = pairs_off
        self.pair_key = pair_key
        self.pair_val = pair_val
        self.str_off = str_off
        self.str_data = str_data
        self.string_info = {} if string_info is None else string_info
        self._area_by_tags = None

    def __len__(self):
        return len(self.id)

    @classmethod
    def from_shards(cls, shard_paths):
        """The ways of the threaded decode's shards, or ``None`` when they hold none."""
        joined = _joined_shards(shard_paths, "way_id", _WAY_KEYS.values())
        if joined is None:
            return None
        return cls(**{key: joined[shard_key] for key, shard_key in _WAY_KEYS.items()})

    def take(self, rows):
        """The store of the ways ``rows``, sharing this store's strings."""
        rows = np.asarray(rows, dtype=np.int64)
        refs_off, refs = wc.take(self.refs_off, self.refs, rows)
        pairs_off, pair_key = wc.take(self.pairs_off, self.pair_key, rows)
        _, pair_val = wc.take(self.pairs_off, self.pair_val, rows)
        return ElementStore(
            id=self.id[rows],
            refs_off=refs_off,
            refs=refs,
            version=self.version[rows],
            timestamp=self.timestamp[rows],
            visible=self.visible[rows],
            pairs_off=pairs_off,
            pair_key=pair_key,
            pair_val=pair_val,
            str_off=self.str_off,
            str_data=self.str_data,
            string_info=self.string_info,
        )

    def area_by_tags(self):
        """Whether the tags of each way make it an area when it is closed (see
        :func:`pyrosm.way_columns.area_by_tags`), found once per store."""
        if self._area_by_tags is None:
            self._area_by_tags = wc.area_by_tags(
                self.pairs_off,
                self.pair_key,
                self.pair_val,
                self.str_off,
                self.str_data,
            )
        return self._area_by_tags

    def string(self, i):
        """String ``i`` as the Python path decodes it."""
        return bytes(self.str_data[self.str_off[i] : self.str_off[i + 1]]).decode(
            "utf-8", "replace"
        )

    def tags(self, w, keys=None):
        """The tag dict of way ``w`` as the Python path builds it, with only the keys
        ``keys`` when given."""
        tags = {}
        for p in range(self.pairs_off[w], self.pairs_off[w + 1]):
            key = self.string(self.pair_key[p])
            if keys is None or key in keys:
                tags[key] = self.string(self.pair_val[p])
        return tags

    def node_lists(self):
        """Each way's node refs, as an object array of arrays."""
        nodes = np.empty(len(self), dtype=object)
        nodes[:] = np.split(self.refs, self.refs_off[1:-1])
        return nodes

    def way_columns(self, keys=None):
        """The ways as :func:`~pyrosm.engine.collect._read_way_columns` gives them, with
        only the tag keys ``keys`` when given."""
        tags = np.empty(len(self), dtype=object)
        tags[:] = [self.tags(w, keys) for w in range(len(self))]
        return {
            "id": self.id,
            "nodes": self.node_lists(),
            "tags": tags,
            "version": self.version,
            "timestamp": self.timestamp,
            "visible": self.visible.astype(bool),
        }


def _filter_entries(data_filter):
    """The entries of a dict value filter for :func:`pyrosm.way_columns.value_filter`, or
    ``None`` for a filter it does not cover (a compiled filter, an unknown value type, a
    string UTF-8 cannot encode)."""
    if not isinstance(data_filter, dict):
        return None
    try:
        return _encoded_entries(data_filter)
    except UnicodeEncodeError:
        return None  # a string no PBF holds, such as a lone surrogate


def _encoded_entries(data_filter):
    """The entries of :func:`_filter_entries` for a dict filter, its strings encoded; raises
    ``UnicodeEncodeError`` for a string UTF-8 cannot encode."""
    from pyrosm.config.osm_filters import _CLOSED, AccessRule, MultiValue

    entries = []
    for key, values in data_filter.items():
        if isinstance(values, AccessRule):
            entries.append(
                (
                    wc.FILTER_ACCESS,
                    [k.encode("utf-8") for k in values.keys],
                    [v.encode("utf-8") for v in _CLOSED],
                )
            )
            continue
        if values is True or (isinstance(values, list) and values == [True]):
            entries.append((wc.FILTER_ANY, [key.encode("utf-8")], []))
            continue
        if not isinstance(values, (list, tuple, set, frozenset)):
            return None
        strings = [v.encode("utf-8") for v in values if isinstance(v, str)]
        kind = wc.FILTER_MULTI if isinstance(values, MultiValue) else wc.FILTER_VALUES
        entries.append((kind, [key.encode("utf-8")], strings))
    return entries


def keep_mask(store, filter_spec):
    """Whether each way of ``store`` passes the value filter ``filter_spec`` (see
    :func:`~pyrosm.engine.collect._keep_fn`). The C filter decides the ways it can; the
    others (``wc.CHECK``) go through the Python filter on their tag dicts."""
    from pyrosm.engine.collect import _keep_fn

    osm_keys, data_filter, filter_type = filter_spec
    n = len(store)
    if data_filter is None:
        return np.ones(n, dtype=bool)
    keep = _keep_fn(filter_spec)
    entries = _filter_entries(data_filter)
    if entries is None:
        return np.fromiter((keep(store.tags(w)) for w in range(n)), dtype=bool, count=n)
    valid, _ = wc.string_columns(store.str_off, store.str_data, [])
    decided = wc.value_filter(
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid.view(np.uint8),
        np.arange(n, dtype=np.int64),
        [k.encode("utf-8") for k in osm_keys],
        entries,
        filter_type == "keep",
    )
    mask = decided == wc.KEEP
    for w in np.flatnonzero(decided == wc.CHECK):
        mask[w] = keep(store.tags(w))
    return mask


def _string_columns(store, keys):
    """``(valid, column, special)`` for the tag columns ``keys`` of ``store``: whether each
    string is valid UTF-8 and the column it names as a key (see
    :func:`pyrosm.way_columns.string_columns`), and the special elements, whose tags only
    their tag dicts give (see :func:`pyrosm.way_columns.special_ways`). The first two
    are found once for the strings of a store and ``keys`` (see ``store.string_info``).
    """
    keys = tuple(keys)
    if keys not in store.string_info:
        store.string_info[keys] = wc.string_columns(
            store.str_off, store.str_data, [_key_bytes(k) for k in keys]
        )
    valid, column = store.string_info[keys]
    special = wc.special_ways(
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid,
        column,
    )
    return valid, column, special


def result_tag_columns(store, tags_as_columns, keep_metadata, keep_other_tags):
    """The tag columns of all the ways of ``store``, as
    :func:`~pyrosm.engine.collect._result_tag_columns` gives them."""
    from pyrosm.engine.collect import _way_column_keys

    column_keys, tag_cols = _way_column_keys(tags_as_columns, keep_metadata)
    tag_col_set = set(tag_cols)
    _, column, special = _string_columns(store, tag_cols)
    present, leftover = wc.column_presence(
        store.pairs_off,
        store.pair_key,
        column,
        np.flatnonzero(~special),
        len(tag_cols),
    )
    columns = {tag_cols[i] for i in np.flatnonzero(present)}
    leftover = leftover or not {"version", "timestamp", "visible"}.issubset(column_keys)
    requested = None if keep_other_tags else set(tags_as_columns)
    for w in np.flatnonzero(special):
        keys = set(store.tags(w, requested))
        if "id" in keys:
            keys.discard("id")
            keys.add("id_tag")
        columns |= keys & tag_col_set
        leftover = leftover or not keys <= tag_col_set
    if keep_other_tags and leftover and len(store) > 0:
        columns.add("tags")
    return columns


class ChunkContext:
    """What building a store's ways as the chunks of a result needs of a read: the output
    columns (in the order of the Python path), each string's column as a key, the special
    ways, the node coordinates and the area the geometries must intersect. ``network``
    builds a network's edges: lines, with a ``length`` column."""

    def __init__(
        self,
        store,
        tags_as_columns,
        keep_metadata,
        keep_other_tags,
        way_columns,
        node_coordinates=None,
        keep_node_info=False,
        bounding_box=None,
        network=False,
    ):
        from pyrosm.engine.collect import _way_column_keys

        self.store = store
        self.column_keys, self.tag_cols = _way_column_keys(
            tags_as_columns, keep_metadata
        )
        self.version_col = "version" in self.column_keys
        self.timestamp_col = "timestamp" in self.column_keys
        self.visible_col = "visible" in self.column_keys
        self.keep_other_tags = keep_other_tags
        self.keep_node_info = keep_node_info
        self.network = network
        # The area the geometries must intersect, as prepare_geodataframe filters them.
        if isinstance(bounding_box, list):
            from shapely import box

            bounding_box = box(*bounding_box)
        self.area = bounding_box
        # Whether the JSON ``tags`` column is built, and the metadata fields it holds.
        self.json = keep_other_tags and "tags" in way_columns
        self.json_version = self.json and not self.version_col
        self.json_timestamp = self.json and not self.timestamp_col
        self.json_visible = self.json and not self.visible_col
        # The output tag columns: the tag columns the result has, in column order.
        # ``out_column[c]`` is tag column ``c``'s position among them, -1 when not output.
        self.out_names = [k for k in self.tag_cols if k in way_columns]
        position = {k: i for i, k in enumerate(self.out_names)}
        self.out_column = np.array(
            [position.get(k, -1) for k in self.tag_cols], dtype=np.int32
        )
        self.valid, self.column, self.special = _string_columns(store, self.tag_cols)
        self.requested = None if keep_other_tags else set(tags_as_columns)
        self.node_coordinates = node_coordinates
        # Whether each way's tags make it an area when closed; a network's ways are lines.
        self.area_by_tags = None if network else store.area_by_tags()

    def special_rows(self, ways):
        """The tag columns (``{column: list}``) and JSON values of the ways ``ways`` of the
        store, from their tag dicts as :func:`~pyrosm.engine.collect._ways_arrays` builds
        them."""
        from pyrosm.engine.collect import _route_tags

        store = self.store
        col_lists, leftover, _ = _route_tags(
            [store.tags(w, self.requested) for w in ways],
            store.version[ways],
            store.timestamp[ways],
            store.visible[ways],
            self.tag_cols,
            self.keep_other_tags,
            self.json_version,
            self.json_timestamp,
            self.json_visible,
        )
        return col_lists, leftover

    def tag_arrays(self, rows):
        """:func:`pyrosm.way_columns.tag_columns` of the ways ``rows`` of the store."""
        store = self.store
        return wc.tag_columns(
            rows,
            store.pairs_off,
            store.pair_key,
            store.pair_val,
            store.str_off,
            store.str_data,
            valid=self.valid,
            column=self.column,
            out_column=self.out_column,
            n_out=len(self.out_names),
            special=self.special[rows],
            version=store.version,
            timestamp=store.timestamp,
            visible=store.visible,
            json=self.json,
            json_version=self.json_version,
            json_timestamp=self.json_timestamp,
            json_visible=self.json_visible,
        )


def tag_lists(store, tags_as_columns, keep_metadata, keep_other_tags):
    """The tag columns (``{column: object array}``, None where a way lacks the tag) and the
    JSON ``tags`` values of all the ways of ``store``, as the loop of
    :func:`~pyrosm.engine.collect._ways_arrays` builds them from tag dicts."""
    from pyrosm.engine.collect import _way_column_keys

    _, tag_cols = _way_column_keys(tags_as_columns, keep_metadata)
    ctx = ChunkContext(
        store, tags_as_columns, keep_metadata, keep_other_tags, set(tag_cols) | {"tags"}
    )
    built = ctx.tag_arrays(np.arange(len(store), dtype=np.int64))
    columns = {
        name: _string_array(*arrays).to_numpy(zero_copy_only=False)
        for name, arrays in zip(ctx.out_names, built["columns"])
    }
    if built["json"] is None:
        json = np.full(len(store), None, dtype=object)
    else:
        json = _string_array(*built["json"]).to_numpy(zero_copy_only=False)
    special = np.flatnonzero(ctx.special)
    if len(special):
        col_lists, leftover = ctx.special_rows(special)
        for i, w in enumerate(special.tolist()):
            json[w] = leftover[i]
            for name in ctx.out_names:
                columns[name][w] = col_lists[name][i]
    return columns, json


def _string_array(offsets, data, valid):
    """The Arrow string array of the values ``data[offsets[i]:offsets[i + 1]]``, null where
    ``valid`` is 0."""
    import pyarrow as pa

    valid = np.asarray(valid, dtype=bool)
    nulls = int(len(valid) - valid.sum())
    return pa.LargeStringArray.from_buffers(
        len(valid),
        pa.py_buffer(np.ascontiguousarray(offsets, dtype=np.int64)),
        pa.py_buffer(np.ascontiguousarray(data, dtype=np.uint8)),
        pa.py_buffer(np.packbits(valid, bitorder="little")) if nulls else None,
        nulls,
    )


# The largest offset of an Arrow binary array (32-bit offsets).
_BINARY_LIMIT = 2**31 - 1


def _binary_array(offsets, data):
    """A binary array of the values ``data[offsets[i]:offsets[i + 1]]``, in several chunks
    when the data exceed what 32-bit offsets reach."""
    import pyarrow as pa

    parts, start = [], 0
    n = len(offsets) - 1
    # Each chunk runs from ``start`` to ``stop``: the last row whose data end within
    # _BINARY_LIMIT bytes of the chunk's start, at least one row. An empty input still
    # gives one (empty) chunk.
    while start < n or not parts:
        stop = n
        if offsets[n] - offsets[start] > _BINARY_LIMIT:
            stop = max(
                start + 1,
                int(np.searchsorted(offsets, offsets[start] + _BINARY_LIMIT, "right"))
                - 1,
            )
        part = (offsets[start : stop + 1] - offsets[start]).astype(np.int32)
        values = data[offsets[start] : offsets[stop]]
        parts.append(
            pa.Array.from_buffers(
                pa.binary(),
                stop - start,
                [None, pa.py_buffer(part), pa.py_buffer(values)],
            )
        )
        start = stop
    return parts[0] if len(parts) == 1 else pa.chunked_array(parts, type=pa.binary())


def _replace(array, positions, values):
    """``array`` with the values at ``positions`` replaced by ``values`` (str or None)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    mask = np.zeros(len(array), dtype=bool)
    mask[positions] = True
    return pc.replace_with_mask(
        array, pa.array(mask), pa.array(values, type=array.type)
    )


def _constant_strings(text, n):
    """A string array of ``n`` times ``text``."""
    import pyarrow as pa

    size = len(text)
    return pa.LargeStringArray.from_buffers(
        n,
        pa.py_buffer(np.arange(0, size * n + 1, size, dtype=np.int64)),
        pa.py_buffer(text.encode("utf-8") * n),
    )


def _fix_orientation(chunk, positions):
    """Write the WKB of the rings at ``positions`` of ``chunk``, oriented by GEOS."""
    import shapely

    wkb, wkb_off = chunk["wkb"], chunk["wkb_off"]
    coords, coords_off = chunk["coords"], chunk["coords_off"]
    for k in positions:
        ring = coords[coords_off[k] : coords_off[k + 1]]
        polygon = shapely.orient_polygons(shapely.polygons(ring), exterior_cw=False)
        data = shapely.to_wkb(polygon)
        wkb[wkb_off[k] : wkb_off[k + 1]] = np.frombuffer(data, dtype=np.uint8)
        coords[coords_off[k] : coords_off[k + 1]] = shapely.get_coordinates(polygon)


def _build_chunk(ctx, rows):
    """The chunk of the ways ``rows`` of ``ctx.store`` (see
    :func:`pyrosm.way_columns.build_chunk`), with the special ways' columns and the
    orientation GEOS decides filled in, the tag columns and the JSON as Arrow arrays
    (``tag_arrays``, ``json_array``). Returns ``(chunk, kept_rows)``."""
    store = ctx.store
    rows = np.asarray(rows, dtype=np.int64)
    special = ctx.special[rows]
    chunk = wc.build_chunk(
        rows,
        store.refs_off,
        store.refs,
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid=ctx.valid,
        column=ctx.column,
        out_column=ctx.out_column,
        n_out=len(ctx.out_names),
        special=special,
        version=store.version,
        timestamp=store.timestamp,
        visible=store.visible,
        nodes=ctx.node_coordinates,
        network=ctx.network,
        area_by_tags=ctx.area_by_tags,
        json=ctx.json,
        json_version=ctx.json_version,
        json_timestamp=ctx.json_timestamp,
        json_visible=ctx.json_visible,
    )
    if len(chunk["check"]):
        _fix_orientation(chunk, chunk["check"])
    # ``kept`` indexes ``rows`` (the ways with a geometry); ``positions`` index ``kept``.
    kept = chunk["kept"]
    columns = [_string_array(*c) for c in chunk["columns"]]
    json = _string_array(*chunk["json"]) if chunk["json"] is not None else None
    positions = np.flatnonzero(special[kept])
    if len(positions):
        col_lists, leftover = ctx.special_rows(rows[kept[positions]])
        columns = [
            _replace(array, positions, col_lists[name])
            for array, name in zip(columns, ctx.out_names)
        ]
        if json is not None:
            json = _replace(json, positions, leftover)
    chunk["tag_arrays"] = columns
    chunk["json_array"] = json
    return chunk, rows[kept]


def _chunk_parts(ctx, start, stop):
    """The Arrow table of the ways ``start:stop`` of ``ctx.store`` as the Python path
    writes a chunk of a layer, or the edges of a network with ``ctx.network`` (without
    schema metadata), with the bounds and the geometry types (bits of
    ``way_columns.LINESTRING`` / ``POLYGON``) of its geometries; ``(None, None, 0)`` when
    none of them has a geometry (in ``ctx.area``)."""
    import pyarrow as pa

    store = ctx.store
    chunk, rows = _build_chunk(ctx, np.arange(start, stop))
    if len(rows) == 0:
        return None, None, 0
    names, arrays = [], []
    tags = dict(zip(ctx.out_names, chunk["tag_arrays"]))
    for key in ctx.column_keys:
        if key == "id":
            array = pa.array(store.id[rows])
        elif key == "nodes":
            if not ctx.keep_node_info:
                continue
            offsets, refs = wc.take(store.refs_off, store.refs, rows)
            array = pa.ListArray.from_arrays(
                pa.array(offsets.astype(np.int32)), pa.array(refs)
            )
        elif key == "version" and ctx.version_col:
            array = pa.array(store.version[rows].astype(np.int32))
        elif key == "timestamp" and ctx.timestamp_col:
            array = pa.array(store.timestamp[rows].astype(np.uint32))
        elif key == "visible" and ctx.visible_col:
            array = pa.array(store.visible[rows].astype(bool))
        elif key in tags:
            array = tags[key]
        else:
            continue
        names.append(key)
        arrays.append(array)
    if chunk["json_array"] is not None:
        names.append("tags")
        arrays.append(chunk["json_array"])
    names.append("osm_type")
    arrays.append(_constant_strings("way", len(rows)))
    names.append("geometry")
    arrays.append(_binary_array(chunk["wkb_off"], chunk["wkb"]))
    if ctx.network:
        from pyrosm.distance import line_lengths

        owner = np.repeat(np.arange(len(rows)), np.diff(chunk["coords_off"]))
        lengths = pa.array(line_lengths(chunk["coords"], owner, len(rows)))
        # As ``DataFrame.assign`` does, a tag column ``length`` takes the lengths in place.
        if "length" in names:
            arrays[names.index("length")] = lengths
        else:
            names.append("length")
            arrays.append(lengths)
    table = pa.Table.from_arrays(arrays, names=names)
    if ctx.area is not None:
        return _in_area(table, ctx.area)
    return table, chunk["bounds"], chunk["types"]


def _in_area(table, area):
    """:func:`_chunk_parts`'s parts of the rows of ``table`` whose geometry intersects
    ``area``, as the spatial filter of ``prepare_geodataframe`` keeps them."""
    import pyarrow as pa
    import shapely

    geometries = shapely.from_wkb(
        table.column("geometry").to_numpy(zero_copy_only=False)
    )
    inside = shapely.intersects(geometries, area)
    if not inside.any():
        return None, None, 0
    if not inside.all():
        table = table.filter(pa.array(inside))
        geometries = geometries[inside]
    kinds = set(shapely.get_type_id(geometries).tolist())
    types = (wc.LINESTRING if 1 in kinds else 0) | (wc.POLYGON if 3 in kinds else 0)
    bounds = shapely.bounds(geometries)
    return (
        table,
        (
            bounds[:, 0].min(),
            bounds[:, 1].min(),
            bounds[:, 2].max(),
            bounds[:, 3].max(),
        ),
        types,
    )


def chunk_tables(ctx, starts, chunk_size):
    """The tables (see :func:`_chunk_parts`) of the chunks of ``chunk_size`` ways of
    ``ctx.store`` that begin at ``starts``, in order, built on the read's threads with at
    most one more chunk than threads in flight."""
    for table, _, _ in _chunks(ctx, starts, chunk_size):
        yield table


def _chunks(ctx, starts, chunk_size):
    """:func:`chunk_tables` with each table's bounds and geometry types."""
    n = len(ctx.store)
    yield from thread_map(
        lambda start: _chunk_parts(ctx, start, min(start + chunk_size, n)),
        starts,
        window=read_threads() + 1,
    )


class ArrowFrame:
    """A result held as an Arrow table (the network edges or the point features a threaded
    read builds), with the bounds and the geometry type names of its geometries. Written
    with :meth:`write` as ``GeoDataFrame.to_parquet`` writes the frame it holds; ``index``
    is that of ``geopandas.io.arrow._geopandas_to_arrow``."""

    def __init__(self, table, bounds, types, index=None):
        self.table = table
        self.bounds = bounds
        self.types = types
        self.index = index

    def schema(self):
        """The schema ``_geopandas_to_arrow`` gives the frame: that of one row holding each
        column's first value that is not null (so the column types the installed pandas
        infers for the whole frame), with the row count, the bounds and the geometry types
        of all rows in its metadata."""
        import json

        import geopandas as gpd
        import pyarrow as pa
        import pyarrow.compute as pc
        import shapely
        from geopandas.io.arrow import _geopandas_to_arrow

        values = []
        for column in self.table.columns:
            first = pc.index(pc.is_valid(column), True).as_py()
            values.append(column.take([max(first, 0)]))
        head = pa.Table.from_arrays(values, names=self.table.column_names).to_pandas()
        head["geometry"] = gpd.GeoSeries(
            shapely.from_wkb(head["geometry"].to_numpy()), crs="epsg:4326"
        )
        head = gpd.GeoDataFrame(head, geometry="geometry", crs="epsg:4326")
        schema = _geopandas_to_arrow(head, index=self.index).schema
        metadata = dict(schema.metadata)
        # The head's RangeIndex, if listed, stands for that of the frame, which has all
        # the rows.
        frame = json.loads(metadata[b"pandas"])
        for index in frame["index_columns"]:
            index["stop"] = self.table.num_rows
        metadata[b"pandas"] = json.dumps(frame).encode("utf-8")
        geo = json.loads(metadata[b"geo"])
        column = geo["columns"]["geometry"]
        column["geometry_types"] = sorted(self.types)
        column["bbox"] = [float(v) for v in self.bounds]
        metadata[b"geo"] = json.dumps(geo).encode("utf-8")
        return schema.with_metadata(metadata)

    def to_arrow(self):
        """The table as ``_geopandas_to_arrow`` gives the frame: its schema's types and
        metadata (a column without values, which arrow types as null, as nulls)."""
        import pyarrow as pa

        schema = self.schema()
        columns = [
            (
                pa.nulls(self.table.num_rows)
                if field.type == pa.null()
                else column.cast(field.type)
            )
            for column, field in zip(self.table.columns, schema)
        ]
        return pa.Table.from_arrays(columns, schema=schema)

    def write(self, path):
        """Write the frame to the GeoParquet file ``path``."""
        import pyarrow.parquet as pq

        pq.write_table(self.to_arrow(), path, compression="snappy")


def network_frame(ctx, chunk_size):
    """The edges of ``ctx.store`` (a network context) as an :class:`ArrowFrame`, built in
    chunks of ``chunk_size`` ways, or ``None`` when no way has a geometry."""
    import pyarrow as pa

    tables, bounds, types = [], [], 0
    starts = range(0, len(ctx.store), chunk_size)
    for table, box, kinds in _chunks(ctx, starts, chunk_size):
        if table is not None:
            tables.append(table)
            bounds.append(box)
            types |= kinds
    if not tables:
        return None
    bounds = np.array(bounds)
    return ArrowFrame(
        pa.concat_tables(tables),
        (
            bounds[:, 0].min(),
            bounds[:, 1].min(),
            bounds[:, 2].max(),
            bounds[:, 3].max(),
        ),
        [
            name
            for bit, name in ((wc.LINESTRING, "LineString"), (wc.POLYGON, "Polygon"))
            if types & bit
        ],
    )


def _node_store(shard_paths):
    """The node features of a threaded decode's shards: an :class:`ElementStore` of their
    ids, tags and metadata (without node refs), with their ``lon``, ``lat`` and metadata
    rows (version, timestamp, changeset, visible). ``None`` when no shard holds one."""
    joined = _joined_shards(
        shard_paths,
        "nfeat_id",
        (
            "nfeat_id",
            "nfeat_lon",
            "nfeat_lat",
            "nfeat_meta",
            "nfeat_pairs_off",
            "nfeat_pair_key",
            "nfeat_pair_val",
            "str_off",
            "str_data",
        ),
    )
    if joined is None:
        return None
    ids, meta = joined["nfeat_id"], joined["nfeat_meta"]
    store = ElementStore(
        id=ids,
        refs_off=np.zeros(len(ids) + 1, dtype=np.int64),
        refs=np.empty(0, np.int64),
        version=meta[:, 0],
        timestamp=meta[:, 1],
        visible=meta[:, 3],
        pairs_off=joined["nfeat_pairs_off"],
        pair_key=joined["nfeat_pair_key"],
        pair_val=joined["nfeat_pair_val"],
        str_off=joined["str_off"],
        str_data=joined["str_data"],
    )
    return store, joined["nfeat_lon"], joined["nfeat_lat"], meta


def _node_tag_columns(store, tags_as_columns):
    """The tag columns and JSON ``tags`` column of the nodes of ``store`` as
    ``tagparser.explode_node_tag_array`` builds them from their tag dicts -- the requested
    columns some node has, in order, then ``tags`` when some node has other tags -- as
    Arrow string arrays."""
    column_keys = list(dict.fromkeys(tags_as_columns))
    column_set = set(column_keys)
    n = len(store)
    valid, column, special = _string_columns(store, column_keys)
    present, _ = wc.column_presence(
        store.pairs_off,
        store.pair_key,
        column,
        np.flatnonzero(~special),
        len(column_keys),
    )
    present = present.copy()
    rows = {}
    for w in np.flatnonzero(special):
        other, values = {}, {}
        for k, v in store.tags(w).items():
            key = "id_tag" if k == "id" else k
            if key in column_set:
                values[key] = v
                present[column_keys.index(key)] = True
            else:
                other[key] = v
        rows[w] = (values, dumps(other) if other else None)
    names = [k for c, k in enumerate(column_keys) if present[c]]
    position = {k: i for i, k in enumerate(names)}
    built = wc.tag_columns(
        np.arange(n, dtype=np.int64),
        store.pairs_off,
        store.pair_key,
        store.pair_val,
        store.str_off,
        store.str_data,
        valid=valid,
        column=column,
        out_column=np.array([position.get(k, -1) for k in column_keys], dtype=np.int32),
        n_out=len(names),
        special=special,
        version=store.version,
        timestamp=store.timestamp,
        visible=store.visible,
        json=True,
        json_version=False,
        json_timestamp=False,
        json_visible=False,
    )
    columns = [_string_array(*arrays) for arrays in built["columns"]]
    json = _string_array(*built["json"])
    if rows:
        positions = np.array(sorted(rows), dtype=np.int64)
        columns = [
            _replace(
                array, positions, [rows[w][0].get(name) for w in positions.tolist()]
            )
            for array, name in zip(columns, names)
        ]
        json = _replace(json, positions, [rows[w][1] for w in positions.tolist()])
    data = dict(zip(names, columns))
    if json.null_count < len(json):
        data["tags"] = json
    return data


def python_columns(node_arrays):
    """``node_arrays`` with its Arrow columns as lists, as the Python path builds them."""
    import pyarrow as pa

    return {
        k: (v.to_pylist() if isinstance(v, pa.Array) else v)
        for k, v in node_arrays.items()
    }


def node_features(shard_paths, tags_as_columns, keep_metadata, filter_spec):
    """The node features of a threaded decode, as
    :func:`~pyrosm.engine.collect._collect_node_features` gives them (the tag columns as
    Arrow arrays when pyarrow is available and some node has other tags; see
    :func:`python_columns`), or ``None``."""
    from pyrosm.engine.collect import _explode_node_tags, _node_arrays
    from pyrosm.utils import _compat

    found = _node_store(shard_paths)
    if found is None:
        return None
    store, lon, lat, meta = found
    mask = keep_mask(store, filter_spec)
    if not mask.any():
        return None
    rows = np.flatnonzero(mask)
    store = store.take(rows)
    node_arrays = _node_arrays(
        store.id, lon[rows], lat[rows], meta[rows], keep_metadata
    )
    if _compat.HAS_PYARROW:
        data = _node_tag_columns(store, tags_as_columns)
        if "tags" in data:
            node_arrays.update(data)
            return node_arrays
    tags = np.empty(len(store), dtype=object)
    tags[:] = [store.tags(w) for w in range(len(store))]
    return _explode_node_tags(node_arrays, tags, tags_as_columns)


# The node columns that are not tags, with their dtypes.
_NODE_FIELDS = {
    "id": np.int64,
    "lon": np.float64,
    "lat": np.float64,
    "visible": np.bool_,
    "version": np.int32,
    "timestamp": np.uint32,
    "changeset": np.int32,
}


def node_table(node_arrays, bounding_box, keep_other_tags=True):
    """The point features ``node_arrays`` (from :func:`node_features`) as the Arrow table,
    with its metadata, that the Python path's node chunk gives (without the JSON ``tags``
    column when not ``keep_other_tags``), or ``None`` when no point is in
    ``bounding_box``. ``False`` when the Python path must build it: a tag column took a
    non-tag column's name (also ``geometry`` or ``osm_type``), or the ``tags`` column is not
    an Arrow array (no node has tags outside the columns, or pyarrow built none)."""
    import pyarrow as pa
    import shapely

    if not keep_other_tags:
        node_arrays = {k: v for k, v in node_arrays.items() if k != "tags"}
    for key, dtype in _NODE_FIELDS.items():
        values = node_arrays.get(key)
        if values is not None and not (
            isinstance(values, np.ndarray) and values.dtype == dtype
        ):
            return False
    tags = node_arrays.get("tags")
    if tags is not None and not isinstance(tags, (pa.Array, pa.ChunkedArray)):
        return False
    if "geometry" in node_arrays or "osm_type" in node_arrays:
        return False
    names, arrays = [], []
    for key, values in node_arrays.items():
        names.append(key)
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            arrays.append(values)
        elif isinstance(values, np.ndarray):
            arrays.append(pa.array(values))
        else:
            arrays.append(pa.array(values, type=pa.large_string()))
    n = len(node_arrays["id"])
    # Each point's WKB: the byte order (1 = little-endian), the type (1 = Point), x, y.
    points = np.empty(
        n, dtype=[("order", "u1"), ("type", "=u4"), ("x", "=f8"), ("y", "=f8")]
    )
    points["order"] = 1 if sys.byteorder == "little" else 0
    points["type"] = 1
    points["x"] = node_arrays["lon"]
    points["y"] = node_arrays["lat"]
    size = points.dtype.itemsize
    names += ["geometry", "osm_type"]
    arrays.append(
        _binary_array(
            np.arange(0, size * n + 1, size, dtype=np.int64), points.view(np.uint8)
        )
    )
    arrays.append(_constant_strings("node", n))
    table = pa.Table.from_arrays(arrays, names=names)
    if bounding_box is not None:
        area = (
            shapely.box(*bounding_box)
            if isinstance(bounding_box, list)
            else bounding_box
        )
        table = _in_area(table, area)[0]
        if table is None:
            return None
    lon = table.column("lon").to_numpy()
    lat = table.column("lat").to_numpy()
    frame = ArrowFrame(
        table,
        (lon.min(), lat.min(), lon.max(), lat.max()),
        ["Point"],
        index=False,
    )
    return frame.to_arrow()


def kept_ways(store, exclude_ids, filter_spec, in_box):
    """The ways of ``store`` a read outputs, as
    :func:`~pyrosm.engine.collect._filter_way_columns` selects them: those passing the value
    filter ``filter_spec`` (see :func:`keep_mask`), with a node in ``in_box`` (the ids of the
    nodes in the bounding box, or None), and not in ``exclude_ids``. ``None`` when none is
    left."""
    from pyrosm.int64_hash import Int64Set

    if store is None:
        return None
    mask = keep_mask(store, filter_spec)
    if in_box is not None:
        mask &= wc.with_node_in(store.refs_off, store.refs, Int64Set(in_box))
    if len(exclude_ids):
        mask &= ~wc.in_set(store.id, Int64Set(np.asarray(exclude_ids, dtype=np.int64)))
    if not mask.any():
        return None
    return store.take(np.flatnonzero(mask))


def node_coordinates(shard_paths, refs):
    """The coordinate store of the nodes ``refs`` (a list of id arrays) refer to, as
    :func:`~pyrosm.engine.collect._node_lookup` builds it: their ids in ascending order
    (the coordinates of the last shard holding an id) and coordinates, gathered from the
    shards on the read's threads."""
    import pandas as pd

    from pyrosm.int64_hash import Int64Set
    from pyrosm.node_lookup import NodeLocations

    refs = [r for r in refs if len(r)]
    if not refs:
        return _no_nodes()
    wanted = Int64Set(np.concatenate(refs))

    def gather(shard):
        z = _open_shard(shard)
        ids = z["node_id"]
        if len(ids) == 0:
            return None
        mask = wc.in_set(ids, wanted)
        return ids[mask], z["node_lon"][mask], z["node_lat"][mask]

    parts = [p for p in thread_map(gather, shard_paths) if p is not None]
    if not parts:
        return _no_nodes()
    ids, lon, lat = (np.concatenate(column) for column in zip(*parts))
    # Sorted by id with each id's last copy, unless the ids already ascend strictly.
    if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
        order = np.argsort(ids, kind="stable")
        ids, lon, lat = ids[order], lon[order], lat[order]
        last = np.append(ids[1:] != ids[:-1], True)
        ids, lon, lat = ids[last], lon[last], lat[last]
    return NodeLocations(pd.DataFrame({"id": ids, "lon": lon, "lat": lat}))


def relation_ways(shard_paths, member_ids, in_box=None):
    """The member ways of the kept relations, as
    :func:`~pyrosm.engine.collect._collect_relation_ways` gives them (the last shard
    holding a way wins), gathered from a threaded decode's shards on the read's threads,
    their node refs held in one array."""
    from pyrosm.int64_hash import Int64Set

    members = Int64Set(member_ids)
    box = None if in_box is None else Int64Set(in_box)

    def gather(shard):
        z = _open_shard(shard)
        ids = z["all_id"]
        if len(ids) == 0:
            return None
        rows = np.flatnonzero(wc.in_set(ids, members))
        off, refs = wc.take(z["all_refs_off"], z["all_refs"], rows)
        if box is not None:
            inside = np.flatnonzero(wc.with_node_in(off, refs, box))
            rows = rows[inside]
            off, refs = wc.take(off, refs, inside)
        return (ids[rows], np.diff(off), refs) if len(rows) else None

    parts = [part for part in thread_map(gather, shard_paths) if part is not None]
    if not parts:
        return None
    ids, counts, refs = (np.concatenate(column) for column in zip(*parts))
    starts = _offsets_from_lengths(counts)[:-1]
    # Each id once, ascending, at its last copy.
    unique, last_reversed = np.unique(ids[::-1], return_index=True)
    last = len(ids) - 1 - last_reversed
    nodes = np.empty(len(unique), dtype=object)
    nodes[:] = [
        refs[a : a + c] for a, c in zip(starts[last].tolist(), counts[last].tolist())
    ]
    return {"id": unique, "nodes": nodes}


def node_record_blocks(filepath, wanted, keep_metadata):
    """The records of the nodes ``wanted`` (an :class:`~pyrosm.int64_hash.Int64Set`) in
    each data block of ``filepath``, for
    :func:`~pyrosm.engine.collect._gather_node_records`, decoded without the GIL on the
    read's threads: yields ``(end, item)`` for each data blob in file order,
    ``end`` the blob's end in the file and ``item`` None (no wanted node), the block's
    bytes (for the Python path), or ``(record, counts, keys, vals, strings)``: the
    arrays of the wanted nodes and their tags, ``counts`` per node as ids of
    ``strings``."""
    from pyrosm.engine.blobs import _data_blob_spans, _index_blobs, _read_batches
    from pyrosm.engine.decode import _THREAD_BATCH, _decoded_parts

    blobs, _ = _data_blob_spans(_index_blobs(filepath))

    def decode(batch):
        out = []
        with open(filepath, "rb") as f:
            for part, buffer, starts in _decoded_parts(f, batch):
                out.extend(_part_records(part, buffer, starts, wanted, keep_metadata))
                del buffer, starts
        return out

    for items in thread_map(decode, _read_batches(blobs, _THREAD_BATCH)):
        yield from items


def _part_records(part, buffer, starts, wanted, keep_metadata):
    """The ``(end, item)`` of :func:`node_record_blocks` for the blocks of a part."""
    from pyrosm.primitive_block_decoder import BLOCK_COUNTS, decode_batch_nogil

    decoded = decode_batch_nogil(
        buffer, starts, keys=[], features=True, way_strings=True, node_ids=wanted
    )
    counts = decoded["counts"]
    status = counts[:, BLOCK_COUNTS.index("status")]
    # Block b's features are at[b]:at[b + 1]; feature_meta holds version, timestamp,
    # changeset, visible. The tags are ids of the part's strings, so ``strings`` (decoded
    # once) serves all its blocks.
    at = _offsets_from_lengths(counts[:, BLOCK_COUNTS.index("features")])
    rows, meta = decoded["feature_row"], decoded["feature_meta"]
    pairs_off = decoded["feature_pairs_off"]
    strings = None
    out = []
    for b, (offset, size, _) in enumerate(part):
        end = offset + size
        f0, f1 = at[b], at[b + 1]
        if status[b]:
            out.append((end, bytes(buffer[starts[b] : starts[b + 1]])))
            continue
        if f0 == f1:
            out.append((end, None))
            continue
        r = rows[f0:f1]
        record = {
            "id": decoded["node_id"][r],
            "lon": decoded["node_lon"][r],
            "lat": decoded["node_lat"][r],
            "visible": meta[f0:f1, 3].astype(bool),
        }
        if keep_metadata:
            for c, name in enumerate(("version", "timestamp", "changeset")):
                record[name] = meta[f0:f1, c].copy()
        if strings is None:
            off, data = decoded["string_off"], decoded["string_data"]
            strings = [
                data[a:b].tobytes().decode("utf-8", "replace")
                for a, b in zip(off[:-1].tolist(), off[1:].tolist())
            ]
        p0, p1 = pairs_off[f0], pairs_off[f1]
        out.append(
            (
                end,
                (
                    record,
                    np.diff(pairs_off[f0 : f1 + 1]),
                    decoded["feature_keys"][p0:p1],
                    decoded["feature_vals"][p0:p1],
                    strings,
                ),
            )
        )
    return out
