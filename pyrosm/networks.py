import numpy as np
from rapidjson import loads

from pyrosm.config.osm_filters import country_rule
from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe


def get_network_data(
    node_coordinates,
    way_records,
    tags_as_columns,
    network_filter,
    bounding_box,
    slice_to_segments,
    filter_type="exclude",
    keep_metadata=True,
    osm_keys="highway",
    steps=None,
):
    """The ways of the network that ``network_filter`` selects as ``(edges, nodes)``
    GeoDataFrames, read from the in-memory reader's parsed arrays, the ways sliced into
    edges between their nodes with ``slice_to_segments`` (nodes is None otherwise). The
    selection and the edges are steps of ``steps`` (a
    :class:`~pyrosm.utils.progress.Steps`, or None). A way built for another mode without
    a tag of the mode stays where its country lets the mode use it (see
    :func:`~pyrosm.config.osm_filters.country_rule`)."""
    # Structural columns are always kept; element metadata only when requested.
    tags_as_columns += ["id", "nodes"]
    if keep_metadata:
        tags_as_columns += ["timestamp", "changeset", "version"]

    # Call signature for fetching network data. Predefined networks keep only ways with a
    # 'highway' tag; an advanced custom filter selects ways by its own positive keys instead.
    network_filter, rule = country_rule(network_filter, filter_type)
    nodes, ways, relation_ways, relations = get_osm_data(
        node_arrays=None,
        way_records=way_records,
        relations=None,
        tags_as_columns=tags_as_columns,
        data_filter=network_filter,
        filter_type=filter_type,
        osm_keys=osm_keys,
        keep_metadata=keep_metadata,
        steps=steps,
    )

    if ways is not None and rule is not None:
        ways = _ways_by_country(ways, node_coordinates, rule, tags_as_columns)

    # If there weren't any data, return None
    if ways is None:
        return None, None

    # Prepare GeoDataFrame
    edges, nodes = prepare_geodataframe(
        nodes,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        tags_as_columns,
        bounding_box,
        parse_network=True,
        calculate_seg_lengths=slice_to_segments,
        keep_metadata=keep_metadata,
        steps=steps,
    )

    return edges, nodes


def _ways_by_country(ways, node_coordinates, rule, tags_as_columns):
    """The way arrays ``ways`` without the ways that the access rule ``rule`` leaves to
    their country (:meth:`~pyrosm.config.osm_filters.AccessRule.lacks_mode_tag`) and whose
    country, by their first node in ``node_coordinates``, does not let the mode use them;
    ``None`` when none is left. The mode's tag is read from its column, or else from the
    JSON ``tags`` column of the other tags. A column of ``tags_as_columns`` that only the
    dropped ways had a value in is left out, as for the ways a read never selects."""
    mode = rule.keys[0]
    highway = ways["highway"]
    rows = []
    for row in np.flatnonzero([h in rule.other_mode_highway for h in highway]):
        if mode in ways:
            value = ways[mode][row]
        else:
            other = ways["tags"][row] if "tags" in ways else None
            value = loads(other).get(mode) if isinstance(other, str) else None
        if rule.lacks_mode_tag({"highway": highway[row], mode: value}):
            rows.append(row)
    if not rows:
        return ways
    rows = np.asarray(rows)
    nodes = ways["nodes"]
    first = np.array(
        [nodes[row][0] if len(nodes[row]) else -1 for row in rows], dtype=np.int64
    )
    _, lon, lat = node_coordinates.gather(first)
    keep = np.ones(len(highway), dtype=bool)
    keep[rows[~rule.allowed_in(highway[rows], lon, lat)]] = False
    if not keep.any():
        return None
    kept = {}
    for key, values in ways.items():
        values = values[keep]
        if key in tags_as_columns and all(v is None for v in values):
            continue
        kept[key] = values
    return kept
