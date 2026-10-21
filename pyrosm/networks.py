from pyrosm.utils import warn_empty_result
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
    :class:`~pyrosm.utils.progress.Steps`, or None)."""
    # Structural columns are always kept; element metadata only when requested.
    tags_as_columns += ["id", "nodes"]
    if keep_metadata:
        tags_as_columns += ["timestamp", "changeset", "version"]

    # Call signature for fetching network data. Predefined networks keep only ways with a
    # 'highway' tag; an advanced custom filter selects ways by its own positive keys instead.
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

    # If there weren't any data, return None
    if ways is None:
        warn_empty_result("network")
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
