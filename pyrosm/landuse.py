from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter, require_layer_key, warn_empty_result


def get_landuse_data(
    nodes,
    node_coordinates,
    way_records,
    relations,
    tags_as_columns,
    custom_filter,
    bounding_box,
    keep_metadata=True,
    relation_member_ways=None,
    complete_relations=False,
):
    """The landuse features as a GeoDataFrame, read from the in-memory reader's parsed arrays;
    ``custom_filter`` narrows them. Returns ``None`` when nothing matches."""
    # If custom_filter has not been defined, initialize with default
    if custom_filter is None:
        custom_filter = {"landuse": [True]}
    else:
        # Check that the custom filter is in correct format, then narrow it to the
        # "landuse" layer, so every element kept is one of the landuse elements.
        custom_filter = validate_custom_filter(custom_filter)
        custom_filter = require_layer_key(custom_filter, "landuse")

    # Call signature for fetching buildings
    nodes, ways, relation_ways, relations = get_osm_data(
        node_arrays=nodes,
        way_records=way_records,
        relations=relations,
        tags_as_columns=tags_as_columns,
        data_filter=custom_filter,
        filter_type="keep",
        osm_keys=None,
        keep_metadata=keep_metadata,
        relation_member_ways=relation_member_ways,
    )

    # If there weren't any data, return empty GeoDataFrame
    if nodes is None and ways is None and relations is None:
        warn_empty_result("landuse")
        return None

    # Prepare GeoDataFrame
    gdf = prepare_geodataframe(
        nodes,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        tags_as_columns,
        bounding_box,
        keep_metadata=keep_metadata,
        complete_relations=complete_relations,
    )
    return gdf
