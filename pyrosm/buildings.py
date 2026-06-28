from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter, ensure_filter_key
import warnings


def get_building_data(
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
    # If custom_filter has not been defined, initialize with default
    if custom_filter is None:
        custom_filter = {"building": [True]}
    else:
        # Check that the custom filter is in correct format, then ensure the "building"
        # key is present (an OR term) so buildings are always included.
        custom_filter = validate_custom_filter(custom_filter)
        custom_filter = ensure_filter_key(custom_filter, "building")

    # Call signature for fetching buildings
    nodes, ways, relation_ways, relations = get_osm_data(
        node_arrays=None,
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
        warnings.warn(
            "Could not find any building elements for given area.",
            UserWarning,
            stacklevel=2,
        )
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
