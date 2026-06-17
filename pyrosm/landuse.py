from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter
import warnings


def landuse_relation_filter(custom_filter):
    """The effective relation ``custom_filter`` a landuse read applies (ensuring a
    ``landuse`` key). Shared with ``read_tiled`` so it can scope relation completion
    to the same relations this layer keeps."""
    if custom_filter is None:
        return {"landuse": [True]}
    custom_filter = validate_custom_filter(custom_filter)
    if "landuse" not in custom_filter.keys():
        custom_filter = {**custom_filter, "landuse": [True]}
    return custom_filter


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
    # Default/validate the filter (ensures a "landuse" key).
    custom_filter = landuse_relation_filter(custom_filter)

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
        warnings.warn(
            "Could not find any landuse elements for given area.",
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
