from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
import warnings


def get_user_defined_data(
    nodes,
    node_coordinates,
    way_records,
    relations,
    tags_as_columns,
    custom_filter,
    osm_keys,
    filter_type,
    keep_nodes,
    keep_ways,
    keep_relations,
    bounding_box,
):
    if not keep_nodes:
        nodes = None

    # If wanting to parse relations but not ways,
    # it is still necessary to parse ways as well at this point
    if keep_ways is False and keep_relations is True:
        pass
    # If ways are not wanted, neither should relations be parsed
    elif not keep_ways:
        way_records = None
        relations = None

    if not keep_relations:
        relations = None

    # Call signature for fetching POIs
    nodes, ways, relation_ways, relations = get_osm_data(
        node_arrays=nodes,
        way_records=way_records,
        relations=relations,
        tags_as_columns=tags_as_columns,
        data_filter=custom_filter,
        filter_type=filter_type,
        osm_keys=osm_keys,
    )

    # If there weren't any data, return empty GeoDataFrame
    if nodes is None and ways is None and relations is None:
        warnings.warn(
            "Could not find any OSM data for given area.", UserWarning, stacklevel=2
        )
        return None

    # Ensure that ways are None if returning those are not requested
    if not keep_ways:
        ways = None

    # Prepare GeoDataFrame
    gdf = prepare_geodataframe(
        nodes,
        node_coordinates,
        ways,
        relations,
        relation_ways,
        tags_as_columns,
        bounding_box,
    )

    return gdf
