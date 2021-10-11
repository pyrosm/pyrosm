from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter
import warnings


def get_poi_data(
    nodes,
    node_coordinates,
    way_records,
    relations,
    tags_as_columns,
    custom_filter,
    bounding_box,
):
    # Validate filter
    custom_filter = validate_custom_filter(custom_filter)

    # Call signature for fetching POIs
    nodes, ways, relation_ways, relations = get_osm_data(
        node_arrays=nodes,
        way_records=way_records,
        relations=relations,
        tags_as_columns=tags_as_columns,
        data_filter=custom_filter,
        filter_type="keep",
        osm_keys=None,
    )

    # If there weren't any data, return empty GeoDataFrame
    if nodes is None and ways is None and relations is None:
        warnings.warn(
            "Could not find any POIs for given area.", UserWarning, stacklevel=2
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
    )

    return gdf
