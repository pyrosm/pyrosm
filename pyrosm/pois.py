from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter
import warnings


def poi_relation_filter(custom_filter):
    """The effective relation ``custom_filter`` a POI read applies -- the
    amenity/shop/tourism default when none is given, validated/normalised the same way
    ``get_poi_data`` does. Shared with ``read_tiled`` so it can scope relation
    completion to the same relations this layer keeps."""
    if custom_filter is None:
        custom_filter = {"amenity": True, "shop": True, "tourism": True}
    return validate_custom_filter(custom_filter)


def get_poi_data(
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
        keep_metadata=keep_metadata,
        relation_member_ways=relation_member_ways,
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
        keep_metadata=keep_metadata,
        complete_relations=complete_relations,
    )

    return gdf
