from pyrosm.data_manager import get_osm_data
from pyrosm.frames import prepare_geodataframe
from pyrosm.utils import validate_custom_filter, require_layer_key, warn_empty_result


def get_boundary_data(
    node_coordinates,
    way_records,
    relations,
    tags_as_columns,
    custom_filter,
    boundary_type,
    name,
    bounding_box,
    keep_metadata=True,
    relation_member_ways=None,
    complete_relations=False,
):
    """The boundaries of ``boundary_type`` (``"all"`` for any) as a GeoDataFrame, read from the
    in-memory reader's parsed arrays; ``custom_filter`` narrows them (a filter on ``boundary``
    itself chooses the boundary values in place of ``boundary_type``), and ``name`` keeps the
    boundaries whose name contains that text. Returns ``None`` when nothing matches."""
    if boundary_type == "all":
        boundary_type = True
    else:
        boundary_type = [boundary_type]

    # If custom_filter has not been defined, initialize with default
    if custom_filter is None:
        custom_filter = {"boundary": boundary_type}

    # Validate / normalize the filter (normalizes True -> [True]; compiles advanced forms),
    # then narrow it to the boundaries of boundary_type.
    custom_filter = validate_custom_filter(custom_filter)
    custom_filter = require_layer_key(custom_filter, "boundary", boundary_type)

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
        warn_empty_result("boundaries")
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

    if gdf is None:
        return None

    # Filter by name
    # (use Pandas for filtering, which allows using 'contains' more easily)
    if name is not None:
        if "name" not in gdf.columns:
            raise ValueError(
                "Could not filter by name from given area. "
                "Any of the OSM elements did not have a name tag."
            )
        gdf = gdf.dropna(subset=["name"])
        gdf = gdf.loc[gdf["name"].str.contains(name)].reset_index(drop=True).copy()

    return gdf
