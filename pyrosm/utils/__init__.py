from shapely import ops
from shapely.geometry import MultiLineString, \
    Polygon, MultiPolygon


def validate_custom_filter(custom_filter):
    # Check that the custom filter is in correct format
    if not isinstance(custom_filter, dict):
        raise ValueError(f"'custom_filter' should be a Python dictionary. "
                         f"Got {custom_filter} with type {type(custom_filter)}.")

    for k, v in custom_filter.items():
        if not isinstance(k, str):
            raise ValueError(f"OSM key in 'custom_filter' should be string. "
                             f"Got {k} of type {type(k)}")
        if v is True:
            continue

        if not isinstance(v, list):
            raise ValueError(f"OSM tags in 'custom_filter' should be inside a list. "
                             f"Got {v} of type {type(v)}")

        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"OSM tag (value) in 'custom_filter' should be string. "
                                 f"Got {item} of type {type(item)}")


def validate_osm_keys(osm_keys):
    if osm_keys is not None:
        if type(osm_keys) not in [str, list]:
            raise ValueError(f"'osm_keys_to_keep' -parameter should be of type str or list. "
                             f"Got {osm_keys} of type {type(osm_keys)}.")


def validate_tags_as_columns(tags_as_columns):
    if not isinstance(tags_as_columns, list):
        raise ValueError(f"'tags_as_columns' should be a list. "
                         f"Got {tags_as_columns} of type {type(tags_as_columns)}.")
    for col in tags_as_columns:
        if not isinstance(col, str):
            raise ValueError(f"All tags listed in 'tags_as_columns' should be strings. "
                             f"Got {col} of type {type(col)}.")


def validate_booleans(keep_nodes, keep_ways, keep_relations):
    if not isinstance(keep_nodes, bool):
        raise ValueError("'keep_nodes' should be boolean type: True or False")

    if not isinstance(keep_ways, bool):
        raise ValueError("'keep_ways' should be boolean type: True or False")

    if not isinstance(keep_relations, bool):
        raise ValueError("'keep_relations' should be boolean type: True or False")

    if keep_nodes is False and keep_ways is False and keep_relations is False:
        raise ValueError("At least on of the following parameters should be True: "
                         "'keep_nodes', 'keep_ways', or 'keep_relations'")


def validate_boundary_type(boundary_type):
    allowed_boundary_types = ["administrative", "national_park", "political",
                              "postal_code", "protected_area", "aboriginal_lands",
                              "maritime", "marker",
                              # There is no consensus whether allowing the following ones should be done
                              # but as they exist, allow using them here as well.
                              # https://wiki.openstreetmap.org/wiki/Parcel
                              "lot", "parcel", "tract",
                              "all"]
    allowed_text = ", ".join(allowed_boundary_types)
    if not isinstance(boundary_type, str):
        raise ValueError(f"'boundary_type' should be one of the following: {allowed_text}."
                         f"Got '{boundary_type}' of type {type(boundary_type)}.")

    boundary_type = boundary_type.strip().lower()
    if boundary_type not in allowed_boundary_types:
        raise ValueError(f"'boundary_type' should be one of the following: {allowed_text}."
                         f"Got '{boundary_type}' of type {type(boundary_type)}.")
    return boundary_type


def validate_bounding_box(geom):
    if type(geom) in [Polygon, MultiPolygon]:
        return geom

    elif isinstance(geom, MultiLineString):
        geom = ops.linemerge(geom)

    if not geom.is_closed:
        raise ValueError(
            "Provided bounding box is not a closed geometry. "
            "Ensure that you pass a Polygon or LinearRing."
        )
    return Polygon(geom)

