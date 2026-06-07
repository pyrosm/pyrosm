"""Coverage of the input validators in pyrosm.utils (#272).

These assert the error branches (bad argument types / values) that the parsing
APIs rely on, so the validators are exercised directly rather than only through
the happy path.
"""

import pytest
from shapely.geometry import Polygon, MultiLineString, LineString, box
from pyrosm.utils import (
    validate_custom_filter,
    validate_osm_keys,
    validate_tags_as_columns,
    validate_boundary_type,
    validate_bounding_box,
    validate_input_file,
    validate_graph_type,
)


def test_validate_custom_filter_rejects_non_dict():
    with pytest.raises(ValueError, match="dictionary"):
        validate_custom_filter(["building"])


def test_validate_custom_filter_rejects_non_string_key():
    with pytest.raises(ValueError, match="string"):
        validate_custom_filter({1: ["yes"]})


def test_validate_custom_filter_rejects_non_list_value():
    with pytest.raises(ValueError, match="inside a list"):
        validate_custom_filter({"amenity": "restaurant"})


def test_validate_custom_filter_rejects_non_string_value_item():
    with pytest.raises(ValueError, match="should be string"):
        validate_custom_filter({"amenity": [123]})


def test_validate_custom_filter_coerces_true_value():
    # True means "keep all values for this key" -> coerced to [True]
    out = validate_custom_filter({"building": True})
    assert out["building"] == [True]


def test_validate_osm_keys_rejects_bad_type():
    with pytest.raises(ValueError, match="str or list"):
        validate_osm_keys(123)


def test_validate_osm_keys_accepts_none_str_list():
    validate_osm_keys(None)
    validate_osm_keys("highway")
    validate_osm_keys(["highway", "building"])


def test_validate_tags_as_columns_rejects_non_list():
    with pytest.raises(ValueError, match="should be a list"):
        validate_tags_as_columns("highway")


def test_validate_tags_as_columns_rejects_non_string_element():
    with pytest.raises(ValueError, match="should be strings"):
        validate_tags_as_columns(["highway", 5])


def test_validate_boundary_type_rejects_non_string():
    with pytest.raises(ValueError, match="should be one of"):
        validate_boundary_type(123)


def test_validate_boundary_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="should be one of"):
        validate_boundary_type("not_a_boundary_type")


def test_validate_boundary_type_normalizes_known_value():
    assert validate_boundary_type("  Administrative ") == "administrative"


def test_validate_bounding_box_rejects_open_geometry():
    with pytest.raises(ValueError, match="not a closed geometry"):
        validate_bounding_box(LineString([(0, 0), (1, 0), (1, 1)]))


def test_validate_bounding_box_accepts_polygon_and_multilinestring():
    poly = box(0, 0, 1, 1)
    assert validate_bounding_box(poly) is poly
    # A closed MultiLineString is merged and polygonized
    ring = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 0)]])
    assert isinstance(validate_bounding_box(ring), Polygon)


def test_validate_input_file_rejects_non_string():
    with pytest.raises(ValueError, match="should be a string"):
        validate_input_file(123)


def test_validate_input_file_rejects_non_pbf():
    with pytest.raises(ValueError, match="Protobuf"):
        validate_input_file("data.txt")


def test_validate_input_file_rejects_missing_file():
    with pytest.raises(ValueError, match="does not exist"):
        validate_input_file("/nonexistent/path/data.osm.pbf")


def test_validate_graph_type_rejects_non_string():
    with pytest.raises(ValueError, match="should be a string"):
        validate_graph_type(123)


def test_validate_graph_type_rejects_unknown():
    with pytest.raises(ValueError, match="pandarm"):
        validate_graph_type("bogus")
