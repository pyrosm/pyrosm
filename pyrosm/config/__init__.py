from pyrosm.config.default_tags import (
    basic_info_tags,
    highway_columns,
    building_columns,
    amenity_columns,
    shop_columns,
    craft_columns,
    leisure_columns,
    tourism_columns,
    historic_columns,
    emergency_columns,
    office_columns,
    landuse_columns,
    natural_columns,
    waterway_columns,
    geological_columns,
    power_columns,
    public_transport_columns,
    railway_columns,
    route_columns,
    place_columns,
    aerialway_columns,
    aeroway_columns,
    boundary_columns,
)
from pyrosm.config.osm_filters import NETWORK_TYPES, get_osm_filter


class NetworkFilter:
    driving = get_osm_filter("driving")
    driving_service = get_osm_filter("driving+service")
    walking = get_osm_filter("walking")
    cycling = get_osm_filter("cycling")
    all = get_osm_filter("all")
    all_public = get_osm_filter("all_public")


class Tags:
    # Tags object contains configuration about the default
    # tag combinations (key:value) that will be kept as columns
    # in the resulting GeoDataFrame. All other possible tags are
    # inserted into a JSON that is stored in "tags" column.
    # These follow more or less OSM Wiki documentation:
    # https://wiki.openstreetmap.org/wiki/Map_Features
    available = [
        "aerialway",
        "aeroway",
        "amenity",
        "boundary",
        "building",
        "craft",
        "emergency",
        "geological",
        "highway",
        "historic",
        "landuse",
        "leisure",
        "natural",
        "office",
        "power",
        "public_transport",
        "railway",
        "route",
        "place",
        "shop",
        "tourism",
        "waterway",
    ]

    aerialway = aerialway_columns
    aeroway = aeroway_columns
    amenity = amenity_columns
    boundary = boundary_columns
    building = building_columns
    craft = craft_columns
    emergency = emergency_columns
    geological = geological_columns
    highway = highway_columns
    historic = historic_columns
    landuse = landuse_columns
    leisure = leisure_columns
    natural = natural_columns
    office = office_columns
    power = power_columns
    public_transport = public_transport_columns
    railway = railway_columns
    route = route_columns
    place = place_columns
    shop = shop_columns
    tourism = tourism_columns
    waterway = waterway_columns

    # Basic info tags
    _basic_tags = basic_info_tags


class Conf:
    network_filters = NetworkFilter()
    tags = Tags()
    _possible_network_filters = NETWORK_TYPES

    # One way tags
    oneway_values = ["yes", "true", "1", "-1", "T", "F"]
