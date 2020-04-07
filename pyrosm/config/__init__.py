from pyrosm.config.default_tags import highway_tags_to_keep, amenity_tags_to_keep
from pyrosm.config.osm_filters import get_osm_filter


class NetworkFilter:
    driving = get_osm_filter("driving")
    driving_psv = get_osm_filter("driving+psv")
    walking = get_osm_filter("walking")
    cycling = get_osm_filter("cycling")


class Tags:
    networks = highway_tags_to_keep
    amenities = amenity_tags_to_keep
    buildings = ["TODO"]


class Conf:
    network_filters = NetworkFilter()
    tag_filters = Tags()





