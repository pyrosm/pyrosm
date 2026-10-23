"""Which closed ways are areas: the OSM wiki's list of polygon features
(https://wiki.openstreetmap.org/wiki/Overpass_turbo/Polygon_Features), as OSMnx uses it.
"""

ALL = "all"
PASSLIST = "passlist"
BLOCKLIST = "blocklist"

# Per key: every value makes a closed way an area (ALL), only the listed values do
# (PASSLIST), or every value but the listed ones does (BLOCKLIST).
POLYGON_FEATURES = {
    "aeroway": (BLOCKLIST, frozenset({"taxiway"})),
    "amenity": (ALL, frozenset()),
    "area": (ALL, frozenset()),
    "area:highway": (ALL, frozenset()),
    "barrier": (
        PASSLIST,
        frozenset({"city_wall", "ditch", "hedge", "retaining_wall", "spikes"}),
    ),
    "boundary": (ALL, frozenset()),
    "building": (ALL, frozenset()),
    "building:part": (ALL, frozenset()),
    "craft": (ALL, frozenset()),
    "golf": (ALL, frozenset()),
    "highway": (PASSLIST, frozenset({"elevator", "escape", "rest_area", "services"})),
    "historic": (ALL, frozenset()),
    "indoor": (ALL, frozenset()),
    "landuse": (ALL, frozenset()),
    "leisure": (ALL, frozenset()),
    "man_made": (BLOCKLIST, frozenset({"cutline", "embankment", "pipeline"})),
    "military": (ALL, frozenset()),
    "natural": (
        BLOCKLIST,
        frozenset({"arete", "cliff", "coastline", "ridge", "tree_row"}),
    ),
    "office": (ALL, frozenset()),
    "place": (ALL, frozenset()),
    "power": (
        PASSLIST,
        frozenset({"generator", "plant", "substation", "transformer"}),
    ),
    "public_transport": (ALL, frozenset()),
    "railway": (
        PASSLIST,
        frozenset({"platform", "roundhouse", "station", "turntable"}),
    ),
    "ruins": (ALL, frozenset()),
    "shop": (ALL, frozenset()),
    "tourism": (ALL, frozenset()),
    "waterway": (PASSLIST, frozenset({"boatyard", "dam", "dock", "riverbank"})),
}


def is_area(tags):
    """Whether a closed way with the tags ``tags`` (a mapping of key to value) is an area:
    not when it is tagged ``area=no``, otherwise when one of its tags is a polygon feature
    of :data:`POLYGON_FEATURES`."""
    if tags.get("area") == "no":
        return False
    for key, value in tags.items():
        feature = POLYGON_FEATURES.get(key)
        if feature is None:
            continue
        rule, values = feature
        if (
            rule == ALL
            or (rule == PASSLIST and value in values)
            or (rule == BLOCKLIST and value not in values)
        ):
            return True
    return False
