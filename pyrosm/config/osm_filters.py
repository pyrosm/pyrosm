"""Predefined network filters, following the way filters of OSMnx.

Each filter is an "exclude" dict of ``{osm_key: [values]}``: a way is dropped when it
carries one of the listed values for that key. The filters mirror
``osmnx._overpass._get_network_filter``, with the OSMnx regular expressions written out
as the OSM values they match (``"motor"`` matches ``motor``, ``motorway`` and
``motorway_link``).
"""


class MultiValue(list):
    """Filter values that match a multi-value tag as well.

    An OSM tag can hold several values separated by a semicolon
    (``access=private;delivery``). OSMnx matches its filters against the raw tag value
    with a regular expression, so such a tag matches when any of its values is listed.
    These values do the same, which keeps the predefined networks equal to the OSMnx
    ones. A ``custom_filter`` is unaffected and keeps matching values exactly; use a
    compiled regular expression there to match a multi-value tag.
    """

    def __contains__(self, value):
        if list.__contains__(self, value):
            return True
        if isinstance(value, str) and ";" in value:
            return any(
                list.__contains__(self, part.strip()) for part in value.split(";")
            )
        return False


# Excluded by every network type: areas are not routable ways, and these highway values
# are either not a physical way yet (or anymore) or not part of the street network.
_ALWAYS_EXCLUDED_HIGHWAY = [
    "abandoned",
    "construction",
    "no",
    "planned",
    "platform",
    "proposed",
    "raceway",
    "razed",
    "rest_area",
    "services",
]

# Motorized ways, as matched by the "motor" expression of OSMnx.
_MOTOR_HIGHWAY = ["motor", "motorway", "motorway_link"]

# OSMnx spellings accepted for the pyrosm network types.
NETWORK_TYPE_ALIASES = {
    "drive": "driving",
    "drive_service": "driving+service",
    "walk": "walking",
    "bike": "cycling",
}


def normalize_network_type(network_type):
    """Lower-case a network type and resolve an OSMnx alias to the pyrosm name."""
    if not isinstance(network_type, str):
        raise ValueError(_UNKNOWN_NETWORK_TYPE)
    net_type = network_type.lower().strip()
    return NETWORK_TYPE_ALIASES.get(net_type, net_type)


def get_osm_filter(network_type):
    """Get the OSM filter for a network type.

    Accepts the pyrosm names ('driving', 'driving+service', 'walking', 'cycling',
    'all', 'all_public') and their OSMnx aliases ('drive', 'drive_service', 'walk',
    'bike'), and raises a ValueError for anything else.
    """
    try:
        builder = _NETWORK_FILTERS[normalize_network_type(network_type)]
    except KeyError:
        raise ValueError(_UNKNOWN_NETWORK_TYPE) from None
    return builder()


def driving_filter(include_service_roads=False):
    """Drivable ways, as in OSMnx for 'drive'.

    Filters out un-drivable ways, private ways, anything specifying motor=no, and
    service roads. With 'include_service_roads' service roads are kept except those
    providing parking, private or emergency access, as in OSMnx for 'drive_service'.
    """
    excluded_highway = _ALWAYS_EXCLUDED_HIGHWAY + [
        "bridleway",
        "bus_guideway",
        "corridor",
        "cycleway",
        "elevator",
        "escalator",
        "footway",
        "path",
        "pedestrian",
        "steps",
        "track",
    ]
    excluded_service = ["emergency_access", "parking", "parking_aisle", "private"]

    if not include_service_roads:
        excluded_highway.append("service")
        excluded_service += ["alley", "driveway"]

    return dict(
        area=MultiValue(["yes"]),
        access=MultiValue(["private"]),
        highway=MultiValue(excluded_highway),
        motor_vehicle=MultiValue(["no"]),
        motorcar=MultiValue(["no"]),
        service=MultiValue(excluded_service),
    )


def walking_filter():
    """Walkable ways, as in OSMnx for 'walk'.

    Filters out cycle ways, motor ways, private ways, and anything specifying foot=no.
    Service roads are kept, permitting things like parking lot lanes and alleys that you
    *can* walk on even if they are not exactly pleasant walks. Ways whose sidewalk is
    mapped as a separate way are excluded, so that the sidewalk is not counted twice.
    """
    return dict(
        area=MultiValue(["yes"]),
        access=MultiValue(["private"]),
        highway=MultiValue(
            _ALWAYS_EXCLUDED_HIGHWAY + ["bus_guideway", "cycleway"] + _MOTOR_HIGHWAY
        ),
        foot=MultiValue(["no"]),
        service=MultiValue(["private"]),
        sidewalk=MultiValue(["separate"]),
        **{
            "sidewalk:both": MultiValue(["separate"]),
            "sidewalk:left": MultiValue(["separate"]),
            "sidewalk:right": MultiValue(["separate"]),
        },
    )


def cycling_filter():
    """Cyclable ways, as in OSMnx for 'bike'.

    Filters out foot ways, motor ways, private ways, and anything specifying
    bicycle=no.
    """
    return dict(
        area=MultiValue(["yes"]),
        access=MultiValue(["private"]),
        highway=MultiValue(
            _ALWAYS_EXCLUDED_HIGHWAY
            + [
                "bus_guideway",
                "corridor",
                "elevator",
                "escalator",
                "footway",
                "steps",
            ]
            + _MOTOR_HIGHWAY
        ),
        bicycle=MultiValue(["no"]),
        service=MultiValue(["private"]),
    )


def all_filter(public_only=False):
    """Every way of the street network, as in OSMnx for 'all'.

    No mode-specific restrictions are applied. With 'public_only' the privately
    accessible ways are left out as well, as in OSMnx for 'all_public'.
    """
    network_filter = dict(
        area=MultiValue(["yes"]), highway=MultiValue(_ALWAYS_EXCLUDED_HIGHWAY)
    )
    if public_only:
        network_filter["access"] = MultiValue(["private"])
        network_filter["service"] = MultiValue(["private"])
    return network_filter


_NETWORK_FILTERS = {
    "driving": driving_filter,
    "driving+service": lambda: driving_filter(include_service_roads=True),
    "walking": walking_filter,
    "cycling": cycling_filter,
    "all": all_filter,
    "all_public": lambda: all_filter(public_only=True),
}

# Every network type accepted by the reading methods, aliases included.
NETWORK_TYPES = list(_NETWORK_FILTERS) + list(NETWORK_TYPE_ALIASES)

_UNKNOWN_NETWORK_TYPE = "'network_type' should be one of the following: " + ", ".join(
    NETWORK_TYPES
)
