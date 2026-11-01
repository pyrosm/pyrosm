"""Predefined network filters.

Each filter is an "exclude" dict of ``{osm_key: [values]}``: a way is dropped when it
carries one of the listed values for that key. The filters follow the way filters of
OSMnx (``osmnx._overpass._get_network_filter``) closely and in most cases select the
same ways, with the OSMnx regular expressions written out as the OSM values they match
(``"motor"`` matches ``motor``, ``motorway`` and ``motorway_link``). Where the two
differ, the filters follow OSM's access rules: the ``access`` entry of a filter is an
:class:`AccessRule`, so ``access=no`` excludes a way unless a tag of the network's mode
of transport, such as ``foot=yes``, allows the use. A way built for another mode is left
out when it has no tag of the network's mode: the walking filter leaves out a
``highway=cycleway`` without a ``foot`` tag, and the cycling filter a
``highway=footway`` without a ``bicycle`` tag. With such a tag, the access rule decides
as on any other way, so ``foot=designated`` lets a cycle way stay in the walking network
and ``foot=no`` leaves it out. A network read keeps a way without a tag of the network's
mode where the default of the way's country lets the mode use it (see
:func:`country_rule`); ``shared_paths`` leaves all of them out or keeps them all instead
(see :func:`with_shared_paths`). OSMnx leaves all cycle ways out of its 'walk' network
and all foot ways out of its 'bike' network.
"""

import numpy as np


class MultiValue(list):
    """Filter values that match a multi-value tag as well.

    An OSM tag can hold several values separated by a semicolon
    (``access=private;delivery``). OSMnx matches its filters against the raw tag value
    with a regular expression, so such a tag matches when any of its values is listed.
    These values match it the same way. A ``custom_filter`` is unaffected and keeps
    matching values exactly; use a compiled regular expression there to match a
    multi-value tag.
    """

    def __contains__(self, value):
        if list.__contains__(self, value):
            return True
        if isinstance(value, str) and ";" in value:
            return any(
                list.__contains__(self, part.strip()) for part in value.split(";")
            )
        return False


def _parts(value):
    """The values of a tag that holds one or several ``;``-separated values."""
    return [part.strip() for part in value.split(";")]


# The values of an access tag that close a way to the public.
_CLOSED = MultiValue(["no", "private"])

# These and the value that sends the mode to the way mapped beside this one, such as a
# sidewalk or a cycle track.
_CLOSED_OR_SIDEPATH = MultiValue(["no", "private", "use_sidepath"])


class AccessRule:
    """The access tags of a mode of transport, in OSM's access hierarchy.

    ``mode_keys`` are the access tags of the mode from the most specific one
    (``"motorcar", "motor_vehicle", "vehicle"`` for driving), followed by the general
    ``access`` tag. The most specific of them that a way carries decides: the way is
    closed when the value of that tag is one of the values in ``closed`` (``no`` and
    ``private`` by default), also as one of several values (``private;delivery``). In a
    walking network, ``access=no``
    with ``foot=yes`` and ``access=private`` with ``foot=yes`` stay, while ``access=no``
    alone and with ``foot=private`` are excluded. The walking and cycling networks also
    close a way tagged ``use_sidepath``, which sends the mode to the way mapped beside it.

    ``other_mode_highway`` lists the ``highway`` values of ways built for another mode,
    such as ``cycleway`` in a walking network. Such a way is also closed when it has no
    tag of the mode, the first of ``mode_keys`` (``foot`` for walking). With that tag,
    it is closed only when the tag's value is one of the values in ``closed``, as any
    other way is: a cycleway
    with ``foot=designated`` stays in a walking network, and one with ``foot=no`` or
    without a ``foot`` tag is excluded. With ``by_country``, a network read keeps such a
    way without a tag of the mode where its country lets the mode use it (see
    :func:`filter_for_extent` and :func:`country_rule`).
    """

    def __init__(
        self, *mode_keys, other_mode_highway=(), by_country=False, closed=_CLOSED
    ):
        self.keys = mode_keys + ("access",)
        self.other_mode_highway = MultiValue(other_mode_highway)
        self.by_country = by_country
        self.closed = MultiValue(closed)

    def closes(self, tags):
        """Whether ``tags`` close the way. A way whose ``highway`` value is one of
        ``other_mode_highway`` is also closed when it has no tag of the first of
        ``mode_keys``."""
        for key in self.keys:
            value = tags.get(key)
            if value is not None:
                if value in self.closed:
                    return True
                break
        if tags.get("highway") in self.other_mode_highway:
            return tags.get(self.keys[0]) is None
        return False

    def without_other_mode(self):
        """This rule's access hierarchy alone, without ``other_mode_highway``."""
        return AccessRule(*self.keys[:-1], closed=self.closed)

    def without_country(self):
        """This rule with ``by_country`` off: a way built for another mode without a tag of
        the mode is closed, whatever its country."""
        return AccessRule(
            *self.keys[:-1],
            other_mode_highway=self.other_mode_highway,
            closed=self.closed,
        )

    def lacks_mode_tag(self, tags):
        """Whether the way of ``tags`` is built for another mode (by its ``highway`` tag)
        and has no tag of the mode, the first of ``mode_keys``."""
        return tags.get(self.keys[0]) is None and (
            tags.get("highway") in self.other_mode_highway
        )

    def allowed_in(self, highway, lon, lat):
        """Whether the country of each way, by a point of it (``lon``, ``lat`` in
        EPSG:4326), lets the mode use it without a tag of the mode, by the way's
        ``highway`` value (one of ``other_mode_highway``, also among several values; see
        :func:`pyrosm.data.countries.countries_allowing`). ``highway`` may be None when
        ``other_mode_highway`` has one value. A way without a country, or without
        coordinates (NaN), is not allowed."""
        from pyrosm.data.countries import countries_allowing, country_codes

        mode = self.keys[0]
        codes = country_codes(lon, lat)
        if len(self.other_mode_highway) == 1:
            allowed = countries_allowing(mode, self.other_mode_highway[0])
            return np.isin(codes, list(allowed))
        return np.fromiter(
            (
                code is not None
                and any(
                    code in countries_allowing(mode, part)
                    for part in _parts(value)
                    if part in self.other_mode_highway
                )
                for value, code in zip(highway, codes)
            ),
            dtype=bool,
            count=len(codes),
        )

    def __repr__(self):
        return (
            "AccessRule(keys=%r, closed=%r, other_mode_highway=%r, by_country=%r)"
            % (
                self.keys,
                list(self.closed),
                list(self.other_mode_highway),
                self.by_country,
            )
        )


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


def get_osm_filter(network_type, shared_paths="local"):
    """Get the OSM filter for a network type.

    Accepts the pyrosm names ('driving', 'driving+service', 'walking', 'cycling',
    'all', 'all_public') and their OSMnx aliases ('drive', 'drive_service', 'walk',
    'bike'), and raises a ValueError for anything else. ``shared_paths`` sets which ways
    built for another mode the walking and cycling filters keep (see
    :func:`with_shared_paths`).
    """
    try:
        builder = _NETWORK_FILTERS[normalize_network_type(network_type)]
    except KeyError:
        raise ValueError(_UNKNOWN_NETWORK_TYPE) from None
    return with_shared_paths(builder(), shared_paths)


def with_shared_paths(network_filter, shared_paths):
    """``network_filter`` with the ways built for another mode that have no tag of the
    network's mode (a cycle way without a ``foot`` tag in a walking network, a foot way
    without a ``bicycle`` tag in a cycling network) kept as ``shared_paths`` says:
    ``"tagged"`` leaves them out, ``"local"`` keeps them where their country lets the
    mode use them, and ``"all"`` keeps them all. A tag of the mode decides as on any other
    way; ``use_sidepath``, which sends the mode to the way mapped beside it, closes the
    way with ``"tagged"`` and ``"local"`` and not with ``"all"``. Raises a ValueError for
    any other level, and for ``"tagged"`` or ``"all"`` with a filter that has no such
    ways: the driving and 'all' filters, and a ``custom_filter``."""
    if shared_paths not in SHARED_PATHS:
        raise ValueError(_UNKNOWN_SHARED_PATHS)
    if shared_paths == "local":
        return network_filter
    rule = network_filter.get("access") if isinstance(network_filter, dict) else None
    if not (isinstance(rule, AccessRule) and rule.other_mode_highway):
        raise ValueError(_SHARED_PATHS_NETWORKS)
    if shared_paths == "tagged":
        return dict(network_filter, access=rule.without_country())
    return dict(network_filter, access=AccessRule(*rule.keys[:-1]))


def driving_filter(include_service_roads=False):
    """Drivable ways, following OSMnx's 'drive'.

    Filters out un-drivable ways, ways closed to cars (motorcar, motor_vehicle, vehicle
    and access tags, see AccessRule), and service roads. With 'include_service_roads'
    service roads are kept except those providing parking, private or emergency access,
    following OSMnx's 'drive_service'. Unlike OSMnx, it also filters out busways, ladders
    and trails, which cars cannot use.
    """
    excluded_highway = _ALWAYS_EXCLUDED_HIGHWAY + [
        "bridleway",
        "bus_guideway",
        "busway",
        "corridor",
        "cycleway",
        "elevator",
        "escalator",
        "footway",
        "ladder",
        "path",
        "pedestrian",
        "steps",
        "track",
        "trail",
    ]
    excluded_service = ["emergency_access", "parking", "parking_aisle", "private"]

    if not include_service_roads:
        excluded_highway.append("service")
        excluded_service += ["alley", "driveway"]

    return dict(
        area=MultiValue(["yes"]),
        access=AccessRule("motorcar", "motor_vehicle", "vehicle"),
        highway=MultiValue(excluded_highway),
        service=MultiValue(excluded_service),
    )


def walking_filter():
    """Walkable ways, following OSMnx's 'walk'.

    Filters out motorways and the ways that AccessRule closes, including ways with
    ``foot=no``, ``foot=use_sidepath`` (pedestrians use the sidewalk mapped beside the
    way) or ``access=no`` and no ``foot`` tag. A cycle way without a ``foot`` tag is
    filtered out; a network read keeps it where the way's country lets pedestrians use
    cycle ways (see country_rule). With a ``foot`` tag, AccessRule decides as for any
    other way, so the shared foot and cycle paths and the sidewalks mapped as cycle ways,
    tagged ``foot=designated`` or ``foot=yes``, can stay in the network. OSMnx leaves all
    cycle ways out.
    Service roads are kept, permitting things like parking lot lanes and alleys that you
    *can* walk on even if they are not exactly pleasant walks. Ways whose sidewalk is
    mapped as a separate way are excluded, so that the sidewalk is not counted twice.
    """
    return dict(
        area=MultiValue(["yes"]),
        access=AccessRule(
            "foot",
            other_mode_highway=["cycleway"],
            by_country=True,
            closed=_CLOSED_OR_SIDEPATH,
        ),
        highway=MultiValue(
            _ALWAYS_EXCLUDED_HIGHWAY + ["bus_guideway"] + _MOTOR_HIGHWAY
        ),
        service=MultiValue(["private"]),
        sidewalk=MultiValue(["separate"]),
        **{
            "sidewalk:both": MultiValue(["separate"]),
            "sidewalk:left": MultiValue(["separate"]),
            "sidewalk:right": MultiValue(["separate"]),
        },
    )


def cycling_filter():
    """Cyclable ways, following OSMnx's 'bike'.

    Filters out motorways and the ways that AccessRule closes, including ways with
    ``bicycle=no``, ``bicycle=use_sidepath`` (cyclists use the cycle track mapped beside
    the way) or ``access=no`` and no ``bicycle`` or ``vehicle`` tag. A foot
    way without a ``bicycle`` tag is filtered out; a network read keeps it where the
    way's country lets cyclists use foot ways (see country_rule). With a ``bicycle``
    tag, AccessRule decides as for any other way, so ``bicycle=yes`` or
    ``bicycle=dismount`` can keep a foot way in the network and ``bicycle=no`` leaves it
    out. OSMnx leaves all foot ways out.
    """
    return dict(
        area=MultiValue(["yes"]),
        access=AccessRule(
            "bicycle",
            "vehicle",
            other_mode_highway=["footway"],
            by_country=True,
            closed=_CLOSED_OR_SIDEPATH,
        ),
        highway=MultiValue(
            _ALWAYS_EXCLUDED_HIGHWAY
            + ["bus_guideway", "corridor", "elevator", "escalator", "steps"]
            + _MOTOR_HIGHWAY
        ),
        service=MultiValue(["private"]),
    )


def all_filter(public_only=False):
    """Every way of the street network, following OSMnx's 'all'.

    No mode-specific restrictions are applied. With 'public_only' the ways that the
    access tag closes to the public (see AccessRule) are left out as well, following
    OSMnx's 'all_public'.
    """
    network_filter = dict(
        area=MultiValue(["yes"]), highway=MultiValue(_ALWAYS_EXCLUDED_HIGHWAY)
    )
    if public_only:
        network_filter["access"] = AccessRule()
        network_filter["service"] = MultiValue(["private"])
    return network_filter


def _country_rule_of(data_filter, filter_type):
    """The access rule of the ``exclude`` filter ``data_filter`` that leaves the ways built
    for another mode without a tag of the mode to their country (``by_country``), or
    None."""
    rule = data_filter.get("access") if isinstance(data_filter, dict) else None
    if filter_type != "exclude" or not isinstance(rule, AccessRule):
        return None
    if not (rule.other_mode_highway and rule.by_country):
        return None
    return rule


def filter_for_extent(data_filter, filter_type, extent):
    """``data_filter`` for a network read of the area ``extent`` (a shapely geometry in
    EPSG:4326, or None for an unknown area). When its access rule leaves the ways built for
    another mode without a tag of the mode to their country and the countries ``extent``
    touches all let the mode use such ways, the filter with the rule's access hierarchy
    alone, which keeps them; when none of them does, the filter with the rule's
    ``by_country`` off, which leaves them out. Otherwise ``data_filter`` itself, whose ways
    a read decides by their own country (see :func:`country_rule`). So a way that reaches
    beyond ``extent`` follows the decision for ``extent``, and so does a way out at sea
    beyond every country's coastal water."""
    from pyrosm.data.countries import countries_allowing, countries_in

    rule = _country_rule_of(data_filter, filter_type)
    if rule is None or extent is None:
        return data_filter
    allows = [
        countries_allowing(rule.keys[0], highway) for highway in rule.other_mode_highway
    ]
    codes = countries_in(extent)
    if codes and all(code in allowed for code in codes for allowed in allows):
        return dict(data_filter, access=rule.without_other_mode())
    if not any(code in allowed for code in codes for allowed in allows):
        return dict(data_filter, access=rule.without_country())
    return data_filter


def filter_for_read(data_filter, filter_type, filepath, bounding_box=None):
    """``data_filter`` for a network read of ``filepath`` within ``bounding_box`` (see
    :func:`filter_for_extent` and :func:`pyrosm.data.countries.read_extent`). The file's
    header is read only for a filter that leaves ways to their country."""
    if _country_rule_of(data_filter, filter_type) is None:
        return data_filter
    from pyrosm.data.countries import read_extent

    extent = read_extent(filepath, bounding_box)
    return filter_for_extent(data_filter, filter_type, extent)


def country_rule(data_filter, filter_type):
    """``(data_filter, rule)`` for a network read: when the ``exclude`` filter
    ``data_filter`` has an access rule that leaves the ways built for another mode without
    a tag of the mode to their country (``by_country``), the filter with that rule's
    access hierarchy alone and the rule; otherwise ``(data_filter, None)``. The read
    filters with the first and then drops the ways the rule leaves to their country
    (:meth:`AccessRule.lacks_mode_tag`) where the country of their first node with
    coordinates does not let the mode use them (:meth:`AccessRule.allowed_in`)."""
    rule = _country_rule_of(data_filter, filter_type)
    if rule is None:
        return data_filter, None
    return dict(data_filter, access=rule.without_other_mode()), rule


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

# The levels of ``shared_paths``, from the fewest ways built for another mode to all.
SHARED_PATHS = ("tagged", "local", "all")

_UNKNOWN_SHARED_PATHS = "'shared_paths' should be one of the following: " + ", ".join(
    SHARED_PATHS
)
_SHARED_PATHS_NETWORKS = (
    "The shared_paths values 'tagged' and 'all' apply only to the 'walking' and "
    "'cycling' networks read without a custom_filter"
)
