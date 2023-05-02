from enum import Enum
import numpy as np
from shapely.coordinates import get_coordinates


class Unit(Enum):
    """
    Enumeration of supported units.
    The full list can be checked by iterating over the class; e.g.
    the expression `tuple(Unit)`.

    Inspired by: https://github.com/mapado/haversine

    """

    KILOMETERS = "km"
    METERS = "m"
    MILES = "mi"
    NAUTICAL_MILES = "nmi"
    FEET = "ft"
    INCHES = "in"


# mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
_AVG_EARTH_RADIUS_KM = 6371.0088

# Unit values taken from http://www.unitconversion.org/unit_converter/length.html
_CONVERSIONS = {
    Unit.KILOMETERS: 1.0,
    Unit.METERS: 1000.0,
    Unit.MILES: 0.621371192,
    Unit.NAUTICAL_MILES: 0.539956803,
    Unit.FEET: 3280.839895013,
    Unit.INCHES: 39370.078740158,
}


def haversine(lat1, lng1, lat2, lng2, unit=Unit.KILOMETERS):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.

    Inspired by: https://github.com/mapado/haversine

    Parameters
    ==========
    lat1 : np.array
        Latitude in decimal degrees
    lng1 : np.array
        Longitude in decimal degrees
    lat2 : np.array
        Latitude in decimal degrees
    lng2 : np.arrays
        Longitude in decimal degrees
    unit : pyrosm.distance.Unit
        A member of pyrosm.distance.Unit, or, equivalently, a string containing the
        initials of its corresponding unit of measurement (i.e. miles = mi)
        default 'km' (kilometers).
    """

    # get earth radius in required units
    unit = Unit(unit)
    avg_earth_radius = _AVG_EARTH_RADIUS_KM * _CONVERSIONS[unit]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.deg2rad, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    return 2 * avg_earth_radius * np.arcsin(np.sqrt(d))


def calculate_geom_length(geom):
    return calculate_geom_array_length(geom).sum().round(0)


def calculate_geom_array_length(geom_array):
    coords = get_coordinates(geom_array).T

    # Only every second element should be taken from the coordinates
    lon1, lat1 = coords[0][:-1:2], coords[1][:-1:2]
    lon2, lat2 = coords[0][1::2], coords[1][1::2]

    # Length of the segments
    geom_lengths = haversine(lat1, lon1, lat2, lon2, unit=Unit.METERS).round(3)
    return geom_lengths
