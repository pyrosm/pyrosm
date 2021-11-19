import numpy as np


def test_distance_calculations():
    from pyrosm.distance import Unit, haversine

    # Example line from Null Island to 10'10 --> should be ~1569 km
    lon1, lat1 = np.array(0.0, dtype=np.float64), np.array(0.0, dtype=np.float64)
    lon2, lat2 = np.array(10.0, dtype=np.float64), np.array(10.0, dtype=np.float64)
    correct_distance_km = 1568.52272
    correct_distance_miles = 974.634834
    correct_distance_nautical_miles = 846.93452
    correct_distance_feet = 5146072
    correct_distance_inches = 61752863

    # Test kilometers
    l_km = haversine(lat1, lon1, lat2, lon2, unit=Unit.KILOMETERS)
    assert round(l_km, 5) == correct_distance_km

    # Meters
    l_m = haversine(lat1, lon1, lat2, lon2, unit=Unit.METERS)
    assert round(l_m, 2) == correct_distance_km * 1000

    # Miles
    l_mi = haversine(lat1, lon1, lat2, lon2, unit=Unit.MILES)
    assert round(l_mi, 6) == correct_distance_miles

    # Nautical miles
    l_nmi = haversine(lat1, lon1, lat2, lon2, unit=Unit.NAUTICAL_MILES)
    assert round(l_nmi, 5) == correct_distance_nautical_miles

    # Feet
    l_f = haversine(lat1, lon1, lat2, lon2, unit=Unit.FEET)
    assert round(l_f, 0) == correct_distance_feet

    # Inches
    l_f = haversine(lat1, lon1, lat2, lon2, unit=Unit.INCHES)
    assert round(l_f, 0) == correct_distance_inches
