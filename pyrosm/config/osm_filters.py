def get_osm_filter(network_type):
    """
    Get OSM filter for different kinds of network types: 'driving', 'walking', 'cycling'.

    Applies same filters as in OSMnx,
    see: https://github.com/gboeing/osmnx/blob/master/osmnx/downloader.py#L19

    """
    if network_type == "driving":
        return driving_filter()
    elif network_type == "walking":
        return walking_filter()
    elif network_type == "cycling":
        return cycling_filter()
    elif network_type == "driving+psv":
        return driving_filter(exclude_public_service_vehicle_paths=False)


def driving_filter(exclude_public_service_vehicle_paths=True):
    """
    Driving filters for different tags (almost) as in OSMnx for 'drive+service'.

    Filter out un-drivable roads, private ways, and
    anything specifying motor=no. also filter out any non-service roads that
    are tagged as providing parking, private, or emergency-access
    services.

    If 'exclude_public_service_vehicle_paths' == False, also paths that are only accessible
    for public transport vehicles are included.

    Applied filters:
        '["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|'
        'elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
        '["motor_vehicle"!~"no"]["motorcar"!~"no"]{}'
        '["service"!~"parking|parking_aisle|private|emergency_access"]'

    """
    drive_filter = dict(
        area=["yes"],
        highway=[
            "cycleway",
            "footway",
            "path",
            "pedestrian",
            "steps",
            "track",
            "corridor",
            "elevator",
            "escalator",
            "proposed",
            "construction",
            "bridleway",
            "abandoned",
            "platform",
            "raceway",
        ],
        motor_vehicle=["no"],
        motorcar=["no"],
        service=["parking", "parking_aisle", "private", "emergency_access"],
    )

    if exclude_public_service_vehicle_paths:
        drive_filter["psv"] = ["yes"]

    return drive_filter


def walking_filter():
    """
    Walking filters for different tags as in OSMnx for 'walk'.

    Filter out cycle ways, motor ways, private ways, and anything
    specifying foot=no. allow service roads, permitting things like parking
    lot lanes, alleys, etc that you *can* walk on even if they're not exactly
    pleasant walks. some cycleways may allow pedestrians, but this filter ignores
    such cycleways.

    Applied filters:
    '["area"!~"yes"]["highway"!~"cycleway|motor|proposed|construction|abandoned|
      platform|raceway|motorway|motorway_link"]'

    '["foot"!~"no"]["service"!~"private"]'


    """
    return dict(
        area=["yes"],
        highway=[
            "cycleway",
            "motor",
            "proposed",
            "construction",
            "abandoned",
            "platform",
            "raceway",
            "motorway",
            "motorway_link",
        ],
        foot=["no"],
        service=["private"],
    )


def cycling_filter():
    """
    Cycling filters for different tags as in OSMnx for 'bike'.

    Filter out foot ways, motor ways, private ways, and anything
    specifying biking=no.

    Applied filters:
        '["area"!~"yes"]["highway"!~"footway|steps|corridor|elevator|escalator|motor|proposed|'
        'construction|abandoned|platform|raceway"]'
        '["bicycle"!~"no"]["service"!~"private"

    """
    return dict(
        area=["yes"],
        highway=[
            "footway",
            "steps",
            "corridor",
            "elevator",
            "escalator",
            "motor",
            "proposed",
            "construction",
            "abandoned",
            "platform",
            "raceway",
            "motorway",
            "motorway_link",
        ],
        bicycle=["no"],
        service=["private"],
    )
