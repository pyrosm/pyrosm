# ========================
# HIGHWAY TAGS
# ========================

# Default tags to keep with highways
# Mostly based on: https://wiki.openstreetmap.org/wiki/Key:highway
highway_tags_to_keep = ["access",
                        "area",
                        "bicycle",
                        "bicycle_road",
                        "bridge",
                        "busway",
                        "cycleway",
                        "est_width",
                        "foot",
                        "footway",
                        "highway",
                        "int_ref",
                        "junction",
                        "lanes",
                        "lit",
                        "maxspeed",
                        "motorcar",
                        "motorroad",
                        "motor_vehicle",
                        "name",
                        "oneway",
                        "overtaking",
                        "path",
                        "passing_places",
                        "psv",
                        "ref",
                        "service",
                        "segregated",
                        "sidewalk",
                        "smoothness",
                        "surface",
                        "tracktype",
                        "tunnel",
                        "turn",
                        "width",
                        "winter_road",

                        # Other highway tags which are not kept by default
                        # (more tags slows down the parsing)
                        # =====================================

                        # "abutters",
                        # "driving_side",
                        # "embedded_rails",
                        # "ford",
                        # "ice_road",
                        # "incline",
                        # "mtb:scale",
                        # "mtb:scale:uphill",
                        # "mtb:scale:imba",
                        # "mtb:description",
                        # "parking:condition",
                        # "parking:lane",
                        # "sac_scale",
                        # "tactile_paving",
                        # "traffic_calming",
                        # "trail_visibility",

                        ]

# ========================
# BUILDING / AMENITY TAGS
# ========================
# See:
# https://wiki.openstreetmap.org/wiki/Key:building
# https://wiki.openstreetmap.org/wiki/Key:addr

building_tags_to_keep = ['building',
                         'addr:city',
                         'addr:country',
                         'addr:full',
                         'addr:housenumber',
                         'addr:housename',
                         'addr:postcode',
                         'addr:place',
                         'addr:province',
                         'addr:state',
                         'addr:street',
                         'amenity',
                         'building:flats',
                         'building:levels',
                         'building:material',
                         'building:max_level',
                         'building:min_level',
                         'building:fireproof',
                         'building:use',
                         'craft',
                         'email',
                         'height',
                         'internet_access',
                         'landuse',
                         'levels',
                         'mml:class'
                         'name',
                         'office',
                         'opening_hours',
                         'operator',
                         'phone',
                         'ref',
                         'shop',
                         'soft_storey',
                         'source',
                         'start_date',
                         'takeaway',
                         'url',
                         'website',
                         'wikipedia',

                         # Other tags that are not kept by default
                         # 'addr:conscriptionnumber',
                         # 'addr:district',

                         ]
