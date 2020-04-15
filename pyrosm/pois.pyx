from pyrosm.data_manager cimport get_osm_data

cpdef get_poi_data(way_records, relations, tags_to_keep, custom_filter):
    # If custom_filter has not been defined, initialize with default
    if custom_filter is None:
        custom_filter = {"amenity": True,
                         "craft": True,
                         "historic": True,
                         "leisure": True,
                         "shop": True,
                         "tourism": True
                         }
    else:
        # Check that the custom filter is in correct format
        if not isinstance(custom_filter, dict):
            raise ValueError(f"'custom_filter' should be a Python dictionary. "
                         f"Got {custom_filter} with type {type(custom_filter)}.")

    # Call signature for fetching buildings
    return get_osm_data(way_records=way_records,
                        relations=relations,
                        tags_as_columns=tags_to_keep,
                        custom_filter=custom_filter,
                        filter_type="keep"
                        )