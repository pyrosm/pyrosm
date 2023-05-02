import numpy as np
from shapely import linestrings, polygons, points, linearrings, \
    multilinestrings, multipolygons, get_geometry
from shapely import Geometry
from shapely import GEOSException
from shapely.linear import line_merge
from shapely.coordinates import get_coordinates
from shapely.predicates import is_geometry
from shapely.geometry import MultiPolygon
from shapely.ops import polygonize
from pyrosm.distance import Unit, haversine


cpdef fix_geometry(geometry, diff_threshold=20):
    """
    Fix for invalid geometries using two strategies:
        1. buffer(0) --> works in most cases.
        2. bowtie fix --> used if buffer breaks down the geometry
        3. If the difference is still huge between the
        original geometry and fix-candidate, returns
        invalid geometry
    """
    # Then try fixing with buffer
    fix_candidate = geometry.buffer(0)
    if fix_candidate.is_valid:
        # Ensure that the area of the geometry
        # hasn't changed dramatically
        # Sometimes taking buffer 0 totally breaks down
        # the original geometry having hundred/thousand-fold
        # difference in area
        try:
            diff = abs(1 - geometry.area / fix_candidate.area)
            if diff < diff_threshold:
                return fix_candidate
        except ZeroDivisionError:
            pass
        except Exception as e:
            raise e

    # If geometry is MultiPolygon do not try fix bowtie
    if isinstance(geometry, MultiPolygon):
        return geometry

    # Try fixing "bowtie" geometry
    ext = geometry.exterior
    mls = ext.intersection(ext)
    polys = polygonize(mls)
    fix_candidate = MultiPolygon(polys)
    if fix_candidate.is_valid:
        try:
            diff = abs(1 - geometry.area / fix_candidate.area)
            if diff < diff_threshold:
                return fix_candidate
        except ZeroDivisionError:
            pass
        except Exception as e:
            raise e
    # Otherwise return original geometry
    return geometry


cdef get_way_coordinates_for_polygon(node_coordinate_lookup, way_elements):
    cdef int i, ii, nn, n = len(way_elements["id"])
    features = []
    for i in range(0, n):
        way_nodes = way_elements["nodes"][i]
        nn = len(way_nodes)
        coords = []
        for ii in range(0, nn):
            node = way_nodes[ii]
            try:
                # Ensure lat/lon values are valid
                lon = node_coordinate_lookup[node]["lon"]
                lat = node_coordinate_lookup[node]["lat"]

                if is_valid_coordinate_pair(lat, lon):
                    coords.append((lon, lat))
            except:
                pass
        features.append(coords)
    return features


cdef create_linear_ring(coordinates):
    try:
        return linearrings(coordinates)
    # pygeos 0.7.1 throws GEOSException
    except GEOSException as e:
        if "Invalid number of points" in str(e):
            return None
        elif "point array must contain" in str(e):
            return None
        raise e
    # pygeos 0.8.0 throws ValueError
    except ValueError as e:
        if "Provide at least 4 coordinates" in str(e):
            return None
    except Exception as e:
        raise e


cdef create_linestring(coordinates):
    try:
        return linestrings(coordinates)
    except GEOSException as e:
        if "Invalid number of points" in str(e):
            return None
        elif "point array must contain" in str(e):
            return None
        raise e
    except ValueError as e:
        if "Provide at least 2 coordinates" in str(e):
            return None
        if "not have enough dimensions" in str(e):
            raise e
    except Exception as e:
        raise e


cdef _create_point_geometries(xarray, yarray):
    cdef:
        int N = len(xarray)
        float x, y

    geometries = []
    for i in range(0, N):
        coords = (xarray[i], yarray[i])
        geometries.append(coords)

    return [
        points(geom) if geom is not None 
        else None for geom in geometries
    ]

cdef create_relation_geometry(node_coordinates, ways,
                              member_roles, force_linestring,
                              make_multipolygon):
    cdef int i, m_cnt
    cdef str role
    # Get coordinates for relation
    coordinates = get_way_coordinates_for_polygon(node_coordinates, ways)

    shell = []
    holes = []
    m_cnt = len(member_roles)

    for i in range(0, m_cnt):
        role = member_roles[i]
        coords = coordinates[i]

        # Points are skipped
        if len(coords) < 2:
            continue

        # In case element should constitute a multipolygon,
        # geometries having less than 3 coordinates should be
        # skipped as it is not possible to create a polygon
        if make_multipolygon:
            if len(coords) < 3:
                continue

        geometry = create_linestring(coords)

        if geometry is None:
            continue

        if role == "inner":
            holes.append(geometry)
        else:
            shell.append(geometry)

    if len(shell) == 0:
        if len(holes) == 0:
            return None
        # If shell wasn't found at all, but holes were,
        # use the holes to construct the geometry
        # (might happen sometimes with incorrect tagging)
        else:
            shell = holes
            holes = []

    # Check if should build a LineString
    # e.g. routes should be linestrings
    if force_linestring:
        geoms = shell + holes

        if len(geoms) == 1:
            return geoms
        else:
            geom = line_merge(multilinestrings(geoms))

            if isinstance(geom, np.ndarray):
                return geom.tolist()
            else:
                return [geom]

    if len(holes) == 0:
        holes = None

    # Ensure holes are valid LinearRings
    else:
        # Parse rings
        rings = []
        for hole in holes:
            # In some cases, there are insufficient number
            # of coordinates for constructing LinearRing
            ring = create_linear_ring(get_coordinates(hole))
            if ring is not None:
                rings.append(ring)
        holes = rings
        if len(holes) == 0:
            holes = None

    if len(shell) > 1:
        if not make_multipolygon:
            ring = create_linear_ring(
                get_coordinates(
                    line_merge(multilinestrings(shell))
                ))
            if ring is None:
                return None
            geom = polygons(ring, holes)
        else:
            # Parse rings
            rings = []
            for part in shell:
                # In some cases, there are insufficient number
                # of coordinates for constructing LinearRing
                ring = create_linear_ring(get_coordinates(part))
                if ring is not None:
                    rings.append(ring)

            if len(rings) == 0:
                return None

            if len(rings) > 1:
                geom = multipolygons(polygons(rings, holes))
            else:
                geom = polygons(rings, holes)

    else:
        ring = create_linear_ring(get_coordinates(shell))
        if ring is None:
            return None

        geom = polygons(ring,
                        holes)

    if isinstance(geom, np.ndarray):
        return geom.tolist()
    elif is_geometry(geom):
        return [geom]
    else:
        # TODO: Remove this if no errors arise
        raise NotImplementedError(
            "'create_relation_geometry': "
            "not a geometry or ndarray.\n"
            "Raise an issue at: "
            "https://github.com/HTenkanen/pyrosm/issues"
        )


cpdef create_point_geometries(xarray, yarray):
    return _create_point_geometries(xarray, yarray)


cdef is_valid_coordinate_pair(lat, lon):
    if lon > 180 or lon < -180:
        return False
    if lat > 90 or lat < -90:
        return False
    return True


cdef create_linestring_geometry(nodes, node_coordinates):
    coords = []
    kept_nodes = []
    node_data = []
    cdef int i, n = len(nodes)
    for i in range(0, n):
        node = nodes[i]
        try:
            data = node_coordinates[node]
            # Ensure coordinates are valid
            lon = data["lon"]
            lat = data["lat"]

            if is_valid_coordinate_pair(lat, lon):
                coords.append([(lon, lat)])
                kept_nodes.append(node)
                data["id"] = node
                node_data.append(data)
        except:
            pass

    if len(coords) > 1:
        try:
            # Each geom segment should be constructed separately
            # (i.e. becomes a multilinestring)
            coords = np.array(coords, dtype=np.float64)
            coords = np.hstack([coords[:-1], coords[1:]])
            coord_cnt = len(coords)
            # Get an array of linestrings (segments of the way geometry)
            geom = linestrings(coords)

            # Get from and to-ids
            from_ids = kept_nodes[:-1]
            to_ids = kept_nodes[1:]

            return geom, from_ids, to_ids, node_data
        except GEOSException as e:
            if "Invalid number of points" in str(e):
                # node_data should always be a list
                return None, None, None, []
            else:
                raise e
        except Exception as e:
            raise e

    else:
        # node_data should always be a list
        return None, None, None, []


cdef create_polygon_geometry(nodes, node_coordinates):
    cdef int i, n = len(nodes)
    coords = []
    for i in range(0, n):
        node = nodes[i]
        try:
            # Ensure lat/lon values are valid
            lon = node_coordinates[node]["lon"]
            lat = node_coordinates[node]["lat"]
            if is_valid_coordinate_pair(lat, lon):
                coords.append((lon, lat))
        except:
            pass

    if len(coords) > 2:
        try:
            return polygons(coords)
        except GEOSException as e:
            # Some geometries might not be valid for creating a Polygon
            # These might occur e.g. at the edge of the spatial extent
            if "Invalid number of points in LinearRing" in str(e):
                return None
            else:
                raise e
        # pygeos 0.8.0 throws ValueError
        except ValueError as e:
            if "Provide at least 4 coordinates" in str(e):
                return None
        except Exception as e:
            raise e
    else:
        return None


cdef _create_way_geometries(node_coordinates,
                            way_elements,
                            parse_network):
    # Info for constructing geometries:
    # https://wiki.openstreetmap.org/wiki/Way

    # 'parse_network' determines whether the way is part of a network (and not e.g. a building)
    # if true, the length of the way (or its segments) will be calculated

    cdef long long node
    cdef list coords
    cdef int n, i
    n = len(way_elements['id'])
    keys = list(way_elements.keys())

    # Containers for geoms and node-ids
    geometries = []
    from_ids, to_ids = [], []
    node_attributes = []
    parsed_way_indices = []

    for i in range(0, n):
        nodes = way_elements['nodes'][i]
        # In some cases (e.g. when using clipped pbf file) the nodes list can be empty
        if len(nodes) == 0:
            continue
        u = nodes[0]
        v = nodes[-1]

        # If first and last node are the same, it's a closed way
        if  u == v:
            tag_keys = way_elements.keys()
            # Create Polygon by default unless way is of type 'highway', 'barrier' or 'route'
            if "highway" in tag_keys or "barrier" in tag_keys or "route" in tag_keys:
                geom, from_id, to_id, node_data = create_linestring_geometry(nodes, node_coordinates)
            else:
                geom = create_polygon_geometry(nodes, node_coordinates)

        # Otherwise create LineString
        else:
            geom, from_id, to_id, node_data = create_linestring_geometry(nodes, node_coordinates)

        if parse_network:
            from_ids.append(from_id)
            to_ids.append(to_id)
            node_attributes += node_data
            # Geometries should be an array of LineStrings at this point
            geometries.append(geom)

        # In case e.g. amenities have line features,
        # ensure that geometry is in correct form
        else:
            if isinstance(geom, Geometry):
                geometries.append(geom)
            elif geom is None:
                geometries.append(geom)
            else:
                # LineStrings are in an array
                if len(geom) == 1:
                    geometries.append(geom[0])
                else:
                    geometries.append(multilinestrings(geom))

        # Add index
        parsed_way_indices.append(i)

    # Select valid ways
    for key in keys:
        way_elements[key] = way_elements[key][parsed_way_indices]

    return way_elements, geometries, from_ids, to_ids, node_attributes


cpdef create_way_geometries(node_coordinates, way_elements, parse_network):
    return _create_way_geometries(node_coordinates, way_elements, parse_network)
