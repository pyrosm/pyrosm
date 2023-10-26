from shapely import ops
from shapely.geometry import MultiLineString, Polygon, MultiPolygon, box
from pyrosm_proto import BlobHeader, Blob, HeaderBlock
from pyrosm.exceptions import PBFNotImplemented
import zlib
from struct import unpack
import os
import geopandas as gpd
import pandas as pd
import warnings


def validate_custom_filter(custom_filter):
    # Check that the custom filter is in correct format
    if not isinstance(custom_filter, dict):
        raise ValueError(
            f"'custom_filter' should be a Python dictionary. "
            f"Got {custom_filter} with type {type(custom_filter)}."
        )

    for k, v in custom_filter.items():
        if not isinstance(k, str):
            raise ValueError(
                f"OSM key in 'custom_filter' should be string. "
                f"Got {k} of type {type(k)}"
            )
        if v is True:
            custom_filter[k] = [v]
            continue

        if not isinstance(v, list):
            raise ValueError(
                f"OSM tags in 'custom_filter' should be inside a list. "
                f"Got {v} of type {type(v)}"
            )

        for item in v:
            if item is True:
                continue
            if not isinstance(item, str):
                raise ValueError(
                    f"OSM tag (value) in 'custom_filter' should be string. "
                    f"Got {item} of type {type(item)}"
                )
    return custom_filter


def validate_osm_keys(osm_keys):
    if osm_keys is not None:
        if type(osm_keys) not in [str, list]:
            raise ValueError(
                f"'osm_keys_to_keep' -parameter should be of type str or list. "
                f"Got {osm_keys} of type {type(osm_keys)}."
            )


def validate_tags_as_columns(tags_as_columns):
    if not isinstance(tags_as_columns, list):
        raise ValueError(
            f"'tags_as_columns' should be a list. "
            f"Got {tags_as_columns} of type {type(tags_as_columns)}."
        )
    for col in tags_as_columns:
        if not isinstance(col, str):
            raise ValueError(
                f"All tags listed in 'tags_as_columns' should be strings. "
                f"Got {col} of type {type(col)}."
            )


def validate_booleans(keep_nodes, keep_ways, keep_relations):
    if not isinstance(keep_nodes, bool):
        raise ValueError("'keep_nodes' should be boolean type: True or False")

    if not isinstance(keep_ways, bool):
        raise ValueError("'keep_ways' should be boolean type: True or False")

    if not isinstance(keep_relations, bool):
        raise ValueError("'keep_relations' should be boolean type: True or False")

    if keep_nodes is False and keep_ways is False and keep_relations is False:
        raise ValueError(
            "At least on of the following parameters should be True: "
            "'keep_nodes', 'keep_ways', or 'keep_relations'"
        )


def validate_boundary_type(boundary_type):
    allowed_boundary_types = [
        "administrative",
        "national_park",
        "political",
        "postal_code",
        "protected_area",
        "aboriginal_lands",
        "maritime",
        "marker",
        # There is no consensus whether allowing the following ones should be done
        # but as they exist, allow using them here as well.
        # https://wiki.openstreetmap.org/wiki/Parcel
        "lot",
        "parcel",
        "tract",
        "all",
    ]
    allowed_text = ", ".join(allowed_boundary_types)
    if not isinstance(boundary_type, str):
        raise ValueError(
            f"'boundary_type' should be one of the following: {allowed_text}."
            f"Got '{boundary_type}' of type {type(boundary_type)}."
        )

    boundary_type = boundary_type.strip().lower()
    if boundary_type not in allowed_boundary_types:
        raise ValueError(
            f"'boundary_type' should be one of the following: {allowed_text}."
            f"Got '{boundary_type}' of type {type(boundary_type)}."
        )
    return boundary_type


def validate_bounding_box(geom):
    if type(geom) in [Polygon, MultiPolygon]:
        return geom

    elif isinstance(geom, MultiLineString):
        geom = ops.linemerge(geom)

    if not geom.is_closed:
        raise ValueError(
            "Provided bounding box is not a closed geometry. "
            "Ensure that you pass a Polygon or LinearRing."
        )
    return Polygon(geom)


def validate_input_file(filepath):
    if not isinstance(filepath, str):
        raise ValueError("'filepath' should be a string.")
    if not filepath.endswith(".pbf"):
        raise ValueError(
            f"Input data should be in Protobuf format (*.osm.pbf). "
            f"Found: {filepath.split('.')[-1]}"
        )
    if not os.path.exists(filepath):
        raise ValueError(f"File does not exist: " f"Found: {filepath}")
    return filepath


def validate_graph_type(graph_type):
    if not isinstance(graph_type, str):
        raise ValueError("'graph_type' should be a string.")
    graph_type = graph_type.lower()
    if graph_type not in ["networkx", "igraph", "pandana"]:
        raise ValueError(
            f"'graph_type' should be 'networkx', 'igraph', or 'panadana'. "
            f"Got '{graph_type}'."
        )
    return graph_type


def validate_node_gdf(nodes):
    if not isinstance(nodes, gpd.GeoDataFrame):
        raise ValueError(f"'nodes' should be a GeoDataFrame, got '{type(nodes)}'.")
    geom_types = nodes.geometry.geom_type.unique().tolist()
    if len(geom_types) != 1 or geom_types[0] != "Point":
        raise ValueError("'nodes' should contain only 'Point' geometries.")


def validate_edge_gdf(edges):
    if not isinstance(edges, gpd.GeoDataFrame):
        raise ValueError(f"'edges' should be a GeoDataFrame, got '{type(edges)}'.")
    geom_types = edges.geometry.geom_type.unique().tolist()
    for gtype in geom_types:
        if gtype not in ["LineString", "MultiLineString"]:
            raise ValueError(
                "'edges' should contain only 'LineString' or 'MultiLineString' geometries."
            )


def valid_header_block(header_block):
    for feature in header_block.required_features:
        if not (feature in ("OsmSchema-V0.6", "DenseNodes", "HistoricalInformation")):
            raise PBFNotImplemented("Required feature %s not implemented!", feature)
    return True


def get_bounding_box(filepath):
    with open(filepath, "rb") as f:
        # Check that the data stream is valid OSM
        # =======================================

        buf = f.read(4)
        msg_len = unpack("!L", buf)[0]
        msg = BlobHeader()
        msg.ParseFromString(f.read(msg_len))
        blob_header = msg

        msg = Blob()
        msg.ParseFromString(f.read(blob_header.datasize))
        blob_data = zlib.decompress(msg.zlib_data)
        header_block = HeaderBlock()
        header_block.ParseFromString(blob_data)

        # Validate header
        if valid_header_block(header_block):
            # Parse bounding box
            try:
                bb = header_block.bbox.SerializeToDict()
                div = 1000000000
                bbox = box(
                    bb["left"] / div,
                    bb["bottom"] / div,
                    bb["right"] / div,
                    bb["top"] / div,
                )
            except Exception:
                bbox = None
            return bbox


def datetime_to_unix_time(dt):
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")


def unix_time_to_datetime(unix_time):
    return pd.Timestamp.utcfromtimestamp(unix_time)


def get_unix_time(timestamp, osh_file):
    if not osh_file:
        raise ValueError(
            "The input file does not seem to be OSH.PBF -file. "
            "Timestamp can only be used with OSH.PBF files. "
            "You can download OSH.PBF files from Geofabrik (requires OSM account): "
            "https://osm-internal.download.geofabrik.de/"
        )

    if not isinstance(timestamp, int):
        dt = pd.to_datetime(timestamp, utc=True)
        unix_time = datetime_to_unix_time(dt)
    else:
        # If integer is provided test that it can be parsed to datetime
        unix_time = datetime_to_unix_time(unix_time_to_datetime(timestamp))

    # If the time is in the future raise exception
    if unix_time > datetime_to_unix_time(pd.Timestamp.utcnow()):
        raise ValueError(f"timestamp cannot be in the future. Got: {timestamp}.")

    # If the time is older than the first changeset in OSM (2005-04-09 19:54:13), raise exception
    first_changeset = "2005-04-09 19:54:13"
    if unix_time < datetime_to_unix_time(pd.to_datetime(first_changeset, utc=True)):
        raise ValueError(
            f"The first changeset to OSM was made in '{first_changeset}' by Steve. "
            f"You attempt to extract older data which won't work."
        )
    return unix_time


def warn_about_timestamp_not_set(unix_time):
    warnings.warn(
        f"Reading OSH.PBF file without user-defined timestamp. Using the current UTC"
        f" time as timestamp: {unix_time_to_datetime(unix_time)}",
        UserWarning,
    )
