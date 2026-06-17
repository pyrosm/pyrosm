import pandas as pd
import geopandas as gpd
import numpy as np
from pyrosm._arrays cimport concatenate_dicts_of_arrays
from pyrosm.geometry cimport _create_point_geometries
from pyrosm.geometry cimport create_way_geometries
from pyrosm.geometry import orient_polygons
from pyrosm.relations import prepare_relations
from shapely.geometry import box
from pyrosm.data_filter import get_mask_by_osmid, _filter_array_dict_by_indices_or_mask
from shapely import multilinestrings
from pyrosm.distance import calculate_geom_length, calculate_geom_array_length

cpdef create_nodes_gdf(nodes, osmids_to_keep=None):
    cdef str k
    if isinstance(nodes, list):
        nodes = concatenate_dicts_of_arrays(nodes)

    # Check if nodes should be filtered
    if osmids_to_keep is not None:
        if isinstance(osmids_to_keep, np.ndarray):
            # Get mask for nodeid array
            mask = get_mask_by_osmid(nodes["id"], osmids_to_keep)
            nodes = _filter_array_dict_by_indices_or_mask(nodes, mask)
        else:
            raise ValueError("'indices_to_keep' should be a numpy array.")

    geometry = _create_point_geometries(nodes['lon'], nodes['lat'])
    return gpd.GeoDataFrame(dict(nodes), geometry=geometry, crs='epsg:4326')

cpdef create_gdf(data_arrays, geometry_array):
    return gpd.GeoDataFrame(
        create_df(data_arrays), geometry=geometry_array, crs='epsg:4326'
    )

cpdef create_df(data_arrays):
    cdef str key
    # Build in one constructor call; per-column df[k]=... triggers a spurious
    # pandas chained-assignment warning under Cython.
    columns = {}
    for key, data in data_arrays.items():
        # 'nodes' must be a list of arrays to avoid a block-consolidation error
        columns[key] = data.tolist() if key == "nodes" else data
    return pd.DataFrame(columns)

cpdef prepare_way_gdf(node_coordinates, ways, parse_network, calculate_seg_lengths):
    if ways is not None:
        # from/to ids and node-attribute records are only consumed when building
        # segment-level graph edges; skip them otherwise (plain get_network).
        ways, geometries, from_ids, to_ids, node_attributes = create_way_geometries(
            node_coordinates,
            ways,
            parse_network,
            calculate_seg_lengths
        )

        # .assign (not df[col]=...) avoids the spurious Cython CoW warning
        way_gdf = create_df(ways).assign(osm_type="way", geometry=geometries)

        # In case network is parsed, include way-level length info
        if parse_network and not calculate_seg_lengths:
            # Drop rows without geometry
            way_gdf = way_gdf.dropna(subset=['geometry']).reset_index(drop=True)

            # Create MultiLineStrings and calculate the length
            geoms = [multilinestrings(geom) for geom in way_gdf["geometry"]]
            way_gdf = way_gdf.assign(
                geometry=geoms,
                length=[calculate_geom_length(geom) for geom in geoms],
            )
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")

            # If only edges are requested, clean node_attributes
            node_attributes = None

        # In case network is parsed and requested for graph export,
        # include segment-level length info
        elif parse_network and calculate_seg_lengths:
            # Insert way-level from/to ids
            way_gdf = way_gdf.assign(u=from_ids, v=to_ids)

            # Drop rows without geometry
            way_gdf = way_gdf.dropna(subset=['geometry']).reset_index(drop=True)

            # Parse segment level from/to-ids
            u = np.concatenate(way_gdf["u"].to_list())
            v = np.concatenate(way_gdf["v"].to_list())

            # Explode multi-geometries
            way_gdf = way_gdf.explode("geometry").reset_index(drop=True)
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")

            # Update from/to-ids and calculate the length of the geometries
            way_gdf = way_gdf.assign(
                u=u,
                v=v,
                length=calculate_geom_array_length(way_gdf.geometry.values.to_numpy()),
            )

        # For cases not related to networks
        else:
            way_gdf = gpd.GeoDataFrame(way_gdf, geometry="geometry", crs="epsg:4326")
            node_attributes = None

    else:
        way_gdf = gpd.GeoDataFrame()
        node_attributes = None

    return way_gdf, node_attributes

cpdef prepare_node_gdf(nodes):
    if nodes is not None:
        # Create GeoDataFrame from nodes
        node_gdf = create_nodes_gdf(nodes).assign(osm_type="node")
    else:
        node_gdf = gpd.GeoDataFrame()
    return node_gdf

cpdef prepare_relation_gdf(node_coordinates, relations, relation_ways, tags_as_columns, bint keep_metadata=True):
    if relations is not None:
        relations = prepare_relations(relations, relation_ways,
                                      node_coordinates,
                                      tags_as_columns,
                                      keep_metadata)

        # prepare_relations returns an empty GeoDataFrame when no relation could
        # be assembled into a geometry (e.g. every boundary relation is incomplete
        # and was dropped). It then has no geometry column, so building a
        # GeoDataFrame(..., crs=...) from it would raise -- return empty instead.
        if "geometry" not in relations:
            relation_gdf = gpd.GeoDataFrame()
        else:
            relation_gdf = gpd.GeoDataFrame(relations, crs="epsg:4326").assign(
                osm_type="relation"
            )

    else:
        relation_gdf = gpd.GeoDataFrame()
    return relation_gdf

cpdef prepare_geodataframe(nodes, node_coordinates, ways,
                           relations, relation_ways,
                           tags_as_columns, bounding_box,
                           parse_network=False,
                           calculate_seg_lengths=False,
                           bint keep_metadata=True):

    # Prepare ways
    way_gdf, node_attr = prepare_way_gdf(node_coordinates,
                                         ways,
                                         parse_network,
                                         calculate_seg_lengths)

    # Prepare relation data
    relation_gdf = prepare_relation_gdf(node_coordinates, relations, relation_ways, tags_as_columns, keep_metadata)

    # When not parsing the network,
    # nodes should be kept as part of the main output
    if not parse_network:
        # Prepare nodes
        node_gdf = prepare_node_gdf(nodes)
    else:
        node_gdf = gpd.GeoDataFrame()

    # Merge all
    gdf = pd.concat([node_gdf, way_gdf, relation_gdf])

    if len(gdf) == 0:
        if parse_network:
            return None, None
        return None

    gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)

    # Normalize polygon ring orientation to the OGC/GeoJSON right-hand rule
    # (exterior CCW, holes CW); non-polygonal geometries are untouched (#230).
    gdf["geometry"] = gpd.GeoSeries(
        orient_polygons(gdf["geometry"].to_numpy()),
        index=gdf.index,
        crs=gdf.crs,
    )

    # When parsing the network with nodes, prepare the nodes frame
    if node_attr is not None:
        node_attr = pd.DataFrame(node_attr)
        node_attr = gpd.GeoDataFrame(node_attr,
                                     crs="epsg:4326",
                                     geometry=gpd.points_from_xy(node_attr["lon"],
                                                                 node_attr["lat"])
                                     ).drop_duplicates("id").reset_index(drop=True)

    # Filter by bounding box if it was used
    if bounding_box is not None:
        if isinstance(bounding_box, list):
            bounding_box = box(*bounding_box)
        # Filter data spatially
        orig_cols = list(gdf.columns)
        filter_gdf = gpd.GeoDataFrame({"geometry": [bounding_box]},
                                      crs="epsg:4326",
                                      index=[0])
        gdf = gpd.sjoin(gdf, filter_gdf, how="inner")
        gdf = gdf[orig_cols].reset_index(drop=True)

        if node_attr is not None:
            # Keep every node referenced by the kept (whole) edges, including
            # endpoints just outside the box. Spatially clipping the nodes to the
            # box instead would drop boundary endpoints of edges that straddle the
            # edge, leaving dangling u/v that break graph export (#199).
            referenced = pd.unique(
                np.concatenate([gdf["u"].to_numpy(), gdf["v"].to_numpy()])
            )
            node_attr = node_attr[node_attr["id"].isin(referenced)].reset_index(
                drop=True
            )

    if len(gdf) == 0:
        if parse_network:
            return None, None
        return None

    if parse_network:
        return gdf, node_attr
    return gdf
