"""Read an OSM PBF in spatial tiles and stitch the pieces back together.

``read_tiled`` covers the data extent with a grid of bounding-box tiles, reads
each tile with a normal ``OSM(filepath, bounding_box=tile)`` call, and
concatenates the per-tile GeoDataFrames into one result that is identical to an
untiled read. Only one tile's worth of node coordinates is resident at a time, so
the peak memory of the parse is bounded by the tile size rather than the whole
file -- at the cost of re-reading the source once per tile.
"""

import copy
import math
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidjson import dumps
from shapely.geometry import box

from pyrosm.boundary import boundary_relation_filter
from pyrosm.buildings import building_relation_filter
from pyrosm.data_manager import get_osm_data
from pyrosm.landuse import landuse_relation_filter
from pyrosm.natural import natural_relation_filter
from pyrosm.pbfreader import fetch_member_nodes, fetch_member_ways, parse_relations_only
from pyrosm.pois import poi_relation_filter
from pyrosm.pyrosm import OSM
from pyrosm.utils import get_bounding_box, validate_custom_filter

# Layers read_tiled can stitch, mapped to the OSM method that produces them. Each
# returns a single GeoDataFrame with one row per element, so ``(osm_type, id)`` is a
# unique identity key. ``boundaries`` is included for API completeness, but its
# features are relations (see read_tiled's ``layer`` docs).
LAYER_METHODS = {
    "network": "get_network",
    "buildings": "get_buildings",
    "pois": "get_pois",
    "landuse": "get_landuse",
    "natural": "get_natural",
    "boundaries": "get_boundaries",
    "custom_criteria": "get_data_by_custom_criteria",
}

# Native OSM coordinate quantum (10^-7 degrees); tile edges are placed on this
# integer grid so the boundaries are exact and reproducible.
_E7 = 10_000_000

# Ground-distance-per-degree constants for converting a target tile area in km^2
# into degree steps (mean Earth radii): one degree of latitude is ~110.574 km
# everywhere; one degree of longitude is ~111.320 km at the equator and scales by
# cos(latitude).
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_EQUATOR = 111.320

# Auto-sizing knobs (used only when ``tile_size=None``). They are conservative,
# whole-file estimates -- a future calibration pass can tune them without any API
# or test change. ``K_MB_PER_MB`` over-predicts the peak resident memory of a whole
# read as a multiple of the file size (MB per MB); ``SAFETY`` is the fraction of
# available memory the read may use; ``DEFAULT_TILE_KM2`` is the fixed fallback tile
# area used when the heuristic cannot run (no ``psutil``, unknown file size, or an
# absurd result).
K_MB_PER_MB = 15.0
SAFETY = 0.5
DEFAULT_TILE_KM2 = 25.0


def _extent_centre_latitude(extent):
    return (extent[1] + extent[3]) / 2.0


def _km2_to_degree_steps(area_km2, centre_lat):
    """Convert a target tile area in km^2 to (delta_lon, delta_lat) degree steps at
    the extent's centre latitude. The tile is square on the ground, so it is a
    latitude-dependent rectangle in degrees."""
    side_km = math.sqrt(area_km2)
    delta_lat = side_km / _KM_PER_DEG_LAT
    cos_phi = math.cos(math.radians(centre_lat))
    # Guard the poles, where a degree of longitude collapses to zero ground distance.
    cos_phi = max(cos_phi, 1e-6)
    delta_lon = side_km / (_KM_PER_DEG_LON_EQUATOR * cos_phi)
    return delta_lon, delta_lat


def _extent_area_km2(extent):
    """Approximate ground area of an extent in km^2 (centre-latitude longitude scale)."""
    minx, miny, maxx, maxy = extent
    cos_phi = max(math.cos(math.radians(_extent_centre_latitude(extent))), 1e-6)
    width_km = (maxx - minx) * _KM_PER_DEG_LON_EQUATOR * cos_phi
    height_km = (maxy - miny) * _KM_PER_DEG_LAT
    return width_km * height_km


def _auto_tile_km2(filepath, extent_area_km2):
    """Pick a tile area (km^2) from available memory and the file size, working in
    megabytes throughout. Falls back to ``DEFAULT_TILE_KM2`` whenever the estimate
    cannot be made or is non-positive/absurd."""
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / 1e6
    except Exception:
        return DEFAULT_TILE_KM2
    try:
        file_size_mb = os.path.getsize(filepath) / 1e6
    except OSError:
        return DEFAULT_TILE_KM2

    budget_mb = available_mb * SAFETY
    peak_mb = K_MB_PER_MB * file_size_mb
    if budget_mb <= 0 or peak_mb <= 0:
        return DEFAULT_TILE_KM2

    n_tiles = max(1, math.ceil(peak_mb / budget_mb))
    area = extent_area_km2 / n_tiles
    if area <= 0 or not math.isfinite(area):
        return DEFAULT_TILE_KM2
    return area


def _build_tile_grid(extent, step_x_deg, step_y_deg, mask=None):
    """Cover ``extent`` with a grid of bounding-box tiles of the given degree steps
    (``step_x_deg`` in longitude, ``step_y_deg`` in latitude), snapped to the E7
    integer grid. The last row/column is clamped to ``extent``; adjacent tiles share
    an edge. When ``mask`` is given, tiles whose box does not intersect it are
    dropped (a grid-reduction control only)."""
    minx, miny, maxx, maxy = extent
    if minx >= maxx or miny >= maxy:
        raise ValueError(
            "Invalid extent {ext}: expected [minx, miny, maxx, maxy] with "
            "minx < maxx and miny < maxy.".format(ext=list(extent))
        )

    min_x, max_x = round(minx * _E7), round(maxx * _E7)
    min_y, max_y = round(miny * _E7), round(maxy * _E7)
    step_x = round(step_x_deg * _E7)
    step_y = round(step_y_deg * _E7)
    if step_x <= 0 or step_y <= 0:
        raise ValueError("'tile_size' is too small to form a tile grid.")

    tiles = []
    x = min_x
    while x < max_x:
        x2 = min(x + step_x, max_x)
        y = min_y
        while y < max_y:
            y2 = min(y + step_y, max_y)
            bbox = [x / _E7, y / _E7, x2 / _E7, y2 / _E7]
            if mask is None or box(*bbox).intersects(mask):
                tiles.append(bbox)
            y = y2
        x = x2
    return tiles


def generate_tiles(extent, tile_size, mask=None):
    """Build a grid of bounding-box tiles covering ``extent``.

    Parameters
    ----------
    extent : list
        Area to cover as ``[minx, miny, maxx, maxy]`` in decimal degrees.
    tile_size : float
        Tile edge length in decimal degrees. Must be positive.
    mask : shapely geometry, optional
        When given, tiles whose bounding box does not intersect ``mask`` are
        dropped (a grid-reduction control only; the kept tiles are still read
        with their full bounding box).

    Returns
    -------
    list
        Tile bounding boxes, each ``[minx, miny, maxx, maxy]`` in degrees. The
        last row/column is clamped to ``extent``; adjacent tiles share an edge.
    """
    if tile_size <= 0:
        raise ValueError("'tile_size' must be a positive number of degrees.")
    return _build_tile_grid(extent, tile_size, tile_size, mask)


def _relation_member_way_ids(members):
    """The way-member ids of one relation's ``members`` record (member_type 'way')."""
    ids = []
    member_ids = members["member_id"]
    member_types = members["member_type"]
    for j in range(len(member_ids)):
        if member_types[j] == b"way":
            ids.append(int(member_ids[j]))
    return ids


def _candidate_relations(relations, extent_ways):
    """Slice ``relations`` (a ``parse_relations_only`` dict) to the relations that
    have at least one member way present in the kept tiles (``extent_ways``), plus
    the set of all those candidates' member way-ids. Membership-based, so a relation
    whose tiled geometry was dropped is still a candidate."""
    members_col = relations.get("members")
    if members_col is None:
        return None, set()

    keep_idx = []
    member_ids = set()
    for i in range(len(members_col)):
        way_ids = _relation_member_way_ids(members_col[i])
        if any(wid in extent_ways for wid in way_ids):
            keep_idx.append(i)
            member_ids.update(way_ids)
    if not keep_idx:
        return None, set()

    idx = np.array(keep_idx, dtype=np.int64)
    candidates = {col: relations[col][idx] for col in relations}
    return candidates, member_ids


def _layer_relation_filter_spec(layer, layer_kwargs):
    """The ``(custom_filter, osm_keys, filter_type)`` the layer applies to relations,
    reused from the feature modules so completion can be narrowed to the relations the
    layer would actually keep (rather than every relation that touches the tiles)."""
    custom_filter = copy.deepcopy(layer_kwargs.get("custom_filter"))
    if layer == "buildings":
        return building_relation_filter(custom_filter), None, "keep"
    if layer == "landuse":
        return landuse_relation_filter(custom_filter), None, "keep"
    if layer == "natural":
        return natural_relation_filter(custom_filter), None, "keep"
    if layer == "boundaries":
        boundary_type = layer_kwargs.get("boundary_type", "administrative")
        return boundary_relation_filter(custom_filter, boundary_type), None, "keep"
    if layer == "pois":
        return poi_relation_filter(custom_filter), None, "keep"
    # custom_criteria -- mirror get_data_by_custom_criteria's normalisation so the
    # prefilter matches the read: osm_keys str -> list, filter_type lowercased.
    osm_keys = layer_kwargs.get("osm_keys_to_keep")
    if isinstance(osm_keys, str):
        osm_keys = [osm_keys]
    # filter_type has already been validated as a string by the per-tile read.
    filter_type = layer_kwargs.get("filter_type", "keep").lower()
    return validate_custom_filter(custom_filter), osm_keys, filter_type


def _filter_relations_to_layer(layer, candidates, member_ways, layer_kwargs):
    """Narrow the candidate relations (and their member ways) to the relations the
    requested layer keeps, by running the layer's own relation filter over the cheap
    member-way records (no node coordinates needed). Returns
    ``(layer_relations, layer_member_ways)`` -- ``(None, [])`` when the layer keeps
    none."""
    data_filter, osm_keys, filter_type = _layer_relation_filter_spec(
        layer, layer_kwargs
    )
    # get_osm_data filters relations by their tags and keeps only relations whose
    # member ways are present; tags_as_columns is irrelevant here (we use only the
    # filtered relation set, not the assembled rows).
    _, _, _, layer_relations = get_osm_data(
        None, member_ways, candidates, [], data_filter, filter_type, osm_keys
    )
    if layer_relations is None:
        return None, []

    layer_member_ids = set()
    for members in layer_relations["members"]:
        layer_member_ids.update(_relation_member_way_ids(members))
    layer_member_ways = [w for w in member_ways if w["id"] in layer_member_ids]
    return layer_relations, layer_member_ways


def _read_relations_from_members(
    filepath,
    method,
    candidates,
    member_ways,
    member_coords,
    keep_metadata,
    unix_time,
    layer_kwargs,
):
    """Assemble the candidate relations from pre-fetched member ways/nodes by feeding
    them through the normal layer read (no bbox, no re-parse), and return only the
    relation rows. Reuses each layer's relation filter and tag handling."""
    if member_coords is None:
        return None
    osm = OSM(filepath, keep_metadata=keep_metadata)
    # Pre-load the member-only data so the layer method assembles relations without
    # re-parsing or loading the bulk; bounding_box is None, so nothing is clipped.
    # _osh_file is cleared and no timestamp is passed: the members were already
    # materialised at the right time, and this guarantees no re-parse is triggered.
    osm._osh_file = False
    osm._current_timestamp = unix_time
    osm._nodes = {}
    osm._node_coordinates = member_coords
    osm._way_records = member_ways
    osm._relations = candidates
    osm._relation_member_ways = None

    # Forward the layer's filter/tag kwargs (e.g. custom_filter), but not timestamp
    # (the members are already materialised and re-parsing must not be triggered).
    completion_kwargs = {
        k: v for k, v in copy.deepcopy(layer_kwargs).items() if k != "timestamp"
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r".*(did not contain any OSM nodes|[Cc]ould not find any|"
                r"extend beyond the bounding box).*"
            ),
        )
        gdf = getattr(osm, method)(**completion_kwargs)

    if gdf is None or len(gdf) == 0:
        return None
    rel = gdf[gdf["osm_type"] == "relation"]
    return rel.reset_index(drop=True) if len(rel) else None


def read_tiled(
    filepath,
    layer="network",
    tile_size=None,
    mask=None,
    extent=None,
    relations="complete",
    keep_metadata=True,
    **layer_kwargs,
):
    """Read one OSM layer in spatial tiles and return one stitched GeoDataFrame.

    The result has the same rows, the same set of columns and the same values as
    the equivalent untiled ``OSM(filepath).get_<layer>`` call, but only one tile's
    node coordinates are held in memory at a time. The source file is re-read once
    per tile. Row and column order may differ from the untiled read (column order
    is deterministic with the geometry column last; the untiled order is itself
    parse-order dependent). Structural columns keep their dtype; free-form string
    columns hold identical values but their pandas string representation (``object``
    vs ``StringDtype``) is content-inferred and may differ. For layers with node
    features (e.g. POIs, ``natural``) read with small tiles, pyrosm's bounding-box
    behaviour can additionally place a node's own tags in the ``tags`` column; those
    values are still available in their dedicated columns, so ids, geometries and
    the dedicated columns match the untiled read.

    Parameters
    ----------
    filepath : str
        Path to an ``*.osm.pbf`` file.
    layer : str
        One of ``"network"``, ``"buildings"``, ``"pois"``, ``"landuse"``,
        ``"natural"``, ``"boundaries"``, ``"custom_criteria"``.
        ``"custom_criteria"`` maps to ``OSM.get_data_by_custom_criteria`` and
        requires a ``custom_filter`` keyword. ``get_network(nodes=True)`` (which
        returns a node/edge tuple) is not supported. ``"boundaries"`` features are
        relations; with the default ``relations="complete"`` they are rebuilt from
        their full member set across tiles.
    tile_size : float, optional
        Target tile **area in square kilometres**. Each tile is a square on the
        ground (a latitude-dependent rectangle in degrees, sized at the extent's
        centre latitude). Smaller tiles lower peak memory but re-read the file more
        times. When ``None`` (default) the tile area is chosen automatically from
        the available memory and the file size (a conservative whole-file estimate),
        falling back to ``DEFAULT_TILE_KM2`` when that cannot be determined. The
        centre-latitude longitude scaling is accurate for city/region extents; a
        very tall multi-latitude extent gets a coarser area approximation.
    mask : shapely geometry, optional
        Restrict the tile grid to tiles intersecting this geometry. Features are
        not clipped to it; the result equals an untiled read over the union of the
        kept tile boxes.
    extent : list, optional
        Area to tile as ``[minx, miny, maxx, maxy]``. Defaults to the file's
        bounding box from the PBF header; required when the header has none.
        Extents that cross the antimeridian (longitude wrapping past +/-180) are
        not supported -- the planar tile grid would span the wrong side of the
        globe; pass a non-wrapping extent for such data.
    relations : str
        How relation-derived rows (e.g. multipolygons, boundaries) are handled.
        ``"complete"`` (default) rebuilds every relation that has a member way in the
        kept tiles from its full member set -- fetched once per call with two
        id-filtered passes over the file, never loading the bulk -- so relations are
        correct and not cut at tile edges; ``"drop"`` excludes all relation rows; and
        ``"error"`` raises if any relation row appears. ``network`` carries no
        relations, so the setting has no effect there.
    keep_metadata : bool
        Passed to each tile's ``OSM`` object (default ``True``).
    **layer_kwargs
        Extra keyword arguments forwarded to the layer's ``OSM`` method
        (e.g. ``custom_filter``, ``tags_to_keep``, ``extra_attributes``).

    Returns
    -------
    geopandas.GeoDataFrame or None
        The stitched layer, or ``None`` when no tile yields any feature.
    """
    if layer not in LAYER_METHODS:
        raise ValueError(
            "Unsupported layer '{layer}'. Supported layers: {ok}.".format(
                layer=layer, ok=", ".join(LAYER_METHODS)
            )
        )
    if relations not in ("complete", "drop", "error"):
        raise ValueError("'relations' should be one of 'complete', 'drop' or 'error'.")
    if layer == "network" and layer_kwargs.get("nodes"):
        raise ValueError(
            "read_tiled does not support get_network(nodes=True): it returns a "
            "(nodes, edges) tuple and slices ways into segments. Use nodes=False."
        )

    if extent is None:
        header_bbox = get_bounding_box(filepath)
        if header_bbox is None:
            raise ValueError(
                "The PBF header has no bounding box, so the extent to tile is "
                "unknown. Pass extent=[minx, miny, maxx, maxy]."
            )
        extent = list(header_bbox.bounds)

    # Choose a tile area (km^2) and turn it into latitude-aware degree steps.
    if tile_size is None:
        tile_size = _auto_tile_km2(filepath, _extent_area_km2(extent))
    if tile_size <= 0:
        raise ValueError("'tile_size' must be a positive area in km^2.")
    delta_lon, delta_lat = _km2_to_degree_steps(
        tile_size, _extent_centre_latitude(extent)
    )
    tiles = _build_tile_grid(extent, delta_lon, delta_lat, mask)

    method = LAYER_METHODS[layer]
    # Networks carry no relation geometries, so completion is a no-op there; skip its
    # cost and treat "complete" like "drop" (there are no relation rows to drop). A
    # custom_criteria read with keep_relations=False also yields no relation rows, so
    # there is nothing to complete -- short-circuit before any member fetch.
    completing = relations == "complete" and layer != "network"
    if layer == "custom_criteria" and layer_kwargs.get("keep_relations") is False:
        completing = False

    # Relation completion (once per call) needs every relation's definition and the
    # member ways present in the kept tiles. Parse the relation definitions up front.
    unix_time = None
    all_relations = None
    all_member_ids = set()
    extent_ways = set()
    if completing:
        probe = OSM(filepath, keep_metadata=keep_metadata)
        probe._set_current_time(layer_kwargs.get("timestamp"))
        unix_time = probe._current_timestamp
        all_relations = parse_relations_only(filepath, unix_time)
        members_col = all_relations.get("members")
        if members_col is not None:
            for members in members_col:
                all_member_ids.update(_relation_member_way_ids(members))

    frames = []
    for tile in tiles:
        osm = OSM(filepath, bounding_box=tile, keep_metadata=keep_metadata)
        # pyrosm normalises a passed custom_filter in place (e.g. True -> [True]);
        # copy per tile so the caller's arguments are never mutated.
        tile_kwargs = copy.deepcopy(layer_kwargs)
        with warnings.catch_warnings():
            # Suppress per-tile bbox-read warnings that are not actionable here:
            # empty tiles ("no nodes" / "could not find any") when the grid covers
            # areas with no data, and the incomplete-relation warning ("extend beyond
            # the bounding box") -- relations are handled once per call by `relations`,
            # not by each tile's read.
            warnings.filterwarnings(
                "ignore",
                message=(
                    r".*(did not contain any OSM nodes|[Cc]ould not find any|"
                    r"extend beyond the bounding box).*"
                ),
            )
            gdf = getattr(osm, method)(**tile_kwargs)

        # Record which relation member ways are present in this kept tile (used to
        # decide which relations to complete). The tile's parsed ways are available
        # even when the layer yields no feature rows.
        if completing and osm._way_records:
            extent_ways |= {w["id"] for w in osm._way_records} & all_member_ids

        if gdf is None or len(gdf) == 0:
            continue
        if "osm_type" not in gdf.columns or "id" not in gdf.columns:
            raise ValueError(
                "Layer '{layer}' output lacks the 'osm_type'/'id' columns needed "
                "to stitch tiles; it is not supported by read_tiled.".format(
                    layer=layer
                )
            )
        if gdf.duplicated(subset=["osm_type", "id"]).any():
            raise ValueError(
                "Layer '{layer}' produced multiple rows per (osm_type, id) within "
                "a tile, so tiles cannot be de-duplicated safely; it is not "
                "supported by read_tiled.".format(layer=layer)
            )
        frames.append(gdf)

    # The per-tile (non-relation) result.
    bulk = None
    if frames:
        bulk = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        has_relation = (bulk["osm_type"] == "relation").any()
        if relations == "error" and has_relation:
            raise ValueError(
                "Tiled read of '{layer}' contains relation-derived features, "
                "which cannot be reconstructed exactly across tiles. Pass "
                "relations='drop' to exclude them, or relations='complete' to "
                "rebuild them.".format(layer=layer)
            )
        if relations in ("drop", "complete"):
            # Per-tile relation rows are partial (cut at tile edges); drop them and,
            # for "complete", replace them with the rebuilt relations below.
            bulk = bulk[bulk["osm_type"] != "relation"]

    # Rebuild the relations once, from the complete member set of every relation that
    # has a member way in the kept tiles.
    relation_gdf = None
    if completing and extent_ways:
        candidates, member_ids = _candidate_relations(all_relations, extent_ways)
        # Cheap pass: member ways of every candidate relation (way records only, no
        # coordinates). Then narrow to the relations the requested layer keeps, so the
        # expensive node fetch never touches other relation types' members.
        member_ways = fetch_member_ways(filepath, member_ids, unix_time, keep_metadata)
        layer_relations, layer_member_ways = _filter_relations_to_layer(
            layer, candidates, member_ways, layer_kwargs
        )
        if layer_relations is not None and layer_member_ways:
            node_ids = set()
            for way in layer_member_ways:
                node_ids.update(way["nodes"])
            member_coords = fetch_member_nodes(
                filepath, node_ids, unix_time, keep_metadata
            )
            relation_gdf = _read_relations_from_members(
                filepath,
                method,
                layer_relations,
                layer_member_ways,
                member_coords,
                keep_metadata,
                unix_time,
                layer_kwargs,
            )

    parts = [p for p in (bulk, relation_gdf) if p is not None and len(p) > 0]
    if not parts:
        return None
    stitched = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)

    stitched = stitched.drop_duplicates(
        subset=["osm_type", "id"], keep="first"
    ).reset_index(drop=True)

    # pyrosm's bounding-box node path can leave a row's "tags" as a raw dict
    # instead of the canonical JSON string; normalise so the column is uniform.
    if "tags" in stitched.columns:
        stitched["tags"] = stitched["tags"].map(
            lambda v: dumps(v) if isinstance(v, dict) else v
        )

    # The stitched column set and dtypes match the untiled read, but pd.concat
    # appends tag columns that first appear in a later tile, so the order can
    # differ from the untiled parse order (which is itself data-dependent). Keep a
    # deterministic, conventional order with the geometry column last.
    geom = stitched.geometry.name
    stitched = stitched[[c for c in stitched.columns if c != geom] + [geom]]
    return stitched
