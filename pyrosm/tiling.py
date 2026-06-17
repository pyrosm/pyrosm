"""Read an OSM PBF in spatial tiles and stitch the pieces back together.

``read_tiled`` covers the data extent with a grid of bounding-box tiles, reads
each tile with a normal ``OSM(filepath, bounding_box=tile)`` call, and
concatenates the per-tile GeoDataFrames into one result that is identical to an
untiled read. Only one tile's worth of node coordinates is resident at a time, so
the peak memory of the parse is bounded by the tile size rather than the whole
file -- at the cost of re-reading the source once per tile.
"""

import copy
import warnings

import geopandas as gpd
import pandas as pd
from rapidjson import dumps
from shapely.geometry import box

from pyrosm.pyrosm import OSM
from pyrosm.utils import get_bounding_box

# Layers that return a single GeoDataFrame with one row per element, for which
# ``(osm_type, id)`` is a unique identity key.
SUPPORTED_LAYERS = ("network", "buildings", "pois", "landuse", "natural")

# Native OSM coordinate quantum (10^-7 degrees); tile edges are placed on this
# integer grid so the boundaries are exact and reproducible.
_E7 = 10_000_000


def generate_tiles(extent, tile_size, aoi=None):
    """Build a grid of bounding-box tiles covering ``extent``.

    Parameters
    ----------
    extent : list
        Area to cover as ``[minx, miny, maxx, maxy]`` in decimal degrees.
    tile_size : float
        Tile edge length in decimal degrees. Must be positive.
    aoi : shapely geometry, optional
        When given, tiles whose bounding box does not intersect ``aoi`` are
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

    minx, miny, maxx, maxy = extent
    if minx >= maxx or miny >= maxy:
        raise ValueError(
            "Invalid extent {ext}: expected [minx, miny, maxx, maxy] with "
            "minx < maxx and miny < maxy.".format(ext=list(extent))
        )

    min_x, max_x = round(minx * _E7), round(maxx * _E7)
    min_y, max_y = round(miny * _E7), round(maxy * _E7)
    step = round(tile_size * _E7)
    if step <= 0:
        raise ValueError("'tile_size' is too small to form a tile grid.")

    tiles = []
    x = min_x
    while x < max_x:
        x2 = min(x + step, max_x)
        y = min_y
        while y < max_y:
            y2 = min(y + step, max_y)
            bbox = [x / _E7, y / _E7, x2 / _E7, y2 / _E7]
            if aoi is None or box(*bbox).intersects(aoi):
                tiles.append(bbox)
            y = y2
        x = x2
    return tiles


def read_tiled(
    filepath,
    layer="network",
    tile_size=0.5,
    aoi=None,
    extent=None,
    relations="error",
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
        ``"natural"``. ``get_network(nodes=True)`` (which returns a node/edge
        tuple) is not supported.
    tile_size : float
        Tile edge length in decimal degrees (default ``0.5``). Smaller tiles
        lower peak memory but re-read the file more times.
    aoi : shapely geometry, optional
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
        How relation-derived rows (e.g. multipolygons) are handled. ``"error"``
        (default) raises if any appear, because relations spanning tiles cannot be
        reconstructed exactly here; ``"drop"`` excludes them and returns only the
        node/way rows.
    keep_metadata : bool
        Passed to each tile's ``OSM`` object (default ``True``).
    **layer_kwargs
        Extra keyword arguments forwarded to the ``get_<layer>`` method
        (e.g. ``custom_filter``, ``tags_to_keep``, ``extra_attributes``).

    Returns
    -------
    geopandas.GeoDataFrame or None
        The stitched layer, or ``None`` when no tile yields any feature.
    """
    if layer not in SUPPORTED_LAYERS:
        raise ValueError(
            "Unsupported layer '{layer}'. Supported layers: {ok}.".format(
                layer=layer, ok=", ".join(SUPPORTED_LAYERS)
            )
        )
    if relations not in ("error", "drop"):
        raise ValueError("'relations' should be either 'error' or 'drop'.")
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

    tiles = generate_tiles(extent, tile_size, aoi)

    method = "get_" + layer
    frames = []
    for tile in tiles:
        osm = OSM(filepath, bounding_box=tile, keep_metadata=keep_metadata)
        # pyrosm normalises a passed custom_filter in place (e.g. True -> [True]);
        # copy per tile so the caller's arguments are never mutated.
        tile_kwargs = copy.deepcopy(layer_kwargs)
        with warnings.catch_warnings():
            # Empty tiles are expected when the grid (or an AOI) covers areas with
            # no data; their "no nodes" / "could not find any" warnings are not
            # actionable here.
            warnings.filterwarnings(
                "ignore",
                message=r".*(did not contain any OSM nodes|[Cc]ould not find any).*",
            )
            gdf = getattr(osm, method)(**tile_kwargs)

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

    if not frames:
        return None

    stitched = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

    has_relation = (stitched["osm_type"] == "relation").any()
    if relations == "error":
        if has_relation:
            raise ValueError(
                "Tiled read of '{layer}' contains relation-derived features, "
                "which cannot be reconstructed exactly across tiles. Pass "
                "relations='drop' to exclude them.".format(layer=layer)
            )
    else:
        stitched = stitched[stitched["osm_type"] != "relation"]

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
