"""Parse buildings from one PBF with one tool and print {"seconds", "features"} as JSON.

`benchmarks_scaling.ipynb` runs this as a separate subprocess per (tool, area) so each parse
is isolated: the parent process samples the whole process tree's peak memory with psutil and
enforces a timeout, and a crash or out-of-memory kill is recorded rather than taking the
notebook kernel down with it. Imports are kept lazy so the pyrosm out-of-core engine's spawned
pool workers (which re-import this module) stay cheap; the ``__main__`` guard keeps them from
re-running the parse.

Usage: python bench_worker.py <tool> <pbf>
"""
import json
import sys
import time

POLYGON_TYPES = {"Polygon", "MultiPolygon"}
LINE_TYPES = {"LineString", "MultiLineString"}


def _count_polygons(geodataframe):
    if geodataframe is None:
        return 0
    return int(geodataframe.geometry.geom_type.isin(POLYGON_TYPES).sum())


def _count_lines(geodataframe):
    if geodataframe is None:
        return 0
    return int(geodataframe.geometry.geom_type.isin(LINE_TYPES).sum())


def pyrosm_in_memory(pbf):
    from pyrosm import OSM

    return OSM(pbf).get_buildings()


def pyrosm_out_of_core(pbf):
    from pyrosm import OSM

    OSM.clear_cache(pbf)  # measure parsing, not a cache hit
    return OSM(pbf, engine="out_of_core", workers="auto").get_buildings()


def quackosm_with_columns(pbf):
    import os
    import tempfile

    import quackosm as qosm

    workdir = os.path.join(tempfile.gettempdir(), "quackosm_cache")
    return qosm.convert_pbf_to_geodataframe(
        pbf, tags_filter={"building": True}, keep_all_tags=True, explode_tags=True,
        verbosity_mode="silent", working_directory=workdir, ignore_cache=True)

def quackosm_no_columns(pbf):
    import os
    import tempfile

    import quackosm as qosm

    workdir = os.path.join(tempfile.gettempdir(), "quackosm_cache")
    return qosm.convert_pbf_to_geodataframe(
        pbf, tags_filter={"building": True}, keep_all_tags=True, explode_tags=False,
        verbosity_mode="silent", working_directory=workdir, ignore_cache=True)


def pyosmium(pbf):
    import geopandas as gpd
    import osmium
    from shapely import from_wkb

    wkb_factory = osmium.geom.WKBFactory()
    rows = []
    fp = (osmium.FileProcessor(pbf)
          .with_locations("flex_mem")
          .with_areas(osmium.filter.KeyFilter("building"))
          .with_filter(osmium.filter.KeyFilter("building")))
    for element in fp:
        if not element.is_area():
            continue
        try:
            wkb = wkb_factory.create_multipolygon(element)
        except Exception:
            continue
        if wkb:
            row = {tag.k: tag.v for tag in element.tags}
            row["geometry"] = from_wkb(bytes.fromhex(wkb))
            rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def osmnx_buildings(pbf, west, south, east, north):
    # OSMnx ignores the local file and downloads buildings for the bounding box from the
    # Overpass API; the work happens in the cloud, so its time and memory are not directly
    # comparable to the local-file readers.
    import osmnx as ox
    from shapely.geometry import box

    ox.settings.use_cache = False  # each run really hits Overpass
    polygon = box(float(west), float(south), float(east), float(north))
    return ox.features_from_polygon(polygon, tags={"building": True})


def osmium_tool(pbf):
    import os
    import subprocess
    import tempfile

    import geopandas as gpd

    work = tempfile.gettempdir()
    filtered = os.path.join(work, "bw_osmium_building.osm.pbf")
    geojson = os.path.join(work, "bw_osmium_building.geojsonseq")
    subprocess.run(["osmium", "tags-filter", "--overwrite", pbf, "w/building", "-o", filtered],
                   check=True, capture_output=True)
    subprocess.run(["osmium", "export", "--overwrite", filtered, "--geometry-types", "polygon",
                    "-f", "geojsonseq", "-o", geojson], check=True, capture_output=True)
    return gpd.read_file(geojson)


# --- road network (highway=* ways as lines) ---

def net_pyrosm_in_memory(pbf):
    from pyrosm import OSM

    return OSM(pbf).get_data_by_custom_criteria(
        {"highway": True}, filter_type="keep", keep_nodes=False, keep_relations=False)


def net_pyrosm_out_of_core(pbf):
    from pyrosm import OSM

    OSM.clear_cache(pbf)  # measure parsing, not a cache hit
    return OSM(pbf, engine="out_of_core", workers="auto").get_data_by_custom_criteria(
        {"highway": True}, filter_type="keep", keep_nodes=False, keep_relations=False)


def net_quackosm_with_columns(pbf):
    import os
    import tempfile

    import quackosm as qosm

    workdir = os.path.join(tempfile.gettempdir(), "quackosm_cache")
    return qosm.convert_pbf_to_geodataframe(
        pbf, tags_filter={"highway": True}, keep_all_tags=True, explode_tags=True,
        verbosity_mode="silent", working_directory=workdir, ignore_cache=True)


def net_quackosm_no_columns(pbf):
    import os
    import tempfile

    import quackosm as qosm

    workdir = os.path.join(tempfile.gettempdir(), "quackosm_cache")
    return qosm.convert_pbf_to_geodataframe(
        pbf, tags_filter={"highway": True}, keep_all_tags=True, explode_tags=False,
        verbosity_mode="silent", working_directory=workdir, ignore_cache=True)

def net_pyosmium(pbf):
    import geopandas as gpd
    import osmium
    from shapely import from_wkb

    wkb_factory = osmium.geom.WKBFactory()
    rows = []
    fp = (osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY)
          .with_locations("flex_mem")
          .with_filter(osmium.filter.KeyFilter("highway")))
    for element in fp:
        if not (element.is_way() and len(element.nodes) >= 2):
            continue
        try:
            wkb = wkb_factory.create_linestring(element)
        except Exception:
            continue
        if wkb:
            row = {tag.k: tag.v for tag in element.tags}
            row["geometry"] = from_wkb(bytes.fromhex(wkb))
            rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def net_osmium_tool(pbf):
    import os
    import subprocess
    import tempfile

    import geopandas as gpd

    work = tempfile.gettempdir()
    filtered = os.path.join(work, "bw_osmium_highway.osm.pbf")
    geojson = os.path.join(work, "bw_osmium_highway.geojsonseq")
    subprocess.run(["osmium", "tags-filter", "--overwrite", pbf, "w/highway", "-o", filtered],
                   check=True, capture_output=True)
    subprocess.run(["osmium", "export", "--overwrite", filtered, "--geometry-types", "linestring",
                    "-f", "geojsonseq", "-o", geojson], check=True, capture_output=True)
    return gpd.read_file(geojson)


def net_osmnx(pbf, west, south, east, north):
    # OSMnx returns a simplified routing graph from Overpass (not the raw ways), so its line count
    # is higher and not directly comparable; the work happens in the cloud.
    import osmnx as ox
    from shapely.geometry import box

    ox.settings.use_cache = False  # each run really hits Overpass
    polygon = box(float(west), float(south), float(east), float(north))
    graph = ox.graph_from_polygon(polygon, network_type="all", retain_all=True)
    return ox.graph_to_gdfs(graph, nodes=False)


# --- geometry + key only (head-to-head: both tools return just geometry + the key column) ---

def geom_pyrosm_out_of_core(pbf, key="building"):
    from pyrosm import OSM

    OSM.clear_cache(pbf)  # measure parsing, not a cache hit
    gdf = OSM(pbf, engine="out_of_core", workers="auto").get_data_by_custom_criteria(
        {key: True}, keep_other_tags=False, filter_type="keep", tags_as_columns=[key],
        keep_nodes=False, keep_relations=False)
    keep = [c for c in (key, "geometry") if c in gdf.columns]
    return gdf[keep]


def geom_quackosm(pbf, key="building"):
    import os
    import tempfile

    import quackosm as qosm

    workdir = os.path.join(tempfile.gettempdir(), "quackosm_cache")
    return qosm.convert_pbf_to_geodataframe(
        pbf, tags_filter={key: True}, keep_all_tags=False, explode_tags=True,
        verbosity_mode="silent", working_directory=workdir, ignore_cache=True)


TOOLS = {
    "pyrosm-in-memory": pyrosm_in_memory,
    "pyrosm-out-of-core": pyrosm_out_of_core,
    "quackosm-with-columns": quackosm_with_columns,
    "quackosm-no-columns": quackosm_no_columns,
    "pyosmium": pyosmium,
    "osmium-tool": osmium_tool,
    "osmnx": osmnx_buildings,
    "geom-pyrosm-out-of-core": geom_pyrosm_out_of_core,
    "geom-quackosm": geom_quackosm,
    "net-pyrosm-in-memory": net_pyrosm_in_memory,
    "net-pyrosm-out-of-core": net_pyrosm_out_of_core,
    "net-quackosm-with-columns": net_quackosm_with_columns,
    "net-quackosm-no-columns": net_quackosm_no_columns,
    "net-pyosmium": net_pyosmium,
    "net-osmium-tool": net_osmium_tool,
    "net-osmnx": net_osmnx,
}


if __name__ == "__main__":
    tool_name, pbf_path = sys.argv[1], sys.argv[2]
    extra_args = sys.argv[3:]   # e.g. the bounding box osmnx queries from Overpass
    started = time.perf_counter()
    result = TOOLS[tool_name](pbf_path, *extra_args)
    seconds = time.perf_counter() - started
    count = _count_lines(result) if tool_name.startswith("net-") else _count_polygons(result)
    print(json.dumps({"seconds": seconds, "features": count}))
