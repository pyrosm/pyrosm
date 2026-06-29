"""Generate OSMnx reference fixtures for the pyrosm <-> OSMnx parity tests.

Downloads OSM data from the Overpass API at the same historical moment as the
``helsinki_region_pbf`` test extract (an Overpass "attic" query via the ``[date:...]``
setting) using OSMnx, and saves each layer to disk as a GeoParquet file. The pyrosm
tests then read the same area, time and filter from the PBF and compare against these
fixtures, so the comparison isolates query behaviour from data drift.

The anchor is the latest element edit timestamp found in ``helsinki_region_pbf``.

The area is a small central-Helsinki bounding box (an attic query over the whole region
times out). It is a subset of ``helsinki_region_pbf``, so the parity test reads the PBF
restricted to the same box -- ``OSM(fp, bounding_box=CENTRAL_HELSINKI_BBOX)`` -- and
compares against these fixtures.

Usage:
    python scripts/generate_osmnx_reference.py
"""

from pathlib import Path

import osmnx as ox

# Latest edit timestamp in helsinki_region_pbf (see module docstring).
ATTIC_DATE = "2019-04-21T19:43:42Z"

# Central Helsinki (Kamppi / city centre / Kruununhaka / Hakaniemi), a small subset of the
# helsinki_region_pbf extent: (minx, miny, maxx, maxy) == OSMnx (left, bottom, right, top).
CENTRAL_HELSINKI_BBOX = (24.92, 60.16, 24.97, 60.18)

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "osmnx_reference"

# The queries to capture. Each mirrors a pyrosm read of the same area/time, exercising a
# different kind of filter. ``pyrosm`` shows the equivalent pyrosm call for the test.
NETWORK_QUERIES = [
    {
        "name": "network_paths",
        # regex value with alternation (Overpass-style)
        "custom_filter": '["highway"~"cycleway|footway|path"]',
        "pyrosm": "get_network(custom_filter='[\"highway\"~\"cycleway|footway|path\"]', filter_type='keep')",
    },
    {
        "name": "network_path_bicycle_designated",
        # AND of two brackets in one string
        "custom_filter": '["highway"~"path"]["bicycle"~"designated"]',
        "pyrosm": 'get_network(custom_filter=\'["highway"~"path"]["bicycle"~"designated"]\', filter_type=\'keep\')',
    },
]

BUILDING_QUERIES = [
    {
        "name": "buildings_all",
        # the simplest building query: every building (one node/way/relation triple, so the
        # Overpass attic request stays light -- a multi-value list expands to one triple per
        # value and tends to time out on the public attic endpoint).
        "tags": {"building": True},
        "pyrosm": "get_buildings()",
    },
    {
        "name": "buildings_residential",
        # only residential buildings (a single value -> one node/way/relation triple,
        # so it stays light for the attic endpoint).
        "tags": {"building": "residential"},
        "pyrosm": "get_buildings(custom_filter={'building': ['residential']})",
    },
]


def file_safe(gdf):
    """Coerce list/dict cells (e.g. an edge's node-id list) to strings so the frame can
    be written to (Geo)Parquet without arrow type errors."""
    out = gdf.copy()
    geom_col = out.geometry.name
    for col in out.columns:
        if col == geom_col:
            continue
        out[col] = out[col].map(
            lambda v: (
                ";".join(map(str, v))
                if isinstance(v, (list, tuple, set))
                else (str(v) if isinstance(v, dict) else v)
            )
        )
    return out


def save(gdf, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.parquet"
    file_safe(gdf).to_parquet(path)
    print(f"  wrote {len(gdf):6d} features -> {path.relative_to(OUT_DIR.parents[2])}")


def main():
    # Make every Overpass query return the state of the data at ATTIC_DATE.
    ox.settings.overpass_settings = (
        '[out:json][timeout:{timeout}][date:"' + ATTIC_DATE + '"]'
    )
    ox.settings.use_cache = False
    ox.settings.log_console = True

    # Central-Helsinki subset of the PBF; the parity test reads the PBF with the same box.
    bbox = CENTRAL_HELSINKI_BBOX
    print(f"attic date : {ATTIC_DATE}")
    print(f"bbox       : {bbox}")
    print(f"output dir : {OUT_DIR}")

    for q in NETWORK_QUERIES:
        print(f"\n[network] {q['name']}  custom_filter={q['custom_filter']}")
        print(f"  pyrosm equivalent: {q['pyrosm']}")
        # truncate_by_edge=True keeps ways that cross the bbox edge (it retains an
        # out-of-bbox node when a neighbour is inside), matching pyrosm's complete-ways
        # bounding-box behaviour, so the two select the same OSM ways.
        graph = ox.graph_from_bbox(
            bbox,
            custom_filter=q["custom_filter"],
            retain_all=True,
            simplify=False,
            truncate_by_edge=True,
        )
        edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
        save(edges, q["name"])

    for q in BUILDING_QUERIES:
        print(f"\n[buildings] {q['name']}  tags={q['tags']}")
        print(f"  pyrosm equivalent: {q['pyrosm']}")
        feats = ox.features_from_bbox(bbox, tags=q["tags"])
        feats = feats[feats.geometry.type.isin(["Polygon", "MultiPolygon"])]
        save(feats, q["name"])

    print("\nDone.")


if __name__ == "__main__":
    main()
