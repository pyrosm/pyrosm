"""Crop a PBF to a bounding box with one tool, or count the buildings of a PBF, and print the
result as JSON.

benchmarks_scaling.ipynb runs every crop as a separate process (measured by run_monitored.py),
so each crop starts from a fresh process. Every tool keeps complete ways. The crop is written to
a temporary file next to the output and renamed to it once complete, so the output path never
holds a partial file. The output is then read back with pyrosm in another process ("count"),
outside the measured crop, to count its buildings, which checks that it is a valid subset.

Usage: python crop_worker.py <tool> <input.pbf> <output.pbf> <west> <south> <east> <north>
                             [<pyrosm workers>]
       python crop_worker.py count <file.pbf>

Tools: "pyrosm" (to_pbf with repack=True), "pyrosm-fast" (repack=False), "pyosmium",
"osmium-tool" and "osmosis". pyosmium and osmium-tool take the size of libosmium's thread
pool from the OSMIUM_POOL_THREADS environment variable.

A crop prints {"seconds", "out_mb"}; "count" prints {"buildings"}.
"""

import json
import os
import shutil
import subprocess
import sys
import time


def crop_with_pyrosm(input_pbf, output_pbf, bbox, workers=1, repack=True):
    from pyrosm import OSM

    OSM(input_pbf, bounding_box=bbox).to_pbf(output_pbf, workers=workers, repack=repack)


def crop_with_pyosmium(input_pbf, output_pbf, bbox):
    """Crop with three passes over the file: find the ids of the nodes inside the box, find
    the ways that touch the box (keeping all of their nodes, so ways crossing the border stay
    whole), and write the kept nodes, ways and relations."""
    import osmium

    if os.path.exists(output_pbf):
        os.remove(output_pbf)
    west, south, east, north = bbox

    nodes_in_box = set()
    for node in osmium.FileProcessor(input_pbf).with_filter(
        osmium.filter.EntityFilter(osmium.osm.NODE)
    ):
        if west <= node.location.lon <= east and south <= node.location.lat <= north:
            nodes_in_box.add(node.id)

    ways_to_keep = set()
    nodes_to_keep = set(nodes_in_box)
    for way in osmium.FileProcessor(input_pbf).with_filter(
        osmium.filter.EntityFilter(osmium.osm.WAY)
    ):
        way_node_ids = [node.ref for node in way.nodes]
        if any(node_id in nodes_in_box for node_id in way_node_ids):
            ways_to_keep.add(way.id)
            nodes_to_keep.update(way_node_ids)

    writer = osmium.SimpleWriter(output_pbf)
    try:
        for element in osmium.FileProcessor(input_pbf):
            if element.is_node() and element.id in nodes_to_keep:
                writer.add_node(element)
            elif element.is_way() and element.id in ways_to_keep:
                writer.add_way(element)
            elif element.is_relation() and any(
                (member.type == "w" and member.ref in ways_to_keep)
                or (member.type == "n" and member.ref in nodes_to_keep)
                for member in element.members
            ):
                writer.add_relation(element)
    finally:
        writer.close()


def crop_with_osmium_tool(input_pbf, output_pbf, bbox):
    west, south, east, north = bbox
    subprocess.run(
        ["osmium", "extract", "--overwrite", "--strategy", "complete_ways",
         "--bbox", f"{west},{south},{east},{north}", input_pbf, "-o", output_pbf],
        check=True, capture_output=True,
    )


def crop_with_osmosis(input_pbf, output_pbf, bbox):
    if os.path.exists(output_pbf):
        os.remove(output_pbf)
    west, south, east, north = bbox
    subprocess.run(
        [shutil.which("osmosis") or "osmosis", "--read-pbf", input_pbf,
         "--bounding-box", f"left={west}", f"bottom={south}", f"right={east}", f"top={north}",
         "completeWays=yes", "--write-pbf", output_pbf],
        check=True, capture_output=True,
    )


def crop(tool, input_pbf, output_pbf, bbox, workers):
    if tool in ("pyrosm", "pyrosm-fast"):
        crop_with_pyrosm(input_pbf, output_pbf, bbox, workers, repack=tool == "pyrosm")
    elif tool == "pyosmium":
        crop_with_pyosmium(input_pbf, output_pbf, bbox)
    elif tool == "osmium-tool":
        crop_with_osmium_tool(input_pbf, output_pbf, bbox)
    elif tool == "osmosis":
        crop_with_osmosis(input_pbf, output_pbf, bbox)
    else:
        raise ValueError(f"unknown crop tool: {tool}")


def partial_path(output_pbf):
    """The temporary file a crop is written to before it is renamed to ``output_pbf``."""
    directory, name = os.path.split(os.path.abspath(output_pbf))
    return os.path.join(directory, ".partial-" + name)


def count_buildings(pbf):
    from pyrosm import OSM

    buildings = OSM(pbf, engine="in_memory").get_buildings()
    return 0 if buildings is None else len(buildings)


if __name__ == "__main__":
    if sys.argv[1] == "count":
        print(json.dumps({"buildings": count_buildings(sys.argv[2])}))
        sys.exit()

    tool, input_pbf, output_pbf = sys.argv[1:4]
    bbox = [float(value) for value in sys.argv[4:8]]
    workers = int(sys.argv[8]) if len(sys.argv) > 8 else 1

    # osmium-tool takes the file format from the name, so the temporary file keeps ".osm.pbf".
    partial = partial_path(output_pbf)
    if os.path.lexists(partial):
        os.remove(partial)   # left by a crop that was killed
    started = time.perf_counter()
    try:
        crop(tool, input_pbf, partial, bbox, workers)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    seconds = time.perf_counter() - started
    os.replace(partial, output_pbf)
    print(json.dumps({"seconds": seconds, "out_mb": round(os.path.getsize(output_pbf) / 1e6, 1)}))
