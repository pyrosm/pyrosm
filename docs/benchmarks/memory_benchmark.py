"""Memory + speed benchmark for the Bucket A read-path optimizations.

Measures peak process RSS and wall-clock for the multi-layer extraction
workflow (network + buildings + POIs from one ``OSM`` object) on a real,
freshly downloaded extract, under three configurations:

  (D)  default                            : ``OSM(fp)``                  -- today's behavior
  (O1) keep_metadata=False                : drop way/relation metadata
  (O2) keep_metadata=False + tags_to_keep : also keep only a minimal set of tag columns

Each configuration runs in its own subprocess so the reported peak RSS
(``resource.getrusage(RUSAGE_SELF).ru_maxrss``, an OS/process-level number,
not ``tracemalloc``) reflects only that run. Run before and after each Bucket A
measure and record the numbers in ``benchmarks/memory_results.md``.

Usage::

    python benchmarks/memory_benchmark.py            # uses get_data("Helsinki")
    python benchmarks/memory_benchmark.py /path/to/area.osm.pbf
"""

import resource
import subprocess
import sys
import time

# Minimal, representative tag-column sets for the tags_to_keep configuration.
NETWORK_TAGS = ["highway", "name", "maxspeed", "oneway"]
BUILDING_TAGS = ["building"]
POI_TAGS = ["amenity", "shop", "name"]

# (label, keep_metadata, restrict_tags)
CONFIGS = [
    ("default", True, False),
    ("keep_metadata=False", False, False),
    ("keep_metadata=False+tags_to_keep", False, True),
]


def _peak_rss_mb():
    # ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _run_once(filepath, keep_metadata, restrict_tags):
    """Child entry point: parse the multi-layer workflow and print 'rss_mb\\tseconds'."""
    from pyrosm import OSM, get_data

    fp = filepath if filepath else get_data("Helsinki")
    net_tags = NETWORK_TAGS if restrict_tags else None
    bld_tags = BUILDING_TAGS if restrict_tags else None
    poi_tags = POI_TAGS if restrict_tags else None
    start = time.perf_counter()
    osm = OSM(fp, keep_metadata=keep_metadata)
    osm.get_network(tags_to_keep=net_tags)
    osm.get_buildings(tags_to_keep=bld_tags)
    osm.get_pois(custom_filter={"amenity": True, "shop": True}, tags_to_keep=poi_tags)
    elapsed = time.perf_counter() - start
    print(f"{_peak_rss_mb():.1f}\t{elapsed:.2f}")


def _spawn(filepath, keep_metadata, restrict_tags):
    cmd = [sys.executable, __file__, "--child", str(keep_metadata), str(restrict_tags)]
    if filepath:
        cmd.append(filepath)
    out = subprocess.check_output(cmd, text=True).strip().splitlines()[-1]
    rss_mb, seconds = out.split("\t")
    return float(rss_mb), float(seconds)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--child":
        keep_metadata = args[1] == "True"
        restrict_tags = args[2] == "True"
        filepath = args[3] if len(args) > 3 else None
        _run_once(filepath, keep_metadata, restrict_tags)
        return

    filepath = args[0] if args else None
    print("config                              peak_rss(MB)   wall(s)")
    results = {}
    for label, keep_metadata, restrict_tags in CONFIGS:
        rss_mb, seconds = _spawn(filepath, keep_metadata, restrict_tags)
        results[label] = (rss_mb, seconds)
        print(f"{label:35s} {rss_mb:10.1f}   {seconds:8.2f}")

    d_rss, d_t = results["default"]
    for label, _, _ in CONFIGS[1:]:
        o_rss, o_t = results[label]
        print(
            f"\ndelta ({label} vs default): RSS {o_rss - d_rss:+.1f} MB "
            f"({100 * (o_rss - d_rss) / d_rss:+.1f}%), "
            f"wall {o_t - d_t:+.2f} s ({100 * (o_t - d_t) / d_t:+.1f}%)"
        )


if __name__ == "__main__":
    main()
