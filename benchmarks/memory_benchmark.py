"""Memory + speed benchmark for the Bucket A read-path optimizations.

Measures peak process RSS and wall-clock for the multi-layer extraction
workflow (network + buildings + POIs from one ``OSM`` object) on a real,
freshly downloaded extract, under two configurations:

  (D) default  : ``OSM(fp)``                       -- today's behavior
  (O) opt-in   : ``OSM(fp, keep_metadata=False)``  -- the A5 memory option

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


def _peak_rss_mb():
    # ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _run_once(filepath, keep_metadata):
    """Child entry point: parse the multi-layer workflow and print 'rss_mb\\tseconds'."""
    from pyrosm import OSM, get_data

    fp = filepath if filepath else get_data("Helsinki")
    start = time.perf_counter()
    osm = OSM(fp, keep_metadata=keep_metadata)
    osm.get_network()
    osm.get_buildings()
    osm.get_pois(custom_filter={"amenity": True, "shop": True})
    elapsed = time.perf_counter() - start
    print(f"{_peak_rss_mb():.1f}\t{elapsed:.2f}")


def _spawn(filepath, keep_metadata):
    cmd = [sys.executable, __file__, "--child", str(keep_metadata)]
    if filepath:
        cmd.append(filepath)
    out = subprocess.check_output(cmd, text=True).strip().splitlines()[-1]
    rss_mb, seconds = out.split("\t")
    return float(rss_mb), float(seconds)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--child":
        keep_metadata = args[1] == "True"
        filepath = args[2] if len(args) > 2 else None
        _run_once(filepath, keep_metadata)
        return

    filepath = args[0] if args else None
    print("config                         peak_rss(MB)   wall(s)")
    results = {}
    for label, keep_metadata in [("default", True), ("keep_metadata=False", False)]:
        rss_mb, seconds = _spawn(filepath, keep_metadata)
        results[label] = (rss_mb, seconds)
        print(f"{label:30s} {rss_mb:10.1f}   {seconds:8.2f}")

    d_rss, d_t = results["default"]
    o_rss, o_t = results["keep_metadata=False"]
    print(
        f"\ndelta (opt-in vs default): RSS {o_rss - d_rss:+.1f} MB "
        f"({100 * (o_rss - d_rss) / d_rss:+.1f}%), "
        f"wall {o_t - d_t:+.2f} s ({100 * (o_t - d_t) / d_t:+.1f}%)"
    )


if __name__ == "__main__":
    main()
