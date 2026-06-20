# Out-of-core engine cache: pyrosm vs quackosm

The out-of-core engine (`pyrosm/engine/`) caches each feature read to its own
GeoParquet keyed by the source file (mtime + size) and the read parameters, so a
repeated read of the same layer skips decoding and is served straight from disk.
This page measures that cache against [quackosm](https://github.com/kraina-ai/quackosm)
(a DuckDB-based decode-once/query-many reader) across three extract sizes, on
speed and peak memory, for both the first (cold) and repeat (warm) read.

## TL;DR

- **Cold reads cross over with file size.** On a ~36 MB extract pyrosm is several
  times faster cold (single-process decode beats quackosm's fixed DuckDB
  start-up); around the ~116 MB mark they tie; on a 1.9 GB country file quackosm
  is **~3.5× faster** cold, because pyrosm's per-element Python/Shapely assembly
  of millions of features is the bottleneck.
- **Warm reads are tied everywhere** — both libraries just reload a cached
  parquet, so the cost is dominated by materialising the GeoDataFrame, not by
  the cache format. The one exception is pyrosm's network layer, which is **not**
  cached and re-decodes on every read (see Estonia below).
- **Peak memory is comparable** on the large file (both ~13 GB for ~17.7 M
  buildings — the in-memory result frame is the binding constraint for either
  tool); quackosm runs ~30 % lower RSS on the mid-size multi-layer workload.

## Setup

- **Machine:** macOS (Darwin), Apple Silicon, 10 logical cores, ~26 GB RAM.
- **Libraries:** pyrosm 0.9.1 (out-of-core engine), quackosm 0.17.1, on Python
  3.12, geopandas 1.1.3, shapely 2.1.2, pyarrow 24.0.0, duckdb 1.5.3.
- **Datasets:** `Helsinki_region.osm.pbf` (36.7 MB), `estonia-latest.osm.pbf`
  (115 MB), `poland-latest.osm.pbf` (1.92 GB).
- **Metric:** peak process RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
  (OS-level), wall-clock via `time.perf_counter()`. Each phase runs in its own
  subprocess so the reported peak RSS is that phase alone.
- **Cold vs warm:** *cold* is the first read into an empty cache / quackosm
  working directory (pyrosm decodes → assembles → writes the result parquet;
  quackosm converts the PBF → parquet, then loads it). *Warm* is the immediately
  following identical read against the now-populated cache.
- **Both tools return GeoDataFrames** (pyrosm `get_<layer>`, quackosm
  `convert_pbf_to_geodataframe`) so the comparison is end-to-end including the
  in-memory frame.
- **Worker policy:** the pyrosm engine decodes single-process below ~70 MB and a
  worker per CPU above it ([`pool.py`](../../pyrosm/engine/pool.py)), so the
  36.7 MB extract is decoded serially and the 115 MB / 1.9 GB files use 10
  workers; DuckDB is internally multi-threaded throughout.
- **Row counts** differ by <0.5 % between the libraries from slightly different
  layer definitions; the network counts differ more because pyrosm returns a
  routable edge network while quackosm returns raw `highway=*` ways (the two
  network rows are not the same object — see the Estonia note).

## Helsinki region (36.7 MB) — buildings, POIs, landuse, natural

Four layers read on one process; pyrosm decodes single-process at this size.

| layer | rows (pyrosm / quack) | pyrosm cold | quack cold | pyrosm warm | quack warm |
| --- | --- | ---: | ---: | ---: | ---: |
| buildings | 175,975 / 176,674 | 11.44 s | 30.26 s | 0.23 s | 0.26 s |
| pois | 32,225 / 32,213 | 6.61 s | 28.38 s | 0.11 s | 0.12 s |
| landuse | 15,606 / 15,535 | 5.43 s | 28.95 s | 0.03 s | 0.09 s |
| natural | 34,476 / 34,656 | 6.29 s | 29.35 s | 0.04 s | 0.11 s |
| **total** | — | **29.8 s** | **116.9 s** | **0.42 s** | **0.58 s** |
| **peak RSS** | — | **1239 MB** | **1332 MB** | **563 MB** | **440 MB** |
| cache on disk | — | 41 MB | 29 MB | — | — |

- **pyrosm is ~3.9× faster cold** (29.8 s vs 116.9 s). quackosm pays a roughly
  fixed ~28–30 s per layer here — its DuckDB conversion has a high floor that the
  small file cannot amortise — whereas pyrosm's serial decode of a 36.7 MB file
  is quick and each layer is cheap.
- **Warm is a wash** (0.42 s vs 0.58 s): both reload a small cached parquet.
- Peak RSS is comparable (~1.2–1.3 GB cold); quackosm is a touch lower warm.

## Estonia (115 MB) — buildings, POIs, network

Three layers on one process; pyrosm decodes with 10 workers at this size.

| layer | rows (pyrosm / quack) | pyrosm cold | quack cold | pyrosm warm | quack warm |
| --- | --- | ---: | ---: | ---: | ---: |
| buildings | 902,739 / 902,893 | 50.97 s | 40.00 s | 0.72 s | 0.70 s |
| pois | 104,972 / 104,980 | 21.80 s | 32.43 s | 0.38 s | 0.47 s |
| network | 340,167 / 429,685 | 34.10 s | 35.33 s | **32.48 s** | 0.33 s |
| **total** | — | **106.9 s** | **107.8 s** | **33.6 s** | **1.50 s** |
| **peak RSS** | — | **3841 MB** | **2686 MB** | **3389 MB** | **1173 MB** |
| cache on disk | — | 105 MB | — | — | — |

- **Cold is a tie** (106.9 s vs 107.8 s): this is the crossover size. pyrosm's
  parallel decode keeps buildings (50 s) competitive, and POIs are faster than
  quackosm.
- **The warm gap is entirely the network layer.** pyrosm's per-layer cache
  covers buildings/POIs (warm 0.72 s / 0.38 s), but `get_network` returns edges
  through a separate path that is **not** cached, so it re-decodes the whole file
  on every read (warm 32.5 s) — that single layer is essentially all of pyrosm's
  33.6 s warm total. quackosm caches every layer uniformly, so its warm total is
  1.5 s.
- The network row counts (340,167 vs 429,685) are not comparable: pyrosm builds
  a routable edge network while quackosm returns every `highway=*` way.
- quackosm runs **~30 % lower peak RSS** on this multi-layer workload (2686 MB vs
  3841 MB cold).

## Poland (1.92 GB) — buildings only

A country file, single layer, read on a 10-worker decode.

| | rows | cold | warm | disk |
| --- | ---: | ---: | ---: | ---: |
| pyrosm cache | 17,741,410 | 1066.2 s (17.8 min) | 13.5 s | 1.7 GB cache |
| quackosm | 17,746,624 | 308.3 s (5.1 min) | 13.7 s | 1.4 GB workdir |
| peak RSS (pyrosm) | — | 12,603 MB | 12,918 MB | — |
| peak RSS (quack) | — | 13,070 MB | 13,629 MB | — |

- **quackosm is ~3.5× faster cold** (5.1 min vs 17.8 min). At 17.7 M features the
  bottleneck is building the result, and DuckDB's columnar conversion scales far
  better than pyrosm assembling millions of Python way-records and Shapely
  geometries.
- **Warm is a dead tie** (13.5 s vs 13.7 s): both read their cached parquet and
  materialise the same ~17.7 M-row frame, so the cost is identical and dominated
  by frame construction, not the cache.
- **Peak RSS is comparable at ~12.5–13.6 GB** for both, in both phases — the
  17.7 M-building GeoDataFrame is the binding constraint regardless of tool or
  caching; neither returns Poland buildings as a single frame in much under
  ~13 GB.

## Cold-read scaling (buildings)

The buildings layer measured cold across all three sizes shows the crossover:

| extract | pyrosm cold | quackosm cold |
| --- | ---: | ---: |
| Helsinki region (36.7 MB) | **11.4 s** | 30.3 s |
| Estonia (115 MB) | 51.0 s | **40.0 s** |
| Poland (1.92 GB) | 1066 s | **308 s** |

pyrosm's low fixed overhead wins on small extracts; the crossover sits around
Estonia; on a large country file quackosm's DuckDB throughput pulls clearly
ahead. Warm reads are tied at every size (both reload the cached parquet), and
peak memory is comparable (the result frame dominates) — so the per-layer
cache's value is the fast repeat read, while the *first* read of a very large
file is gated by pyrosm's per-element assembly rather than by the cache.

## Scope of the cache

- The cache is **per layer**, keyed by the source file (mtime + size) plus the
  read parameters (filter, tags-as-columns, metadata/nodes flags, bounding box,
  relation completion), and is reused across processes and sessions. An empty
  result is recorded with a marker file so it is not recomputed.
- It requires `pyarrow`; without it the engine falls back to the uncached read.
- `get_network` is **not** covered (it returns edges via a separate path), so
  network reads re-decode every time — visible as the 32.5 s warm network read on
  Estonia.
- Layers whose assembly is dominated by heavy multipolygon relations (e.g.
  `get_natural` / `get_boundaries` on a country file) are bounded by that
  relation step, not by the cache.
