# Read-path memory & speed benchmark results

Tracked results of [`memory_benchmark.py`](memory_benchmark.py): peak process
RSS and wall-clock for the multi-layer workflow `get_network` + `get_buildings`
+ `get_pois` on a single `OSM` object. Run before/after each read-path
optimization (rebuilding pyrosm in between) and append a row so the cumulative
effect stays auditable.

## Setup

- **Machine:** macOS (Darwin 25.5.0), 10 logical cores, ~26 GB RAM.
- **Dataset:** Helsinki region extract via `get_data("Helsinki")` (BBBike),
  138 MB (137,971,308 bytes), parsed 2026-06-11.
- **Metric:** peak process RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
  (OS-level, one subprocess per config — not `tracemalloc`); wall-clock via
  `time.perf_counter()`.
- **Method:** 3 repeats per config, reported as `median [min–max]`.
- **Configs:**
  - `default` — `OSM(fp)`, no `tags_to_keep` (today's behaviour).
  - `keep_metadata=False` — drop way/relation metadata columns (#87).
  - `keep_metadata=False + tags_to_keep` — additionally keep only a minimal
    tag-column set per layer: network `[highway, name, maxspeed, oneway]`,
    buildings `[building]`, POIs `[amenity, shop, name]` (#87).

## Results

| config | peak RSS (MB) | wall (s) | ΔRSS vs default | Δwall vs default |
| --- | --- | --- | --- | --- |
| `default` | 4688 [4610–4756] | 44.5 [43.8–44.8] | — | — |
| `keep_metadata=False` | 4614 [4521–4725] | 43.9 [43.6–44.5] | −73 MB (−1.6%) | −0.6 s (−1.4%) |
| `keep_metadata=False + tags_to_keep` | 4670 [4395–4679] | 36.1 [35.5–37.0] | −18 MB (−0.4%) | −8.4 s (−19.0%) |

## Reading

- **`tags_to_keep`** cuts multi-layer wall-clock by **~19%** (every repeat
  35.5–37.0 s vs 43.8–44.8 s for `default`) — fewer tag columns to resolve and
  materialise across the three layers. Peak RSS is unchanged within run-to-run
  variance (the config RSS ranges overlap): the workflow's peak is reached while
  the shared parse-once cache (`_nodes`, `_way_records`, `_node_coordinates`) is
  built, which this API-layer option does not alter. Its memory benefit is the
  smaller returned GeoDataFrames (fewer columns), not a lower workflow peak.
  Against the acceptance gate (peak RSS flat-or-down, multi-layer wall within
  ≤5%) it passes as a pure-speed change.
- **`keep_metadata=False`** leaves peak RSS and wall-clock flat within noise,
  consistent with it dropping only the small way/relation metadata columns; the
  large per-node metadata arrays are not touched by this option.
- Peak-RSS run-to-run variance at this ~4.5 GB scale is ~±3% (≈±150 MB), larger
  than the way/relation-metadata and tag-column savings, so peak-RSS deltas below
  that threshold are not significant on this workflow.
