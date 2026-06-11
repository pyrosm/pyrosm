# Read-path memory & speed benchmark results

Tracked results of [`memory_benchmark.py`](memory_benchmark.py): peak process
RSS and wall-clock for the multi-layer workflow `get_network` + `get_buildings`
+ `get_pois` on a single `OSM` object. One measurement session is run
before/after each read-path optimization (rebuilding pyrosm in between) and
appended below, so the cumulative effect stays auditable.

## Setup

- **Machine:** macOS (Darwin 25.5.0), 10 logical cores, ~26 GB RAM.
- **Dataset:** Helsinki region extract via `get_data("Helsinki")` (BBBike),
  138 MB (137,971,308 bytes).
- **Metric:** peak process RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
  (OS-level, one subprocess per config — not `tracemalloc`); wall-clock via
  `time.perf_counter()`. Reported as `median [min–max]` over the session's repeats.
- **Configs:**
  - `default` — `OSM(fp)` (today's behaviour).
  - `keep_metadata=False` — drop element metadata (`timestamp`/`version`/
    `changeset`); from #150 this also skips decoding the per-node metadata.
  - `keep_metadata=False + tags_to_keep` — additionally keep only a minimal
    tag-column set per layer: network `[highway, name, maxspeed, oneway]`,
    buildings `[building]`, POIs `[amenity, shop, name]`.
- **Note:** absolute peak RSS drifts a few percent between sessions on this
  shared machine, so compare configs **within** a session, not across them.

## Sessions

### `tags_to_keep` added (#87) — 3 repeats

| config | peak RSS (MB) | wall (s) | ΔRSS vs default | Δwall vs default |
| --- | --- | --- | --- | --- |
| `default` | 4688 [4610–4756] | 44.5 [43.8–44.8] | — | — |
| `keep_metadata=False` | 4614 [4521–4725] | 43.9 [43.6–44.5] | −73 MB (−1.6%) | −0.6 s (−1.4%) |
| `keep_metadata=False + tags_to_keep` | 4670 [4395–4679] | 36.1 [35.5–37.0] | −18 MB (−0.4%) | −8.4 s (−19.0%) |

- **`tags_to_keep`** cuts multi-layer wall-clock by **~19%** (every repeat
  35.5–37.0 s vs 43.8–44.8 s) — fewer tag columns to resolve/materialise across
  the three layers. Peak RSS is flat within run-to-run variance: the workflow
  peak is reached building the shared parse-once cache, which this API-layer
  option does not alter; its memory benefit is the smaller returned GeoDataFrames.
- **`keep_metadata=False`** here drops only way/relation metadata (per-node
  metadata is still decoded), so peak RSS and wall stay flat within noise.

### Per-node metadata skip (#150) — 5 repeats

`keep_metadata=False` now also skips decoding the per-node `version`/`changeset`/
`timestamp` while parsing.

| config | peak RSS (MB) | wall (s) | ΔRSS vs default | Δwall vs default |
| --- | --- | --- | --- | --- |
| `default` | 4964 [4819–5013] | 43.1 [42.5–44.6] | — | — |
| `keep_metadata=False` | 4740 [4673–5066] | 40.0 [39.8–40.5] | −224 MB (−4.5%) | −3.1 s (−7.2%) |
| `keep_metadata=False + tags_to_keep` | 4646 [4583–4678] | 33.0 [32.9–33.4] | −318 MB (−6.4%) | −10.1 s (−23.4%) |

- **`keep_metadata=False`** is now a clear parse-time win: wall **−7.2%** with no
  range overlap (39.8–40.5 s vs 42.5–44.6 s), from skipping the per-node
  version/changeset/timestamp decode — versus ~neutral for the same config before
  this change. Peak RSS drops ~**224 MB** (−4.5%) at the median; the saving is the
  per-node metadata arrays no longer held in the parse cache. It scales with node
  count, so on this extract it is of the same order as the ±~4% run-to-run RSS
  variance (one repeat spiked to 5066 MB). Passes the acceptance gate (RSS down,
  wall faster).
- **`tags_to_keep`** on top keeps its per-layer speedup; the two stack to −23% wall.
- Peak-RSS run-to-run variance at this ~5 GB scale is ~±4% (≈±180 MB), so RSS
  deltas are read at the median and the wall-clock is the cleaner signal.

### Compact node-coordinate store (#53) — `master` vs branch, default config, 3 repeats

This measure changes the **default** path (it replaces the per-node
dict-of-dicts coordinate lookup with a cykhash id→index map plus column arrays),
so it is measured as `master` vs the branch on the **default** config, built
back-to-back in one session.

| build | peak RSS (MB) | wall (s) | ΔRSS | Δwall |
| --- | --- | --- | --- | --- |
| `master` (default) | 5084 [4693–5200] | 42.6 [42.1–44.8] | — | — |
| branch (default) | 4493 [4353–4567] | 39.1 [38.6–44.3] | −591 MB (−11.6%) | −3.5 s (−8.2%) |

- The default-path peak RSS drops **~590 MB (−11.6%)** with no range overlap
  (branch 4353–4567 MB vs master 4693–5200 MB) — the dict-of-dicts (one Python
  dict per node, each holding boxed scalars) was a major share of the parse-once
  cache; the compact arrays remove that overhead. Wall-clock is ~8% faster
  (C-level khash lookups replace per-node Python dict hashing). This is the
  largest peak-RSS reduction in the sequence so far and it benefits every config
  (all layers read coordinates through the same store).
- Output is unchanged: every getter's column set, per-column dtype, per-column
  values, row count and geometry match `master` (verified by an order-independent
  fingerprint; the node-feature column *order* is nondeterministic on `master`
  itself and is unaffected by this change).

### Batched network geometry (#53) — `master` vs branch, default config, 3 repeats

Builds each network way's per-segment LineStrings with one batched
`shapely.linestrings` call across ways instead of one call per way, and skips the
from/to ids + node-attribute records that plain `get_network()` discards.

| build | peak RSS (MB) | wall (s) | ΔRSS | Δwall |
| --- | --- | --- | --- | --- |
| `master` (default) | 4300 [4232–4350] | 39.0 [38.9–40.3] | — | — |
| branch (default) | 4369 [4270–4378] | 32.3 [32.2–32.4] | +69 MB (+1.6%) | −6.6 s (−17%) |

- Multi-layer wall-clock drops **~17%** with no range overlap (32.2–32.4 s vs
  38.9–40.3 s). Measured in isolation, the network geometry construction itself
  falls **15.96 s → 10.00 s (−37%)** (same session, parse cached): the batched
  Shapely call replaces per-way construction, and plain `get_network` no longer
  builds the per-node records it would discard. Peak RSS is flat within the ±~4%
  run-to-run noise (the transient batched coordinate/segment arrays roughly offset
  the avoided per-feature object churn). Passes the gate (RSS flat, wall faster).
- Output is unchanged: every getter (incl. walking/driving/cycling networks and
  the graph nodes/edges) matches `master` by the order-independent fingerprint.

### Streaming block read (#53) — `master` vs branch, by bounding-box size, 1 run each

The block reader becomes a generator: each PrimitiveBlock is parsed and discarded
instead of holding the whole decompressed file in a list. Peak RSS + wall for
`OSM(fp, bbox)._read_pbf()` + `get_network()`, one subprocess per box (boxes are
centred sub-boxes of the data extent; output byte-identical, full suite passes).

| config | rows kept | peak RSS master → branch | Δ memory | wall master → branch | Δ wall |
| --- | --- | --- | --- | --- | --- |
| full (no bbox) | 469,800 | 3887 → 3068 MB | −819 MB (−21%) | 19.9 → 20.6 s | ~neutral |
| bbox 75% | 315,400 | 3560 → 2472 MB | −1088 MB (−31%) | 68.5 → 74.4 s | +8.6% |
| bbox 50% | 153,800 | 3452 → 1709 MB | −1743 MB (−50%) | 35.0 → 38.3 s | +9.3% |
| bbox 25% | 46,400 | 3140 → 1269 MB | −1871 MB (−60%) | 26.4 → 28.4 s | +7.5% |
| bbox 10% | 10,600 | 2912 → 939 MB | −1973 MB (−68%) | 12.1 → 13.5 s | +11.8% |

- **Memory is the win, and it grows as the box shrinks.** `master` loads the
  entire decompressed file into a list regardless of the box, so a 10% box still
  costs ~2.9 GB; streaming makes memory scale with what is kept (~0.9 GB) —
  **−68%**. Whole-file is a more modest −21% (there the resident parse cache,
  built either way, dominates the peak). The parse step alone was marginally
  *faster* streaming (9.6 s vs 9.9 s).
- **Speed: neutral for whole-file; a bounded +8–12% for bounding-box reads** —
  the boundary-node pass re-reads/re-decompresses the file because the blocks are
  no longer retained. (Bounding-box reads are already several times slower than
  whole-file in pyrosm independent of this change — e.g. master 68 s for a 75%
  box vs 20 s whole-file — so this adds ~10% on top of an existing cost.)

### Build only non-empty node tag-columns (#53) — `master` vs branch, default config, 3 repeats

Node tag-splitting built *every* candidate tag-column (mostly all-`None`) and then
dropped the empty ones with a per-column `set()` check; now it builds only the
columns that actually occur, in one pass.

| build | peak RSS (MB) | wall (s) | ΔRSS | Δwall |
| --- | --- | --- | --- | --- |
| `master` (default) | 3745 [3680–3801] | 33.4 [33.0–33.8] | — | — |
| branch (default) | 3807 [3772–4081] | 28.8 [28.7–29.1] | +62 MB (+1.7%) | −4.6 s (−13.7%) |

- Multi-layer wall-clock drops **~14%** with no range overlap. Isolated,
  `get_pois` assembly falls **9.31 s → 5.81 s (−37.6%)**; in a microbenchmark the
  node tag-handling step alone is **6.2×** (4.13 s → 0.67 s) because the previous
  build-all-then-`set`-drop did two O(elements × candidate-columns) passes over
  ~250 columns to keep ~40. Peak RSS is flat within noise. Output byte-identical
  (order-independent fingerprint across pois/buildings/landuse/natural/network on
  two datasets).
- The way path (`convert_way_records_to_lists` + `convert_to_arrays_and_drop_empty`)
  has the same build-all-then-drop pattern and would benefit similarly; left as a
  follow-up.

### Build only non-empty way/relation tag-columns (#53) — `master` vs branch, 138 MB extract

The way/relation follow-up to the node change: `convert_way_records_to_lists`
built *every* candidate tag-column per way (a fresh `dict.fromkeys(candidates)`
plus one append per candidate, n times) and `convert_to_arrays_and_drop_empty`
then dropped the all-`None` ones; now it builds only the columns that occur, in
one pass. `convert_to_arrays_and_drop_empty` is unchanged (its all-`None` drop
stays as a safety net but now sees only kept columns).

Way features have far fewer candidate tag-columns than the POI case (building 38,
landuse 35, natural 41, vs ~250 for `amenity`+`shop`), so the win is smaller. The
tag-assembly step (`get_osm_data`, geometry excluded) on the cached arrays:

| feature | `master` (s) | branch (s) | Δ |
| --- | --- | --- | --- |
| `get_osm_data` building | 0.703 | 0.620 | −11.7% |
| `get_osm_data` landuse | 0.436 | 0.368 | −15.6% |
| `get_osm_data` natural | 0.222 | 0.190 | −14.3% |

- The full feature call improves less (`get_buildings` ~−8%, `get_landuse` ~−9%,
  `get_network` ~−2%) because polygon/line geometry construction dominates those
  getters and is unchanged. Peak RSS flat within noise.
- Output byte-identical: across `get_network`/`get_buildings`/`get_landuse`/
  `get_natural`/POIs/custom-criteria on two datasets with `keep_metadata` True and
  False, every column's dtype, values and geometry match `master` (order-independent
  fingerprint; the only run-to-run variation is node-feature column *order*, which
  already varies on `master` and is unaffected here).
