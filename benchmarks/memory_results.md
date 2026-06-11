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
