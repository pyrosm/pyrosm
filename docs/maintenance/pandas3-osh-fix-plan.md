# Plan — Restore OSH/timestamp parsing under pandas 3.0 (keep py3.9–3.12 green)

## 1. Problem

On Python 3.11 and 3.12 the dependency stack resolves to **pandas 3.0.3**, and
3 tests fail (identical on both Python versions); on Python 3.10 (pandas 2.3.3)
all pass. The failures are confined to OSH (OpenStreetMap *history* `.osh.pbf`)
timestamp parsing:

| Test | Observed | Expected |
|------|----------|----------|
| `tests/test_network_parsing.py::test_reading_network_from_osh` | shape `(383, 27)` | `(210, 25)` |
| `tests/test_building_parsing.py::test_reading_buildings_from_osh` | `gdf.loc[0,"geometry"]` not a `Polygon` | a `Polygon` |
| `tests/test_graph_exports.py::test_nxgraph_export_from_osh` | shortest path `277.0` | one of `{470, 478}` (test: `round(...) in [478, 470]`) |

All non-history functionality passes on pandas 3.0. The regression tracks the
**pandas version (2.x vs 3.0), not the Python version.**

## 2. Root cause (reproduced and confirmed)

The OSH-only code path in `pyrosm/pbfreader.pyx` (the `if unix_time_filter is not
None:` block, lines ~419–443) selects the latest version of each element and
then cleans empty tag values:

```python
all_ways = get_latest_version(ways_df).to_dict(orient="records")   # pbfreader.pyx:429
all_ways = clean_empty_values_from_ways(all_ways)                  # pbfreader.pyx:432
```

`get_latest_version` is `df.groupby("id").last().reset_index()`
(`pyrosm/data_filter.pyx:301`). `clean_empty_values_from_ways`
(`pyrosm/data_filter.pyx:307`) strips empty tags with an **identity check**:

```python
cleaned.append({x: y for x, y in ways[i].items() if y != None})   # data_filter.pyx:311
```

**pandas 3.0 (PDEP-14) makes the string dtype the default.** Object tag columns
are now `str`/`StringDtype`, and missing values surface as `pd.NA`/`NaN` — **never
Python `None`**. Direct measurements on the two stacks (same code, same fixture
`Helsinki-test.osh.pbf`, timestamp `2010-01-01`):

- `parse_osm_data(...)` returns **401 ways / 401 unique ids on BOTH** pandas
  versions → version dedup is fine, this is **not** a `groupby.last` row bug.
- The per-record **key count diverges**: pandas 2.3.3 records carry ~33 keys;
  pandas 3.0.3 records carry ~75 keys.
- Minimal `groupby("id").last()` probe on object tag columns:
  - pandas 2.3.3: missing group cells come back as **`None`** → `y != None`
    strips them (record keeps only its real tags).
  - pandas 3.0.3: tag columns are `str` dtype, missing cells are **`pd.NA`** (no
    value is `None`) → `y != None` keeps **everything**, so each way record is
    bloated with the union of all tag keys (NA-valued).

**Cascade to the observed failures:** every OSH way record now carries a
`highway` key (NA) and dozens of other tag keys it should not. The network's
`highway` exclude-filter (`record_should_be_kept`, `data_filter.pyx`) and the
tags-as-columns expansion then admit extra ways and extra columns → `(383, 27)`
instead of `(210, 25)`. The buildings (row-0 geometry) and nx-graph (shortest
path) failures are downstream consequences of the same bloated, mis-filtered
record set.

A secondary, **non-causal** symptom: 405 `ChainedAssignmentError` warnings under
Copy-on-Write (from `networks.py:37`, `landuse.py:48`, `natural.py:48`,
`pyrosm.py:916`) and one `Pandas4Warning` for `pd.Timestamp.utcfromtimestamp`
(`utils/__init__.py:227`). These do not cause the 3 failures but are latent
CoW/deprecation risks.

## 3. Goals / non-goals

**Goals**
- The 3 OSH tests pass on pandas 3.0 (py3.11/3.12) **without regressing** pandas
  2.x (py3.9/3.10).
- The null-handling fix is dtype-agnostic so future pandas 3.x string-dtype
  behaviour does not silently corrupt records.
- Lock the invariant with a regression test and a both-pandas CI matrix.

**Non-goals**
- No public API changes, no parsing-architecture refactor, no performance work.
- Chained-assignment cleanup and the `utcfromtimestamp` deprecation are
  in-scope-optional (same PR, clearly separated), not required to make the 3
  tests pass.

## 4. Fix design

### 4.1 Primary fix (root cause) — null-aware, dtype-agnostic tag cleaning
Replace the `y != None` identity check in `clean_empty_values_from_ways`
(`data_filter.pyx:307`) with a scalar-null check that treats a tag cell as
*empty* iff it is a **missing-tag sentinel** — `None`, `NaN`, or `pd.NA` — and
otherwise keeps it, **including non-scalar cells**. The `nodes` key holds a numpy
array (real data, never a missing sentinel), and `pd.isna(arr)` on an array
returns an array (calling `bool()` on it raises), so array/list cells are kept
unconditionally. Proposed helper (Cython-compatible):

```python
import numpy as np
import pandas as pd

cdef _is_empty(v):
    # Array/list/tuple cells (e.g. the "nodes" key) are always real data here,
    # never a missing-tag sentinel -> keep them regardless of length.
    if isinstance(v, (np.ndarray, list, tuple)):
        return False
    if v is None:
        return True
    # scalar missing value: NaN-float, or pandas-3.0 string/object pd.NA
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False
```

Then `{x: y for x, y in ways[i].items() if not _is_empty(y)}`.

**Semantics (resolves the only ambiguity):** a cell is dropped iff it is a
missing-tag sentinel (`None` / `NaN` / `pd.NA`). **Real values are always kept,
including `0`, `0.0`, and empty string `""`; array cells are always kept.** This
both fixes pandas 3.0 (where missing tags are `pd.NA`, not `None`) *and*
reproduces the pandas-2.3.3 result exactly, because in 2.x missing tags were
already `None` (dropped), and any numeric-`NaN` cell is itself a missing tag that
should also be dropped — it does not change the 33-key baseline, since such cells
were not real values there either.

**Equivalence requirement (must be tested, §6):** on the OSH fixture the new
cleaner must produce the **same key-set per record as the passing pandas-2.3.3
baseline**. The test encodes that as a fixed expected key-set so it runs
identically on both pandas legs. Add explicit cases that `0` and `""` survive
and that `None`/`NaN`/`pd.NA` are dropped.

### 4.2 Nodes and relations — audit first, with a concrete output contract
`get_latest_version` is also used for nodes (`pbfreader.pyx:428`) and relations
(`:434`). Unlike ways, these do **not** pass through the dict-cleaner; they are
converted via `{col: df[col].values for col in df.columns}` (`:437`–`:438`) and
`set_index("id").to_dict(orient="index")` (`:441`).

This is an **audit-only** step unless a concrete defect is shown. Procedure and
output contract:

1. After §4.1 alone, re-run `test_reading_buildings_from_osh` and
   `test_nxgraph_export_from_osh`. **If both pass, make no code change here** —
   record §4.2 as audited/closed.
2. If either still fails, the required output contract is:
   - **Key/index columns** (`id`, `version`, `timestamp`, `changeset`) consumed
     by Cython/khash and `set_index("id")` must be real NumPy `int64` (not
     pandas nullable `Int64`/`NA`). Coerce immediately after
     `get_latest_version`, before the `.values` extraction, e.g.
     `df[int_cols] = df[int_cols].astype("int64")` (assert no NA present first).
   - **Object/string attribute columns** consumed by `create_df` and the
     geometry/relation builders must use Python `None` for missing (the
     pre-3.0 sentinel), produced **before** the `.values` / `to_dict(orient=
     "index")` conversion, e.g. normalize with
     `df = df.astype(object).where(df.notna(), None)` applied only to the
     non-array object columns (never the `nodes`/member-list columns).
   - `node_coordinates_lookup` values must therefore contain `None`, not `NA`,
     for absent attributes.
3. Add/extend a test asserting the dtype/None contract on the relations path
   (the building OSH test already exercises relations end-to-end).

### 4.3 Broader null-sentinel audit (prevent recurrence)
Grep and review every place pyrosm relies on `None` identity for pandas-derived
values: `!= None`, `== None`, `is None` on cell values, and `.to_dict(` sites.
Convert value-level null tests to `pd.isna`/`pd.notna`. Keep changes minimal and
covered by tests.

### 4.4 Optional, same PR — CoW chained-assignment cleanup
For each warning site, ensure assignments mutate an owned frame (explicit
`.copy()` or single-step `.loc[rows, col] = val`). Not required for the 3
failures, but removes latent silent-no-op risk under pandas 3.x. Gate behind a
test that asserts zero `ChainedAssignmentError` warnings in the suite.

### 4.5 Trivial — deprecation
`utils/__init__.py:227`: `pd.Timestamp.utcfromtimestamp(unix_time)` →
`pd.Timestamp.fromtimestamp(unix_time, tz="UTC")` (then drop tz or keep per the
caller's expectation; verify against `unix_time_to_datetime` callers).

## 5. Implementation steps

1. **Repro harness.** Use the two existing envs (lockfiles in
   `docs/maintenance/phase0-environment-py3{11,12}.lock.txt` for pandas 3.0;
   `…py310/…lock` analogue for pandas 2.3 via `ci/310-conda.yaml`). Confirm the
   3 OSH tests fail on pandas 3.0 and pass on 2.3 before changing code.
2. **Implement §4.1** in `data_filter.pyx`; rebuild (`pip install -e .
   --no-build-isolation`); run the 3 OSH tests on pandas 3.0 → expect pass.
3. **Run full suite on pandas 2.3.3** → expect no regression (still 106 passed).
4. **Implement §4.2/§4.3** as needed if any OSH test still diverges; re-run.
5. **Apply §4.5** (deprecation) and, if included, **§4.4** (CoW cleanup).
6. **Add regression tests** (§6).
7. **Run the full matrix** (py3.9/3.10 pandas 2.x; py3.11/3.12 pandas 3.0).
   Note: verify what pandas py3.9 actually resolves — if it pins `<3.0`, it only
   exercises the 2.x branch; that is acceptable but should be recorded.

## 6. Tests / validation

- **Invariant regression test (new):** parse the OSH fixture at a fixed
  timestamp and assert **no way record contains an NA/None-valued tag key**
  (directly guards the exact bug, independent of pandas version).
- **Cross-version key-set equivalence (new, dev-only):** on the same fixture,
  assert the cleaned record key-sets match between pandas 2.x and 3.0 (run in CI
  on both legs; the assertion is the same fixed expected set).
- Keep the existing shape/geometry/shortest-path assertions in the 3 OSH tests.
- If §4.4 is included: assert the suite emits **zero** `ChainedAssignmentError`
  warnings (e.g. `filterwarnings = error::FutureWarning` scoped to pyrosm, or an
  explicit `recwarn` check).

## 7. Risks

- **`nodes` array cell (critical):** `pd.isna` on an array raises / returns an
  array — the helper must special-case array/list cells (handled in §4.1).
- **Semantic equivalence:** the new cleaner must keep legitimate `0` / `0.0` /
  `""` tag values and drop only missing sentinels (`None` / `NaN` / `pd.NA`).
  Validate by per-record key-set equivalence against the 2.x baseline (§6). (Per
  §4.1, numeric `NaN` *is* a missing tag and is dropped on both legs — this does
  not change the 2.x baseline.)
- **Integer columns turning nullable:** pandas 3.0 may surface `id`/`version`/
  `timestamp` as nullable; downstream khash/int code requires real `int64`.
  Assert dtypes after cleaning; coerce if needed.
- **Hidden reliance on bloat:** confirm no non-OSH path secretly depended on the
  old `None`-stripping behaviour (full suite on both pandas versions covers it).

## 8. Acceptance criteria

- The 3 OSH tests pass on pandas 3.0 (py3.11 and py3.12).
- Full suite green on **both** pandas 2.3.3 and 3.0.3.
- New invariant test passes on both legs; OSH cleaned-record key-set equals the
  pandas-2.3 baseline on the fixture.
- (If §4.4 included) no `ChainedAssignmentError` warnings remain.
- `ci/39/310/311/312-conda.yaml` all resolve (already fixed) and CI runs both a
  pandas-2.x and a pandas-3.0 leg.
