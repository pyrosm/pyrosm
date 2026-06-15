# Pyrosm PBF backend: pyrobuf → Google protobuf

Pyrosm decodes `*.osm.pbf` files via Protocol Buffers. This change replaces the
unmaintained [`pyrobuf`](https://github.com/appnexus/pyrobuf) backend with
Google's [`protobuf`](https://protobuf.dev/) library and its C `upb`
implementation. This page shows the impact on parsing speed and on
cross-version compatibility.

## TL;DR

- **No speed regression.** protobuf (`upb`) performs on par with pyrobuf across
  Python 3.10–3.14; differences are within run-to-run noise.
- **The fast C backend (`upb`) is active on every Python version, 3.10–3.14**
  (including 3.14), so the slow pure-Python path never kicks in.
- **Better forward compatibility.** protobuf ships maintained wheels and
  conda-forge builds for current Python; pyrobuf is unmaintained (last release
  0.9.3) and its source build fails under modern setuptools.

## Method

- Input: `helsinki_region_pbf` (`Helsinki_region.osm.pbf`, 36.7 MB, 478,485 ways)
  fetched via `pyrosm.get_data("helsinki_region_pbf")`.
- Machine: macOS arm64 (Apple Silicon). conda-forge packages.
- One untimed warm-up, then 5 timed repetitions per stage; medians reported.
- Identical pyrosm source for both backends (differs only in the protobuf
  import + generated message modules).

## Backend availability (conda-forge, per Python version)

| Python | protobuf | backend impl | pyrobuf |
| ------ | -------- | ------------ | ------- |
| 3.10   | 7.34.2   | **upb**      | 0.9.3   |
| 3.11   | 7.34.2   | **upb**      | 0.9.3   |
| 3.12   | 7.34.2*  | **upb**      | 0.9.3   |
| 3.13   | 7.34.2   | **upb**      | 0.9.3   |
| 3.14   | 7.34.2   | **upb**      | 0.9.3   |

conda-forge serves the same current protobuf (7.34.2) for every supported
Python — no version is forced onto an older protobuf. The vendored message
modules are generated against protobuf 6.33.5 and run unchanged on the newer
7.x runtime. (*The 3.12 benchmark below was run on protobuf 6.33.5, confirming
both 6.x and 7.x work.)

## Benchmark (median seconds, lower is better)

`ratio` = protobuf ÷ pyrobuf; values near 1.0 indicate comparable speed.

| Python | `parse_osm_data` pyrobuf | protobuf | ratio | `get_network` pyrobuf | protobuf | ratio |
| ------ | -----: | -----: | :--: | -----: | -----: | :--: |
| 3.10   | 8.75 | 7.05 | 0.81 | 13.48 | 12.81 | 0.95 |
| 3.11   | 8.28 | 6.80 | 0.82 | 12.56 | 11.48 | 0.91 |
| 3.12   | 6.74 | 7.03 | 1.04 | 12.26 | 12.49 | 1.02 |
| 3.13   | 6.86 | 6.95 | 1.01 | 12.08 | 11.99 | 0.99 |
| 3.14   | 7.84 | 7.88 | 1.00 | 12.53 | 12.20 | 0.97 |

`get_buildings` tracks `get_network` (ratios 0.87–1.02) and is omitted for
brevity.

## Reading the numbers

- **`parse_osm_data`** is the pure parse + array-extraction stage — the part the
  protobuf backend actually drives. protobuf and pyrobuf land close together on
  every version; the larger gaps on 3.10–3.11 most likely reflect run-to-run
  variance rather than a consistent advantage for either backend.
- **`get_network`** is end-to-end (parse + Shapely geometry + GeoDataFrame
  assembly). Geometry assembly dominates wall time and is backend-independent,
  so the backend swap barely moves the total — ratios sit around 1.0.
- The overall picture is parity: no version shows a meaningful protobuf
  slowdown, and we make no claim that protobuf is faster.

## Why migrate, given parity?

Speed was never the risk — compatibility was. pyrobuf is unmaintained and its
PyPI source build breaks on modern setuptools, which blocks `pip install pyrosm`
on current toolchains. Google protobuf is actively maintained, ships prebuilt
wheels and conda-forge packages for Python 3.10–3.14, and keeps the fast `upb`
backend available everywhere we tested. The migration removes a fragile,
abandoned dependency with no measurable performance cost.
