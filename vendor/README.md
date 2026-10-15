# Vendored libraries

C code from other projects that pyrosm compiles into its extension modules. Each library
sits in its own folder with its licence. The files are copied from upstream unchanged and
keep their upstream paths; to update a library, copy the same files from a newer release
and update its entry below.

## khash

- What: the hash tables behind `pyrosm.int64_hash` (`Int64Set`, `SortedInt64Set`,
  `Int64Map`).
- Upstream: klib, https://github.com/attractivechaos/klib, `khash.h` version 0.2.8.
- Licence: MIT, `khash/LICENSE` (also at the top of `khash.h`).
- Files: `khash.h`.

## libdeflate

- What: zlib compression and decompression of PBF blobs, through `pyrosm.zlib_blob`.
- Upstream: https://github.com/ebiggers/libdeflate, release 1.26 (commit 92e6a0d).
- Licence: MIT, `libdeflate/COPYING`.
- Files: the ones zlib compression and decompression need, at their upstream paths:
  `libdeflate.h`, `common_defs.h`, and in `lib/`: `adler32.c`, `bt_matchfinder.h`,
  `cpu_features_common.h`, `decompress_template.h`, `deflate_compress.c`,
  `deflate_compress.h`, `deflate_constants.h`, `deflate_decompress.c`,
  `hc_matchfinder.h`, `ht_matchfinder.h`, `lib_common.h`, `matchfinder_common.h`,
  `utils.c`, `zlib_compress.c`, `zlib_constants.h`, `zlib_decompress.c`, with
  `arm/adler32_impl.h`, `arm/cpu_features.c`, `arm/cpu_features.h`,
  `arm/matchfinder_impl.h`, `riscv/matchfinder_impl.h`, `x86/adler32_impl.h`,
  `x86/adler32_template.h`, `x86/cpu_features.c`, `x86/cpu_features.h`,
  `x86/decompress_impl.h` and `x86/matchfinder_impl.h`.
