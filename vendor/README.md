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
