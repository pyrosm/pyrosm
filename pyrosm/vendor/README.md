# Vendored third-party code

## cykhash

- **Upstream:** https://github.com/realead/cykhash
- **Version:** 2.0.1 (PyPI sdist, released 2023-02-05)
- **License:** MIT, see [cykhash/LICENSE](cykhash/LICENSE)

pyrosm reads OSM ids through cykhash's khash-backed int64 set and int64->int64 map:
`Int64Set`, `Int64Set_from_buffer`, `isin_int64` and `any_int64_from_iter` from
`khashsets`, and `Int64toInt64Map`, `Int64toInt64Map_from_buffers` and
`Int64toInt64Map_to` from `khashmaps`.

It is vendored because cykhash publishes no wheels -- every release on PyPI is an sdist --
so depending on it meant every `pip install pyrosm` compiled it from source, and needed a
C compiler even on the platforms where pyrosm ships a binary wheel.

### What was copied

The `khashsets` and `khashmaps` modules and the files they include, from `src/cykhash/` of
the sdist:

```
khashsets.pyx  khashsets.pxd  khashmaps.pyx  khashmaps.pxd  floatdef.pxd
common.pxi  memory.pxi  khash.pxi  murmurhash.pxi  hash_functions.pxi
sets/set_header.pxi  sets/set_impl.pxi  sets/set_init.pxi
maps/map_header.pxi  maps/map_impl.pxi  maps/map_init.pxi
```

Every one of those is byte-for-byte upstream. The `.pxi` files ship pre-generated in the
sdist, so upstream's Tempita templating step is not needed here.

**The one exception is `cykhash/__init__.py`**, which is not upstream's: pyrosm imports the
two extension modules directly, so the initializer here is a docstring that makes the
directory a package. Upstream's re-exports both modules and additionally imports `unique`
and `utils`, which pyrosm does not use and which are not vendored.

`unique.pyx`, `utils.pyx`, the `.pxi.in` templates and upstream's own packaging files are
left out.

### Updating

Download the sdist of the new version, copy the files listed above over the ones here,
rewrite `__init__.py` the same way, and update the version above. Nothing else in pyrosm
refers to the vendored files except the imports in `data_filter.pyx`, `node_lookup.pyx`,
`node_lookup.pxd` and `pbf_export.pyx`.
