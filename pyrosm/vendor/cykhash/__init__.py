"""Vendored cykhash: the khash-backed int64 set and map pyrosm reads OSM ids with.

pyrosm imports the two extension modules directly (``pyrosm.vendor.cykhash.khashsets``
and ``.khashmaps``), so this initializer only makes the directory a package. Upstream's
own initializer re-exports both modules and additionally imports ``unique`` and
``utils``, which are not vendored. See ../README.md.
"""
