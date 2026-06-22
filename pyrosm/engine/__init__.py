"""Out-of-core PBF reading engine (the ``engine="out_of_core"`` backend).

The file is read in one pass: data blobs are decoded in parallel with the raw Cython
``primitive_block_decoder`` (protobuf is used only for the small ``BlobHeader`` / ``Blob``
framing), each worker spills the node coordinates and the features it finds to a
per-worker shard on disk, and the main process then gathers only the coordinates the kept
features reference and assembles the geometries. Peak memory is bounded by the working
set rather than the whole file.

The public ``get_*`` readers re-exported here mirror the in-memory reader's output
column-for-column.

Parallel reading and the ``__main__`` guard: the engine reads on a single core by default.
Pass ``workers="auto"`` to choose the count automatically (a single core for small files,
one per CPU core for larger files), or ``workers=N`` for an explicit count (reduced to the
CPU-core count, with a warning, if it exceeds it). On macOS and
Windows those workers start with ``spawn`` and re-import the program's entry point, so a
parallel read that runs at import time must be placed under an ``if __name__ == "__main__":``
guard. Without the guard a parallel read still completes -- it falls back to a single process
and emits a warning -- but it is not parallel. On Linux (``fork``) no guard is needed; the
default single-core read needs none anywhere.
"""

from pyrosm.engine.readers import (
    get_buildings,
    get_landuse,
    get_natural,
    get_pois,
    get_boundaries,
    get_data_by_custom_criteria,
    get_network,
)

__all__ = [
    "get_buildings",
    "get_landuse",
    "get_natural",
    "get_pois",
    "get_boundaries",
    "get_data_by_custom_criteria",
    "get_network",
]
