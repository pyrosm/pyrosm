"""Out-of-core PBF reading engine (the ``engine="out_of_core"`` backend).

The file is read in one pass: data blobs are decoded in parallel with the raw Cython
``primitive_block_decoder`` (protobuf is used only for the small ``BlobHeader`` / ``Blob``
framing), each decode task spills the node coordinates and the features it finds to
shards on disk, and the main process then gathers only the coordinates the kept
features reference and assembles the geometries. Peak memory is bounded by the working
set rather than the whole file.

The public ``get_*`` readers re-exported here mirror the in-memory reader's output
column-for-column.

By default (``workers="auto"``) the engine chooses the thread count by the file's size: a
single core for files below about 20 MB; for larger files the CPU cores this process may
use less two, at most 4 below about 70 MB and at most one per data blob. Pass
``workers=N`` for an explicit count (reduced to the cores this process may use, with a
warning, if it exceeds them). The workers are threads, so a parallel read needs no
``if __name__ == "__main__":`` guard. With ``PYROSM_DECODE_BACKEND=processes``
(deprecated: it warns, and it will be removed in a future release) the default is one
core, and ``"auto"`` or ``workers=N`` start worker processes, at most eight with
``"auto"``; on macOS and Windows those start with ``spawn`` and re-import the program's
entry point, so a parallel read that runs at import time must then be placed under the
guard. Without it the read still completes -- it falls back to a single process and emits
a warning -- but it is not parallel.
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
