"""Out-of-core PBF reading engine (the ``engine="out_of_core"`` backend).

The file is read in one pass: data blobs are decoded in parallel with the raw Cython
``primitive_block_decoder`` (protobuf is used only for the small ``BlobHeader`` / ``Blob``
framing), each worker spills the node coordinates and the features it finds to a
per-worker shard on disk, and the main process then gathers only the coordinates the kept
features reference and assembles the geometries. Peak memory is bounded by the working
set rather than the whole file.

The public ``get_*`` readers re-exported here mirror the in-memory reader's output
column-for-column.
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
