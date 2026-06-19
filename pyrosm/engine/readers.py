"""Public out-of-core readers (one per layer). Each indexes the file's blobs, decodes
them into per-worker shards, then collects and assembles (or streams to GeoParquet) the
requested layer."""

import tempfile
import shutil

from pyrosm.utils._compat import require_pyarrow
from pyrosm.engine.blobs import _index_blobs
from pyrosm.engine.pool import _auto_workers, _decode_all
from pyrosm.engine.assemble import _assemble_buildings
from pyrosm.engine import geoparquet


def get_buildings(filepath, workers=None, output=None, keep_metadata=True):
    """Read building geometries from ``filepath`` with the out-of-core engine.

    Returns the building ways and relations with the same columns as the in-memory reader:
    every occurring ``Conf.tags.building`` tag as its own column, the remaining tags as a
    JSON ``tags`` column, and -- when ``keep_metadata`` -- the ``version`` / ``timestamp``
    / ``changeset`` / ``visible`` element metadata.

    With ``output=None`` (the default) returns an in-memory GeoDataFrame. With ``output``
    set to a path, the buildings are streamed to a GeoParquet file in chunks (never fully
    materialised) and the path is returned; this needs the optional ``pyarrow`` dependency.

    ``workers`` defaults to one for small files (no multiprocessing overhead) and
    otherwise to a worker per CPU, bounded by the blob count.
    """
    if output is not None:
        require_pyarrow()

    data_blobs = [
        (offset, size)
        for (blob_type, offset, size) in _index_blobs(filepath)
        if blob_type == "OSMData"
    ]
    if workers is None:
        workers = _auto_workers(filepath, len(data_blobs))

    shard_dir = tempfile.mkdtemp(prefix="pyrosm_ooc_")
    try:
        shard_paths = _decode_all(filepath, data_blobs, workers, shard_dir)
        if output is None:
            return _assemble_buildings(shard_paths, keep_metadata)
        return geoparquet._stream_buildings_to_parquet(
            shard_paths, output, geoparquet._OUTPUT_CHUNK_SIZE, keep_metadata
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
