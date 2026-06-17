"""Phase 1 parity: the streaming buildings reader vs OSM(fp).get_buildings() way rows."""

import zlib
from struct import pack, unpack

import geopandas as gpd
import numpy as np
import pytest
import shapely

from pyrosm import OSM, get_data
from pyrosm import streaming
from pyrosm.proto.fileformat_pb2 import BlobHeader, Blob


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def _rewrite_pbf_raw(src, dst):
    """Re-encode every blob of ``src`` with an uncompressed ``raw`` payload. Bundled
    extracts are all zlib, so this synthesises a file that exercises the streaming
    reader's ``raw`` decompression branch."""
    with open(src, "rb") as f, open(dst, "wb") as out:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = unpack("!L", head)
            bh = BlobHeader()
            bh.ParseFromString(f.read(n))
            blob = Blob()
            blob.ParseFromString(f.read(bh.datasize))
            raw = (
                zlib.decompress(blob.zlib_data)
                if blob.HasField("zlib_data")
                else blob.raw
            )
            new_blob = Blob()
            new_blob.raw = raw
            blob_bytes = new_blob.SerializeToString()
            bh.datasize = len(blob_bytes)
            header_bytes = bh.SerializeToString()
            out.write(pack("!L", len(header_bytes)))
            out.write(header_bytes)
            out.write(blob_bytes)


def _untiled_way_buildings(fp):
    gdf = OSM(fp).get_buildings()
    gdf = gdf[gdf["osm_type"] == "way"]
    return gdf.sort_values("id").reset_index(drop=True)


def _assert_matches(mine, ref):
    a = mine.sort_values("id").reset_index(drop=True)
    # Same building way ids.
    np.testing.assert_array_equal(a["id"].to_numpy(), ref["id"].to_numpy())
    # Same geometries, exact coordinates (order-canonical via normalize).
    na = gpd.GeoSeries(shapely.normalize(a.geometry.values))
    nb = gpd.GeoSeries(shapely.normalize(ref.geometry.values))
    assert na.geom_equals_exact(nb, tolerance=0).all()


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_buildings_match_untiled(fixture, request):
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_buildings(fp)
    ref = _untiled_way_buildings(fp)
    assert mine is not None and len(mine) == len(ref) > 0
    _assert_matches(mine, ref)


def test_streaming_buildings_parallel_matches_single(helsinki_pbf):
    # The multiprocessing path must produce the same result as the in-process path.
    single = streaming.get_buildings(helsinki_pbf, workers=1)
    parallel = streaming.get_buildings(helsinki_pbf, workers=3)
    _assert_matches(parallel, single.sort_values("id").reset_index(drop=True))


def test_streaming_buildings_raw_blob_matches(helsinki_pbf, tmp_path):
    # Uncompressed `raw` blobs must decode to the same result as the zlib originals.
    raw_fp = str(tmp_path / "helsinki_raw.osm.pbf")
    _rewrite_pbf_raw(helsinki_pbf, raw_fp)
    zlib_buildings = streaming.get_buildings(helsinki_pbf)
    raw_buildings = streaming.get_buildings(raw_fp)
    _assert_matches(
        raw_buildings, zlib_buildings.sort_values("id").reset_index(drop=True)
    )
