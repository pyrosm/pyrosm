"""Streaming reader parity vs OSM(fp).get_buildings() way rows, plus output= GeoParquet."""

import sys
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


@pytest.fixture
def helsinki_region_pbf():
    return get_data("helsinki_region_pbf")


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


def _untiled_buildings(fp):
    # Way and relation building rows from the in-memory reader (the streaming backend
    # must reproduce both). osm_type makes the comparison key unambiguous since a way and
    # a relation can share an id number.
    gdf = OSM(fp).get_buildings()
    return gdf.sort_values(["osm_type", "id"]).reset_index(drop=True)


def _assert_matches(mine, ref):
    a = mine.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = ref.sort_values(["osm_type", "id"]).reset_index(drop=True)
    # Same element kinds and ids.
    np.testing.assert_array_equal(a["osm_type"].to_numpy(), b["osm_type"].to_numpy())
    np.testing.assert_array_equal(a["id"].to_numpy(), b["id"].to_numpy())
    # Same geometries, exact coordinates (order-canonical via normalize).
    na = gpd.GeoSeries(shapely.normalize(a.geometry.values))
    nb = gpd.GeoSeries(shapely.normalize(b.geometry.values))
    assert na.geom_equals_exact(nb, tolerance=0).all()


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_buildings_match_untiled(fixture, request):
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_buildings(fp)
    ref = _untiled_buildings(fp)
    assert mine is not None and len(mine) == len(ref) > 0
    _assert_matches(mine, ref)


def _assert_full_parity(mine, ref):
    import pandas as pd

    a = mine.sort_values(["osm_type", "id"]).reset_index(drop=True)
    b = ref.sort_values(["osm_type", "id"]).reset_index(drop=True)
    assert set(a.columns) == set(b.columns), set(a.columns).symmetric_difference(
        b.columns
    )
    a = a[b.columns]
    na = gpd.GeoSeries(shapely.normalize(a.geometry.values))
    nb = gpd.GeoSeries(shapely.normalize(b.geometry.values))
    assert na.geom_equals_exact(nb, tolerance=0).all()
    for col in b.columns:
        if col == "geometry":
            continue
        pd.testing.assert_series_equal(
            a[col], b[col], check_dtype=False, check_names=False
        )


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_buildings_full_column_parity(fixture, request):
    # Every tag column, the JSON 'tags' column and the metadata columns must match
    # OSM().get_buildings() column-for-column and value-for-value.
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_buildings(fp)
    ref = OSM(fp).get_buildings()
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_landuse_full_column_parity(fixture, request):
    # The generalized layer reader must match OSM().get_landuse() (a different layer:
    # different filter key and tag columns) column-for-column and value-for-value.
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_landuse(fp)
    ref = OSM(fp).get_landuse()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_natural_full_column_parity(fixture, request):
    # natural includes NODE features (points), so this exercises the node path too.
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_natural(fp)
    ref = OSM(fp).get_natural()
    if ref is None:
        assert mine is None
        return
    n_node = int((ref["osm_type"] == "node").sum())
    assert int((mine["osm_type"] == "node").sum()) == n_node
    _assert_full_parity(mine, ref)


def test_streaming_natural_has_node_rows(helsinki_pbf):
    # The bundled Helsinki extract has natural node features; assert they are produced.
    mine = streaming.get_natural(helsinki_pbf)
    assert int((mine["osm_type"] == "node").sum()) > 0


@pytest.mark.parametrize("fixture", ["test_pbf", "helsinki_pbf"])
def test_streaming_pois_default_parity(fixture, request):
    # The default POI filter ({"amenity": True, "shop": True, "tourism": True}) over
    # nodes + ways + relations.
    fp = request.getfixturevalue(fixture)
    mine = streaming.get_pois(fp)
    ref = OSM(fp).get_pois()
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize("network_type", ["walking", "driving", "cycling"])
def test_streaming_network_parity(network_type, helsinki_pbf):
    # Street network: highway ways as LineString edges + a 'length' column, via the
    # predefined exclude filters.
    mine = streaming.get_network(helsinki_pbf, network_type=network_type)
    ref = OSM(helsinki_pbf).get_network(network_type=network_type)
    assert ref is not None and len(ref) > 0
    _assert_full_parity(mine, ref)


def _central_bbox(gdf, frac=0.4):
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = (1 - frac) / 2
    return [
        minx + (maxx - minx) * pad,
        miny + (maxy - miny) * pad,
        minx + (maxx - minx) * (1 - pad),
        miny + (maxy - miny) * (1 - pad),
    ]


@pytest.mark.parametrize("complete_relations", [False, True])
def test_streaming_buildings_bounding_box_parity(helsinki_pbf, complete_relations):
    # A bounding box restricts buildings (ways + relations) to that area; relations are
    # partial by default, complete with complete_relations=True -- matching the in-memory
    # reader either way.
    bbox = _central_bbox(OSM(helsinki_pbf).get_buildings())
    mine = streaming.get_buildings(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    )
    ref = OSM(
        helsinki_pbf, bounding_box=bbox, complete_relations=complete_relations
    ).get_buildings()
    assert ref is not None and (ref["osm_type"] == "relation").any()
    _assert_full_parity(mine, ref)


def test_streaming_pois_bounding_box_parity(helsinki_pbf):
    # bbox over a layer with node features (POIs): nodes + ways restricted to the box.
    bbox = _central_bbox(OSM(helsinki_pbf).get_pois())
    mine = streaming.get_pois(helsinki_pbf, bounding_box=bbox)
    ref = OSM(helsinki_pbf, bounding_box=bbox).get_pois()
    assert ref is not None and (ref["osm_type"] == "node").any()
    _assert_full_parity(mine, ref)


def test_streaming_network_bounding_box_parity(helsinki_pbf):
    # A bounding box restricts the network to ways with >=1 node inside it (kept whole),
    # then the final spatial filter clips to the box -- matching the in-memory reader.
    full = OSM(helsinki_pbf).get_network(network_type="driving")
    minx, miny, maxx, maxy = full.total_bounds
    bbox = [
        minx + (maxx - minx) * 0.25,
        miny + (maxy - miny) * 0.25,
        minx + (maxx - minx) * 0.75,
        miny + (maxy - miny) * 0.75,
    ]
    ref = OSM(helsinki_pbf, bounding_box=bbox).get_network(network_type="driving")
    mine = streaming.get_network(
        helsinki_pbf, network_type="driving", bounding_box=bbox
    )
    assert ref is not None and 0 < len(ref) < len(full)
    _assert_full_parity(mine, ref)


def test_streaming_network_custom_filter_parity(helsinki_pbf):
    flt = {"highway": ["footway", "path", "pedestrian"]}
    mine = streaming.get_network(helsinki_pbf, custom_filter=flt, filter_type="keep")
    ref = OSM(helsinki_pbf).get_network(custom_filter=flt, filter_type="keep")
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


def test_streaming_boundaries_parity(helsinki_region_pbf):
    # The Helsinki region extract has administrative boundaries (relations + ways).
    mine = streaming.get_boundaries(helsinki_region_pbf)
    ref = OSM(helsinki_region_pbf).get_boundaries()
    assert ref is not None and (ref["osm_type"] == "relation").any()
    _assert_full_parity(mine, ref)


def test_streaming_custom_criteria_value_filter_parity(helsinki_pbf):
    flt = {"amenity": ["restaurant", "cafe", "pub"]}
    mine = streaming.get_data_by_custom_criteria(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=flt)
    assert mine is not None and len(mine) > 0
    _assert_full_parity(mine, ref)


@pytest.mark.parametrize(
    "flags",
    [
        {"keep_nodes": False},
        {"keep_ways": False},
        {"keep_relations": False},
        {"keep_nodes": False, "keep_relations": False},
    ],
)
def test_streaming_custom_criteria_keep_flags_parity(helsinki_pbf, flags):
    # keep_nodes / keep_ways / keep_relations must select element kinds exactly as the
    # in-memory reader.
    flt = {"amenity": True}
    mine = streaming.get_data_by_custom_criteria(
        helsinki_pbf, custom_filter=flt, **flags
    )
    ref = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=flt, **flags)
    if ref is None:
        assert mine is None
        return
    _assert_full_parity(mine, ref)


def test_streaming_pois_value_filter_parity(helsinki_pbf):
    # A value-level custom_filter must refine to the exact values (not just key presence),
    # matching OSM().get_pois(custom_filter=...).
    flt = {"amenity": ["restaurant", "cafe", "bar"]}
    mine = streaming.get_pois(helsinki_pbf, custom_filter=flt)
    ref = OSM(helsinki_pbf).get_pois(custom_filter=flt)
    assert mine is not None and len(mine) > 0
    # Every kept element really has one of the requested amenity values.
    assert mine["amenity"].isin(flt["amenity"]).all()
    _assert_full_parity(mine, ref)


def test_streaming_buildings_keep_metadata_false_parity(helsinki_pbf):
    # keep_metadata=False must drop the element-metadata columns exactly as the in-memory
    # reader does (whatever its handling) -- full column + value parity either way.
    mine = streaming.get_buildings(helsinki_pbf, keep_metadata=False)
    ref = OSM(helsinki_pbf, keep_metadata=False).get_buildings()
    assert "changeset" not in mine.columns
    _assert_full_parity(mine, ref)


def test_streaming_building_relations_match(helsinki_pbf):
    # The bundled Helsinki extract has building relations (multipolygons); they must
    # match the in-memory reader's relation rows exactly.
    mine = streaming.get_buildings(helsinki_pbf)
    ref = OSM(helsinki_pbf).get_buildings()
    n_rel = int((ref["osm_type"] == "relation").sum())
    assert n_rel > 0
    assert int((mine["osm_type"] == "relation").sum()) == n_rel
    _assert_matches(
        mine[mine["osm_type"] == "relation"], ref[ref["osm_type"] == "relation"]
    )


def test_streaming_buildings_parallel_matches_single(helsinki_pbf):
    # The multiprocessing path must produce the same result as the in-process path.
    single = streaming.get_buildings(helsinki_pbf, workers=1)
    parallel = streaming.get_buildings(helsinki_pbf, workers=3)
    _assert_matches(parallel, single.sort_values("id").reset_index(drop=True))


def test_auto_workers_decides_on_file_size_not_blob_count(tmp_path):
    # The default worker count is chosen by file size: parallelising only pays off above
    # ~70 MB, and blob count must not force a pool for a small file (sparse files give a
    # size without occupying disk).
    import os

    small = tmp_path / "small.osm.pbf"
    small.touch()
    os.truncate(small, streaming._PARALLEL_MIN_FILE_BYTES - 1)
    assert streaming._auto_workers(str(small), 10_000) == 1

    big = tmp_path / "big.osm.pbf"
    big.touch()
    os.truncate(big, streaming._PARALLEL_MIN_FILE_BYTES + 1)
    cpus = os.cpu_count() or 1
    assert streaming._auto_workers(str(big), 10_000) == cpus
    assert streaming._auto_workers(str(big), 2) == min(cpus, 2)


def test_streaming_buildings_raw_blob_matches(helsinki_pbf, tmp_path):
    # Uncompressed `raw` blobs must decode to the same result as the zlib originals.
    raw_fp = str(tmp_path / "helsinki_raw.osm.pbf")
    _rewrite_pbf_raw(helsinki_pbf, raw_fp)
    zlib_buildings = streaming.get_buildings(helsinki_pbf)
    raw_buildings = streaming.get_buildings(raw_fp)
    _assert_matches(
        raw_buildings, zlib_buildings.sort_values("id").reset_index(drop=True)
    )


def test_streaming_buildings_output_parquet_matches(helsinki_pbf, tmp_path):
    # The streamed GeoParquet must reload equal to the in-memory frame.
    out = str(tmp_path / "buildings.parquet")
    in_memory = streaming.get_buildings(helsinki_pbf)
    returned = streaming.get_buildings(helsinki_pbf, output=out)
    assert returned == out
    reloaded = gpd.read_parquet(out)
    assert isinstance(reloaded, gpd.GeoDataFrame)
    assert reloaded.crs == in_memory.crs
    _assert_matches(reloaded, in_memory.sort_values("id").reset_index(drop=True))


def test_streaming_buildings_output_is_chunked(helsinki_pbf, tmp_path, monkeypatch):
    # A small chunk size must stream multiple row groups (output not materialised at
    # once) while still reloading equal to the in-memory frame.
    import pyarrow.parquet as pq

    monkeypatch.setattr(streaming, "_OUTPUT_CHUNK_SIZE", 50)
    out = str(tmp_path / "buildings_chunked.parquet")
    in_memory = streaming.get_buildings(helsinki_pbf)
    streaming.get_buildings(helsinki_pbf, output=out)
    assert pq.ParquetFile(out).metadata.num_row_groups > 1
    reloaded = gpd.read_parquet(out)
    _assert_matches(reloaded, in_memory.sort_values("id").reset_index(drop=True))


def test_streaming_output_requires_pyarrow(helsinki_pbf, tmp_path, monkeypatch):
    # Without pyarrow, output= must fail fast with an actionable error (and not decode).
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError, match="pyarrow"):
        streaming.get_buildings(helsinki_pbf, output=str(tmp_path / "x.parquet"))
