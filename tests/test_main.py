import pytest
from pyrosm import get_data


@pytest.fixture
def test_pbf():
    pbf_path = get_data("test_pbf")
    return pbf_path


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def helsinki_region_pbf():
    pbf_path = get_data("helsinki_region_pbf")
    return pbf_path


def test_lazy_public_api():
    import pyrosm

    # The public names are lazily resolved via module __getattr__.
    assert pyrosm.OSM.__name__ == "OSM"
    assert callable(pyrosm.get_data)
    assert callable(pyrosm.get_path)
    # Unknown attributes still raise AttributeError.
    with pytest.raises(AttributeError):
        pyrosm.does_not_exist


@pytest.mark.parametrize("run", ["read", "crop", "merge"])
def test_default_workers_warns_once_that_it_changes_in_1_0(
    run, helsinki_pbf, tmp_path, monkeypatch
):
    """A read, crop or merge of a file large enough for ``workers="auto"`` to run in
    parallel that leaves ``workers`` unset warns once per process, at the caller's line,
    that pyrosm 1.0 runs it in parallel by default; one that passes ``workers``, or reads a
    smaller file, does not warn. A read served from the result cache warns too."""
    import warnings

    import pyrosm.engine.pool as pool
    import pyrosm.utils
    from pyrosm import OSM, merge_pbf
    from pyrosm.pbf_export import crop_pbf

    out, box = str(tmp_path / "out.osm.pbf"), [24.9424, 60.1701, 24.9461, 60.1731]
    runs = {
        "read": lambda **kw: OSM(helsinki_pbf, **kw).get_pois(),
        "crop": lambda **kw: crop_pbf(helsinki_pbf, out, box, **kw),
        "merge": lambda **kw: merge_pbf([helsinki_pbf], out, **kw),
    }
    line = runs[run].__code__.co_firstlineno
    pyrosm.utils._future_warnings_shown.discard(pyrosm.utils._DEFAULT_WORKERS_WARNING)

    def future_warnings(**options):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runs[run](**options)
        return [w for w in caught if issubclass(w.category, FutureWarning)]

    assert future_warnings() == []  # the sample is too small for "auto" to go parallel
    monkeypatch.setattr(pool, "_PARALLEL_MIN_FILE_BYTES", 1)
    for options, expected in [({"workers": 1}, 0), ({}, 1), ({}, 0)]:
        future = future_warnings(**options)
        assert len(future) == expected
        for w in future:
            assert "In pyrosm 1.0" in str(w.message)
            assert (w.filename, w.lineno) == (__file__, line)


def test_default_workers_warning_as_an_error_raises_at_every_default_call(
    helsinki_pbf, monkeypatch
):
    """With the warning turned into an error, each call that uses the default for a file
    large enough for ``workers="auto"`` to run in parallel raises."""
    import warnings

    import pyrosm.engine.pool as pool
    import pyrosm.utils

    monkeypatch.setattr(pool, "_PARALLEL_MIN_FILE_BYTES", 1)
    pyrosm.utils._future_warnings_shown.discard(pyrosm.utils._DEFAULT_WORKERS_WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for _ in range(2):
            with pytest.raises(FutureWarning, match="In pyrosm 1.0"):
                pyrosm.utils.default_workers(None, helsinki_pbf)
        assert pyrosm.utils.default_workers(2, helsinki_pbf) == 2
        # a file that cannot be read counts as no data, so the default does not warn
        assert pyrosm.utils.default_workers(None, "missing.osm.pbf") == 1


def test_processes_backend_warns_once_that_it_is_deprecated(monkeypatch):
    """Selecting the processes backend warns once per process that it is deprecated,
    pointing at the caller; the threads backend does not warn."""
    import warnings

    import pyrosm.utils
    from pyrosm.engine.pool import _PROCESSES_BACKEND_WARNING, _decode_backend

    pyrosm.utils._future_warnings_shown.discard(_PROCESSES_BACKEND_WARNING)
    for backend, expected in [("threads", 0), ("processes", 1), ("processes", 0)]:
        monkeypatch.setenv("PYROSM_DECODE_BACKEND", backend)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _decode_backend() == backend
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert [str(w.message) for w in future] == [_PROCESSES_BACKEND_WARNING] * expected
        assert all(w.filename == __file__ for w in future)


def test_network(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_network()
    assert isinstance(gdf, GeoDataFrame)


def test_buildings(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_buildings()
    assert isinstance(gdf, GeoDataFrame)


def test_landuse(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_landuse()
    assert isinstance(gdf, GeoDataFrame)


def test_pois(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_pois()
    assert isinstance(gdf, GeoDataFrame)


def test_natural(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_natural()
    assert isinstance(gdf, GeoDataFrame)


def test_custom(test_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(test_pbf)
    gdf = osm.get_data_by_custom_criteria({"highway": ["secondary"]})
    assert isinstance(gdf, GeoDataFrame)


def test_boundaries(helsinki_region_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    # helsinki_pbf has only incomplete boundaries (all dropped, #154); the region
    # extract contains complete ones.
    osm = OSM(helsinki_region_pbf)
    gdf = osm.get_boundaries()
    assert isinstance(gdf, GeoDataFrame)


def test_passing_pathlib_path(test_pbf):
    from pathlib import Path
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(Path(test_pbf))
    assert isinstance(osm.filepath, str)
    gdf = osm.get_network()
    assert isinstance(gdf, GeoDataFrame)


def test_passing_incorrect_filepath():
    from pyrosm import OSM

    try:
        OSM(11)
    except ValueError:
        pass
    except Exception as e:
        raise e


def test_passing_wrong_file_format():
    from pyrosm import OSM

    try:
        OSM("test.osm")
    except ValueError:
        pass
    except Exception as e:
        raise e


def test_invalid_osm_pbf_raises_meaningful_error(tmp_path):
    """A '.pbf' that is not a valid OSM PBF should raise InvalidOSMFileError."""
    import struct
    from pyrosm import OSM
    from pyrosm.exceptions import InvalidOSMFileError
    from pyrosm.proto.fileformat_pb2 import BlobHeader

    # 1) Too short to contain a header.
    short = tmp_path / "short.pbf"
    short.write_bytes(b"\x00\x00")
    with pytest.raises(InvalidOSMFileError):
        OSM(short)

    # 2) Declared BlobHeader size beyond the 64 KiB maximum.
    oversized = tmp_path / "oversized.pbf"
    oversized.write_bytes(struct.pack("!L", 5_000_000) + b"garbage" * 100)
    with pytest.raises(InvalidOSMFileError):
        OSM(oversized)

    # 3) A parseable BlobHeader whose first block is not an 'OSMHeader'
    #    (e.g. a non-OSM PBF such as a vector-tile PBF).
    header = BlobHeader(type="NotOSMHeader", datasize=5).SerializeToString()
    wrong_type = tmp_path / "wrong_type.pbf"
    wrong_type.write_bytes(struct.pack("!L", len(header)) + header + b"\x00" * 5)
    with pytest.raises(InvalidOSMFileError):
        OSM(wrong_type)

    # 4) Correct 'OSMHeader' type but a payload that is not zlib-compressed
    #    (reproduces the cryptic "Error -5 while decompressing data" from #156).
    header = BlobHeader(type="OSMHeader", datasize=20).SerializeToString()
    bad_zlib = tmp_path / "bad_zlib.pbf"
    bad_zlib.write_bytes(
        struct.pack("!L", len(header)) + header + b"notzlibdata000000000"
    )
    with pytest.raises(InvalidOSMFileError):
        OSM(bad_zlib)

    # 5) A BlobHeader whose bytes are not valid protobuf (truncated varint).
    malformed = tmp_path / "malformed_header.pbf"
    malformed.write_bytes(struct.pack("!L", 1) + b"\x08")
    with pytest.raises(InvalidOSMFileError):
        OSM(malformed)

    # 6) A valid header requiring a feature pyrosm does not support: the message
    #    names the file.
    import zlib
    from pyrosm.exceptions import PBFNotImplemented
    from pyrosm.proto.fileformat_pb2 import Blob
    from pyrosm.proto.osmformat_pb2 import HeaderBlock

    header_block = HeaderBlock(required_features=["OsmSchema-V0.6", "LocationsOnWays"])
    data = header_block.SerializeToString()
    blob = Blob(raw_size=len(data), zlib_data=zlib.compress(data)).SerializeToString()
    header = BlobHeader(type="OSMHeader", datasize=len(blob)).SerializeToString()
    unsupported = tmp_path / "locations_on_ways.pbf"
    unsupported.write_bytes(struct.pack("!L", len(header)) + header + blob)
    with pytest.raises(PBFNotImplemented, match="'LocationsOnWays'") as err:
        OSM(unsupported)
    assert str(unsupported) in str(err.value)
