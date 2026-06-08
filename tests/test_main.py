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


def test_boundaries(helsinki_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(helsinki_pbf)
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
    bad_zlib.write_bytes(struct.pack("!L", len(header)) + header + b"notzlibdata000000000")
    with pytest.raises(InvalidOSMFileError):
        OSM(bad_zlib)

    # 5) A BlobHeader whose bytes are not valid protobuf (truncated varint).
    malformed = tmp_path / "malformed_header.pbf"
    malformed.write_bytes(struct.pack("!L", 1) + b"\x08")
    with pytest.raises(InvalidOSMFileError):
        OSM(malformed)
