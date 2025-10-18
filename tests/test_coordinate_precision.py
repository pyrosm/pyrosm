from pathlib import Path

import pytest

pytest.importorskip("shapely", reason="shapely required by pyrosm import.")
pytest.importorskip("pyrosm_proto", reason="pyrosm_proto extension must be built.")
pyproj = pytest.importorskip("pyproj", reason="pyproj required to validate distance.")
Geod = pyproj.Geod

from pyrosm import OSM


def test_dense_node_coordinates_match_osm_api():
    """Ensure dense nodes keep full double precision when decoded."""
    pbf_path = Path(__file__).parent / "data" / "pyrosm_test.pbf"
    osm = OSM(str(pbf_path))

    gdf = osm.get_data_by_custom_criteria(
        {"power": ["tower"]}, filter_type="keep", keep_nodes=True
    )

    row = gdf.loc[gdf["id"] == 623850466].iloc[0]
    lon_py, lat_py = row.geometry.coords[0]

    expected_lat, expected_lon = 26.0914866, -80.4410082
    assert abs(lat_py - expected_lat) < 1e-9
    assert abs(lon_py - expected_lon) < 1e-9

    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(expected_lon, expected_lat, lon_py, lat_py)
    assert dist_m < 1e-4
