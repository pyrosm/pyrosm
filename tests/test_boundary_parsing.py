import pytest
from pyrosm import get_data


@pytest.fixture
def helsinki_pbf():
    pbf_path = get_data("helsinki_pbf")
    return pbf_path


@pytest.fixture
def helsinki_region_pbf():
    pbf_path = get_data("helsinki_region_pbf")
    return pbf_path


REQUIRED_COLUMNS = [
    "name",
    "admin_level",
    "boundary",
    "id",
    "timestamp",
    "version",
    "changeset",
    "geometry",
    "tags",
    "osm_type",
]


def test_reading_boundaries_with_defaults(helsinki_region_pbf):
    # The small helsinki_pbf extract has only incomplete boundaries (all dropped,
    # see test_regressions::test_incomplete_boundaries_dropped_not_force_closed),
    # so the feature tests use the region extract, which contains complete ones.
    from pyrosm import OSM

    osm = OSM(helsinki_region_pbf)
    gdf = osm.get_boundaries()

    assert len(gdf) == 247
    for col in REQUIRED_COLUMNS:
        assert col in gdf.columns

    # Complete boundaries assemble into valid polygons; results are relations
    # (with a few standalone boundary ways).
    assert "relation" in set(gdf.osm_type.unique())
    assert (gdf.geometry.geom_type == "Polygon").any()
    assert gdf.geometry.is_valid.all()


def test_reading_boundaries_with_name_search(helsinki_region_pbf):
    from pyrosm import OSM

    osm = OSM(helsinki_region_pbf)

    # Full name -> a single boundary polygon.
    gdf = osm.get_boundaries(name="Punavuori")
    assert len(gdf) == 1
    for col in REQUIRED_COLUMNS:
        assert col in gdf.columns
    assert gdf.geometry.geom_type.iloc[0] == "Polygon"

    # Partial name -> one or more, every returned name contains the substring.
    gdf = osm.get_boundaries(name="saari")
    assert len(gdf) >= 1
    assert gdf["name"].str.contains("saari").all()


def test_passing_incorrect_parameters(helsinki_pbf):
    from pyrosm import OSM

    osm = OSM(helsinki_pbf)
    try:
        osm.get_boundaries(boundary_type="incorrect")
    except ValueError as e:
        if "should be one of the following" in str(e):
            pass
    except Exception as e:
        raise e

    try:
        osm.get_boundaries(name=1)
    except ValueError as e:
        if "should be text" in str(e):
            pass
    except Exception as e:
        raise e


def test_adding_extra_attribute(helsinki_region_pbf):
    from pyrosm import OSM
    from geopandas import GeoDataFrame

    osm = OSM(filepath=helsinki_region_pbf)
    gdf = osm.get_boundaries()
    extra_col = "wikidata"
    extra = osm.get_boundaries(extra_attributes=[extra_col])

    # The extra should have one additional column compared to the original one
    assert extra.shape[1] == gdf.shape[1] + 1
    # Should have same number of rows
    assert extra.shape[0] == gdf.shape[0]
    assert extra_col in extra.columns
    assert len(extra[extra_col].dropna().unique()) > 0
    assert isinstance(gdf, GeoDataFrame)


def test_reading_all_boundaries(helsinki_region_pbf):
    from pyrosm import OSM

    osm = OSM(helsinki_region_pbf)
    gdf = osm.get_boundaries(boundary_type="all")

    assert len(gdf) == 699
    for col in REQUIRED_COLUMNS:
        assert col in gdf.columns

    # Test filtering different types of boundaries
    value_counts = gdf["boundary"].value_counts()

    for boundary_type, cnt in value_counts.items():
        # Some incorrect boundary types exists in the data
        if boundary_type in ["lot 1", "imagery", "historic"]:
            continue

        gdf = osm.get_boundaries(boundary_type=boundary_type)
        assert len(gdf) >= cnt, f"Got incorrect number of rows with {boundary_type}"


def test_relation_members_in_tags(helsinki_region_pbf):
    """#216 — a relation's members are exposed under the 'members' key of the
    JSON 'tags' column (not a separate full-length column)."""
    import json
    from pyrosm import OSM

    osm = OSM(helsinki_region_pbf)
    gdf = osm.get_boundaries()
    rel = gdf[gdf.osm_type == "relation"].iloc[0]

    # Folded into the tags JSON, not a standalone column.
    assert "members" not in gdf.columns

    tags = json.loads(rel["tags"])
    assert "members" in tags
    members = tags["members"]
    assert isinstance(members, list) and len(members) > 0
    for member in members:
        assert set(member) == {"member_id", "member_type", "member_role"}
        assert isinstance(member["member_id"], int)
        assert member["member_type"] in ("node", "way", "relation")
        assert isinstance(member["member_role"], str)
