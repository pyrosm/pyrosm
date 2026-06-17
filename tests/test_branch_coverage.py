"""Branch coverage for feature modules and the graph-export prep (#272).

Uses the real bundled PBF data and derives malformed inputs from it (dropping a
column / using a no-data bbox) to exercise the validation / no-data / warning
branches that the happy-path tests don't reach.
"""

import pytest
from geopandas import GeoDataFrame
from pyrosm import OSM, get_data


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


@pytest.fixture
def helsinki_region_pbf():
    return get_data("helsinki_region_pbf")


# --- natural.py ---------------------------------------------------------------


def test_get_natural_custom_filter_without_natural_key(helsinki_pbf):
    """A custom_filter lacking the 'natural' key gets natural=[True] injected, so
    the result still contains the full set of natural features (natural.py)."""
    osm = OSM(helsinki_pbf)
    default = osm.get_natural()
    via_filter = osm.get_natural(custom_filter={"wetland": ["bog", "marsh"]})
    assert isinstance(via_filter, GeoDataFrame)
    assert "natural" in via_filter.columns
    # The injected natural=[True] dominates, so the filtered call returns the
    # same set of natural features as the default call.
    assert len(via_filter) == len(default)
    assert len(via_filter) > 100
    assert "coastline" in set(via_filter["natural"].dropna())


# --- boundary.py --------------------------------------------------------------


def test_get_boundaries_returns_none_for_empty_area(helsinki_pbf):
    """A bbox with no boundaries warns and returns None (boundary.py)."""
    osm = OSM(helsinki_pbf, bounding_box=[24.940, 60.173, 24.942, 60.175])
    with pytest.warns(UserWarning, match="Could not find any boundaries"):
        gdf = osm.get_boundaries()
    assert gdf is None


def test_get_boundaries_filter_by_name(helsinki_region_pbf):
    """Filtering boundaries by name returns only the matching boundary
    (boundary.py name-filter branch). Helsinki has named neighbourhood
    boundaries (e.g. 'Kallio'). The region extract is used because helsinki_pbf's
    boundaries are all incomplete and dropped (#154)."""
    osm = OSM(helsinki_region_pbf)
    all_boundaries = osm.get_boundaries()
    assert "Kallio" in set(all_boundaries["name"].dropna())

    only_kallio = osm.get_boundaries(name="Kallio")
    assert isinstance(only_kallio, GeoDataFrame)
    assert len(only_kallio) >= 1
    assert set(only_kallio["name"]) == {"Kallio"}
    assert len(only_kallio) < len(all_boundaries)


# --- graphs.py (get_directed_edges prep) --------------------------------------


def test_to_graph_requires_edge_endpoint_columns(test_pbf):
    """Edges missing the 'u'/'v' columns raise a clear error (graphs.py)."""
    from pyrosm.graphs import to_igraph

    pytest.importorskip("igraph")
    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    with pytest.raises(ValueError, match="does not exist in edges"):
        to_igraph(nodes, edges.drop(columns=["u"]), retain_all=True)


def test_to_graph_requires_node_id_column(test_pbf):
    """Nodes missing the 'id' column raise a clear error (graphs.py)."""
    from pyrosm.graphs import to_igraph

    pytest.importorskip("igraph")
    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    with pytest.raises(ValueError, match="does not exist in nodes"):
        to_igraph(nodes.drop(columns=["id"]), edges, retain_all=True)


def test_to_graph_warns_when_direction_column_missing(test_pbf):
    """Edges without the oneway/direction column warn and are treated as
    bidirectional (graphs.py)."""
    from pyrosm.graphs import to_igraph

    pytest.importorskip("igraph")
    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    if "oneway" in edges.columns:
        edges = edges.drop(columns=["oneway"])
    with pytest.warns(UserWarning, match="oneway"):
        g = to_igraph(nodes, edges, retain_all=True)
    assert g is not None


def test_to_graph_retain_all_keeps_all_components(test_pbf):
    """retain_all=True skips the connected-component pruning (graphs.py)."""
    from pyrosm.graphs import to_igraph

    pytest.importorskip("igraph")
    osm = OSM(test_pbf)
    nodes, edges = osm.get_network(nodes=True)
    g_all = to_igraph(nodes, edges, retain_all=True)
    g_conn = to_igraph(nodes, edges, retain_all=False)
    assert g_all.vcount() >= g_conn.vcount()
