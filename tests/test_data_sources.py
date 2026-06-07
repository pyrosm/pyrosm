"""Offline coverage of the Geofabrik / BBBike data-source accessors (#272).

These walk the static source tables shipped with the package (no network I/O)
and assert every region/city resolves to a real ``*.osm.pbf`` URL, exercising the
per-region ``__getattr__`` / ``__call__`` accessors on every source class.
"""

import pytest
from pyrosm.data import sources, available, search_source

CONTINENTS = [
    "africa",
    "antarctica",
    "asia",
    "australia_oceania",
    "europe",
    "north_america",
    "south_america",
    "central_america",
]


def test_available_top_level_keys():
    assert set(available.keys()) >= {"test_data", "regions", "subregions", "cities"}


def _walk_source(node):
    """Resolve a source node recursively: a leaf dict must carry a ``*.osm.pbf``
    URL; a group object (with ``.available``) is walked member by member. Returns
    the number of leaf records seen. Exercises every ``__getattr__`` / ``__call__``."""
    if isinstance(node, dict):
        assert node["url"].endswith(".osm.pbf")
        return 1
    avail = getattr(node, "available", None)
    if callable(node):  # exercise __call__ where defined
        node()
    count = 0
    members = avail if isinstance(avail, (list, dict)) else []
    for name in members:
        count += _walk_source(getattr(node, name))  # __getattr__
    return count


@pytest.mark.parametrize("continent", CONTINENTS)
def test_continent_region_sources_resolve(continent):
    group = getattr(sources, continent)
    assert isinstance(group.available, list) and len(group.available) > 0
    assert _walk_source(group) > 0


def test_subregion_sources_resolve():
    groups = sources.subregions.available
    assert isinstance(groups, dict) and len(groups) > 0
    assert _walk_source(sources.subregions) > 0


def test_city_sources_resolve():
    cities = sources.cities.available
    assert isinstance(cities, list) and len(cities) > 0
    if callable(sources.cities):
        assert sources.cities() == cities
    for city in cities:
        rec = getattr(sources.cities, city)
        assert rec["url"].endswith(".osm.pbf")


@pytest.mark.parametrize(
    "name,fragment",
    [
        ("helsinki", "Helsinki"),
        ("algeria", "algeria-latest.osm.pbf"),
        ("alsace", "france/alsace-latest.osm.pbf"),
        ("new_york", "us/new-york-latest.osm.pbf"),
    ],
)
def test_search_source_resolves_known_names(name, fragment):
    rec = search_source(name)
    assert isinstance(rec, dict)
    assert rec["url"].endswith(".osm.pbf")
    assert fragment.lower() in rec["url"].lower()


def test_search_source_unknown_name_raises():
    with pytest.raises((ValueError, KeyError)):
        search_source("not_a_real_place_xyz")
