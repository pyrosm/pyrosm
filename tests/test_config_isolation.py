def test_get_methods_do_not_mutate_shared_tag_config():
    """Regression test: get_* must not mutate the shared Conf default-tag lists.

    The feature methods used to assign ``tags_as_columns = self.conf.tags.<x>``
    (a reference to the shared default list) and then extend it in place --
    directly via ``+= extra_attributes`` and downstream via the ``+=`` in
    ``data_manager``/``networks``. As a result the default columns leaked and
    accumulated across calls within a single process (e.g. ``id``/``nodes``/
    ``timestamp``/``version`` were appended on every call, and any
    ``extra_attributes`` stuck permanently).
    """
    from pyrosm import OSM, get_data
    from pyrosm.config import Conf

    osm = OSM(get_data("test_pbf"))

    building_before = list(Conf.tags.building)
    highway_before = list(Conf.tags.highway)
    natural_before = list(Conf.tags.natural)

    # Plain calls (would leak id/nodes/timestamp/version via the downstream +=)
    osm.get_buildings()
    osm.get_network()
    osm.get_natural()
    # extra_attributes (would leak the custom attribute permanently)
    osm.get_buildings(extra_attributes=["my_extra_attr"])

    assert Conf.tags.building == building_before
    assert Conf.tags.highway == highway_before
    assert Conf.tags.natural == natural_before
    assert "my_extra_attr" not in Conf.tags.building
