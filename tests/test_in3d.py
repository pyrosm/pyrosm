import importlib.machinery
import sys
import types

import pytest

from pyrosm import OSM, get_data


def test_in3d_gives_the_views_of_pyrosm3d(monkeypatch):
    """``OSM.in3d`` is pyrosm3d's ``In3D`` for the OSM object; without pyrosm3d it
    raises an ImportError with the command that installs it."""
    osm = OSM(get_data("test_pbf"))

    class In3D:
        def __init__(self, osm):
            self.osm = osm

    pyrosm3d = types.ModuleType("pyrosm3d")
    pyrosm3d.__spec__ = importlib.machinery.ModuleSpec("pyrosm3d", None)
    pyrosm3d.In3D = In3D
    monkeypatch.setitem(sys.modules, "pyrosm3d", pyrosm3d)
    views = osm.in3d
    assert isinstance(views, In3D) and views.osm is osm

    monkeypatch.setitem(sys.modules, "pyrosm3d", None)
    with pytest.raises(ImportError, match=r"pip install \"pyrosm\[3d\]\""):
        osm.in3d
