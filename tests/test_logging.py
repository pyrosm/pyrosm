import logging
import shlex
from pathlib import Path

import pytest

from pyrosm import OSM, enable_logging, get_data
from pyrosm import _log
from pyrosm.utils import progress


@pytest.fixture
def pyrosm_logger():
    """The ``pyrosm`` logger, with its handlers, level and propagation, and the progress setting
    of enable_logging restored afterwards."""
    logger = logging.getLogger("pyrosm")
    state = (list(logger.handlers), logger.level, logger.propagate, _log._handler)
    setting = progress._setting
    yield logger
    for handler in logger.handlers:
        if handler not in state[0]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(state[1])
    logger.propagate = state[2]
    _log._handler = state[3]
    progress._setting = setting


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_enable_logging(pyrosm_logger, tmp_path, monkeypatch):
    """Without enable_logging a pyrosm message reaches the root logger's handler once; with it,
    the message goes to pyrosm's one handler (stderr or a file) and not to the root logger,
    however often it is called. The package's NullHandler stays; other levels are refused."""
    root = _Records()
    monkeypatch.setattr(logging.getLogger(), "handlers", [root])
    pyrosm_logger.setLevel(logging.INFO)
    pyrosm_logger.info("before")
    assert root.messages == ["before"]

    enable_logging("debug")
    enable_logging(to_file=tmp_path / "pyrosm.log")
    outputs = [
        h for h in pyrosm_logger.handlers if not isinstance(h, logging.NullHandler)
    ]
    assert len(outputs) == 1 and isinstance(outputs[0], logging.FileHandler)
    assert any(isinstance(h, logging.NullHandler) for h in pyrosm_logger.handlers)
    logging.getLogger("pyrosm.engine").info("after Hämeenlinna")
    outputs[0].flush()
    lines = (tmp_path / "pyrosm.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].endswith("pyrosm.engine INFO after Hämeenlinna")
    assert root.messages == ["before"]
    for level in ("LOUD", True, None):
        with pytest.raises(ValueError, match="level must be"):
            enable_logging(level)


def test_enable_logging_sets_progress(pyrosm_logger, tmp_path, monkeypatch):
    """enable_logging(progress=...) sets the progress of every operation not given its own,
    also of an OSM object made before the call: reads with both engines and a crop report to
    the function given; an operation's own progress overrides it; None is refused."""
    from pyrosm.engine import cache

    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path)
    fp = get_data("test_pbf")
    made_before = OSM(fp, engine="in_memory")
    calls = []
    enable_logging(progress=lambda done, total: calls.append((done, total)))
    box = [26.94, 60.52, 26.96, 60.53]
    for run in (
        made_before.get_buildings,
        OSM(fp, engine="out_of_core", workers=1).get_buildings,
        lambda: OSM(fp, bounding_box=box).to_pbf(tmp_path / "crop.osm.pbf"),
    ):
        calls.clear()
        run()
        assert calls and calls[-1][0] == calls[-1][1]
    calls.clear()
    OSM(fp, engine="in_memory", progress=False).get_buildings()
    assert calls == []
    with pytest.raises(ValueError, match="progress must be True, False or a callable"):
        enable_logging(progress=None)


def test_download_progress_covers_the_crop(pyrosm_logger, tmp_path, monkeypatch):
    """The progress given to get_data_by_bbox (whose crop get_data_by_geocoding shares) and to
    get_data_by_area reports their crop too, in place of the setting of enable_logging."""
    from shapely.geometry import box
    import pyrosm.data.extract_index as ei
    from pyrosm import get_data_by_bbox

    fp = get_data("test_pbf")
    monkeypatch.setattr("pyrosm.utils.download.download", lambda *a, **k: fp)
    bounds = [26.94, 60.52, 26.96, 60.53]
    setting, own = [], []
    enable_logging(progress=lambda done, total: setting.append(done))

    def report(done, total):
        own.append(done)

    source = ei.ExtractSource("geofabrik", "test", "url", 0, fp)
    for crop in (
        lambda: get_data_by_bbox(bounds, directory=tmp_path, progress=report),
        lambda: ei._write_with_provenance(
            box(*bounds), [source], True, None, tmp_path, 1, report
        ),
    ):
        own.clear()
        crop()
        assert own and setting == []


def test_graph_export_reports_its_steps(pyrosm_logger):
    """A graph export reports its steps to the function set with enable_logging, done from 0
    to the number of steps, which depends on simplify and retain_all; with progress off it
    reports nothing and still builds the graph."""
    from pyrosm.graphs import graph_tables

    osm = OSM(get_data("test_pbf"), progress=False)
    nodes, edges = osm.get_network(nodes=True, network_type="driving")
    calls = []
    enable_logging(progress=lambda done, total: calls.append((done, total)))
    for export, steps in (
        (lambda: OSM.to_graph(nodes, edges, graph_type="networkx"), 4),
        (
            lambda: OSM.to_graph(
                nodes, edges, graph_type="networkx", simplify=False, retain_all=True
            ),
            2,
        ),
        (lambda: graph_tables(nodes, edges), 3),
    ):
        calls.clear()
        export()
        assert calls == [(done, steps) for done in range(steps + 1)]
    calls.clear()
    enable_logging(progress=False)
    assert OSM.to_graph(nodes, edges, graph_type="networkx").number_of_edges()
    assert calls == []


def test_log_event_and_timed(pyrosm_logger, caplog):
    """Messages are ``op key=value ...`` lines that shlex.split reads back, with control
    characters as spaces; nothing is formatted or timed when the level is off, and a block that
    raises logs nothing."""

    class Unprintable:
        def __str__(self):
            raise AssertionError("formatted while logging is off")

    caplog.set_level(logging.INFO, logger="pyrosm")
    _log.log_event(pyrosm_logger, logging.DEBUG, "off", value=Unprintable())
    with _log.timed(pyrosm_logger, "off", value=Unprintable()) as fields:
        pass
    assert "seconds" not in fields and caplog.records == []

    _log.log_event(
        pyrosm_logger,
        logging.INFO,
        "read",
        count=3,
        share=0.12345,
        flag=True,
        none=None,
        path=Path("a b/it's.pbf"),
        query="x\r\ny\x00\x1b[2J\u2028z",
    )
    message = caplog.records[0].getMessage()
    assert len(message.splitlines()) == 1 and "\x1b" not in message
    assert shlex.split(message) == [
        "read",
        "count=3",
        "share=0.123",
        "flag=True",
        "none=None",
        "path=%s" % Path("a b/it's.pbf"),  # the native separator, \ on Windows
        "query=x  y  [2J z",
    ]
    with _log.timed(pyrosm_logger, "phase", logging.INFO, blobs=3) as fields:
        fields["pool"] = False
    assert caplog.records[1].getMessage().startswith("phase blobs=3 pool=False seconds=")
    with pytest.raises(ValueError):
        with _log.timed(pyrosm_logger, "failed", logging.INFO):
            raise ValueError("boom")
    assert len(caplog.records) == 2


def test_reads_are_logged(caplog, tmp_path, monkeypatch):
    """A read logs the reader's configuration, the in-memory read or the engine's phases and
    cache, and one line per feature read with its size; to_graph logs its options."""
    from pyrosm.engine import cache

    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path)
    caplog.set_level(logging.DEBUG, logger="pyrosm")
    fp = get_data("test_pbf")
    osm = OSM(fp, engine="in_memory", progress=False)
    buildings = osm.get_buildings()
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    OSM.to_graph(nodes, edges, graph_type="networkx")
    engine = OSM(fp, engine="out_of_core", workers=1, progress=False)
    engine.get_buildings()
    engine.get_buildings()
    for _ in range(2):
        engine.get_data_by_custom_criteria(custom_filter={"no_such_key": True})
    messages = [r.getMessage() for r in caplog.records]

    def find(prefix):
        return [m for m in messages if m.startswith(prefix)]

    assert find("OSM file=test.osm.pbf engine=in_memory workers=None bounding_box=None")
    assert find("read_pbf file=test.osm.pbf bytes=%d " % Path(fp).stat().st_size)
    assert find("get_buildings features=%d seconds=" % len(buildings))
    assert find(
        "get_network network_type=driving nodes=True node_rows=%d edge_rows=%d "
        % (len(nodes), len(edges))
    )
    assert find("to_graph graph_type=networkx network_type=None simplify=True ")
    assert find("OSM file=test.osm.pbf engine=out_of_core workers=1")
    # One decode for the buildings and one for the empty layer; the cached reads decode nothing.
    assert len(find("index file=test.osm.pbf blobs=3 ")) == 2
    assert len(find("decode workers=1 blobs=3 pool=False ")) == 2
    assert len(find("collect seconds=")) == 2
    cache_events = [shlex.split(m)[1:] for m in find("cache ")]
    assert [status for status, _ in cache_events] == [
        "status=miss",
        "status=hit",
        "status=miss",
        "status=empty",
    ]
    assert cache_events[3][1].endswith(".empty")
