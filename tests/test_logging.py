import logging
import shlex
from pathlib import Path

import pytest

from pyrosm import OSM, enable_logging, get_data
from pyrosm import _log


@pytest.fixture
def pyrosm_logger():
    """The ``pyrosm`` logger, with its handlers, level and propagation restored afterwards."""
    logger = logging.getLogger("pyrosm")
    state = (list(logger.handlers), logger.level, logger.propagate, _log._handler)
    yield logger
    for handler in logger.handlers:
        if handler not in state[0]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(state[1])
    logger.propagate = state[2]
    _log._handler = state[3]


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
        "path=a b/it's.pbf",
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
    osm = OSM(fp, progress=False)
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
    assert find("to_graph graph_type=networkx network_type=None simplify=False ")
    assert find("OSM file=test.osm.pbf engine=out_of_core workers=1")
    # One decode for the buildings and one for the empty layer; the cached reads decode nothing.
    assert len(find("index file=test.osm.pbf blobs=3 ")) == 2
    assert len(find("decode workers=1 blobs=3 pool=False ")) == 2
    assert len(find("collect workers=1 ")) == 2
    cache_events = [shlex.split(m)[1:] for m in find("cache ")]
    assert [status for status, _ in cache_events] == [
        "status=miss",
        "status=hit",
        "status=miss",
        "status=empty",
    ]
    assert cache_events[3][1].endswith(".empty")
