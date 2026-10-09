"""pyrosm's log messages: one line per event, ``"<op> key=value ..."``, written through the
standard ``logging`` module under the ``pyrosm`` logger."""

import contextlib
import functools
import inspect
import logging
import re
import shlex
import threading
import time

logger = logging.getLogger("pyrosm")

# The handler enable_logging() attached last, removed when it is called again.
_handler = None
_handler_lock = threading.Lock()

# C0 and C1 control characters, DEL and the Unicode line and paragraph separators, which
# could move the cursor or forge a line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def _printable(text):
    """``text`` with each control character replaced by a space."""
    return _CONTROL.sub(" ", text)


def enable_logging(level="INFO", to_file=None):
    """Show pyrosm's log messages.

    Attaches one handler to the ``pyrosm`` logger, writing to stderr, or to the file
    ``to_file`` when given, and sets the logger's level: ``"INFO"`` (default) logs the
    configuration of each ``OSM`` object and a line per read with the number of features and the
    seconds it took; ``"DEBUG"`` adds the time of each phase of a read. Calling it again replaces
    the handler it attached before. pyrosm's messages then no longer reach the root logger, so
    they are not shown twice when the application has configured logging too.

    Parameters
    ----------
    level : str | int
        A logging level name, such as ``"INFO"`` or ``"DEBUG"`` (any letter case), or number.

    to_file : str | pathlib.Path, optional
        A file to append the messages to instead of writing them to stderr.

    Returns
    -------
    logging.Handler
        The handler attached.
    """
    global _handler
    if isinstance(level, str) and isinstance(logging.getLevelName(level.upper()), int):
        level = logging.getLevelName(level.upper())
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError(
            "level must be a logging level name or number; got %r." % (level,)
        )
    handler = (
        logging.StreamHandler()
        if to_file is None
        else logging.FileHandler(to_file, encoding="utf-8", errors="backslashreplace")
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    with _handler_lock:
        if _handler is not None:
            logger.removeHandler(_handler)
            _handler.close()
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _handler = handler
    return handler


def _format(value):
    """``value`` as it appears in a message: a float with three decimals, ``bool``, ``None`` and
    integers as they are, anything else with its control characters as spaces (so a message stays
    one line) and quoted for ``shlex.split`` when needed."""
    if isinstance(value, float):
        return "%.3f" % value
    if value is None or isinstance(value, (bool, int)):
        return str(value)
    return shlex.quote(_printable(str(value)))


def log_event(log, level, op, **fields):
    """Log ``"<op> key=value ..."`` from ``fields`` on ``log`` at ``level``, formatting nothing
    when ``log`` is not enabled for it."""
    if log.isEnabledFor(level):
        log.log(
            level,
            " ".join([op] + ["%s=%s" % (k, _format(v)) for k, v in fields.items()]),
        )


@contextlib.contextmanager
def timed(log, op, level=logging.DEBUG, **fields):
    """Time the block and log ``"<op> key=value ... seconds=<elapsed>"`` when it completes.
    The block gets ``fields`` and can add to them; nothing is logged when it raises, and nothing
    is timed when ``log`` is not enabled for ``level``."""
    if not log.isEnabledFor(level):
        yield fields
        return
    start = time.perf_counter()
    yield fields
    fields["seconds"] = time.perf_counter() - start
    log_event(log, level, op, **fields)


def _sizes(result):
    """The number of rows of a read's result: ``features`` for a frame (0 for ``None``), or
    ``node_rows`` and ``edge_rows`` for a ``(nodes, edges)`` pair."""
    if isinstance(result, tuple) and len(result) == 2:
        nodes, edges = result
        return {
            "node_rows": 0 if nodes is None else len(nodes),
            "edge_rows": 0 if edges is None else len(edges),
        }
    return {"features": 0 if result is None else len(result)}


def logged(*keys, level=logging.INFO, count=True):
    """Decorate a function so that each call is logged as ``"<name> <keys> <sizes> seconds=..."``
    on its module's logger: ``keys`` are argument names whose values are logged (defaults
    included), and with ``count`` the size of the result (see :func:`_sizes`)."""

    def decorate(func):
        log = logging.getLogger(func.__module__)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not log.isEnabledFor(level):
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            fields = {key: bound.arguments[key] for key in keys}
            with timed(log, func.__name__, level, **fields) as fields:
                result = func(*args, **kwargs)
                if count:
                    fields.update(_sizes(result))
            return result

        return wrapper

    return decorate
