"""Progress of downloads and reads on stderr: a tqdm bar, or a widget in Jupyter when
ipywidgets is installed."""

import contextlib
import importlib.util
import sys
import threading
import time

from pyrosm._log import _printable

_clock = time.monotonic
# How often an open bar is redrawn, so that its time moves on during a long step.
_REDRAW_SECONDS = 1.0

# The progress of every operation whose own ``progress`` is None, set by
# pyrosm.enable_logging(progress=...).
_setting = True


def _bar_class():
    """``(tqdm class, disable)`` for a bar on stderr.

    In a Jupyter kernel it is the widget bar when ipywidgets is installed, else the text bar,
    which the notebook redraws in place; both are shown although the kernel's stderr is no
    terminal. Elsewhere it is the text bar, hidden when stderr is no terminal.
    """
    from tqdm import std

    # A kernel has imported IPython already, so it is looked up, not imported.
    ipython = sys.modules.get("IPython")
    shell = getattr(ipython, "get_ipython", lambda: None)()
    if type(shell).__name__ != "ZMQInteractiveShell":
        return std.tqdm, None
    if importlib.util.find_spec("ipywidgets") is None:
        return std.tqdm, False
    from tqdm import notebook

    return notebook.tqdm, False


def validate_progress(progress, allow_none=True):
    """Raise ``ValueError`` unless ``progress`` is ``True``, ``False``, a callable, or
    ``None`` with ``allow_none``."""
    if not (
        (allow_none and progress is None)
        or isinstance(progress, bool)
        or callable(progress)
    ):
        raise ValueError(
            "progress must be %sTrue, False or a callable; got %r."
            % ("None, " if allow_none else "", progress)
        )


@contextlib.contextmanager
def reporting(progress, desc, **options):
    """The progress callback for a validated ``progress``, which is ``None`` to follow the
    setting of :func:`pyrosm.enable_logging` (``True`` unless set): no callback for ``False``,
    the callable itself, or for ``True`` a :class:`Bar` described ``desc`` with ``options``,
    closed on exit."""
    if progress is None:
        progress = _setting
    if progress is True:
        bar = Bar(desc, **options)
        try:
            yield bar
        finally:
            bar.close()
    else:
        yield progress if callable(progress) else None


class Bar:
    """A progress callback, called as ``bar(done, total)``, that draws a bar on stderr
    described ``desc`` (see :func:`_bar_class`), of bytes, or of another ``unit`` counted
    as given.

    The bar opens at the first call and starts over when ``done`` goes back or ``total``
    changes, keeping its start time. It shows only once it has run ``delay`` seconds, is
    redrawn every second while open, and ``leave=False`` clears it when it closes. A bar
    of the ``unit`` ``"%"`` shows the percentage and the times. :meth:`describe`
    changes the label it shows. When the bar is hidden, ``desc`` is printed once as a
    line when it opens, or with ``timed`` a line ``"<desc> took <time>"`` when it
    closes, if it ran at least ``delay`` seconds.
    """

    def __init__(self, desc, delay=0.0, leave=True, timed=False, unit="B"):
        self.desc = _printable(desc)
        self.label = self.desc
        self.delay = delay
        self.leave = leave
        self.timed = timed
        self.unit = unit
        self.bar = None
        self.hidden = False
        self.done = 0
        self.start = _clock()
        self._closed = threading.Event()
        self._redraw = None

    def describe(self, label):
        """Show ``label`` in place of the description from the next redraw on."""
        self.label = _printable(label)
        if self.bar is not None:
            self.bar.set_description(self.label, refresh=False)

    def __call__(self, done, total):
        if self.bar is None:
            tqdm, disable = _bar_class()
            percent = {}
            if self.unit == "%":
                percent["bar_format"] = "{l_bar}{bar}| {elapsed}<{remaining}"
            self.bar = tqdm(
                total=total,
                file=sys.stderr,
                unit=self.unit,
                unit_scale=self.unit == "B",
                unit_divisor=1000,
                mininterval=0.2,
                dynamic_ncols=True,
                delay=self.delay,
                leave=self.leave,
                disable=disable,
                **percent,
            )
            # Set after: the widget shows a description given to it as HTML until it redraws.
            # A delayed bar is not redrawn here, which would show it before its delay.
            self.bar.set_description(self.label, refresh=self.delay <= 0)
            # Kept, as tqdm marks every bar disabled once it closes.
            self.hidden = self.bar.disable
            if self.hidden and not self.timed:
                self.bar.write(self.desc, file=sys.stderr)
            if not self.hidden:
                self._redraw = threading.Thread(target=self._redraw_while_open)
                self._redraw.daemon = True
                self._redraw.start()
        elif done < self.done or total != self.bar.total:
            if _clock() - self.start < self.delay:
                # Not drawn yet: reset() would draw it before its delay.
                self.bar.n = self.bar.last_print_n = 0
            else:
                # reset() restarts the bar's clock, and with it the delay.
                started = self.bar.start_t
                self.bar.reset(total)
                self.bar.start_t = started
            # reset() keeps the old total when the new one is unknown.
            self.bar.total = total
        self.bar.update(done - self.bar.n)
        self.done = done

    def _redraw_while_open(self):
        """Redraw the bar every :data:`_REDRAW_SECONDS` once its delay has passed, until
        it closes, so its time moves on while no report comes."""
        opened = time.monotonic()
        while not self._closed.wait(_REDRAW_SECONDS):
            if time.monotonic() - opened >= self.delay:
                self.bar.refresh()

    def close(self):
        """Stop redrawing and close the bar, printing its time line when it is hidden
        and ``timed``."""
        if self.bar is None:
            return
        self._closed.set()
        if self._redraw is not None:
            self._redraw.join()
        self.bar.close()
        elapsed = _clock() - self.start
        if self.timed and self.hidden and elapsed >= self.delay:
            took = self.bar.format_interval(elapsed)
            self.bar.write("%s took %s" % (self.desc, took), file=sys.stderr)


@contextlib.contextmanager
def stepping(progress, desc, shares=None, **options):
    """The :class:`Steps` of the work described ``desc``, reported as ``progress`` says
    (see :func:`reporting`, whose ``options`` these are), at 100 % when the work ends.
    """
    with reporting(progress, desc, unit="%", **options) as report:
        steps = Steps(report, desc, shares)
        yield steps
        steps.finish()


class Steps:
    """The progress of an operation made of named steps, each a fixed share of it: one
    ``report(done, 100)``, with ``done`` the percentage, from 0 to 100 and never back.

    ``report`` is a :class:`Bar` (whose label then names the current step), a progress
    function or None. :meth:`plan` spreads what is left of the 100 % over the steps to
    come, by their ``shares`` (a step name -> share dict; 1 for a step not in it), and
    :meth:`start_step` starts one, returning the function that moves it through its
    share.
    """

    def __init__(self, report, desc, shares=None):
        self.report = report
        self.desc = desc
        self.shares = shares or {}
        self.done = 0.0
        self.sent = None
        self.planned = {}  # step name -> (start, width) in percent

    def plan(self, names):
        """Spread what is left of the 100 % over the steps ``names``, in this order."""
        weights = [self.shares.get(name, 1.0) for name in names]
        start, left = self.done, 100.0 - self.done
        self.planned = {}
        for name, weight in zip(names, weights):
            width = left * weight / sum(weights)
            self.planned[name] = (start, width)
            start += width

    def start_step(self, name):
        """Start the step ``name`` (a step not planned takes what is left) and return
        its ``advance(done, total)``, which moves the progress through the step's share
        by ``done / total``."""
        if name not in self.planned:
            self.plan([name])
        start, width = self.planned[name]
        if isinstance(self.report, Bar):
            self.report.describe("%s: %s" % (self.desc, name))
        self._move(start)

        def advance(done, total):
            if total:
                self._move(start + width * min(max(done / total, 0.0), 1.0))

        return advance

    def finish(self):
        """Report the operation done, at 100 %."""
        self._move(100.0)

    def _move(self, percent):
        """Report the progress at ``percent``, or where it is when that is less."""
        self.done = max(self.done, round(percent, 1))
        if self.report is not None and self.done != self.sent:
            self.sent = self.done
            self.report(self.done, 100)
