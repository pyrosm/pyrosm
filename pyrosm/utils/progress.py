"""Progress of downloads and reads on stderr: a tqdm bar, or a widget in Jupyter when
ipywidgets is installed."""

import contextlib
import importlib.util
import re
import sys
import time

# C0 and C1 control characters and DEL, which could move the cursor or forge a line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_clock = time.monotonic


def _printable(text):
    """``text`` with each control character replaced by a space."""
    return _CONTROL.sub(" ", text)


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


def validate_progress(progress):
    """Raise ``ValueError`` unless ``progress`` is ``True``, ``False`` or a callable."""
    if not (isinstance(progress, bool) or callable(progress)):
        raise ValueError(
            "progress must be True, False or a callable; got %r." % (progress,)
        )


@contextlib.contextmanager
def reporting(progress, desc, **options):
    """The progress callback for a validated ``progress``: ``None`` for ``False``, the callable
    itself, or for ``True`` a :class:`Bar` described ``desc`` with ``options``, closed on exit.
    """
    if progress is True:
        bar = Bar(desc, **options)
        try:
            yield bar
        finally:
            bar.close()
    else:
        yield progress if callable(progress) else None


class Bar:
    """A progress callback, called as ``bar(done, total)``, that draws a bar of bytes on stderr
    described ``desc`` (see :func:`_bar_class`).

    The bar opens at the first call and starts over when ``done`` goes back or ``total``
    changes. It shows only once it has run ``delay`` seconds, and ``leave=False`` clears it
    when it closes. When the bar is hidden, ``desc`` is printed once as a line when it opens,
    or with ``timed`` a line ``"<desc> took <time>"`` when it closes, if it ran at least
    ``delay`` seconds.
    """

    def __init__(self, desc, delay=0.0, leave=True, timed=False):
        self.desc = _printable(desc)
        self.delay = delay
        self.leave = leave
        self.timed = timed
        self.bar = None
        self.hidden = False
        self.done = 0
        self.start = _clock()

    def __call__(self, done, total):
        if self.bar is None:
            tqdm, disable = _bar_class()
            self.bar = tqdm(
                total=total,
                file=sys.stderr,
                unit="B",
                unit_scale=True,
                unit_divisor=1000,
                mininterval=0.2,
                dynamic_ncols=True,
                delay=self.delay,
                leave=self.leave,
                disable=disable,
            )
            # Set after: the widget shows a description given to it as HTML until it redraws.
            # A delayed bar is not redrawn here, which would show it before its delay.
            self.bar.set_description(self.desc, refresh=self.delay <= 0)
            # Kept, as tqdm marks every bar disabled once it closes.
            self.hidden = self.bar.disable
            if self.hidden and not self.timed:
                self.bar.write(self.desc, file=sys.stderr)
        elif done < self.done or total != self.bar.total:
            if _clock() - self.start < self.delay:
                # Not drawn yet: reset() would draw it before its delay.
                self.bar.n = self.bar.last_print_n = 0
            else:
                self.bar.reset(total)
            # reset() keeps the old total when the new one is unknown.
            self.bar.total = total
        self.bar.update(done - self.bar.n)
        self.done = done

    def close(self):
        if self.bar is None:
            return
        self.bar.close()
        elapsed = _clock() - self.start
        if self.timed and self.hidden and elapsed >= self.delay:
            took = self.bar.format_interval(elapsed)
            self.bar.write("%s took %s" % (self.desc, took), file=sys.stderr)
