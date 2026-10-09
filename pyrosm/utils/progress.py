"""Download progress on stderr: a tqdm bar, or a widget in Jupyter when ipywidgets is
installed."""

import importlib.util
import re
import sys

# C0 and C1 control characters and DEL, which could move the cursor or forge a line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


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


class Bar:
    """A progress callback, called as ``bar(written, total)``, that draws a bar of bytes on
    stderr described ``desc`` (see :func:`_bar_class`).

    The bar opens at the first call and starts over when ``written`` goes back or ``total``
    changes. When the bar is hidden, ``desc`` is printed once as a line instead.
    """

    def __init__(self, desc):
        self.desc = _printable(desc)
        self.bar = None
        self.written = 0

    def __call__(self, written, total):
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
                disable=disable,
            )
            # Set after: the widget shows a description given to it as HTML until it redraws.
            self.bar.set_description(self.desc)
            if self.bar.disable:
                self.bar.write(self.desc, file=sys.stderr)
        elif written < self.written or total != self.bar.total:
            self.bar.reset(total)
            # reset() keeps the old total when the new one is unknown.
            self.bar.total = total
        self.bar.update(written - self.bar.n)
        self.written = written

    def close(self):
        if self.bar is not None:
            self.bar.close()
