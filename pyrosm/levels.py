"""The floors an OSM element is on, from its ``level`` tag."""

import re

# One part of a level value: a level, such as "-1" or "0.5", or a range, such as "0-2"
# or "-4--2".
_LEVEL_PART = re.compile(r"\s*(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?\s*")


def level_ranges(value):
    """The levels of an OSM ``level`` value as ``(lowest, highest)`` pairs of floats,
    one for each part separated by a semicolon: ``"-1;0"`` gives ``[(-1.0, -1.0),
    (0.0, 0.0)]`` and ``"0-2"`` gives ``[(0.0, 2.0)]``. Parts that are not a level or a
    range are skipped; None when no part is."""
    if value is None or value != value:
        return None
    ranges = []
    for part in str(value).split(";"):
        match = _LEVEL_PART.fullmatch(part)
        if match:
            first = float(match.group(1))
            last = float(match.group(2)) if match.group(2) is not None else first
            ranges.append((min(first, last), max(first, last)))
    return ranges or None
