"""The floors an OSM element is on, from its ``level`` tag, and its layer."""

import math
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


def tag_layer(tags):
    """The layer of an element with the OSM ``tags``: its ``layer`` tag as a float (0
    when missing or not a finite number), at most -1 for a tunnel without a positive
    layer."""
    try:
        layer = float(tags.get("layer", 0))
    except (TypeError, ValueError):
        layer = 0.0
    if not math.isfinite(layer):
        layer = 0.0
    if tags.get("tunnel") == "yes" and layer <= 0:
        layer = min(layer, -1.0)
    return layer
