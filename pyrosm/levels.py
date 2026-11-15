"""The floors an OSM element is on, from its ``level`` tag, and its layer."""

import math
import re
from fractions import Fraction

# One part of a level value: a level, such as "-1" or "0.5", or a range, such as "0-2"
# or "-4--2".
_LEVEL_PART = re.compile(r"\s*(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?\s*")
# The most levels that a range of a level value may span; the tallest buildings have
# fewer than 200 floors.
MAX_LEVEL_SPAN = 200


def _level_parts(value):
    """The parts of an OSM ``level`` value that are a level or a range, as ``(first,
    last)`` pairs of the numbers as written; ``last`` is None for a single level. A
    missing value has none."""
    if value is None or value != value:
        return []
    matches = (_LEVEL_PART.fullmatch(part) for part in str(value).split(";"))
    return [match.groups() for match in matches if match]


def level_ranges(value):
    """The levels of an OSM ``level`` value as ``(lowest, highest)`` pairs of floats,
    one for each part separated by a semicolon: ``"-1;0"`` gives ``[(-1.0, -1.0),
    (0.0, 0.0)]`` and ``"0-2"`` gives ``[(0.0, 2.0)]``. Parts that are not a level or a
    range are skipped; None when no part is."""
    ranges = []
    for first, last in _level_parts(value):
        first = float(first)
        last = first if last is None else float(last)
        ranges.append((min(first, last), max(first, last)))
    return ranges or None


def level_list(value):
    """The levels of an OSM ``level`` value written out as OSM lists them: the ends of
    each range and the whole levels between them, lowest first, joined by semicolons.
    ``"0-2"`` gives ``"0;1;2"``, ``"-2;0"`` stays ``"-2;0"`` and ``"0.5-2"`` gives
    ``"0.5;1;2"``. Every digit of a level is kept. A range over more than
    :data:`MAX_LEVEL_SPAN` levels is left out; None when no level is left."""
    levels = {}
    for first, last in _level_parts(value):
        ends = (first, first if last is None else last)
        low, high = sorted(Fraction(number) for number in ends)
        if high - low > MAX_LEVEL_SPAN:
            continue
        for whole in range(math.ceil(low), math.floor(high) + 1):
            levels[Fraction(whole)] = str(whole)
        for number in ends:
            levels[Fraction(number)] = _level_text(number)
    return ";".join(levels[level] for level in sorted(levels)) or None


def _level_text(number):
    """The level written as ``number`` (digits with an optional sign and fraction) as
    OSM writes a level: without the zeros that say nothing and without the sign of
    zero, so ``"01.50"`` gives ``"1.5"`` and ``"-0"`` gives ``"0"``."""
    digits = number.lstrip("-")
    whole, _, fraction = digits.partition(".")
    fraction = fraction.rstrip("0")
    text = (whole.lstrip("0") or "0") + ("." + fraction if fraction else "")
    return text if text == "0" or digits == number else "-" + text


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
