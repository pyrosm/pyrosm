"""Compile advanced (opt-in) ``custom_filter`` forms into an evaluable predicate.

Pyrosm's default ``custom_filter`` is a plain dict of exact string values combined with
OR. Two richer forms are supported on top of that, and both are strictly opt-in:

1. **Regex values** inside a dict, e.g. ``{"ref": [re.compile(r"I[ -]?20")]}``. A compiled
   pattern is matched against the tag value with ``re.search`` (substring match), which finds
   inconsistently tagged values a literal filter misses.

2. **Overpass-style bracket strings**, e.g.
   ``['["highway"~"cycleway"]', '["highway"~"path"]["bicycle"~"designated"]']``. Each string is
   the AND of its brackets; a list of strings is the OR of the strings. This is the
   tag-filter subset of Overpass QL that osmnx users already type — not the full language.

Both forms lower into the same internal representation: a ``CompiledFilter`` holding a
**disjunctive normal form** predicate (OR of AND-groups of :class:`Condition`). A plain
dict with only string / list / ``True`` values is left untouched and keeps pyrosm's existing
fast dict path; only the two opt-in forms produce a ``CompiledFilter``.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

# Operators whose condition requires the key to be present with a matching value. They
# alone define the candidate-key gate: an element that carries none of these keys cannot
# satisfy any AND-group, so it can be discarded before evaluation.
POSITIVE_OPERATORS = frozenset({"eq", "regex", "exists"})


@lru_cache(maxsize=None)
def _compiled_pattern(pattern, flags):
    """Compile and cache a regex source string (per process, so it survives pickling)."""
    return re.compile(pattern, flags)


@dataclass(frozen=True)
class Condition:
    """A single tag test, e.g. ``highway = residential`` or ``ref ~ "I[ -]?20"``.

    ``operator`` is one of ``eq``, ``ne``, ``regex``, ``nregex``, ``exists``, ``nexists``.
    For ``eq``/``ne`` ``value`` is a literal string; for ``regex``/``nregex`` it is the regex
    source and ``flags`` the ``re`` flags to compile it with (preserving e.g. ``IGNORECASE`` or
    ``DOTALL`` from a caller's compiled pattern). ``value``/``flags`` are unused for
    ``exists``/``nexists``. Only regex sources and integer flags (not compiled objects) are
    stored, so the condition pickles cleanly across the engine's worker processes.
    """

    key: str
    operator: str
    value: str = ""
    flags: int = 0

    @property
    def is_positive(self):
        return self.operator in POSITIVE_OPERATORS

    def matches(self, tags):
        """Return whether ``tags`` (a key -> value mapping) satisfies this condition.

        A negative operator (``ne``/``nregex``/``nexists``) is satisfied when the key is
        absent, mirroring Overpass: ``["bicycle"!="no"]`` keeps ways that have no bicycle tag.
        """
        if self.operator == "exists":
            return self.key in tags
        if self.operator == "nexists":
            return self.key not in tags
        if self.key not in tags:
            # Positive operators need the key present; negative ones are satisfied without it.
            return self.operator in ("ne", "nregex")

        tag_value = tags[self.key]
        if self.operator == "eq":
            return tag_value == self.value
        if self.operator == "ne":
            return tag_value != self.value
        pattern = _compiled_pattern(self.value, self.flags)
        found = pattern.search(str(tag_value)) is not None
        return found if self.operator == "regex" else not found


@dataclass(frozen=True)
class CompiledFilter:
    """An OR of AND-groups of :class:`Condition`, evaluated per element.

    ``groups`` is a tuple of groups; each group is a tuple of conditions. An element matches
    when *any* group matches, and a group matches when *all* its conditions hold.
    """

    groups: tuple = field(default_factory=tuple)

    def matches(self, tags):
        return any(
            all(condition.matches(tags) for condition in group) for group in self.groups
        )

    @property
    def positive_keys(self):
        """Keys from positive conditions, used as pyrosm's candidate-key gate (``osm_keys``)."""
        return sorted(
            {
                condition.key
                for group in self.groups
                for condition in group
                if condition.is_positive
            }
        )

    def keys(self):
        """Every referenced key, so callers can expose filter keys as result columns."""
        return sorted({condition.key for group in self.groups for condition in group})

    def to_json(self):
        """The filter as nested lists that JSON can hold (a frame's ``attrs`` must be JSON
        for pandas to write them): a list per AND-group of ``[key, operator, value, flags]``
        per condition."""
        return [
            [[c.key, c.operator, c.value, c.flags] for c in group]
            for group in self.groups
        ]

    @classmethod
    def from_json(cls, groups):
        """The filter whose :meth:`to_json` is ``groups``."""
        return cls(tuple(tuple(Condition(*c) for c in group) for group in groups))

    def renamed(self, old, new):
        """The filter with every condition on the key ``old`` reading the key ``new``."""
        return CompiledFilter(
            tuple(
                tuple(
                    Condition(new, c.operator, c.value, c.flags) if c.key == old else c
                    for c in group
                )
                for group in self.groups
            )
        )

    def and_require(self, key, values=True):
        """The filter with every AND-group that has no positive condition on ``key`` (one an
        element can only meet by carrying ``key``) also requiring it: that ``key`` exists
        (``values=True``) or has one of ``values`` (a list of strings). A group with such a
        condition keeps its own conditions on ``key``."""
        if values is True:
            required = [Condition(key, "exists")]
        else:
            required = [Condition(key, "eq", value) for value in values]
        groups = []
        for group in self.groups:
            if any(c.key == key and c.is_positive for c in group):
                groups.append(group)
            else:
                groups.extend(group + (condition,) for condition in required)
        return CompiledFilter(tuple(groups))


def _read_quoted(text):
    """Read a leading single- or double-quoted token; return (unquoted_value, remainder)."""
    if not text:
        raise ValueError("expected a quoted key or value, got empty text")
    quote = text[0]
    if quote not in "\"'":
        raise ValueError(f"expected a quoted key or value, got: {text!r}")
    end = text.find(quote, 1)
    if end == -1:
        raise ValueError(f"unterminated quote in: {text!r}")
    return text[1:end], text[end + 1 :]


def _read_operator(text):
    """Read a leading comparison operator; return (condition_operator, remainder)."""
    for token, operator in (
        ("!=", "ne"),
        ("!~", "nregex"),
        ("=", "eq"),
        ("~", "regex"),
    ):
        if text.startswith(token):
            return operator, text[len(token) :]
    raise ValueError(f"expected one of = != ~ !~, got: {text!r}")


def _split_brackets(spec):
    """Split ``'["a"="b"]["c"]'`` into the bracket interiors ``['"a"="b"', '"c"']``.

    Quote-aware so a ``]`` inside a quoted value does not end the bracket early.
    """
    interiors = []
    i, n = 0, len(spec)
    while i < n:
        if spec[i].isspace():
            i += 1
            continue
        if spec[i] != "[":
            raise ValueError(f"expected '[' at position {i} in filter string: {spec!r}")
        j = i + 1
        quote = None
        while j < n:
            char = spec[j]
            if quote is not None:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "]":
                break
            j += 1
        else:
            raise ValueError(f"unbalanced '[' in filter string: {spec!r}")
        interiors.append(spec[i + 1 : j])
        i = j + 1
    return interiors


def _parse_bracket(interior):
    """Parse one bracket interior (e.g. ``'"highway"~"path"'``) into a :class:`Condition`."""
    text = interior.strip()
    if not text:
        raise ValueError("empty bracket '[]' in filter string")

    # [!"key"] -> the key must be absent.
    if text.startswith("!"):
        key, remainder = _read_quoted(text[1:].strip())
        if not key:
            raise ValueError(f"empty key in filter bracket: {interior!r}")
        if remainder.strip():
            raise ValueError(f'unexpected text after [!"key"]: {interior!r}')
        return Condition(key, "nexists")

    # [~"keyregex"~"valueregex"] -> regex on the key name. Unsupported: it would force a
    # full scan of every tag of every element, defeating the candidate-key gate.
    if text.startswith("~"):
        raise ValueError(
            'key-regex filters (e.g. [~"^addr:.*$"~"."]) are not supported'
        )

    key, remainder = _read_quoted(text)
    if not key:
        raise ValueError(f"empty key in filter bracket: {interior!r}")
    remainder = remainder.strip()

    # ["key"] -> the key must be present (any value).
    if not remainder:
        return Condition(key, "exists")

    operator, remainder = _read_operator(remainder)
    value, remainder = _read_quoted(remainder.strip())
    remainder = remainder.strip()

    flags = 0
    if remainder:
        if remainder.replace(" ", "") == ",i":
            if operator not in ("regex", "nregex"):
                raise ValueError(
                    "the ',i' flag is only valid on the ~ and !~ operators"
                )
            flags = re.IGNORECASE
        else:
            raise ValueError(f"unexpected text after value: {interior!r}")

    return Condition(key, operator, value, flags)


def parse_bracket_filter(spec):
    """Parse a bracket string (or list of strings) into AND-groups of conditions.

    One string becomes one AND-group; a list of strings becomes one group each (OR). Every
    group must hold at least one positive condition so the candidate-key gate stays sound.
    """
    specs = [spec] if isinstance(spec, str) else list(spec)
    groups = []
    for one in specs:
        if not isinstance(one, str):
            raise ValueError(
                f"each bracket filter must be a string, got {one!r} of type {type(one)}"
            )
        conditions = tuple(_parse_bracket(b) for b in _split_brackets(one))
        if not conditions:
            raise ValueError(f"filter string has no brackets: {one!r}")
        if not any(condition.is_positive for condition in conditions):
            raise ValueError(
                f"filter string {one!r} has only negative conditions; add at least one "
                f"positive condition (=, ~, or a bare key) so it can select elements"
            )
        groups.append(conditions)
    return tuple(groups)


def _dict_to_groups(custom_filter):
    """Lower a dict that contains at least one regex value into OR-of-singleton groups.

    Each ``key: value`` pair becomes its own AND-group (preserving the dict's OR semantics).
    Values follow the same contract as a plain dict filter: ``True`` (-> ``exists``) or a list
    whose items are strings (-> ``eq``), compiled patterns (-> ``regex``), or ``True``.
    """
    groups = []
    for key, values in custom_filter.items():
        if values is True:
            groups.append((Condition(key, "exists"),))
            continue
        # A bare compiled pattern is allowed (it is unambiguously a regex intent); a string
        # must still be wrapped in a list, as in a plain dict filter.
        if isinstance(values, re.Pattern):
            groups.append((Condition(key, "regex", values.pattern, values.flags),))
            continue
        if not isinstance(values, list):
            raise ValueError(
                f"value for key {key!r} should be inside a list. Got {values!r}."
            )
        for value in values:
            if value is True:
                groups.append((Condition(key, "exists"),))
            elif isinstance(value, re.Pattern):
                groups.append((Condition(key, "regex", value.pattern, value.flags),))
            elif isinstance(value, str):
                groups.append((Condition(key, "eq", value),))
            else:
                raise ValueError(
                    f"value {value!r} for key {key!r} must be a string, a compiled regex, "
                    f"or True"
                )
    return tuple(groups)


def _dict_has_regex(custom_filter):
    # A regex value is recognised as a bare compiled pattern or one inside a value list (matching
    # the plan); a tuple-wrapped pattern is not "advanced" and is rejected by the validator.
    return any(
        isinstance(values, re.Pattern)
        or (isinstance(values, list) and any(isinstance(v, re.Pattern) for v in values))
        for values in custom_filter.values()
    )


def is_advanced_filter(custom_filter):
    """Whether ``custom_filter`` uses an opt-in advanced form (string, list, or regex dict)."""
    if isinstance(custom_filter, (str, list, tuple, CompiledFilter)):
        return True
    if isinstance(custom_filter, dict):
        return _dict_has_regex(custom_filter)
    return False


def rename_filter_key(data_filter, osm_keys, old, new):
    """``(data_filter, osm_keys)`` with the key ``old`` read as ``new``: in a dict filter, in
    a :class:`CompiledFilter` and in the candidate keys ``osm_keys`` (a key or a list).
    """
    if isinstance(data_filter, CompiledFilter):
        data_filter = data_filter.renamed(old, new)
    elif isinstance(data_filter, dict) and old in data_filter:
        data_filter = {(new if k == old else k): v for k, v in data_filter.items()}
    if isinstance(osm_keys, str):
        osm_keys = new if osm_keys == old else osm_keys
    else:
        osm_keys = [new if k == old else k for k in osm_keys]
    return data_filter, osm_keys


def as_compiled_filter(custom_filter):
    """``custom_filter`` (a plain dict filter or a :class:`CompiledFilter`) as a
    :class:`CompiledFilter`, whose :meth:`~CompiledFilter.matches` tests an element's tags.
    """
    if isinstance(custom_filter, CompiledFilter):
        return custom_filter
    return CompiledFilter(_dict_to_groups(custom_filter))


def dict_as_bracket_filter(custom_filter):
    """The bracket filter that keeps what ``custom_filter``, a plain dict as
    :func:`pyrosm.utils.validate_custom_filter` returns it, keeps: a string for one
    condition, otherwise a list of strings (their OR). None when the dict has no
    condition, or a key or value holds both quote characters, which a bracket cannot
    quote."""
    brackets = []
    for key, values in custom_filter.items():
        for value in values:
            quoted = []
            for part in [key] if value is True else [key, value]:
                quote = "'" if '"' in part else '"'
                if quote in part:
                    return None
                quoted.append(quote + part + quote)
            brackets.append("[%s]" % "=".join(quoted))
    if not brackets:
        return None
    return brackets[0] if len(brackets) == 1 else brackets


def compile_custom_filter(custom_filter):
    """Turn any ``custom_filter`` the user passed into the form the readers expect.

    Returns ``None`` for ``None``, a :class:`CompiledFilter` for the opt-in advanced forms
    (bracket string(s) or a regex-bearing dict), and the unchanged dict for a plain dict so
    the existing fast path is preserved. Calling it again on its own result changes nothing,
    so it is safe to call more than once on the same filter.
    """
    if custom_filter is None or isinstance(custom_filter, CompiledFilter):
        return custom_filter
    if isinstance(custom_filter, (str, list, tuple)):
        return CompiledFilter(parse_bracket_filter(custom_filter))
    if isinstance(custom_filter, dict):
        if _dict_has_regex(custom_filter):
            return CompiledFilter(_dict_to_groups(custom_filter))
        return custom_filter
    raise ValueError(
        f"'custom_filter' should be a dict, a bracket-filter string, or a list of such "
        f"strings. Got {custom_filter!r} of type {type(custom_filter)}."
    )
