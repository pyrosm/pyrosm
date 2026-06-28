"""Advanced (opt-in) filtering: regex dict values and Overpass-style bracket strings.

Covers the filter compiler (parsing + predicate semantics) directly, then the end-to-end
behaviour through the in-memory and out-of-core readers on bundled data. The opt-in forms add
regex value matching (issue #116) and Overpass-style bracket filters with AND-within-a-string /
OR-across-strings, including non-highway networks (issue #341).
"""

import re
import pickle

import pytest

from pyrosm import OSM, get_data
from pyrosm.utils import validate_custom_filter
from pyrosm.filter_compiler import (
    compile_custom_filter,
    is_advanced_filter,
    CompiledFilter,
    Condition,
)


@pytest.fixture
def test_pbf():
    return get_data("test_pbf")


@pytest.fixture
def helsinki_pbf():
    return get_data("helsinki_pbf")


def ids(gdf):
    """Sorted id list of a result frame (None -> empty), for set-equality comparisons."""
    if gdf is None:
        return []
    return sorted(gdf["id"].tolist())


# --------------------------------------------------------------------------------------------
# Compiler unit tests (no PBF)
# --------------------------------------------------------------------------------------------


def test_regex_dict_matches_per_key_variants():
    # issue #116: the same road tagged inconsistently across keys. Conditions are per-key, so
    # each variant is targeted by its own key.
    f = compile_custom_filter(
        {"ref": [re.compile(r"I[ -]?20")], "tiger:name_base": [re.compile(r"I[ -]?20")]}
    )
    assert f.matches({"ref": "I 20"})
    assert f.matches({"ref": "I 20;US 259"})  # substring via re.search
    assert f.matches({"tiger:name_base": "I-20"})
    assert not f.matches({"ref": "US 259"})
    # the literal equivalent misses the ;-joined value and the I-20 variant
    literal = compile_custom_filter({"ref": [re.compile(r"^I 20$")]})
    assert literal.matches({"ref": "I 20"})
    assert not literal.matches({"ref": "I 20;US 259"})


def test_bracket_string_is_and_list_is_or():
    # AND within a string
    f = compile_custom_filter('["highway"~"path"]["bicycle"~"designated"]')
    assert f.matches({"highway": "path", "bicycle": "designated"})
    assert not f.matches({"highway": "path"})
    # OR across a list of strings
    f = compile_custom_filter(['["highway"~"cycleway"]', '["railway"~"subway"]'])
    assert f.matches({"highway": "cycleway"})
    assert f.matches({"railway": "subway"})
    assert not f.matches({"highway": "primary"})


def test_all_operators():
    eq = compile_custom_filter('["a"="x"]')
    assert eq.matches({"a": "x"}) and not eq.matches({"a": "y"})

    # negative ops are satisfied when the key is absent (Overpass semantics), so each is
    # paired with a positive condition (a group needs at least one positive condition)
    ne = compile_custom_filter('["k"]["a"!="x"]')
    assert ne.matches({"k": "1"}) and not ne.matches({"k": "1", "a": "x"})
    assert ne.matches({"k": "1", "a": "y"})

    regex = compile_custom_filter('["a"~"^foo"]')
    assert regex.matches({"a": "foobar"}) and not regex.matches({"a": "barfoo"})

    nregex = compile_custom_filter('["k"]["a"!~"^foo"]')
    assert nregex.matches({"k": "1"}) and not nregex.matches({"k": "1", "a": "foobar"})

    exists = compile_custom_filter('["a"]')
    assert exists.matches({"a": ""}) and not exists.matches({"b": "1"})

    nexists = compile_custom_filter('["k"][!"a"]')
    assert nexists.matches({"k": "1"}) and not nexists.matches({"k": "1", "a": "x"})


def test_regex_dict_mixes_true_str_and_regex():
    # a dict routed to the advanced path lowers every value the same way a plain dict does:
    # bare True or True-in-a-list -> exists, list items str -> eq / pattern -> regex, all OR'd.
    f = compile_custom_filter(
        {"building": True, "shop": [True], "name": ["Foo"], "ref": [re.compile("A1")]}
    )
    assert f.matches({"building": "yes"})
    assert f.matches({"shop": "kiosk"})
    assert f.matches({"name": "Foo"})
    assert f.matches({"ref": "A1 road"})
    assert not f.matches({"highway": "primary"})


def test_regex_dict_value_forms():
    # a compiled pattern is accepted both as a bare value and inside a list (the plan's two
    # forms); both lower to a CompiledFilter and match the same way.
    bare = compile_custom_filter({"ref": re.compile("I[ -]?20")})
    listed = compile_custom_filter({"ref": [re.compile("I[ -]?20")]})
    assert isinstance(bare, CompiledFilter) and isinstance(listed, CompiledFilter)
    assert bare.matches({"ref": "I-20"}) and listed.matches({"ref": "I-20"})
    # a tuple-wrapped pattern is not "advanced" -> falls to the validator, which requires a list
    with pytest.raises(ValueError, match="inside a list"):
        validate_custom_filter({"ref": (re.compile("x"),)})


def test_whitespace_between_brackets():
    f = compile_custom_filter(' ["a"="x"]  ["b"~"y"] ')
    assert f.matches({"a": "x", "b": "yy"})


def test_case_insensitive_flag():
    f = compile_custom_filter('["name"~"oxford",i]')
    assert f.matches({"name": "OXFORD Street"})
    assert not compile_custom_filter('["name"~"oxford"]').matches({"name": "OXFORD Street"})


def test_quote_aware_value_with_bracket():
    # a ] inside a quoted value must not end the bracket early
    f = compile_custom_filter('["name"~"a]b"]')
    assert f.matches({"name": "xa]by"})


def test_positive_keys_exclude_negative_conditions():
    f = compile_custom_filter('["highway"="path"]["bicycle"!="no"]')
    # only the positive condition's key gates candidacy
    assert f.positive_keys == ["highway"]
    assert sorted(f.keys()) == ["bicycle", "highway"]


def test_or_require_appends_only_when_absent():
    f = compile_custom_filter('["amenity"="cafe"]')
    augmented = f.or_require("building")
    assert "building" in augmented.keys()
    assert augmented.matches({"building": "yes"})  # OR term
    assert augmented.matches({"amenity": "cafe"})
    # no-op when the key is already referenced
    assert f.or_require("amenity") is f


@pytest.mark.parametrize(
    "bad",
    [
        '["highway"',  # unbalanced
        '["highway"="a"',  # unbalanced
        '[~"^addr:.*$"~"."]',  # key-regex unsupported
        '["a"!="b"]',  # only a negative condition
        '["a"#"b"]',  # unknown operator
        '["a"="b",x]',  # bad flag
        '["a"="b",i]',  # ,i only valid on regex ops
        '["a"=b]',  # unquoted value
        "[]",  # empty bracket
        "foo",  # not bracket syntax
        '[!"a"junk]',  # trailing text after [!"key"]
        "[!]",  # negated existence with no key
        "[! ]",  # negated existence with blank key
        '[""="x"]',  # empty key
        '[!""]',  # empty key (negated existence)
        "",  # empty filter string (no brackets)
        {"ref": [re.compile("x")], "name": "Foo"},  # bare-string value in a regex dict
        ["building"],  # a bare key, not bracket syntax
        [123],  # non-string entry in a bracket list
        123,  # unsupported type
        {"a": [123], "b": [re.compile("x")]},  # bad value in a regex-bearing dict
    ],
)
def test_invalid_filters_raise(bad):
    with pytest.raises(ValueError):
        compile_custom_filter(bad)


def test_is_advanced_filter_detection():
    assert is_advanced_filter('["a"="b"]')
    assert is_advanced_filter(['["a"="b"]'])
    assert is_advanced_filter({"ref": [re.compile("x")]})
    assert not is_advanced_filter({"highway": ["primary"]})
    assert not is_advanced_filter({"building": True})
    assert not is_advanced_filter(None)


def test_compile_is_idempotent_and_passthrough():
    assert compile_custom_filter(None) is None
    plain = {"highway": ["primary"]}
    assert compile_custom_filter(plain) is plain  # plain dict untouched
    compiled = compile_custom_filter('["a"="b"]')
    assert compile_custom_filter(compiled) is compiled


def test_pickle_roundtrip():
    # the engine ships the filter to spawn workers, so it must pickle (and only sources are
    # stored, never compiled regex objects)
    f = compile_custom_filter(['["highway"~"path"]["bicycle"~"designated"]', '["a"="b"]'])
    restored = pickle.loads(pickle.dumps(f))
    assert restored == f
    assert isinstance(restored, CompiledFilter)
    assert restored.matches({"highway": "path", "bicycle": "designated"})


def test_condition_value_object():
    c = Condition("highway", "regex", "^foo", re.IGNORECASE)
    assert c.is_positive
    assert Condition("a", "ne").is_positive is False


def test_read_quoted_unterminated():
    # a defensive guard in the quote reader (the bracket splitter normally guarantees balanced
    # quotes before this is reached)
    from pyrosm.filter_compiler import _read_quoted

    with pytest.raises(ValueError, match="unterminated"):
        _read_quoted('"abc')


def test_regex_dict_preserves_pattern_flags():
    # a compiled pattern's flags (e.g. DOTALL) survive lowering, not just IGNORECASE
    f = compile_custom_filter({"note": [re.compile("a.b", re.DOTALL)]})
    assert f.matches({"note": "a\nb"})  # '.' matches newline only with DOTALL
    no_dotall = compile_custom_filter({"note": [re.compile("a.b")]})
    assert not no_dotall.matches({"note": "a\nb"})


# --------------------------------------------------------------------------------------------
# In-memory reader (helsinki_pbf)
# --------------------------------------------------------------------------------------------


def test_regex_value_union_matches_literal_union(helsinki_pbf):
    # issue #116 end-to-end: a regex value matching several alternatives equals the literal
    # union a plain dict expresses.
    osm = OSM(helsinki_pbf)
    regex = osm.get_data_by_custom_criteria(
        custom_filter={"highway": [re.compile("footway|cycleway")]}
    )
    literal = osm.get_data_by_custom_criteria(
        custom_filter={"highway": ["footway", "cycleway"]}
    )
    assert ids(regex) == ids(literal)
    assert len(ids(regex)) > 0


def test_regex_matches_semicolon_joined_value(helsinki_pbf):
    # issue #116 end-to-end: a regex value matches a multi-value (;-joined) tag via substring,
    # which a literal exact-match filter misses. The bundled extract has
    # surface="paved;cobblestone".
    osm = OSM(helsinki_pbf)
    regex = osm.get_data_by_custom_criteria(
        custom_filter={"surface": [re.compile("cobblestone")]}
    )
    literal = osm.get_data_by_custom_criteria(custom_filter={"surface": ["cobblestone"]})
    assert regex is not None
    assert "paved;cobblestone" in set(regex["surface"].dropna())
    literal_surfaces = set() if literal is None else set(literal["surface"].dropna())
    assert "paved;cobblestone" not in literal_surfaces


def test_network_bracket_union_matches_dict(helsinki_pbf):
    # issue #341: a list of bracket strings (OR) selecting highway values equals the dict form.
    osm = OSM(helsinki_pbf)
    bracket = osm.get_network(
        custom_filter=['["highway"~"^footway$"]', '["highway"~"^cycleway$"]'],
        filter_type="keep",
    )
    osm2 = OSM(helsinki_pbf)
    plain = osm2.get_network(
        custom_filter={"highway": ["footway", "cycleway"]}, filter_type="keep"
    )
    assert ids(bracket) == ids(plain)
    assert len(ids(bracket)) > 0
    assert set(bracket["highway"].unique()) <= {"footway", "cycleway"}


def test_network_and_condition_is_subset(helsinki_pbf):
    # issue #341 AND case: every returned way satisfies both brackets and is a subset of the
    # first bracket alone (the bundled extract has path/footway ways with a bicycle tag).
    both = OSM(helsinki_pbf).get_network(
        custom_filter='["highway"~"path|footway"]["bicycle"~"."]', filter_type="keep"
    )
    first = OSM(helsinki_pbf).get_network(
        custom_filter='["highway"~"path|footway"]', filter_type="keep"
    )
    assert both is not None and len(both) > 0
    assert set(ids(both)).issubset(set(ids(first)))
    assert both["bicycle"].notna().all()
    assert both["highway"].str.contains("path|footway").all()


def test_network_non_highway_key(helsinki_pbf):
    # issue #341: advanced filters select by their own positive keys, so a non-highway network
    # (e.g. railway) is possible -- the historical highway-only candidacy did not allow this.
    rail = OSM(helsinki_pbf).get_network(
        custom_filter='["railway"~"subway|tram|rail"]', filter_type="keep"
    )
    assert rail is not None and len(rail) > 0
    assert "railway" in rail.columns
    assert rail["railway"].notna().all()
    assert rail["railway"].str.contains("subway|tram|rail").all()


def test_network_filter_type_matrix(helsinki_pbf):
    # advanced filters default to keep; explicit keep matches the default; keep and exclude
    # partition the candidate universe (here: ways carrying a highway tag).
    keep_default = OSM(helsinki_pbf).get_network(custom_filter='["highway"~"^footway$"]')
    keep_explicit = OSM(helsinki_pbf).get_network(
        custom_filter='["highway"~"^footway$"]', filter_type="keep"
    )
    exclude = OSM(helsinki_pbf).get_network(
        custom_filter='["highway"~"^footway$"]', filter_type="exclude"
    )
    universe = OSM(helsinki_pbf).get_network(custom_filter='["highway"~"."]', filter_type="keep")

    assert ids(keep_default) == ids(keep_explicit)
    keep_set, exclude_set, universe_set = set(ids(keep_default)), set(ids(exclude)), set(ids(universe))
    assert keep_set.isdisjoint(exclude_set)
    assert keep_set | exclude_set == universe_set


def test_buildings_layer_injection_parity(helsinki_pbf):
    # the layer key is OR-injected for advanced filters exactly as for dicts: building OR amenity.
    advanced = OSM(helsinki_pbf).get_buildings(custom_filter='["amenity"="restaurant"]')
    plain = OSM(helsinki_pbf).get_buildings(custom_filter={"amenity": ["restaurant"]})
    assert ids(advanced) == ids(plain)


def test_plain_dict_unchanged(helsinki_pbf):
    # backward compatibility: a plain-dict filter is untouched by the advanced path.
    a = OSM(helsinki_pbf).get_buildings(custom_filter={"building": ["residential"]})
    b = OSM(helsinki_pbf).get_buildings(custom_filter={"building": ["residential"]})
    assert ids(a) == ids(b)
    assert len(ids(a)) > 0


# --------------------------------------------------------------------------------------------
# Out-of-core engine parity (single worker)
# --------------------------------------------------------------------------------------------


def test_engine_parity_buildings(helsinki_pbf):
    advanced = '["building"~"."]'
    in_memory = OSM(helsinki_pbf).get_buildings(custom_filter=advanced)
    engine = OSM(helsinki_pbf, engine="out_of_core", workers=1).get_buildings(
        custom_filter=advanced
    )
    assert ids(engine) == ids(in_memory)


def test_engine_parity_network(helsinki_pbf):
    advanced = '["highway"~"footway|cycleway"]'
    in_memory = OSM(helsinki_pbf).get_network(custom_filter=advanced, filter_type="keep")
    engine = OSM(helsinki_pbf, engine="out_of_core", workers=1).get_network(
        custom_filter=advanced, filter_type="keep"
    )
    assert ids(engine) == ids(in_memory)


def test_engine_parity_custom_criteria(helsinki_pbf):
    advanced = '["highway"~"footway|cycleway"]'
    in_memory = OSM(helsinki_pbf).get_data_by_custom_criteria(custom_filter=advanced)
    engine = OSM(helsinki_pbf, engine="out_of_core", workers=1).get_data_by_custom_criteria(
        custom_filter=advanced
    )
    assert ids(engine) == ids(in_memory)


def test_engine_parity_pois(helsinki_pbf):
    # POIs do not inject a layer key, so candidacy is the filter's own positive keys.
    advanced = '["amenity"~"restaurant|cafe"]'
    in_memory = OSM(helsinki_pbf).get_pois(custom_filter=advanced)
    engine = OSM(helsinki_pbf, engine="out_of_core", workers=1).get_pois(
        custom_filter=advanced
    )
    assert ids(engine) == ids(in_memory)


def test_engine_get_network_resolves_default_filter_type(helsinki_pbf):
    # called directly (not via OSM, which passes a concrete filter_type), the engine reader
    # resolves an omitted filter_type to keep for an advanced filter.
    import pyrosm.engine as engine_backend

    advanced = '["highway"~"footway|cycleway"]'
    direct = engine_backend.get_network(helsinki_pbf, custom_filter=advanced, workers=1)
    in_memory = OSM(helsinki_pbf).get_network(custom_filter=advanced, filter_type="keep")
    assert ids(direct) == ids(in_memory)


def test_engine_parity_landuse_injection(helsinki_pbf):
    # advanced filter on a layer that OR-injects its key (landuse): engine matches in-memory.
    advanced = '["landuse"~"residential|forest"]'
    in_memory = OSM(helsinki_pbf).get_landuse(custom_filter=advanced)
    engine = OSM(helsinki_pbf, engine="out_of_core", workers=1).get_landuse(
        custom_filter=advanced
    )
    assert ids(engine) == ids(in_memory)
