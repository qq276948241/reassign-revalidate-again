"""Tests for the type-tagged round-trip serialization in ``attr._roundtrip``."""

import pytest

import attr
from attr import RoundtripError, dumpb, dumps, loadb, loads


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        42,
        -7,
        3.5,
        -0.25,
        1e100,
        "",
        "hello",
        [1, "two", 3.0, False, None],
        {"a": 1, "b": "two", "c": True, "d": None},
        {"a": ["b", "c"]},
        [[{"deep": [1, 2]}], {"x": -1, "list": "n"}],
        {},
        ["", None],
        None,
    ],
)
def test_roundtrip_preserves_type_and_value(value):
    restored = loads(dumps(value))

    assert restored == value
    assert type(restored) is type(value)


def test_numbers_do_not_become_strings():
    value = {"i": 42, "f": 3.5, "neg": -9}

    restored = loads(dumps(value))

    assert isinstance(restored["i"], int)
    assert isinstance(restored["f"], float)
    assert isinstance(restored["neg"], int)


def test_bool_is_not_int():
    assert loads(dumps(True)) is True
    assert loads(dumps(False)) is False
    assert dumps(True) != dumps(1)


def test_four_empty_values_stay_distinct():
    value = {"arr": [], "obj": {}, "str": "", "nil": None}

    restored = loads(dumps(value))

    assert restored["arr"] == []
    assert isinstance(restored["arr"], list)
    assert restored["obj"] == {}
    assert isinstance(restored["obj"], dict)
    assert restored["str"] == ""
    assert isinstance(restored["str"], str)
    assert restored["nil"] is None


def test_four_empty_placeholders_are_distinct():
    restored = loads(dumps({"arr": [], "obj": {}, "str": "", "nil": None}))

    assert restored["str"] != restored["nil"]
    assert type(restored["str"]) is str
    assert restored["nil"] is None
    assert type(restored["arr"]) is list
    assert type(restored["obj"]) is dict


def test_nested_layers_keep_their_types():
    value = {"a": [{"b": [[1, 2], [3]]}]}

    restored = loads(dumps(value))

    assert isinstance(restored, dict)
    assert isinstance(restored["a"], list)
    assert isinstance(restored["a"][0], dict)
    assert isinstance(restored["a"][0]["b"], list)
    assert isinstance(restored["a"][0]["b"][0], list)
    assert isinstance(restored["a"][0]["b"][1], list)


def test_nested_structures_keep_layers():
    value = {"a": [{"b": [1, [2, [3]]]}]}

    restored = loads(dumps(value))

    assert restored == value
    assert isinstance(restored["a"][0]["b"][1][1], list)


def test_keys_round_trip_with_markers_and_escapes():
    value = {'a:b,c[d]e{f}"g\\h': 1, "é键": 2}

    restored = loads(dumps(value))

    assert restored == value


def test_keys_with_special_characters_round_trip():
    value = {'"': 1, "i": 2, "~": 3, "[": 4, "{": 5}

    assert loads(dumps(value)) == value


def test_escaped_marker_is_content_not_tag():
    text = dumps("[not a list")

    restored = loads(text)

    assert restored == "[not a list"
    assert isinstance(restored, str)


@pytest.mark.parametrize(
    "content",
    [
        '"',
        "[",
        "{",
        "n;",
        "i123;",
        "semi;colon",
        "bt;",
        "\\;",
        '\\"',
        "back\\slash",
        "looks ; like , separators =",
    ],
)
def test_markers_inside_content_are_escaped(content):
    restored = loads(dumps(content))

    assert restored == content
    assert isinstance(restored, str)


def test_escape_takes_precedence_over_marker():
    text = dumps("i1")

    restored = loads(text)

    assert restored == "i1"
    assert isinstance(restored, str)


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-32", "latin-1", "gb18030", "big5"]
)
def test_charset_round_trip(encoding):
    value = {
        "msg": "café" if encoding == "latin-1" else "中文",
        "n": 1,
        "none": None,
    }

    assert loadb(dumpb(value, encoding=encoding), encoding=encoding) == value


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "gb18030"])
def test_escape_before_charset_keeps_tags(encoding):
    value = {'"\\': ["中文", 1, None], "q": ["é", 1]}

    assert loadb(dumpb(value, encoding=encoding), encoding=encoding) == value


def test_unsupported_characters_for_charset_raise():
    with pytest.raises(RoundtripError):
        dumpb("中文", encoding="ascii")


def test_legacy_untagged_string_still_loads():
    assert loads("plain legacy content") == "plain legacy content"


def test_bare_top_level_string_passes_through():
    assert loads(dumps("hello")) == "hello"


def test_legacy_unmarked_top_level_still_works():
    assert loads("anything goes here") == "anything goes here"


def test_legacy_loads_accepts_marked_object():
    assert loads("{a:i1}") == {"a": 1}


def test_tagged_and_legacy_mix_without_pollution():
    restored = loads('["tagged",plain,42]')

    assert restored == ["tagged", "plain", "42"]
    assert isinstance(restored[0], str)
    assert isinstance(restored[1], str)
    assert isinstance(restored[2], str)

    restored = loads('{a:i1,"b":b1,c:~}')

    assert restored == {"a": 1, "b": True, "c": None}
    assert isinstance(restored["a"], int)
    assert restored["b"] is True
    assert restored["c"] is None


def test_marked_and_legacy_forms_coexist():
    payload = "[" + dumps("plain,42") + ",i7;]"

    assert loads(payload) == ["plain,42", 7]


def test_error_names_key_and_types():
    with pytest.raises(RoundtripError) as exc:
        loads("{a:i1,")

    assert "a" in str(exc.value)


def test_type_mismatch_error_names_key_and_types():
    with pytest.raises(RoundtripError) as exc:
        loads('{"a.b":ixyz;}')

    message = str(exc.value)
    assert "a.b" in message
    assert "int" in message


def test_trailing_content_is_rejected():
    with pytest.raises(RoundtripError):
        loads("i1x")


def test_trailing_data_rejected():
    with pytest.raises(RoundtripError):
        loads("~junk")


@pytest.mark.parametrize(
    "payload", ["", "{", "[", "i;", "b2;", '"unterminated', "{a:i1,}"]
)
def test_malformed_input_raises(payload):
    with pytest.raises(RoundtripError):
        loads(payload)


def test_bad_charset_input_raises():
    with pytest.raises(RoundtripError):
        loadb(b"\xff\xfe\xfd", "ascii")


def test_types_are_preserved_not_stringified():
    value = {"s": "1", "i": 1, "f": 1.0, "b": True, "n": None}

    restored = loads(dumps(value))

    assert type(restored["s"]) is str
    assert type(restored["i"]) is int
    assert type(restored["f"]) is float
    assert type(restored["b"]) is bool
    assert restored["n"] is None


def test_assert_roundtrip_reports_value_difference():
    with pytest.raises(RoundtripError):
        loads("{oops")


def test_public_api_exported():
    assert attr.dumps is dumps
    assert attr.loads is loads
    assert attr.dumpb is dumpb
    assert attr.loadb is loadb
    assert attr.RoundtripError is RoundtripError
