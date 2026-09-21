import pytest

from attr.roundtrip import RoundtripError, dumpb, dumps, loadb, loads


class TestTags:
    def test_scalar_types_keep_their_tag(self):
        assert dumps(42) == "i42"
        assert dumps(3.5) == "d3.5"
        assert dumps(True) == "b1"
        assert dumps(False) == "b0"
        assert dumps(None) == "~"
        assert dumps("text") == '"text"'

    def test_scalar_types_come_back_as_the_same_type(self):
        for value in (42, -7, 3.5, -0.25, True, False, None, "text"):
            restored = loads(dumps(value))
            assert restored == value
            assert type(restored) is type(value)

    def test_number_does_not_become_string(self):
        restored = loads("i42")
        assert isinstance(restored, int)
        assert restored == 42
        restored = loads("d3.5")
        assert isinstance(restored, float)
        assert restored == 3.5

    def test_bool_is_not_a_number(self):
        assert dumps(True) == "b1"
        assert isinstance(loads("b1"), bool)
        assert dumps(False) == "b0"
        assert isinstance(loads("b0"), bool)


class TestEmpty:
    def test_four_empty_values_keep_their_shape(self):
        assert dumps([]) == "[]"
        assert dumps({}) == "{}"
        assert dumps("") == '""'
        assert dumps(None) == "~"

    def test_four_empty_values_read_back_distinctly(self):
        assert loads("[]") == []
        assert loads("{}") == {}
        assert loads('""') == ""
        assert loads("~") is None

    def test_empty_string_is_not_null(self):
        assert loads(dumps("")) == ""
        assert loads(dumps(None)) is None


class TestNesting:
    def test_nested_object_and_array(self):
        value = {"a": [1, 2], "b": [3, 4]}
        assert loads(dumps(value)) == value

    def test_deep_array_is_not_flattened(self):
        value = [[1], [2], [[3]]]
        restored = loads(dumps(value))
        assert restored == value
        assert isinstance(restored[2], list)

    def test_deep_object_is_not_flattened(self):
        value = {"a": {"b": {"c": 1}}}
        restored = loads(dumps(value))
        assert restored == value
        assert isinstance(restored["a"]["b"], dict)

    def test_mixed_levels_keep_their_types(self):
        value = {"k": [1, "x", None, True], "y": {"n": None}}
        restored = loads(dumps(value))
        assert restored == value
        assert type(restored["k"]) is list
        assert type(restored["k"][0]) is int
        assert type(restored["y"]) is dict

    def test_tuple_becomes_list(self):
        assert loads(dumps((1, 2))) == [1, 2]

    def test_string_keys_round_trip(self):
        for index, key in enumerate(("a.b[c]", "x:y~z", "")):
            value = {key: index}
            assert loads(dumps(value)) == value

    def test_non_string_keys_are_rejected_with_path(self):
        value = {1: 2}
        with pytest.raises(RoundtripError) as excinfo:
            dumps(value)
        assert "keys must be strings" in str(excinfo.value)
        assert "int" in str(excinfo.value)

    def test_unsupported_type_names_the_path(self):
        value = {"a": object()}
        with pytest.raises(RoundtripError) as excinfo:
            dumps(value)
        assert "a: cannot encode object" in str(excinfo.value)


class TestEscape:
    def test_marker_characters_are_escaped(self):
        assert dumps("a[b]{c}:d,e~") == '"a\\x005bb\\x005d\\x007bc\\x007d\\x003ad\\x002ce~"'

    def test_escaped_marker_survives_round_trip(self):
        text = "{}[]:,\\"
        assert loads(dumps(text)) == text

    def test_quote_and_backslash_round_trip(self):
        text = '"\\'
        assert loads(dumps(text)) == text

    def test_control_characters_round_trip(self):
        text = "\x00\x08\x0c\n\r\t\x1f\x7f"
        assert loads(dumps(text)) == text

    def test_marker_does_not_swallow_escape(self):
        assert loads('"\\x005b"') == "["
        assert loads('"\\u00e9"') == "é"

    def test_bad_escape_is_rejected(self):
        with pytest.raises(RoundtripError):
            loads('"\\u00zz"')


class TestCharset:
    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32", "latin-1"])
    def test_bytes_round_trip_keeps_types(self, encoding):
        value = {"v": [1, 2.5, True], "k": "n"}
        assert loadb(dumpb(value, encoding), encoding) == value

    def test_escape_happens_before_charset_change(self):
        value = "é"
        data = dumpb(value, "utf-16")
        assert loadb(data, "utf-16") == value

    def test_bad_bytes_raise_roundtrip_error(self):
        with pytest.raises(RoundtripError):
            loadb(b"\xff\xfe\x00", "utf-8")


class TestLegacy:
    def test_plain_content_is_a_legacy_string(self):
        assert loads("plain text") == "plain text"

    def test_legacy_and_tagged_mix_in_one_container(self):
        encoded = dumps({"legacy": "i-am-legacy", "tagged": 1})
        value = loads(encoded)
        assert value == {"legacy": "i-am-legacy", "tagged": 1}

    def test_legacy_key_supported_next_to_tagged_key(self):
        restored = loads('{legacy:i42,"tagged":i43}')
        assert restored["legacy"] == 42
        assert restored["tagged"] == 43

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("i", "malformed number"),
            ("d-x", "malformed number"),
            ("b2", "bad bool tag"),
            ("[", "missing value"),
            ("{", "expected ':'"),
            ("~x", "trailing content after value"),
            ("[i1 2]", "expected ',' or ']'"),
            ("i1x", "trailing content after value"),
        ],
    )
    def test_malformed_input_reports_path_and_position(self, text, message):
        with pytest.raises(RoundtripError) as excinfo:
            loads(text)
        assert message in str(excinfo.value)
        assert "at position" in str(excinfo.value)

    def test_nested_error_names_the_key_path(self):
        with pytest.raises(RoundtripError) as excinfo:
            loads('{"a":{"b":"oops')
        assert "a.b" in str(excinfo.value)

    def test_nested_error_names_the_index_path(self):
        with pytest.raises(RoundtripError) as excinfo:
            loads("[1,[2,b3]]")
        assert "[1][1]" in str(excinfo.value)


class TestTypeGuards:
    def test_loads_requires_str(self):
        with pytest.raises(RoundtripError) as excinfo:
            loads(1)
        assert "loads expects str, got int" in str(excinfo.value)

    def test_loadb_requires_bytes(self):
        with pytest.raises(RoundtripError) as excinfo:
            loadb("x")
        assert "loadb expects bytes, got str" in str(excinfo.value)

    def test_non_finite_float_rejected(self):
        for value in (float("nan"), float("inf")):
            with pytest.raises(RoundtripError):
                dumps(value)
