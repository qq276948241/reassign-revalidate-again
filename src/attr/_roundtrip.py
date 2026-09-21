"""
Type-tagged, charset-aware round-trip serialization.

Every value is written with a marker that records its type, so that
``loads(dumps(value))`` reproduces both the value *and* its type.  Legacy,
unmarked strings are still accepted: unknown content is treated as plain
string data and never interpreted as a tag.

The wire format is a small, self-delimiting one::

    list   [item,item,...]
    object {key:value,key:value,...}
    string "..."
    int    i<repr>;
    float  f<repr>;
    bool   b0; / b1;
    null   ~

String content is escaped before it leaves the encoder, and only the decoder
removes the escapes; therefore a marker character appearing inside escaped
content can never be mistaken for the real marker.
"""

from __future__ import annotations


class RoundtripError(ValueError):
    """Raised when a serialized payload cannot be decoded."""


_TAG_INT = "i"
_TAG_FLOAT = "f"
_TAG_BOOL = "b"
_TAG_NULL = "~"
_LIST_OPEN = "["
_LIST_CLOSE = "]"
_OBJ_OPEN = "{"
_OBJ_CLOSE = "}"
_STRING_OPEN = '"'
_ESCAPE = "\\"
_PAIR_SEP = ":"
_ITEM_SEP = ","

_SPECIAL = frozenset('[]{}":,~ifb\\')

_ENCODE_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
}
_DECODE_ESCAPES = {
    "\\": "\\",
    '"': '"',
}


def _escape(text: str) -> str:
    return "".join(_ENCODE_ESCAPES.get(ch, ch) for ch in text)


def _encode_string(text: str) -> str:
    return _STRING_OPEN + _escape(text) + _STRING_OPEN


def _needs_tag(text: str) -> bool:
    if text == "":
        return True
    if text[0] in _SPECIAL:
        return True
    return any(ch in _SPECIAL for ch in text)


def _encode_value(value: object) -> str:
    if value is None:
        return _TAG_NULL
    if value is True:
        return _TAG_BOOL + "1;"
    if value is False:
        return _TAG_BOOL + "0;"
    if isinstance(value, bool):
        raise RoundtripError("unsupported boolean value")
    if isinstance(value, int):
        return _TAG_INT + repr(int(value)) + ";"
    if isinstance(value, float):
        return _TAG_FLOAT + repr(float(value)) + ";"
    if isinstance(value, str):
        if _needs_tag(value):
            return _encode_string(value)
        return value
    if isinstance(value, (list, tuple)):
        return (
            _LIST_OPEN
            + _ITEM_SEP.join(_encode_value(item) for item in value)
            + _LIST_CLOSE
        )
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise RoundtripError(
                    f"unsupported key type {type(key).__name__!r}; keys must be strings"
                )
            parts.append(_encode_string(key) + _PAIR_SEP + _encode_value(item))
        return _OBJ_OPEN + _ITEM_SEP.join(parts) + _OBJ_CLOSE
    raise RoundtripError(
        f"value of type {type(value).__name__!r} is not serializable"
    )


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, message: str) -> RoundtripError:
        return RoundtripError(
            f"{message} at position {self.pos} of {self.text!r}"
        )

    def expect(self, ch: str) -> None:
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise self.error(f"expected {ch!r}")
        self.pos += 1

    def parse_value(self, key_path: str | None = None) -> object:
        if self.pos >= len(self.text):
            raise self.error("unexpected end of input")
        ch = self.text[self.pos]
        if ch == _LIST_OPEN:
            return self.parse_list(key_path)
        if ch == _OBJ_OPEN:
            return self.parse_object(key_path)
        if ch == _STRING_OPEN:
            return self.parse_string()
        if ch == _TAG_NULL:
            self.pos += 1
            return None
        if ch == _TAG_BOOL:
            return self.parse_bool(key_path)
        if ch == _TAG_INT:
            return self.parse_number(int, key_path)
        if ch == _TAG_FLOAT:
            return self.parse_number(float, key_path)
        return self.parse_legacy()

    def parse_list(self, key_path: str | None) -> list:
        self.expect(_LIST_OPEN)
        items = []
        index = 0
        if self.pos < len(self.text) and self.text[self.pos] == _LIST_CLOSE:
            self.pos += 1
            return items
        while True:
            items.append(
                self.parse_value(
                    f"{key_path}[{index}]" if key_path else f"[{index}]"
                )
            )
            index += 1
            if self.pos >= len(self.text):
                raise self.error("unterminated list")
            ch = self.text[self.pos]
            if ch == _ITEM_SEP:
                self.pos += 1
                continue
            if ch == _LIST_CLOSE:
                self.pos += 1
                return items
            raise self.error("expected ',' or ']' in list")

    def parse_object(self, key_path: str | None) -> dict:
        self.expect(_OBJ_OPEN)
        result = {}
        if self.pos < len(self.text) and self.text[self.pos] == _OBJ_CLOSE:
            self.pos += 1
            return result
        last_key = None
        while True:
            if self.pos >= len(self.text):
                raise self.error(f"unterminated object after key {last_key!r}")
            if self.text[self.pos] == _STRING_OPEN:
                key = self.parse_string()
            else:
                key = self.parse_key()
            last_key = key
            self.expect(_PAIR_SEP)
            path = f"{key_path}.{key}" if key_path else key
            result[key] = self.parse_value(path)
            if self.pos >= len(self.text):
                raise self.error(f"unterminated object after key {last_key!r}")
            ch = self.text[self.pos]
            if ch == _ITEM_SEP:
                self.pos += 1
                if (
                    self.pos < len(self.text)
                    and self.text[self.pos] == _OBJ_CLOSE
                ):
                    raise self.error("trailing separator in object")
                continue
            if ch == _OBJ_CLOSE:
                self.pos += 1
                return result
            raise self.error("expected ',' or '}' in object")

    def parse_string(self) -> str:
        self.expect(_STRING_OPEN)
        chars = []
        while True:
            if self.pos >= len(self.text):
                raise self.error("unterminated string")
            ch = self.text[self.pos]
            if ch == _STRING_OPEN:
                self.pos += 1
                return "".join(chars)
            if ch == _ESCAPE:
                self.pos += 1
                if self.pos >= len(self.text):
                    raise self.error("dangling escape in string")
                escaped = self.text[self.pos]
                if escaped not in _DECODE_ESCAPES:
                    raise self.error(f"invalid escape sequence \\{escaped}")
                chars.append(_DECODE_ESCAPES[escaped])
                self.pos += 1
                continue
            chars.append(ch)
            self.pos += 1

    def parse_key(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != _PAIR_SEP:
            self.pos += 1
        if self.pos == start:
            raise self.error("empty object key")
        return self.text[start : self.pos]

    def _parse_tag_payload(self) -> str:
        end = self.pos
        while end < len(self.text) and self.text[end] not in ";,}]":
            end += 1
        if end >= len(self.text):
            if ";" not in self.text[self.pos :]:
                payload = self.text[self.pos :]
                self.pos = len(self.text)
                return payload
            raise self.error("unterminated type tag")
        payload = self.text[self.pos : end]
        if self.text[end] == ";":
            self.pos = end + 1
        else:
            self.pos = end
        return payload

    def parse_bool(self, key_path: str | None) -> bool:
        self.pos += 1
        payload = self._parse_tag_payload()
        if payload == "1":
            return True
        if payload == "0":
            return False
        raise self._type_error("bool", payload, key_path)

    def parse_number(self, kind, key_path: str | None):
        tag = self.text[self.pos]
        self.pos += 1
        payload = self._parse_tag_payload()
        expected = "int" if tag == _TAG_INT else "float"
        try:
            return kind(payload)
        except ValueError:
            raise self._type_error(expected, payload, key_path)

    def _type_error(
        self, expected: str, payload: str, key_path: str | None
    ) -> RoundtripError:
        where = f" for key {key_path!r}" if key_path else ""
        return RoundtripError(
            f"malformed {expected} value {payload!r}{where}: encoded "
            f"{expected}, read {payload!r}"
        )

    def parse_legacy(self) -> str:
        start = self.pos
        depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in (_LIST_OPEN, _OBJ_OPEN):
                depth += 1
            elif ch in (_LIST_CLOSE, _OBJ_CLOSE):
                if depth == 0:
                    break
                depth -= 1
            elif ch == _ITEM_SEP and depth == 0:
                break
            self.pos += 1
        return self.text[start : self.pos]


def dumps(value: object) -> str:
    return _encode_value(value)


def loads(text: str) -> object:
    if not isinstance(text, str):
        raise RoundtripError(f"loads expects str, got {type(text).__name__!r}")
    parser = _Parser(text)
    value = parser.parse_value()
    if parser.pos != len(text):
        raise parser.error("trailing content after value")
    return value


def dumpb(value: object, encoding: str = "utf-8") -> bytes:
    text = dumps(value)
    try:
        return text.encode(encoding)
    except (UnicodeEncodeError, LookupError) as exc:
        raise RoundtripError(
            f"cannot encode serialized value with {encoding!r}: {exc}"
        )


def loadb(data: bytes, encoding: str = "utf-8") -> object:
    if isinstance(data, str):
        text = data
    elif isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            raise RoundtripError(
                f"cannot decode {len(data)} bytes with {encoding!r}: {exc}"
            )
    else:
        raise RoundtripError(
            f"loadb expects bytes, got {type(data).__name__!r}"
        )
    return loads(text)
