"""
Tagged round-trip serialization.

Objects become strings and strings become objects through the *same*
channel.  Every value carries a type tag, so the value that is read back
always has the same type -- and the same nested structure -- as the value
that was written:

``[i1,i2]``          list / array
``{key:value}``      dict / object
``"text"``           string (escaped, ASCII only)
``i42`` / ``d3.5``   number (int / float)
``b1`` / ``b0``      bool
``~``                ``None``

The four empty values therefore keep their own shape:

``[]`` empty list, ``{}`` empty dict, ``""`` empty string, ``~`` null.

Strings are escaped *before* being encoded to bytes (UTF-8 by default),
so marker characters can never be swallowed by a charset change, while
plain, untagged content keeps being accepted as a legacy string.  Tagged
and legacy content may be mixed inside one container.
"""

from __future__ import annotations

import math

from typing import Any


__all__ = ["RoundtripError", "dumpb", "dumps", "loadb", "loads"]


class RoundtripError(ValueError):
    """A value could not be encoded or decoded through the tagged channel."""


_TAG_INT = "i"
_TAG_FLOAT = "d"
_TAG_TRUE = "b1"
_TAG_FALSE = "b0"
_TAG_NULL = "~"

_NO_TAG = object()

_STRUCTURAL = set("{}[]:,")
_STOP_CHARS = "{}[]:,"

_SIMPLE_ESCAPES = {
    "\\": "\\",
    '"': '"',
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _where(path):
    return path or "<root>"


def _escape_string(text):
    out = ['"']
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char in _STRUCTURAL:
            out.append("\\x" + format(ord(char), "04x"))
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 32 or ord(char) > 126:
            out.append("\\u" + format(ord(char), "04x"))
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _encode_value(value: Any, path: str) -> str:
    scalar = _encode_scalar(value, path)
    if scalar is not None:
        return scalar

    if isinstance(value, (list, tuple)):
        items = ",".join(
            _encode_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
        return "[" + items + "]"

    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{_where(path)}: keys must be strings, got {type(key).__name__}"
                raise RoundtripError(msg)
            key_path = f"{path}.{key}" if path else key
            parts.append(_escape_string(key) + ":" + _encode_value(item, key_path))
        return "{" + ",".join(parts) + "}"

    msg = f"{_where(path)}: cannot encode {type(value).__name__}"
    raise RoundtripError(msg)


def _encode_scalar(value: Any, path: str) -> str | None:
    if value is None:
        return _TAG_NULL
    if isinstance(value, bool):
        return _TAG_TRUE if value else _TAG_FALSE
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, int):
        return _TAG_INT + repr(int(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            msg = f"{_where(path)}: {value!r} is not a finite number"
            raise RoundtripError(msg)
        return _TAG_FLOAT + repr(value)
    return None


def dumps(obj: Any) -> str:
    """
    Encode *obj* into a tagged string from which :func:`loads` restores
    the exact same types and nesting.
    """
    return _encode_value(obj, "")


def dumpb(obj: Any, encoding: str = "utf-8") -> bytes:
    """
    Encode *obj* and return bytes in *encoding*.

    Escaping happens before the charset change, so tags survive.
    """
    return dumps(obj).encode(encoding)


def loads(text: str) -> Any:
    """
    Decode *text* produced by :func:`dumps` back to the same value.

    Plain content without a tag is accepted as a legacy string.  The
    whole input must be consumed, so a tagged value followed by stray
    content is rejected rather than silently flattened.
    """
    if not isinstance(text, str):
        msg = f"loads expects str, got {type(text).__name__}"
        raise RoundtripError(msg)

    scanner = _Scanner(text)
    value = scanner.read_value("")
    if scanner.pos != scanner.length:
        scanner.error("trailing content after value", "")
    return value


def loadb(data: bytes | bytearray, encoding: str = "utf-8") -> Any:
    """Decode *data* (written by :func:`dumpb` with *encoding*) to a value."""
    if not isinstance(data, (bytes, bytearray)):
        msg = f"loadb expects bytes, got {type(data).__name__}"
        raise RoundtripError(msg)
    try:
        return loads(bytes(data).decode(encoding))
    except UnicodeError as exc:
        msg = f"cannot decode input as {encoding}: {exc}"
        raise RoundtripError(msg) from None


class _Scanner:
    def __init__(self, text):
        self.text = text
        self.length = len(text)
        self.pos = 0

    def error(self, message, path):
        msg = f"{_where(path)}: {message} at position {self.pos}"
        raise RoundtripError(msg)

    def expect(self, char, path):
        if self.pos >= self.length or self.text[self.pos] != char:
            found = self.text[self.pos] if self.pos < self.length else "<end>"
            msg = f"expected {char!r}, found {found!r}"
            self.error(msg, path)
        self.pos += 1

    def read_string(self, path):
        self.expect('"', path)
        out = []
        text = self.text
        while True:
            if self.pos >= self.length:
                self.error("unterminated string", path)
            char = text[self.pos]
            if char == '"':
                self.pos += 1
                return "".join(out)
            if char != "\\":
                out.append(char)
                self.pos += 1
                continue

            self.pos += 1
            if self.pos >= self.length:
                self.error("dangling escape", path)
            esc = text[self.pos]
            if esc in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[esc])
                self.pos += 1
            elif esc in ("x", "u"):
                start = self.pos + 1
                end = start + 4
                if end > self.length:
                    self.error("short unicode escape", path)
                digits = text[start:end]
                try:
                    out.append(chr(int(digits, 16)))
                except ValueError:
                    self.error(f"bad unicode escape {digits!r}", path)
                self.pos = end
            else:
                out.append(esc)
                self.pos += 1

    def read_list(self, path):
        self.expect("[", path)
        result = []
        if self.pos < self.length and self.text[self.pos] == "]":
            self.pos += 1
            return result
        while True:
            result.append(self.read_value(f"{path}[{len(result)}]"))
            if self.pos >= self.length:
                self.error("unterminated list", path)
            char = self.text[self.pos]
            self.pos += 1
            if char == "]":
                return result
            if char != ",":
                self.error(f"expected ',' or ']', found {char!r}", path)

    def read_dict(self, path):
        self.expect("{", path)
        result = {}
        if self.pos < self.length and self.text[self.pos] == "}":
            self.pos += 1
            return result
        while True:
            if self.pos < self.length and self.text[self.pos] == '"':
                key = self.read_string(f"{path}.<key>")
            else:
                key = self.read_legacy_token()
            self.expect(":", f"{path}.{key}")
            value_path = f"{path}.{key}" if path else key
            result[key] = self.read_value(value_path)
            if self.pos >= self.length:
                self.error("unterminated object", path)
            char = self.text[self.pos]
            self.pos += 1
            if char == "}":
                return result
            if char != ",":
                self.error(f"expected ',' or '}}', found {char!r}", path)

    def read_legacy_token(self):
        start = self.pos
        text = self.text
        while self.pos < self.length and text[self.pos] not in _STOP_CHARS:
            self.pos += 1
        return text[start:self.pos]

    def read_number(self, tag, path):
        start = self.pos
        text = self.text
        if self.pos < self.length and text[self.pos] == "-":
            self.pos += 1
        while self.pos < self.length and text[self.pos].isdigit():
            self.pos += 1
        if tag == _TAG_FLOAT and self.pos < self.length and text[self.pos] in ".eE":
            self.pos += 1
            if self.pos < self.length and text[self.pos] in "+-":
                self.pos += 1
            while self.pos < self.length and (
                text[self.pos].isdigit() or text[self.pos] in "eE"
            ):
                if text[self.pos] in "eE":
                    self.pos += 1
                    if self.pos < self.length and text[self.pos] in "+-":
                        self.pos += 1
                else:
                    self.pos += 1
        raw = text[start:self.pos]
        if not any(char.isdigit() for char in raw):
            self.error(f"malformed number {raw!r}", path)
        try:
            if tag == _TAG_INT:
                return int(raw)
            return float(raw)
        except ValueError:
            self.error(f"malformed number {raw!r}", path)

    def read_scalar_tag(self, path):
        text = self.text
        char = text[self.pos]
        if char == '"':
            return self.read_string(path)
        if char == _TAG_NULL:
            self.pos += 1
            return None
        if char == "b":
            if text.startswith(_TAG_TRUE, self.pos):
                self.pos += 2
                return True
            if text.startswith(_TAG_FALSE, self.pos):
                self.pos += 2
                return False
            self.error("bad bool tag", path)
        if char in (_TAG_INT, _TAG_FLOAT):
            self.pos += 1
            return self.read_number(char, path)
        return _NO_TAG

    def read_tagged(self, path):
        if self.pos >= self.length:
            self.error("missing value", path)
        char = self.text[self.pos]
        if char == "[":
            return self.read_list(path)
        if char == "{":
            return self.read_dict(path)
        return self.read_scalar_tag(path)

    def read_value(self, path):
        tagged = self.read_tagged(path)
        if tagged is not _NO_TAG:
            return tagged
        return self.read_legacy_token()
