"""Inline keyboards: keyboard_json, picker_rows, EMPTY_KEYBOARD and the CallbackData codec with its 64-byte guard.

Port of keyboard_json and picker_rows of tg-bridge.py (C16/C19). Telegram rejects an inline button whose callback_data
is longer than 64 bytes (UTF-8), so every callback passes through check_callback_data() before it reaches a keyboard.
Pure functions and frozen values only: no I/O, no clock, no module state beyond the constants below.
"""

import json
from dataclasses import dataclass

CALLBACK_DATA_LIMIT = 64  # bytes, UTF-8: Telegram's limit for an inline button's callback_data
SEPARATOR = ":"
PICKER_PREFIX = "cm"  # picker callbacks are cm:<thread>:<kind>:<arg>
PICKER_COLUMNS = 2  # buttons per picker row
CHECK_MARK = "✓ "  # marks the option that is currently set

EMPTY_KEYBOARD = '{"inline_keyboard": []}'  # takes the inline keyboard off a finished message


class CallbackDataError(ValueError):
    """A callback_data value cannot be sent: too long for Telegram, or not decodable without ambiguity."""


def check_callback_data(data: str) -> str:
    """The same string when it fits Telegram's 64-byte limit, else CallbackDataError."""
    size = len(data.encode("utf-8"))
    if size > CALLBACK_DATA_LIMIT:
        raise CallbackDataError(f"callback_data is {size} bytes, the limit is {CALLBACK_DATA_LIMIT}: {data[:20]!r}...")
    return data


@dataclass(frozen=True)
class CallbackData:
    """A typed callback: a prefix that names the handler, then its arguments, joined with ':' on the wire.

    Only the last argument may contain ':' (a model name such as 'qwen3:8b'); the prefix and every other argument
    must not. decode() is unambiguous when given the field count (see decode), as the live parser is.
    Arguments may be ints; they are sent as their decimal text.
    """

    prefix: str
    args: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not self.prefix or SEPARATOR in self.prefix:
            raise CallbackDataError(f"invalid callback prefix: {self.prefix!r}")
        for arg in self.args[:-1]:
            if SEPARATOR in str(arg):
                raise CallbackDataError(f"callback argument may not contain {SEPARATOR!r}: {arg!r}")

    def encode(self) -> str:
        """The wire string, checked against the 64-byte limit."""
        return check_callback_data(SEPARATOR.join((self.prefix, *(str(arg) for arg in self.args))))

    @classmethod
    def decode(cls, data: str, fields: int | None = None) -> "CallbackData":
        """The typed callback for a wire string, as received in a callback_query.

        fields is the number of ':'-separated fields the wire string has, prefix included (the live picker parser
        uses 4 for cm:<thread>:<kind>:<arg>). The last field keeps any ':' it contains. Without fields every ':'
        splits, so a last argument that contains ':' only round-trips when fields is given.
        """
        head, *rest = data.split(SEPARATOR, fields - 1 if fields else -1)
        return cls(prefix=head, args=tuple(rest))


def _button(text: str, data: str | CallbackData) -> dict[str, str]:
    wire = data.encode() if isinstance(data, CallbackData) else check_callback_data(data)
    return {"text": text, "callback_data": wire}


def keyboard_json(rows: list[list[tuple[str, str | CallbackData]]]) -> str:
    """Rows of (text, callback_data) pairs -> the reply_markup JSON Telegram expects.

    callback_data is either a raw string or a CallbackData; both are checked against the 64-byte limit.
    """
    return json.dumps({"inline_keyboard": [[_button(text, data) for text, data in row] for row in rows]})


def picker_rows(
    thread: int | str, kind: str, options: list[tuple[str, str, int | str]], current: str | None
) -> list[list[tuple[str, CallbackData]]]:
    """Picker keyboard rows from (label, value, arg) options, PICKER_COLUMNS per row.

    The callback is cm:<thread>:<kind>:<arg>, where arg names the option itself (a model name, a stable path id, a mode
    index), never a position that another change could shift. The arg may be an int (mode index, path id) or a string
    (a model name, which may contain ':'). The option whose value is current gets a '✓ ' mark.
    """
    pairs = []
    for label, value, arg in options:
        data = CallbackData(PICKER_PREFIX, (str(thread), kind, arg))
        data.encode()  # the 64-byte guard runs now, so an oversized picker fails where it is built
        pairs.append(((CHECK_MARK if value == current else "") + label, data))
    return [pairs[i : i + PICKER_COLUMNS] for i in range(0, len(pairs), PICKER_COLUMNS)]
