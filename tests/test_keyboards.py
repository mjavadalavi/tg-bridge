"""keyboards: port of the keyboard checks of fakecli/test_buttons.py (labels A, B, H, L, Q, R kept in docstrings).

The 64-byte guard is new in v3 and is tested with exact and multi-byte boundary values. Nothing here does I/O.
"""

import json
import unittest

from tg_bridge.telegram.keyboards import (
    CALLBACK_DATA_LIMIT,
    EMPTY_KEYBOARD,
    CallbackData,
    CallbackDataError,
    check_callback_data,
    keyboard_json,
    picker_rows,
)

MODELS = ("opus", "sonnet", "haiku", "fable", "default")
MODES = (("bypass", "bypassPermissions"), ("acceptEdits", "acceptEdits"), ("default", "default"), ("plan", "plan"))


def _buttons(markup: str) -> list[list[str]]:
    """The button labels of a reply_markup JSON string, row by row."""
    return [[b["text"] for b in row] for row in json.loads(markup)["inline_keyboard"]]


def _callbacks(markup: str) -> list[str]:
    return [b["callback_data"] for row in json.loads(markup)["inline_keyboard"] for b in row]


def _model_options() -> list[tuple[str, str, str]]:
    return [(m, m, m) for m in MODELS]


class EmptyKeyboardTests(unittest.TestCase):
    """The constant that takes a keyboard off a finished message."""

    def test_empty_keyboard_is_the_old_literal(self) -> None:
        self.assertEqual(EMPTY_KEYBOARD, '{"inline_keyboard": []}')

    def test_keyboard_json_of_no_rows_is_the_empty_keyboard(self) -> None:
        self.assertEqual(keyboard_json([]), EMPTY_KEYBOARD)


class KeyboardJsonTests(unittest.TestCase):
    """keyboard_json keeps the row layout and the text/callback pairs."""

    def test_row_layout_is_kept(self) -> None:
        """A: keyboard_json keeps the row layout."""
        rows = picker_rows(9301, "mo", _model_options(), "sonnet")
        markup = keyboard_json(rows)
        self.assertEqual(_buttons(markup), [["opus", "✓ sonnet"], ["haiku", "fable"], ["default"]])

    def test_raw_string_callbacks_are_sent_as_given(self) -> None:
        markup = keyboard_json([[("Allow", "pa:abc"), ("Deny", "pd:abc")]])
        self.assertEqual(json.loads(markup), {"inline_keyboard": [[
            {"text": "Allow", "callback_data": "pa:abc"},
            {"text": "Deny", "callback_data": "pd:abc"},
        ]]})

    def test_typed_callbacks_are_encoded(self) -> None:
        markup = keyboard_json([[("Stop", CallbackData("st", ("9304",)))]])
        self.assertEqual(_callbacks(markup), ["st:9304"])

    def test_output_is_compact_json(self) -> None:
        markup = keyboard_json([[("x", "y")]])
        self.assertEqual(markup, '{"inline_keyboard": [[{"text": "x", "callback_data": "y"}]]}')


class SixtyFourByteGuardTests(unittest.TestCase):
    """Telegram refuses callback_data over 64 bytes; the guard raises before anything is sent."""

    def test_limit_is_sixty_four_bytes(self) -> None:
        self.assertEqual(CALLBACK_DATA_LIMIT, 64)

    def test_exactly_sixty_four_bytes_is_accepted(self) -> None:
        data = "x" * 64
        self.assertEqual(check_callback_data(data), data)
        self.assertEqual(_callbacks(keyboard_json([[("ok", data)]])), [data])

    def test_sixty_five_bytes_raises_for_a_raw_string(self) -> None:
        """Old test_buttons A: a 65-byte callback is refused."""
        with self.assertRaises(CallbackDataError):
            keyboard_json([[("too long", "x" * 65)]])

    def test_sixty_five_bytes_raises_for_a_typed_callback(self) -> None:
        with self.assertRaises(CallbackDataError):
            CallbackData("cm", ("9301", "cw", "x" * 54)).encode()  # 11 + 54 = 65 bytes

    def test_guard_counts_utf8_bytes_not_characters(self) -> None:
        """'✓' is 3 bytes: 22 of them are 66 bytes even though they are only 22 characters."""
        data = "✓" * 22
        self.assertEqual(len(data), 22)
        with self.assertRaises(CallbackDataError):
            check_callback_data(data)

    def test_guard_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(CallbackDataError, ValueError))


class CallbackDataCodecTests(unittest.TestCase):
    """CallbackData encodes to prefix:arg:... and decodes back without ambiguity."""

    def test_encode_joins_prefix_and_arguments(self) -> None:
        self.assertEqual(CallbackData("cm", ("9301", "mo", "sonnet")).encode(), "cm:9301:mo:sonnet")

    def test_prefix_only_encodes_to_the_prefix(self) -> None:
        self.assertEqual(CallbackData("pa").encode(), "pa")

    def test_decode_inverts_encode(self) -> None:
        original = CallbackData("q", ("nonce1", "0", "2"))
        self.assertEqual(CallbackData.decode(original.encode()), original)

    def test_decode_splits_the_wire_string(self) -> None:
        self.assertEqual(
            CallbackData.decode("cm:9304:sn:0"),
            CallbackData("cm", ("9304", "sn", "0")),
        )

    def test_argument_with_separator_is_refused_unless_it_is_the_last(self) -> None:
        with self.assertRaises(CallbackDataError):
            CallbackData("cm", ("a:b", "c"))
        self.assertEqual(CallbackData("cm", ("c", "a:b")).encode(), "cm:c:a:b")

    def test_last_argument_may_contain_separator(self) -> None:
        """A model name such as qwen3:8b is a valid last argument (the live /model picker sends it)."""
        self.assertEqual(CallbackData("cm", ("9301", "mo", "qwen3:8b")).encode(), "cm:9301:mo:qwen3:8b")

    def test_decode_with_field_count_keeps_colons_in_last_field(self) -> None:
        """decode(fields=4) is the live parser's split(':', 3): the last field keeps its colon."""
        wire = CallbackData("cm", ("9301", "mo", "qwen3:8b")).encode()
        self.assertEqual(CallbackData.decode(wire, fields=4), CallbackData("cm", ("9301", "mo", "qwen3:8b")))

    def test_int_arguments_encode_as_decimal_text(self) -> None:
        """The live calling shape passes ints (mode index, cwd path id)."""
        self.assertEqual(CallbackData("cm", ("9304", "md", 0)).encode(), "cm:9304:md:0")
        self.assertEqual(CallbackData("cm", ("9304", "cw", 12)).encode(), "cm:9304:cw:12")

    def test_empty_or_separated_prefix_is_refused(self) -> None:
        for prefix in ("", "a:b"):
            with self.subTest(prefix=prefix), self.assertRaises(CallbackDataError):
                CallbackData(prefix)

    def test_instances_are_frozen_values(self) -> None:
        value = CallbackData("st", ("1",))
        self.assertEqual(value, CallbackData("st", ("1",)))
        with self.assertRaises(AttributeError):
            value.prefix = "other"  # type: ignore[misc]


class PickerRowsTests(unittest.TestCase):
    """picker_rows: two per row, a check mark on the current value, callback cm:<thread>:<kind>:<arg>."""

    def test_five_models_make_rows_of_two_two_and_one(self) -> None:
        """A: 5 models -> rows of 2,2,1."""
        rows = picker_rows(9301, "mo", _model_options(), "sonnet")
        self.assertEqual([len(r) for r in rows], [2, 2, 1])

    def test_check_mark_only_on_the_current_value(self) -> None:
        """A: ✓ marks only the current value."""
        rows = picker_rows(9301, "mo", _model_options(), "sonnet")
        labels = [label for row in rows for label, _ in row]
        self.assertEqual(labels, ["opus", "✓ sonnet", "haiku", "fable", "default"])

    def test_callback_is_thread_kind_and_option_argument(self) -> None:
        """A: callback is cm:<thread>:<kind>:<arg>."""
        rows = picker_rows(9301, "mo", _model_options(), "sonnet")
        callbacks = [data.encode() for row in rows for _, data in row]
        self.assertEqual(callbacks, [f"cm:9301:mo:{m}" for m in MODELS])

    def test_no_check_mark_when_nothing_matches(self) -> None:
        """The default model is not in the list, so no option is marked."""
        rows = picker_rows(9301, "mo", _model_options(), "gpt-x")
        labels = [label for row in rows for label, _ in row]
        self.assertFalse(any(label.startswith("✓") for label in labels))

    def test_mode_picker_marks_the_current_mode(self) -> None:
        """L: the mode picker marks the current mode."""
        options = [(label, value, index) for index, (label, value) in enumerate(MODES)]
        rows = picker_rows(9304, "md", [(label, value, str(arg)) for label, value, arg in options], "acceptEdits")
        self.assertEqual(_buttons(keyboard_json(rows)), [["bypass", "✓ acceptEdits"], ["default", "plan"]])

    def test_huge_thread_id_still_fits(self) -> None:
        """A: the picker stays within 64 bytes for an unusually large thread id."""
        rows = picker_rows(10**12 + 7, "md", [(label, value, str(i)) for i, (label, value) in enumerate(MODES)], "x")
        for row in rows:
            for _, data in row:
                self.assertLessEqual(len(data.encode()), 64)

    def test_long_argument_is_refused_by_the_guard(self) -> None:
        """The 64-byte guard reaches the picker: a 60-byte argument overflows the callback."""
        with self.assertRaises(CallbackDataError):
            picker_rows(9301, "rf", [("rule", "rule", "x" * 60)], "")

    def test_integer_args_are_sent_as_text(self) -> None:
        """The live mode and cwd pickers pass int args; they must not raise TypeError."""
        rows = picker_rows(9304, "md", [("bypass", "bypassPermissions", 0), ("plan", "plan", 3)], "plan")
        self.assertEqual(_buttons(keyboard_json(rows)), [["bypass", "✓ plan"]])
        self.assertEqual(_callbacks(keyboard_json(rows)), ["cm:9304:md:0", "cm:9304:md:3"])

    def test_model_name_with_colon_is_a_valid_picker_option(self) -> None:
        """Live shape: models ["opus", "qwen3:8b"] build the /model picker and the tap decodes like the live parser."""
        models = ["opus", "qwen3:8b"]
        rows = picker_rows(9301, "mo", [(m, m, m) for m in models], "qwen3:8b")
        wire = _callbacks(keyboard_json(rows))
        self.assertEqual(wire, ["cm:9301:mo:opus", "cm:9301:mo:qwen3:8b"])
        self.assertEqual(_buttons(keyboard_json(rows)), [["opus", "✓ qwen3:8b"]])
        tapped = CallbackData.decode(wire[1], fields=4)
        self.assertEqual(tapped.args, ("9301", "mo", "qwen3:8b"))

    def test_no_options_make_no_rows(self) -> None:
        self.assertEqual(picker_rows(9301, "mo", [], "sonnet"), [])

    def test_picker_output_is_a_valid_keyboard(self) -> None:
        """Picker rows go straight into keyboard_json and keep their order."""
        rows = picker_rows(9304, "rs", [("first", "a", "11"), ("second", "b", "12")], "")
        self.assertEqual(_callbacks(keyboard_json(rows)), ["cm:9304:rs:11", "cm:9304:rs:12"])


if __name__ == "__main__":
    unittest.main()
