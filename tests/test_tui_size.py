"""The tui transport's terminal size (config key 'tui_size', as 'COLSxROWS'). Config.tui_size() gives (rows, cols):
a valid value is read with its sides in the right order, an absent key is the default with no line in the log, and any
other value gives the default plus one config line that names the key. parse_tui_size is the same rule on its own:
both sides from 1 to TUI_SIZE_MAX, digits only (no signs, no spaces, no other script's digits)."""

import unittest

from tg_bridge.config import DEFAULT_TUI_SIZE, Config, parse_tui_size

INVALID = ("", "200", "200 x 50", " 200x50", "x50", "200x", "0x50", "200x0", "1001x50", "200x1001", "200.5x50",
           "-1x5", "2e2x50", "٢٠٠x50", 200, ["200x50"])


class TuiSizeTests(unittest.TestCase):
    def config(self, **data: object) -> tuple[Config, list[str]]:
        lines: list[str] = []
        return Config(data, home="/home/tester", env={}, log=lines.append), lines

    def test_an_absent_key_is_the_default_and_logs_nothing(self) -> None:
        cfg, lines = self.config()
        self.assertEqual(cfg.tui_size(), (50, 200))
        self.assertEqual(DEFAULT_TUI_SIZE, (50, 200))
        self.assertEqual(lines, [])

    def test_cols_by_rows_is_read_back_as_rows_then_cols(self) -> None:
        cfg, lines = self.config(tui_size="120x40")
        self.assertEqual(cfg.tui_size(), (40, 120))
        self.assertEqual(lines, [])

    def test_the_largest_size_is_accepted(self) -> None:
        self.assertEqual(parse_tui_size("1000x1000"), (1000, 1000))
        self.assertEqual(parse_tui_size("1x1"), (1, 1))

    def test_an_invalid_value_is_the_default_with_one_warning_naming_the_key(self) -> None:
        for value in INVALID:
            with self.subTest(value=value):
                cfg, lines = self.config(tui_size=value)
                self.assertEqual(cfg.tui_size(), (50, 200))
                self.assertEqual(len(lines), 1)
                self.assertIn("tui_size", lines[0])

    def test_parse_tui_size_returns_none_for_each_invalid_form(self) -> None:
        for value in INVALID:
            with self.subTest(value=value):
                self.assertIsNone(parse_tui_size(value))


if __name__ == "__main__":
    unittest.main()
