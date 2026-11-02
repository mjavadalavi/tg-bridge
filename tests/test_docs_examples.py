"""The shipped examples and the README stay in step with the code: config.example.json loads through Config with the
teamclaude tui launcher, and the README documents the transport key and the tui limits the backend declares.

These are plain text checks. They read the files beside the package, so a renamed key or a dropped limit fails here.
"""

import json
import os
import unittest

from tg_bridge.config import TRANSPORT_PRINT, TRANSPORT_TUI, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "config.example.json")
README = os.path.join(ROOT, "README.md")
HOME = "/home/tester"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class ConfigExampleTests(unittest.TestCase):
    def test_example_is_valid_json_with_the_teamclaude_tui_launcher(self) -> None:
        data = json.loads(_read(EXAMPLE))
        self.assertEqual(data["launchers"]["teamclaude"], {
            "cmd": "teamclaude run",
            "config_dir": "~/.claude-team",
            "transport": "tui",
        })

    def test_example_loads_through_config_with_print_defaults_and_tui_for_teamclaude(self) -> None:
        cfg = Config(json.loads(_read(EXAMPLE)), home=HOME)
        specs = cfg.launcher_specs()
        self.assertEqual(specs["teamclaude"].transport, TRANSPORT_TUI)
        self.assertEqual(specs["teamclaude"].argv, ("teamclaude", "run"))
        self.assertEqual(specs["teamclaude"].config_dir, HOME + "/.claude-team")
        self.assertEqual(specs["claude"].transport, TRANSPORT_PRINT)
        self.assertEqual(specs["fast"].transport, TRANSPORT_PRINT)

    def test_example_tui_size_is_read_as_rows_then_cols_with_no_warning(self) -> None:
        lines: list[str] = []
        cfg = Config(json.loads(_read(EXAMPLE)), home=HOME, env={}, log=lines.append)
        self.assertEqual(cfg.tui_size(), (50, 200))
        self.assertEqual(lines, [])


class ReadmeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.readme = _read(README)

    def test_readme_documents_the_transport_key_and_both_values(self) -> None:
        self.assertIn("`transport`", self.readme)
        self.assertIn("`print`", self.readme)
        self.assertIn("`tui`", self.readme)
        self.assertIn("Non-interactive use isn't supported", self.readme)

    def test_readme_explains_how_the_tui_transport_works(self) -> None:
        self.assertIn("### The `tui` transport", self.readme)
        self.assertIn("`/exit`", self.readme)
        self.assertIn("session transcript", self.readme)
        self.assertIn("`tg-bridge hook permission`", self.readme)
        self.assertIn("Unix socket", self.readme)
        self.assertIn("folder-trust dialog", self.readme)
        self.assertIn("Stop sends Esc", self.readme)

    def test_readme_lists_the_tui_limits(self) -> None:
        for phrase in ("`/mode` and `/model` apply from the next message", "No cost in the footer",
                       "`AskUserQuestion` is disabled", "Tool calls made by subagents are not in the live feed",
                       "`TG_TITLE_LAUNCHER`", "sessions run at once"):
            self.assertIn(phrase, self.readme)

    def test_config_reference_row_mentions_transport(self) -> None:
        row = next(line for line in self.readme.splitlines() if line.startswith("| `launchers` / `agent_launchers`"))
        self.assertIn("`transport`", row)

    def test_readme_documents_sessions_in_the_terminal_with_its_keys(self) -> None:
        self.assertIn("## Sessions in the terminal: `tg-bridge view`", self.readme)
        self.assertNotIn("## Watching sessions", self.readme)
        self.assertIn("tg-bridge view", self.readme)
        self.assertIn("tui_size", self.readme)
        self.assertIn("click a tab with the mouse", self.readme)
        for key in ("`Ctrl-b n`", "`Ctrl-b p`", "`Ctrl-b w`", "`Ctrl-b d`"):
            self.assertIn(key, self.readme)

    def test_readme_explains_the_terminal_session_handoff(self) -> None:
        for phrase in ("real Claude Code session", "Exit Claude Code to give the session back to Telegram",
                       "Telegram messages for that topic wait", "busy screen", "`<thread>.term`"):
            self.assertIn(phrase, self.readme)

    def test_readme_lists_only_the_view_states_the_code_writes(self) -> None:
        self.assertIn("(`running`, `idle` or `lingering`)", self.readme)


if __name__ == "__main__":
    unittest.main()
