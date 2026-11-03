"""The screen reading (agents/claude/tmux_pane.py) on real captures of the Claude Code 2.1 screen, taken from tmux panes.

The captures live in fixtures/tmux_screens/: the claude_* files are Claude Code screens (empty or typed input box), the
non_claude_* files are other screens of the bridge (its view window and session picker). Each expectation below is what
the CLI state shows on that screen: the input box empty or not, the CLI busy or not, and so idle (a message can be
pasted now)."""

import unittest
from pathlib import Path

from tg_bridge.agents.claude.tmux_pane import REASON_IDLE, _is_rule, idle, input_empty, screen_reason, working

SCREENS = Path(__file__).resolve().parent / "fixtures" / "tmux_screens"


def read_screen(name: str) -> str:
    return (SCREENS / name).read_text(encoding="utf-8")


# name -> (input box empty, working, idle)
EXPECTED = {
    "claude_input_empty_1.txt": (True, False, True),
    "claude_input_empty_2.txt": (True, False, True),
    "claude_input_typed_1.txt": (False, False, False),
    "claude_input_typed_2.txt": (False, False, False),
    "non_claude_0.txt": (False, False, False),
    "non_claude_4.txt": (False, False, False),
    "non_claude_5.txt": (False, False, False),
    "non_claude_9.txt": (False, False, False),
}


class ClaudeScreenCaptureTest(unittest.TestCase):
    def test_every_capture_has_its_expected_state(self) -> None:
        self.assertEqual(sorted(p.name for p in SCREENS.iterdir()), sorted(EXPECTED))
        for name, (empty, busy, both) in EXPECTED.items():
            screen = read_screen(name)
            with self.subTest(capture=name):
                self.assertIs(input_empty(screen), empty)
                self.assertIs(working(screen), busy)
                self.assertIs(idle(screen), both)

    def test_an_empty_input_box_on_a_real_screen_is_idle(self) -> None:
        for name in ("claude_input_empty_1.txt", "claude_input_empty_2.txt"):
            screen = read_screen(name)
            with self.subTest(capture=name):
                self.assertTrue(input_empty(screen))
                self.assertFalse(working(screen))
                self.assertTrue(idle(screen))

    def test_typed_text_in_the_input_box_on_a_real_screen_is_not_empty(self) -> None:
        for name in ("claude_input_typed_1.txt", "claude_input_typed_2.txt"):
            screen = read_screen(name)
            with self.subTest(capture=name):
                self.assertFalse(input_empty(screen))
                self.assertFalse(idle(screen))

    def test_a_spinner_line_above_the_top_rule_of_a_real_screen_is_busy(self) -> None:
        lines = read_screen("claude_input_empty_1.txt").splitlines()
        top = [index for index, line in enumerate(lines) if _is_rule(line)][-2]
        lines.insert(top, "✻ Cogitating… (12s · esc to interrupt)")  # inserted here in the test, not in the fixture
        busy = "\n".join(lines) + "\n"
        self.assertTrue(input_empty(busy))
        self.assertTrue(working(busy))
        self.assertFalse(idle(busy))

    def test_reason_idle_agrees_with_idle_on_every_capture(self) -> None:
        for path in sorted(SCREENS.iterdir()):
            screen = read_screen(path.name)
            with self.subTest(capture=path.name):
                self.assertEqual(screen_reason(screen) == REASON_IDLE, idle(screen))

    def test_the_labelled_top_rule_is_a_rule(self) -> None:
        self.assertTrue(_is_rule("─" * 20 + " tg:master-thisis ─"))
        self.assertTrue(_is_rule("─" * 8))
        self.assertTrue(_is_rule("   " + "─" * 40 + "  "))

    def test_a_short_run_or_the_prompt_line_is_not_a_rule(self) -> None:
        self.assertFalse(_is_rule("── 2026-11-01 17:43:33 · claude-work ──"))
        self.assertFalse(_is_rule("─" * 7))
        self.assertFalse(_is_rule("❯ " + "─" * 20))
        self.assertFalse(_is_rule(""))


if __name__ == "__main__":
    unittest.main()
