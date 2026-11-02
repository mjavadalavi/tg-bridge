"""plain_line (feed/format.py): the readable terminal line of each shown event, and None for the events it hides."""

import unittest

from tg_bridge.agents.events import (
    ApprovalRequest,
    Decision,
    ProcessExited,
    SessionStarted,
    StatusEvent,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnResult,
    TurnStatus,
    UsageEvent,
)
from tg_bridge.feed.format import LINE_LABEL_CHARS, NO_DETAIL, plain_line

CWD = "/home/x/proj"


class TextAndToolTests(unittest.TestCase):
    def test_text_is_shown_stripped(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="  hello there \n"), CWD), "hello there")

    def test_blank_text_is_hidden(self) -> None:
        self.assertIsNone(plain_line(TextEvent(text="   \n"), CWD))

    def test_subagent_text_is_indented_under_its_parent(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="sub", parent_id="t1"), CWD), "  ↳ sub")

    def test_tool_start_shows_the_label_with_the_tool_mark(self) -> None:
        self.assertEqual(plain_line(ToolStart(tool_id="a", name="Bash", label="Bash: ls"), CWD), "🔧 Bash: ls")

    def test_tool_start_label_is_cut_to_the_line_limit(self) -> None:
        line = plain_line(ToolStart(tool_id="a", name="Bash", label="x" * 500), CWD)
        self.assertEqual(len(line), len("🔧 ") + LINE_LABEL_CHARS)
        self.assertTrue(line.endswith("…"))

    def test_subagent_tool_start_is_indented(self) -> None:
        line = plain_line(ToolStart(tool_id="a", name="Read", label="Read: a.py", parent_id="t1"), CWD)
        self.assertEqual(line, "  ↳ 🔧 Read: a.py")

    def test_successful_tool_end_is_hidden(self) -> None:
        self.assertIsNone(plain_line(ToolEnd(tool_id="a", ok=True), CWD))

    def test_failed_tool_end_shows_the_detail(self) -> None:
        self.assertEqual(plain_line(ToolEnd(tool_id="a", ok=False, detail="exit 2"), CWD), "❌ exit 2")

    def test_failed_tool_end_without_detail_says_so(self) -> None:
        self.assertEqual(plain_line(ToolEnd(tool_id="a", ok=False), CWD), f"❌ {NO_DETAIL}")


class TerminalJunkTests(unittest.TestCase):
    """Escape sequences and control characters in agent text or tool names never reach the feed's line."""

    def test_clear_screen_escape_is_removed_from_text(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="before\x1b[2Jafter"), CWD), "beforeafter")

    def test_cursor_home_escape_is_removed_from_a_tool_label(self) -> None:
        line = plain_line(ToolStart(tool_id="a", name="Bash", label="Bash: \x1b[H\x1b[31mls\x1b[0m"), CWD)
        self.assertEqual(line, "🔧 Bash: ls")

    def test_osc_52_clipboard_write_is_removed_when_bel_terminated(self) -> None:
        line = plain_line(TextEvent(text="a\x1b]52;c;SGVsbG8=\x07b"), CWD)
        self.assertEqual(line, "ab")

    def test_osc_terminated_by_esc_backslash_is_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="x\x1b]0;title\x1b\\y"), CWD), "xy")

    def test_bel_is_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="ding\x07dong"), CWD), "dingdong")

    def test_lone_escape_is_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="a\x1b\nb"), CWD), "a\nb")

    def test_single_character_escape_is_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="a\x1bcb"), CWD), "ab")

    def test_c1_control_csi_is_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="a\u009b2Jb"), CWD), "ab")

    def test_other_c1_controls_are_removed(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="a\u0085\u0090b"), CWD), "ab")

    def test_failed_tool_detail_and_turn_error_are_cleaned_too(self) -> None:
        self.assertEqual(plain_line(ToolEnd(tool_id="a", ok=False, detail="x\x1b[2J"), CWD), "❌ x")
        result = TurnResult(TurnStatus.ERROR, answer=None, error="bad\x1b[2J\x07")
        self.assertEqual(plain_line(result, CWD), "⚠ error: bad")

    def test_tab_and_newline_inside_text_are_kept(self) -> None:
        self.assertEqual(plain_line(TextEvent(text="a\tb\nc"), CWD), "a\tb\nc")

    def test_normal_text_emoji_and_persian_are_unchanged(self) -> None:
        text = "سلام، 👋 دنیا — the [bracket] \\ ok"
        self.assertEqual(plain_line(TextEvent(text=text), CWD), text)


class ResultTests(unittest.TestCase):
    def test_done_with_duration(self) -> None:
        result = TurnResult(TurnStatus.OK, answer="x", duration_ms=65_000)
        self.assertEqual(plain_line(result, CWD), "✅ done · 1:05")

    def test_done_without_duration_has_no_suffix(self) -> None:
        self.assertEqual(plain_line(TurnResult(TurnStatus.OK, answer="x"), CWD), "✅ done")

    def test_interrupted(self) -> None:
        self.assertEqual(plain_line(TurnResult(TurnStatus.INTERRUPTED, answer=None), CWD), "⏹ interrupted")

    def test_error_shows_the_error_text(self) -> None:
        result = TurnResult(TurnStatus.ERROR, answer=None, error="boom", duration_ms=3_000)
        self.assertEqual(plain_line(result, CWD), "⚠ error: boom · 0:03")

    def test_error_without_text_falls_back_to_the_subtype(self) -> None:
        result = TurnResult(TurnStatus.ERROR, answer=None, subtype="error_max_turns")
        self.assertEqual(plain_line(result, CWD), "⚠ error: error_max_turns")


class UsageStatusAndHiddenTests(unittest.TestCase):
    def test_usage_adds_cached_tokens_to_the_input_side(self) -> None:
        usage = UsageEvent(input_tokens=1200, output_tokens=300, cached_tokens=800)
        self.assertEqual(plain_line(usage, CWD), "🔢 tokens 2.0k in · 300 out")

    def test_status_uses_the_status_text(self) -> None:
        self.assertEqual(plain_line(StatusEvent(kind="compacting"), CWD), "🗜 compacting")
        self.assertEqual(plain_line(StatusEvent(kind="retry", attempt=2, max_attempts=5), CWD),
                         "🔁 API retry 2/5")

    def test_status_without_a_note_is_hidden(self) -> None:
        self.assertIsNone(plain_line(StatusEvent(kind="warning", detail="x"), CWD))

    def test_events_the_feed_does_not_show_are_hidden(self) -> None:
        hidden = (
            SessionStarted(session_id="s1"),
            ProcessExited(returncode=0),
            ApprovalRequest(approval_id="p", kind="tool", tool="Bash", input={}, summary="s",
                            choices=frozenset({Decision.ALLOW_ONCE})),
        )
        for event in hidden:
            with self.subTest(event=type(event).__name__):
                self.assertIsNone(plain_line(event, CWD))


if __name__ == "__main__":
    unittest.main()
