"""Feed formatting: port of the footer and header checks of fakecli/test_feed.py (F) and test_tool_log.py (A, F).

Scenario labels are kept in the test docstrings. Retry and stream-cut header texts are new in v3 (API retry
for Claude, stream cut-off for Codex) and are covered in their own classes.
"""

import unittest

from tg_bridge.agents.events import StatusEvent, TurnResult, TurnStatus, UsageEvent
from tg_bridge.feed.format import (
    final_header,
    footer,
    fmt_k,
    mmss,
    rel,
    running_header,
    short,
    status_text,
)


def _ok(duration_ms: int | None = None, num_turns: int | None = None) -> TurnResult:
    return TurnResult(status=TurnStatus.OK, answer="done", duration_ms=duration_ms, num_turns=num_turns)


class NumberAndTimeTests(unittest.TestCase):
    """Helpers ported from the live _fmt_k and _mmss."""

    def test_fmt_k_keeps_small_numbers_and_shortens_thousands(self) -> None:
        """F) token counts: 0 and 999 as they are, 1000 and up with one decimal in k."""
        cases = {0: "0", 999: "999", 1000: "1.0k", 1200: "1.2k", 13400: "13.4k"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(fmt_k(value), expected)

    def test_mmss_pads_seconds(self) -> None:
        """A) running header time: 0:00, 1:05, 3:12."""
        self.assertEqual(mmss(0), "0:00")
        self.assertEqual(mmss(65.9), "1:05")
        self.assertEqual(mmss(192), "3:12")

    def test_short_cuts_with_an_ellipsis_at_exactly_n(self) -> None:
        """E) a long tool line is cut to exactly n characters, the last one an ellipsis."""
        self.assertEqual(short("abc", 5), "abc")
        self.assertEqual(short("abcde", 5), "abcde")
        cut = short("a" * 200, 120)
        self.assertEqual(len(cut), 120)
        self.assertTrue(cut.endswith("…"))

    def test_rel_inside_cwd_is_relative(self) -> None:
        self.assertEqual(rel("/home/x/proj/src/a.py", "/home/x/proj"), "src/a.py")

    def test_rel_outside_cwd_stays_absolute(self) -> None:
        self.assertEqual(rel("/home/x/other/a.py", "/home/x/proj"), "/home/x/other/a.py")
        self.assertEqual(rel("/home/x/proj2/a.py", "/home/x/proj"), "/home/x/proj2/a.py")

    def test_rel_relative_or_empty_inputs(self) -> None:
        self.assertEqual(rel("src/a.py", "/home/x/proj"), "src/a.py")
        self.assertEqual(rel("", "/home/x/proj"), "")
        self.assertEqual(rel("/home/x/proj/a.py", ""), "/home/x/proj/a.py")


class FooterTests(unittest.TestCase):
    """F) summary header: tool calls, duration, tokens in and out, cost, turns; '*' when the cost is unknown."""

    def test_footer_with_known_cost(self) -> None:
        """F) footer: 14 tools, 3m12s, cache counted into input, $0.10, 3 turns.

        Ported labels:
        F: footer ('✅ 14 tools · 3m12s · ↑13.4k ↓1.2k · $0.10 · 3 turns')
        """
        usage = UsageEvent(input_tokens=10000, cached_tokens=3400, output_tokens=1200, cost_usd=0.1,
                           cost_known=True)
        text = footer(_ok(duration_ms=192000, num_turns=3), usage, tools=14, wall=0)
        self.assertEqual(text, "✅ 14 tools · 3m12s · ↑13.4k ↓1.2k · $0.10 · 3 turns")

    def test_footer_marks_unknown_cost_with_a_star(self) -> None:
        """F) unknown cost marked: '$0.00*' when the backend gave a figure that is not exact."""
        usage = UsageEvent(cost_usd=0.0, cost_known=False)
        text = footer(_ok(duration_ms=4000, num_turns=0), usage, tools=0, wall=0)
        self.assertEqual(text, "✅ 0 tools · 4s · ↑0 ↓0 · $0.00* · 0 turns")

    def test_footer_shows_dash_when_cost_is_none(self) -> None:
        """F) cost '—' when the backend reports no cost at all (Codex)."""
        usage = UsageEvent(input_tokens=5, output_tokens=6, cost_usd=None, cost_known=False)
        text = footer(_ok(duration_ms=1000, num_turns=1), usage, tools=2, wall=0)
        self.assertEqual(text, "✅ 2 tools · 1s · ↑5 ↓6 · — · 1 turns")

    def test_footer_tokens_come_before_cost(self) -> None:
        usage = UsageEvent(input_tokens=1, output_tokens=2, cost_usd=0.5, cost_known=True)
        parts = footer(_ok(duration_ms=1000, num_turns=1), usage, tools=1, wall=0).split(" · ")
        self.assertTrue(parts[2].startswith("↑"))
        self.assertTrue(parts[3].startswith("$"))

    def test_footer_without_usage_shows_zero_tokens(self) -> None:
        text = footer(_ok(duration_ms=4000, num_turns=0), None, tools=0, wall=0)
        self.assertEqual(text, "✅ 0 tools · 4s · ↑0 ↓0 · — · 0 turns")

    def test_footer_uses_wall_time_when_duration_is_missing(self) -> None:
        text = footer(_ok(duration_ms=None, num_turns=1), None, tools=1, wall=12.7)
        self.assertTrue(text.split(" · ")[1] == "12s")

    def test_footer_has_five_fields_joined_by_separators(self) -> None:
        """A) the summary header has tools, time, tokens, cost and turns: four separators in all."""
        usage = UsageEvent(cost_usd=0.2, cost_known=True)
        text = footer(_ok(duration_ms=1000, num_turns=1), usage, tools=3, wall=0)
        self.assertTrue(text.startswith("✅ 3 tools · "))
        self.assertEqual(text.count(" · "), 4)

    def test_footer_minutes_only_past_sixty_seconds(self) -> None:
        usage = UsageEvent(cost_usd=0.0, cost_known=True)
        self.assertEqual(footer(_ok(duration_ms=59000, num_turns=1), usage, 0, 0).split(" · ")[1], "59s")
        self.assertEqual(footer(_ok(duration_ms=60000, num_turns=1), usage, 0, 0).split(" · ")[1], "1m00s")


class FinalHeaderTests(unittest.TestCase):
    """F) the last header of a turn: footer when it answered, else the reason the turn ended."""

    def test_ok_outcome_uses_the_footer(self) -> None:
        usage = UsageEvent(input_tokens=10000, cached_tokens=3400, output_tokens=1200, cost_usd=0.1,
                           cost_known=True)
        text = final_header("ok", result=_ok(duration_ms=192000, num_turns=3), usage=usage,
                            tools=14, wall=0)
        self.assertEqual(text, "✅ 14 tools · 3m12s · ↑13.4k ↓1.2k · $0.10 · 3 turns")

    def test_stopped_header_says_session_kept(self) -> None:
        text = final_header("stopped", result=None, usage=None, tools=3, wall=0)
        self.assertEqual(text, "⏹ stopped after 3 tools — session kept")

    def test_capped_header_names_the_limit(self) -> None:
        text = final_header("capped", result=None, usage=None, tools=2, wall=0, limit_hit=600)
        self.assertEqual(text, "⏱ stopped after 600s of work · 2 tools")

    def test_interrupted_header_names_the_restart(self) -> None:
        text = final_header("interrupted", result=None, usage=None, tools=1, wall=0)
        self.assertEqual(text, "⏹ interrupted by a bridge restart · 1 tools")

    def test_error_outcome_is_a_failed_turn(self) -> None:
        text = final_header("error", result=None, usage=None, tools=2, wall=0)
        self.assertEqual(text, "❌ turn failed · 2 tools")

    def test_ok_without_a_result_falls_back_to_failed(self) -> None:
        self.assertEqual(final_header("ok", result=None, usage=None, tools=0, wall=0),
                         "❌ turn failed · 0 tools")


class StatusTextTests(unittest.TestCase):
    """Header state texts: API retry n/m (Claude), stream cut-off (Codex), compacting."""

    def test_claude_api_retry_with_max(self) -> None:
        """A) an api_retry line: '🔁 API retry 2/10'."""
        ev = StatusEvent(kind="retry", attempt=2, max_attempts=10)
        self.assertEqual(status_text(ev), "🔁 API retry 2/10")

    def test_retry_without_max_has_no_slash(self) -> None:
        """Codex retry: the translator counts attempts, there is no maximum."""
        self.assertEqual(status_text(StatusEvent(kind="retry", attempt=1)), "🔁 API retry 1")

    def test_retry_without_attempt_shows_a_question_mark(self) -> None:
        self.assertEqual(status_text(StatusEvent(kind="retry")), "🔁 API retry ?")

    def test_stream_cut_text(self) -> None:
        """A) a Codex stream cut-off with willRetry: '📡 stream cut, retrying 1'."""
        self.assertEqual(status_text(StatusEvent(kind="stream_cut", attempt=1)), "📡 stream cut, retrying 1")

    def test_stream_cut_without_attempt(self) -> None:
        self.assertEqual(status_text(StatusEvent(kind="stream_cut")), "📡 stream cut, retrying ?")

    def test_compacting_text(self) -> None:
        self.assertEqual(status_text(StatusEvent(kind="compacting")), "🗜 compacting")

    def test_other_kinds_do_not_change_the_header(self) -> None:
        self.assertIsNone(status_text(StatusEvent(kind="warning", detail="x")))
        self.assertIsNone(status_text(StatusEvent(kind="subagent")))


class RunningHeaderTests(unittest.TestCase):
    """A) the running header: count, elapsed time, idle time past the ping threshold, state note."""

    def test_plain_running_header(self) -> None:
        text = running_header(2, 65, 0, ping_seconds=60, parked=False, state=None)
        self.assertEqual(text, "⚙️ working · 2 tools · 1:05")

    def test_idle_time_shows_past_the_ping_threshold(self) -> None:
        text = running_header(1, 300, 130, ping_seconds=60, parked=False, state=None)
        self.assertEqual(text, "⚙️ working · 1 tools · 5:00 · ⏳ idle 2 min")

    def test_idle_time_hidden_while_a_card_is_open(self) -> None:
        """Ported labels:
            C: no idle time in the log header while a card is open (0)"""
        text = running_header(1, 300, 130, ping_seconds=60, parked=True, state=None)
        self.assertNotIn("idle", text)

    def test_stream_cut_note_ends_the_header(self) -> None:
        """A) stream cut shown in the header for the rest of the turn."""
        state = status_text(StatusEvent(kind="stream_cut", attempt=2))
        text = running_header(3, 10, 0, ping_seconds=60, parked=False, state=state)
        self.assertEqual(text, "⚙️ working · 3 tools · 0:10 · 📡 stream cut, retrying 2")

    def test_api_retry_note_in_the_header(self) -> None:
        """A) '🔁 API retry 2/10' appears in the first line of the header."""
        state = status_text(StatusEvent(kind="retry", attempt=2, max_attempts=10))
        text = running_header(0, 5, 0, ping_seconds=60, parked=False, state=state)
        self.assertIn("🔁 API retry 2/10", text.split("\n")[0])


if __name__ == "__main__":
    unittest.main()
