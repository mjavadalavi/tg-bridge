"""TmuxPane (agents/claude/tmux_pane.py): the tmux calls with a fake runner, the screen reading on realistic captures, and one
real tmux on a private socket (the only test that starts a process; skipped when tmux is missing)."""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from tg_bridge.agents.claude.tmux_pane import (BUSY_HINT, CLEAR_KEY_GAP_SECONDS, CLEAR_ROUNDS_MAX, DIALOG_FOOTER, NBSP,
                                               PASTE_GAP_MAX_SECONDS, PASTE_GAP_SECONDS, REASON_BUSY, REASON_DIALOG,
                                               REASON_GONE, REASON_IDLE, REASON_MODE, REASON_NO_CLI, REASON_TYPED,
                                               TmuxPane, _faint_after, idle, input_empty, input_text, paste_gap,
                                               plain_screen, screen_reason, shows_cli, working)

RULE = "─" * 40

# Realistic captures of the Claude Code 2.1 screen (the input box is framed by two rules; the footer sits below it).
IDLE = f"""\
⏺ Done. The three steps are in the plan above.

{RULE}
❯ {NBSP}
{RULE}
  ? for shortcuts
"""

TYPED = f"""\
⏺ Done. The three steps are in the plan above.

{RULE}
❯ متن مراحل رو با شکل هماهنگ کن
{RULE}
  ? for shortcuts
"""

BUSY = f"""\
⏺ Reading the plan file
✻ Thinking… (12s · ↑ 320 tokens · esc to interrupt)

{RULE}
❯ {NBSP}
{RULE}
  ? for shortcuts
"""

DIALOG = """\
 Do you want to make this edit to tasks.py?
 ❯ 1. Yes
   2. Yes, allow all edits during this session (shift+tab)
   3. No, and tell Claude what to do differently (esc)

 Enter to confirm · Esc to cancel
"""


class RecordingClock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class FakeTmux:
    """Stands in for `tmux`: records each call (subcommand, argv, stdin) and answers with the pane id or a screen. The
    calls named in fail exit 1; raise_exc is raised instead of running."""

    def __init__(self, *, pane: str = "%3", screen: str = IDLE, fail: tuple[str, ...] = (),
                 raise_exc: BaseException | None = None, in_mode: bool = False) -> None:
        self.pane = pane
        self.in_mode = in_mode
        self.screen = screen
        self.fail = fail
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, list[str], str | None]] = []

    def __call__(self, args, timeout, text=None):
        args = list(args)
        if self.raise_exc is not None:
            raise self.raise_exc
        sub = args[2] if args[:1] == ["-S"] else args[0]
        self.calls.append((sub, args, text))
        if sub in self.fail:
            return subprocess.CompletedProcess(args, 1, "", "no such pane")
        if sub == "display-message" and "#{pane_in_mode}" in args:
            out = ("1" if self.in_mode else "0") + "\n"
        else:
            out = {"display-message": self.pane + "\n", "capture-pane": self.screen}.get(sub, "")
        return subprocess.CompletedProcess(args, 0, out, "")

    def subs(self) -> list[str]:
        return [sub for sub, _args, _text in self.calls]


def box_screen(lines: list[str]) -> str:
    """A CLI screen whose input box holds these lines: the first on the prompt line, the rest indented below it."""
    body = [f"❯ {lines[0]}" if lines else f"❯ {NBSP}", *(f"  {line}" for line in lines[1:])]
    return "\n".join(["⏺ Done.", "", RULE, *body, RULE, "  ? for shortcuts", ""])


class BoxTmux(FakeTmux):
    """A FakeTmux whose input box holds lines and reacts to the clear keys as the CLI's box does: C-u empties the line
    the cursor is on (the last one), BSpace on an empty line joins it to the line above (drops it)."""

    def __init__(self, lines: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.lines = list(lines)
        self.screen = box_screen(self.lines)

    def __call__(self, args, timeout, text=None):
        args = list(args)
        if args[-1:] == ["C-u"] and self.lines:
            self.lines[-1] = ""
        elif args[-1:] == ["BSpace"] and len(self.lines) > 1 and self.lines[-1] == "":
            self.lines.pop()
        self.screen = box_screen(self.lines)
        return super().__call__(args, timeout, text)


class DialogAfterClearTmux(FakeTmux):
    """A FakeTmux whose box turns into a selection dialog once the first C-u is sent."""

    def __call__(self, args, timeout, text=None):
        if list(args)[-1:] == ["C-u"]:
            self.screen = DIALOG
        return super().__call__(args, timeout, text)


BUSY_TYPED = BUSY.replace(f"❯ {NBSP}", "❯ typed")  # a spinner above a box that holds text


def keys(fake: FakeTmux) -> list[str]:
    """The keys sent with send-keys, in order."""
    return [args[-1] for sub, args, _text in fake.calls if sub == "send-keys"]


class TmuxPaneCallTests(unittest.TestCase):
    def test_capture_runs_capture_pane_on_the_socket_and_returns_the_screen(self) -> None:
        fake = FakeTmux()
        self.assertEqual(TmuxPane("/tmp/x/s", "%3", run=fake).capture(), IDLE)
        self.assertEqual(fake.calls[0][1], ["-S", "/tmp/x/s", "capture-pane", "-p", "-t", "%3"])

    def test_an_empty_socket_means_tmux_default_server(self) -> None:
        fake = FakeTmux()
        TmuxPane("", "%3", run=fake).capture()
        self.assertEqual(fake.calls[0][1], ["capture-pane", "-p", "-t", "%3"])

    def test_paste_loads_the_text_pastes_it_waits_then_presses_enter(self) -> None:
        fake, clock = FakeTmux(), RecordingClock()
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=clock).paste("hello\nworld"))
        self.assertEqual(fake.calls[1], ("load-buffer", ["load-buffer", "-b", "tgbridge-3", "-"], "hello\nworld"))
        self.assertEqual(fake.calls[2][1], ["paste-buffer", "-p", "-r", "-d", "-b", "tgbridge-3", "-t", "%3"])
        self.assertEqual(fake.calls[3][1], ["send-keys", "-t", "%3", "Enter"])
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], PASTE_GAP_SECONDS, places=2)

    def test_a_pane_in_copy_mode_is_left_before_the_paste(self) -> None:
        fake, clock = FakeTmux(in_mode=True), RecordingClock()
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=clock).paste("hello"))
        self.assertEqual(fake.calls[1][1], ["send-keys", "-t", "%3", "-X", "cancel"])
        self.assertEqual(fake.subs()[2:], ["load-buffer", "paste-buffer", "send-keys"])

    def test_a_failed_cancel_pastes_nothing(self) -> None:
        fake, clock = FakeTmux(in_mode=True, fail=("send-keys",)), RecordingClock()
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).paste("hello"))
        self.assertNotIn("load-buffer", fake.subs())

    def test_the_paste_gap_grows_with_the_text_and_is_capped(self) -> None:
        self.assertAlmostEqual(paste_gap("x" * 10), PASTE_GAP_SECONDS, places=2)
        self.assertAlmostEqual(paste_gap("x" * 20480), PASTE_GAP_SECONDS + 1.0, places=6)  # 10 chunks of 2 KB
        self.assertEqual(paste_gap("x" * 500_000), PASTE_GAP_MAX_SECONDS)
        self.assertAlmostEqual(paste_gap("ш" * 1024), PASTE_GAP_SECONDS + 0.1, places=6)  # 2 KB of UTF-8, two bytes a glyph

    def test_ready_needs_an_existing_pane_out_of_any_mode_with_an_idle_screen(self) -> None:
        self.assertTrue(TmuxPane("", "%3", run=FakeTmux()).ready())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(in_mode=True)).ready())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(screen=BUSY)).ready())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(pane="%4")).ready())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(fail=("display-message",))).ready())

    def test_reason_is_gone_for_a_missing_pane_and_mode_in_copy_mode(self) -> None:
        self.assertEqual(TmuxPane("", "%3", run=FakeTmux(pane="%4")).reason(), REASON_GONE)
        self.assertEqual(TmuxPane("", "%3", run=FakeTmux(fail=("display-message",))).reason(), REASON_GONE)
        self.assertEqual(TmuxPane("", "%3", run=FakeTmux(in_mode=True, screen=BUSY)).reason(), REASON_MODE)
        self.assertEqual(TmuxPane("", "%3", run=FakeTmux()).reason(), REASON_IDLE)

    def test_a_failed_paste_presses_no_enter(self) -> None:
        for failing in ("load-buffer", "paste-buffer"):
            with self.subTest(failing=failing):
                fake, clock = FakeTmux(fail=(failing,)), RecordingClock()
                self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).paste("hello"))
                self.assertNotIn("send-keys", fake.subs())
                self.assertEqual(clock.sleeps, [])

    def test_a_failed_enter_is_a_failed_paste(self) -> None:
        fake = FakeTmux(fail=("send-keys",))
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).paste("hello"))

    def test_escape_presses_escape_in_the_pane(self) -> None:
        fake = FakeTmux()
        self.assertTrue(TmuxPane("", "%3", run=fake).escape())
        self.assertEqual(fake.calls[0][1], ["send-keys", "-t", "%3", "Escape"])
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(fail=("send-keys",))).escape())

    def test_exists_is_true_only_when_tmux_shows_the_same_pane(self) -> None:
        self.assertTrue(TmuxPane("", "%3", run=FakeTmux(pane="%3")).exists())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(pane="%4")).exists())
        self.assertFalse(TmuxPane("", "%3", run=FakeTmux(fail=("display-message",))).exists())

    def test_a_tmux_that_cannot_run_is_an_answer_not_an_exception(self) -> None:
        for error in (FileNotFoundError("tmux"), subprocess.TimeoutExpired("tmux", 10)):
            with self.subTest(error=type(error).__name__):
                pane = TmuxPane("", "%3", run=FakeTmux(raise_exc=error), clock=RecordingClock())
                self.assertEqual(pane.capture(), "")
                self.assertFalse(pane.paste("hello"))
                self.assertFalse(pane.escape())
                self.assertFalse(pane.exists())

    def test_without_a_pane_id_nothing_runs(self) -> None:
        fake = FakeTmux()
        pane = TmuxPane("", "", run=fake)
        self.assertEqual((pane.capture(), pane.escape(), pane.exists()), ("", False, False))
        self.assertEqual(fake.calls, [])


class ScreenTests(unittest.TestCase):
    def test_an_empty_input_box_is_empty_and_idle(self) -> None:
        self.assertTrue(input_empty(IDLE))
        self.assertTrue(idle(IDLE))

    def test_typed_text_in_the_input_box_is_not_empty(self) -> None:
        self.assertFalse(input_empty(TYPED))
        self.assertFalse(idle(TYPED))
        self.assertFalse(working(TYPED))

    def test_a_second_typed_line_in_the_box_is_not_empty(self) -> None:
        screen = f"{RULE}\n❯ {NBSP}\n  second line\n{RULE}\n  ? for shortcuts\n"
        self.assertFalse(input_empty(screen))

    def test_a_busy_cli_is_not_idle_even_with_an_empty_box(self) -> None:
        self.assertTrue(input_empty(BUSY))
        self.assertTrue(working(BUSY))
        self.assertFalse(idle(BUSY))

    def test_a_selection_dialog_is_not_idle(self) -> None:
        self.assertTrue(working(DIALOG))
        self.assertFalse(idle(DIALOG))
        self.assertIn(DIALOG_FOOTER, DIALOG)
        self.assertIn(BUSY_HINT, BUSY)

    def test_padding_rows_below_the_box_do_not_hide_the_busy_hint(self) -> None:
        self.assertTrue(working(BUSY + "\n\n\n   \n"))

    def test_a_spinner_above_the_box_is_busy_even_with_a_statusline_below_it(self) -> None:
        screen = f"⏺ Working\n✻ Cogitating… (12s · esc to interrupt)\n{RULE}\n❯ {NBSP}\n{RULE}\n  model\n  ctx 3%\n  ? for shortcuts\n"
        self.assertTrue(working(screen))
        self.assertFalse(idle(screen))

    def test_the_spinner_is_looked_for_three_lines_above_the_top_rule_only(self) -> None:
        spinner = "✻ Thinking… (2s · esc to interrupt)"
        self.assertTrue(working(f"{spinner}\nl1\nl2\n{RULE}\n❯ {NBSP}\n{RULE}\n"))  # three lines above the top rule
        self.assertFalse(working(f"{spinner}\nl1\nl2\nl3\n{RULE}\n❯ {NBSP}\n{RULE}\n"))  # four lines above

    def test_shows_cli_is_the_box_or_a_working_state(self) -> None:
        self.assertTrue(shows_cli(IDLE))
        self.assertTrue(shows_cli(TYPED))
        self.assertTrue(shows_cli(DIALOG))
        self.assertFalse(shows_cli("user@host:~$ \n"))

    def test_a_busy_hint_scrolled_far_above_the_box_is_not_busy(self) -> None:
        old = "✻ Thinking… (2s · esc to interrupt)\n" + "old answer line\n" * 12
        self.assertFalse(working(old + IDLE))
        self.assertTrue(idle(old + IDLE))

    def test_no_input_box_is_not_an_idle_prompt(self) -> None:
        self.assertFalse(input_empty("user@host:~$ \n"))
        self.assertFalse(idle("user@host:~$ \n"))
        self.assertFalse(input_empty(""))

    def test_reason_of_each_screen(self) -> None:
        self.assertEqual(screen_reason(IDLE), REASON_IDLE)
        self.assertEqual(screen_reason(TYPED), REASON_TYPED)
        self.assertEqual(screen_reason(BUSY), REASON_BUSY)
        self.assertEqual(screen_reason(DIALOG), REASON_DIALOG)
        self.assertEqual(screen_reason("user@host:~$ \n"), REASON_NO_CLI)
        self.assertEqual(screen_reason(""), REASON_NO_CLI)

    def test_input_text_of_an_empty_box_is_empty(self) -> None:
        self.assertEqual(input_text(IDLE), "")
        self.assertEqual(input_text(BUSY), "")

    def test_input_text_of_a_one_line_box_is_that_line(self) -> None:
        self.assertEqual(input_text(TYPED), "متن مراحل رو با شکل هماهنگ کن")

    def test_input_text_of_a_multi_line_box_keeps_the_indented_lines_and_drops_blank_ones(self) -> None:
        screen = f"{RULE}\n❯ first line\n  {NBSP}\n    second line  \n  third\n{RULE}\n  ? for shortcuts\n"
        self.assertEqual(input_text(screen), "first line\nsecond line\nthird")

    def test_input_text_of_a_collapsed_paste_is_its_placeholder(self) -> None:
        screen = f"{RULE}\n❯ [Pasted text #2 +12 lines]\n{RULE}\n  ? for shortcuts\n"
        self.assertEqual(input_text(screen), "[Pasted text #2 +12 lines]")

    def test_input_text_without_a_box_is_empty(self) -> None:
        self.assertEqual(input_text("user@host:~$ \n"), "")
        self.assertEqual(input_text(""), "")

    def test_an_indented_glyph_row_inside_the_box_is_text_not_the_prompt(self) -> None:
        """A restored message whose second line starts with the glyph: the prompt is the row at column 0, the glyph row
        below it is text."""
        screen = f"{RULE}\n❯ first line\n  ❯ second line\n{RULE}\n  ? for shortcuts\n"
        self.assertEqual(input_text(screen), "first line\n❯ second line")
        self.assertFalse(input_empty(screen))

    def test_a_box_whose_prompt_row_is_indented_still_reads_its_text(self) -> None:
        screen = f"{RULE}\n  ❯ typed\n{RULE}\n  ? for shortcuts\n"
        self.assertEqual(input_text(screen), "typed")
        self.assertFalse(input_empty(screen))
        self.assertTrue(input_empty(f"{RULE}\n  ❯ {NBSP}\n{RULE}\n"))


class ClearInputTests(unittest.TestCase):
    def test_an_empty_box_is_cleared_at_once_with_no_key(self) -> None:
        fake, clock = FakeTmux(), RecordingClock()
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=clock).clear_input(5))
        self.assertEqual(keys(fake), [])
        self.assertEqual(clock.sleeps, [])

    def test_a_one_line_box_is_cleared_by_one_c_u(self) -> None:
        fake, clock = BoxTmux(["hello"]), RecordingClock()
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=clock).clear_input(3))
        self.assertEqual(keys(fake), ["C-u"])
        self.assertEqual(clock.sleeps, [CLEAR_KEY_GAP_SECONDS])

    def test_a_three_line_box_is_cleared_by_c_u_and_bspace_pairs(self) -> None:
        fake, clock = BoxTmux(["a", "b", "c"]), RecordingClock()
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=clock).clear_input(3))
        self.assertEqual(keys(fake), ["C-u", "BSpace", "C-u", "BSpace", "C-u"])
        self.assertEqual(fake.lines, [""])
        self.assertEqual(len(clock.sleeps), 5)

    def test_a_box_the_owner_does_not_own_gets_no_key(self) -> None:
        fake, clock = BoxTmux(["hello"]), RecordingClock()
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).clear_input(3, owned=lambda box: False))
        self.assertEqual(keys(fake), [])
        self.assertEqual(clock.sleeps, [])

    def test_owned_is_asked_about_the_box_before_each_key(self) -> None:
        """The owner's check runs on the screen read before every key: a box that turns foreign in the second round gets
        no third key."""
        fake, clock = BoxTmux(["a", "b", "c"]), RecordingClock()
        seen: list[str] = []

        def owned(box: str) -> bool:
            seen.append(box)
            return len(seen) <= 2
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).clear_input(3, owned=owned))
        self.assertEqual(keys(fake), ["C-u", "BSpace"])
        self.assertEqual(seen, ["a\nb\nc", "a\nb", "a\nb"])

    def test_a_pane_in_a_tmux_mode_is_not_cleared(self) -> None:
        fake = FakeTmux(in_mode=True, screen=TYPED)
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(keys(fake), [])

    def test_a_failed_key_stops_the_clear(self) -> None:
        fake = FakeTmux(screen=TYPED, fail=("send-keys",))
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(keys(fake), ["C-u"])

    def test_a_box_that_never_empties_stops_at_the_round_limit(self) -> None:
        for rounds, pairs in ((0, 1), (2, 2), (100, CLEAR_ROUNDS_MAX)):
            with self.subTest(rounds=rounds):
                fake, clock = FakeTmux(screen=TYPED), RecordingClock()
                self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).clear_input(rounds))
                self.assertEqual(keys(fake), ["C-u", "BSpace"] * pairs)
                self.assertEqual(clock.sleeps, [CLEAR_KEY_GAP_SECONDS] * (2 * pairs))

    def test_a_dialog_on_the_screen_sends_no_key_and_returns_false(self) -> None:
        fake = FakeTmux(screen=DIALOG)
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(keys(fake), [])

    def test_a_busy_spinner_sends_no_key_and_returns_false(self) -> None:
        fake = FakeTmux(screen=BUSY_TYPED)
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(keys(fake), [])

    def test_a_screen_without_an_input_box_sends_no_key_and_returns_false(self) -> None:
        fake = FakeTmux(screen="user@host:~$ typed\n")
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(keys(fake), [])

    def test_a_box_that_turns_into_a_dialog_mid_clear_stops_there(self) -> None:
        fake, clock = DialogAfterClearTmux(screen=TYPED), RecordingClock()
        self.assertFalse(TmuxPane("", "%3", run=fake, clock=clock).clear_input(3))
        self.assertEqual(keys(fake), ["C-u"])
        self.assertEqual(clock.sleeps, [CLEAR_KEY_GAP_SECONDS])

    def test_each_check_reads_the_screen_once(self) -> None:
        fake = BoxTmux(["a", "b", "c"])
        self.assertTrue(TmuxPane("", "%3", run=fake, clock=RecordingClock()).clear_input(3))
        self.assertEqual(fake.subs().count("capture-pane"), 6)  # the first read, then one after each of the 5 keys


# The input box as capture-pane -e shows it: the CLI's prompt suggestion is faint (SGR 2) text after the prompt.
SUGGESTED = (f"⏺ Done.\n{RULE}\n\x1b[39m❯{NBSP}\x1b[2mخب نتیجه کدکس چی شد؟\x1b[0m\n{RULE}\n"
             "\x1b[2m  ? for shortcuts\x1b[0m\n")
TYPED_COLOURED = f"⏺ Done.\n{RULE}\n\x1b[39m❯{NBSP}\x1b[38;5;16mمتن\x1b[39m\n{RULE}\n"
TYPED_AFTER_FAINT = f"⏺ Done.\n{RULE}\n❯ \x1b[2mghost\x1b[22m typed\n{RULE}\n"
TRUECOLOUR_TYPED = f"⏺ Done.\n{RULE}\n\x1b[38;2;153;153;153m❯\x1b[39m typed\n{RULE}\n"
TRUECOLOUR_PLACEHOLDER = f"{RULE}\n❯ \x1b[38;2;1;2;3m[Pasted text #2 +12 lines]\x1b[39m\n{RULE}\n  ? for shortcuts\n"


class FaintParameterTests(unittest.TestCase):
    def test_the_faint_state_after_each_sgr_parameter_list(self) -> None:
        cases = [
            ("2", False, True),
            ("", True, False),
            ("0", True, False),
            ("22", True, False),
            ("1;2", False, True),
            ("2;22", False, False),
            ("39", True, True),
            ("38;5;2", False, False),
            ("38;5;16;2", False, True),
            ("38;2;153;153;153", False, False),
            ("48;2;2;2;2", False, False),
            ("48;5;2", False, False),
            ("58;5;2", False, False),
            ("2;38;5;7;22", False, False),
            ("38;5", False, False),
            ("38:2::1:2:3", False, False),
            ("38:2::1:2:3", True, True),
        ]
        for params, faint, expected in cases:
            with self.subTest(params=params, faint=faint):
                self.assertEqual(_faint_after(params, faint), expected)


class PlainScreenTests(unittest.TestCase):
    def test_the_faint_suggestion_in_the_box_reads_as_empty(self) -> None:
        self.assertEqual(screen_reason(plain_screen(SUGGESTED)), REASON_IDLE)

    def test_coloured_typed_text_is_kept(self) -> None:
        self.assertEqual(screen_reason(plain_screen(TYPED_COLOURED)), REASON_TYPED)

    def test_text_after_the_faint_part_ends_is_kept(self) -> None:
        self.assertIn("❯  typed", plain_screen(TYPED_AFTER_FAINT))

    def test_faint_text_off_the_prompt_line_is_kept(self) -> None:
        self.assertIn("  ? for shortcuts", plain_screen(SUGGESTED).splitlines())

    def test_a_screen_without_escapes_is_unchanged(self) -> None:
        self.assertEqual(plain_screen(TYPED), TYPED)

    def test_truecolour_prompt_glyph_with_typed_text_keeps_the_text(self) -> None:
        screen = plain_screen(TRUECOLOUR_TYPED)
        self.assertEqual(screen_reason(screen), REASON_TYPED)
        self.assertEqual(input_text(screen), "typed")

    def test_truecolour_placeholder_in_the_box_is_kept(self) -> None:
        screen = plain_screen(TRUECOLOUR_PLACEHOLDER)
        self.assertFalse(input_empty(screen))
        self.assertEqual(input_text(screen), "[Pasted text #2 +12 lines]")

    def test_colour_parameters_that_are_not_faint_keep_the_text(self) -> None:
        for sgr in ("38;5;2", "48;2;2;2;2", "48;5;2", "58;5;2", "38:2::1:2:3"):
            with self.subTest(sgr=sgr):
                screen = f"{RULE}\n❯ \x1b[{sgr}mtyped\x1b[0m\n{RULE}\n"
                self.assertEqual(input_text(plain_screen(screen)), "typed")

    def test_a_colon_form_colour_changes_no_faint_state(self) -> None:
        self.assertEqual(plain_screen("❯ \x1b[2mghost\x1b[38:2::1:2:3m more\x1b[22m typed"), "❯  typed")

    def test_ready_and_reason_read_the_coloured_capture(self) -> None:
        fake = FakeTmux(screen=SUGGESTED)
        pane = TmuxPane("", "%3", run=fake)
        self.assertTrue(pane.ready())
        self.assertEqual(pane.reason(), REASON_IDLE)
        self.assertIn(["capture-pane", "-p", "-e", "-t", "%3"], [args for _sub, args, _text in fake.calls])


class RealTmuxTests(unittest.TestCase):
    """One real tmux on a private socket: the text pasted into `cat` shows in the capture."""

    @unittest.skipUnless(shutil.which("tmux"), "tmux is not installed")
    def test_paste_reaches_a_real_pane_and_capture_shows_it(self) -> None:
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("TMUX", None)  # never the caller's own server
        folder = tempfile.mkdtemp(prefix="tgbt-")
        self.addCleanup(shutil.rmtree, folder, True)
        socket = os.path.join(folder, "s")
        self.addCleanup(subprocess.run, ["tmux", "-S", socket, "kill-server"], capture_output=True, check=False)
        made = subprocess.run(["tmux", "-S", socket, "new-session", "-d", "-P", "-F", "#{pane_id}", "cat"],
                              capture_output=True, text=True, check=True, timeout=10)
        pane = TmuxPane(socket, made.stdout.strip())

        self.assertTrue(pane.exists())
        self.assertTrue(pane.paste("hello"))
        deadline = time.monotonic() + 10
        screen = pane.capture()
        while "hello" not in screen and time.monotonic() < deadline:
            time.sleep(0.1)
            screen = pane.capture()
        self.assertIn("hello", screen)

    @unittest.skipUnless(shutil.which("tmux"), "tmux is not installed")
    def test_paste_into_a_pane_in_copy_mode_reaches_the_app(self) -> None:
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("TMUX", None)
        folder = tempfile.mkdtemp(prefix="tgbt-")
        self.addCleanup(shutil.rmtree, folder, True)
        socket = os.path.join(folder, "s")
        self.addCleanup(subprocess.run, ["tmux", "-S", socket, "kill-server"], capture_output=True, check=False)
        # the app echoes GOT:<line> only when a whole line, Enter included, reached it; the typed text alone echoes too
        app = "sh -c 'read x; echo GOT:$x; sleep 30'"
        made = subprocess.run(["tmux", "-S", socket, "new-session", "-d", "-P", "-F", "#{pane_id}", app],
                              capture_output=True, text=True, check=True, timeout=10)
        pane_id = made.stdout.strip()
        pane = TmuxPane(socket, pane_id)
        subprocess.run(["tmux", "-S", socket, "copy-mode", "-t", pane_id], capture_output=True, check=True, timeout=10)

        self.assertTrue(pane.in_mode())
        self.assertFalse(pane.ready())
        self.assertTrue(pane.paste("hello"))
        self.assertFalse(pane.in_mode())
        deadline = time.monotonic() + 10
        screen = pane.capture()
        while "GOT:hello" not in screen and time.monotonic() < deadline:
            time.sleep(0.1)
            screen = pane.capture()
        self.assertIn("GOT:hello", screen)


if __name__ == "__main__":
    unittest.main()
