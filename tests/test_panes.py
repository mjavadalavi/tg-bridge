"""panes: port of the tmux helpers of the live file (C21): panes_alive, type_to_pane and spawn_seat, with the
SPAWN_PREAMBLE and the prompt wait of the live file (tg-bridge.py:1992-2032).

Scenarios: the pane list is parsed and an unreachable tmux reads as no panes; typing sends the text flattened with
Enter after the 0.4 s gap, refuses empty text and dead panes, and reports a send that could not run; a seat opens in
the claude session's new window named after the task, waits for the prompt, then gets the preamble and the message; a
seat with no prompt in 30 s is reported and nothing is typed; a failed or missing tmux is an answer, never a raise.
The tmux runner and the clock are fakes, so no process starts and no time passes.
"""

import subprocess
import unittest
from collections.abc import Sequence

from tests.fakes.clock import FakeClock
from tg_bridge.daemon.panes import (
    NO_PROMPT_TEXT,
    SPAWN_CWD,
    SPAWN_PREAMBLE,
    Panes,
    at_prompt,
    seat_window_name,
)

LIST_OUTPUT = "%1\n%7\n"
NEW_PANE = "%9"
READY = "Claude Code\n❯ \n"
BUSY = "working\nesc to interrupt\n"
BOOTING = "loading the workspace\n"


class ScriptedTmux:
    """Records every tmux call. Replies come from `answers` by subcommand; `raises` makes a subcommand fail to run."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.answers: dict[str, subprocess.CompletedProcess[str]] = {
            "list-panes": _done(LIST_OUTPUT),
            "new-window": _done(NEW_PANE + "\n"),
            "send-keys": _done(""),
        }
        self.screens: list[str] = [READY]
        self.raises: dict[str, Exception] = {}

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(args), timeout))
        command = args[0]
        if command in self.raises:
            raise self.raises[command]
        if command == "capture-pane":
            screen = self.screens.pop(0) if len(self.screens) > 1 else self.screens[0]
            return _done(screen)
        return self.answers[command]

    def subcommands(self) -> list[str]:
        return [args[0] for args, _ in self.calls]

    def sent_keys(self) -> list[list[str]]:
        return [args for args, _ in self.calls if args[0] == "send-keys"]


def _done(stdout: str, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["tmux"], returncode=code, stdout=stdout, stderr=stderr)


class PanesCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tmux = ScriptedTmux()
        self.panes = Panes(clock=self.clock, run=self.tmux, spawn_cwd=SPAWN_CWD)


class ListAndTypeTests(PanesCase):
    def test_panes_alive_parses_the_pane_list(self) -> None:
        self.assertEqual(self.panes.panes_alive(), {"%1", "%7"})
        self.assertEqual(self.tmux.calls[0], (["list-panes", "-a", "-F", "#{pane_id}"], 10))

    def test_unreachable_tmux_reads_as_no_panes(self) -> None:
        self.tmux.raises["list-panes"] = FileNotFoundError("tmux")
        self.assertEqual(self.panes.panes_alive(), set())

    def test_empty_text_is_refused_without_calling_tmux(self) -> None:
        self.assertEqual(self.panes.type_to_pane("%7", "   "), (False, "empty text — nothing typed"))
        self.assertEqual(self.tmux.calls, [])

    def test_dead_pane_is_refused_and_nothing_is_typed(self) -> None:
        ok, info = self.panes.type_to_pane("%3", "ls")
        self.assertFalse(ok)
        self.assertIn("pane %3 is not alive", info)
        self.assertEqual(self.tmux.sent_keys(), [])

    def test_live_pane_gets_the_flattened_text_then_enter(self) -> None:
        ok, info = self.panes.type_to_pane("%7", "line one\nline two")
        self.assertEqual((ok, info), (True, "typed into pane %7"))
        self.assertEqual(self.tmux.sent_keys(), [
            ["send-keys", "-t", "%7", "-l", "--", "line one line two"],
            ["send-keys", "-t", "%7", "Enter"],
        ])
        self.assertEqual(self.clock.sleeps, [0.4])

    def test_a_send_that_cannot_run_is_a_failure(self) -> None:
        self.tmux.raises["send-keys"] = OSError("tmux gone")
        ok, info = self.panes.type_to_pane("%7", "ls")
        self.assertFalse(ok)
        self.assertIn("send-keys did not run", info)


class SpawnTests(PanesCase):
    def test_seat_opens_in_the_claude_session_then_gets_the_preamble_and_text(self) -> None:
        self.tmux.answers["list-panes"] = _done(LIST_OUTPUT + NEW_PANE + "\n")  # the new window is a live pane
        self.tmux.screens = [BOOTING, READY]
        ok, info, pane = self.panes.spawn_seat("/opt/budget-claude", "budget.mytask", "budget.mytask", "Tester", "fix it")
        self.assertEqual((ok, info, pane), (True, "typed into pane %9", NEW_PANE))
        new_window = next(args for args, _ in self.tmux.calls if args[0] == "new-window")
        self.assertEqual(new_window, ["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", "claude",
                                      "-n", "budget-mytask", "-c", SPAWN_CWD, "/opt/budget-claude"])
        typed = self.tmux.sent_keys()[0][-1]
        self.assertEqual(typed, SPAWN_PREAMBLE.format(topic="budget.mytask", sender="Tester") + "fix it")
        self.assertEqual(self.clock.sleeps, [1.5, 0.4])  # one poll while the window boots, then the gap before Enter

    def test_no_prompt_in_thirty_seconds_types_nothing(self) -> None:
        self.tmux.screens = [BOOTING]
        ok, info, pane = self.panes.spawn_seat("/opt/budget-claude", "budget.mytask", "budget.mytask", "Tester", "hi")
        self.assertEqual((ok, info, pane), (True, NO_PROMPT_TEXT.format(pane=NEW_PANE), NEW_PANE))
        self.assertEqual(self.tmux.sent_keys(), [])
        self.assertEqual(self.tmux.subcommands().count("capture-pane"), 20)  # 30 s polled every 1.5 s

    def test_refused_new_window_is_reported_with_its_stderr(self) -> None:
        self.tmux.answers["new-window"] = _done("", code=1, stderr="no server running\n")
        self.assertEqual(self.panes.spawn_seat("/opt/x", "a.b", "a.b", "T", "hi"),
                         (False, "spawn failed: no server running", None))

    def test_missing_tmux_is_reported_not_raised(self) -> None:
        self.tmux.raises["new-window"] = FileNotFoundError("tmux")
        self.assertEqual(self.panes.spawn_seat("/opt/x", "a.b", "a.b", "T", "hi"),
                         (False, "tmux did not run: tmux could not start", None))

    def test_spawn_cwd_is_the_constructor_value(self) -> None:
        self.assertEqual(self.panes.spawn_cwd, SPAWN_CWD)
        self.assertEqual(Panes(clock=self.clock, run=self.tmux, spawn_cwd="/w").spawn_cwd, "/w")


class HelperTests(unittest.TestCase):
    def test_prompt_detection_accepts_the_glyph_the_busy_hint_or_a_shell_prompt(self) -> None:
        self.assertTrue(at_prompt(READY))
        self.assertTrue(at_prompt(BUSY))
        self.assertTrue(at_prompt("user@host:~$\n"))
        self.assertFalse(at_prompt(BOOTING))

    def test_window_names_replace_dots_and_spaces_and_are_capped(self) -> None:
        self.assertEqual(seat_window_name("budget.my task"), "budget-my-task")
        self.assertEqual(seat_window_name(""), "seat")
        self.assertEqual(len(seat_window_name("x" * 80)), 40)


if __name__ == "__main__":
    unittest.main()
