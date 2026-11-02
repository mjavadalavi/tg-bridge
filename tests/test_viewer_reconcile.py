"""Viewer (B3), one reconcile round of view-sync against the windows tmux lists: the duplicate windows of a topic, the
windows of removed topics, renamed topics, and the window-size of each window (latest, set once per window).

The tmux runner is the ScriptedTmux fake of viewer_support, so no tmux server starts.
"""

import contextlib
import io
import json
import subprocess
from collections.abc import Sequence

from tests.fakes.clock import FakeClock
from tests.viewer_support import (
    DEFAULT_SIZE,
    OTHER_THREAD,
    THREAD,
    HomeCase,
    ScriptedTmux,
)
from tg_bridge.config import Config
from tg_bridge.daemon.viewer import (
    Viewer,
    run_sync,
)


class FailingWindowSizeTmux(ScriptedTmux):
    """A ScriptedTmux whose first `set-option ... window-size` exits 1, as tmux does when it refuses the option."""

    def __init__(self, existing: Sequence[tuple[str, str, str]] = ()) -> None:
        super().__init__(existing=existing)
        self.failures_left = 1

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if args[0] == "set-option" and "window-size" in args and self.failures_left:
            self.failures_left -= 1
            self.calls.append(list(args))
            return subprocess.CompletedProcess(args=["tmux"], returncode=1, stdout="", stderr="")
        return super().__call__(args, timeout)


class ReconcileTests(HomeCase):
    """One round of view-sync against the windows tmux lists: duplicates, removed and renamed topics, and the
    window-size of the windows."""

    def config_with(self, size: str) -> Config:
        return Config({"tui_size": size}, home=self.home, env={}, log=lambda _line: None)

    def test_two_windows_of_one_topic_keep_the_lowest_id_and_the_other_is_closed(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha"), (str(THREAD), DEFAULT_SIZE, "alpha")])
        self.assertEqual(Viewer(self.paths, self.config, run=tmux).sync_windows(), [])
        self.assertEqual(tmux.named("kill-window"), [["kill-window", "-t", "@2"]])
        self.assertEqual(tmux.named("new-window"), [])

    def test_a_window_of_a_removed_topic_is_closed_and_the_sync_window_is_not(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha"), (str(OTHER_THREAD), DEFAULT_SIZE, "beta")])
        self.assertEqual(Viewer(self.paths, self.config, run=tmux).sync_windows(), [])
        self.assertEqual(tmux.named("kill-window"), [["kill-window", "-t", "@2"]])

    def test_a_window_without_a_thread_tag_is_never_closed(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha"), ("", DEFAULT_SIZE, "mine")])
        Viewer(self.paths, self.config, run=tmux).sync_windows()
        self.assertEqual(tmux.named("kill-window"), [])

    def test_a_renamed_topic_gets_its_new_name_with_each_hash_doubled(self) -> None:
        self.write_sessions([(THREAD, "Bug #S fix")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha")])
        Viewer(self.paths, self.config, run=tmux).sync_windows()
        self.assertEqual(tmux.named("rename-window"), [["rename-window", "-t", "@1", "Bug-##S-fix"]])
        self.assertEqual(tmux.windows["@1"][2], "Bug-#S-fix")

    def test_a_window_already_named_with_the_seat_name_is_not_renamed(self) -> None:
        self.write_sessions([(THREAD, "Bug #S fix")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "Bug-#S-fix")])
        Viewer(self.paths, self.config, run=tmux).sync_windows()
        self.assertEqual(tmux.named("rename-window"), [])

    def test_an_unreadable_database_changes_no_topic_window_but_still_closes_duplicates(self) -> None:
        with open(self.paths.db, "wb") as handle:
            handle.write(b"this is not a sqlite database\n" * 4)
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha"), (str(THREAD), DEFAULT_SIZE, "alpha"),
                                      (str(OTHER_THREAD), DEFAULT_SIZE, "gone")])
        self.assertEqual(Viewer(self.paths, self.config, run=tmux).sync_windows(), [])
        self.assertEqual(tmux.named("kill-window"), [["kill-window", "-t", "@2"]])
        self.assertEqual(tmux.named("rename-window"), [])
        self.assertEqual(tmux.named("new-window"), [])

    def test_a_window_of_another_size_is_left_at_its_size_and_follows_the_terminal(self) -> None:
        self.write_sessions([(THREAD, "alpha"), (OTHER_THREAD, "beta")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "alpha"), (str(OTHER_THREAD), "100x30", "beta")])
        self.assertEqual(Viewer(self.paths, self.config_with("100x30"), run=tmux).sync_windows(), [])
        self.assertEqual(tmux.named("resize-window"), [])
        self.assertEqual(tmux.windows["@2"][1], "100x30")
        self.assertIn(["set-option", "-w", "-t", "@0", "window-size", "latest"], tmux.calls)
        self.assertIn(["set-option", "-w", "-t", "@2", "window-size", "latest"], tmux.calls)

    def test_an_old_window_is_set_to_latest_once_over_several_rounds_with_one_config_read(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux(existing=[(str(THREAD), "100x30", "alpha")])  # an older version's window: manual, fixed size
        clock = FakeClock()
        loads: list[int] = []

        def load() -> Config:
            loads.append(1)
            return self.config

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_sync(self.paths, load, clock, stop_after=3, run=tmux), 0)
        self.assertEqual(tmux.calls.count(["set-option", "-w", "-t", "@1", "window-size", "latest"]), 1)
        self.assertEqual(tmux.calls.count(["set-option", "-w", "-t", "@0", "window-size", "latest"]), 1)
        self.assertEqual(tmux.named("resize-window"), [])
        self.assertEqual(clock.sleeps, [5.0, 5.0, 5.0])
        self.assertEqual(len(loads), 1)

    def test_a_failed_window_size_is_retried_next_round_and_a_successful_one_is_not_repeated(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = FailingWindowSizeTmux(existing=[(str(THREAD), "100x30", "alpha")])  # the first window-size call fails
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_sync(self.paths, lambda: self.config, FakeClock(), stop_after=3, run=tmux), 0)
        self.assertEqual(tmux.failures_left, 0)
        self.assertEqual(tmux.calls.count(["set-option", "-w", "-t", "@0", "window-size", "latest"]), 2)  # fail, then ok
        self.assertEqual(tmux.calls.count(["set-option", "-w", "-t", "@1", "window-size", "latest"]), 1)
