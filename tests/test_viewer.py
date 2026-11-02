"""Viewer (B3): list_topics reads the sessions table read-only; Viewer makes the tg-bridge tmux session and one window per
topic (the reconcile rounds are in test_viewer_reconcile.py); run_sync adds windows on a loop; run_pane follows one topic's files (the feed of a print topic, the raw bytes of a
tui topic, the waiting line before any file, a RESET whenever a file is replaced, shrunk or switched) and keeps a terminal
quiet while it runs; attach replaces the process with tmux attach.

The tmux runner is a scripted fake, the clock is a FakeClock whose sleep can change the files between two polls, and the
paths live in a TempHome, so no tmux server starts and no time passes.
"""
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import termios
import unittest
from collections.abc import Sequence
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.viewer_support import (
    DEFAULT_SIZE,
    ENTRY,
    OTHER_THREAD,
    SYNC_CMD,
    THREAD,
    CrashClock,
    HomeCase,
    InterruptClock,
    ScriptedTmux,
    StepClock,
    pane_cmd,
)
from tg_bridge.agents.pty_mirror import RESET
from tg_bridge.config import Config
from tg_bridge.core.view_recorder import FEED_SUFFIX, META_SUFFIX
from tg_bridge.daemon import viewer as viewer_module
from tg_bridge.daemon.viewer import (
    PTY_SUFFIX,
    SESSION,
    SESSION_TARGET,
    SESSION_TARGET_COLON,
    THREAD_OPTION,
    WINDOW_FORMAT,
    Viewer,
    attach,
    list_topics,
    run_pane,
    run_sync,
)
from tg_bridge.daemon.view_menu import MENU_KEY


class ListTopicsTests(HomeCase):
    def test_named_topics_come_back_in_thread_order(self) -> None:
        self.write_sessions([(7, "beta"), (5, "alpha"), (9, None), (11, "")])
        self.assertEqual(list_topics(self.paths), [(5, "alpha"), (7, "beta")])

    def test_no_database_is_none_and_none_is_created(self) -> None:
        self.assertIsNone(list_topics(self.paths))
        self.assertFalse(os.path.exists(self.paths.db))

    def test_a_database_without_the_table_is_none(self) -> None:
        with contextlib.closing(sqlite3.connect(self.paths.db)) as db:
            db.execute("CREATE TABLE other(x INTEGER)")
            db.commit()
        self.assertIsNone(list_topics(self.paths))

    def test_a_table_without_a_named_topic_is_an_empty_list(self) -> None:
        self.write_sessions([(9, None), (11, "")])
        self.assertEqual(list_topics(self.paths), [])

    def test_a_file_that_is_not_a_database_is_none(self) -> None:
        with open(self.paths.db, "wb") as handle:
            handle.write(b"this is not a sqlite database\n" * 4)
        self.assertIsNone(list_topics(self.paths))


class ViewerSessionTests(HomeCase):
    def viewer(self, tmux: ScriptedTmux) -> Viewer:
        return Viewer(self.paths, self.config, run=tmux)

    def test_a_missing_session_is_made_with_its_sync_window_and_options(self) -> None:
        tmux = ScriptedTmux(has_session=1)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.calls, [
            ["has-session", "-t", SESSION_TARGET],
            ["new-session", "-d", "-s", SESSION, "-n", "sync", "-x", "200", "-y", "50",
             "-P", "-F", "#{window_id}", SYNC_CMD],
            ["set-option", "-w", "-t", "@0", "window-size", "latest"],
            ["set-option", "-t", SESSION_TARGET_COLON, "status-position", "bottom"],
            ["set-option", "-t", SESSION_TARGET_COLON, "mouse", "on"],
            ["list-keys", "-T", "root", MENU_KEY],
        ])

    def test_an_existing_session_keeps_its_windows_and_only_gets_the_mouse_on(self) -> None:
        tmux = ScriptedTmux(has_session=0)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.calls, [["has-session", "-t", SESSION_TARGET],
                                      ["set-option", "-t", SESSION_TARGET_COLON, "mouse", "on"],
                                      ["list-keys", "-T", "root", MENU_KEY]])

    def test_a_topic_without_a_window_gets_one_running_its_pane_command_and_a_thread_tag(self) -> None:
        self.write_sessions([(THREAD, "alpha"), (OTHER_THREAD, "budget.task")])
        tmux = ScriptedTmux(tagged=[THREAD])
        self.assertEqual(self.viewer(tmux).sync_windows(), ["budget-task"])
        self.assertEqual(tmux.named("new-window"), [
            ["new-window", "-d", "-P", "-F", "#{window_id}", "-t", SESSION_TARGET_COLON, "-n", "budget-task",
             pane_cmd(OTHER_THREAD)],
        ])
        self.assertIn(["set-option", "-w", "-t", "@2", THREAD_OPTION, str(OTHER_THREAD)], tmux.calls)

    def test_an_already_tagged_thread_is_not_added_again_but_its_window_gets_the_current_name(self) -> None:
        self.write_sessions([(THREAD, "renamed")])
        tmux = ScriptedTmux(tagged=[THREAD])
        self.assertEqual(self.viewer(tmux).sync_windows(), [])
        self.assertEqual(tmux.named("new-window"), [])
        self.assertEqual(tmux.named("rename-window"), [["rename-window", "-t", "@1", "renamed"]])

    def test_the_window_listing_asks_for_the_thread_tags_of_the_exact_session(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux()
        self.viewer(tmux).sync_windows()
        self.assertEqual(tmux.named("list-windows"), [["list-windows", "-t", SESSION_TARGET, "-F", WINDOW_FORMAT]])

    def test_a_listing_row_without_the_separator_changes_no_window(self) -> None:
        # a non-UTF-8 tmux server turns the tab of the old format into "_": no row parses, nothing is taken as "no window"
        self.write_sessions([(THREAD, "alpha"), (OTHER_THREAD, "beta")])
        tmux = ScriptedTmux(raw_listing="@0\t\t200x50\tsync\n@1\t5\t200x50\talpha\n@2_5_200x50_alpha\n")
        self.assertEqual(self.viewer(tmux).sync_windows(), [])
        self.assertEqual(tmux.named("new-window"), [])
        self.assertEqual(tmux.named("kill-window"), [])
        self.assertEqual(tmux.named("rename-window"), [])
        self.assertEqual(tmux.named("resize-window"), [])

    def test_a_window_name_with_the_separator_is_still_matched_by_its_tag(self) -> None:
        self.write_sessions([(THREAD, "a|b"), (OTHER_THREAD, "c")])
        tmux = ScriptedTmux(existing=[(str(THREAD), DEFAULT_SIZE, "a|b")])
        self.assertEqual(self.viewer(tmux).sync_windows(), ["c"])
        self.assertEqual(tmux.named("rename-window"), [])
        self.assertEqual(len(tmux.named("new-window")), 1)

    def test_two_topics_with_one_name_each_get_their_own_window(self) -> None:
        self.write_sessions([(THREAD, "same"), (OTHER_THREAD, "same")])
        tmux = ScriptedTmux()
        self.assertEqual(self.viewer(tmux).sync_windows(), ["same", "same"])
        self.assertEqual(len(tmux.named("new-window")), 2)

    def test_a_topic_name_with_a_hash_is_added_once_over_three_sync_rounds(self) -> None:
        self.write_sessions([(THREAD, "Bug #S fix")])
        tmux = ScriptedTmux()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_sync(self.paths, lambda: self.config, FakeClock(), stop_after=3, run=tmux)
        self.assertEqual(out.getvalue(), "view: window Bug-#S-fix added\n")
        [new_window] = tmux.named("new-window")
        self.assertEqual(new_window[new_window.index("-n") + 1], "Bug-##S-fix")  # tmux expands '#', so it is doubled

    def test_each_new_topic_window_follows_the_terminal_and_is_never_resized(self) -> None:
        self.write_sessions([(THREAD, "alpha"), (OTHER_THREAD, "beta")])
        tmux = ScriptedTmux()
        self.viewer(tmux).sync_windows()
        self.assertEqual(tmux.named("resize-window"), [])
        self.assertNotIn("manual", [arg for call in tmux.calls for arg in call])
        for window_id in ("@0", "@1", "@2"):  # the sync window and the two new topic windows, each set once
            self.assertEqual(tmux.calls.count(["set-option", "-w", "-t", window_id, "window-size", "latest"]), 1)

    def test_the_configured_tui_size_only_sizes_the_new_session(self) -> None:
        config = Config({"tui_size": "100x30"}, home=self.home, env={}, log=lambda _line: None)
        tmux = ScriptedTmux(has_session=1)
        Viewer(self.paths, config, run=tmux).ensure_session()
        [new_session] = tmux.named("new-session")
        at = new_session.index("-x")
        self.assertEqual(new_session[at:at + 4], ["-x", "100", "-y", "30"])
        self.assertEqual(tmux.named("resize-window"), [])

    def test_a_failed_window_listing_adds_nothing(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux(list_code=1)
        self.assertEqual(self.viewer(tmux).sync_windows(), [])
        self.assertEqual(tmux.named("new-window"), [])

    def test_a_tmux_that_cannot_run_is_not_an_exception(self) -> None:
        def missing(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("tmux")

        viewer = Viewer(self.paths, self.config, run=missing)
        viewer.ensure_session()
        self.assertEqual(viewer.sync_windows(), [])

    def test_the_sync_command_names_this_checkout_entry_script(self) -> None:
        tmux = ScriptedTmux(has_session=1)
        self.viewer(tmux).ensure_session()
        self.assertIn(ENTRY, tmux.named("new-session")[0][-1])


class RunSyncTests(HomeCase):
    def test_one_round_prints_the_window_it_added_and_sleeps_five_seconds(self) -> None:
        self.write_sessions([(THREAD, "alpha")])
        tmux = ScriptedTmux()
        clock = FakeClock()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(run_sync(self.paths, lambda: self.config, clock, stop_after=1, run=tmux), 0)
        self.assertEqual(out.getvalue(), "view: window alpha added\n")
        self.assertEqual(clock.sleeps, [5.0])


class PaneCase(HomeCase):
    def setUp(self) -> None:
        super().setUp()
        os.makedirs(self.paths.view, mode=0o700, exist_ok=True)
        self.write_sessions([(THREAD, "alpha")])
        self.stem = os.path.join(self.paths.view, str(THREAD))
        self.out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.pipe = io.StringIO()  # a stdin that is not a terminal: run_pane must leave it alone

    def write_meta(self, transport: str) -> None:
        meta = {"topic": "alpha", "transport": transport, "launcher": "claude", "state": "running", "updated": 1}
        with open(self.stem + META_SUFFIX, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    def write_file(self, suffix: str, data: bytes, mode: str = "wb") -> None:
        with open(self.stem + suffix, mode) as handle:
            handle.write(data)

    def replace_file(self, suffix: str, data: bytes) -> None:
        """Puts data in place the way PtyMirror and ViewRecorder do: a new file under a temp name, then os.replace."""
        temp = self.stem + suffix + ".new"
        with open(temp, "wb") as handle:
            handle.write(data)
        os.replace(temp, self.stem + suffix)

    def output(self) -> str:
        self.out.flush()
        return self.out.buffer.getvalue().decode("utf-8")


class RunPaneTests(PaneCase):
    def test_before_any_file_it_says_it_is_waiting_and_says_so_once(self) -> None:
        self.assertEqual(run_pane(self.paths, THREAD, self.out, FakeClock(), stop_after=3, stdin=self.pipe), 0)
        self.assertEqual(self.output(), "waiting for the first turn of alpha…\n")

    def test_a_print_topic_prints_only_whole_new_lines(self) -> None:
        self.write_meta("print")
        self.write_file(FEED_SUFFIX, b"one\n")
        seen: list[str] = []  # what the terminal shows after each poll

        def add_two_and_a_partial() -> None:
            seen.append(self.output())
            self.write_file(FEED_SUFFIX, b"two\npart", "ab")

        def finish_the_partial() -> None:
            seen.append(self.output())
            self.write_file(FEED_SUFFIX, b"ial\n", "ab")

        clock = StepClock([add_two_and_a_partial, finish_the_partial])
        run_pane(self.paths, THREAD, self.out, clock, stop_after=3, stdin=self.pipe)
        self.assertEqual(seen, ["one\n", "one\ntwo\n"])  # the partial line is held back until its newline
        self.assertEqual(self.output(), "one\ntwo\npartial\n")

    def test_a_tui_topic_copies_the_raw_bytes_and_resets_when_the_file_restarts(self) -> None:
        self.write_meta("tui")
        self.write_file(PTY_SUFFIX, (RESET + "HELLO").encode("utf-8"))
        clock = StepClock([
            lambda: self.write_file(PTY_SUFFIX, b" more", "ab"),
            lambda: self.write_file(PTY_SUFFIX, (RESET + "NEW").encode("utf-8")),  # shorter: the file restarted
        ])
        run_pane(self.paths, THREAD, self.out, clock, stop_after=3, stdin=self.pipe)
        self.assertEqual(self.output(), RESET + "HELLO" + " more" + RESET + RESET + "NEW")

    def test_ctrl_c_ends_the_follower_with_status_0(self) -> None:
        self.write_meta("print")
        self.assertEqual(run_pane(self.paths, THREAD, self.out, InterruptClock(), stop_after=10, stdin=self.pipe), 0)

    def test_a_restart_that_outgrows_the_old_offset_in_one_poll_gets_a_reset(self) -> None:
        self.write_meta("tui")
        self.write_file(PTY_SUFFIX, (RESET + "HELLO").encode("utf-8"))  # 7 bytes followed so far
        longer = (RESET + "A NEW RUN THAT IS LONGER THAN THE OLD FILE").encode("utf-8")
        clock = StepClock([lambda: self.replace_file(PTY_SUFFIX, longer)])
        run_pane(self.paths, THREAD, self.out, clock, stop_after=2, stdin=self.pipe)
        self.assertEqual(self.output(), RESET + "HELLO" + RESET + longer.decode("utf-8"))

    def test_a_feed_replaced_by_the_trim_gets_a_reset_and_no_duplicate_lines(self) -> None:
        self.write_meta("print")
        self.write_file(FEED_SUFFIX, b"a\nb\n")  # 4 bytes followed so far
        # the new file is longer than the old offset, so only its new inode tells the follower it was replaced
        clock = StepClock([lambda: self.replace_file(FEED_SUFFIX, b"c\nd\ne\n")])
        run_pane(self.paths, THREAD, self.out, clock, stop_after=2, stdin=self.pipe)
        self.assertEqual(self.output(), "a\nb\n" + RESET + "c\nd\ne\n")

    def test_a_switch_from_the_feed_to_the_raw_mirror_gets_a_reset_before_the_replay(self) -> None:
        self.write_meta("print")
        self.write_file(FEED_SUFFIX, b"one\n")

        def switch_to_tui() -> None:
            self.write_meta("tui")
            self.write_file(PTY_SUFFIX, (RESET + "RAW").encode("utf-8"))

        run_pane(self.paths, THREAD, self.out, StepClock([switch_to_tui]), stop_after=2, stdin=self.pipe)
        self.assertEqual(self.output(), "one\n" + RESET + RESET + "RAW")

    def test_a_switch_from_the_raw_mirror_to_the_feed_gets_a_reset_before_the_replay(self) -> None:
        self.write_meta("tui")
        self.write_file(PTY_SUFFIX, (RESET + "RAW").encode("utf-8"))

        def switch_to_print() -> None:
            self.write_meta("print")
            self.write_file(FEED_SUFFIX, b"plain\n")

        run_pane(self.paths, THREAD, self.out, StepClock([switch_to_print]), stop_after=2, stdin=self.pipe)
        self.assertEqual(self.output(), RESET + "RAW" + RESET + "plain\n")

    def test_the_inode_and_the_bytes_come_from_the_same_file_when_a_replace_lands_before_the_read(self) -> None:
        self.write_meta("print")
        self.write_file(FEED_SUFFIX, b"one\n")
        feed = self.stem + FEED_SUFFIX
        looks: list[str] = []

        def open_after_a_replace(path: str, *args: object, **kwargs: object) -> object:
            if path == feed:
                looks.append(path)
                if len(looks) == 2:  # between the first look and the second, the feed is replaced by a new file
                    self.replace_file(FEED_SUFFIX, b"abcd\nfresh-file-content\n")  # a newline at byte 4
            return open(path, *args, **kwargs)

        clock = StepClock([lambda: self.write_file(FEED_SUFFIX, b"two\n", "ab")])
        with mock.patch.object(viewer_module, "open", open_after_a_replace, create=True):
            run_pane(self.paths, THREAD, self.out, clock, stop_after=3, stdin=self.pipe)
        self.assertEqual(self.output(), "one\n" + RESET + "abcd\nfresh-file-content\n")

    def test_a_pipe_as_stdin_is_never_put_in_terminal_mode(self) -> None:
        self.write_meta("print")
        with mock.patch.object(termios, "tcsetattr") as set_attr, mock.patch.object(termios, "tcflush") as flush:
            run_pane(self.paths, THREAD, self.out, FakeClock(), stop_after=2, stdin=self.pipe)
        set_attr.assert_not_called()
        flush.assert_not_called()


class TerminalTests(PaneCase):
    """run_pane on a real terminal (os.openpty): echo and line editing are off while it runs, input is flushed each round,
    the attributes come back afterwards (also when the pane fails), and a terminal that fails mid-run ends nothing."""

    def open_terminal(self) -> tuple[int, io.TextIOWrapper]:
        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        stream = os.fdopen(slave, "r", encoding="utf-8")  # closes the slave fd
        self.addCleanup(stream.close)
        return slave, stream

    @staticmethod
    def echo_and_line_editing(fd: int) -> int:
        return termios.tcgetattr(fd)[3] & (termios.ECHO | termios.ICANON)

    def test_echo_and_line_editing_are_off_while_the_pane_runs_and_the_attributes_come_back(self) -> None:
        self.write_meta("print")
        fd, stream = self.open_terminal()
        saved = termios.tcgetattr(fd)
        self.assertNotEqual(saved[3] & termios.ECHO, 0)  # a new terminal echoes, so the check below means something
        during: list[int] = []
        clock = StepClock([lambda: during.append(self.echo_and_line_editing(fd)),
                           lambda: during.append(self.echo_and_line_editing(fd))])
        self.assertEqual(run_pane(self.paths, THREAD, self.out, clock, stop_after=2, stdin=stream), 0)
        self.assertEqual(during, [0, 0])
        self.assertEqual(termios.tcgetattr(fd), saved)

    def test_the_attributes_come_back_even_when_the_pane_fails(self) -> None:
        self.write_meta("print")
        fd, stream = self.open_terminal()
        saved = termios.tcgetattr(fd)
        with self.assertRaises(RuntimeError):
            run_pane(self.paths, THREAD, self.out, CrashClock(), stop_after=2, stdin=stream)
        self.assertEqual(termios.tcgetattr(fd), saved)

    def test_unread_input_is_flushed_once_per_round(self) -> None:
        self.write_meta("print")
        fd, stream = self.open_terminal()
        with mock.patch.object(termios, "tcflush", wraps=termios.tcflush) as flush:
            run_pane(self.paths, THREAD, self.out, FakeClock(), stop_after=3, stdin=stream)
        self.assertEqual(flush.call_args_list, [mock.call(fd, termios.TCIFLUSH)] * 3)

    def test_a_terminal_that_fails_during_a_flush_does_not_end_the_pane(self) -> None:
        self.write_meta("print")
        fd, stream = self.open_terminal()
        saved = termios.tcgetattr(fd)
        with mock.patch.object(termios, "tcflush", side_effect=termios.error(5, "Input/output error")):
            self.assertEqual(run_pane(self.paths, THREAD, self.out, FakeClock(), stop_after=2, stdin=stream), 0)
        self.assertEqual(termios.tcgetattr(fd), saved)


class AttachTests(unittest.TestCase):
    def run_attach(self, env: dict[str, str]) -> list[tuple[str, list[str]]]:
        calls: list[tuple[str, list[str]]] = []
        attach(lambda program, argv: calls.append((program, list(argv))), env=env)
        return calls

    def test_attach_replaces_the_process_with_tmux_attach_to_the_exact_session(self) -> None:
        self.assertEqual(self.run_attach({}), [("tmux", ["tmux", "attach", "-t", "=" + SESSION])])

    def test_inside_tmux_attach_switches_the_client_instead(self) -> None:
        self.assertEqual(self.run_attach({"TMUX": "/tmp/tmux-1000/default,123,0"}),
                         [("tmux", ["tmux", "switch-client", "-t", "=" + SESSION])])


if __name__ == "__main__":
    unittest.main()
