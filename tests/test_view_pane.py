"""The view pane (daemon/view_pane.py): the three screens of `tg-bridge view-pane <thread>`.

ready shows the topic's header and the last lines of its feed; Enter claims the topic and runs the topic's own launcher on
the pane's tty, with --resume, --model and the permission flags of its mode, in the topic's folder. The claim exists while
the CLI runs and is gone after it. busy (meta state running or lingering) shows the mirror with a top line and refuses
Enter. Ctrl+C and q do nothing on ready. The CLI is tests/fakes/fake_view_launcher.py: a stdlib script that records its
argv and cwd and exits, so no claude process starts. Every launcher in these tests is a full command that names the fake,
never a bare name, so the real claude-work or claude on this machine is never run. Keys come from a scripted source, or from a
real pseudo-terminal (os.openpty) in the terminal case. The mouse setting uses the scripted tmux of viewer_support.
"""

import io
import json
import os
import shlex
import sqlite3
import sys
import termios
from collections.abc import Callable, Mapping, Sequence
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.viewer_support import HomeCase, ScriptedTmux
from tg_bridge.config import Config
from tg_bridge.core import terminal_claim
from tg_bridge.core.turn_text import TIMEZONE
from tg_bridge.core.view_recorder import FEED_SUFFIX, META_SUFFIX
from tg_bridge.daemon.code_watch import RESTART_STATUS
from tg_bridge.daemon.view_launcher import CHOICE_HINT_LINE, switch_launcher
from tg_bridge.daemon.view_pane import (
    BUSY_LINE,
    BUSY_REFUSED_LINE,
    CLEAR_SCREEN,
    READY_HINT,
    SWITCH_FAILED_LINE,
    KeySource,
    ViewPane,
    _run_child,
    _TerminalKeys,
    child_env,
    read_topic,
    run_topic_pane,
)
from tg_bridge.daemon.viewer import SESSION_TARGET_COLON, Viewer
from tg_bridge.store.db import Database
from tg_bridge.store.sessions import SessionRepo

THREAD = 42
UUID = "0f3c9a7e-1b2d-4c5e-8f60-7a8b9c0d1e2f"
FRESH_UUID = "6b1e2f3a-9c4d-4e5f-8a7b-0c1d2e3f4a5b"  # the session id a topic without one starts with (injected)
FAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tests", "fakes",
                    "fake_view_launcher.py")
ENTER = b"\r"
CTRL_C = b"\x03"
KINDS = ("claude-work", "claude-alt", "claude")
LOCAL_MODES = termios.ECHO | termios.ICANON | termios.ISIG  # the local flags the raw mode turns off


def fake_command(kind: str) -> str:
    """The launcher command of one kind: this interpreter, -I, the fake script and the kind as its first argument."""
    return shlex.join([sys.executable, "-I", FAKE, kind])


class ScriptedKeys:
    """One key per round, in order; b"" once they run out. Nothing waits."""

    def __init__(self, keys: list[bytes]) -> None:
        self._keys = list(keys)

    def read(self, seconds: float) -> bytes:
        return self._keys.pop(0) if self._keys else b""


class PaneCase(HomeCase):
    """A topic with a sessions row and a view folder, the fake launchers in the config, and the environment the fake CLI
    reads: where it records its run and which claim file it looks for."""

    def setUp(self) -> None:
        super().setUp()
        self.folder = os.path.join(self.home, "work")
        os.makedirs(self.folder)
        self.view = self.paths.view
        os.makedirs(self.view, mode=0o700)
        self.record = os.path.join(self.home, "record.jsonl")
        self.claim = terminal_claim.claim_path(self.view, THREAD)
        self.topic_launcher = ""
        env = mock.patch.dict(os.environ, {"FAKE_VIEW_OUT": self.record, "FAKE_VIEW_CLAIM": self.claim})
        env.start()
        self.addCleanup(env.stop)
        self.launchers = {kind: {"cmd": fake_command(kind), "config_dir": "~/.claude"} for kind in KINDS}

    def make_topic(self, *, launcher: str | None = None, pick: str = "", uuid: str = UUID,
                   mode: str = "bypassPermissions", model: str = "", feed: int = 3, name: str = "alpha") -> None:
        """Writes the topic's sessions row through the real store (replacing any earlier row of the thread) and its feed."""
        db = Database(self.paths)
        self.addCleanup(db.close)
        db.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        repo = SessionRepo(db, FakeClock())
        stored = fake_command("claude-work") if launcher is None else launcher
        repo.ensure(THREAD, name, stored, self.folder, mode)
        if uuid:
            repo.set_uuid(THREAD, uuid)
        if model:
            repo.set_model(THREAD, model)
        if pick:
            repo.set_launcher_pick(THREAD, pick)
        with open(os.path.join(self.view, f"{THREAD}{FEED_SUFFIX}"), "w", encoding="utf-8") as handle:
            handle.write("".join(f"feed line {number}\n" for number in range(1, feed + 1)))

    def write_meta(self, state: str) -> None:
        with open(os.path.join(self.view, f"{THREAD}{META_SUFFIX}"), "w", encoding="utf-8") as handle:
            json.dump({"state": state, "transport": "tui"}, handle)

    def make_config(self) -> Config:
        topics = {"alpha": {"launcher": self.topic_launcher}} if self.topic_launcher else {}
        return Config({"launchers": self.launchers, "topics": topics}, home=self.home, env={}, log=lambda _line: None)

    def pane(self, keys: list[bytes], rounds: int, switch: Callable[[int, str, str], str] | None = None) -> io.StringIO:
        out = io.StringIO()
        self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=out, clock=FakeClock(),
                                        stop_after=rounds, stdin=io.StringIO(), keys=ScriptedKeys(keys),
                                        switch=switch, new_uuid=lambda: FRESH_UUID), 0)
        return out

    def children(self) -> list[dict]:
        if not os.path.exists(self.record):
            return []
        with open(self.record, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def session_start_hook(argv: Sequence[str]) -> dict:
    """The settings the CLI argv carries in its last two arguments: --settings and the JSON of the SessionStart hook."""
    return json.loads(argv[-1])


class ReadyTests(PaneCase):
    def test_enter_runs_the_launcher_with_resume_model_and_the_mode_flags_in_the_folder(self) -> None:
        self.make_topic(model="opus", mode="acceptEdits")
        self.pane([ENTER], rounds=1)
        [child] = self.children()
        self.assertEqual(child["argv"][:-2], ["claude-work", "--resume", UUID, "--model", "opus", "--permission-mode",
                                              "acceptEdits", "--allow-dangerously-skip-permissions"])
        self.assertEqual(child["argv"][-2], "--settings")
        command = session_start_hook(child["argv"])["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        self.assertEqual(shlex.split(command)[-4:], ["hook", "session-start", str(THREAD), str(os.getpid())])
        self.assertEqual(os.path.realpath(child["cwd"]), os.path.realpath(self.folder))

    def test_each_launcher_kind_and_mode_has_its_own_argv_and_session_id_without_a_uuid(self) -> None:
        for kind in KINDS:
            for mode in ("bypassPermissions", "plan"):
                with self.subTest(kind=kind, mode=mode):
                    if os.path.exists(self.record):
                        os.remove(self.record)
                    self.make_topic(launcher=fake_command(kind), uuid="", mode=mode)
                    self.pane([ENTER], rounds=1)
                    [child] = self.children()
                    self.assertEqual(child["argv"][:-2], [kind, "--session-id", FRESH_UUID, "--permission-mode", mode,
                                                          "--allow-dangerously-skip-permissions"])
                    self.assertEqual(child["argv"][-2], "--settings")

    def test_the_cli_gets_the_session_start_hook_and_no_other_settings(self) -> None:
        self.make_topic()
        self.pane([ENTER], rounds=1)
        [child] = self.children()
        settings = session_start_hook(child["argv"])
        self.assertEqual(list(settings), ["hooks"])
        self.assertEqual(list(settings["hooks"]), ["SessionStart"])
        [entry] = settings["hooks"]["SessionStart"]
        [hook] = entry["hooks"]
        self.assertEqual(hook["type"], "command")
        self.assertEqual(shlex.split(hook["command"])[-4:], ["hook", "session-start", str(THREAD), str(os.getpid())])

    def run_claims(self, *, uuid: str = UUID, env: dict[str, str] | None = None) -> list[tuple[list[str], object]]:
        """Runs the pane once on Enter and returns (argv, the claim read while the CLI runs) for each run."""
        seen: list[tuple[list[str], object]] = []

        def run_cli(argv: Sequence[str], cwd: str, environ: Mapping[str, str]) -> int:
            seen.append((list(argv), terminal_claim.read_claim(self.view, THREAD)))
            return 0

        self.make_topic(uuid=uuid)
        with mock.patch.dict(os.environ, env or {}):
            run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(), stop_after=1,
                           stdin=io.StringIO(), keys=ScriptedKeys([ENTER]), run=run_cli,
                           new_uuid=lambda: FRESH_UUID)
        return seen

    def test_open_session_writes_pane_uuid_config_dir_and_cwd_into_the_claim(self) -> None:
        [(argv, claim)] = self.run_claims(env={"TMUX_PANE": "%7"})
        self.assertEqual(argv[argv.index("--resume") + 1], UUID)
        self.assertEqual((claim.pane, claim.uuid), ("%7", UUID))
        self.assertEqual(claim.config_dir, os.path.join(self.home, ".claude"))  # the launcher's folder, expanded
        self.assertEqual(os.path.realpath(claim.cwd), os.path.realpath(self.folder))
        self.assertEqual(claim.pid, os.getpid())

    def test_the_claim_a_new_tab_takes_is_not_hooked_until_the_hook_runs(self) -> None:
        seen: list[bool] = []

        def run_cli(argv: Sequence[str], cwd: str, environ: Mapping[str, str]) -> int:
            seen.append(terminal_claim.read_claim(self.view, THREAD).hooked is False)
            return 0

        self.make_topic(uuid=UUID)
        with mock.patch.dict(os.environ, {"TMUX_PANE": "%7"}):
            run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(), stop_after=1,
                           stdin=io.StringIO(), keys=ScriptedKeys([ENTER]), run=run_cli,
                           new_uuid=lambda: FRESH_UUID)
        self.assertEqual(seen, [True])

    def test_open_session_names_the_tmux_socket_of_the_pane(self) -> None:
        [(_argv, claim)] = self.run_claims(env={"TMUX_PANE": "%7", "TMUX": "/tmp/tmux-1000/default,4242,0"})
        self.assertEqual(claim.socket, "/tmp/tmux-1000/default")

    def test_a_fresh_topic_starts_with_session_id_and_its_claim_names_the_same_uuid(self) -> None:
        [(argv, claim)] = self.run_claims(uuid="", env={"TMUX_PANE": "%9"})
        self.assertEqual(argv[argv.index("--session-id") + 1], FRESH_UUID)
        self.assertNotIn("--resume", argv)
        self.assertEqual((claim.uuid, claim.pane), (FRESH_UUID, "%9"))
        self.assertIsNone(terminal_claim.read_claim(self.view, THREAD), "the claim is gone after the CLI exits")

    def test_the_session_the_cli_ran_is_recorded_when_it_exits(self) -> None:
        self.run_claims(uuid="", env={"TMUX_PANE": "%9"})
        ended = terminal_claim.take_ended(self.view, THREAD)
        self.assertIsNotNone(ended, "the pane did not record the session its CLI ran")
        self.assertEqual(ended.uuid, FRESH_UUID)
        self.assertEqual(ended.started_uuid, FRESH_UUID)
        self.assertEqual(ended.config_dir, os.path.join(self.home, ".claude"))
        self.assertEqual(os.path.realpath(ended.cwd), os.path.realpath(self.folder))
        self.assertGreater(ended.ended, 0.0)

    def test_a_session_moved_while_the_cli_ran_is_recorded_with_its_last_uuid(self) -> None:
        NEW_UUID = "2d4f6a8c-0e1b-4a3c-9d5e-7f9a1b3c5d7e"
        THIRD_UUID = "3e5a7b9d-1f2c-4b4d-8e6f-0a1b2c3d4e5f"

        def run_cli(argv: Sequence[str], cwd: str, environ: Mapping[str, str]) -> int:
            terminal_claim.update_uuid(self.view, THREAD, os.getpid(), NEW_UUID)  # the SessionStart hook after /clear
            terminal_claim.update_uuid(self.view, THREAD, os.getpid(), THIRD_UUID)  # and after a /resume
            return 0

        self.make_topic(uuid=UUID)
        with mock.patch.dict(os.environ, {"TMUX_PANE": "%9"}):
            run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(), stop_after=1,
                           stdin=io.StringIO(), keys=ScriptedKeys([ENTER]), run=run_cli,
                           new_uuid=lambda: FRESH_UUID)
        ended = terminal_claim.take_ended(self.view, THREAD)
        self.assertIsNotNone(ended, "the pane did not record the session its CLI ran")
        self.assertEqual(ended.uuid, THIRD_UUID)
        self.assertEqual(ended.started_uuid, UUID)
        self.assertEqual(ended.seen, (UUID, NEW_UUID))

    def test_a_cli_that_could_not_start_records_no_session(self) -> None:
        def run_fails(argv: Sequence[str], cwd: str, environ: Mapping[str, str]) -> int:
            raise OSError("no such launcher")

        self.make_topic(uuid="")
        with mock.patch.dict(os.environ, {"TMUX_PANE": "%9"}):
            run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(), stop_after=1,
                           stdin=io.StringIO(), keys=ScriptedKeys([ENTER]), run=run_fails,
                           new_uuid=lambda: FRESH_UUID)
        self.assertIsNone(terminal_claim.take_ended(self.view, THREAD))

    def test_the_stored_pick_wins_over_the_stored_launcher(self) -> None:
        self.make_topic(launcher=fake_command("claude"), pick="claude-alt")
        self.pane([ENTER], rounds=1)
        self.assertEqual(self.children()[0]["kind"], "claude-alt")

    def test_the_topic_launcher_from_config_is_used_over_the_stored_one(self) -> None:
        self.topic_launcher = fake_command("claude-alt")
        self.make_topic(launcher=fake_command("claude"))
        self.pane([ENTER], rounds=1)
        self.assertEqual(self.children()[0]["kind"], "claude-alt")

    def test_the_claim_exists_while_the_cli_runs_and_is_gone_after_it(self) -> None:
        self.make_topic()
        self.pane([ENTER], rounds=1)
        [child] = self.children()
        self.assertTrue(child["claim"])
        self.assertFalse(os.path.exists(self.claim))

    def test_ctrl_c_and_q_do_nothing_on_ready(self) -> None:
        self.make_topic()
        out = self.pane([CTRL_C, b"q", CTRL_C], rounds=3)
        self.assertEqual(self.children(), [])
        self.assertIn(READY_HINT, out.getvalue())

    def test_the_ready_screen_shows_the_header_and_only_the_last_feed_lines(self) -> None:
        self.make_topic(feed=40, uuid="")
        out = self.pane([], rounds=1).getvalue()
        self.assertIn("launcher: claude-work", out)
        self.assertIn("session: new", out)
        self.assertIn("feed line 40", out)
        self.assertIn("feed line 11", out)
        self.assertNotIn("feed line 5\n", out)

    def test_a_topic_without_any_launcher_says_why_and_enter_does_nothing(self) -> None:
        self.make_topic(launcher="")
        out = self.pane([ENTER], rounds=1).getvalue()
        self.assertEqual(self.children(), [])
        self.assertIn("the launcher is empty", out)
        self.assertNotIn(READY_HINT, out)


class BusyTests(PaneCase):
    def test_a_running_topic_is_busy_and_enter_does_nothing(self) -> None:
        self.make_topic()
        self.write_meta("running")
        out = self.pane([ENTER], rounds=1)
        self.assertEqual(self.children(), [])
        self.assertFalse(os.path.exists(self.claim))
        self.assertIn(BUSY_LINE, out.getvalue())

    def test_a_lingering_topic_is_busy_and_enter_does_nothing(self) -> None:
        self.make_topic()
        self.write_meta("lingering")
        self.pane([ENTER], rounds=1)
        self.assertEqual(self.children(), [])
        self.assertFalse(os.path.exists(self.claim))

    def test_a_live_claim_of_another_process_refuses_enter_and_is_left_alone(self) -> None:
        self.make_topic()
        with open(self.claim, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": 1.0}, handle)
        self.pane([ENTER], rounds=1)
        self.assertEqual(self.children(), [])
        self.assertTrue(os.path.exists(self.claim))

    def test_a_turn_mark_keeps_the_topic_busy_while_the_meta_says_idle(self) -> None:
        self.make_topic()
        terminal_claim.begin_turn(self.view, THREAD, FakeClock())
        self.write_meta("idle")  # a TurnResult's idle while the CLI of the turn still runs
        out = self.pane([ENTER], rounds=1)
        self.assertEqual(self.children(), [])
        self.assertIn(BUSY_LINE, out.getvalue())

    def test_an_idle_topic_opens_on_enter_after_it_was_busy(self) -> None:
        self.make_topic()
        self.write_meta("running")
        self.pane([ENTER], rounds=1)
        self.assertEqual(self.children(), [])
        self.write_meta("idle")
        self.pane([ENTER], rounds=1)
        self.assertEqual(len(self.children()), 1)


class WatchedKeys:
    """The pane's keys on a terminal (the pane's own reader), with the local flags of the terminal read before each wait:
    the pane must have echo, line editing and signals off while it waits for keys. waiting holds the flags seen."""

    def __init__(self, slave: int) -> None:
        self._slave = slave
        self._keys = _TerminalKeys(slave)
        self.waiting: list[int] = []

    def read(self, seconds: float) -> bytes:
        self.waiting.append(termios.tcgetattr(self._slave)[3] & LOCAL_MODES)
        return self._keys.read(seconds)


class TerminalTests(PaneCase):
    """The pane on a real pseudo-terminal: Ctrl+C arrives as a byte and changes nothing, Enter runs the CLI, the terminal
    is raw while the pane waits for keys and cooked while the CLI runs, and it is the same as before when the pane ends."""

    def test_on_a_pty_ctrl_c_is_ignored_enter_runs_the_cli_and_the_attributes_come_back(self) -> None:
        self.make_topic()
        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        before = termios.tcgetattr(slave)
        self.assertEqual(before[3] & LOCAL_MODES, LOCAL_MODES, "the test needs a terminal in cooked mode")
        os.write(master, CTRL_C + ENTER)
        keys = WatchedKeys(slave)
        seen_while_running: list[bool] = []

        def run_cli(argv: Sequence[str], cwd: str, env: Mapping[str, str]) -> int:
            seen_while_running.append(termios.tcgetattr(slave) == before)
            return _run_child(argv, cwd, env)

        stream = os.fdopen(slave, "r", closefd=False)
        self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(),
                                        stop_after=2, stdin=stream, keys=keys, run=run_cli), 0)
        self.assertEqual(keys.waiting, [0, 0])  # no echo, no line editing, no signals: Ctrl+C is a byte
        self.assertEqual(seen_while_running, [True])  # the CLI runs on the terminal as it was before the pane
        self.assertEqual(termios.tcgetattr(slave), before)
        self.assertEqual(len(self.children()), 1)


class CountingKeys:
    """A key source that counts its reads. hang_up, when given, runs before the first read (the terminal goes away)."""

    def __init__(self, source: KeySource, hang_up: Callable[[], None] | None = None) -> None:
        self._source = source
        self._hang_up = hang_up
        self.reads = 0

    def read(self, seconds: float) -> bytes:
        if self.reads == 0 and self._hang_up is not None:
            self._hang_up()
        self.reads += 1
        return self._source.read(seconds)


class EndOfInputTests(PaneCase):
    """A pane whose terminal went away ends at once. Before the fix, select() kept saying readable and os.read returned
    b'' or raised EIO, so the loop spun on it. The reads are counted: a loop that keeps going reads the whole budget."""

    def test_a_pty_whose_master_closed_ends_the_pane_at_once(self) -> None:
        self.make_topic()
        master, slave = os.openpty()
        self.addCleanup(os.close, slave)
        keys = CountingKeys(_TerminalKeys(slave), hang_up=lambda: os.close(master))
        stream = os.fdopen(slave, "r", closefd=False)
        self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(),
                                        stop_after=1000, stdin=stream, keys=keys), 0)
        self.assertEqual(keys.reads, 1, "the pane kept reading a terminal that was gone")

    def test_a_key_source_at_end_of_file_ends_the_pane_at_once(self) -> None:
        self.make_topic()
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        os.close(write_end)  # no writer is left: the read end is readable and reads b''
        keys = CountingKeys(_TerminalKeys(read_end))
        self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(),
                                        stop_after=1000, stdin=io.StringIO(), keys=keys), 0)
        self.assertEqual(keys.reads, 1, "the pane kept reading an input that had ended")
        self.assertEqual(self.children(), [])


class ChildPidTests(PaneCase):
    """The CLI child's pid: _run_child reports it, and the default runner writes it into the claim the pane took."""

    def new_pane(self) -> ViewPane:
        return ViewPane(self.paths, THREAD, self.make_config, io.StringIO(), FakeClock(), ScriptedKeys([]))

    def test_run_child_reports_the_started_child_pid_then_0_once_it_exited(self) -> None:
        started: list[int] = []
        status = _run_child(["true"], self.home, {"PATH": os.environ.get("PATH", "")}, on_start=started.append)
        self.assertEqual(status, 0)
        [pid, exited] = started
        self.assertGreater(pid, 0)
        self.assertEqual(exited, 0)

    def test_child_started_writes_the_child_into_the_claim_the_pane_took(self) -> None:
        self.make_topic()
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 1.0, uuid=UUID, cwd=self.folder))
        self.new_pane()._child_started(4321)
        claim = terminal_claim.read_claim(self.view, THREAD)
        self.assertEqual((claim.pid, claim.child, claim.uuid), (os.getpid(), 4321, UUID))

    def test_child_started_leaves_a_claim_that_names_another_pid_alone(self) -> None:
        self.make_topic()
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getppid(), 1.0, uuid=UUID))
        self.new_pane()._child_started(4321)
        self.assertEqual(terminal_claim.read_claim(self.view, THREAD).child, 0)

    def test_the_default_runner_hands_the_started_child_to_the_claim_and_clears_it_on_exit(self) -> None:
        self.make_topic()
        with mock.patch.object(terminal_claim, "set_child", wraps=terminal_claim.set_child) as set_child:
            self.pane([ENTER], rounds=1)
        [child] = self.children()
        self.assertTrue(child["claim"])
        first, last = set_child.call_args_list[0][0], set_child.call_args_list[-1][0]
        self.assertEqual((first[0], first[1], first[2]), (self.view, THREAD, os.getpid()))
        self.assertGreater(first[3], 0)
        self.assertEqual((last[0], last[1], last[2], last[3]), (self.view, THREAD, os.getpid(), 0))
        claim = terminal_claim.read_claim(self.view, THREAD)
        self.assertTrue(claim is None or claim.child == 0)


class ChildEnvTests(PaneCase):
    """The CLI's environment: no CLAUDE* variable of the pane reaches it, and TZ is set as the bridge sets it for its
    own turns. The bridge sets no config dir, so none is added for the launcher."""

    def test_the_child_env_drops_every_claude_variable_of_the_pane(self) -> None:
        inherited = {"CLAUDE_CONFIG_DIR": "/work/other-config", "CLAUDE_PID": "123", "CLAUDE_EFFORT": "high",
                     "CLAUDE_PLUGIN_ROOT": "/plugins", "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli"}
        self.make_topic()
        seen: list[Mapping[str, str]] = []

        def run_cli(argv: Sequence[str], cwd: str, env: Mapping[str, str]) -> int:
            seen.append(env)
            return 0

        with mock.patch.dict(os.environ, inherited):
            self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(),
                                            stop_after=1, stdin=io.StringIO(), keys=ScriptedKeys([ENTER]),
                                            run=run_cli), 0)
        [env] = seen
        self.assertEqual([name for name in env if name.startswith("CLAUDE")], [])
        self.assertEqual(env["TZ"], TIMEZONE)
        self.assertIn("PATH", env)

    def test_child_env_keeps_the_other_variables_and_sets_only_tz(self) -> None:
        self.assertEqual(child_env({"PATH": "/bin", "HOME": "/h", "CLAUDE_CONFIG_DIR": "/c", "TZ": "UTC"}),
                         {"PATH": "/bin", "HOME": "/h", "TZ": TIMEZONE})


class SessionTests(HomeCase):
    def test_a_new_tg_bridge_session_turns_the_mouse_on(self) -> None:
        tmux = ScriptedTmux(has_session=1)
        Viewer(self.paths, self.config, tmux).ensure_session()
        self.assertIn(["set-option", "-t", SESSION_TARGET_COLON, "mouse", "on"], tmux.calls)


class RecordingSwitch:
    """A launcher change that records its calls and replies with the launcher's name."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str]] = []

    def __call__(self, thread: int, name: str, picked: str) -> str:
        self.calls.append((thread, name, picked))
        return f"switched to {picked}"


class ChangedBetweenScreensKey:
    """A key source whose first key arrives after `change` ran on the topic's view folder: the pane drew the idle screen,
    then the daemon's turn mark or a lingering meta arrived before the key."""

    def __init__(self, view_dir: str, change: Callable[[str], None], key: bytes) -> None:
        self._view = view_dir
        self._change = change
        self._key = key
        self._read = False

    def read(self, seconds: float) -> bytes:
        if self._read:
            return b""
        self._read = True
        self._change(self._view)
        return self._key


class ClaimedByAnotherPaneKey:
    """A key source whose first key arrives while another live pane (this test process) claims the topic. The topic
    looks idle to the pane's screen (no meta, no turn mark), so the ready screen is drawn again with the refusal."""

    def __init__(self, view_dir: str, thread: int, key: bytes) -> None:
        self._claim = terminal_claim.claim_path(view_dir, thread)
        self._key = key
        self._read = False

    def read(self, seconds: float) -> bytes:
        if self._read:
            return b""
        self._read = True
        with open(self._claim, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": 1.0}, handle)
        return self._key


class LauncherKeyTests(PaneCase):
    """The number keys of the ready screen: the list line, the switch, the reply, and the refusals."""

    def test_the_list_line_marks_the_current_claude_launcher(self) -> None:
        self.make_topic(pick="claude-alt")
        out = self.pane([], rounds=1).getvalue()
        self.assertIn("  1 claude-work   2 claude-alt \u2713   3 claude", out)
        self.assertIn(CHOICE_HINT_LINE, out)

    def test_a_digit_changes_the_launcher_and_shows_the_reply(self) -> None:
        self.make_topic()
        switch = RecordingSwitch()
        out = self.pane([b"3"], rounds=2, switch=switch)
        self.assertEqual(switch.calls, [(THREAD, "alpha", "claude")])
        self.assertIn("switched to claude", out.getvalue())

    def test_the_reply_stays_until_the_next_key(self) -> None:
        self.make_topic()
        out = self.pane([b"3", b"x"], rounds=3, switch=RecordingSwitch()).getvalue()
        screens = out.split(CLEAR_SCREEN)
        self.assertIn("switched to claude", screens[-2])
        self.assertNotIn("switched to claude", screens[-1])

    def test_a_digit_with_no_launcher_behind_it_is_ignored(self) -> None:
        self.make_topic()
        switch = RecordingSwitch()
        self.pane([b"9"], rounds=1, switch=switch)
        self.assertEqual(switch.calls, [])

    def test_a_claimed_topic_refuses_the_switch_under_the_lock(self) -> None:
        self.make_topic()
        switch = RecordingSwitch()
        out = io.StringIO()
        run_topic_pane(self.paths, THREAD, self.make_config, out=out, clock=FakeClock(), stop_after=2,
                       stdin=io.StringIO(), keys=ClaimedByAnotherPaneKey(self.view, THREAD, b"2"), switch=switch)
        self.assertEqual(switch.calls, [])
        self.assertIn(BUSY_REFUSED_LINE, out.getvalue())

    def test_a_digit_on_the_busy_screen_is_ignored(self) -> None:
        self.make_topic()
        self.write_meta("running")
        switch = RecordingSwitch()
        out = self.pane([b"2"], rounds=1, switch=switch).getvalue()
        self.assertEqual(switch.calls, [])
        self.assertIn(BUSY_LINE, out)
        self.assertNotIn(BUSY_REFUSED_LINE, out)

    def test_a_failed_switch_keeps_the_pane_alive_and_shows_the_failed_line(self) -> None:
        self.make_topic()
        for exc in (ValueError("bad pick"), OSError("disk gone"), sqlite3.Error("db locked"),
                    RuntimeError("schema too new"), KeyError("claude-alt")):
            with self.subTest(exc=type(exc).__name__):
                def failing(thread: int, name: str, picked: str, error: Exception = exc) -> str:
                    raise error

                out = io.StringIO()
                self.assertEqual(run_topic_pane(self.paths, THREAD, self.make_config, out=out, clock=FakeClock(),
                                                stop_after=2, stdin=io.StringIO(), keys=ScriptedKeys([b"3"]),
                                                switch=failing), 0)
                self.assertIn(SWITCH_FAILED_LINE.format(detail=exc), out.getvalue())

    def test_a_turn_mark_set_before_the_digit_refuses_the_switch_and_does_not_call_it(self) -> None:
        self.make_topic()
        switch = RecordingSwitch()
        out = io.StringIO()
        run_topic_pane(self.paths, THREAD, self.make_config, out=out, clock=FakeClock(), stop_after=2,
                       stdin=io.StringIO(), keys=ChangedBetweenScreensKey(
                           self.view, lambda view: terminal_claim.begin_turn(view, THREAD, FakeClock()), b"2"),
                       switch=switch)
        self.assertEqual(switch.calls, [])
        self.assertIn(BUSY_LINE, out.getvalue())  # the next screen is the busy one, which drops the refusal reply

    def test_a_lingering_meta_set_before_the_digit_refuses_the_switch_and_does_not_call_it(self) -> None:
        self.make_topic()
        switch = RecordingSwitch()
        out = io.StringIO()
        run_topic_pane(self.paths, THREAD, self.make_config, out=out, clock=FakeClock(), stop_after=2,
                       stdin=io.StringIO(), keys=ChangedBetweenScreensKey(self.view, lambda _view: self.write_meta("lingering"),
                                                                          b"2"),
                       switch=switch)
        self.assertEqual(switch.calls, [])
        self.assertIn(BUSY_LINE, out.getvalue())  # the next screen is the busy one, which drops the refusal reply


class SwitchTests(PaneCase):
    """The real switch (view_launcher.switch_launcher) on a migrated database in the temp state dir."""

    def make_split_config(self) -> Config:
        """Two Claude launchers with their own config folders: a switch between them is not a same-folder change."""
        launchers = {kind: {"cmd": fake_command(kind), "config_dir": os.path.join(self.home, f"cfg-{kind}")}
                     for kind in ("claude-work", "claude-alt")}
        return Config({"launchers": launchers}, home=self.home, env={}, log=lambda _line: None)

    def test_the_real_switch_stores_the_pick_the_telegram_command_stores(self) -> None:
        config = self.make_split_config()
        self.make_topic(uuid="")
        reply = switch_launcher(self.paths, lambda: config, THREAD, "alpha", "claude-alt")
        self.assertIn("launcher: claude-alt", reply)
        row = read_topic(self.paths, THREAD)
        self.assertIsNotNone(row)
        self.assertEqual(row.launcher_pick, "claude-alt")

    def test_the_next_enter_after_a_digit_opens_the_new_launcher(self) -> None:
        self.make_topic(uuid="")
        self.pane([b"2", ENTER], rounds=2)
        self.assertEqual([child["kind"] for child in self.children()], ["claude-alt"])


class CodeChangeTests(PaneCase):
    def run_pane(self, changed: list[bool], rounds: int) -> tuple[int, int]:
        """The pane's status and how often code_changed was asked."""
        asked = []
        answers = iter(changed)
        self.make_topic()
        status = run_topic_pane(self.paths, THREAD, self.make_config, out=io.StringIO(), clock=FakeClock(),
                                stop_after=rounds, stdin=io.StringIO(), keys=ScriptedKeys([]),
                                code_changed=lambda: asked.append(1) or next(answers))
        return status, len(asked)

    def test_changed_code_ends_the_pane_with_the_restart_status(self) -> None:
        self.assertEqual(self.run_pane([False, True, False], rounds=3), (RESTART_STATUS, 2))

    def test_unchanged_code_keeps_the_pane_running(self) -> None:
        self.assertEqual(self.run_pane([False, False, False], rounds=3), (0, 3))
