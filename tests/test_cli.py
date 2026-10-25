"""CLI (STEP 54): argv routing and exit codes of the main entry, the send, sendfile, type and spawn commands, the
daemon probe, and the globals scan (tests/check_globals.py).

Routing is tested with stub runners, so no App or tmux is needed. The send and sendfile commands run against an App over
FakeTelegram and FakeUploader in a TempHome. The type and spawn commands run against a scripted tmux. The probe starts the
real entry script `tg-bridge.py daemon` in a child process with TG_BRIDGE_FAKE_TG=1 under a TempHome, then checks the
second-daemon refusal, the SIGTERM exit (status 0, under 5 s), the stop log lines, the kept active marker and the released
lock. The probe waits for the poll loop's first log line: the fake answers getUpdates with a non-list, so that pass logs
'poll loop:' once, and by then start-up recovery and the inbox replay are over.
"""

import contextlib
import dataclasses
import io
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock

import tg_bridge
from tests import check_globals
from tests.fakes.backend import FakeBackend
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tg_bridge import cli
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.app import App
from tg_bridge.config import BACKEND_CLAUDE, Config
from tg_bridge.core.turn_runner import active_key
from tg_bridge.daemon.inbox import GROUP_KEY
from tg_bridge.daemon.lifecycle import ALREADY_RUNNING, acquire_lock
from tg_bridge.daemon.panes import SPAWN_PREAMBLE, Panes
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.telegram.proxy import OpenerCache
from tg_bridge.telegram.upload import Uploader

V3_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ENTRY = os.path.join(V3_DIR, "tg-bridge.py")
THREAD = 9201
TOPIC = "side"
GROUP = -1009201
CLAUDE_LAUNCHER = "claude-cmd"
PROBE_STOP_SECONDS = 5.0
PROBE_READY_SECONDS = 30.0
READY_MARKER = "poll loop:"
STOP_LOG = "daemon stop (signal 15)"
STOPPED_LOG = "daemon stop: 0 session(s) signalled"
RESTART_MARKER = json.dumps({"chat_id": GROUP, "name": TOPIC, "started": "2026-10-25T00:00:00Z"})


class StubRunners:
    """Records each runner call and returns a scripted status."""

    def __init__(self, status: int = 0) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.status = status

    def runners(self) -> cli.Runners:
        def record(name: str, returns: int | None = None) -> Callable[..., int]:
            def call(*args: Any) -> int:
                self.calls.append((name, args))
                return self.status if returns is None else returns
            return call

        return cli.Runners(daemon=lambda: self.calls.append(("daemon", ())), send=record("send"),
                           sendfile=record("sendfile"), type_text=record("type"), spawn=record("spawn"))


def _run_quietly(action: Callable[[], Any]) -> tuple[Any, str, str]:
    """Runs action with stdout and stderr captured. Returns (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = action()
    return result, out.getvalue(), err.getvalue()


class RouteTests(unittest.TestCase):
    """The argv of each subcommand reaches its runner with the arguments the live main() used."""

    def setUp(self) -> None:
        self.stub = StubRunners()
        self.runners = self.stub.runners()

    def test_no_subcommand_runs_the_daemon(self) -> None:
        self.assertEqual(cli.route([], self.runners), 0)
        self.assertEqual(self.stub.calls, [("daemon", ())])

    def test_daemon_ignores_extra_arguments(self) -> None:
        self.assertEqual(cli.route(["daemon", "extra"], self.runners), 0)
        self.assertEqual(self.stub.calls, [("daemon", ())])

    def test_send_passes_topic_and_text(self) -> None:
        self.assertEqual(cli.route(["send", TOPIC, "hello there"], self.runners), 0)
        self.assertEqual(self.stub.calls, [("send", (TOPIC, "hello there"))])

    def test_send_ignores_extra_arguments_as_before(self) -> None:
        cli.route(["send", TOPIC, "one", "two"], self.runners)
        self.assertEqual(self.stub.calls, [("send", (TOPIC, "one"))])

    def test_send_without_its_text_is_usage_and_status_1(self) -> None:
        status, _, err = _run_quietly(lambda: cli.route(["send", TOPIC], self.runners))
        self.assertEqual(status, 1)
        self.assertEqual(err.strip(), cli.USAGE["send"])
        self.assertEqual(self.stub.calls, [])

    def test_sendfile_and_send_file_are_the_same_command(self) -> None:
        cli.route(["sendfile", TOPIC, "/tmp/a.png", "a caption"], self.runners)
        cli.route(["send-file", TOPIC, "/tmp/b.pdf"], self.runners)
        self.assertEqual(self.stub.calls, [("sendfile", (TOPIC, "/tmp/a.png", "a caption")),
                                           ("sendfile", (TOPIC, "/tmp/b.pdf", ""))])

    def test_sendfile_without_a_path_is_usage(self) -> None:
        status, _, err = _run_quietly(lambda: cli.route(["sendfile", TOPIC], self.runners))
        self.assertEqual((status, err.strip()), (1, cli.USAGE["sendfile"]))

    def test_type_passes_pane_and_text(self) -> None:
        cli.route(["type", "%3", "ls -la"], self.runners)
        self.assertEqual(self.stub.calls, [("type", ("%3", "ls -la"))])

    def test_type_without_text_is_usage(self) -> None:
        status, _, err = _run_quietly(lambda: cli.route(["type", "%3"], self.runners))
        self.assertEqual((status, err.strip()), (1, cli.USAGE["type"]))

    def test_spawn_passes_command_name_and_text(self) -> None:
        cli.route(["spawn", "/opt/budget-claude", "budget.task", "fix it"], self.runners)
        self.assertEqual(self.stub.calls, [("spawn", ("/opt/budget-claude", "budget.task", "fix it"))])

    def test_spawn_needs_all_three_arguments(self) -> None:
        status, _, err = _run_quietly(lambda: cli.route(["spawn", "/opt/x", "name"], self.runners))
        self.assertEqual((status, err.strip()), (1, cli.USAGE["spawn"]))
        self.assertEqual(self.stub.calls, [])

    def test_the_runner_status_is_the_exit_status(self) -> None:
        stub = StubRunners(status=1)
        self.assertEqual(cli.route(["type", "%3", "x"], stub.runners()), 1)

    def test_help_and_unknown_words_print_the_spec_and_exit_0(self) -> None:
        for word in ("help", "no-such-command"):
            status, out, _ = _run_quietly(lambda word=word: cli.route([word], self.runners))
            self.assertEqual((status, out), (0, tg_bridge.__doc__), word)
        self.assertEqual(self.stub.calls, [])


class ConfigLogTests(unittest.TestCase):
    def test_config_warnings_reach_the_bridge_log(self) -> None:
        """A rejected launcher is logged once by the daemon's and the short commands' config load (STEP 3)."""
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            with open(paths.config, "w", encoding="utf-8") as handle:
                json.dump({"launchers": {"x": {"cmd": "xcmd", "backend": "codex"}}}, handle)
            cli.load_config(paths)
            with open(paths.log, encoding="utf-8") as handle:
                hidden = [line for line in handle if "hidden" in line]
            self.assertEqual(len(hidden), 1, hidden)


class HelpTextTests(unittest.TestCase):
    def test_help_text_describes_the_codex_launcher_and_its_handoff(self) -> None:
        """DESIGN section 1: the help text is the old spec plus the Codex lines (agent_launchers, /launcher codex)."""
        text = tg_bridge.__doc__ or ""
        self.assertIn("agent_launchers", text)
        self.assertIn("/launcher codex", text)
        self.assertIn("summary", text)


class ExitStatusTests(unittest.TestCase):
    """A SystemExit maps to the status Python itself would use."""

    def test_no_code_is_0_an_int_is_itself_and_a_message_is_1(self) -> None:
        self.assertEqual(cli._exit_status(SystemExit()), 0)
        self.assertEqual(cli._exit_status(SystemExit(3)), 3)
        status, _, err = _run_quietly(lambda: cli._exit_status(SystemExit("tg-bridge: boom")))
        self.assertEqual(status, 1)
        self.assertEqual(err.strip(), "tg-bridge: boom")


class TopicCommandTests(unittest.TestCase):
    """send and sendfile against a real App over fakes: the topic is found by name, the group comes from kv."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        self.telegram = FakeTelegram(self.clock)
        self.uploader = FakeUploader()
        config = Config({}, home=self.home.home, env={})
        logger = Logger(os.path.join(self.home.state, "bridge.log"))
        registry = BackendRegistry(config, logger, self.clock)
        registry.register(BACKEND_CLAUDE, lambda cfg, log, clk: FakeBackend([]))
        self.app = App.build(Paths.from_home(self.home.home), config, telegram=self.telegram, clock=self.clock,
                             registry=registry, env={})
        self.addCleanup(self.app.close)
        self.app.sessions.ensure(THREAD, TOPIC, CLAUDE_LAUNCHER, self.home.home)

    def test_send_to_a_known_topic_reaches_its_thread_in_the_group(self) -> None:
        self.app.kv.set(GROUP_KEY, str(GROUP))
        self.assertEqual(cli.send_to_topic(self.app, TOPIC, "hello"), 0)
        self.assertEqual(self.telegram.texts(THREAD), ["hello"])
        sent = [c for c in self.telegram.calls if c.method == "sendMessage"]
        self.assertEqual(sent[0].params["chat_id"], GROUP)

    def test_send_before_the_group_is_learned_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli.send_to_topic(self.app, TOPIC, "hello")
        self.assertIn("unknown topic 'side' or group not learned yet", str(caught.exception))
        self.assertEqual(self.telegram.calls, ())

    def test_send_to_an_unknown_topic_is_refused(self) -> None:
        self.app.kv.set(GROUP_KEY, str(GROUP))
        with self.assertRaises(SystemExit) as caught:
            cli.send_to_topic(self.app, "nope", "hello")
        self.assertIn("tg-bridge send: unknown topic 'nope'", str(caught.exception))

    def test_sendfile_hands_the_file_to_the_uploader_with_the_caption(self) -> None:
        self.app.kv.set(GROUP_KEY, str(GROUP))
        path = os.path.join(self.home.home, "shot.png")
        with open(path, "wb") as handle:
            handle.write(b"png")
        app = dataclasses.replace(self.app, uploader=self.uploader)
        self.assertEqual(cli.send_file_to_topic(app, TOPIC, path, "a caption"), 0)
        self.assertEqual(self.uploader.files, ((GROUP, THREAD, path, "a caption"),))

    def test_sendfile_of_a_missing_file_is_refused_with_the_reason(self) -> None:
        self.app.kv.set(GROUP_KEY, str(GROUP))
        refusing = Uploader(self.app.paths, OpenerCache(lambda: ""), self.app.logger)
        app = dataclasses.replace(self.app, uploader=refusing)
        with self.assertRaises(SystemExit) as caught:
            cli.send_file_to_topic(app, TOPIC, os.path.join(self.home.home, "missing.pdf"), "")
        self.assertIn("tg-bridge sendfile: no such file", str(caught.exception))
        self.assertEqual(self.telegram.calls, ())

    def test_sendfile_to_an_unknown_topic_is_refused_before_any_upload(self) -> None:
        self.app.kv.set(GROUP_KEY, str(GROUP))
        app = dataclasses.replace(self.app, uploader=self.uploader)
        with self.assertRaises(SystemExit):
            cli.send_file_to_topic(app, "nope", "/tmp/whatever.pdf", "")
        self.assertEqual(self.uploader.files, ())


class ScriptedTmux:
    """Records every tmux call; new-window answers with a pane id and the capture shows a prompt."""

    def __init__(self, new_window: subprocess.CompletedProcess[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.new_window = new_window or _done("%9\n")

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if args[0] == "list-panes":
            return _done("%1\n%3\n%9\n")
        if args[0] == "new-window":
            return self.new_window
        if args[0] == "capture-pane":
            return _done("Claude Code\n❯ \n")
        return _done("")

    def keys(self) -> list[list[str]]:
        return [args for args in self.calls if args[0] == "send-keys"]


def _done(stdout: str, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["tmux"], returncode=code, stdout=stdout, stderr=stderr)


class TmuxCommandTests(unittest.TestCase):
    """type and spawn talk to tmux only: no App, no database, no preamble added to what the owner typed."""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tmux = ScriptedTmux()
        self.panes = Panes(clock=self.clock, run=self.tmux, spawn_cwd="/srv/work")

    def test_type_into_a_live_pane_prints_the_result_and_exits_0(self) -> None:
        status, out, _ = _run_quietly(lambda: cli.type_into_pane(self.panes, "%3", "ls\nnow"))
        self.assertEqual((status, out), (0, "typed into pane %3\n"))
        self.assertEqual(self.tmux.keys()[0], ["send-keys", "-t", "%3", "-l", "--", "ls now"])

    def test_type_into_a_dead_pane_exits_1(self) -> None:
        status, out, _ = _run_quietly(lambda: cli.type_into_pane(self.panes, "%404", "ls"))
        self.assertEqual(status, 1)
        self.assertIn("pane %404 is not alive", out)
        self.assertEqual(self.tmux.keys(), [])

    def test_spawn_types_the_text_verbatim_without_the_topic_preamble(self) -> None:
        status, out, _ = _run_quietly(lambda: cli.spawn_in_tmux(self.panes, "/opt/budget-claude", "budget.task", "fix it"))
        self.assertEqual((status, out), (0, "typed into pane %9\n"))
        self.assertEqual(self.tmux.keys()[0][-1], "fix it")
        self.assertNotIn(SPAWN_PREAMBLE[:20], " ".join(self.tmux.keys()[0]))
        new_window = next(args for args in self.tmux.calls if args[0] == "new-window")
        self.assertEqual(new_window[-1], "/opt/budget-claude")
        self.assertIn("budget-task", new_window)

    def test_spawn_that_cannot_open_a_window_exits_1(self) -> None:
        tmux = ScriptedTmux(new_window=_done("", 1, "no server running"))
        status, out, _ = _run_quietly(
            lambda: cli.spawn_in_tmux(Panes(clock=self.clock, run=tmux, spawn_cwd="/srv"), "/opt/x", "a.b", "hi")
        )
        self.assertEqual(status, 1)
        self.assertIn("spawn failed: no server running", out)


class MainTests(unittest.TestCase):
    """main() in process: the umask, the spec, usage and the TempHome paths."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.enterContext(mock.patch.dict(os.environ, {"TG_BRIDGE_FAKE_TG": "1"}))
        saved = os.umask(0o022)
        self.addCleanup(os.umask, saved)

    def test_main_sets_the_private_umask(self) -> None:
        _run_quietly(lambda: cli.main(["help"]))
        self.assertEqual(os.umask(0o022), cli.UMASK)

    def test_main_prints_the_spec_for_help(self) -> None:
        status, out, _ = _run_quietly(lambda: cli.main(["help"]))
        self.assertEqual((status, out), (0, tg_bridge.__doc__))

    def test_main_send_with_an_unknown_topic_exits_1_with_the_message(self) -> None:
        status, _, err = _run_quietly(lambda: cli.main(["send", "nope", "hi"]))
        self.assertEqual(status, 1)
        self.assertIn("tg-bridge send: unknown topic 'nope'", err)

    def test_main_send_without_arguments_is_usage(self) -> None:
        status, _, err = _run_quietly(lambda: cli.main(["send"]))
        self.assertEqual((status, err.strip()), (1, cli.USAGE["send"]))


class DaemonProbeTests(unittest.TestCase):
    """The real entry script as a daemon in a child process, with the fake Telegram and a TempHome."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        self.process: subprocess.Popen[bytes] | None = None
        self.addCleanup(self._stop_child)

    def _stop_child(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=PROBE_STOP_SECONDS)

    def _env(self) -> dict[str, str]:
        return {"PATH": os.environ.get("PATH", ""), "HOME": self.home.home, "LANG": "C.UTF-8",
                "TG_BRIDGE_FAKE_TG": "1"}

    def _start_daemon(self, stderr_path: str) -> subprocess.Popen[bytes]:
        with open(stderr_path, "wb") as stderr:
            return subprocess.Popen([sys.executable, "-I", ENTRY, "daemon"], env=self._env(),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr)

    def _log(self) -> str:
        try:
            with open(self.paths.log, encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            return ""

    def _wait_for(self, text: str) -> bool:
        deadline = time.monotonic() + PROBE_READY_SECONDS
        while time.monotonic() < deadline:
            if text in self._log():
                return True
            time.sleep(0.05)
        return False

    def test_daemon_refuses_a_second_start_stops_on_sigterm_and_keeps_the_marker(self) -> None:
        stderr_path = os.path.join(self.home.home, "daemon.err")
        self.process = self._start_daemon(stderr_path)
        self.assertTrue(self._wait_for(READY_MARKER), "the first daemon never reached its poll loop")

        second = self._start_daemon(os.path.join(self.home.home, "second.err"))
        self.assertEqual(second.wait(timeout=PROBE_READY_SECONDS), 1)
        with open(os.path.join(self.home.home, "second.err"), encoding="utf-8") as handle:
            self.assertIn(ALREADY_RUNNING, handle.read())

        db = Database(self.paths)
        try:
            KvRepo(db).set(active_key(THREAD), RESTART_MARKER)
        finally:
            db.close()

        started = time.monotonic()
        self.process.send_signal(signal.SIGTERM)
        status = self.process.wait(timeout=PROBE_STOP_SECONDS)
        elapsed = time.monotonic() - started
        self.assertEqual(status, 0, self._log())
        self.assertLess(elapsed, PROBE_STOP_SECONDS)
        log = self._log()
        self.assertIn(STOP_LOG, log)
        self.assertIn(STOPPED_LOG, log)

        db = Database(self.paths)
        try:
            self.assertEqual(KvRepo(db).get(active_key(THREAD)), RESTART_MARKER, "shutdown keeps the active marker")
        finally:
            db.close()
        lock = acquire_lock(self.paths.lock)  # released by the exit: a new daemon could start now
        lock.close()


class GlobalsScanTests(unittest.TestCase):
    """tests/check_globals.py: the package is clean, and the scan flags what the rule forbids."""

    def test_the_package_has_no_module_level_mutable_state(self) -> None:
        self.assertEqual(check_globals.scan(), [])

    def test_mutable_containers_and_calls_are_flagged(self) -> None:
        cases = {
            "X = [1]": "bound to a mutable value",
            "X = {}": "bound to a mutable value",
            "_LOCK = threading.Lock()": "bound to a mutable value",
            "X = dict(a=1)": "bound to a mutable value",
            "if True:\n    X = []": "bound to a mutable value",
        }
        for source, expected in cases.items():
            problems = check_globals.check_source(source)
            self.assertTrue(any(expected in line for line in problems), (source, problems))

    def test_lowercase_names_and_global_statements_are_flagged(self) -> None:
        self.assertTrue(check_globals.check_source("x = 1"))
        self.assertTrue(any("'global Z'" in line for line in
                            check_globals.check_source("def f():\n    global Z\n    Z = 1")))

    def test_immutable_values_and_type_aliases_pass(self) -> None:
        clean = (
            "LIMIT = 3\nNAMES = ('a', 'b')\nBOUNDS = -1, 60 * 60\nSEEN = frozenset({1})\nPATTERN = re.compile('x')\n"
            "Alias = dict[str, int]\nUnion = int | None\nMAP = MappingProxyType({'a': 1})\nSENTINEL = object()\n"
            "ALIAS = SOME_OTHER\nSTATE_DIR = os.path.join('a', 'b')\n__version__ = '3.0.0'\n"
            "def f():\n    local = []\n    return local\n"
        )
        self.assertEqual(check_globals.check_source(clean), [])

    def test_a_frozen_dataclass_value_passes_only_when_the_package_declares_it_frozen(self) -> None:
        source = "CAPS = Capabilities(live=True)\n"
        self.assertEqual(check_globals.check_source(source), [
            "<source>:1: 'CAPS' is bound to a mutable value"])
        self.assertEqual(check_globals.check_source(source, frozen=frozenset({"Capabilities"})), [])
        declared = "import dataclasses\n@dataclasses.dataclass(frozen=True, slots=True)\nclass Capabilities:\n    live: bool\n"
        import ast
        self.assertEqual(check_globals.frozen_dataclass_names(ast.parse(declared)), frozenset({"Capabilities"}))


if __name__ == "__main__":
    unittest.main()
