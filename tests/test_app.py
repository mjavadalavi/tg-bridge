"""App (STEP 53): the composition root wires every service once, by constructor injection.

The end-to-end test builds the App over a TempHome with a FakeTelegram, a FakeClock and a registry whose Claude backend
is a FakeBackend, then pushes one text update through the same entry points the poller uses (ingest, then process) and
waits for the answer to reach the topic. The TG_BRIDGE_FAKE_TG hook is checked both ways: honoured under a TempHome,
refused for the real home (checked without building anything there, so no real state directory is touched).
"""

import json
import os
import stat
import time
import unittest
from collections.abc import Callable
from unittest import mock

from tests.fakes.backend import FakeBackend
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.app import FAKE_TELEGRAM_ENV, App, fake_telegram_requested, real_home
from tg_bridge.bot.callbacks.handlers import CHOICE_EXPIRED_TOAST
from tg_bridge.bot.callbacks.terminal_taps import TerminalTaps
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.bot.terminal_guard import TerminalGuard
from tg_bridge.config import BACKEND_CLAUDE, Config
from tg_bridge.core.linger import LingerService
from tg_bridge.core.self_turn import SelfTurns
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.client import HttpTelegramClient

CHAT = -1009801
THREAD = 9801
TOPIC = "side"
ALLOWED_UID = 42
ANSWER = "hello back"
WAIT_SECONDS = 20.0


def _config(home: str) -> Config:
    project = os.path.join(home, "project")
    os.makedirs(project, exist_ok=True)
    return Config({
        "allow_user_ids": [ALLOWED_UID],
        "topics": {TOPIC: {"launcher": "claude-cmd", "cwd": project}},
    }, home=home, env={})


def _registry(config: Config, logger: Logger, clock: FakeClock, backend: FakeBackend) -> BackendRegistry:
    registry = BackendRegistry(config, logger, clock)
    registry.register(BACKEND_CLAUDE, lambda cfg, log, clk: backend)
    return registry


def _settle(clock: FakeClock, predicate: Callable[[], bool], timeout: float = WAIT_SECONDS) -> bool:
    """Waits for predicate() while the FakeClock moves on, one second at a time. The feed's close() joins its sender on
    the clock's timers, so a turn only finishes when time passes. Never moves more than needed for the answer."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        clock.advance(1.0)
        time.sleep(0.005)
    return True


def _text_update(update_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id, "message_thread_id": THREAD, "text": text,
            "chat": {"id": CHAT, "type": "supergroup", "title": "team", "is_forum": True},
            "from": {"id": ALLOWED_UID, "first_name": "Tester", "username": "tester"},
        },
    }


class AppEndToEndTests(unittest.TestCase):
    """One text update through the wired App: ingested, processed, answered by a turn on the FakeBackend."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        self.config = _config(self.home.home)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.backend = FakeBackend([[TurnResult(TurnStatus.OK, answer=ANSWER, duration_ms=5, num_turns=1,
                                                subtype="success")]])
        self.telegram = FakeTelegram(self.clock)
        self.app = App.build(Paths.from_home(self.home.home), self.config, telegram=self.telegram,
                             clock=self.clock, registry=_registry(self.config, self.logger, self.clock,
                                                                  self.backend),
                             env={})
        self.addCleanup(self.app.close)
        # the topic's name, as a forum_topic_created message would have left it (kept out of the test's path so no
        # folder card search runs)
        self.app.kv.set(f"thread:{THREAD}", TOPIC)

    def test_one_text_update_runs_one_turn_and_the_answer_reaches_the_topic(self) -> None:
        update = _text_update(1, "hello")
        self.assertTrue(self.app.inbox.ingest_update(update))
        self.app.inbox.process_update(update)
        answered = _settle(self.clock, lambda: ANSWER in self.telegram.texts(THREAD))
        self.assertTrue(answered, f"no answer in the topic: {self.telegram.texts(THREAD)!r}")
        self.assertEqual(len(self.backend.specs), 1, "exactly one session was opened for the turn")
        self.assertEqual(self.backend.specs[0].thread_id, THREAD)
        self.assertEqual(self.app.sessions.get(THREAD).uuid, self.backend.specs[0].new_id,
                         "the session id that was preassigned to the spawn is stored for the topic")
        self.assertEqual(self.app.shutdown_sessions(), 0, "the finished turn left no live session")

    def test_the_turn_leaves_the_topic_view_files_and_the_view_folder_is_private(self) -> None:
        update = _text_update(1, "hello")
        self.assertTrue(self.app.inbox.ingest_update(update))
        self.app.inbox.process_update(update)
        self.assertTrue(_settle(self.clock, lambda: ANSWER in self.telegram.texts(THREAD)))
        view = self.app.paths.view
        self.assertEqual(stat.S_IMODE(os.stat(view).st_mode), 0o700)
        with open(os.path.join(view, f"{THREAD}.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual((meta["topic"], meta["transport"], meta["state"]), (TOPIC, "print", "idle"))
        with open(os.path.join(view, f"{THREAD}.feed"), encoding="utf-8") as handle:
            self.assertIn("✅ done · 0:00", handle.read().splitlines())

    def test_the_services_share_one_live_registry_and_the_injected_registry(self) -> None:
        self.assertIs(self.app.registry.get(BACKEND_CLAUDE), self.backend)
        self.assertIs(self.app.client, self.telegram)
        self.assertIs(self.app.lifecycle._live, self.app.live)
        self.assertIs(self.app.inbox._messages, self.app.handler)
        self.assertIs(self.app.inbox._callbacks, self.app.router)

    def test_app_wires_one_linger_service_into_runner_and_dispatcher(self) -> None:
        """The turn runner and the dispatcher share one LingerService, which runs its self turns through a SelfTurns."""
        self.assertIsInstance(self.app.linger, LingerService)
        self.assertIsInstance(self.app.linger._self_turns, SelfTurns)
        self.assertIs(self.app.dispatcher._linger, self.app.linger)
        self.assertIs(self.app.dispatcher._turns._linger, self.app.linger)

    def test_the_terminal_guard_gets_the_live_registry_and_the_clock_and_settles_before_a_bridge_turn(self) -> None:
        """One TerminalGuard, built with the app's live registry and clock, is the dispatcher's before_turn hook and
        the guard the session commands and the terminal taps hold."""
        with TempHome() as home, \
                mock.patch("tg_bridge.app.TerminalGuard", wraps=TerminalGuard) as guard_cls, \
                mock.patch("tg_bridge.app.SessionCommands", wraps=SessionCommands) as session_cmds, \
                mock.patch("tg_bridge.app.TerminalTaps", wraps=TerminalTaps) as taps:
            config = _config(home.home)
            app = App.build(Paths.from_home(home.home), config, telegram=self.telegram, clock=self.clock,
                            registry=_registry(config, self.logger, self.clock, self.backend), env={})
            self.addCleanup(app.close)
        guard = app.dispatcher._before_turn.__self__
        self.assertIsInstance(guard, TerminalGuard)
        self.assertEqual(app.dispatcher._before_turn, guard.before_turn)
        self.assertIs(guard_cls.call_args.kwargs["live"], app.live)
        self.assertIs(guard_cls.call_args.kwargs["clock"], self.clock)
        self.assertIs(session_cmds.call_args.kwargs["terminal"], guard)
        self.assertIs(taps.call_args.kwargs["terminal"], guard)

    def test_the_router_handles_the_stop_choice_card_taps(self) -> None:
        """The D2 choice card's buttons (sq:) are registered on the App's router: a tap reaches its handler, which
        answers 'this choice has expired' when no kept update with the card's hold and self turn runs (the stop path is
        covered by test_callbacks)."""
        tap = {"id": "cb-choice", "from": {"id": ALLOWED_UID}, "data": f"cm:{THREAD}:sq:c:1:1",
               "message": {"message_id": 5, "chat": {"id": CHAT}, "text": "card"}}
        self.app.router.dispatch(tap)
        answers = [c.params.get("text") for c in self.telegram.calls if c.method == "answerCallbackQuery"]
        self.assertEqual(answers, [CHOICE_EXPIRED_TOAST])

    def test_the_menu_and_help_list_the_commands_in_the_live_order(self) -> None:
        """The /help text and the setMyCommands menu both follow the live BRIDGE_COMMANDS order (STEP 57 compares them)."""
        words = [word for word, _ in self.app.commands.bridge_commands()]
        self.assertEqual(words, ["status", "stop", "mode", "model", "launcher", "new", "clear", "resume", "cwd", "queue",
                                 "cost", "rules", "forget", "help"])

    def test_the_command_registry_holds_the_info_session_and_settings_commands(self) -> None:
        for word in ("help", "status", "cost", "stop", "new", "resume", "mode", "model", "launcher", "cwd"):
            self.assertIsNotNone(self.app.commands.find(word), word)

    def test_a_pending_row_is_replayed_once_at_start(self) -> None:
        """A row stored before a restart runs once on replay, is marked done, and a re-delivery never stores it again.

        Ported labels:
        d: no in-memory state for topic before replay
        d: replay dispatched the pending message
        d: run_turn_v2 called with its text
        d: replayed row done (got done)
        d: re-delivered update is not inserted again
        d: message ran exactly once
        """
        update = _text_update(3, "replayed")
        self.assertTrue(self.app.inbox.ingest_update(update))  # stored, then the process stopped before handling it
        self.assertEqual(self.app.dispatcher.queue_snapshot(THREAD), ())
        self.assertEqual(self.backend.specs, [], "no turn runs before the replay")
        self.app.inbox.replay_inbox()
        self.assertTrue(_settle(self.clock, lambda: ANSWER in self.telegram.texts(THREAD)))
        sent = [call[1] for session in self.backend.sessions for call in session.calls if call[0] == "send_user"]
        self.assertEqual(sent, ["replayed"])
        rows = self.app.db.query("SELECT state FROM inbox_queue WHERE update_id=?", (3,))
        self.assertEqual([row["state"] for row in rows], ["done"])
        self.app.inbox.replay_inbox()
        self.assertFalse(self.app.inbox.ingest_update(update))
        _settle(self.clock, lambda: False, timeout=0.2)  # gives a second run the chance to start; none may
        self.assertEqual(len(self.backend.specs), 1)

    def test_the_inbox_row_is_done_after_the_update_was_processed(self) -> None:
        update = _text_update(2, "again")
        self.app.inbox.ingest_update(update)
        self.app.inbox.process_update(update)
        self.assertTrue(_settle(self.clock, lambda: ANSWER in self.telegram.texts(THREAD)))
        rows = self.app.db.query("SELECT state FROM inbox_queue WHERE update_id=?", (2,))
        self.assertEqual([row["state"] for row in rows], ["done"])


class FakeTelegramHookTests(unittest.TestCase):
    """TG_BRIDGE_FAKE_TG=1 is honoured under a TempHome and refused for the real home."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.lines: list[str] = []

    def test_the_hook_is_honoured_under_a_temp_home(self) -> None:
        paths = Paths.from_home(self.home.home)
        self.assertTrue(fake_telegram_requested(paths, {FAKE_TELEGRAM_ENV: "1"}, self.lines.append))
        self.assertEqual(self.lines, [])

    def test_the_hook_is_refused_for_the_real_home_with_one_log_line(self) -> None:
        paths = Paths.from_home(real_home())
        self.assertFalse(fake_telegram_requested(paths, {FAKE_TELEGRAM_ENV: "1"}, self.lines.append))
        self.assertEqual(self.lines, ["TG_BRIDGE_FAKE_TG ignored: HOME is the real home"])

    def test_the_hook_is_off_unless_the_variable_is_exactly_one(self) -> None:
        paths = Paths.from_home(self.home.home)
        self.assertFalse(fake_telegram_requested(paths, {}, self.lines.append))
        self.assertFalse(fake_telegram_requested(paths, {FAKE_TELEGRAM_ENV: "0"}, self.lines.append))
        self.assertEqual(self.lines, [])

    def test_build_uses_the_fake_client_only_when_the_hook_is_on(self) -> None:
        clock = FakeClock()
        config = _config(self.home.home)
        logger = Logger(os.path.join(self.home.state, "bridge.log"))
        registry = _registry(config, logger, clock, FakeBackend())
        paths = Paths.from_home(self.home.home)
        hooked = App.build(paths, config, clock=clock, registry=registry, env={FAKE_TELEGRAM_ENV: "1"})
        self.addCleanup(hooked.close)
        self.assertIsInstance(hooked.client, FakeTelegram)
        plain = App.build(paths, config, clock=clock, registry=registry, env={})
        self.addCleanup(plain.close)
        self.assertIsInstance(plain.client, HttpTelegramClient)


class ViewFolderStartupTests(unittest.TestCase):
    """The view folder at start-up. A folder that cannot be created turns the views off: the App still builds and a turn
    still answers. Only the daemon's start-up (sweep_stale_views) clears the 'running' state a killed daemon left."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        self.config = _config(self.home.home)
        self.log_path = os.path.join(self.home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.paths = Paths.from_home(self.home.home)
        self.telegram = FakeTelegram(self.clock)
        self.backend = FakeBackend([[TurnResult(TurnStatus.OK, answer=ANSWER, duration_ms=5, num_turns=1,
                                                subtype="success")]])

    def build(self, sweep: bool = False) -> App:
        app = App.build(self.paths, self.config, telegram=self.telegram, clock=self.clock,
                        registry=_registry(self.config, self.logger, self.clock, self.backend), env={},
                        sweep_stale_views=sweep)
        self.addCleanup(app.close)
        return app

    def write_running_meta(self, thread: int) -> None:
        with open(os.path.join(self.paths.view, f"{thread}.json"), "w", encoding="utf-8") as handle:
            json.dump({"topic": TOPIC, "transport": "tui", "launcher": "claude-cmd", "state": "running",
                       "updated": 1.0}, handle)

    def stored_state(self, thread: int) -> str:
        with open(os.path.join(self.paths.view, f"{thread}.json"), encoding="utf-8") as handle:
            return json.load(handle)["state"]

    def test_a_view_folder_that_is_a_file_turns_the_views_off_and_the_turn_still_answers(self) -> None:
        with open(self.paths.view, "w", encoding="utf-8") as handle:
            handle.write("not a folder")
        app = self.build()
        app.kv.set(f"thread:{THREAD}", TOPIC)
        update = _text_update(1, "hello")
        self.assertTrue(app.inbox.ingest_update(update))
        app.inbox.process_update(update)
        self.assertTrue(_settle(self.clock, lambda: ANSWER in self.telegram.texts(THREAD)))
        self.assertTrue(os.path.isfile(self.paths.view))  # nothing was replaced or removed
        with open(self.log_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len([line for line in lines if "views are off" in line]), 1)
        self.assertEqual([line for line in lines if "view: cannot write" in line], [], "no recorder was built")

    def test_the_daemon_start_clears_the_running_state_a_killed_daemon_left(self) -> None:
        os.makedirs(self.paths.view, mode=0o700, exist_ok=True)
        self.write_running_meta(9001)
        self.build(sweep=True)
        self.assertEqual(self.stored_state(9001), "idle")

    def test_a_short_command_build_leaves_the_running_state_alone(self) -> None:
        os.makedirs(self.paths.view, mode=0o700, exist_ok=True)
        self.write_running_meta(9001)
        self.build(sweep=False)
        self.assertEqual(self.stored_state(9001), "running")


if __name__ == "__main__":
    unittest.main()
