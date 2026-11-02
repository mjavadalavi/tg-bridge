"""Lifecycle (STEP 52): port of the recover and shutdown scenarios of fakecli/test_inbox.py (labels kept in the test names
and docstrings), plus the lock, signal and command-menu parts of cmd_daemon.

Sessions are FakeBackend sessions (no subprocess), except one straggler test that gives a real child its own process group
so the SIGKILL path is exercised for real. Everything runs in a TempHome with FakeTelegram and FakeClock.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from collections.abc import Callable
from typing import Any

from tests.fakes.backend import FakeBackend, FakeSession, fake_spec
from tests.support import TempHome
from tests.turn_support import TurnHarness
from tg_bridge.agents.base import AgentSession
from tg_bridge.agents.events import AgentEvent
from tg_bridge.bot.commands.base import Alias, Command, CommandRegistry
from tg_bridge.core.linger import linger_key
from tg_bridge.core.turn_runner import active_key
from tg_bridge.daemon.lifecycle import (
    ALREADY_RUNNING,
    LINGER_RESTART_NOTICE,
    RESTART_NOTICE,
    Lifecycle,
    acquire_lock,
    ignore_stop_signals,
    install_stop_handlers,
)
from tg_bridge.paths import Paths
from tg_bridge.store.kv import KvRepo
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD

THREAD = 9001
CHAT = -100
STOP_SECONDS = 0.1  # the grace handed to every stop in these tests
BUDGET_SECONDS = 0.3  # the shutdown budget used where a session is expected to outlive it


def _marker(chat_id: int | None = CHAT, name: str = "side") -> str:
    """The JSON an active:<thread> marker stores (the same shape turn_runner writes)."""
    return json.dumps({"chat_id": chat_id, "name": name, "started": "2026-10-25T00:00:00Z"})


class StopSession(FakeSession):
    """A FakeBackend session whose stop() runs a test hook first. pgid, when given, is its child's process group."""

    def __init__(self, backend: FakeBackend, spec: Any, sink: Callable[[AgentEvent], None],
                 on_stop: Callable[[float], None], pgid: int | None = None) -> None:
        super().__init__(backend, spec, sink)
        self._on_stop = on_stop
        self._pgid = pgid

    @property
    def pgid(self) -> int | None:
        return self._pgid

    def stop(self, grace: float = 5.0) -> None:
        self._on_stop(grace)
        super().stop(grace)


class LifecycleCase(unittest.TestCase):
    """Builds the services a Lifecycle needs on real repositories under a TempHome."""

    def setUp(self) -> None:
        self._home = TempHome()
        self._home.__enter__()
        self.h = TurnHarness(self._home.home)
        self.backend = FakeBackend()
        self.commands = CommandRegistry()
        self.lifecycle = Lifecycle(
            live=self.h.live,
            permissions=self.h.permissions,
            kv=self.h.kv,
            messenger=self.h.messenger,
            telegram=self.h.tg,
            commands=self.commands,
            clock=self.h.clock,
            logger=self.h.logger,
        )

    def tearDown(self) -> None:
        self.h.db.close()
        self._home.__exit__(None, None, None)

    def live_session(self, thread: int, on_stop: Callable[[float], None] | None = None,
                     pgid: int | None = None) -> AgentSession:
        """A session registered as running a turn in `thread`. With on_stop it is a StopSession."""
        spec = fake_spec(thread_id=thread, chat_id=CHAT, topic=f"t{thread}", cwd=self._home.home,
                         errfile_dir=self._home.home)
        sink: Callable[[AgentEvent], None] = lambda event: None
        if on_stop is None:
            session: AgentSession = FakeSession(self.backend, spec, sink)
        else:
            session = StopSession(self.backend, spec, sink, on_stop, pgid)
        self.assertTrue(self.h.live.begin_turn(thread, session, object()))
        return session

    def log_text(self) -> str:
        return "\n".join(self.h.log_lines())

    def open_ask(self, nonce: str, thread: int, command: str, message_id: int | None) -> None:
        self.h.asks.insert(nonce=nonce, thread_id=thread, chat_id=CHAT, request_id=f"perm_{nonce}",
                           tool_name="Bash", input={"command": command})
        if message_id is not None:
            self.h.asks.set_message_id(nonce, message_id)


class RecoverTests(LifecycleCase):
    def test_recover_active_marker_sends_restart_notice_and_deletes_marker(self) -> None:
        """recover: active marker -> restart notice in its topic; the marker is removed.

        Ported labels:
            restart notice sent to topic 9001 (got 1)
            active:9001 key deleted
        """
        self.h.kv.set(active_key(THREAD), _marker())
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [RESTART_NOTICE])
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))
        self.assertEqual((recovery.markers, recovery.notices, recovery.asks), (1, 1, 0))

    def test_recover_marker_without_chat_is_removed_without_notice(self) -> None:
        """recover: a marker that names no chat gets no notice, but it is still cleared."""
        self.h.kv.set(active_key(THREAD), _marker(chat_id=None))
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [])
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))
        self.assertEqual((recovery.markers, recovery.notices), (1, 0))

    def test_recover_malformed_marker_is_removed_quietly(self) -> None:
        """recover: a key without a topic id or a value that is not JSON is removed and sends nothing."""
        self.h.kv.set("active:abc", "not json")
        self.h.kv.set(active_key(THREAD), "[1, 2]")
        recovery = self.lifecycle.recover()
        self.assertEqual([c for c in self.h.tg.calls if c.method == "sendMessage"], [])
        self.assertEqual(self.h.kv.items_with_prefix("active:"), [])
        self.assertEqual((recovery.markers, recovery.notices), (2, 0))

    def test_recover_unanswered_ask_card_is_edited_and_ask_voided(self) -> None:
        """recover: unanswered ask -> card edited with the void note, command kept, keyboard emptied, ask voided.

        Ported labels:
            edited card has empty keyboard
            ask card edited with void note, command kept
            ask answer set to void
        """
        self.open_ask("n-open", THREAD, "ls -la", message_id=777)
        recovery = self.lifecycle.recover()
        edits = [c for c in self.h.tg.calls if c.method == "editMessageText" and c.params.get("message_id") == 777]
        self.assertEqual(len(edits), 1)
        self.assertIn("⌛ void — bridge restarted", edits[0].params["text"])
        self.assertIn("ls -la", edits[0].params["text"])
        self.assertEqual(edits[0].params["reply_markup"], EMPTY_KEYBOARD)
        self.assertEqual(self.h.asks.by_nonce("n-open").answer, "void")
        self.assertEqual(recovery.asks, 1)

    def test_recover_leaves_answered_asks_alone(self) -> None:
        """recover: an ask that was answered before the restart keeps its answer and its card is not touched."""
        self.open_ask("n-done", THREAD, "git status", message_id=888)
        self.assertTrue(self.h.asks.answer("n-done", "allow"))
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.asks.by_nonce("n-done").answer, "allow")
        self.assertEqual(self.h.tg.edits(888), [])
        self.assertEqual(recovery.asks, 0)

    def test_recover_twice_finds_nothing_the_second_time(self) -> None:
        """recover: running it again after it finished sends no second notice and voids nothing more."""
        self.h.kv.set(active_key(THREAD), _marker())
        self.open_ask("n-open", THREAD, "ls", message_id=777)
        self.lifecycle.recover()
        second = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [RESTART_NOTICE])
        self.assertEqual((second.markers, second.notices, second.asks), (0, 0, 0))

    def test_linger_marker_gets_a_notice_and_is_removed(self) -> None:
        """recover: a kept-session marker (linger:<thread>) -> linger restart notice in its topic; the marker is removed."""
        self.h.kv.set(linger_key(THREAD), _marker())
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [LINGER_RESTART_NOTICE])
        self.assertEqual(self.h.kv.items_with_prefix("linger:"), [])
        self.assertEqual((recovery.markers, recovery.notices, recovery.lingers), (0, 1, 1))

    def test_active_and_linger_markers_give_one_notice(self) -> None:
        """recover: a topic with both an active and a linger marker gets the turn notice only; both markers go."""
        self.h.kv.set(active_key(THREAD), _marker())
        self.h.kv.set(linger_key(THREAD), _marker())
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [RESTART_NOTICE])
        self.assertEqual(self.h.kv.items_with_prefix("linger:"), [])
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))
        self.assertEqual((recovery.markers, recovery.notices, recovery.lingers), (1, 1, 1))

    def test_active_marker_without_chat_does_not_suppress_the_linger_notice(self) -> None:
        """recover: an active marker that names no chat sends nothing, so the topic's kept-session marker still gets
        the linger notice (the notified set only holds topics that were actually told)."""
        self.h.kv.set(active_key(THREAD), _marker(chat_id=None))
        self.h.kv.set(linger_key(THREAD), _marker())
        recovery = self.lifecycle.recover()
        self.assertEqual(self.h.tg.texts(THREAD), [LINGER_RESTART_NOTICE])
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))
        self.assertEqual(self.h.kv.items_with_prefix("linger:"), [])
        self.assertEqual((recovery.markers, recovery.notices, recovery.lingers), (1, 1, 1))

    def test_bad_linger_marker_is_removed_without_notice(self) -> None:
        """recover: a linger marker with no topic id, a value that is not JSON, or no chat is removed and sends nothing."""
        self.h.kv.set("linger:abc", _marker())
        self.h.kv.set(linger_key(THREAD), "not json")
        self.h.kv.set("linger:9002", _marker(chat_id=None))
        recovery = self.lifecycle.recover()
        self.assertEqual([c for c in self.h.tg.calls if c.method == "sendMessage"], [])
        self.assertEqual(self.h.kv.items_with_prefix("linger:"), [])
        self.assertEqual((recovery.lingers, recovery.notices), (3, 0))


class ShutdownTests(LifecycleCase):
    def test_shutdown_stops_kept_sessions(self) -> None:
        """shutdown_sessions: a kept session (lingering, not running a turn) is stopped like a running one."""
        spec = fake_spec(thread_id=THREAD, chat_id=CHAT, topic="t", cwd=self._home.home, errfile_dir=self._home.home)
        session = FakeSession(self.backend, spec, lambda event: None)
        self.assertTrue(self.h.live.begin_linger(THREAD, session, limit=3))
        count = self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS)
        self.assertEqual(count, 1)
        self.assertFalse(session.alive())
        self.assertIn(("stop", STOP_SECONDS), session.calls)

    def test_shutdown_sessions_stops_the_live_session_and_keeps_its_marker(self) -> None:
        """shutdown_sessions: the live session is stopped with the grace; its active marker is kept on purpose.

        Ported labels:
            shutdown_sessions stopped the session (n=1, took 0.0s)
            hanging fake session is running before shutdown
            active:9009 marker kept after shutdown
        """
        self.h.kv.set(active_key(THREAD), _marker())
        session = self.live_session(THREAD)
        count = self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS)
        self.assertEqual(count, 1)
        self.assertFalse(session.alive())
        self.assertIn(("stop", STOP_SECONDS), session.calls)  # FakeSession records the grace it was stopped with
        self.assertIsNotNone(self.h.kv.get(active_key(THREAD)))

    def test_shutdown_sessions_stops_all_sessions_in_parallel(self) -> None:
        """shutdown_sessions: two sessions stop at the same time. A barrier that needs both to arrive would break the
        second stop (and log it) if they were run one after the other."""
        barrier = threading.Barrier(2, timeout=5)
        stopped: list[str] = []

        def meet(grace: float) -> None:
            barrier.wait()
            stopped.append("met")

        first = self.live_session(9011, on_stop=meet)
        second = self.live_session(9012, on_stop=meet)
        count = self.lifecycle.shutdown_sessions(budget=10.0, grace=STOP_SECONDS)
        self.assertEqual(count, 2)
        self.assertEqual(stopped, ["met", "met"])
        self.assertFalse(first.alive())
        self.assertFalse(second.alive())
        self.assertNotIn("shutdown:", self.log_text())

    def test_shutdown_refuses_a_turn_that_starts_after_it_began(self) -> None:
        """shutdown_sessions: after the snapshot no new turn can be registered (the spawn is refused)."""
        self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS)
        late = FakeSession(self.backend, fake_spec(thread_id=9013), lambda event: None)
        self.assertFalse(self.h.live.begin_turn(9013, late, object()))
        self.assertTrue(self.h.live.is_shutting_down())

    def test_shutdown_with_no_live_session_returns_zero(self) -> None:
        """shutdown_sessions: nothing running means nothing to stop."""
        self.assertEqual(self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS), 0)

    def test_failing_stop_is_logged_and_the_other_sessions_still_stop(self) -> None:
        """shutdown_sessions: one session whose stop raises is logged; the others are stopped all the same."""

        def broken(grace: float) -> None:
            raise RuntimeError("stop went wrong")

        healthy = self.live_session(9014)
        self.live_session(9015, on_stop=broken)
        count = self.lifecycle.shutdown_sessions(budget=10.0, grace=STOP_SECONDS)
        self.assertEqual(count, 2)
        self.assertFalse(healthy.alive())
        self.assertIn("shutdown: stop went wrong", self.log_text())

    def test_straggler_process_group_is_sigkilled_after_the_budget(self) -> None:
        """shutdown_sessions: live session killed as a group. A child that outlives the budget is SIGKILLed, so no
        orphan process is left behind.

        Ported labels:
            process group gone after shutdown
        """
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        release = threading.Event()
        try:
            self.live_session(9020, on_stop=lambda grace: release.wait(30), pgid=child.pid)
            started = time.monotonic()
            count = self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS)
            elapsed = time.monotonic() - started
            self.assertEqual(count, 1)
            self.assertEqual(child.wait(timeout=10), -signal.SIGKILL)
            self.assertLess(elapsed, 5.0)
        finally:
            release.set()
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_session_without_a_process_group_is_not_signalled(self) -> None:
        """shutdown_sessions: a fake session has no process group, so nothing is killed for it (no error either)."""
        session = self.live_session(9016)
        self.assertEqual(self.lifecycle.shutdown_sessions(budget=BUDGET_SECONDS, grace=STOP_SECONDS), 1)
        self.assertFalse(session.alive())


class LockTests(LifecycleCase):
    def test_second_daemon_lock_is_refused_until_the_first_is_released(self) -> None:
        """cmd_daemon flock: a second acquire of the lock is refused with 'already running'; after release it works."""
        path = Paths.from_home(self._home.home).lock
        first = acquire_lock(path)
        try:
            with self.assertRaises(SystemExit) as refused:
                acquire_lock(path)
            self.assertEqual(refused.exception.code, ALREADY_RUNNING)
        finally:
            first.close()
        second = acquire_lock(path)
        second.close()


class SignalTests(LifecycleCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def tearDown(self) -> None:
        for sig, handler in self._saved.items():
            signal.signal(sig, handler)
        super().tearDown()

    def test_sigterm_raises_system_exit_zero_and_logs_the_stop(self) -> None:
        """SIGTERM during a blocking poll: the handler logs 'daemon stop (signal 15)' and exits with code 0.

        Ported labels:
            SIGTERM during blocking poll exits cleanly (rc=0, took 0.03s)
            signal stop logged
        """
        install_stop_handlers(self.h.logger)
        with self.assertRaises(SystemExit) as stopped:
            os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("daemon stop (signal 15)", self.log_text())

    def test_ignore_stop_signals_makes_a_second_signal_harmless(self) -> None:
        """The shutdown path ignores SIGTERM and SIGINT, so a second signal cannot cut the shutdown short."""
        ignore_stop_signals()
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
        self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)


class RegisterCommandsTests(LifecycleCase):
    def test_register_commands_publishes_the_registry_menu_in_order(self) -> None:
        """register_commands: setMyCommands gets every word of the registry with its help, aliases included."""
        self.commands.register(Command("ping", "a test command", lambda ctx, args: None))
        self.commands.register(Command("stop", "stop the turn", lambda ctx, args: None,
                                       aliases=(Alias("halt", "stop it too"),)))
        self.assertTrue(self.lifecycle.register_commands())
        sent = [c for c in self.h.tg.calls if c.method == "setMyCommands"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(json.loads(sent[0].params["commands"]), [
            {"command": "ping", "description": "a test command"},
            {"command": "stop", "description": "stop the turn"},
            {"command": "halt", "description": "stop it too"},
        ])

    def test_register_commands_failure_is_logged_and_not_raised(self) -> None:
        """register_commands is best effort: a failed call is logged and the bridge runs on.

        Ported labels:
        N: a failing setMyCommands is logged, not raised
        """
        self.h.tg.fail_next("setMyCommands", code=400, description="Bad Request")
        self.assertFalse(self.lifecycle.register_commands())
        self.assertIn("setMyCommands failed", self.log_text())


class KvPrefixTests(LifecycleCase):
    def test_items_with_prefix_returns_only_matching_keys_in_key_order(self) -> None:
        """The start-up scan reads active:<thread> keys only: 'activex' and bare 'active' do not match."""
        kv: KvRepo = self.h.kv
        kv.set("active:9002", "b")
        kv.set("active:9001", "a")
        kv.set("activex:1", "no")
        kv.set("active", "no")
        kv.set("offset", "5")
        self.assertEqual(kv.items_with_prefix("active:"), [("active:9001", "a"), ("active:9002", "b")])


if __name__ == "__main__":
    unittest.main()
