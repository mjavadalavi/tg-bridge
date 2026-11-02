"""Linger end to end (C9c): a kept session from the inbound message to the 🔔 reply, with the fake CLIs as child processes.

The stack is the one the App builds: the default backend registry, a named launcher picked for the topic (the print
transport, the tui transport, or the Codex fake), the real repositories, PermissionService, LingerService with a real
SelfTurns, the SessionRouter and TurnRunner, and the real Dispatcher whose runner thread holds the topic. Only the
Telegram client, the feed (a recorder) and the clock are doubles. The clock is the FakeClock: a test moves it, so the
quiet window and the cap are reached without waiting, and the hold loop ticks in real time under it.

Background work is kept open by a fake whose end the test controls: a flag file (FAKE_BG_FLAG / FAKE_TUI_BG_FLAG) releases
it, and the seconds are set high so only the flag, the cap, the quiet window or a stop ends it.
"""

import os
import re
import sys
import threading
import unittest

import tests.fakes as fakes_package
from tests.fakes.clock import FakeClock
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, wait_until
from tg_bridge.agents.claude.tui_turn import USER_ECHO_SECONDS
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import CommandRegistry
from tg_bridge.bot.commands.info import LINGER_STATUS_TEXT, _turn_line
from tg_bridge.bot.commands.session import SessionCommands, StopKind
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import LINGER_CAP_TEXT, linger_key
from tg_bridge.core.topic_queues import Admission, Delivery
from tg_bridge.core.turn_text import BACKGROUND_PREFIX
from tg_bridge.daemon.lifecycle import LINGER_RESTART_NOTICE, Lifecycle
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.telegram.answers import AnswerSender

FAKES = os.path.dirname(os.path.abspath(fakes_package.__file__))
FAKE_CLAUDE = os.path.join(FAKES, "fake_claude.py")
FAKE_TUI = os.path.join(FAKES, "fake_claude_tui.py")
FAKE_CODEX = os.path.join(FAKES, "fake_codex.py")
SENDER = "owner"
WAIT = 30.0  # real seconds a test waits for a child process or a thread
BELL = BACKGROUND_PREFIX  # the first line of every reply of a self turn
PID = re.compile(r"pid=(\d+)")


class HeldEchoClock(FakeClock):
    """A FakeClock whose message watch (USER_ECHO_SECONDS, the TUI's echo of a typed message) is held until teardown. A
    live TUI reads the typed line within a second; a FakeClock would end every TUI turn at once, before the fake reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.echo_release = threading.Event()

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if seconds == USER_ECHO_SECONDS:
            self.echo_release.wait()


class NoTitles:
    """Stands in for TitleService: auto-titles are off in these tests."""

    def submit(self, *args: object) -> None:
        pass


def launcher_config(name: str, home: str) -> dict:
    """The config keys of one launcher: the print and tui fakes as Claude launchers, the Codex fake as an agent."""
    if name == "fakecodex":
        return {"agent_launchers": {name: {"backend": "codex", "command": f"{sys.executable} -I {FAKE_CODEX}",
                                           "config_dir": home}}}
    fake, transport = (FAKE_TUI, "tui") if name == "tui" else (FAKE_CLAUDE, "print")
    return {"launchers": {name: {"cmd": f"{sys.executable} -I {fake}", "config_dir": home, "transport": transport}}}


class LingerEndToEnd(unittest.TestCase):
    """One topic, one kept session at a time. start() builds the stack for a launcher; the tests drive it with send()."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        clock = HeldEchoClock()
        self.addCleanup(clock.echo_release.set)
        self.h = TurnHarness(self.home.home, poll_seconds=0.005, clock=clock)
        self.flag = os.path.join(self.home.state, "bg.flag")  # the file that ends the fake's background work
        self.inbox = InboxRepo(self.h.db, self.h.clock)
        self.started: list[threading.Thread] = []
        self.addCleanup(self.h.db.close)  # registered first, so it runs after the teardown below
        self.addCleanup(self._teardown)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "default")

    def _teardown(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)
        for thread in self.started:
            thread.join(WAIT)

    def _thread(self, **kwargs: object) -> threading.Thread:
        """The dispatcher's thread factory: records every runner thread so the teardown can join it."""
        thread = threading.Thread(**kwargs)  # type: ignore[arg-type]
        self.started.append(thread)
        return thread

    # the stack

    def start(self, launcher: str, env: dict[str, str] | None = None, *, linger_seconds: float = 120,
              max_sessions: int = 3) -> None:
        """Builds the registry, the linger service, the runner and the dispatcher for a launcher of this topic."""
        self.h.base_env = {"PATH": os.environ.get("PATH", ""), "FAKE_TUI_CONFIG_DIR": self.home.home, **(env or {})}
        self.h.set_config(**launcher_config(launcher, self.home.home), linger_seconds=linger_seconds,
                          linger_max_sessions=max_sessions)
        self.h.sessions.set_launcher_pick(THREAD, launcher)
        self.registry = build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher="")
        self.linger = self.h.linger(self.registry)
        self.runner = self.h.runner(self.registry, linger=self.linger)
        self.dispatcher = Dispatcher(
            turns=self.runner, slots=self.h.slots, inbox=self.inbox, messenger=self.h.messenger,
            answers=AnswerSender(self.h.tg, FakeUploader(), self.h.messenger, self.h.logger, self.h.clock),
            titles=NoTitles(), client=self.h.tg, live=self.h.live, config=self.h.config, logger=self.h.logger,
            start_thread=self._thread, linger=self.linger)

    def print_background(self, kind: str, **extra: str) -> dict[str, str]:
        """The fake print CLI's background work: it starts in the first turn and ends when the flag file exists."""
        return {"FAKE_BG": kind, "FAKE_BG_FLAG": self.flag, "FAKE_BG_SECONDS": "600", **extra}

    def tui_background(self, scenario: str) -> dict[str, str]:
        """The fake TUI's background work, the same way: it ends when the flag file exists (not after the seconds)."""
        return {"FAKE_TUI_SCENARIO": scenario, "FAKE_TUI_BG_FLAG": self.flag, "FAKE_TUI_BG_SECONDS": "600"}

    def commands(self) -> SessionCommands:
        """The /stop command of the bot, on the same services as the runner."""
        return SessionCommands(
            config=self.h.config, sessions=self.h.sessions, history=self.h.history, live=self.h.live,
            backends=self.registry, messenger=self.h.messenger, clock=self.h.clock, logger=self.h.logger,
            handoff=HandoffService(self.h.kv, self.registry, self.h.logger), linger=self.linger)

    # the topic

    def send(self, text: str, update_id: int) -> Delivery:
        """A message from the owner, as the inbound handler delivers it to the dispatcher."""
        self.assertTrue(self.inbox.ingest(update_id, {"update_id": update_id}, thread_id=THREAD, chat_id=CHAT,
                                          sender=SENDER))
        return self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, update_id, text)

    def texts(self) -> list[str]:
        return self.h.tg.texts(THREAD)

    def wait_text(self, fragment: str) -> bool:
        return wait_until(lambda: any(fragment in text for text in self.texts()), WAIT)

    def kept(self):
        return self.h.live.lingering(THREAD)

    def wait_kept(self, pending: int) -> None:
        """Waits until the topic's session is kept with this many background tasks pending and its user turn has ended
        (the record is made before the turn leaves the live registry, so both are checked)."""
        self.assertTrue(wait_until(lambda: self.linger.status(THREAD) is not None and self.h.live.running(THREAD) is None,
                                   WAIT), "the session was not kept")
        self.assertEqual(self.linger.status(THREAD).pending, pending)

    def release_work(self) -> None:
        """Ends the fake's background work: the flag file exists, the fake's thread writes its end."""
        open(self.flag, "w", encoding="utf-8").close()

    def advance_until(self, predicate, limit: int = 90) -> bool:
        """Moves the FakeClock a second at a time until predicate() holds. The hold loop ticks in real time meanwhile."""
        for _ in range(limit):
            if wait_until(predicate, 0.2):
                return True
            self.h.clock.advance(1.0)
        return wait_until(predicate, 0.2)

    def status_turn_line(self) -> str:
        """The turn line of /status for this topic, from the same function the command renders it with."""
        return _turn_line(self.h.live.running(THREAD), self.linger.status(THREAD))

    def assertGone(self, session) -> None:
        """The process of a closed session has exited."""
        self.assertTrue(wait_until(lambda: not session.alive(), WAIT), "the kept process is still running")

    def turn_rows(self) -> int:
        return self.h.db.query("SELECT COUNT(*) AS n FROM turns WHERE thread_id=?", (THREAD,))[0]["n"]

    def assertNotKept(self) -> None:
        self.assertIsNone(self.kept())
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertFalse(self.linger.holding(THREAD))


class PrintLingerTests(LingerEndToEnd):
    """The print transport (stream-json): the fake CLI reports its background work on stdout."""

    def test_print_bash_background_reports_back_with_the_bell(self) -> None:
        self.start("print", self.print_background("bash"))
        self.assertEqual(self.send("run it in the background", 1).admission, Admission.STARTED)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.wait_text("STARTED"))
        # /status while the session is kept: the FakeClock has not moved, so the window is the full linger_seconds (120)
        self.assertEqual(self.linger.status(THREAD).seconds_left, 120)
        self.assertEqual(self.status_turn_line(), LINGER_STATUS_TEXT.format(pending=1, minutes=2))
        self.release_work()
        self.assertTrue(self.wait_text(f"{BELL}\n\nFINISHED"))
        self.assertTrue(self.advance_until(lambda: self.kept() is None))
        self.assertGone(session)
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertEqual(self.turn_rows(), 2, "the user turn and the self turn each have a turns row")

    def test_print_message_during_linger_reaches_the_same_process(self) -> None:
        self.start("print", self.print_background("bash", FAKE_PID_ANSWER="1"))
        self.assertEqual(self.send("start", 1).admission, Admission.STARTED)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.wait_text("STARTED"))
        delivery = self.send("again", 2)
        self.assertIs(delivery.admission, Admission.HANDED, "a message to an idle kept session is handed over quietly")
        self.assertTrue(self.wait_text("ok:2"))
        pids = {PID.search(text).group(1) for text in self.texts() if PID.search(text)}
        self.assertEqual(len(pids), 1, "both answers come from one process")
        self.assertIs(self.kept(), session)

    def test_print_stop_closes_the_kept_session(self) -> None:
        self.start("print", self.print_background("bash"))
        self.send("start", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        result = self.commands().stop_turn(THREAD)
        self.assertIs(result.kind, StopKind.RELEASED)
        self.assertTrue(wait_until(lambda: self.kept() is None, WAIT))
        self.assertGone(session)
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_print_cap_closes_with_notice(self) -> None:
        self.start("print", self.print_background("bash"), linger_seconds=5)
        self.send("start", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.advance_until(lambda: self.kept() is None), "the cap did not close the session")
        self.assertGone(session)
        self.assertTrue(self.wait_text(LINGER_CAP_TEXT.split("{")[0]), "the cap notice was not posted")  # posted after the close

    def test_print_wakeup_reports_back(self) -> None:
        self.start("print", self.print_background("wakeup"))
        self.send("wake me later", 1)
        self.wait_kept(pending=1)  # the ScheduleWakeup is counted from its tool result
        session = self.kept()
        self.release_work()
        self.assertTrue(self.wait_text(f"{BELL}\n\nAWAKE"))
        self.assertTrue(self.advance_until(lambda: self.kept() is None))
        self.assertGone(session)

    def test_print_background_subagent_ask_gets_a_card_while_idle(self) -> None:
        self.start("print", self.print_background("agent", FAKE_BG_ASK="1"))
        self.send("delegate it", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.wait_text("STARTED"))
        self.release_work()  # the subagent asks for a permission while the main session is idle
        self.assertTrue(wait_until(lambda: bool(self.h.asks.pending(THREAD)), WAIT), "no ask reached the topic")
        self.assertTrue(wait_until(lambda: bool(self.h.tg.buttons(THREAD)), WAIT), "the ask has no card")
        nonce = self.h.asks.pending(THREAD)[0].nonce
        self.assertTrue(self.h.permissions.answer_ask(nonce, True))
        self.assertTrue(self.wait_text(f"{BELL}\n\nFINISHED"))
        self.assertTrue(self.advance_until(lambda: self.kept() is None))
        self.assertGone(session)

    def test_print_quick_background_end_reports_back_with_the_bell(self) -> None:
        # the fake ends its work before the first result and starts the self turn right after it: the self turn begins
        # while the user turn is still ending, and its 🔔 reply must still come
        self.start("print", {"FAKE_BG": "bash", "FAKE_BG_QUICK": "1"})
        self.assertEqual(self.send("go", 1).admission, Admission.STARTED)
        self.assertTrue(self.wait_text(f"{BELL}\n\nFINISHED"), "the self turn's 🔔 reply never came")
        self.assertTrue(self.advance_until(lambda: self.kept() is None))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))


class TuiLingerTests(LingerEndToEnd):
    """The tui transport: the fake TUI writes the transcript lines, the bridge tails them."""

    def test_tui_bash_background_reports_back_with_the_bell(self) -> None:
        self.start("tui", self.tui_background("bg_bash"))
        self.send("run it in the background", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.wait_text("STARTED"))
        self.release_work()
        self.assertTrue(self.wait_text(f"{BELL}\n\nFINISHED"))
        self.assertTrue(self.advance_until(lambda: self.kept() is None))
        self.assertGone(session)
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_tui_message_during_linger_reaches_the_same_session(self) -> None:
        self.start("tui", self.tui_background("bg_bash"))
        self.send("start", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertTrue(self.wait_text("STARTED"))
        self.assertIs(self.send("again", 2).admission, Admission.HANDED)
        self.assertTrue(wait_until(lambda: sum("STARTED" in text for text in self.texts()) == 2, WAIT),
                        "the second message got no answer")
        self.assertIs(self.kept(), session, "the second message reached the kept session, not a new one")

    def test_tui_wakeup_then_close_falls_back_to_stop(self) -> None:
        # the deaf TUI ignores the first /exit typed while the wakeup is pending: the signal ladder must end it
        self.start("tui", self.tui_background("bg_wakeup") | {"FAKE_TUI_EXIT_DEAF": "1"})
        self.send("wake me later", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        result = self.commands().stop_turn(THREAD)
        self.assertIs(result.kind, StopKind.RELEASED)
        self.assertGone(session)
        self.assertTrue(wait_until(lambda: self.kept() is None, WAIT))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))


class CodexNeverLingersTests(LingerEndToEnd):
    def test_codex_never_lingers(self) -> None:
        self.start("fakecodex", {"FAKE_CODEX_MODE": "normal", "FAKE_CODEX_ANSWER": "codex answer"})
        self.send("hi", 1)
        self.assertTrue(self.wait_text("codex answer"))
        for thread in self.started:  # the runner ends at once: nothing is held
            thread.join(WAIT)
        self.assertNotKept()
        self.assertFalse(any(text.startswith(LINGER_CAP_TEXT.split("{")[0]) for text in self.texts()))


class RestartMarkerTests(LingerEndToEnd):
    def lifecycle(self) -> Lifecycle:
        return Lifecycle(live=self.h.live, permissions=self.h.permissions, kv=self.h.kv, messenger=self.h.messenger,
                         telegram=self.h.tg, commands=CommandRegistry(), clock=self.h.clock, logger=self.h.logger)

    def test_restart_marker_survives_shutdown(self) -> None:
        self.start("print", self.print_background("bash"))
        self.send("start", 1)
        self.wait_kept(pending=1)
        session = self.kept()
        self.assertEqual(self.lifecycle().shutdown_sessions(grace=1), 1)
        self.assertGone(session)
        # the hold tick that sees the shutdown releases the record: no stop and no removal of the marker. The runner
        # may have made that tick already, so the test makes it too (then the record is simply gone).
        self.linger.wait_once(THREAD)
        self.assertFalse(self.linger.holding(THREAD))
        self.assertIsNotNone(self.h.kv.get(linger_key(THREAD)), "the shutdown must leave the marker for the notice")
        recovery = self.lifecycle().recover()
        self.assertEqual(recovery.lingers, 1)
        self.assertIn(LINGER_RESTART_NOTICE, self.texts())
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))


if __name__ == "__main__":
    unittest.main()
