"""A turn in a terminal session (spec T3): TurnRunner with an attach target, on the real ClaudeBackend whose tmux is a fake.

The message is typed into the pane (load-buffer, paste-buffer, Enter), the answer is read from the claim's transcript, and
the turn is written like any other. The session's uuid is the claim's: a fresh one is adopted into the topic's row. No
linger is offered, and a ⏹ (halt_turn) sends Escape without closing anything. The bot's ⏹ and /status are checked too.
A BlockingClock keeps the echo net and the interrupt fallback quiet, so only the transcript ends a turn.
"""

import json
import os
import subprocess
import threading
import time
import unittest

from tests.support import TempHome
from tests.test_cmd_session import THREAD as CMD_THREAD
from tests.test_cmd_session import SessionCase, StubSession
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, wait_until
from tg_bridge.agents.base import AttachSpec, always_attached
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.transcript_turns import ATTACH_RELOCATE_SECONDS
from tg_bridge.agents.claude.tui_turn import ATTACH_ECHO_MISSED_ERROR, USER_ECHO_SECONDS
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.bot.commands.info import ATTACHED_STATUS_TEXT, TERMINAL_STATUS_TEXT, _turn_line
from tg_bridge.core import terminal_claim
from tg_bridge.core.live import RunningTurn
from tg_bridge.core.turn_state import TurnOutcome

UUID = "11111111-2222-3333-4444-555555555555"
NEW_UUID = "66666666-7777-8888-9999-000000000000"
PANE = "%3"
TURN_DURATION_LINE = {"type": "system", "subtype": "turn_duration", "durationMs": 900, "messageCount": 2}
INTERRUPT_LINE = {"type": "user", "message": {"role": "user", "content": "[Request interrupted by user]"}}
TURN_WAIT = 10.0


def echo_line(text: str) -> dict:
    return {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": text}}


def assistant_line(text: str) -> dict:
    message = {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}],
               "usage": {"input_tokens": 3, "output_tokens": 2}}
    return {"type": "assistant", "isSidechain": False, "message": message}


SHORT_SLEEP_SECONDS = 2.0  # a sleep up to this long is real: the paste gap and the restore's polls


class BlockingClock:
    """Real time for now and monotonic; sleep() blocks until the test ends, so no timer fires by itself. A short sleep
    (a paste gap, a poll of the restore after an Escape) passes in real time, so the keys and the restore still run."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds <= SHORT_SLEEP_SECONDS:
            time.sleep(seconds)
            return
        self.release.wait(TURN_WAIT)


CLI_SCREEN = "─" * 40 + "\n❯ \n" + "─" * 40 + "\n  ? for shortcuts\n"  # the CLI's empty input box


class FakeTmux:
    """Stands in for tmux: the pane exists (pane_shown is its id; another id means the pane is gone), its screen is
    CLI_SCREEN, a paste runs on_paste, an Escape runs on_escape. Every call is recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, tuple[str, ...]]] = []
        self.on_paste = None
        self.on_escape = None
        self.pane_shown = PANE
        self.screen = CLI_SCREEN

    def __call__(self, args, timeout, text=None):
        args = list(args)
        sub = args[2] if args[:1] == ["-S"] else args[0]
        self.calls.append((sub, text, tuple(args)))
        out = self.pane_shown + "\n" if sub == "display-message" else self.screen if sub == "capture-pane" else ""
        if sub == "paste-buffer" and self.on_paste is not None:
            self.on_paste()
        if sub == "send-keys" and "Escape" in args and self.on_escape is not None:
            self.on_escape()
        return subprocess.CompletedProcess(args, 0, out, "")

    def typed(self) -> list[str]:
        """The subcommands that type into the pane, in order."""
        return [sub for sub, _text, _args in self.calls if sub in ("load-buffer", "paste-buffer", "send-keys")]

    def killed(self) -> bool:
        return any(sub.startswith("kill") for sub, _text, _args in self.calls)


class AttachedTurnCase(unittest.TestCase):
    def setUp(self) -> None:
        home = self.enterContext(TempHome()).home
        self.h = TurnHarness(home)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.cwd = os.path.join(home, "work")
        self.config_dir = os.path.join(home, "claude-config")
        os.makedirs(self.cwd)
        self.h.sessions.ensure(THREAD, TOPIC, "topic-cmd", self.cwd, "bypassPermissions")
        self.h.sessions.set_uuid(THREAD, UUID)
        self.clock = BlockingClock()
        self.addCleanup(self.clock.release.set)
        self.tmux = FakeTmux()
        self.tmux.on_paste = self.cli_answers
        self.backend = ClaudeBackend(logger=self.h.logger, clock=self.clock, tmux_run=self.tmux)
        self.registry = self.h.registry_for(self.backend)
        self.linger = self.h.linger(self.registry)
        self.runner = self.h.runner(self.registry, linger=self.linger)
        self.transcript = self.write_transcript(UUID)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def write_transcript(self, uuid: str) -> str:
        path = os.path.join(self.config_dir, "projects", history.encoded_cwd(os.path.realpath(self.cwd)), uuid + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        return path

    def cli_answers(self) -> None:
        """What the CLI writes when the pasted message 'hello' reaches it."""
        with open(self.transcript, "a", encoding="utf-8") as handle:
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

    def cli_interrupts(self) -> None:
        """What the CLI writes when Escape stops the turn."""
        with open(self.transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(INTERRUPT_LINE) + "\n")

    def attach(self, uuid: str = UUID, still_attached=always_attached) -> AttachSpec:
        return AttachSpec(socket="", pane=PANE, uuid=uuid, config_dir=self.config_dir, cwd=self.cwd,
                          still_attached=still_attached)

    def start_turn(self, text: str = "hello", uuid: str = UUID, still_attached=always_attached):
        box: list = []
        thread = threading.Thread(target=lambda: box.append(self.runner.run(
            thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text=text, sender="tester",
            attach=self.attach(uuid, still_attached))), daemon=True)
        thread.start()
        return thread, box


class AttachedTurnTests(AttachedTurnCase):
    def test_the_message_is_typed_into_the_pane_and_the_answer_comes_from_the_transcript(self) -> None:
        offers: list[object] = []
        self.linger.offer = lambda offer: offers.append(offer) or False  # the turn's end asks linger; it must not
        thread, box = self.start_turn()
        thread.join(TURN_WAIT)
        self.assertFalse(thread.is_alive(), "the attached turn did not end")
        self.assertEqual(box[0].answer, "ANSWER")
        self.assertEqual(self.tmux.typed(), ["load-buffer", "paste-buffer", "send-keys"], "not typed as one message")
        loaded = [text for sub, text, _args in self.tmux.calls if sub == "load-buffer"]
        self.assertEqual(loaded, ["hello"], "the pasted text is not the message")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, UUID)
        self.assertEqual(self.h.turns.totals(THREAD).turns, 1, "the turn has no turns row")
        self.assertEqual(offers, [], "the terminal's session was offered to linger")
        self.assertFalse(self.linger.holding(THREAD))
        self.assertIsNone(self.linger.status(THREAD))

    def test_a_fresh_session_started_in_the_tab_is_adopted_into_the_topic(self) -> None:
        self.transcript = self.write_transcript(NEW_UUID)
        thread, box = self.start_turn(uuid=NEW_UUID)
        thread.join(TURN_WAIT)
        self.assertEqual(box[0].answer, "ANSWER")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, NEW_UUID, "the fresh session was not stored")
        self.assertIsNotNone(self.h.history.find(THREAD, NEW_UUID), "the fresh session has no history row")

    def test_the_cli_gone_mid_turn_ends_the_turn_with_an_error(self) -> None:
        self.tmux.on_paste = None  # nothing answers, and the CLI then exits: its pane is gone
        thread, box = self.start_turn()
        self.assertTrue(wait_until(lambda: "paste-buffer" in self.tmux.typed()), "the message was not typed")
        self.tmux.pane_shown = "%9"
        for _ in range(200):  # the turn's dead-process timer runs on the harness clock: advance it until the turn ends
            self.h.clock.advance(1.0)
            thread.join(0.05)
            if not thread.is_alive():
                break
        self.assertFalse(thread.is_alive(), "the turn waited on a CLI that is gone")
        self.assertEqual(box[0].outcome, TurnOutcome.ERROR)

    def test_a_claim_released_mid_turn_ends_the_turn_with_an_error_within_seconds(self) -> None:
        """The CLI exits while the turn waits for its answer: the view pane releases the claim, the pane stays. The turn
        ends as an error after the dead-process delay (a few seconds of the harness clock), not at the turn's caps."""
        held = [True]
        self.tmux.on_paste = None  # nothing answers; the CLI exits instead
        thread, box = self.start_turn(still_attached=lambda: held[0])
        self.assertTrue(wait_until(lambda: self.tmux.typed().count("send-keys") == 1), "the message was not sent")
        time.sleep(0.05)  # the turn is in its wait now: the clock moves only from here on
        held[0] = False  # the claim is released; the pane still exists and shows its ready screen
        seconds = 0
        for _ in range(200):
            self.h.clock.advance(1.0)
            seconds += 1
            thread.join(0.05)
            if not thread.is_alive():
                break
        self.assertFalse(thread.is_alive(), "the turn waited on a CLI that already exited")
        self.assertLessEqual(seconds, 6, "the turn took too long to notice the CLI was gone")
        self.assertEqual(box[0].outcome, TurnOutcome.ERROR)

    def test_a_pane_that_is_not_ready_refuses_the_turn_without_a_reply_or_a_row(self) -> None:
        """The owner typed into the tab after the wait: nothing is pasted, no turns row is written, and the reply says
        the terminal was not ready, so the dispatcher waits for the terminal again."""
        self.tmux.screen = "─" * 40 + "\n❯ typed by the owner\n" + "─" * 40 + "\n  ? for shortcuts\n"
        reply = self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester",
                                attach=self.attach())
        self.assertTrue(reply.terminal_busy)
        self.assertIsNone(reply.answer)
        self.assertIsNone(reply.err)
        self.assertEqual(self.tmux.typed(), [], "a key was typed into the busy pane")
        self.assertEqual(self.h.turns.totals(THREAD).turns, 0, "a refused turn was recorded")
        self.assertIsNone(self.h.live.running(THREAD), "the refused turn is still running")

    def test_halt_turn_presses_escape_and_never_closes_the_terminal_session(self) -> None:
        self.tmux.on_paste = None  # nothing answers: the turn stays open until the interrupt
        self.tmux.on_escape = self.cli_interrupts
        thread, box = self.start_turn()
        self.assertTrue(wait_until(lambda: "paste-buffer" in self.tmux.typed()), "the message was not typed")
        running = self.h.live.request_abort(THREAD, stop_tap=True)
        self.assertIsNotNone(running, "no turn was running to stop")
        running.session.halt_turn()  # what the ⏹ tap and /stop do
        thread.join(TURN_WAIT)
        self.assertFalse(thread.is_alive(), "the stopped turn did not end")
        self.assertEqual(self.tmux.typed(), ["load-buffer", "paste-buffer", "send-keys", "send-keys"])
        keys = [args[-1] for sub, _text, args in self.tmux.calls if sub == "send-keys"]
        self.assertEqual(keys[-1], "Escape", "the stop did not press Escape")
        self.assertFalse(self.tmux.killed(), "the terminal session was closed")
        self.assertIn("session kept", box[0].err or "", "the reply says the session is closed")


class HaltRecorder(StubSession):
    """A running session whose ⏹ is halt_turn: a halt is counted apart from stop(), which the stub also counts."""

    def __init__(self) -> None:
        super().__init__()
        self.halts = 0

    def halt_turn(self) -> None:
        self.halts += 1


class BotLevelTests(SessionCase):
    """The ⏹ of /stop (and the other taps) ends an attached turn with halt_turn, not with stop()."""

    def test_stop_during_an_attached_turn_halts_it_and_does_not_stop_it(self) -> None:
        session = HaltRecorder()
        self.start_turn(session)
        self.commands.stop_now(CMD_THREAD)  # the session commands' topic (its own THREAD, not the harness's)
        self.assertTrue(wait_until(lambda: session.halts == 1), "the turn was not halted")
        self.assertEqual(session.stops, 0, "the attached turn's session was stopped")
        self.end_turn(session)

    def test_the_status_line_names_a_turn_in_the_terminal_session(self) -> None:
        running = RunningTurn(session=None, feed=None)  # the line reads neither the session nor the feed when terminal
        self.assertEqual(_turn_line(running, None, True), ATTACHED_STATUS_TEXT)
        self.assertEqual(ATTACHED_STATUS_TEXT, "turn: in the terminal session (tg-bridge view)")
        self.assertEqual(_turn_line(None, None, True), TERMINAL_STATUS_TEXT)


class RelocatingClock(BlockingClock):
    """The attached session's echo window (user-echo thread): the echo rounds (ATTACH_RELOCATE_SECONDS each) return at
    once and the round block_at blocks until the test releases it, so the three rounds of the 30 s window are checked
    without waiting for them."""

    def __init__(self, block_at: int = 3) -> None:
        super().__init__()
        self.echo_sleeps: list[float] = []
        self.block_at = block_at

    def sleep(self, seconds: float) -> None:
        if threading.current_thread().name != "user-echo":
            super().sleep(seconds)
            return
        self.echo_sleeps.append(seconds)
        if len(self.echo_sleeps) >= self.block_at:
            self.release.wait()


class StaleClaimTests(AttachedTurnCase):
    """A /clear or /resume in the tab moved the session to another transcript while the claim still names the old uuid:
    the echo is found in the new transcript, the turn follows it, and the claim and the topic take the new uuid."""

    def setUp(self) -> None:
        super().setUp()
        self.clock = RelocatingClock()
        self.addCleanup(self.clock.release.set)
        self.backend = ClaudeBackend(logger=self.h.logger, clock=self.clock, tmux_run=self.tmux)
        self.runner = self.h.runner(self.h.registry_for(self.backend), view_dir=self.view_dir())
        self.claim_held()

    def view_dir(self) -> str:
        path = os.path.join(self.h.home, "view")
        os.makedirs(path, exist_ok=True)
        return path

    def claim_held(self) -> None:
        self.assertTrue(terminal_claim.take_claim(self.view_dir(), THREAD, os.getpid(), time.time(), pane=PANE,
                                                  uuid=UUID, config_dir=self.config_dir, cwd=self.cwd, socket=""))

    def start(self):
        box: list = []
        spec = AttachSpec(socket="", pane=PANE, uuid=UUID, config_dir=self.config_dir, cwd=self.cwd,
                          pid=os.getpid())
        thread = threading.Thread(target=lambda: box.append(self.runner.run(
            thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester", attach=spec)), daemon=True)
        thread.start()
        return thread, box

    def test_a_stale_claim_uuid_is_followed_to_the_session_the_tab_moved_to(self) -> None:
        self.transcript = self.write_transcript(NEW_UUID)  # the CLI writes the message's echo into the new session
        thread, box = self.start()
        thread.join(TURN_WAIT)
        self.assertFalse(thread.is_alive(), "the turn did not end in the new transcript")
        self.assertEqual(box[0].answer, "ANSWER")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, NEW_UUID, "the topic kept the stale uuid")
        self.assertEqual(terminal_claim.read_claim(self.view_dir(), THREAD).uuid, NEW_UUID,
                         "the claim still names the stale uuid")
        self.assertEqual(self.h.turns.totals(THREAD).turns, 1)

    def test_with_no_transcript_holding_the_message_the_turn_ends_with_the_echo_error_at_thirty_seconds(self) -> None:
        self.tmux.on_paste = None  # nothing is written: the message never reaches a transcript
        thread, box = self.start()
        self.assertTrue(wait_until(lambda: len(self.clock.echo_sleeps) == 3), "the third echo round did not start")
        self.assertEqual(self.clock.echo_sleeps, [10.0, 10.0, 10.0], "the echo window is not 30 s in three rounds")
        self.assertTrue(thread.is_alive(), "the turn ended before the echo window")
        self.clock.release.set()
        thread.join(TURN_WAIT)
        self.assertFalse(thread.is_alive())
        self.assertEqual(box[0].outcome, TurnOutcome.ERROR)
        self.assertIn(ATTACH_ECHO_MISSED_ERROR.format(session=UUID, seconds=USER_ECHO_SECONDS), box[0].err)
        self.assertEqual(self.h.sessions.get(THREAD).uuid, UUID, "the topic moved without a transcript")

