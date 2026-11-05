"""AttachedSession (agents/claude/attached_session.py) with a fake tmux and a temp transcript: a turn read from the file the
paste writes, the failed paste, Escape for an interrupt, and stop() that never sends a key."""

import dataclasses
import json
import os
import subprocess
import threading
import time
import unittest
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tests.turn_support import wait_until
from tg_bridge.agents.base import AttachSpec, SessionSpec, TerminalNotReady
from tg_bridge.agents.claude import attached_session, history
from tg_bridge.agents.claude.attached_session import (ATTACHED_CAPABILITIES, ESCAPE_REPEAT_SECONDS, RESTORE_CAP_SECONDS,
                                                     RESTORE_JOIN_SECONDS, RESTORE_POLL_SECONDS, RESTORE_WAIT_SECONDS,
                                                     UNREACHABLE_ERROR, AttachedSession, _still_ours, is_restored,
                                                     pane_reason)
from tg_bridge.agents.claude.tmux_pane import (CLEAR_KEY_GAP_SECONDS, REASON_GONE, REASON_IDLE, REASON_TYPED,
                                               input_empty)
from tg_bridge.agents.claude.transcript_turns import ATTACH_QUEUED_MAX_SECONDS, ATTACH_RELOCATE_SECONDS
from tg_bridge.agents.claude.tui_turn import ATTACH_ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE, USER_ECHO_SECONDS
from tg_bridge.agents.events import (ApprovalRequest, Decision, SessionStarted, TextEvent, ToolStart,
                                    TurnBegan, TurnResult, TurnStatus)
from tg_bridge.config import LauncherSpec

UUID = "11111111-2222-3333-4444-555555555555"
PANE = "%3"
CLI_BOX = "─" * 40 + "\n❯ \n" + "─" * 40 + "\n  ? for shortcuts\n"  # the CLI's empty input box
TURN_DURATION_LINE = {"type": "system", "subtype": "turn_duration", "durationMs": 900, "messageCount": 2}
WAIT_SECONDS = 10.0


def echo_line(text: str) -> dict:
    return {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": text}}


def assistant_line(text: str) -> dict:
    message = {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}],
               "usage": {"input_tokens": 3, "output_tokens": 2}}
    return {"type": "assistant", "isSidechain": False, "message": message}


class BlockingClock:
    """sleep() blocks until the test ends on the other threads, so no echo net or interrupt fallback fires during a test
    by itself. The test's own thread (the gap after a paste) does not wait."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if threading.current_thread() is threading.main_thread():
            return
        self.release.wait(WAIT_SECONDS)


class FakeTmux:
    """Stands in for tmux. display-message answers with the pane id, capture-pane with screen, and paste-buffer runs on_paste
    (the CLI writing its echo and answer). send-keys records its key in keys_sent and runs on_keys(key), which may
    change the screen. The first Escape also puts after_escape up (the CLI stops working once it is interrupted). The
    subcommands named in fail exit 1."""

    def __init__(self, *, screen: str = CLI_BOX, fail: tuple[str, ...] = (), on_paste=None, on_keys=None) -> None:
        self.pane_shown = PANE  # the pane tmux shows under this id; another id means the pane is gone
        self.screen = screen
        self.after_escape: str | None = None  # the screen once an Escape is sent; None keeps the screen
        self.fail = fail
        self.on_paste = on_paste
        self.on_keys = on_keys
        self.calls: list[tuple[str, str | None]] = []
        self.keys_sent: list[str] = []

    def __call__(self, args, timeout, text=None):
        args = list(args)
        sub = args[2] if args[:1] == ["-S"] else args[0]
        self.calls.append((sub, text))
        if sub in self.fail:
            return subprocess.CompletedProcess(args, 1, "", "")
        out = ""
        if sub == "display-message":
            out = self.pane_shown + "\n"
        elif sub == "capture-pane":
            out = self.screen
        elif sub == "paste-buffer" and self.on_paste is not None:
            self.on_paste()
        elif sub == "send-keys":
            self.keys_sent.append(args[-1])
            if args[-1] == "Escape" and self.after_escape is not None:
                self.screen, self.after_escape = self.after_escape, None
            if self.on_keys is not None:
                self.on_keys(args[-1])
        return subprocess.CompletedProcess(args, 0, out, "")

    def keys(self) -> list[str]:
        """The subcommands that type into the pane."""
        return [sub for sub, _text in self.calls if sub in ("load-buffer", "paste-buffer", "send-keys")]


def box_screen(text: str) -> str:
    """The CLI's screen with text in its input box."""
    return "─" * 40 + "\n❯ " + text + "\n" + "─" * 40 + "\n  ? for shortcuts\n"


def box_rows(*rows: str) -> str:
    """The CLI's screen with a multi-line input box: the first row on the prompt line, the rest indented below it."""
    body = [f"❯ {rows[0]}", *(f"  {row}" for row in rows[1:])]
    return "\n".join(["─" * 40, *body, "─" * 40, "  ? for shortcuts", ""])


def busy_screen(text: str) -> str:
    """The CLI's screen while it works: the busy hint above its input box, with text in the box."""
    return "✻ Thinking… (esc to interrupt)\n" + box_screen(text)


class RestoreClock(FakeClock):
    """The restore runs in the test's own thread (its spawn is synchronous), so its polls are the sleeps of the main
    thread: they move the fake time at once and are recorded. Every other thread (the echo net, the interrupt fallback)
    blocks until the test ends, so none of them fires by itself."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def sleep(self, seconds: float) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.release.wait(WAIT_SECONDS)
            return
        super().sleep(seconds)


class PollClock(RestoreClock):
    """RestoreClock for a restore run on the test's own thread (its spawn defers it to run_restores). Before each poll's
    sleep, on_poll(the sleeps so far) may change the screen, so a test says what the CLI shows while the window runs."""

    def __init__(self) -> None:
        super().__init__()
        self.on_poll = lambda polls: None

    def sleep(self, seconds: float) -> None:
        if threading.current_thread() is threading.main_thread():
            self.on_poll(len(self.sleeps))
        super().sleep(seconds)


class GatedRestoreClock(RestoreClock):
    """RestoreClock for a restore on a thread of its own ('restored-prompt', see threaded_spawn): each of its polls
    waits for gate first, so a test decides when the restore goes on. The other threads are RestoreClock's."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    def sleep(self, seconds: float) -> None:
        if threading.current_thread().name == "restored-prompt":
            self.gate.wait(WAIT_SECONDS)
            FakeClock.sleep(self, seconds)  # the poll moves the fake time on, as any sleep does
            return
        super().sleep(seconds)


def threaded_spawn(work) -> None:
    """The production spawn: the restore runs on a thread of its own."""
    threading.Thread(target=work, name="restored-prompt", daemon=True).start()


class NamedGateClock(RestoreClock):
    """RestoreClock for several restores at once: the restore thread named n waits at its polls for gates[n], so a test
    lets one restore go on while another still waits. Threads not named in gates are RestoreClock's."""

    def __init__(self, names) -> None:
        super().__init__()
        self.gates = {name: threading.Event() for name in names}

    def sleep(self, seconds: float) -> None:
        gate = self.gates.get(threading.current_thread().name)
        if gate is not None:
            gate.wait(WAIT_SECONDS)
            FakeClock.sleep(self, seconds)
            return
        super().sleep(seconds)


def named_spawn(names):
    """A spawn that starts each restore on a thread of its own, named by the next of names (in the order they start)."""
    pending = iter(names)

    def spawn(work) -> None:
        threading.Thread(target=work, name=next(pending), daemon=True).start()
    return spawn


class AttachedSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        home = self.enterContext(TempHome()).home
        self.cwd = os.path.join(home, "work")
        self.config_dir = os.path.join(home, "config")
        os.makedirs(self.cwd)
        self.transcript = os.path.join(self.config_dir, "projects", history.encoded_cwd(os.path.realpath(self.cwd)),
                                       UUID + ".jsonl")
        os.makedirs(os.path.dirname(self.transcript))
        open(self.transcript, "w", encoding="utf-8").close()  # the session exists; its turns are appended to it
        self.clock = BlockingClock()
        self.addCleanup(self.clock.release.set)
        self.events: list[object] = []
        self.logs: list[str] = []
        self.restores: list = []  # the restore work the deferred spawn kept, until run_restores runs it
        self.tmux = FakeTmux(on_paste=self.cli_answers)
        self.spec = SessionSpec(thread_id=1, chat_id=2, topic="t", launcher=LauncherSpec(
            name="l", backend="claude", command="c", argv=("c",), config_dir=self.config_dir), cwd=self.cwd,
            mode="default", model=None, resume_id=UUID, new_id=None, allow_rules=(), deny_tools=(), system_rules="",
            env={}, errfile_dir=self.cwd)

    def make(self, still_attached=None, spawn=None) -> AttachedSession:
        session = AttachedSession(self.spec, self.events.append, socket="", pane=PANE, uuid=UUID,
                                  config_dir=self.config_dir, cwd=self.cwd, logger=self, clock=self.clock,
                                  run=self.tmux, still_attached=still_attached or (lambda: True),
                                  spawn=spawn or self.defer_restore)
        self.addCleanup(session.stop)
        self.addCleanup(self.run_restores)  # runs before stop (LIFO): a deferred restore must not hold stop() up
        return session

    def defer_restore(self, work) -> None:
        """The restore's spawn in most tests: the work is kept. A restore takes _keys, and the Escape that starts it
        holds _keys, so the work runs after interrupt() returned: run_restores() does that on the test's own thread."""
        self.restores.append(work)

    def run_restores(self) -> None:
        while self.restores:
            self.restores.pop(0)()

    def interrupt_and_restore(self, session: AttachedSession) -> bool:
        sent = session.interrupt()
        self.run_restores()
        return sent

    def log(self, msg: object) -> None:  # the Logger protocol of the session
        self.logs.append(str(msg))

    def cli_answers(self) -> None:
        """What the CLI writes to the transcript when the pasted message 'hello' reaches it."""
        with open(self.transcript, "a", encoding="utf-8") as handle:
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

    def results(self) -> list[TurnResult]:
        return [e for e in self.events if isinstance(e, TurnResult)]

    def wait_for_result(self) -> TurnResult:
        deadline = time.monotonic() + WAIT_SECONDS
        while not self.results() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(self.results()), 1, "no turn result came from the transcript")
        return self.results()[0]

    def wait_restores(self, session: AttachedSession, count: int) -> bool:
        """True when the running restores of session reach count within WAIT_SECONDS."""
        with session._restores_idle:
            return session._restores_idle.wait_for(lambda: session._restores == count, WAIT_SECONDS)

    def test_start_reports_the_session_as_resumed_when_the_pane_exists(self) -> None:
        session = self.make()
        session.start()
        self.assertEqual(self.events, [SessionStarted(session_id=UUID, model=None, mode="default", cwd=self.cwd,
                                                      resumed=True)])
        self.assertTrue(session.alive())
        self.assertEqual(self.tmux.keys(), [])  # nothing is typed by start

    def test_start_raises_when_the_pane_is_gone(self) -> None:
        self.tmux.fail = ("display-message",)
        session = self.make()
        with self.assertRaises(TerminalNotReady):
            session.start()
        self.assertFalse(session.alive())
        self.assertEqual(self.events, [])

    def test_a_turn_is_read_from_the_transcript_the_paste_writes(self) -> None:
        session = self.make()
        session.start()
        session.send_user("hello")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertIn(("load-buffer", "hello"), self.tmux.calls)
        self.assertEqual([sub for sub, _ in self.tmux.calls if sub in ("paste-buffer", "send-keys")],
                         ["paste-buffer", "send-keys"])

    def test_a_message_delivered_mid_turn_as_a_queued_command_completes_with_the_answer(self) -> None:
        def cli_delivers_it_mid_turn() -> None:  # no user line: the message comes as an attachment of the busy turn
            with open(self.transcript, "a", encoding="utf-8") as handle:
                for line in (queued_command_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                    handle.write(json.dumps(line) + "\n")

        self.tmux.on_paste = cli_delivers_it_mid_turn
        session = self.make()
        session.start()
        session.send_user("hello")

        result = self.wait_for_result()  # the echo net is blocked by the clock: only the attachment can end the wait
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))

    def test_a_task_notification_queued_during_the_wait_does_not_mark_the_turn_queued(self) -> None:
        notification = {"type": "queue-operation", "operation": "enqueue", "sessionId": UUID,
                        "content": "<task-notification><task-id>b1</task-id></task-notification>"}

        def cli_queues_a_notification_first() -> None:  # a background task finishes while the pasted message waits
            with open(self.transcript, "a", encoding="utf-8") as handle:
                for line in (notification, echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                    handle.write(json.dumps(line) + "\n")

        self.tmux.on_paste = cli_queues_a_notification_first
        session = self.make()
        session.start()
        session.send_user("hello")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertFalse(session._turns._turn.queued, "a background task's queue line marked the message queued")

    def test_a_failed_paste_ends_the_turn_with_an_error_and_presses_no_enter(self) -> None:
        self.tmux.fail = ("paste-buffer",)
        session = self.make()
        session.start()
        session.send_user("hello")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.error), (TurnStatus.ERROR, UNREACHABLE_ERROR))
        self.assertNotIn("send-keys", [sub for sub, _ in self.tmux.calls])

    def test_an_earlier_turn_s_answer_and_duration_do_not_finish_the_pasted_turn(self) -> None:
        def earlier_turn_ends_after_the_paste() -> None:  # the owner's turn in the tab, closed while the message is pasted
            with open(self.transcript, "a", encoding="utf-8") as handle:
                for line in (assistant_line("EARLIER"), TURN_DURATION_LINE):
                    handle.write(json.dumps(line) + "\n")

        self.tmux.on_paste = earlier_turn_ends_after_the_paste
        session = self.make()
        session.start()
        session.send_user("hello")
        with open(self.transcript, "a", encoding="utf-8") as handle:  # the pasted message's own turn, read after those
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))

    def test_an_earlier_turn_s_lines_are_not_sent_to_the_topic(self) -> None:
        earlier_text = assistant_line("EARLIER")
        earlier_tool = {"type": "assistant", "isSidechain": False,
                        "message": {"id": "msg_2", "type": "message", "role": "assistant",
                                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                                 "input": {"command": "ls"}}]}}

        def earlier_turn_keeps_working_while_the_paste_waits() -> None:  # its text and tool use, before the echo
            with open(self.transcript, "a", encoding="utf-8") as handle:
                for line in (earlier_text, earlier_tool):
                    handle.write(json.dumps(line) + "\n")

        self.tmux.on_paste = earlier_turn_keeps_working_while_the_paste_waits
        session = self.make()
        session.start()
        session.send_user("hello")
        with open(self.transcript, "a", encoding="utf-8") as handle:  # the pasted message's own turn, read after those
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        sent = [e for e in self.events if isinstance(e, (TextEvent, ToolStart))]
        self.assertNotIn("EARLIER", [e.text for e in sent if isinstance(e, TextEvent)])
        self.assertFalse(any(isinstance(e, ToolStart) for e in sent), sent)
        self.assertIn("ANSWER", [e.text for e in sent if isinstance(e, TextEvent)])

    def test_a_turn_the_tab_starts_itself_reaches_the_topic_and_ends(self) -> None:
        notification = {"type": "user", "origin": {"kind": "task-notification"},
                        "message": {"role": "user", "content": "<task-notification><task-id>b1</task-id>"
                                                               "</task-notification>"}}
        session = self.make()
        session.start()
        session.send_user("hello")
        self.wait_for_result()  # the owner's turn ends; the tail of the session keeps following the transcript
        with open(self.transcript, "a", encoding="utf-8") as handle:  # then the tab starts a turn by itself
            for line in (notification, assistant_line("FINISHED"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

        deadline = time.monotonic() + WAIT_SECONDS
        while len(self.results()) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual([(r.status, r.answer) for r in self.results()][1:], [(TurnStatus.OK, "FINISHED")],
                         "the self turn never ended")
        self.assertIn("FINISHED", [e.text for e in self.events if isinstance(e, TextEvent)])

    def test_images_are_refused(self) -> None:
        session = self.make()
        session.start()
        with self.assertRaises(ValueError):
            session.send_user("hello", images=("/tmp/a.png",))

    def test_interrupt_presses_escape_in_the_pane(self) -> None:
        session = self.make()
        session.start()
        self.tmux.screen = busy_screen("")  # the CLI works: its spinner shows
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.calls[-1], ("send-keys", None))

    def ctrl_u_clears_the_line(self, key: str) -> None:
        """What the CLI's box does with C-u: its line is emptied."""
        if key == "C-u":
            self.tmux.screen = CLI_BOX

    def paste_then_escape_with_the_turn_open(self, screen: str, *, text: str = "hello", clock=None,
                                             spawn=None) -> tuple[AttachedSession, int]:
        """Pastes text while the CLI has not answered; the CLI works (busy_screen) until the Escape, and shows screen
        once the Escape is sent. Returns the session and the count of keys sent before the interrupt.
        clock and spawn are the session's (the RestoreClock and the deferred spawn by default)."""
        self.tmux.on_paste = None  # the CLI answers later: the Escape comes while the turn is open
        self.tmux.on_keys = self.ctrl_u_clears_the_line
        self.clock = clock if clock is not None else RestoreClock()
        self.addCleanup(self.clock.release.set)
        session = self.make(spawn=spawn)
        session.start()
        session.send_user(text)
        self.tmux.screen = busy_screen("")  # an Escape is pressed only while the CLI works
        self.tmux.after_escape = screen
        return session, len(self.tmux.keys_sent)

    def test_interrupt_after_a_paste_empties_the_message_the_cli_put_back(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"])
        self.assertTrue(input_empty(self.tmux.screen), "the box still holds the message")
        self.assertTrue(any("cleared the message" in line for line in self.logs), self.logs)

    def test_a_long_paste_put_back_as_its_placeholder_is_emptied_too(self) -> None:
        twelve_lines = "\n".join(["line"] * 12)  # 12 lines: the placeholder's count is checked against them
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("[Pasted text #2 +12 lines]"),
                                                                   text=twelve_lines)

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"])
        self.assertTrue(input_empty(self.tmux.screen), "the placeholder is still in the box")

    def test_text_the_user_typed_in_the_tab_is_left_alone(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("my own draft"))

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "a clear key reached the user's text")
        self.assertEqual(self.tmux.screen, box_screen("my own draft"))
        self.assertTrue(any("holds other text" in line for line in self.logs), self.logs)

    def test_an_empty_box_for_the_whole_wait_sends_no_clear_keys(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(CLI_BOX)  # the CLI never put the message back
        polls = len(self.clock.sleeps)

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        rounds = round(RESTORE_WAIT_SECONDS / RESTORE_POLL_SECONDS)
        self.assertEqual(self.clock.sleeps[polls:], [RESTORE_POLL_SECONDS] * rounds)

    def test_an_escape_after_a_paste_starts_one_restore(self) -> None:
        self.tmux.on_paste = None
        self.clock = RestoreClock()
        session = self.make()
        session.start()
        session.send_user("hello")
        self.tmux.screen = busy_screen("")  # the CLI works, so the first Escape is pressed

        self.assertTrue(session.interrupt())
        self.assertTrue(session.interrupt())  # the paste's restore is taken by the first Escape
        self.assertEqual(len(self.restores), 1)

    def test_an_interrupt_with_nothing_pasted_sends_only_escape(self) -> None:
        self.clock = RestoreClock()
        self.addCleanup(self.clock.release.set)
        session = self.make()
        session.start()
        self.tmux.screen = busy_screen("")

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent, ["Escape"])
        self.assertEqual(self.clock.sleeps, [])

    def test_a_failed_escape_starts_no_restore(self) -> None:
        self.tmux.on_paste = None
        session = self.make()
        session.start()
        session.send_user("hello")
        self.tmux.screen = busy_screen("")
        self.tmux.fail = ("send-keys",)  # the Escape cannot be sent

        self.assertFalse(session.interrupt())
        self.assertEqual(self.restores, [])

    def test_an_interrupt_after_the_bridges_turn_ended_sends_only_escape(self) -> None:
        """The CLI answered the pasted message, so the bridge's turn is over: an Escape now restores nothing, even when
        the message is in the box."""
        session = self.make()
        session.start()
        session.send_user("hello")
        self.wait_for_result()
        self.tmux.screen = busy_screen("hello")  # a turn of the tab's own, still working, with the text in its box
        typed = len(self.tmux.keys_sent)

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        self.assertEqual(self.restores, [])

    def test_an_interrupt_during_a_turn_the_tab_started_sends_only_escape(self) -> None:
        notification = {"type": "user", "origin": {"kind": "task-notification"},
                        "message": {"role": "user", "content": "<task-notification><task-id>b1</task-id>"
                                                               "</task-notification>"}}
        session = self.make()
        session.start()
        session.send_user("hello")
        self.wait_for_result()
        with open(self.transcript, "a", encoding="utf-8") as handle:  # the tab starts a turn of its own, still busy
            for line in (notification, assistant_line("WORKING")):
                handle.write(json.dumps(line) + "\n")
        self.assertTrue(wait_until(lambda: any(isinstance(e, TurnBegan) for e in self.events)),
                        "the tab's own turn did not begin")
        self.tmux.screen = busy_screen("hello")  # still working, the text in the box
        typed = len(self.tmux.keys_sent)

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        self.assertEqual(self.restores, [])

    def paste_on_an_idle_cli(self, clock) -> AttachedSession:
        """Pastes hello into the idle CLI, with clock as the session's clock. The CLI has not answered, so the turn is
        open; the screen is left to the test (the CLI's spinner shows or not)."""
        self.tmux.on_paste = None
        self.clock = clock
        self.addCleanup(clock.release.set)
        session = self.make()
        session.start()
        session.send_user("hello")
        return session

    def test_three_interrupts_within_three_seconds_during_one_paste_turn_press_escape_once(self) -> None:
        """Three taps of ⏭ Now within two seconds during one turn: the first Escape interrupts it, and the next two
        press nothing (one Escape per pasted message, and no second Escape within ESCAPE_REPEAT_SECONDS)."""
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)
        self.tmux.screen = busy_screen("")
        typed = len(self.tmux.keys_sent)
        waited = len(clock.sleeps)

        self.assertEqual(session.interrupt(), True)
        clock.advance(1.0)
        self.assertEqual(session.interrupt(), True)
        clock.advance(1.0)
        self.assertEqual(session.interrupt(), True)
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        self.assertEqual(clock.sleeps[waited:], [], "an interrupt waited for the screen")

    def test_an_idle_screen_during_the_paste_turn_still_gets_its_one_escape(self) -> None:
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)  # the box is empty and no spinner shows: the CLI may not show it yet
        typed = len(self.tmux.keys_sent)
        waited = len(clock.sleeps)

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "the open paste turn got no Escape")
        self.assertEqual(clock.sleeps[waited:], [])

    def test_a_dialog_on_screen_gets_one_escape(self) -> None:
        """A selection dialog of the CLI: the Escape cancels it, as it always did, and the turn ends."""
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)
        self.tmux.screen = CLI_BOX + "  Enter to confirm · Esc to cancel\n"
        typed = len(self.tmux.keys_sent)
        waited = len(clock.sleeps)

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "the dialog got no Escape")
        self.assertEqual(clock.sleeps[waited:], [], "the dialog was waited for")

    def test_a_new_paste_more_than_three_seconds_after_an_escape_gets_one_escape(self) -> None:
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)
        self.tmux.screen = busy_screen("")
        self.assertTrue(session.interrupt())
        self.tmux.screen = CLI_BOX  # the CLI stopped: the next message can be pasted
        clock.advance(ESCAPE_REPEAT_SECONDS)
        session.send_user("again")
        self.tmux.screen = busy_screen("")
        typed = len(self.tmux.keys_sent)

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "the new message got no Escape of its own")
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "the new message got a second Escape")

    def test_a_paste_within_three_seconds_of_an_escape_gets_one_escape_and_a_repeat_presses_nothing(self) -> None:
        """Escape interrupts turn A, then message B is pasted 1 s later (inside ESCAPE_REPEAT_SECONDS). B's first interrupt
        presses one Escape: the paste lifts the throttle of A's Escape. A second interrupt right after presses nothing."""
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)  # paste A
        self.tmux.screen = busy_screen("")
        self.assertTrue(session.interrupt())  # Escape for A
        self.tmux.screen = CLI_BOX  # the CLI stopped: B can be pasted
        clock.advance(1.0)
        session.send_user("again")  # paste B, within 3 s of the Escape
        self.tmux.screen = busy_screen("")
        typed = len(self.tmux.keys_sent)

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "turn B got no Escape of its own")
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "a repeat tap on turn B pressed a second Escape")

    def test_a_failed_escape_returns_false(self) -> None:
        session = self.paste_on_an_idle_cli(RestoreClock())
        self.tmux.screen = busy_screen("")
        self.tmux.fail = ("send-keys",)

        self.assertFalse(session.interrupt())

    def test_a_turn_the_tab_started_takes_one_escape_and_a_second_within_three_seconds_presses_nothing(self) -> None:
        clock = RestoreClock()
        self.clock = clock
        self.addCleanup(clock.release.set)
        session = self.make()
        session.start()  # no paste in this session: the turn is the tab's own
        self.tmux.screen = busy_screen("")
        typed = len(self.tmux.keys_sent)

        self.assertTrue(session.interrupt())
        clock.advance(1.0)
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "a second Escape came within 3 s")
        clock.advance(ESCAPE_REPEAT_SECONDS)
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "Escape"], "no Escape after the 3 s")

    def test_a_tab_turn_after_an_escaped_paste_takes_one_escape_and_a_second_within_3_s_presses_nothing(self) -> None:
        """The paste's own turn is over, and the tab then starts a turn of its own: the paste's Escape does not cover
        it. Its first interrupt, more than three seconds after the last Escape, presses one Escape; a second one within
        three seconds presses nothing."""
        clock = RestoreClock()
        session = self.paste_on_an_idle_cli(clock)
        self.tmux.screen = busy_screen("")
        self.assertTrue(session.interrupt())
        self.cli_answers()  # the paste's turn ends
        self.wait_for_result()
        clock.advance(ESCAPE_REPEAT_SECONDS)  # the tab's turn is a later turn than the Escape
        notification = {"type": "user", "origin": {"kind": "task-notification"},
                        "message": {"role": "user", "content": "<task-notification><task-id>b1</task-id>"
                                                               "</task-notification>"}}
        with open(self.transcript, "a", encoding="utf-8") as handle:
            for line in (notification, assistant_line("WORKING")):
                handle.write(json.dumps(line) + "\n")
        self.assertTrue(wait_until(lambda: any(isinstance(e, TurnBegan) for e in self.events)),
                        "the tab's own turn did not begin")
        self.tmux.screen = busy_screen("")
        typed = len(self.tmux.keys_sent)

        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "the tab's own busy turn got no Escape")
        clock.advance(1.0)
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"], "a second Escape came within 3 s")

    def test_stop_waits_for_a_running_restore(self) -> None:
        """The restore thread waits at its first look (the box is empty). stop() returns only when that restore has
        finished, so the next turn's paste cannot meet its keys."""
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=GatedRestoreClock(),
                                                                    spawn=threaded_spawn)
        self.addCleanup(self.clock.gate.set)
        self.assertTrue(session.interrupt())
        stopper = threading.Thread(target=session.stop)
        stopper.start()
        stopper.join(0.3)
        self.assertTrue(stopper.is_alive(), "stop() returned while the restore was still running")

        self.clock.gate.set()
        stopper.join(WAIT_SECONDS)
        self.assertFalse(stopper.is_alive(), "stop() did not return after the restore finished")
        self.assertEqual(session._restores, 0)

    def test_stop_logs_a_restore_that_does_not_finish_in_time(self) -> None:
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=GatedRestoreClock(),
                                                                    spawn=threaded_spawn)
        self.addCleanup(self.clock.gate.set)
        self.assertTrue(session.interrupt())

        with mock.patch.object(attached_session, "RESTORE_JOIN_SECONDS", 0.2):
            session.stop()
        self.assertTrue(any("did not finish" in line for line in self.logs), self.logs)

    def test_stop_waits_until_both_overlapping_restores_have_finished(self) -> None:
        """A second Escape after a new paste starts a second restore while the first one's window is still open. The
        first restore ending must not end stop()'s wait: it waits for both."""
        clock = NamedGateClock(["restore-1", "restore-2"])
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=clock,
                                                                    spawn=named_spawn(["restore-1", "restore-2"]))
        self.addCleanup(clock.gates["restore-2"].set)
        self.addCleanup(clock.gates["restore-1"].set)
        self.assertTrue(session.interrupt())  # restore-1 waits at its first look
        session.send_user("hello")  # a new paste while that window is open
        self.clock.advance(ESCAPE_REPEAT_SECONDS)  # the second Escape is more than 3 s after the first
        self.tmux.screen = busy_screen("")  # the CLI works on it
        self.assertTrue(session.interrupt())  # the second Escape starts restore-2 for it
        self.assertEqual(session._restores, 2)

        stopper = threading.Thread(target=session.stop)
        stopper.start()
        clock.gates["restore-1"].set()
        self.assertTrue(self.wait_restores(session, 1), "the first restore did not end")
        stopper.join(0.3)
        self.assertTrue(stopper.is_alive(), "stop() returned while the second restore was still running")

        clock.gates["restore-2"].set()
        stopper.join(WAIT_SECONDS)
        self.assertFalse(stopper.is_alive(), "stop() did not return after both restores finished")
        self.assertEqual(session._restores, 0)

    def test_a_restore_still_running_when_the_wait_ends_is_abandoned_and_presses_no_key(self) -> None:
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=GatedRestoreClock(),
                                                                   spawn=threaded_spawn)
        self.addCleanup(self.clock.gate.set)
        self.assertTrue(session.interrupt())  # the restore waits at its first look, the box still empty

        with mock.patch.object(attached_session, "RESTORE_JOIN_SECONDS", 0.2):
            session.stop()  # the wait ends with the restore still running: it is abandoned
        self.assertTrue(any("abandoned" in line for line in self.logs), self.logs)

        after_timeout = len(self.tmux.keys_sent)
        self.tmux.screen = box_screen("hello")  # the message is in the box when the restore wakes up
        self.clock.gate.set()
        self.assertTrue(self.wait_restores(session, 0), "the abandoned restore did not end")
        self.assertEqual(self.tmux.keys_sent[after_timeout:], [], "a key was pressed after the abandon")
        self.assertEqual(self.tmux.screen, box_screen("hello"))

    def test_a_clear_started_after_the_abandon_presses_no_key(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))
        session._abandoned = True  # what stop() sets when its wait has ended

        session._clear_box("hello", "hello", session._paste_seq)  # the box the restore read: the pasted text
        self.assertEqual(self.tmux.keys_sent[typed:], [], "a clear key was pressed after the abandon")

    def test_stop_with_no_restore_returns_at_once(self) -> None:
        session = self.make()
        session.start()
        began = time.monotonic()

        session.stop()
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertFalse(any("did not finish" in line for line in self.logs), self.logs)

    def test_a_newer_paste_during_the_restore_wins_even_with_the_same_words(self) -> None:
        """The restore is for the first 'hello'. The same words are pasted again while it waits, and the box then holds
        them: the restore is for a paste, not for a text, so it empties nothing."""
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=GatedRestoreClock(),
                                                                    spawn=threaded_spawn)
        self.addCleanup(self.clock.gate.set)
        self.assertTrue(session.interrupt())  # the restore thread waits at its first look
        session.send_user("hello")  # a newer paste of the same words
        typed = len(self.tmux.keys_sent)
        self.tmux.screen = box_screen("hello")

        self.clock.gate.set()
        self.assertTrue(self.wait_restores(session, 0))
        self.assertNotIn("C-u", self.tmux.keys_sent[typed:], "the restore cleared the newer paste")
        self.assertEqual(self.tmux.screen, box_screen("hello"))

    def test_a_busy_cli_holds_the_window_open_until_the_message_is_put_back(self) -> None:
        """The CLI works for five seconds before it puts the message back. The look is pushed on while it is busy, so
        the message is found, although the window is only RESTORE_WAIT_SECONDS long when the CLI is idle."""
        session, typed = self.paste_then_escape_with_the_turn_open(busy_screen(""), clock=PollClock())
        first = len(self.clock.sleeps)

        def cli_puts_it_back(polls: int) -> None:
            if polls - first == 20:  # 5 s of work, then the CLI is idle with the message in its box
                self.tmux.screen = box_screen("hello")
        self.clock.on_poll = cli_puts_it_back

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"])
        self.assertTrue(input_empty(self.tmux.screen), "the message the CLI put back is still in the box")

    def test_no_key_is_typed_while_the_cli_works_even_with_the_message_in_the_box(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(busy_screen("hello"), clock=PollClock())
        first = len(self.clock.sleeps)
        typed_while_busy: list[str] = []

        def record_key(key: str) -> None:
            if key != "Escape" and self.tmux.screen == busy_screen("hello"):
                typed_while_busy.append(key)
            self.ctrl_u_clears_the_line(key)

        def cli_idle_from_the_fourth_look(polls: int) -> None:
            if polls - first == 4:
                self.tmux.screen = box_screen("hello")
        self.tmux.on_keys = record_key
        self.clock.on_poll = cli_idle_from_the_fourth_look

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(typed_while_busy, [], "a key was typed while the CLI was working")
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"])

    def test_a_cli_busy_beyond_the_cap_gives_up_without_a_clear(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(busy_screen(""), clock=PollClock())
        first = len(self.clock.sleeps)

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        self.assertEqual(sum(self.clock.sleeps[first:]), RESTORE_CAP_SECONDS)

    def test_a_clear_that_fails_once_is_tried_again_after_one_poll_gap(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))
        presses: list[str] = []

        def the_fifth_clear_key_empties_the_box(key: str) -> None:  # the first clear (two rounds) does not empty it
            if key == "Escape":
                return
            presses.append(key)
            if len(presses) == 5:
                self.tmux.screen = CLI_BOX

        self.tmux.on_keys = the_fifth_clear_key_empties_the_box
        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u", "BSpace", "C-u", "BSpace", "C-u"])
        self.assertIn(CLEAR_KEY_GAP_SECONDS, self.clock.sleeps, "the clear's gaps did not run on the session's clock")
        self.assertTrue(any("cleared the message" in line for line in self.logs), self.logs)

    def test_a_box_that_stays_full_after_both_clears_is_logged(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))
        self.tmux.on_keys = None  # nothing empties the box

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(len(self.tmux.keys_sent[typed:]), 1 + 2 * 2 * 2, "Escape, then two clears of two rounds")
        self.assertTrue(any("could not clear" in line for line in self.logs), self.logs)

    def clear_keys_redraw(self, *screens: str) -> None:
        """on_keys for the keys after the Escape: after each one the screen is the next of screens (the last stays)."""
        pending = list(screens)

        def redraw(key: str) -> None:
            if key != "Escape":
                self.tmux.screen = pending.pop(0) if len(pending) > 1 else pending[0]
        self.tmux.on_keys = redraw

    def test_a_turn_closed_during_the_escape_still_gets_its_restore(self) -> None:
        """The transcript answers the turn while the Escape is in flight. The target was decided before the Escape,
        while the turn was open, so the restore still starts and empties the box."""
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))

        def answer_during_the_escape(key: str) -> None:
            if key == "Escape":
                self.cli_answers()
                self.wait_for_result()
            else:
                self.ctrl_u_clears_the_line(key)
        self.tmux.on_keys = answer_during_the_escape

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(len(self.results()), 1)
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"])
        self.assertTrue(input_empty(self.tmux.screen), "the message put back is still in the box")

    def test_an_interrupt_after_stop_presses_nothing_starts_no_restore_and_returns_false(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))
        session.stop()

        self.assertFalse(session.interrupt())
        self.assertEqual(self.tmux.keys_sent[typed:], [], "an Escape was sent after stop()")
        self.assertEqual(self.restores, [])

    def test_stop_during_an_escape_in_flight_waits_for_the_restore_that_escape_starts(self) -> None:
        escape_sent, escape_released = threading.Event(), threading.Event()

        def hold_the_escape(key: str) -> None:
            if key == "Escape":
                escape_sent.set()
                escape_released.wait(WAIT_SECONDS)
            else:
                self.ctrl_u_clears_the_line(key)
        session, _typed = self.paste_then_escape_with_the_turn_open(CLI_BOX, clock=GatedRestoreClock(),
                                                                    spawn=threaded_spawn)
        self.addCleanup(self.clock.gate.set)
        self.tmux.on_keys = hold_the_escape
        interrupter = threading.Thread(target=session.interrupt)
        interrupter.start()
        self.assertTrue(escape_sent.wait(WAIT_SECONDS))
        stopper = threading.Thread(target=session.stop)
        stopper.start()
        stopper.join(0.2)  # stop() waits for the Escape's keys
        escape_released.set()
        interrupter.join(WAIT_SECONDS)

        self.assertEqual(session._restores, 1, "the Escape started no restore")
        stopper.join(0.3)
        self.assertTrue(stopper.is_alive(), "stop() returned while the restore the Escape started was still running")
        self.clock.gate.set()
        stopper.join(WAIT_SECONDS)
        self.assertFalse(stopper.is_alive(), "stop() did not return after the restore finished")

    def test_a_restore_that_cannot_start_is_logged_and_stop_returns_at_once(self) -> None:
        def cannot_start(work) -> None:
            raise RuntimeError("no thread can be started")
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"), spawn=cannot_start)

        self.assertTrue(session.interrupt(), "the Escape is still sent")
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape"])
        began = time.monotonic()
        session.stop()
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertEqual(session._restores, 0, "a restore that never started is still counted")
        self.assertTrue(any("could not start the restore" in line for line in self.logs), self.logs)
        self.assertFalse(any("did not finish" in line for line in self.logs), self.logs)

    def test_a_box_that_turns_into_the_users_own_text_between_clear_keys_gets_no_further_key(self) -> None:
        session, typed = self.paste_then_escape_with_the_turn_open(box_screen("hello"))

        def the_user_types_after_the_first_clear_key(key: str) -> None:
            if key == "C-u":
                self.tmux.screen = box_screen("my own draft")
        self.tmux.on_keys = the_user_types_after_the_first_clear_key

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u"], "a clear key reached the user's text")
        self.assertEqual(self.tmux.screen, box_screen("my own draft"))

    def test_a_box_partly_cleared_by_the_clear_keeps_being_cleared(self) -> None:
        """C-u removes the last row first, so the box is a prefix of the pasted text after the first key: it is still
        the bridge's, and the clear goes on until the box is empty."""
        session, typed = self.paste_then_escape_with_the_turn_open(box_rows("hello", "world"), text="hello\nworld")
        self.clear_keys_redraw(box_rows("hello", ""), box_rows("hello"), box_rows(""))

        self.assertTrue(self.interrupt_and_restore(session))
        self.assertEqual(self.tmux.keys_sent[typed:], ["Escape", "C-u", "BSpace", "C-u"])
        self.assertTrue(input_empty(self.tmux.screen))
        self.assertTrue(any("cleared the message" in line for line in self.logs), self.logs)

    def test_still_ours_accepts_the_pasted_text_and_its_prefixes_only(self) -> None:
        self.assertTrue(_still_ours("hello\nworld", "hello\nworld"))
        self.assertTrue(_still_ours("hello", "hello\nworld"))
        self.assertTrue(_still_ours("[Pasted text #2 +12 lines]", "\n".join(["line"] * 12)))
        self.assertFalse(_still_ours("my own draft", "hello\nworld"))
        self.assertFalse(_still_ours("", "hello\nworld"))
        self.assertFalse(_still_ours("[Pasted text #2 +12 lines]", "[Pasted text #2 +12 lines] and more"))

    def test_the_join_outlasts_the_restore_look_so_a_normal_restore_is_not_abandoned(self) -> None:
        self.assertGreater(RESTORE_JOIN_SECONDS, RESTORE_CAP_SECONDS)

    def test_stop_and_end_never_send_a_key(self) -> None:
        session = self.make()
        session.start()
        session.send_user("hello")
        typed = self.tmux.keys()
        self.assertEqual(typed, ["load-buffer", "paste-buffer", "send-keys"])

        session.stop()
        self.assertTrue(session.end_gracefully(1.0))
        session.stop()
        self.assertEqual(self.tmux.keys(), typed)
        self.assertFalse(session.alive())

    def test_answers_and_mode_and_model_are_refused_without_a_key(self) -> None:
        session = self.make()
        session.start()
        req = ApprovalRequest(approval_id="a1", kind="tool", tool="Bash", input={}, summary="s",
                              choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}))
        self.assertFalse(session.answer(req, Decision.ALLOW_ONCE))
        self.assertFalse(session.set_mode("plan"))
        self.assertFalse(session.set_model("opus"))
        self.assertEqual(self.tmux.keys(), [])

    def test_idle_reads_the_pane_screen(self) -> None:
        session = self.make()
        self.tmux.screen = "❯  \n"  # no input box on the screen: not idle
        self.assertFalse(session.idle())
        self.tmux.screen = "─" * 20 + "\n❯  \n" + "─" * 20 + "\n  ? for shortcuts\n"
        self.assertTrue(session.idle())

    def test_pane_reason_reads_the_pane_through_tmux(self) -> None:
        self.assertEqual(pane_reason("", PANE, run=self.tmux), REASON_IDLE)
        self.tmux.screen = "─" * 40 + "\n❯ hello\n" + "─" * 40 + "\n  ? for shortcuts\n"
        self.assertEqual(pane_reason("", PANE, run=self.tmux), REASON_TYPED)
        self.tmux.pane_shown = "%9"  # tmux does not show the pane under its id any more
        self.assertEqual(pane_reason("", PANE, run=self.tmux), REASON_GONE)

    def test_alive_is_false_until_start_and_after_stop(self) -> None:
        session = self.make()
        self.assertFalse(session.alive())
        session.start()
        self.assertTrue(session.alive())
        session.stop()
        self.assertFalse(session.alive())

    def test_alive_is_false_once_the_pane_is_gone(self) -> None:
        session = self.make()
        session.start()
        self.tmux.screen = CLI_BOX
        self.assertTrue(session.alive())
        self.tmux.pane_shown = "%9"
        self.assertFalse(session.alive())

    def test_alive_follows_the_claim_the_turn_started_with(self) -> None:
        """The view pane releases its claim when the CLI exits, while the pane stays and shows its ready screen. The
        session is alive only while the claim is held, whatever the screen shows."""
        held = [True]
        session = self.make(still_attached=lambda: held[0])
        session.start()
        self.tmux.screen = "user@host:~$ \n"  # the pane shows a shell: the CLI is gone, but its claim is held
        self.assertTrue(session.alive())
        held[0] = False  # the view pane released the claim when the CLI exited
        self.assertFalse(session.alive(), "a released claim still counts as alive")
        self.tmux.screen = CLI_BOX
        self.assertFalse(session.alive(), "a released claim on a CLI screen still counts as alive")

    def test_a_pane_that_is_not_ready_refuses_the_paste_and_begins_nothing(self) -> None:
        """The owner typed into the tab, or started a turn there, after the wait saw the pane idle: nothing is typed and
        no turn is begun, so the message can be sent again later."""
        session = self.make()
        session.start()
        for busy in ("✻ Thinking… (esc to interrupt)\n" + CLI_BOX,
                     "─" * 40 + "\n❯ half a sentence\n" + "─" * 40 + "\n  ? for shortcuts\n"):
            self.tmux.screen = busy
            with self.assertRaises(TerminalNotReady):
                session.send_user("hello")
        self.assertEqual(self.tmux.keys(), [], "a key was typed into a pane that was not ready")
        self.assertEqual(self.results(), [], "a turn was begun for a refused paste")

    def test_the_attached_session_is_not_linger_and_takes_no_images(self) -> None:
        self.assertFalse(ATTACHED_CAPABILITIES.linger)
        self.assertFalse(ATTACHED_CAPABILITIES.images)


NEW_UUID = "66666666-7777-8888-9999-000000000000"
FOREIGN_UUID = "88888888-7777-6666-5555-444444444444"  # another tab of the same folder


def queued_command_line(prompt: str) -> dict:
    """The attachment the CLI writes when it delivers a message pasted while its turn was busy."""
    return {"type": "attachment", "attachment": {"type": "queued_command", "prompt": prompt, "commandMode": "prompt"}}


class RelocateClock(BlockingClock):
    """The echo rounds (ATTACH_RELOCATE_SECONDS each) return at once, so a test does not wait for them, until the round
    block_at: that sleep blocks until the test releases it. The rounds after it wait for after, which the test sets once
    the turn has read what it waits for, so the rounds never run back to back ahead of the tail. Every round is
    recorded, so the echo window can be checked."""

    def __init__(self, block_at: int = 3) -> None:
        super().__init__()
        self.sleeps: list[float] = []
        self.block_at = block_at
        self.after = threading.Event()

    def sleep(self, seconds: float) -> None:
        if threading.current_thread() is threading.main_thread():  # the test's own thread (a paste's gap) does not wait
            return
        if threading.current_thread().name != "user-echo":  # the locate timer of the transcript is not the echo window
            self.release.wait(WAIT_SECONDS)
            return
        self.sleeps.append(seconds)
        if len(self.sleeps) == self.block_at:
            self.release.wait()
        elif len(self.sleeps) > self.block_at:
            self.after.wait()


class QueuedRoundsClock(RelocateClock):
    """The first echo round waits until the test has seen the queued message in the tail; the rounds after it return at
    once, and the fourth blocks, so three rounds are checked."""

    def __init__(self, seen: threading.Event, block_at: int = 4) -> None:
        super().__init__(block_at=block_at)
        self.seen = seen

    def sleep(self, seconds: float) -> None:
        if threading.current_thread().name == "user-echo" and not self.sleeps:
            self.seen.wait(WAIT_SECONDS)
        super().sleep(seconds)


class RelocationTests(unittest.TestCase):
    """A /clear or /resume in the tab: the CLI writes the message's echo into another transcript. The turn finds it within
    the first 10 s and follows that file; with no such file the turn ends with the 30 s echo error. A message the CLI
    queued keeps the turn open past that window."""

    def setUp(self) -> None:
        home = self.enterContext(TempHome()).home
        self.cwd = os.path.join(home, "work")
        self.config_dir = os.path.join(home, "config")
        os.makedirs(self.cwd)
        folder = history.encoded_cwd(os.path.realpath(self.cwd))
        self.old = os.path.join(self.config_dir, "projects", folder, UUID + ".jsonl")
        self.new = os.path.join(self.config_dir, "projects", folder, NEW_UUID + ".jsonl")
        os.makedirs(os.path.dirname(self.old))
        open(self.old, "w", encoding="utf-8").close()
        self.clock = RelocateClock()
        self.addCleanup(self.clock.release.set)
        self.events: list[object] = []
        self.logs: list[str] = []
        self.tmux = FakeTmux(on_paste=self.cli_answers_in_new_session)
        self.spec = SessionSpec(thread_id=1, chat_id=2, topic="t", launcher=LauncherSpec(
            name="l", backend="claude", command="c", argv=("c",), config_dir=self.config_dir), cwd=self.cwd,
            mode="default", model=None, resume_id=UUID, new_id=None, allow_rules=(), deny_tools=(), system_rules="",
            env={}, errfile_dir=self.cwd)

    def log(self, msg: object) -> None:
        self.logs.append(str(msg))

    def cli_answers_in_new_session(self) -> None:
        with open(self.new, "a", encoding="utf-8") as handle:
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

    def cli_answers_in_old_session(self) -> None:
        with open(self.old, "a", encoding="utf-8") as handle:
            for line in (echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE):
                handle.write(json.dumps(line) + "\n")

    def append_to_old_session(self, lines) -> None:
        with open(self.old, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")

    def append_to_new_session(self, lines) -> None:
        with open(self.new, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")

    def attach_to(self, *, hooked: bool, claim_uuid) -> None:
        """The spec of a terminal tab: its claim (claim_uuid reads the session it names) and whether the tab runs
        the hook that keeps the claim current."""
        attach = AttachSpec(socket="", pane=PANE, uuid=UUID, config_dir=self.config_dir, cwd=self.cwd, hooked=hooked,
                            claim_uuid=claim_uuid)
        self.spec = dataclasses.replace(self.spec, attach=attach)

    def foreign_transcript(self, lines) -> None:
        """Another tab's transcript in another project folder, written after the paste: the same message, another
        session."""
        path = os.path.join(self.config_dir, "projects", "other-project", FOREIGN_UUID + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")

    def cli_answers_in_new_session_beside_a_foreign_tab(self) -> None:
        self.cli_answers_in_new_session()
        self.foreign_transcript([echo_line("hello"), assistant_line("FOREIGN"), TURN_DURATION_LINE])

    def cli_queues_in_old_session(self) -> None:
        self.append_to_old_session([{"type": "queue-operation", "operation": "enqueue", "content": "hello"}])

    def make(self) -> AttachedSession:
        session = AttachedSession(self.spec, self.events.append, socket="", pane=PANE, uuid=UUID,
                                  config_dir=self.config_dir, cwd=self.cwd, logger=self, clock=self.clock,
                                  run=self.tmux)
        self.addCleanup(session.stop)
        session.start()
        return session

    def results(self) -> list[TurnResult]:
        return [e for e in self.events if isinstance(e, TurnResult)]

    def wait_for_result(self) -> TurnResult:
        deadline = time.monotonic() + WAIT_SECONDS
        while not self.results() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(self.results()), 1, "no turn result came from the transcript")
        return self.results()[0]

    def test_the_echo_in_the_new_transcript_is_found_and_the_session_moves_to_it(self) -> None:
        session = self.make()
        session.send_user("hello")
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, NEW_UUID)
        self.assertIn(SessionStarted(session_id=NEW_UUID, model=None, mode="default", cwd=self.cwd, resumed=True),
                      self.events, "the new session id was not reported to the turn runner")
        self.assertEqual(self.clock.sleeps[:1], [ATTACH_RELOCATE_SECONDS])

    def test_with_no_new_transcript_the_turn_ends_with_the_echo_error_after_thirty_seconds(self) -> None:
        self.tmux.on_paste = None  # nothing is written anywhere
        session = self.make()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 3), "the third echo round did not start")
        self.assertEqual(self.clock.sleeps, [10.0, 10.0, 10.0], "the echo window is not 30 seconds in three rounds")
        self.assertEqual(self.results(), [], "the turn ended before the echo window")
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertEqual(result.subtype, ECHO_MISSED_SUBTYPE)
        self.assertEqual(result.error, ATTACH_ECHO_MISSED_ERROR.format(session=UUID, seconds=USER_ECHO_SECONDS))
        self.assertIn(UUID, result.error, "the error does not name the session")
        self.assertEqual(session.session_id, UUID, "the session moved without a transcript")

    def test_the_echo_written_after_the_first_round_is_found_in_a_later_round(self) -> None:
        self.clock = RelocateClock(block_at=2)  # the second round blocks: the first one has found nothing by then
        self.addCleanup(self.clock.release.set)
        self.addCleanup(self.clock.after.set)
        self.tmux.on_paste = None
        session = self.make()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 2), "the second echo round did not start")
        self.assertEqual(session.session_id, UUID, "the session moved before the echo was written")
        self.append_to_new_session([echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE])
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, NEW_UUID)
        self.assertEqual(self.clock.sleeps[:2], [10.0, 10.0])

    def test_a_queued_message_keeps_the_turn_open_past_the_echo_window(self) -> None:
        seen = threading.Event()
        self.clock = QueuedRoundsClock(seen)
        self.addCleanup(self.clock.release.set)
        self.addCleanup(self.clock.after.set)
        self.tmux.on_paste = self.cli_queues_in_old_session  # only the enqueue line is written
        session = self.make()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: session._turns._turn.queued), "the enqueue line was not read")
        seen.set()
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 4), "the fourth echo round did not start")
        self.assertEqual(self.results(), [], "the turn ended while its message was still queued")
        self.assertEqual(self.clock.sleeps, [10.0, 10.0, 10.0, 10.0])
        self.append_to_old_session([{"type": "queue-operation", "operation": "dequeue"},
                                    echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE])
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(len(self.results()), 1, "the turn produced more than one result")

    def test_a_queued_turn_does_not_look_in_other_transcripts(self) -> None:
        seen = threading.Event()
        self.clock = QueuedRoundsClock(seen)
        self.addCleanup(self.clock.release.set)
        self.addCleanup(self.clock.after.set)

        def queued_with_a_foreign_echo() -> None:  # the same message is echoed in another transcript too
            self.cli_queues_in_old_session()
            self.append_to_new_session([echo_line("hello"), assistant_line("FOREIGN"), TURN_DURATION_LINE])

        self.tmux.on_paste = queued_with_a_foreign_echo
        session = self.make()
        lookups: list[tuple] = []
        relocate = session._turns._relocate

        def counting_relocate(*args):
            lookups.append(args)
            return relocate(*args)

        session._turns._relocate = counting_relocate
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: session._turns._turn.queued), "the enqueue line was not read")
        seen.set()
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 4), "the fourth echo round did not start")
        self.assertEqual(lookups, [], "a queued turn looked in other transcripts")
        self.assertEqual(session.session_id, UUID, "the turn moved to a foreign transcript")
        self.append_to_old_session([{"type": "queue-operation", "operation": "dequeue"},
                                    echo_line("hello"), assistant_line("ANSWER"), TURN_DURATION_LINE])
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, UUID)

    def test_an_echo_read_in_the_claims_transcript_does_not_relocate(self) -> None:
        self.tmux.on_paste = self.cli_answers_in_old_session
        session = self.make()
        session.send_user("hello")
        result = self.wait_for_result()
        self.assertEqual(result.answer, "ANSWER")
        self.assertEqual(session.session_id, UUID)
        self.assertFalse(any(isinstance(e, SessionStarted) and e.session_id == NEW_UUID for e in self.events))

    def test_a_relocation_found_after_the_echo_arrived_is_dropped(self) -> None:
        self.tmux.on_paste = None  # the echo is written by the test, into the followed transcript, not by the paste
        session = self.make()

        def echo_arrives_during_the_lookup(*args) -> tuple[str, int, str]:
            self.append_to_old_session([echo_line("hello")])  # the echo is read in the followed transcript first
            self.assertTrue(wait_until(lambda: session._turns._turn.echoed), "the echo was not read")
            self.append_to_new_session([echo_line("hello"), assistant_line("FOREIGN"), TURN_DURATION_LINE])
            return self.new, 0, NEW_UUID  # a foreign transcript that holds the same message, found too late

        session._turns._relocate = echo_arrives_during_the_lookup
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: "session relocation dropped: the turn moved on" in self.logs),
                        "the late relocation was not dropped")
        self.assertEqual(session.session_id, UUID, "a relocation found after the echo moved the session")
        self.assertEqual(self.results(), [], "the turn ended before its answer was read")
        self.append_to_old_session([assistant_line("ANSWER"), TURN_DURATION_LINE])
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, UUID)
        self.assertFalse(any(isinstance(e, SessionStarted) and e.session_id == NEW_UUID for e in self.events))

    def test_a_hooked_tab_never_follows_a_foreign_transcript(self) -> None:
        """The claim names the followed session, so the echo is not looked for in any other transcript: another tab of
        the same folder holds the same message, and the turn does not move to it."""
        self.attach_to(hooked=True, claim_uuid=lambda: UUID)
        self.tmux.on_paste = lambda: self.foreign_transcript([echo_line("hello"), assistant_line("FOREIGN"),
                                                              TURN_DURATION_LINE])
        session = self.make()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 3), "the third echo round did not start")
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertEqual(result.subtype, ECHO_MISSED_SUBTYPE)
        self.assertEqual(session.session_id, UUID, "the turn moved to a foreign transcript")
        self.assertFalse(any(isinstance(e, SessionStarted) and e.session_id == FOREIGN_UUID for e in self.events))

    def test_a_hooked_tab_follows_the_session_its_claim_moved_to(self) -> None:
        """A /clear in the tab: the hook told the claim the new session, and the echo is in that session's transcript.
        The foreign transcript with the same echo is not read."""
        self.attach_to(hooked=True, claim_uuid=lambda: NEW_UUID)
        self.tmux.on_paste = self.cli_answers_in_new_session_beside_a_foreign_tab
        session = self.make()
        session.send_user("hello")
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, NEW_UUID)
        self.assertFalse(any(isinstance(e, SessionStarted) and e.session_id == FOREIGN_UUID for e in self.events))

    def test_an_unhooked_tab_keeps_the_folder_scan(self) -> None:
        """A tab started by older code runs no hook: its echo is still found by scanning the folder's transcripts."""
        self.attach_to(hooked=False, claim_uuid=lambda: "")
        self.tmux.on_paste = self.cli_answers_in_new_session
        session = self.make()
        session.send_user("hello")
        result = self.wait_for_result()
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(session.session_id, NEW_UUID)

    def test_a_queued_message_fails_only_at_the_600_s_deadline(self) -> None:
        seen = threading.Event()
        # the 59th round (it starts at 590 s) blocks until the test releases it; the 60th (600 s) returns at once
        self.clock = QueuedRoundsClock(seen, block_at=59)
        self.addCleanup(self.clock.release.set)
        self.addCleanup(self.clock.after.set)
        self.tmux.on_paste = self.cli_queues_in_old_session  # queued and never echoed
        session = self.make()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: session._turns._turn.queued), "the enqueue line was not read")
        seen.set()
        self.assertTrue(wait_until(lambda: len(self.clock.sleeps) == 59), "the 590 s round did not start")
        self.assertEqual(sum(self.clock.sleeps), 590.0)
        self.assertEqual(self.results(), [], "the queued message failed before the 600 s deadline")
        self.clock.after.set()
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertEqual(result.error,
                         ATTACH_ECHO_MISSED_ERROR.format(session=UUID, seconds=ATTACH_QUEUED_MAX_SECONDS))
        self.assertIn(UUID, result.error, "the error does not name the session")
        self.assertEqual(sum(self.clock.sleeps), 600.0)
        self.assertEqual([sub for sub, _ in self.tmux.calls if sub == "paste-buffer"], ["paste-buffer"],
                         "the message was pasted again")


class IsRestoredTests(unittest.TestCase):
    def test_the_placeholder_of_a_long_paste_is_the_message_when_its_line_count_matches(self) -> None:
        twelve_lines = "\n".join(["line"] * 12)
        self.assertTrue(is_restored("[Pasted text #2 +12 lines]", twelve_lines))
        self.assertTrue(is_restored("[Pasted text #2 +11 lines]", twelve_lines))  # within one line of the count
        self.assertTrue(is_restored("[Pasted text #2 +13 lines]", twelve_lines))
        self.assertFalse(is_restored("[Pasted text #2 +5 lines]", twelve_lines))

    def test_a_bare_placeholder_is_the_message_only_for_a_one_line_text(self) -> None:
        self.assertTrue(is_restored("[Pasted text #1]", "short"))
        self.assertFalse(is_restored("[Pasted text #1]", "two\nlines"))

    def test_a_url_wrapped_in_the_middle_of_a_word_is_the_message(self) -> None:
        url = "see https://example.com/a/very/long/path/that/wraps?with=query"
        self.assertTrue(is_restored("see https://example.com/a/very/long/pa\n    th/that/wraps?with=query", url))

    def test_the_same_words_wrapped_and_indented_are_the_message(self) -> None:
        self.assertTrue(is_restored("hello there,\n    how are\n    you", "hello there, how are you"))

    def test_other_text_is_not_the_message(self) -> None:
        self.assertFalse(is_restored("hello there", "hello"))
        self.assertFalse(is_restored("[Pasted text #2 +12 lines] and more", "a long message"))

    def test_a_blank_box_holds_nothing(self) -> None:
        self.assertFalse(is_restored("", ""))
        self.assertFalse(is_restored("  \n ", "hello"))


if __name__ == "__main__":
    unittest.main()
