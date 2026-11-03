"""AttachedSession (agents/claude/attached_session.py) with a fake tmux and a temp transcript: a turn read from the file the
paste writes, the failed paste, Escape for an interrupt, and stop() that never sends a key."""

import dataclasses
import json
import os
import subprocess
import threading
import time
import unittest

from tests.support import TempHome
from tests.turn_support import wait_until
from tg_bridge.agents.base import AttachSpec, SessionSpec, TerminalNotReady
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.attached_session import (ATTACHED_CAPABILITIES, UNREACHABLE_ERROR, AttachedSession,
                                                     pane_reason)
from tg_bridge.agents.claude.tmux_pane import REASON_GONE, REASON_IDLE, REASON_TYPED
from tg_bridge.agents.claude.transcript_turns import ATTACH_QUEUED_MAX_SECONDS, ATTACH_RELOCATE_SECONDS
from tg_bridge.agents.claude.tui_turn import ATTACH_ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE, USER_ECHO_SECONDS
from tg_bridge.agents.events import (ApprovalRequest, Decision, SessionStarted, TextEvent, ToolStart,
                                    TurnResult, TurnStatus)
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
    """sleep() blocks until the test ends, so no echo net or interrupt fallback fires during a test by itself."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.release.wait(WAIT_SECONDS)


class FakeTmux:
    """Stands in for tmux. display-message answers with the pane id, capture-pane with screen, and paste-buffer runs on_paste
    (the CLI writing its echo and answer). The subcommands named in fail exit 1."""

    def __init__(self, *, screen: str = CLI_BOX, fail: tuple[str, ...] = (), on_paste=None) -> None:
        self.pane_shown = PANE  # the pane tmux shows under this id; another id means the pane is gone
        self.screen = screen
        self.fail = fail
        self.on_paste = on_paste
        self.calls: list[tuple[str, str | None]] = []

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
        return subprocess.CompletedProcess(args, 0, out, "")

    def keys(self) -> list[str]:
        """The subcommands that type into the pane."""
        return [sub for sub, _text in self.calls if sub in ("load-buffer", "paste-buffer", "send-keys")]


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
        self.tmux = FakeTmux(on_paste=self.cli_answers)
        self.spec = SessionSpec(thread_id=1, chat_id=2, topic="t", launcher=LauncherSpec(
            name="l", backend="claude", command="c", argv=("c",), config_dir=self.config_dir), cwd=self.cwd,
            mode="default", model=None, resume_id=UUID, new_id=None, allow_rules=(), deny_tools=(), system_rules="",
            env={}, errfile_dir=self.cwd)

    def make(self, still_attached=None) -> AttachedSession:
        session = AttachedSession(self.spec, self.events.append, socket="", pane=PANE, uuid=UUID,
                                  config_dir=self.config_dir, cwd=self.cwd, logger=self, clock=self.clock,
                                  run=self.tmux, still_attached=still_attached or (lambda: True))
        self.addCleanup(session.stop)
        return session

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
        self.assertTrue(session.interrupt())
        self.assertEqual(self.tmux.calls[-1], ("send-keys", None))

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


if __name__ == "__main__":
    unittest.main()
