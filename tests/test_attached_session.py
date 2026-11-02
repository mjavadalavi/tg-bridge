"""AttachedSession (agents/claude/attached_session.py) with a fake tmux and a temp transcript: a turn read from the file the
paste writes, the failed paste, Escape for an interrupt, and stop() that never sends a key."""

import json
import os
import subprocess
import threading
import time
import unittest

from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec, TerminalNotReady
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.attached_session import ATTACHED_CAPABILITIES, UNREACHABLE_ERROR, AttachedSession
from tg_bridge.agents.claude.transcript_turns import ATTACH_RELOCATE_SECONDS
from tg_bridge.agents.claude.tui_turn import ECHO_MISSED_SUBTYPE, USER_ECHO_SECONDS
from tg_bridge.agents.events import ApprovalRequest, Decision, SessionStarted, TurnResult, TurnStatus
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

    def test_a_failed_paste_ends_the_turn_with_an_error_and_presses_no_enter(self) -> None:
        self.tmux.fail = ("paste-buffer",)
        session = self.make()
        session.start()
        session.send_user("hello")

        result = self.wait_for_result()
        self.assertEqual((result.status, result.error), (TurnStatus.ERROR, UNREACHABLE_ERROR))
        self.assertNotIn("send-keys", [sub for sub, _ in self.tmux.calls])

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


class RelocateClock(BlockingClock):
    """The relocation step (ATTACH_RELOCATE_SECONDS) returns at once, so a test does not wait for it; the sleep after it
    blocks until the test releases it. Every sleep is recorded, so the echo window (10 s, then 20 s) can be checked."""

    def __init__(self) -> None:
        super().__init__()
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        if threading.current_thread().name != "user-echo":  # the locate timer of the transcript is not the echo window
            self.release.wait(WAIT_SECONDS)
            return
        self.sleeps.append(seconds)
        if seconds == ATTACH_RELOCATE_SECONDS:
            return
        self.release.wait()


class RelocationTests(unittest.TestCase):
    """A /clear or /resume in the tab: the CLI writes the message's echo into another transcript. The turn finds it within
    the first 10 s and follows that file; with no such file the turn ends with the 30 s echo error."""

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
        deadline = time.monotonic() + WAIT_SECONDS
        while self.clock.sleeps != [ATTACH_RELOCATE_SECONDS, USER_ECHO_SECONDS - ATTACH_RELOCATE_SECONDS] and \
                time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.clock.sleeps, [10.0, 20.0], "the echo window is not 30 seconds in two steps")
        self.assertEqual(self.results(), [], "the turn ended before the echo window")
        self.clock.release.set()
        result = self.wait_for_result()
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertEqual(result.subtype, ECHO_MISSED_SUBTYPE)
        self.assertEqual(session.session_id, UUID, "the session moved without a transcript")

    def test_an_echo_read_in_the_claims_transcript_does_not_relocate(self) -> None:
        self.tmux.on_paste = self.cli_answers_in_old_session
        session = self.make()
        session.send_user("hello")
        result = self.wait_for_result()
        self.assertEqual(result.answer, "ANSWER")
        self.assertEqual(session.session_id, UUID)
        self.assertFalse(any(isinstance(e, SessionStarted) and e.session_id == NEW_UUID for e in self.events))


if __name__ == "__main__":
    unittest.main()
