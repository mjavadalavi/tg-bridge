"""ClaudeTuiSession and build_tui_args (the tui transport): the interactive TUI driven through a PTY, with real child
processes (tests/fakes/fake_claude_tui.py) that write transcripts in the real shape. Scenarios: the argv has the TUI
flags and none of the print ones, AskUserQuestion and KillShell are disallowed once each, allow lists only outside
bypass, --resume or --session-id by the id given. open_session picks the class by transport. End to end: a text turn
(usage counted once per message, one TurnResult), a tool turn (ToolStart and ToolEnd from the transcript), an API
error (TurnResult ERROR with the error text), an interrupt (INTERRUPTED once), a resume that the TUI refuses
(NO_CONVERSATION in stderr_tail), the folder trust dialog accepted with Down and Enter (also when a late remount
swallows the first Down, and never with Enter while the cursor is not on Yes), a resume that starts after the
old lines, and the environment scrub with the socket removed on stop. The interrupt's two ends (the transcript line, and
the 3 s fallback) are pinned with a GatedClock, so the fallback fires only when the test releases it. The transcript
lookup after 15 s and the hook wait before a message (60 s bound) are pinned the same way. Permissions: allow and deny
round trips through answer(), and bypass set live with set_mode. Every test uses short waits; the clock is a FakeClock
unless a test needs a gate or a scaled real sleep, so the TUI's pauses cost nothing. The view mirror (the raw output in
<view_dir>/<thread>.pty) and the PTY size are checked in ViewMirrorTests."""

import io
import json
import os
import sys
import threading
import time
import unittest
import uuid
from dataclasses import replace
from unittest import mock

import tests.fakes as fakes_package
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents import pty_mirror
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.claude.hookbridge import ENV_SOCKET, TURN_ENDED, permission_allow, permission_deny, run_hook_client
from tg_bridge.agents.claude.protocol import DENY_MESSAGE
from tg_bridge.agents.claude.session import BYPASS_MODE, KILL_SWITCH_TOOL, ClaudeSession
from tg_bridge.agents.claude.translate import NO_CONVERSATION
from tg_bridge.agents.claude.trust_dialog import _trust_cursor_on_yes
from tg_bridge.agents.claude.tui_gate import HOOK_POLL_SECONDS, HOOK_WAIT_SECONDS
from tg_bridge.agents.claude.tui_session import (INTERRUPT_FALLBACK_SECONDS, TRANSCRIPT_WAIT_SECONDS, ClaudeTuiSession,
                                                 build_tui_args)
from tg_bridge.agents.claude.tui_turn import (ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE, STARTUP_DIALOG_ERROR,
                                              USER_ECHO_SECONDS)
from tg_bridge.agents.events import (ApprovalRequest, ApprovalWithdrawn, Decision, ProcessExited, SessionStarted,
                                     TextEvent, ToolEnd, ToolStart, TurnResult, TurnStatus, UsageEvent)
from tg_bridge.config import BACKEND_CLAUDE, TRANSPORT_PRINT, TRANSPORT_TUI, LauncherSpec
from tg_bridge.logging_ import Logger

WAIT = 15.0
SCALE = 0.1  # ScaledClock: a sleep really takes this share of its length
FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_claude_tui.py")
SETTINGS = '{"hooks":{"PermissionRequest":[]}}'
PRINT_ONLY_FLAGS = ("-p", "--input-format", "--output-format", "--verbose", "--permission-prompt-tool",
                    "--forward-subagent-text")


def flag_value(args: list[str], flag: str) -> str | None:
    """The argument after flag, or None when the flag is absent."""
    return args[args.index(flag) + 1] if flag in args else None


def pure_spec(**overrides: object) -> SessionSpec:
    """A SessionSpec for the pure argv builder. Keyword arguments replace individual fields."""
    launcher = LauncherSpec(name="teamclaude", backend=BACKEND_CLAUDE, command="teamclaude", argv=("teamclaude",),
                            config_dir="/cfg", transport=TRANSPORT_TUI)
    values: dict = {
        "thread_id": 7, "chat_id": -100, "topic": "side", "launcher": launcher, "cwd": "/work", "mode": "default",
        "model": None, "resume_id": None, "new_id": "new-1", "allow_rules": ("Read", "Glob"),
        "deny_tools": ("Bash(rm *)",), "system_rules": "RULES", "env": {}, "errfile_dir": "/state/turns",
    }
    values.update(overrides)
    return SessionSpec(**values)


def results(sink: RecordingSink) -> list[TurnResult]:
    return [e for e in sink.events if isinstance(e, TurnResult)]


def only_result(sink: RecordingSink) -> TurnResult:
    """The one TurnResult of the turn. Asserts there is exactly one."""
    found = results(sink)
    if len(found) != 1:
        raise AssertionError(f"expected one TurnResult, got {found!r}")
    return found[0]


class GatedClock(FakeClock):
    """A FakeClock whose sleep of gate_seconds blocks until the test sets release. A timer of exactly that length
    therefore fires only when the test lets it, and never by itself: the gate has no timeout of its own, so a test
    that forgets to release it fails on its own wait, and the test's cleanup releases it. The echo watch's sleep
    (USER_ECHO_SECONDS) has a gate of its own, echo_release, so a test that opens one gate does not open the other.
    Other sleeps are instant."""

    def __init__(self, gate_seconds: float | None) -> None:
        super().__init__()
        self.gate_seconds = gate_seconds
        self.release = threading.Event()
        self.echo_release = threading.Event()

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if seconds == self.gate_seconds:
            self.release.wait()
        elif seconds == USER_ECHO_SECONDS:
            self.echo_release.wait()


class ScaledClock(FakeClock):
    """A FakeClock that also really sleeps, SCALE of each length, so a wait loop runs in real time and can be raced."""

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        time.sleep(seconds * SCALE)


def wait_until(predicate: object, timeout: float = WAIT) -> bool:
    """True once predicate() holds, False after timeout real seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_events(sink: RecordingSink, kind: type) -> int:
    return sum(isinstance(e, kind) for e in list(sink.events))


class TrustCursorTests(unittest.TestCase):
    def test_the_cursor_is_on_yes_only_when_the_glyph_sits_before_the_yes_choice(self) -> None:
        on_no = "Entertoconfirm·Esctocancel❯No,exitYes,Itrustthisfolder"
        on_yes = "Entertoconfirm·Esctocancel" + "No,exit❯Yes,Itrustthisfolder"
        self.assertFalse(_trust_cursor_on_yes(on_no))
        self.assertTrue(_trust_cursor_on_yes(on_yes))
        self.assertFalse(_trust_cursor_on_yes("Yes,Itrustthisfolder"))  # no glyph drawn yet


class BuildTuiArgsTests(unittest.TestCase):
    def test_the_launcher_leads_the_tui_flags_and_settings_come_last(self) -> None:
        args = build_tui_args(pure_spec(), SETTINGS)
        self.assertEqual(args[0], "teamclaude")
        self.assertEqual(args[-2:], ["--settings", SETTINGS])
        self.assertEqual(flag_value(args, "--permission-mode"), "default")
        self.assertIn("--allow-dangerously-skip-permissions", args)
        self.assertEqual(flag_value(args, "--append-system-prompt"), "RULES")
        self.assertEqual(flag_value(args, "--name"), "tg:side")

    def test_no_print_only_flag_is_passed(self) -> None:
        args = build_tui_args(pure_spec(), SETTINGS)
        for flag in PRINT_ONLY_FLAGS:
            self.assertNotIn(flag, args)

    def test_ask_user_question_and_the_kill_switch_are_each_disallowed_once(self) -> None:
        args = build_tui_args(pure_spec(deny_tools=(KILL_SWITCH_TOOL, "Bash(rm *)")), SETTINGS)
        self.assertEqual(flag_value(args, "--disallowedTools"), f"{KILL_SWITCH_TOOL},AskUserQuestion,Bash(rm *)")

    def test_allowed_tools_are_passed_outside_bypass_only(self) -> None:
        self.assertEqual(flag_value(build_tui_args(pure_spec(), SETTINGS), "--allowedTools"), "Read,Glob")
        bypass = build_tui_args(pure_spec(mode=BYPASS_MODE), SETTINGS)
        self.assertNotIn("--allowedTools", bypass)

    def test_a_resume_id_is_passed_as_resume_and_a_new_id_as_session_id(self) -> None:
        resumed = build_tui_args(pure_spec(resume_id="old-1", new_id=None), SETTINGS)
        self.assertEqual(flag_value(resumed, "--resume"), "old-1")
        self.assertNotIn("--session-id", resumed)
        fresh = build_tui_args(pure_spec(), SETTINGS)
        self.assertEqual(flag_value(fresh, "--session-id"), "new-1")
        self.assertNotIn("--resume", fresh)
        neither = build_tui_args(pure_spec(new_id=None), SETTINGS)
        self.assertNotIn("--resume", neither)
        self.assertNotIn("--session-id", neither)

    def test_the_model_flag_is_passed_only_when_set(self) -> None:
        self.assertEqual(flag_value(build_tui_args(pure_spec(model="opus"), SETTINGS), "--model"), "opus")
        self.assertNotIn("--model", build_tui_args(pure_spec(), SETTINGS))

    def test_the_memory_dir_joins_the_hook_in_the_one_settings_flag(self) -> None:
        args = build_tui_args(pure_spec(memory_dir="/state/memory/7"), SETTINGS)
        self.assertEqual(args.count("--settings"), 1)
        self.assertEqual(json.loads(flag_value(args, "--settings")),
                         {"hooks": {"PermissionRequest": []}, "autoMemoryDirectory": "/state/memory/7"})

    def test_without_a_memory_dir_the_settings_are_the_hook_alone(self) -> None:
        self.assertEqual(flag_value(build_tui_args(pure_spec(), SETTINGS), "--settings"), SETTINGS)


class OpenSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.backend = ClaudeBackend(logger=self.logger, clock=FakeClock())

    def test_the_tui_transport_opens_the_tui_session(self) -> None:
        spec = pure_spec()
        session = self.backend.open_session(spec, RecordingSink())
        self.assertIsInstance(session, ClaudeTuiSession)

    def test_the_print_transport_keeps_the_print_session(self) -> None:
        spec = pure_spec(launcher=LauncherSpec(name="claude", backend=BACKEND_CLAUDE, command="claude",
                                               argv=("claude",), config_dir="/cfg", transport=TRANSPORT_PRINT))
        session = self.backend.open_session(spec, RecordingSink())
        self.assertIsInstance(session, ClaudeSession)
        self.assertNotIsInstance(session, ClaudeTuiSession)


class TuiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = GatedClock(gate_seconds=None)  # only the echo watch is held: it waits until a test releases it
        self.addCleanup(self.clock.echo_release.set)
        self.sessions: list[ClaudeTuiSession] = []

    def tearDown(self) -> None:
        for session in self.sessions:
            session.stop(grace=1)

    def spec(self, scenario: str = "text", *, resume_id: str | None = None, trust: bool = False,
             extra_env: dict | None = None) -> SessionSpec:
        env = dict(os.environ)
        env.update({"FAKE_TUI_CONFIG_DIR": self.home.home, "FAKE_TUI_SCENARIO": scenario})
        if trust:
            env["FAKE_TUI_TRUST"] = "1"
        env.update(extra_env or {})
        launcher = LauncherSpec(name="fake", backend=BACKEND_CLAUDE, command="fake",
                                argv=(sys.executable, "-I", FAKE), config_dir=self.home.home, transport=TRANSPORT_TUI)
        return SessionSpec(thread_id=7, chat_id=-100, topic="side", launcher=launcher, cwd=self.home.home,
                           mode="default", model=None, resume_id=resume_id,
                           new_id=None if resume_id else str(uuid.uuid4()), allow_rules=("Read",), deny_tools=(),
                           system_rules="RULES", env=env, errfile_dir=os.path.join(self.home.state, "turns"))

    def open(self, scenario: str = "text", *, clock: FakeClock | None = None,
             **kwargs: object) -> tuple[RecordingSink, ClaudeTuiSession]:
        sink = RecordingSink()
        use = clock if clock is not None else self.clock
        if isinstance(use, GatedClock):
            self.addCleanup(use.echo_release.set)
        session = ClaudeTuiSession(self.spec(scenario, **kwargs), sink, logger=self.logger, clock=use)
        self.sessions.append(session)
        return sink, session

    def transcript_path(self, sid: str) -> str:
        return os.path.join(self.home.home, "projects", history.encoded_cwd(os.path.realpath(self.home.home)),
                            sid + ".jsonl")

    def wait_result(self, sink: RecordingSink) -> None:
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), WAIT), "no TurnResult")

    def write_transcript(self, path: str, lines: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(line) + "\n" for line in lines))

    def log_text(self) -> str:
        return read_text(self.logger.path)


class ScenarioTests(TuiTestCase):
    def test_a_text_turn_emits_the_start_the_text_the_usage_and_one_result(self) -> None:
        sink, session = self.open("text")
        session.start()
        self.assertIsInstance(sink.events[0], SessionStarted)
        self.assertEqual(sink.events[0].session_id, session.session_id)
        session.send_user("hello")
        self.wait_result(sink)
        turn = only_result(sink)
        self.assertEqual((turn.status, turn.answer, turn.duration_ms, turn.subtype),
                         (TurnStatus.OK, "ANSWER", 1234, "turn_duration"))
        usage = [e for e in sink.events if isinstance(e, UsageEvent)]
        self.assertEqual(len(usage), 1)
        self.assertEqual((usage[0].input_tokens, usage[0].output_tokens, usage[0].cached_tokens), (10, 5, 120))
        self.assertIsNone(usage[0].cost_usd)
        self.assertFalse(usage[0].cost_known)
        self.assertLess(sink.events.index(usage[0]), sink.events.index(turn))
        self.assertEqual([e.text for e in sink.events if isinstance(e, TextEvent)], ["ANSWER"])
        self.assertTrue(session.end_gracefully(WAIT))
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        self.assertEqual(next(e for e in sink.events if isinstance(e, ProcessExited)).returncode, 0)

    def test_a_tool_turn_emits_tool_start_and_end_and_counts_a_repeated_message_once(self) -> None:
        sink, session = self.open("tool")
        session.start()
        session.send_user("run it")
        self.wait_result(sink)
        starts = [e for e in sink.events if isinstance(e, ToolStart)]
        ends = [e for e in sink.events if isinstance(e, ToolEnd)]
        self.assertEqual([(s.tool_id, s.name) for s in starts], [("tu_1", "Bash")])
        self.assertEqual([(e.tool_id, e.ok) for e in ends], [("tu_1", True)])
        self.assertEqual([e.text for e in sink.events if isinstance(e, TextEvent)], ["Running it.", "DONE"])
        turn = only_result(sink)
        self.assertEqual((turn.status, turn.answer), (TurnStatus.OK, "DONE"))
        usage = next(e for e in sink.events if isinstance(e, UsageEvent))
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.cached_tokens), (20, 10, 240))

    def test_an_api_error_turn_ends_as_error_with_the_error_text(self) -> None:
        sink, session = self.open("api_error")
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        turn = only_result(sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)
        self.assertIsNone(turn.answer)
        self.assertIn("6 Claude Code sessions", turn.error)
        self.assertFalse(turn.resume_failed)
        self.assertEqual(sum(isinstance(e, UsageEvent) for e in sink.events), 1)

    def test_the_interrupt_line_ends_the_turn_and_the_fallback_never_fires(self) -> None:
        clock = GatedClock(gate_seconds=INTERRUPT_FALLBACK_SECONDS)
        self.addCleanup(clock.release.set)
        sink, session = self.open("slow", clock=clock)
        session.start()
        session.send_user("long job")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ToolStart), WAIT))
        self.assertTrue(session.interrupt())
        self.wait_result(sink)  # the gate is still shut: the transcript's interrupt line alone ended the turn
        self.assertTrue(wait_until(lambda: INTERRUPT_FALLBACK_SECONDS in clock.sleeps))  # the fallback is waiting
        clock.release.set()
        time.sleep(0.2)  # the fallback wakes now and must find the turn already ended
        turn = only_result(sink)
        self.assertEqual((turn.status, turn.answer, turn.subtype), (TurnStatus.INTERRUPTED, None, "interrupted"))
        ends = [e for e in sink.events if isinstance(e, ToolEnd)]
        self.assertEqual([e.ok for e in ends], [False])
        self.assertLess(sink.events.index(ends[0]), sink.events.index(turn))
        self.assertEqual(count_events(sink, UsageEvent), 1)

    def test_without_an_interrupt_line_the_fallback_ends_the_turn_after_three_seconds(self) -> None:
        clock = GatedClock(gate_seconds=INTERRUPT_FALLBACK_SECONDS)
        self.addCleanup(clock.release.set)
        sink, session = self.open("slow_quiet", clock=clock)
        session.start()
        session.send_user("long job")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ToolStart), WAIT))
        self.assertTrue(session.interrupt())
        self.assertTrue(wait_until(lambda: INTERRUPT_FALLBACK_SECONDS in clock.sleeps))
        time.sleep(0.2)
        self.assertEqual(results(sink), [])  # the 3 s have not passed yet
        clock.release.set()
        self.wait_result(sink)
        self.assertEqual(only_result(sink).status, TurnStatus.INTERRUPTED)
        self.assertIn(INTERRUPT_FALLBACK_SECONDS, clock.sleeps)  # the wait was the 3 s of the fallback

    def test_every_text_block_of_one_message_is_in_the_answer(self) -> None:
        sink, session = self.open("two_blocks")
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual([e.text for e in sink.events if isinstance(e, TextEvent)], ["FIRST", "SECOND"])
        self.assertEqual(only_result(sink).answer, "FIRST\nSECOND")

    def test_a_transcript_outside_the_cwd_folder_is_found_after_fifteen_seconds(self) -> None:
        clock = GatedClock(gate_seconds=TRANSCRIPT_WAIT_SECONDS)
        self.addCleanup(clock.release.set)
        sink, session = self.open("text", clock=clock, extra_env={"FAKE_TUI_FOLDER": "elsewhere"})
        session.start()
        session.send_user("hello")
        alt_path = os.path.join(self.home.home, "projects", "elsewhere", session.session_id + ".jsonl")
        self.assertTrue(wait_until(lambda: os.path.isfile(alt_path) and "stop_hook_summary" in read_text(alt_path)))
        time.sleep(0.2)
        self.assertEqual(results(sink), [])  # the bridge follows the cwd folder, where nothing is written
        clock.release.set()
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")
        self.assertIn("transcript found at", self.log_text())

    def test_a_resumed_transcript_found_by_lookup_starts_after_its_old_lines(self) -> None:
        sid = str(uuid.uuid4())
        alt_path = os.path.join(self.home.home, "projects", "elsewhere", sid + ".jsonl")
        self.write_transcript(alt_path, [
            {"type": "user", "message": {"role": "user", "content": "old question"}},
            {"type": "assistant", "message": {"id": "msg_old", "content": [{"type": "text", "text": "OLD ANSWER"}]}},
            {"type": "system", "subtype": "turn_duration", "durationMs": 999}])
        clock = GatedClock(gate_seconds=TRANSCRIPT_WAIT_SECONDS)
        self.addCleanup(clock.release.set)
        sink, session = self.open("text", clock=clock, resume_id=sid, extra_env={"FAKE_TUI_FOLDER": "elsewhere"})
        session.start()
        session.send_user("again")
        self.assertTrue(wait_until(lambda: "stop_hook_summary" in read_text(alt_path)))
        clock.release.set()
        self.wait_result(sink)
        self.assertEqual([e.text for e in sink.events if isinstance(e, TextEvent)], ["ANSWER"])
        self.assertEqual(only_result(sink).duration_ms, 1234)  # the old turn_duration (999) was not re-read
        self.assertIn("transcript found at", self.log_text())

    def test_an_interrupt_outside_a_turn_emits_nothing(self) -> None:
        sink, session = self.open("text")
        session.start()
        self.assertTrue(session.interrupt())
        time.sleep(0.3)
        self.assertEqual(results(sink), [])

    def test_a_resume_the_tui_refuses_is_reported_through_stderr_tail(self) -> None:
        sink, session = self.open("no_conversation", resume_id="gone-sid")
        session.start()  # returns once the TUI has exited
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        exited = next(e for e in sink.events if isinstance(e, ProcessExited))
        self.assertEqual(exited.returncode, 1)
        self.assertIn(NO_CONVERSATION, exited.stderr_tail)
        self.assertIn(NO_CONVERSATION, session.stderr_tail())

    def test_the_folder_trust_dialog_is_accepted_with_down_then_enter(self) -> None:
        clock = ScaledClock()  # the fake redraws in real time, so the cursor is read after a real pause
        sink, session = self.open("text", clock=clock, trust=True)
        session.start()
        self.assertIn(0.8, clock.sleeps)  # the pause between Down and the cursor check
        self.assertIsInstance(sink.events[0], SessionStarted)
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")

    def test_a_late_trust_dialog_remount_that_swallows_the_first_down_still_reaches_the_prompt(self) -> None:
        clock = ScaledClock()
        sink, session = self.open("text", clock=clock, trust=True, extra_env={"FAKE_TUI_TRUST_REMOUNT": "1"})
        session.start()
        self.assertIn("accepted the folder trust dialog", self.log_text())
        self.assertIsInstance(sink.events[0], SessionStarted)
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")

    def test_enter_is_never_pressed_while_the_cursor_is_not_on_yes(self) -> None:
        clock = ScaledClock()
        sink, session = self.open("text", clock=clock, trust=True, extra_env={"FAKE_TUI_TRUST_DEAF": "1"})
        with mock.patch.object(session, "press_enter", wraps=session.press_enter) as enter:
            session.start()
        enter.assert_not_called()  # Enter on 'No, exit' would have ended the fake with rc 1
        self.assertTrue(session.alive())
        self.assertIn("the folder trust dialog was not accepted", self.log_text())
        self.assertEqual(results(sink), [])
        self.assertEqual(clock.sleeps.count(0.8), 3)  # one pause after each of the three Down presses

    def test_a_dialog_drawn_in_two_writes_is_still_accepted_and_the_turn_completes(self) -> None:
        clock = ScaledClock()  # the fake's second half comes after a real pause, while the cursor is on 'No, exit'
        sink, session = self.open("text", clock=clock, trust=True, extra_env={"FAKE_TUI_TRUST_SPLIT": "1"})
        session.start()
        self.assertIn("accepted the folder trust dialog", self.log_text())
        self.assertIsInstance(sink.events[0], SessionStarted)
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")

    def test_a_resume_starts_after_the_lines_that_were_already_there(self) -> None:
        sid = str(uuid.uuid4())
        os.makedirs(os.path.dirname(self.transcript_path(sid)), exist_ok=True)
        old = [{"type": "user", "message": {"role": "user", "content": "old question"}},
               {"type": "assistant", "message": {"id": "msg_old", "content": [{"type": "text", "text": "OLD ANSWER"}]}},
               {"type": "system", "subtype": "turn_duration", "durationMs": 999}]
        with open(self.transcript_path(sid), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(line) + "\n" for line in old))
        sink, session = self.open("text", resume_id=sid)
        session.start()
        started = sink.events[0]
        self.assertTrue(started.resumed)
        self.assertEqual(started.session_id, sid)
        session.send_user("again")
        self.wait_result(sink)
        self.assertEqual([e.text for e in sink.events if isinstance(e, TextEvent)], ["ANSWER"])
        self.assertEqual(only_result(sink).duration_ms, 1234)

    def test_the_child_gets_a_scrubbed_environment_and_the_socket_goes_away_on_stop(self) -> None:
        env_out = os.path.join(self.home.home, "env.json")
        sink, session = self.open("text", extra_env={
            "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CONFIG_DIR": "/x/cfg",
            "FAKE_TUI_ENV_OUT": env_out})
        session.start()
        with open(env_out, encoding="utf-8") as f:
            env = json.load(f)
        self.assertNotIn("CLAUDECODE", env)
        self.assertFalse([k for k in env if k.startswith("CLAUDE_CODE_")])
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "/x/cfg")
        self.assertEqual(env["TERM"], "xterm-256color")
        socket_path = env[ENV_SOCKET]
        self.assertTrue(socket_path.endswith(".sock"))
        self.assertTrue(os.path.exists(socket_path))
        session.stop(grace=1)
        self.assertFalse(os.path.exists(socket_path))

    def test_images_are_refused_and_the_live_model_and_mode_are_not_offered(self) -> None:
        _, session = self.open("text")
        with self.assertRaises(ValueError):
            session.send_user("hi", images=("photo.png",))
        self.assertFalse(session.set_model("opus"))
        self.assertFalse(session.set_mode("default"))


class StartupDialogTests(TuiTestCase):
    def test_the_external_imports_dialog_is_declined_with_esc_and_the_turn_runs(self) -> None:
        clock = ScaledClock()  # the dialogs are drawn in real time, so the screen settles between the reads
        sink, session = self.open("text", clock=clock, trust=True, extra_env={"FAKE_TUI_EXTERNAL": "1"})
        with mock.patch.object(session, "press_esc", wraps=session.press_esc) as esc, \
                mock.patch.object(session, "press_enter", wraps=session.press_enter) as enter:
            session.start()
        self.assertEqual(esc.call_count, 1)
        self.assertEqual(enter.call_count, 1)  # the trust dialog's Enter only: the external dialog gets no Enter
        log = self.log_text()
        self.assertIn("accepted the folder trust dialog", log)
        self.assertIn("declined the external imports dialog", log)
        self.assertIn("prompt ready", log)
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")

    def test_an_unknown_confirm_dialog_gets_no_keys_and_is_reported(self) -> None:
        clock = ScaledClock()
        sink, session = self.open("text", clock=clock, extra_env={"FAKE_TUI_UNKNOWN": "1"})
        with mock.patch.object(session, "press_esc", wraps=session.press_esc) as esc, \
                mock.patch.object(session, "press_down", wraps=session.press_down) as down, \
                mock.patch.object(session, "press_enter", wraps=session.press_enter) as enter, \
                mock.patch.object(session, "type_text", wraps=session.type_text) as typed:
            session.start()
        for keys in (esc, down, enter, typed):
            keys.assert_not_called()
        self.assertIn("unknown dialog on startup: ", self.log_text())
        self.assertIn("no prompt: an unknown dialog is open, nothing was typed", self.log_text())
        self.assertTrue(session.alive())
        self.assertEqual(results(sink), [])

    def test_a_message_after_an_unknown_dialog_is_not_typed_and_the_turn_ends_at_once_with_the_startup_error(self) -> None:
        sink, session = self.open("text", clock=ScaledClock(), extra_env={"FAKE_TUI_UNKNOWN": "1"})
        session.start()
        with mock.patch.object(session, "type_text", wraps=session.type_text) as typed:
            session.send_user("hello")
        typed.assert_not_called()  # no key reaches the dialog the bridge does not know
        turn = only_result(sink)  # already there: the turn ended inside send_user, not after a wait
        self.assertEqual((turn.status, turn.error), (TurnStatus.ERROR, STARTUP_DIALOG_ERROR))
        self.assertIn("tg-bridge view", turn.error)

    def test_the_echo_net_is_not_armed_without_a_transcript_to_follow(self) -> None:
        sink = RecordingSink()
        spec = replace(self.spec("text"), new_id=None)  # no session id: there is no transcript to read the echo from
        session = ClaudeTuiSession(spec, sink, logger=self.logger, clock=self.clock)
        self.sessions.append(session)
        session.start()
        session.send_user("hello")
        time.sleep(0.3)  # the TUI runs the message; nothing here follows it
        self.assertNotIn(USER_ECHO_SECONDS, self.clock.sleeps)  # the net never started
        self.assertEqual(results(sink), [])  # so it reports no error for a message that was typed

    def test_a_message_that_never_reaches_the_transcript_ends_the_turn_with_an_error_after_thirty_seconds(self) -> None:
        sink, session = self.open("text", extra_env={"FAKE_TUI_SWALLOW": "1"})  # the TUI takes the keys, no line is written
        session.start()
        session.send_user("hello")
        self.assertTrue(wait_until(lambda: USER_ECHO_SECONDS in self.clock.sleeps))  # the watch is waiting
        time.sleep(0.2)
        self.assertEqual(results(sink), [])  # not one moment before the 30 s
        self.clock.echo_release.set()
        self.wait_result(sink)
        turn = only_result(sink)
        self.assertEqual((turn.status, turn.error, turn.subtype),
                         (TurnStatus.ERROR, ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE))
        self.assertIn("send it again or /stop", turn.error)
        self.assertEqual(count_events(sink, UsageEvent), 1)

    def test_the_echo_watch_does_not_end_a_turn_whose_message_was_read(self) -> None:
        sink, session = self.open("slow")
        session.start()
        session.send_user("long job")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ToolStart), WAIT))  # the message was read: the turn runs
        self.assertTrue(wait_until(lambda: USER_ECHO_SECONDS in self.clock.sleeps))
        self.clock.echo_release.set()
        time.sleep(0.2)
        self.assertEqual(results(sink), [])  # the watch fired and did nothing: the turn is still running
        self.assertTrue(session.interrupt())
        self.wait_result(sink)
        self.assertEqual(only_result(sink).status, TurnStatus.INTERRUPTED)

    def test_an_sdk_shaped_prompt_line_alone_keeps_the_turn_past_the_echo_watch(self) -> None:
        sink, session = self.open("echo_only")  # the default fake: the bridge's prompt line is turnOrigin 'sdk'
        session.start()
        session.send_user("quiet job")
        self.assertTrue(wait_until(lambda: USER_ECHO_SECONDS in self.clock.sleeps))  # the watch is waiting
        self.assertTrue(wait_until(lambda: session._turn.echoed))  # the tail has read the prompt line
        self.clock.echo_release.set()
        time.sleep(0.2)
        self.assertEqual(results(sink), [])  # the prompt line is the echo: the watch does nothing
        self.assertTrue(session.interrupt())
        self.wait_result(sink)
        self.assertEqual(only_result(sink).status, TurnStatus.INTERRUPTED)

    def test_the_echo_watch_does_not_end_a_turn_whose_human_shaped_message_was_read(self) -> None:
        sink, session = self.open("slow", extra_env={"FAKE_TUI_PROMPT_ORIGIN": "human"})  # an interactive prompt's line
        session.start()
        session.send_user("long job")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ToolStart), WAIT))
        self.assertTrue(wait_until(lambda: USER_ECHO_SECONDS in self.clock.sleeps))
        self.clock.echo_release.set()
        time.sleep(0.2)
        self.assertEqual(results(sink), [])
        self.assertTrue(session.interrupt())
        self.wait_result(sink)
        self.assertEqual(only_result(sink).status, TurnStatus.INTERRUPTED)


class PermissionTests(TuiTestCase):
    def test_an_allow_answer_lets_the_tool_run(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertEqual((req.tool, req.kind, req.input), ("Bash", "command", {"command": "echo hi"}))
        session.answer(req, Decision.ALLOW_ONCE)
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "DONE")
        self.assertEqual([e.ok for e in sink.events if isinstance(e, ToolEnd)], [True])

    def test_a_deny_answer_carries_the_message_to_the_tool(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.DENY, deny_message="too risky")
        self.wait_result(sink)
        ends = [e for e in sink.events if isinstance(e, ToolEnd)]
        self.assertEqual([(e.ok, e.detail) for e in ends], [(False, "too risky")])
        self.assertEqual(only_result(sink).answer, "DONE")

    def test_an_owner_deny_without_a_message_uses_the_default_text(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.DENY)
        self.wait_result(sink)
        self.assertEqual([e.detail for e in sink.events if isinstance(e, ToolEnd)], [DENY_MESSAGE])

    def test_an_interrupt_closes_a_permission_request_that_is_still_open(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertTrue(session.interrupt())  # the fake is inside its hook, so no transcript line ends the turn
        self.wait_result(sink)  # the fallback ends it, and the open request is closed before that is reported
        self.assertEqual(only_result(sink).status, TurnStatus.INTERRUPTED)
        session.answer(req, Decision.ALLOW_ONCE)
        self.assertIn("unknown or closed request", self.log_text())

    def raw_request(self, session: ClaudeTuiSession, extra: dict | None = None) -> tuple[threading.Thread, io.StringIO]:
        """A second hook client on the session's socket, as a background subagent's hook would be: it waits for its
        answer, which the test gives through the session's ApprovalRequest."""
        payload = {"session_id": session.session_id, "cwd": self.home.home, "tool_name": "Bash",
                   "tool_input": {"command": "date"}, "permission_suggestions": [], **(extra or {})}
        out = io.StringIO()
        thread = threading.Thread(target=run_hook_client, daemon=True,
                                  args=(io.StringIO(json.dumps(payload)), out, {ENV_SOCKET: session._server.path}))
        thread.start()
        return thread, out

    def test_an_interrupt_withdraws_and_denies_every_open_request(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: count_events(sink, ApprovalRequest) == 1, WAIT))
        client, out = self.raw_request(session)
        self.assertTrue(sink.wait_for(lambda e: count_events(sink, ApprovalRequest) == 2, WAIT))
        requested = [e.approval_id for e in sink.events if isinstance(e, ApprovalRequest)]
        self.assertTrue(session.interrupt())
        self.wait_result(sink)
        withdrawn = [e.approval_id for e in sink.events if isinstance(e, ApprovalWithdrawn)]
        self.assertEqual(sorted(withdrawn), sorted(requested))  # both cards close, one withdrawal each
        self.assertEqual(session._server.pending_count(), 0)
        client.join(WAIT)
        self.assertEqual(json.loads(out.getvalue()), permission_deny(TURN_ENDED))  # the raw request was denied too

    def test_an_interrupt_leaves_a_background_agents_request_open(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: count_events(sink, ApprovalRequest) == 1, WAIT))
        client, out = self.raw_request(session, {"agent_id": "agent-1"})
        self.assertTrue(sink.wait_for(lambda e: count_events(sink, ApprovalRequest) == 2, WAIT))
        requested = [e.approval_id for e in sink.events if isinstance(e, ApprovalRequest)]
        self.assertTrue(session.interrupt())
        self.wait_result(sink)
        withdrawn = [e.approval_id for e in sink.events if isinstance(e, ApprovalWithdrawn)]
        self.assertEqual(withdrawn, [requested[0]])  # only the main turn's request is withdrawn
        self.assertEqual(session._server.pending_count(), 1)
        agent_req = next(e for e in sink.events if isinstance(e, ApprovalRequest) and e.approval_id == requested[1])
        session.answer(agent_req, Decision.ALLOW_ONCE)
        client.join(WAIT)
        self.assertEqual(json.loads(out.getvalue()), permission_allow())

    def test_typing_waits_while_a_permission_request_is_pending(self) -> None:
        sink, session = self.open("permission", clock=ScaledClock())
        session.start()
        with mock.patch.object(session, "type_text", wraps=session.type_text) as typed:
            session.send_user("run it")
            self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
            second = threading.Thread(target=session.send_user, args=("next",), daemon=True)
            second.start()
            time.sleep(0.3)
            self.assertTrue(second.is_alive())  # the wait loop is running
            self.assertEqual(typed.call_args_list, [mock.call("run it")])  # no key goes to the TUI's dialog
            req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
            session.answer(req, Decision.ALLOW_ONCE)
            second.join(WAIT)
            self.assertFalse(second.is_alive())
        self.assertEqual(typed.call_args_list, [mock.call("run it"), mock.call("next")])

    def test_a_request_still_pending_after_the_bound_is_denied_and_the_message_is_typed(self) -> None:
        sink, session = self.open("permission")  # FakeClock: the 60 s bound passes at once
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        session.send_user("next")
        self.assertEqual(self.clock.sleeps.count(HOOK_POLL_SECONDS), round(HOOK_WAIT_SECONDS / HOOK_POLL_SECONDS))
        self.assertTrue(wait_until(lambda: count_events(sink, ToolEnd) >= 1))
        self.assertIn(TURN_ENDED, [e.detail for e in list(sink.events) if isinstance(e, ToolEnd)])
        self.assertTrue(wait_until(lambda: count_events(sink, ApprovalRequest) == 2))  # the second message was typed
        self.assertIn("still pending before the next message", self.log_text())

    def test_bypass_set_live_allows_every_request_without_a_card(self) -> None:
        sink, session = self.open("permission")
        session.start()
        self.assertFalse(session.set_mode("default"))
        self.assertTrue(session.set_mode(BYPASS_MODE))
        session.send_user("run it")
        self.wait_result(sink)
        self.assertFalse([e for e in sink.events if isinstance(e, ApprovalRequest)])
        self.assertEqual([e.ok for e in sink.events if isinstance(e, ToolEnd)], [True])
        self.assertEqual(only_result(sink).answer, "DONE")


class StopLifecycleTests(TuiTestCase):
    def test_a_pending_permission_is_denied_before_the_first_key_on_stop(self) -> None:
        sink, session = self.open("permission")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), WAIT))
        order: list[str] = []
        real_write, real_cancel = session.write, session._server.cancel_all

        def write(data: str, timeout: float | None = None) -> bool:
            order.append("key")
            return real_write(data, timeout)

        def cancel(message: str, *, main_only: bool = False) -> list[str]:
            order.append("deny")
            return real_cancel(message, main_only=main_only)

        with mock.patch.object(session, "write", side_effect=write), \
                mock.patch.object(session._server, "cancel_all", side_effect=cancel):
            session.stop(grace=1)
        self.assertIn("key", order)  # the stop did type keys, so the order below is not vacuous
        self.assertEqual(order[0], "deny")
        self.assertLess(order.index("deny"), order.index("key"))

    def test_a_stop_before_start_spawns_nothing_and_leaves_no_socket(self) -> None:
        env_out = os.path.join(self.home.home, "env.json")  # the fake TUI writes this file when it starts
        _, session = self.open("text", extra_env={"FAKE_TUI_ENV_OUT": env_out})
        self.assertIsNone(session.stop(grace=1))
        session.start()
        self.assertFalse(os.path.exists(env_out))
        self.assertFalse(session.alive())
        self.assertIsNone(session.stop(grace=1))
        self.assertFalse(os.path.exists(session._server.path))


class ViewMirrorTests(TuiTestCase):
    """The view mirror and the terminal size of a session. The mirror is <view_dir>/<thread>.pty: the reset first, then the
    raw output of the TUI (the trust dialog text, the prompt glyph), and it is closed when the process has exited. With no
    view_dir there is no mirror and no file. The child's pty has the configured size, before the TUI starts."""

    def setUp(self) -> None:
        super().setUp()
        self.view_dir = os.path.join(self.home.state, "view")
        os.makedirs(self.view_dir)

    def open_spec(self, spec: SessionSpec) -> tuple[RecordingSink, ClaudeTuiSession]:
        sink = RecordingSink()
        session = ClaudeTuiSession(spec, sink, logger=self.logger, clock=self.clock)
        self.sessions.append(session)
        return sink, session

    def pty_bytes(self) -> bytes:
        with open(os.path.join(self.view_dir, "7.pty"), "rb") as f:
            return f.read()

    def test_the_tui_output_is_mirrored_to_the_thread_pty_file_after_a_reset(self) -> None:
        _, session = self.open_spec(replace(self.spec("text", trust=True), view_dir=self.view_dir))
        session.start()
        self.assertTrue(wait_until(lambda: "❯".encode() in self.pty_bytes()), "the prompt never reached the file")
        data = self.pty_bytes()
        self.assertTrue(data.startswith(b"\x1bc"))
        self.assertIn(b"Quick safety check", data)  # the trust dialog, drawn by the TUI before the prompt

    def test_the_mirror_is_closed_when_the_tui_exits_and_keeps_the_whole_output(self) -> None:
        sink, session = self.open_spec(replace(self.spec("text"), view_dir=self.view_dir))
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        self.assertTrue(session.end_gracefully(WAIT))
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        self.assertIsNone(session._mirror._file, "the mirror must be closed once the process has exited")
        self.assertTrue(self.pty_bytes().startswith(b"\x1bc"))

    def test_a_session_without_a_view_dir_has_no_mirror_and_writes_no_file(self) -> None:
        sink, session = self.open_spec(self.spec("text"))
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        self.assertIsNone(session._mirror)
        self.assertEqual(os.listdir(self.view_dir), [])

    def test_the_pty_of_the_tui_has_the_configured_rows_and_columns(self) -> None:
        size_out = os.path.join(self.home.home, "size.json")
        _, session = self.open_spec(replace(self.spec("text", extra_env={"FAKE_TUI_SIZE_OUT": size_out}),
                                            tui_size=(30, 120)))
        session.start()
        self.assertTrue(wait_until(lambda: os.path.exists(size_out)), "the TUI did not report its size")
        with open(size_out, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"rows": 30, "cols": 120})

    def test_each_restart_of_the_mirror_makes_the_tui_redraw_at_the_session_size(self) -> None:
        """With a small cap the trust dialog and the prompt pass it, so the file restarts. Each restart sends the TUI two
        SIGWINCH: the size one column narrower, then the configured size, as the TUI reads the size on each."""
        winch = os.path.join(self.home.home, "winch.jsonl")
        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 64):
            _, session = self.open_spec(replace(self.spec("text", trust=True, extra_env={"FAKE_TUI_WINCH_OUT": winch}),
                                                view_dir=self.view_dir, tui_size=(30, 120)))
            session.start()
            self.assertTrue(wait_until(lambda: os.path.exists(winch) and len(read_text(winch).splitlines()) >= 2),
                            "the TUI was never asked to redraw")
        sizes = [json.loads(line) for line in read_text(winch).splitlines()]
        for i, size in enumerate(sizes):
            self.assertEqual(size, {"rows": 30, "cols": 119 if i % 2 == 0 else 120})
        session.stop(grace=1)


if __name__ == "__main__":
    unittest.main()
