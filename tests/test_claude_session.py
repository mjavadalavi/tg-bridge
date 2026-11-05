"""ClaudeSession with real child processes (tests/fakes/fake_claude.py), ported from fakecli/test_topic_session.py.

Scenarios: initialize is answered and the session id is learned from the stream; an unhandled can_use_tool is
denied and the CLI says so; approval answers carry the input, the answers and the always-permissions; interrupt is
answered; stop() leaves no process and no group member behind (stubborn child, TERM-ignoring straggler); a resume
the CLI refuses ends the process and is reported; the reader thread cannot wait for a reply. The translator is
replaced by a recording one, because translation belongs to STEP 17.
"""

import json
import os
import sys
import threading
import time
import unittest

import tests.fakes as fakes_package
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.claude.session import ClaudeSession, build_args
from tg_bridge.agents.events import ApprovalRequest, Decision, ProcessExited
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

WAIT = 15.0
FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_claude.py")
WRAPPER = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "launchers", "fake_claude.sh")
PERMISSION_SUGGESTION = {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "echo *"}],
                         "behavior": "allow", "destination": "session"}


def _wait_until(predicate, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _group_gone(pgid: int, timeout: float = 3.0) -> bool:
    """True once no process belongs to process group pgid (zombies still count as present until reaped)."""
    def gone() -> bool:
        try:
            os.killpg(pgid, 0)
            return False
        except ProcessLookupError:
            return True
    return _wait_until(gone, timeout)


class RecordingTranslator:
    """Keeps every raw object the session routes to the translator. With approval=True it claims can_use_tool
    requests by returning an ApprovalRequest, so the session does not deny them itself."""

    def __init__(self, approval: bool = False) -> None:
        self.approval = approval
        self.raw: list[dict] = []
        self.hook = None  # optional callable(session, event), run on the reader thread
        self.session = None
        self._cond = threading.Condition()

    def translate(self, event: dict) -> list:
        with self._cond:
            self.raw.append(event)
            self._cond.notify_all()
        if self.hook is not None:
            self.hook(self.session, event)
        if self.approval and event.get("type") == "control_request" and event["request"]["subtype"] == "can_use_tool":
            return [ApprovalRequest(approval_id=event["request_id"], kind="tool", tool=event["request"]["tool_name"],
                                    input=event["request"]["input"], summary="echo", choices=frozenset(
                                        {Decision.ALLOW_ONCE, Decision.DENY}))]
        return []

    def wait_raw(self, predicate, timeout: float = WAIT) -> dict | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for event in self.raw:
                    if predicate(event):
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)


class ClaudeSessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.sessions: list[ClaudeSession] = []

    def tearDown(self) -> None:
        for session in self.sessions:
            session.stop(grace=1)

    def launcher(self, argv=None, name: str = "fake") -> LauncherSpec:
        return LauncherSpec(name=name, backend="claude", command=name, argv=tuple(argv or (sys.executable, "-I", FAKE)),
                            config_dir=self.home.home)

    def spec(self, **overrides) -> SessionSpec:
        values = {
            "thread_id": 7, "chat_id": -100, "topic": "side", "launcher": self.launcher(), "cwd": self.home.home,
            "mode": "default", "model": None, "resume_id": None, "new_id": None,
            "allow_rules": ("Read", "Glob", "Grep"), "deny_tools": (), "system_rules": "RULES",
            "env": dict(os.environ), "errfile_dir": os.path.join(self.home.state, "turns"),
        }
        values.update(overrides)
        return SessionSpec(**values)

    def session(self, spec: SessionSpec, *, translator=None, sink=None, **kwargs) -> tuple:
        sink = sink if sink is not None else RecordingSink()
        translator = translator if translator is not None else RecordingTranslator()
        session = ClaudeSession(spec, sink, logger=self.logger, clock=self.clock, translator=translator, **kwargs)
        translator.session = session
        self.sessions.append(session)
        return session, sink, translator

    def log_text(self) -> str:
        try:
            with open(self.logger.path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""


class BuildArgsParityTests(ClaudeSessionTestCase):
    """build_args output for various flag combinations."""

    def test_killshell_is_always_denied_and_no_allow_flag_without_rules(self) -> None:
        """Ported labels:
        C: no deny_tools -> KillShell only
        """
        argv = build_args(self.spec(allow_rules=(), deny_tools=()))
        self.assertEqual(argv[argv.index("--disallowedTools") + 1], "KillShell")
        self.assertNotIn("--allowedTools", argv)
        self.assertNotIn("--resume", argv)
        self.assertNotIn("--session-id", argv)

    def test_name_flag_carries_the_topic_name(self) -> None:
        argv = build_args(self.spec(topic="budget"))
        self.assertEqual(argv[argv.index("--name") + 1], "tg:budget")

    def test_memory_dir_adds_one_settings_flag_with_the_memory_folder(self) -> None:
        argv = build_args(self.spec(memory_dir="/state/memory/7"))
        self.assertEqual(argv.count("--settings"), 1)
        self.assertEqual(json.loads(argv[argv.index("--settings") + 1]), {"autoMemoryDirectory": "/state/memory/7"})

    def test_without_a_memory_dir_there_is_no_settings_flag(self) -> None:
        self.assertNotIn("--settings", build_args(self.spec()))


class StartAndTurnTests(ClaudeSessionTestCase):
    def test_start_is_initialized_and_session_id_is_learned_from_the_stream(self) -> None:
        """Ported labels:
        session_id learned from first event
        interrupt returns True on success reply
        """
        session, _, translator = self.session(self.spec())
        session.start()
        self.assertTrue(session.alive())
        self.assertIsNotNone(session.pgid)
        self.assertNotIn("initialize", self.log_text())  # answered, so no timeout or failure line
        session.send_user("hello there")
        self.assertTrue(_wait_until(lambda: session.session_id == "sess-123"))
        self.assertTrue(translator.wait_raw(lambda e: e.get("type") == "system" and e.get("subtype") == "init"))
        self.assertTrue(session.interrupt(), "a control request is answered: interrupt() is True")

    def test_garbage_lines_do_not_stop_the_turn(self) -> None:
        """Ported labels:
        1: garbage lines do not stop the turn
        1: reader finished (end-of-stream path ran)
        """
        spec = self.spec(env=dict(os.environ, FAKE_GARBAGE="1"))
        session, sink, translator = self.session(spec, translator=RecordingTranslator(approval=True))
        session.start()
        session.send_user("hello there")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.ALLOW_ONCE)
        result = translator.wait_raw(lambda e: e.get("type") == "result")
        self.assertIsNotNone(result)
        self.assertEqual(result["result"], "ok:allow")
        session.stop(grace=1)
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited)), "the reader ended and reported the exit")

    def test_unhandled_permission_is_denied_and_the_cli_reports_it(self) -> None:
        """Ported labels:
        result event received
        stderr captured to per-turn err file
        """
        session, _, translator = self.session(self.spec())
        session.start()
        session.send_user("hello there")
        result = translator.wait_raw(lambda e: e.get("type") == "result")
        self.assertIsNotNone(result)
        self.assertEqual(result["result"], "ok:deny")
        self.assertIn("fake-stderr-line", session.stderr_tail())

    def test_approval_answer_sends_input_and_answers(self) -> None:
        """The request's own input goes back as updatedInput next to the answers. For AskUserQuestion that input holds
        the questions, so the original questions are kept in updatedInput (the fixture uses a Bash input).

        Ported labels:
        F: the original questions are kept in updatedInput
        can_use_tool routed to on_control_request
        result is ok:allow
        """
        session, sink, translator = self.session(self.spec(env=dict(os.environ, FAKE_ECHO="1")),
                                                 translator=RecordingTranslator(approval=True))
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.ALLOW_ONCE, answers={"Which?": "this one"})
        result = translator.wait_raw(lambda e: e.get("type") == "result")
        self.assertEqual(result["result"].split(" ", 1)[0], "ok:allow")
        echoed = json.loads(result["result"].split(" ", 1)[1])
        self.assertEqual(echoed, {"updatedInput": {"command": "echo hi", "answers": {"Which?": "this one"}}})

    def test_approval_deny_sends_the_owner_message(self) -> None:
        session, sink, translator = self.session(self.spec(env=dict(os.environ, FAKE_ECHO="1")),
                                                 translator=RecordingTranslator(approval=True))
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.DENY)
        result = translator.wait_raw(lambda e: e.get("type") == "result")
        self.assertEqual(result["result"], 'ok:deny {"message": "denied by owner from Telegram"}')

    def test_bridge_deny_reasons_reach_the_cli_as_the_message(self) -> None:
        """The three bridge-side refusals (closed turn, no questions, card not posted) send their live text to the CLI,
        not the owner's DENY_MESSAGE. Each runs one real CLI turn with its own approval request."""
        for reason in ("turn already ended", "no questions to answer", "could not reach Telegram to ask"):
            with self.subTest(reason=reason):
                session, sink, translator = self.session(self.spec(env=dict(os.environ, FAKE_ECHO="1")),
                                                         translator=RecordingTranslator(approval=True))
                session.start()
                session.send_user("run it")
                self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
                req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
                session.answer(req, Decision.DENY, deny_message=reason)
                result = translator.wait_raw(lambda e: e.get("type") == "result")
                self.assertEqual(result["result"], 'ok:deny ' + json.dumps({"message": reason}))

    def test_always_answer_forwards_updated_permissions(self) -> None:
        session, sink, translator = self.session(self.spec(env=dict(os.environ, FAKE_ECHO="1")),
                                                 translator=RecordingTranslator(approval=True))
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        session.answer(req, Decision.ALLOW_ALWAYS, updated_permissions=[PERMISSION_SUGGESTION])
        result = translator.wait_raw(lambda e: e.get("type") == "result")
        echoed = json.loads(result["result"].split(" ", 1)[1])
        self.assertEqual(echoed["updatedPermissions"], [PERMISSION_SUGGESTION])

    def test_interrupt_is_answered_and_graceful_stop_exits_zero(self) -> None:
        """Ported labels:
        graceful stop after interrupt, rc 0
        stop(): process gone, rc=0
        interrupt returns True on success reply
        """
        session, sink, _ = self.session(self.spec())
        session.start()
        self.assertTrue(session.interrupt(timeout=5))
        session.stop(grace=1)
        self.assertFalse(session.alive())
        self.assertEqual(session.stop(grace=1), 0)
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited) and e.returncode == 0))

    def test_set_mode_and_set_model_are_live_before_and_refused_without_a_process(self) -> None:
        session, _, _ = self.session(self.spec())
        self.assertFalse(session.set_mode("plan"), "no process yet: not live")
        session.start()
        self.assertTrue(session.set_mode("bypassPermissions"))
        self.assertTrue(session.set_model("sonnet"))

    def test_send_user_before_start_is_refused_quietly(self) -> None:
        session, _, _ = self.session(self.spec())
        session.send_user("too early")  # no stdin yet: nothing is written and nothing raises

    def test_images_are_refused_because_the_capability_is_off(self) -> None:
        session, _, _ = self.session(self.spec())
        with self.assertRaises(ValueError):
            session.send_user("look", images=("/tmp/x.jpg",))

    def test_reader_thread_cannot_wait_for_a_reply(self) -> None:
        translator = RecordingTranslator()
        seen: dict = {}

        def hook(session, event) -> None:
            if event.get("type") == "system" and event.get("subtype") == "init" and "value" not in seen:
                seen["value"] = session.set_mode("plan")  # would wait for itself without the guard

        translator.hook = hook
        session, _, _ = self.session(self.spec(), translator=translator)
        session.start()
        session.send_user("hello")
        self.assertTrue(_wait_until(lambda: "value" in seen), "the reader thread returned, no deadlock")
        self.assertIs(seen["value"], False)


class StopAndExitTests(ClaudeSessionTestCase):
    def test_stubborn_child_is_killed_and_its_group_is_empty(self) -> None:
        """Ported labels:
        stubborn fake killed by ladder, rc=-9
        whole process group (incl. child sleep) is gone
        on_exit called once with rc
        """
        spec = self.spec(launcher=self.launcher([sys.executable, "-I", FAKE, "stubborn"]))
        sink = RecordingSink()
        session, sink, _ = self.session(spec, sink=sink, initialize_timeout=0.5)
        session.start()
        pgid = session.pgid
        rc = session.stop(grace=1)
        self.assertFalse(session.alive())
        self.assertIsNotNone(rc)
        self.assertLess(rc, 0, "killed by a signal, not a clean exit")
        self.assertTrue(_group_gone(pgid), "no process of the group (incl. the child sleep) survives")
        self.assertIn("initialize timed out", self.log_text())
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited)))
        exits = [e.returncode for e in sink.events if isinstance(e, ProcessExited)]
        self.assertEqual(exits, [rc], "the exit is reported once, with the return code of stop()")

    def test_term_ignoring_straggler_in_the_group_is_gone_after_stop(self) -> None:
        """Ported labels:
        TERM-ignoring straggler gone after stop()
        """
        spec = self.spec(env=dict(os.environ, FAKE_STRAGGLER="1"))
        session, _, _ = self.session(spec)
        session.start()
        session.send_user("hello")
        pgid = session.pgid
        session.stop(grace=1)
        self.assertTrue(_group_gone(pgid))

    def test_resume_refused_by_the_cli_ends_the_process_and_is_reported(self) -> None:
        spec = self.spec(resume_id="u-1", env=dict(os.environ, FAKE_NOCONV="1"))
        sink = RecordingSink()
        session, sink, translator = self.session(spec, sink=sink)
        session.start()
        exited = sink.wait_for(lambda e: isinstance(e, ProcessExited) and e.returncode == 1)
        self.assertTrue(exited, "the process exit is reported to the sink")
        self.assertIn("initialize failed", self.log_text())
        self.assertIn("No conversation found", session.stderr_tail())
        self.assertFalse(session.alive())

    def test_wrapper_script_launches_the_fake_like_a_topic_launcher(self) -> None:
        spec = self.spec(launcher=self.launcher(["/bin/sh", WRAPPER]))
        session, _, _ = self.session(spec)
        session.start()
        session.send_user("hello")
        self.assertTrue(_wait_until(lambda: session.session_id == "sess-123"))


if __name__ == "__main__":
    unittest.main()
