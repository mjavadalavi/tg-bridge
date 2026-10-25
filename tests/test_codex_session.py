"""CodexSession with a real child process (tests/fakes/fake_codex.py), STEP 27 scope: start, turn, approvals.

Scenarios: SessionStarted is emitted before turn/start; the first initialize carries the optOut list and a refused
optOut is retried once without it; thread/start and thread/resume carry approvalsReviewer 'user' and no developerInstructions; the topic RULES travel
as a leading text item of every turn/start, so a resumed thread gets them too; bypass sends approvalPolicy never with danger-full-access; a resume the server refuses is reported as
resume_failed and nothing else is sent; a normal turn yields text, usage and TurnResult OK; command approvals accept
and cancel (deny aborts the turn, the approval is withdrawn); file approvals accept and decline (no availableDecisions);
an unknown server request gets -32601 and the turn goes on with one warning line; set_mode/set_model return False and
apply to the next turn; local images are sent as localImage input; a missing launcher is a TurnResult error, not an
exception. The FAKE_CODEX_LOG file is the wire record the assertions read.
"""

import dataclasses
import json
import os
import sys
import time
import unittest
from unittest import mock

import tests.fakes as fakes_package
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.codex import approvals
from tg_bridge.agents.codex import session as codex_session
from tg_bridge.agents.codex.instructions import wrap_rules
from tg_bridge.agents.codex.session import OPT_OUT_METHODS, CodexSession
from tg_bridge.agents.events import (ApprovalRequest, ApprovalWithdrawn, Decision, SessionStarted, StatusEvent,
                                     TextEvent, ToolStart, TurnResult, TurnStatus, UsageEvent)
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

WAIT = 15.0
FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_codex.py")
RULES = "RULES-TEXT"
UNSUPPORTED = -32601


class CodexSessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.log_path = os.path.join(self.home.home, "fake-codex-in.jsonl")
        self.sessions: list[CodexSession] = []

    def tearDown(self) -> None:
        for session in self.sessions:
            session.stop(grace=2)

    def spec(self, fake: str = "normal", *, argv: tuple[str, ...] | None = None, **overrides) -> SessionSpec:
        launcher = LauncherSpec(name="codex", backend="codex", command="codex",
                                argv=argv or (sys.executable, "-I", FAKE), config_dir=self.home.home)
        env = dict(os.environ, FAKE_CODEX_MODE=fake, FAKE_CODEX_LOG=self.log_path, FAKE_CODEX_ANSWER="done")
        values = {
            "thread_id": 7, "chat_id": -100, "topic": "codex", "launcher": launcher, "cwd": self.home.home,
            "mode": "default", "model": None, "resume_id": None, "new_id": None, "allow_rules": (),
            "deny_tools": (), "system_rules": RULES, "env": env,
            "errfile_dir": os.path.join(self.home.state, "turns"),
        }
        values.update(overrides)
        return SessionSpec(**values)

    def session(self, spec: SessionSpec, sink: RecordingSink | None = None) -> tuple[CodexSession, RecordingSink]:
        sink = sink if sink is not None else RecordingSink()
        session = CodexSession(spec, sink, logger=self.logger, clock=self.clock)
        self.sessions.append(session)
        return session, sink

    def inbound(self) -> list[dict]:
        """Every frame the bridge wrote to the fake, in order (requests, notifications and replies)."""
        try:
            with open(self.log_path, encoding="utf-8") as handle:
                return [json.loads(line)["in"] for line in handle if line.strip()]
        except FileNotFoundError:
            return []

    def sent(self, method: str) -> list[dict]:
        return [frame for frame in self.inbound() if frame.get("method") == method]

    def replies(self) -> list[dict]:
        return [frame for frame in self.inbound() if "method" not in frame and "id" in frame]

    def turn_result(self, sink: RecordingSink) -> TurnResult:
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult)), "no TurnResult arrived")
        return next(e for e in sink.events if isinstance(e, TurnResult))

    def wait_results(self, sink: RecordingSink, count: int) -> bool:
        """True once sink has recorded count TurnResult events, False after WAIT seconds."""
        deadline = time.monotonic() + WAIT
        while sum(isinstance(e, TurnResult) for e in sink.events) < count:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def approval(self, sink: RecordingSink) -> ApprovalRequest:
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)), "no approval request arrived")
        return next(e for e in sink.events if isinstance(e, ApprovalRequest))


class StartTests(CodexSessionTestCase):
    def test_session_started_is_emitted_before_turn_start(self) -> None:
        session, sink = self.session(self.spec())
        session.start()
        started = [e for e in sink.events if isinstance(e, SessionStarted)]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].session_id, "fake-thread-1")
        self.assertEqual(started[0].model, "gpt-fake")
        self.assertEqual(started[0].mode, "default")
        self.assertFalse(started[0].resumed)
        self.assertEqual(session.session_id, "fake-thread-1")
        self.assertEqual([f["method"] for f in self.inbound() if "method" in f],
                         ["initialize", "initialized", "thread/start"])
        self.assertTrue(all(isinstance(e, SessionStarted) for e in sink.events), "nothing but SessionStarted so far")

    def test_first_initialize_carries_the_optout_list(self) -> None:
        session, _ = self.session(self.spec())
        session.start()
        first = self.sent("initialize")[0]
        self.assertEqual(first["params"]["capabilities"]["optOutNotificationMethods"], list(OPT_OUT_METHODS))
        self.assertEqual(first["params"]["clientInfo"]["name"], "tg-bridge")

    def test_refused_optout_is_retried_once_without_it(self) -> None:
        session, sink = self.session(self.spec("reject_optout"))
        session.start()
        initials = self.sent("initialize")
        self.assertEqual(len(initials), 2)
        self.assertIn("optOutNotificationMethods", initials[0]["params"]["capabilities"])
        self.assertNotIn("optOutNotificationMethods", initials[1]["params"]["capabilities"])
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, SessionStarted)))

    def test_thread_start_carries_reviewer_user_and_no_developer_instructions(self) -> None:
        session, _ = self.session(self.spec())
        session.start()
        params = self.sent("thread/start")[0]["params"]
        self.assertNotIn("developerInstructions", params)
        self.assertEqual(params["approvalsReviewer"], "user")
        self.assertEqual(params["approvalPolicy"], "untrusted")
        self.assertEqual(params["sandbox"], "workspace-write")
        self.assertEqual(params["cwd"], self.home.home)

    def test_bypass_mode_sends_never_and_danger_full_access_on_start_and_turn(self) -> None:
        session, sink = self.session(self.spec(mode="bypassPermissions"))
        session.start()
        params = self.sent("thread/start")[0]["params"]
        self.assertEqual(params["approvalPolicy"], "never")
        self.assertEqual(params["sandbox"], "danger-full-access")
        session.send_user("hi")
        self.turn_result(sink)
        turn = self.sent("turn/start")[0]["params"]
        self.assertEqual(turn["approvalPolicy"], "never")
        self.assertEqual(turn["sandboxPolicy"], {"type": "dangerFullAccess"})

    def test_resume_sends_thread_resume_with_rules_and_reports_the_thread(self) -> None:
        session, sink = self.session(self.spec(resume_id="fake-thread-9"))
        session.start()
        started = next(e for e in sink.events if isinstance(e, SessionStarted))
        self.assertEqual(started.session_id, "fake-thread-9")
        self.assertTrue(started.resumed)
        self.assertEqual(session.session_id, "fake-thread-9")
        self.assertEqual(self.sent("thread/start"), [])
        params = self.sent("thread/resume")[0]["params"]
        self.assertEqual(params["threadId"], "fake-thread-9")
        self.assertTrue(params["excludeTurns"])
        self.assertNotIn("developerInstructions", params)  # accepted by Codex but never reaches the model (DESIGN A.8)
        self.assertEqual(params["approvalsReviewer"], "user")
        self.assertEqual(params["approvalPolicy"], "untrusted")

    def test_resumed_turn_sends_the_rules_as_leading_turn_input(self) -> None:
        session, sink = self.session(self.spec(resume_id="fake-thread-9"))
        session.start()
        session.send_user("hello")
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        items = self.sent("turn/start")[0]["params"]["input"]
        self.assertEqual(items, [{"type": "text", "text": wrap_rules(RULES), "text_elements": []},
                                 {"type": "text", "text": "hello", "text_elements": []}])

    def test_changed_rules_reach_the_next_resumed_turn(self) -> None:
        session, sink = self.session(self.spec(resume_id="fake-thread-9", system_rules="NEW-RULES"))
        session.start()
        session.send_user("again")
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        texts = [item["text"] for item in self.sent("turn/start")[0]["params"]["input"]]
        self.assertIn(wrap_rules("NEW-RULES"), texts)
        self.assertFalse(any(RULES in text for text in texts))

    def test_empty_rules_add_no_input_item(self) -> None:
        session, sink = self.session(self.spec(resume_id="fake-thread-9", system_rules=""))
        session.start()
        session.send_user("plain")
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        items = self.sent("turn/start")[0]["params"]["input"]
        self.assertEqual(items, [{"type": "text", "text": "plain", "text_elements": []}])

    def test_resume_refused_is_resume_failed_and_nothing_more_is_sent(self) -> None:
        session, sink = self.session(self.spec("resume_missing", resume_id="gone-thread"))
        session.start()
        result = self.turn_result(sink)
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertTrue(result.resume_failed)
        self.assertIsNone(session.session_id)
        session.send_user("hello")
        self.assertEqual(self.sent("turn/start"), [])
        self.assertEqual(len([e for e in sink.events if isinstance(e, TurnResult)]), 1)

    def test_missing_launcher_is_a_turn_error_not_an_exception(self) -> None:
        session, sink = self.session(self.spec(argv=("/nonexistent/codex-binary-for-tests",)))
        session.start()
        result = self.turn_result(sink)
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertFalse(session.alive())
        session.send_user("hello")  # nothing to send to: no exception either


class TurnTests(CodexSessionTestCase):
    def test_normal_turn_yields_text_usage_and_ok_result(self) -> None:
        session, sink = self.session(self.spec(mode="bypassPermissions"))
        session.start()
        session.send_user("hello there")
        result = self.turn_result(sink)
        self.assertEqual(result.status, TurnStatus.OK)
        self.assertEqual(result.answer, "done")
        texts = [e for e in sink.events if isinstance(e, TextEvent)]
        self.assertEqual([(t.text, t.final) for t in texts], [("Working on it.\n", False), ("done", True)])
        usage = [e for e in sink.events if isinstance(e, UsageEvent)]
        self.assertEqual(usage[-1].input_tokens, 19078)
        turn = self.sent("turn/start")[0]["params"]
        self.assertEqual(turn["threadId"], "fake-thread-1")
        self.assertEqual(turn["input"], [{"type": "text", "text": wrap_rules(RULES), "text_elements": []},
                                         {"type": "text", "text": "hello there", "text_elements": []}])
        self.assertEqual(turn["approvalsReviewer"], "user")

    def test_turn_start_is_sent_with_default_mode_policy(self) -> None:
        session, sink = self.session(self.spec())
        session.start()
        session.send_user("hi")
        self.turn_result(sink)
        turn = self.sent("turn/start")[0]["params"]
        self.assertEqual(turn["approvalPolicy"], "untrusted")
        self.assertEqual(turn["sandboxPolicy"]["type"], "workspaceWrite")

    def test_local_image_is_sent_as_local_image_input(self) -> None:
        image = os.path.join(self.home.home, "photo.png")
        session, sink = self.session(self.spec(mode="bypassPermissions"))
        session.start()
        session.send_user("look at this", images=(image,))
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        items = self.sent("turn/start")[0]["params"]["input"]
        self.assertEqual(items[2], {"type": "localImage", "path": image})

    def test_unknown_server_request_gets_unsupported_and_the_turn_goes_on(self) -> None:
        session, sink = self.session(self.spec("unknown_request", mode="bypassPermissions"))
        session.start()
        session.send_user("hi")
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        warnings = [e for e in sink.events if isinstance(e, StatusEvent) and e.kind == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("item/tool/requestUserInput", warnings[0].detail)
        refusal = [r for r in self.replies() if r.get("id") == 0]
        self.assertEqual(refusal[0]["error"]["code"], UNSUPPORTED)

    def test_set_mode_and_model_return_false_and_apply_to_the_next_turn(self) -> None:
        session, sink = self.session(self.spec())
        session.start()
        session.send_user("first")
        self.turn_result(sink)
        self.assertIs(session.set_mode("bypassPermissions"), False)
        self.assertIs(session.set_model("gpt-next"), False)
        session.send_user("second")
        self.assertTrue(self.wait_results(sink, 2), "the second turn did not finish")
        first, second = self.sent("turn/start")
        self.assertEqual(first["params"]["approvalPolicy"], "untrusted")
        self.assertNotIn("model", first["params"])
        self.assertEqual(second["params"]["approvalPolicy"], "never")
        self.assertEqual(second["params"]["model"], "gpt-next")

    def test_set_mode_rejects_a_key_codex_does_not_know(self) -> None:
        session, _ = self.session(self.spec())
        with self.assertRaises(approvals.UnknownMode):
            session.set_mode("nonsense")


class ApprovalTests(CodexSessionTestCase):
    def test_command_approval_accept_is_sent_as_accept(self) -> None:
        session, sink = self.session(self.spec("command_approval"))
        session.start()
        session.send_user("run it")
        req = self.approval(sink)
        self.assertEqual(req.kind, "command")
        self.assertEqual(req.approval_id, "rpc:0")
        self.assertIn(Decision.ALLOW_ONCE, req.choices)
        self.assertIn(Decision.DENY, req.choices)
        self.assertIn("touch probe_ok.txt", req.input["command"])
        session.answer(req, Decision.ALLOW_ONCE)
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        self.assertEqual([r["result"] for r in self.replies() if r.get("id") == 0], [{"decision": "accept"}])

    def test_command_approval_deny_cancels_the_turn_and_withdraws_the_card(self) -> None:
        session, sink = self.session(self.spec("command_approval"))
        session.start()
        session.send_user("run it")
        req = self.approval(sink)
        session.answer(req, Decision.DENY)
        self.assertEqual(self.turn_result(sink).status, TurnStatus.INTERRUPTED)
        self.assertEqual([r["result"] for r in self.replies() if r.get("id") == 0], [{"decision": "cancel"}])
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalWithdrawn) and e.approval_id == "rpc:0"))

    def test_file_approval_accept_completes_the_change(self) -> None:
        session, sink = self.session(self.spec("file_approval"))
        session.start()
        session.send_user("write it")
        req = self.approval(sink)
        self.assertEqual(req.kind, "file")
        self.assertTrue(req.input["paths"][0].endswith("probe_ok.txt"))
        self.assertIn(Decision.ALLOW_ONCE, req.choices)
        session.answer(req, Decision.ALLOW_ONCE)
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        self.assertTrue(any(isinstance(e, ToolStart) and e.name == "fileChange" for e in sink.events))
        self.assertEqual([r["result"] for r in self.replies() if r.get("id") == 0], [{"decision": "accept"}])

    def test_file_approval_deny_declines_and_the_turn_goes_on(self) -> None:
        session, sink = self.session(self.spec("file_approval"))
        session.start()
        session.send_user("write it")
        req = self.approval(sink)
        session.answer(req, Decision.DENY)
        self.assertEqual(self.turn_result(sink).status, TurnStatus.OK)
        self.assertEqual([r["result"] for r in self.replies() if r.get("id") == 0], [{"decision": "decline"}])

    def test_answer_for_a_choice_that_was_not_offered_is_dropped(self) -> None:
        session, sink = self.session(self.spec())
        session.start()
        req = ApprovalRequest(approval_id="rpc:0", kind="command", tool="commandExecution", input={},
                              summary="$ x", choices=frozenset({Decision.DENY}), raw={"id": 0})
        before = len(self.inbound())
        session.answer(req, Decision.ALLOW_ONCE)
        self.assertEqual(len(self.inbound()), before)
        self.assertIn("is not offered", self.logger_text())

    def logger_text(self) -> str:
        with open(self.logger.path, encoding="utf-8") as handle:
            return handle.read()


class HomeAndInterruptTests(CodexSessionTestCase):
    def test_launcher_config_dir_wins_over_an_inherited_codex_home(self) -> None:
        base = self.spec()
        session, _ = self.session(self.spec(env=dict(base.env, CODEX_HOME="/elsewhere/.codex")))
        self.assertEqual(session._env["CODEX_HOME"], self.home.home)

    def test_inherited_codex_home_is_kept_when_the_launcher_has_no_config_dir(self) -> None:
        base = self.spec()
        spec = self.spec(env=dict(base.env, CODEX_HOME="/inherited/.codex"),
                         launcher=dataclasses.replace(base.launcher, config_dir=""))
        session, _ = self.session(spec)
        self.assertEqual(session._env["CODEX_HOME"], "/inherited/.codex")

    def test_reply_and_turn_end_share_one_interrupt_deadline(self) -> None:
        # the reply takes 0.6 s and the turn never ends: with a 1 s deadline the stop fallback comes at 1 s, not 1.6 s
        class SlowPeer:
            def request(self, method: str, params: dict, timeout: float) -> dict:
                time.sleep(0.6)
                return {}

        session, _ = self.session(self.spec())
        session.session_id, session._turn_id = "thread-1", "turn-1"
        session.alive = lambda: True
        session._peer = SlowPeer()
        stopped: list[float] = []
        session._stop_after_interrupt = lambda reason: stopped.append(time.monotonic())
        with mock.patch.object(codex_session, "INTERRUPT_TIMEOUT", 1.0):
            started = time.monotonic()
            self.assertTrue(session.interrupt())
        self.assertEqual(len(stopped), 1)
        self.assertLess(stopped[0] - started, 1.3)


if __name__ == "__main__":
    unittest.main()
