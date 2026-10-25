"""ClaudeBackend against the fake Claude CLI: the shared BackendContract plus the Claude-only behaviour.

The contract checks (start, text turn, approval round trip, withdraw, interrupt, stop, resume, refused resume) are in
tests/contract.py and run here with real child processes. The class-specific tests cover what only Claude does: the
exact result text, the interrupt's CLI subtype, the 'always' rule and its permission update, the shared rule grammar,
the one-shot title call (including the process-group kill on timeout), the resume hint, modes and models.
Scenarios ported from fakecli/test_permissions.py, test_p2b_cards.py and test_tool_log.py are in test_rules.py and
test_claude_translate.py; this module adds the backend-level checks. The live file is not loaded here.
"""

import dataclasses
import json
import os
import sys
import time
import unittest
from unittest import mock

import tests.fakes as fakes_package
from tests.contract import BackendContract
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.backend import CLAUDE_CAPABILITIES, ClaudeBackend
from tg_bridge.agents.events import ApprovalRequest, Decision, SessionStarted, TurnResult, TurnStatus
from tg_bridge.config import MODEL_CHOICES, Config, LauncherSpec
from tg_bridge.logging_ import Logger

FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_claude.py")
WAIT = 15.0
SCENARIO_ENV = {
    "text": {"FAKE_NOWAIT": "1"},                              # ask emitted, result 'ok:unanswered' at once
    "approval": {"FAKE_ECHO": "1"},                            # waits for the answer, result echoes it
    "withdraw": {"FAKE_CANCEL": "1"},                          # ask withdrawn, then result
    "hang": {"FAKE_NEVER": "1", "FAKE_INTERRUPT_RESULT": "1"},  # never finishes; the CLI answers an interrupt
    "resume_refused": {"FAKE_NOCONV": "1"},                    # resume of an unknown id: result says so, exit 1
}


class ClaudeBackendTests(BackendContract, unittest.TestCase):
    """The shared contract (tests/contract.py) plus the Claude-only behaviour, against the fake CLI."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.contract_root = self.home.home
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()

    def contract_backend(self) -> ClaudeBackend:
        return ClaudeBackend(logger=self.logger, clock=self.clock, title_argv=(sys.executable, "-I", FAKE))

    def contract_launcher(self) -> LauncherSpec:
        return LauncherSpec(name="fake", backend="claude", command="fake",
                            argv=(sys.executable, "-I", FAKE), config_dir=self.home.home)

    def contract_env(self, scenario: str) -> dict[str, str]:
        return dict(SCENARIO_ENV[scenario])

    def test_text_turn_answer_is_the_result_line(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "text")
        session.start()
        session.send_user("hi")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=WAIT))
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertEqual(result.answer, "ok:unanswered")

    def test_interrupted_turn_carries_the_cli_subtype(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("long")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, SessionStarted), timeout=WAIT))
        self.assertTrue(session.interrupt())
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=10))
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertIs(result.status, TurnStatus.ERROR)
        self.assertEqual(result.subtype, "error_during_execution")

    def test_always_rule_and_permission_update_come_from_the_suggestion(self) -> None:
        backend = self.contract_backend()
        suggestion = {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "echo *"}],
                      "behavior": "allow", "destination": "localSettings"}
        spec = dataclasses.replace(
            self.contract_spec(backend, "approval"),
            env={**os.environ, "FAKE_ECHO": "1", "FAKE_SUGGESTIONS": json.dumps([suggestion])})
        sink = RecordingSink()
        session = backend.open_session(spec, sink)
        self.addCleanup(session.stop, 2)
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertEqual(backend.rules.rule_for(req), "Bash(echo *)")
        update = backend.rules.updated_permissions(req)
        self.assertEqual(update, [dict(suggestion, destination="session")], "destination moved to session")
        session.answer(req, Decision.ALLOW_ALWAYS)  # the session derives the permission update itself
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=WAIT))
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertEqual(json.loads(result.answer.split(" ", 1)[1])["updatedPermissions"], update)

    def test_rules_use_the_shared_grammar(self) -> None:
        backend = self.contract_backend()
        req = ApprovalRequest(approval_id="x", kind="command", tool="Bash", input={"command": "git status -s"},
                              summary="git", choices=frozenset({Decision.ALLOW_ONCE}))
        self.assertTrue(backend.rules.matches("Bash(git status *)", req, self.home.home))
        chained = dataclasses.replace(req, input={"command": "git status; curl x"})
        self.assertFalse(backend.rules.matches("Bash(git status *)", chained, self.home.home))
        self.assertIsNone(backend.rules.rule_for(req), "no 'always' button offered: no rule")

    def test_capabilities_match_the_claude_cli(self) -> None:
        backend = self.contract_backend()
        self.assertIs(backend.capabilities, CLAUDE_CAPABILITIES)
        self.assertTrue(backend.capabilities.live_mode and backend.capabilities.live_model)
        self.assertTrue(backend.capabilities.preassigned_id)
        self.assertFalse(backend.capabilities.images)

    def test_modes_are_the_four_claude_permission_modes_in_picker_order(self) -> None:
        keys = tuple(mode.key for mode in self.contract_backend().modes())
        self.assertEqual(keys, ("bypassPermissions", "acceptEdits", "default", "plan"))

    def test_models_come_from_config_or_the_builtin_list(self) -> None:
        backend = self.contract_backend()
        self.assertEqual(backend.models(Config({}, home=self.home.home)), MODEL_CHOICES)
        custom = Config({"models": ["opus", "haiku"]}, home=self.home.home)
        self.assertEqual(backend.models(custom), ("opus", "haiku"))

    def test_resume_hint_names_the_claude_resume_command(self) -> None:
        self.assertEqual(self.contract_backend().resume_hint("abc-123", "/x"), "claude --resume abc-123")

    def test_one_shot_returns_one_cleaned_line(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_TITLE": "  Fixing   the\n login  bug  "}):
            title = self.contract_backend().one_shot("prompt", timeout=WAIT)
        self.assertEqual(title, "Fixing the login bug")

    def test_one_shot_is_cut_to_sixty_characters(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_TITLE": "word " * 40}):
            title = self.contract_backend().one_shot("prompt", timeout=WAIT)
        self.assertIsNotNone(title)
        self.assertLessEqual(len(title), 60)

    def test_one_shot_timeout_returns_none_and_kills_the_whole_group(self) -> None:
        """Ported labels:
        title_session returned after the timeout
        hung launcher and its child are gone
        """
        pidfile = os.path.join(self.home.state, "grandchild.pid")
        env = {"FAKE_ONESHOT": "hang", "FAKE_PIDFILE": pidfile}
        with mock.patch.dict(os.environ, env):
            started = time.monotonic()
            title = self.contract_backend().one_shot("prompt", timeout=1.0)
        self.assertIsNone(title)
        self.assertLess(time.monotonic() - started, 12, "the call gives up at its timeout")
        with open(pidfile, encoding="utf-8") as handle:
            grandchild = int(handle.read())
        self.assertTrue(_gone(grandchild), "the grandchild in the title group is gone too")

    def test_one_shot_without_a_launcher_is_unsupported(self) -> None:
        self.assertIsNone(ClaudeBackend(logger=self.logger, clock=self.clock).one_shot("p", timeout=1.0))

    def test_one_shot_missing_launcher_returns_none_and_logs(self) -> None:
        backend = ClaudeBackend(logger=self.logger, clock=self.clock, title_argv=("/nonexistent/title-launcher",))
        self.assertIsNone(backend.one_shot("p", timeout=1.0))
        with open(self.logger.path, encoding="utf-8") as handle:
            self.assertIn("title:", handle.read())


def _gone(pid: int, timeout: float = 5.0) -> bool:
    """True when pid no longer exists or is a zombie (reparented and not yet reaped by init)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                state = handle.read().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    unittest.main()
