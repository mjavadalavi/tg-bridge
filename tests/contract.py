"""BackendContract: the behaviour every AgentBackend must show, run against a real child process (DESIGN section 5).

This is a mixin, not a TestCase, so unittest does not collect it. A concrete suite combines it with unittest.TestCase
and provides the hooks below. The scenarios are names; each backend maps them to whatever its fake CLI needs (env
flags for the Claude fake, FAKE_CODEX_MODE for the Codex fake). The checks here are backend-neutral:

  start -> SessionStarted with an id, a text turn completes OK, an approval round trip, a withdrawn approval,
  interrupt -> the turn ends (not OK) within INTERRUPT_WAIT seconds, stop leaves no process in the group,
  resume keeps the stored id, a refused resume is reported as resume_failed.

Hooks a suite must implement:
  contract_backend()          a fresh AgentBackend built with the suite's logger and clock
  contract_launcher()         the LauncherSpec whose argv runs the fake CLI
  contract_env(scenario)      extra environment for one scenario (name -> value)
  contract_root               a temp directory that outlives the test (cwd, stderr files)
"""

import os
import time
import uuid
from tg_bridge.agents.base import AgentBackend, AgentSession, Capabilities, SessionSpec
from tg_bridge.agents.events import (
    ApprovalRequest,
    ApprovalWithdrawn,
    Decision,
    SessionStarted,
    TurnResult,
    TurnStatus,
)
from tg_bridge.config import LauncherSpec
from tests.fakes.backend import RecordingSink

START_WAIT = 15.0          # a child process may take a moment to start and answer
INTERRUPT_WAIT = 10.0      # an interrupted turn must end within this many seconds
GROUP_WAIT = 3.0           # the process group must be empty this long after stop()

SCENARIOS = ("text", "approval", "withdraw", "hang", "resume_refused")


def group_gone(pgid: int, timeout: float = GROUP_WAIT) -> bool:
    """True once no process belongs to process group pgid (a zombie leader counts as gone once reaped)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)


class BackendContract:
    """Backend-neutral checks. Mix in before unittest.TestCase and implement the hooks."""

    contract_root: str

    # ---- hooks ----------------------------------------------------------------------------------------

    def contract_backend(self) -> AgentBackend:
        raise NotImplementedError

    def contract_launcher(self) -> LauncherSpec:
        raise NotImplementedError

    def contract_env(self, scenario: str) -> dict[str, str]:
        raise NotImplementedError

    # ---- helpers --------------------------------------------------------------------------------------

    def contract_spec(self, backend: AgentBackend, scenario: str, resume_id: str | None = None) -> SessionSpec:
        """A SessionSpec for one scenario: default mode, a pre-assigned id when the backend supports one."""
        assert scenario in SCENARIOS, scenario
        preassign = backend.capabilities.preassigned_id and resume_id is None
        return SessionSpec(
            thread_id=1, chat_id=-100, topic="contract", launcher=self.contract_launcher(),
            cwd=self.contract_root, mode="default", model=None, resume_id=resume_id,
            new_id=str(uuid.uuid4()) if preassign else None, allow_rules=("Read",), deny_tools=(),
            system_rules="RULES", env={**os.environ, **self.contract_env(scenario)},
            errfile_dir=os.path.join(self.contract_root, "turns"),
        )

    def contract_open(self, backend: AgentBackend, scenario: str,
                      resume_id: str | None = None) -> tuple[AgentSession, RecordingSink, SessionSpec]:
        """Open a session for a scenario and make sure it is stopped when the test ends."""
        spec = self.contract_spec(backend, scenario, resume_id)
        sink = RecordingSink()
        session = backend.open_session(spec, sink)
        self.addCleanup(session.stop, 2)
        return session, sink, spec

    # ---- contract -------------------------------------------------------------------------------------

    def test_contract_capabilities_modes_and_models_are_declared(self) -> None:
        backend = self.contract_backend()
        self.assertIsInstance(backend.capabilities, Capabilities)
        keys = [mode.key for mode in backend.modes()]
        self.assertTrue(keys, "at least one permission mode")
        self.assertEqual(len(keys), len(set(keys)), "mode keys are unique")
        self.assertIn("default", keys)
        self.assertTrue(callable(backend.rules.matches))
        self.assertTrue(callable(backend.rules.rule_for))

    def test_contract_start_and_first_turn_report_the_session_id(self) -> None:
        backend = self.contract_backend()
        session, sink, spec = self.contract_open(backend, "text")
        session.start()
        self.assertTrue(session.alive(), "the process is running after start()")
        session.send_user("hello there")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, SessionStarted), timeout=START_WAIT))
        started = next(e for e in sink.events if isinstance(e, SessionStarted))
        self.assertTrue(started.session_id, "SessionStarted carries a non-empty id")
        self.assertEqual(session.session_id, started.session_id)
        if spec.new_id is not None:
            self.assertEqual(started.session_id, spec.new_id, "a pre-assigned id is the one the CLI uses")

    def test_contract_text_turn_completes_ok(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "text")
        session.start()
        session.send_user("say something")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=START_WAIT))
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertIs(result.status, TurnStatus.OK)
        self.assertTrue(result.answer, "a finished turn carries the answer text")

    def test_contract_approval_round_trip(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "approval")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=START_WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertIn(Decision.ALLOW_ONCE, req.choices)
        self.assertIn(Decision.DENY, req.choices)
        session.answer(req, Decision.ALLOW_ONCE)
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=START_WAIT))
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertIs(result.status, TurnStatus.OK)

    def test_contract_withdrawn_approval_is_reported(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "withdraw")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=START_WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalWithdrawn), timeout=START_WAIT))
        withdrawn = next(e for e in sink.events if isinstance(e, ApprovalWithdrawn))
        self.assertEqual(withdrawn.approval_id, req.approval_id)
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=START_WAIT))

    def test_contract_interrupt_ends_the_turn_within_ten_seconds(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("take your time")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, SessionStarted), timeout=START_WAIT))
        self.assertTrue(session.interrupt(), "the interrupt is accepted while the turn runs")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=INTERRUPT_WAIT),
                        "the interrupted turn ends with a TurnResult")
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertIsNot(result.status, TurnStatus.OK, "an interrupted turn is never reported as OK")

    def test_contract_stop_leaves_no_process_in_the_group(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "text")
        session.start()
        session.send_user("finish quickly")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=START_WAIT))
        pgid = session.pgid
        self.assertIsNotNone(pgid)
        session.stop(grace=2)
        self.assertFalse(session.alive())
        self.assertTrue(group_gone(pgid), "no process of the session is left in its group")

    def test_contract_resume_keeps_the_stored_id(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "text", resume_id="resume-id-42")
        session.start()
        session.send_user("continue")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, SessionStarted), timeout=START_WAIT))
        started = next(e for e in sink.events if isinstance(e, SessionStarted))
        self.assertEqual(started.session_id, "resume-id-42")

    def test_contract_refused_resume_is_reported_as_resume_failed(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "resume_refused", resume_id="u-9")
        session.start()
        self.assertTrue(sink.wait_for(
            lambda e: isinstance(e, TurnResult) and e.resume_failed, timeout=START_WAIT),
            "the CLI's refusal of the resume is reported on the TurnResult")
