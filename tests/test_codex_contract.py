"""CodexBackend against the fake app-server (tests/fakes/fake_codex.py): the shared BackendContract plus Codex-only checks.

The contract checks (start, text turn, approval round trip, withdraw, interrupt, stop, resume, refused resume) run here
with real child processes. Two adaptations, both documented where they are made: the withdraw check is overridden,
because Codex withdraws a pending approval when the turn is interrupted (the fake sends serverRequest/resolved), not on
its own; and the scenario names map to the FAKE_CODEX_MODE values below. The Codex-only checks cover the interrupt
wire format, the reader-thread refusal, the interrupt fallback, and that interrupt, stubborn and grandchild processes
leave no process behind. The live file is not loaded here.
"""

import dataclasses
import json
import os
import sys
import time
import unittest
from unittest import mock

import tests.fakes as fakes_package
from tests.contract import BackendContract, group_gone
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import Capabilities
from tg_bridge.agents.codex import session as codex_session
from tg_bridge.agents.codex.backend import CodexBackend
from tg_bridge.agents.events import (ApprovalRequest, ApprovalWithdrawn, SessionStarted, TextEvent, TurnResult,
                                     TurnStatus)
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_codex.py")
WAIT = 15.0
INTERRUPT_WAIT = 10.0
# scenario -> FAKE_CODEX_MODE. 'withdraw' interrupts a pending approval (see test_contract_withdrawn_approval_is_reported).
SCENARIO_MODE = {
    "text": "normal",
    "approval": "command_approval",
    "withdraw": "interrupt",
    "hang": "interrupt",
    "resume_refused": "resume_missing",
}


class CodexBackendContractTests(BackendContract, unittest.TestCase):
    """The shared contract against fake_codex, plus the Codex-only behaviour."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.contract_root = self.home.home
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.wire_log = os.path.join(self.home.home, "wire.jsonl")  # every frame the fake received, one per line

    def contract_backend(self) -> CodexBackend:
        return CodexBackend(logger=self.logger, clock=self.clock)

    def contract_launcher(self) -> LauncherSpec:
        return LauncherSpec(name="fake-codex", backend="codex", command="codex",
                            argv=(sys.executable, "-I", FAKE), config_dir=self.home.home)

    def contract_env(self, scenario: str) -> dict[str, str]:
        return {"FAKE_CODEX_MODE": SCENARIO_MODE[scenario], "FAKE_CODEX_ANSWER": "done",
                "FAKE_CODEX_LOG": self.wire_log}

    def contract_open_with_env(self, backend: CodexBackend, scenario: str, extra_env: dict[str, str]):
        """contract_open with more environment values (for example FAKE_CODEX_PIDFILE)."""
        spec = self.contract_spec(backend, scenario)
        spec = dataclasses.replace(spec, env={**spec.env, **extra_env})
        sink = RecordingSink()
        session = backend.open_session(spec, sink)
        self.addCleanup(session.stop, 2)
        return session, sink

    # ---- contract adaptations -------------------------------------------------------------------------

    def test_contract_withdrawn_approval_is_reported(self) -> None:
        """Codex withdraws a pending approval when its turn is interrupted (serverRequest/resolved), so the
        withdraw check interrupts the turn while the approval is open, then expects ApprovalWithdrawn and a turn
        that is not OK. Answering the approval is not a withdrawal in this fake."""
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "withdraw")
        session.start()
        session.send_user("run it")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        req = next(e for e in sink.events if isinstance(e, ApprovalRequest))
        self.assertTrue(session.interrupt())
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalWithdrawn), timeout=INTERRUPT_WAIT))
        withdrawn = next(e for e in sink.events if isinstance(e, ApprovalWithdrawn))
        self.assertEqual(withdrawn.approval_id, req.approval_id)
        result = next(e for e in sink.events if isinstance(e, TurnResult))
        self.assertIsNot(result.status, TurnStatus.OK)

    # ---- capabilities ---------------------------------------------------------------------------------

    def test_codex_capabilities_match_the_design(self) -> None:
        caps = self.contract_backend().capabilities
        self.assertEqual(caps, Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False,
                                            images=True, preassigned_id=False))

    def test_codex_modes_default_to_bypass_labels_and_notes(self) -> None:
        modes = self.contract_backend().modes()
        self.assertEqual([m.key for m in modes], ["bypassPermissions", "acceptEdits", "default", "plan"])
        self.assertTrue(all(m.note == "applies from your next message" for m in modes))

    def test_codex_models_are_empty_until_the_picker_passes_launcher_models(self) -> None:
        self.assertEqual(self.contract_backend().models(mock.Mock()), ())

    def test_codex_backend_has_no_title_or_history_support(self) -> None:
        backend = self.contract_backend()
        self.assertIsNone(backend.one_shot("title please", timeout=1.0))
        self.assertFalse(backend.carry_history(self.contract_launcher(), self.contract_launcher(), "/x", "id", None))

    # ---- interrupt ------------------------------------------------------------------------------------

    def test_interrupt_sends_turn_interrupt_with_thread_and_turn_ids(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("long")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        self.assertTrue(session.interrupt())
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=INTERRUPT_WAIT))
        log = self.wire_frames()
        sent = [f for f in log if f.get("method") == "turn/interrupt"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["params"], {"threadId": "fake-thread-1", "turnId": "fake-turn-1"})
        self.assertIs(next(e for e in sink.events if isinstance(e, TurnResult)).status, TurnStatus.INTERRUPTED)
        self.assertEqual(sum(isinstance(e, TurnResult) for e in sink.events), 1, "exactly one TurnResult")

    def test_interrupt_before_a_turn_is_refused_without_a_request(self) -> None:
        backend = self.contract_backend()
        session, _, _ = self.contract_open(backend, "text")
        session.start()
        self.assertFalse(session.interrupt(), "no turn is running yet")
        self.assertEqual([f for f in self.wire_frames() if f.get("method") == "turn/interrupt"], [])

    def test_interrupt_from_the_reader_thread_is_refused_without_waiting(self) -> None:
        backend = self.contract_backend()
        spec = self.contract_spec(backend, "text")
        seen: list[bool] = []
        holder: dict[str, object] = {}

        def sink(event: object) -> None:
            if isinstance(event, TextEvent):  # emitted on the reader thread, during the turn
                seen.append(holder["session"].interrupt())

        session = backend.open_session(spec, sink)
        holder["session"] = session
        self.addCleanup(session.stop, 2)
        session.start()
        session.send_user("hi")
        deadline = time.monotonic() + WAIT
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(seen, "the sink saw at least one text event on the reader thread")
        self.assertEqual(set(seen), {False}, "every reader-thread call returns False at once")
        self.assertEqual([f for f in self.wire_frames() if f.get("method") == "turn/interrupt"], [])

    def test_interrupt_fallback_stops_the_process_when_the_turn_never_ends(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("long")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        pgid = session.pgid
        # the request is accepted but the fake never ends the turn: make the fallback window short and drop the request
        with mock.patch.object(codex_session, "INTERRUPT_TIMEOUT", 0.3), \
                mock.patch.object(session._peer, "request", return_value={}):
            self.assertTrue(session.interrupt())
        self.assertFalse(session.alive(), "the fallback stopped the process")
        self.assertTrue(group_gone(pgid))
        results = [e for e in sink.events if isinstance(e, TurnResult)]
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, TurnStatus.INTERRUPTED)
        self.assertIn("stopped", results[0].error)

    def test_interrupt_request_failure_falls_back_to_stop(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("long")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        with mock.patch.object(session._peer, "request", side_effect=TimeoutError("turn/interrupt timed out")):
            self.assertTrue(session.interrupt())
        self.assertFalse(session.alive())
        results = [e for e in sink.events if isinstance(e, TurnResult)]
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, TurnStatus.INTERRUPTED)

    def test_interrupt_after_the_turn_ended_returns_false(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "text")
        session.start()
        session.send_user("hi")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=WAIT))
        self.assertFalse(session.interrupt())

    # ---- process cleanup ------------------------------------------------------------------------------

    def test_interrupted_session_leaves_no_process_after_stop(self) -> None:
        backend = self.contract_backend()
        session, sink, _ = self.contract_open(backend, "hang")
        session.start()
        session.send_user("long")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest), timeout=WAIT))
        pgid = session.pgid
        self.assertTrue(session.interrupt())
        session.stop(grace=2)
        self.assertFalse(session.alive())
        self.assertTrue(group_gone(pgid), "no process of the interrupted session is left in its group")

    def test_stubborn_process_is_killed_by_the_ladder(self) -> None:
        backend = self.contract_backend()
        spec = dataclasses.replace(self.contract_spec(backend, "text"),
                                   env={**os.environ, "FAKE_CODEX_MODE": "stubborn", "FAKE_CODEX_ANSWER": "done"})
        sink = RecordingSink()
        session = backend.open_session(spec, sink)
        session.start()
        session.send_user("finish")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=WAIT))
        pgid = session.pgid
        started = time.monotonic()
        session.stop(grace=0.5)
        self.assertFalse(session.alive(), "the process ignored the EOF and must be gone after stop()")
        self.assertLess(time.monotonic() - started, 10, "the ladder ends well inside its budget")
        self.assertTrue(group_gone(pgid))

    def test_grandchild_is_swept_with_its_process_group(self) -> None:
        backend = self.contract_backend()
        session, sink = self.contract_open_with_env(backend, "text", {"FAKE_CODEX_MODE": "grandchild"})
        session.start()
        session.send_user("start a child")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult), timeout=WAIT))
        pgid = session.pgid
        session.stop(grace=2)
        self.assertTrue(group_gone(pgid), "the sleeping grandchild in the group is gone after stop()")

    # ---- helpers --------------------------------------------------------------------------------------

    def wire_frames(self) -> list[dict]:
        """Every frame the bridge wrote to the fake, in order (the fake logs each one as it reads it)."""
        try:
            with open(self.wire_log, encoding="utf-8") as handle:
                return [json.loads(line)["in"] for line in handle if line.strip()]
        except FileNotFoundError:
            return []


if __name__ == "__main__":
    unittest.main()
