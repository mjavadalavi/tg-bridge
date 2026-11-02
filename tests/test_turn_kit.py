"""TurnKit helpers (core/turn_kit.py) that a kept session reuses: build_kit with a given binding, and finish_attempt, which
reads the stop flags and the exit code of one launch. Plain doubles and a TurnHarness on real repositories; no process.
"""

import unittest

from tests.fakes.backend import FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.agents.events import ProcessExited
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import TurnBinding
from tg_bridge.core.turn_kit import TurnPlan, build_kit, finish_attempt
from tg_bridge.core.turn_listeners import SessionRecorder


class _StubSession:
    """The one method finish_attempt reads from a session: its stderr tail."""

    def stderr_tail(self, n: int = 20) -> str:
        return ""


class KitCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.backend = FakeBackend()
        self.plan = TurnPlan(
            thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", images=(),
            launcher=self.h.config.resolve_launcher("", "", "fake-cmd"), backend=self.backend, cwd=self.home.home,
            mode="bypassPermissions", model=None, allow_rules=(), deny_tools=(), system_rules="",
            stored_uuid="")
        self.recorder = SessionRecorder(sessions=self.h.sessions, history=self.h.history, kv=self.h.kv,
                                        thread_id=THREAD, text="hello", cwd=self.home.home, launcher="fake-cmd",
                                        backend="fake", known_uuid="")

    def kit(self, binding: TurnBinding | None = None):
        return build_kit(self.plan, self.recorder, None, clock=self.h.clock, config=self.h.config,
                         feeds=self.h.feeds, permissions=self.h.permissions,
                         handoff=HandoffService(self.h.kv, self.h.registry_for(self.backend), self.h.logger),
                         pending_digest=lambda: "", open_view=lambda: None, binding=binding)


class BuildKitTests(KitCase):
    def test_build_kit_reuses_a_given_binding(self) -> None:
        # a kept session passes the binding of its first turn: the kit uses that object and the permission service
        # registers it again, instead of a new binding for every turn
        binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.home.home, rules=self.backend.rules,
                              mode="bypassPermissions")
        kit = self.kit(binding)
        self.assertIs(kit.binding, binding)
        self.assertIs(self.h.permissions._turns[THREAD], binding)

    def test_build_kit_makes_a_new_binding_when_none_is_given(self) -> None:
        kit = self.kit()
        self.assertEqual((kit.binding.thread_id, kit.binding.cwd, kit.binding.mode),
                         (THREAD, self.home.home, "bypassPermissions"))
        self.assertIs(self.h.permissions._turns[THREAD], kit.binding)


class FinishAttemptTests(KitCase):
    def test_finish_attempt_reads_abort_flags_and_exit_code(self) -> None:
        # aborted: a ⏹ tap on the running turn is read once, marked on the turn and cleared for the next turn
        kit = self.kit()
        live = LiveRegistry()
        session = _StubSession()
        live.begin_turn(THREAD, session, kit.feed)
        self.assertIsNotNone(live.request_abort(THREAD, stop_tap=True))
        attempt = finish_attempt(kit, session, live, refused=False, launch_err=None,
                                  readable_tail=True, logger=self.h.logger, stop_code=None)
        self.assertEqual((attempt.aborted, attempt.stop_tap), (True, True))
        self.assertTrue(kit.turn.stop_requested)  # the same state request_stop() sets on the turn
        self.assertFalse(live.is_aborted(THREAD))
        # exit code: the stop reaped the process; with no exit seen by the reader, its code is the one kept. An exit
        # the reader did see (after the stdout EOF) wins over it.
        reaped = finish_attempt(self.kit(), session, LiveRegistry(), refused=False, launch_err=None,
                                  readable_tail=True, logger=self.h.logger, stop_code=137)
        self.assertEqual(reaped.exit_code, 137)
        seen = self.kit()
        seen.collector.on_event(ProcessExited(returncode=0))
        self.assertEqual(finish_attempt(seen, session, LiveRegistry(), refused=False, launch_err=None,
                                        readable_tail=True, logger=self.h.logger, stop_code=137).exit_code, 0)


if __name__ == "__main__":
    unittest.main()
