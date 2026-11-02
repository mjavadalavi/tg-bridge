"""TurnRunner with linger on: an approval the background work asks for in the window between the user turn's result and
the linger offer. The router is armed before the turn's gate attaches, so the approval reaches its card (the session
sink) even though no hold is installed yet. When the offer does not keep the session, the turn ends as before and the
card is voided with the turn. With linger off the router is not armed, so the same approval is dropped as before.

FakeBackend (with linger capability), TurnHarness with the real permission service, cards and asks repository, and the
real LingerService; the offer is replaced by a stub that delivers the approval at the point the window is.
"""

import dataclasses
import unittest
from unittest import mock

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.agents.events import ApprovalRequest, Decision, TextEvent, TurnResult, TurnStatus
from tg_bridge.store.asks import VOID

OK = TurnResult(TurnStatus.OK, answer="first", duration_ms=5, num_turns=1, subtype="success")
APPROVAL = ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "git status"},
                           summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}))


class ApprovalWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)  # runs before the TempHome is removed (cleanups run last-in first-out)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_mode(THREAD, "default")  # bypass answers an approval itself: the card needs a real mode

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def start(self, **config_overrides) -> None:
        self.h.set_config(**{"linger_seconds": 60, "linger_max_sessions": 3, **config_overrides})
        capabilities = dataclasses.replace(FAKE_CAPABILITIES, linger=True)
        self.backend = FakeBackend([[TextEvent(text="a"), OK]], capabilities=capabilities)
        self.registry = self.h.registry_for(self.backend)
        self.service = self.h.linger(self.registry)
        self.runner = self.h.runner(self.registry, linger=self.service)

    def run_turn(self, offer_stub):
        with mock.patch.object(self.service, "offer", side_effect=offer_stub):
            return self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")

    def test_an_approval_after_the_result_and_before_the_offer_gets_its_card_and_is_voided_when_not_kept(self) -> None:
        self.start()
        seen: dict[str, list[str]] = {}

        def offer_after_an_approval(o):
            o.router(APPROVAL)  # the window: the turn's result is in, the offer has not decided yet
            seen["open"] = [ask.nonce for ask in self.h.asks.pending(THREAD)]
            return False  # not kept: the turn ends as before

        reply = self.run_turn(offer_after_an_approval)
        self.assertEqual((reply.answer, reply.err), ("first", None))
        self.assertEqual(len(seen["open"]), 1, "the approval reached a card while the offer was still pending")
        self.assertEqual(self.h.asks.pending(THREAD), [], "the turn ended: its open card is no longer pending")
        self.assertEqual(self.h.asks.by_nonce(seen["open"][0]).answer, VOID, "the card was closed with the turn")
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertFalse(self.service.holding(THREAD))

    def test_with_linger_off_the_router_is_not_armed_and_the_same_approval_is_dropped_as_before(self) -> None:
        self.start(linger_seconds=0)
        seen: dict[str, list[str]] = {}

        def offer_after_an_approval(o):
            o.router(APPROVAL)
            seen["open"] = [ask.nonce for ask in self.h.asks.pending(THREAD)]
            return False

        self.run_turn(offer_after_an_approval)
        self.assertEqual(seen["open"], [], "no card: the router was never armed")


if __name__ == "__main__":
    unittest.main()
