"""Port of fakecli/test_slot_release.py (the two-parked-turns scenario): two turns parked on permission cards give
their slot permits back, a third turn runs at once, and the parked turns answer after their taps.

The busy-bridge notice and the one-permit balance are in test_dispatcher and test_slots; this module covers the
parked-turn scenario with the TurnRunner, FakeBackend and a real two-permit SlotPool of the turn harness.
"""

import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, TurnHarness, join_turn, run_async, wait_until
from tg_bridge.agents.events import ApprovalRequest, Decision, TurnResult, TurnStatus

TOPIC_A, TOPIC_B, TOPIC_C = 9801, 9802, 9803
OK = TurnResult(TurnStatus.OK, answer="ok:allow", duration_ms=5, num_turns=1, subtype="success")


def ask(approval_id: str) -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, kind="command", tool="Bash", input={"command": "echo hi"},
                           summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}))


class ParkedTurnsPort(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.backend = FakeBackend([[ask("rA"), OK], [ask("rB"), OK], [OK]], capabilities=FAKE_CAPABILITIES)
        self.runner = self.h.runner(self.h.registry_for(self.backend))
        for thread in (TOPIC_A, TOPIC_B, TOPIC_C):
            self.h.sessions.ensure(thread, f"T{thread}", "fake-cmd", self.home.home, "default")
            self.h.sessions.set_mode(thread, "default")

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def _start(self, thread: int):
        slot = self.h.slots.new_slot(CHAT, thread)
        self.assertTrue(slot.take())
        thread_obj, box = run_async(self.runner, thread_id=thread, chat_id=CHAT, topic=f"T{thread}", text="go",
                                    sender="tester", slot=slot)
        return slot, thread_obj, box

    def test_two_parked_turns_give_their_permits_back_and_answer_after_their_taps(self) -> None:
        """A and B answered ok:allow; A and B ended after the taps; A and B still parked after C finished;
        A spawned before B starts; all permits back after the turns (2); parked: both A and B are waiting on a card."""
        slot_a, thread_a, box_a = self._start(TOPIC_A)
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(TOPIC_A)) == 1), "A is not parked on its card")
        self.assertEqual(len(self.backend.specs), 1, "A spawned before B starts")
        slot_b, thread_b, box_b = self._start(TOPIC_B)
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(TOPIC_B)) == 1), "B is not parked on its card")
        self.assertTrue(wait_until(lambda: self.h.slots.free() == 2), "parked turns kept their permits")
        self.assertEqual(len(self.h.asks.pending(TOPIC_A)), 1)
        self.assertEqual(len(self.h.asks.pending(TOPIC_B)), 1)

        slot_c, thread_c, box_c = self._start(TOPIC_C)
        join_turn(self.h.clock, thread_c)
        self.assertEqual((box_c[0].answer, box_c[0].err), ("ok:allow", None))
        self.assertEqual(len(self.h.asks.pending(TOPIC_A)), 1, "A still parked after C finished")
        self.assertEqual(len(self.h.asks.pending(TOPIC_B)), 1, "B still parked after C finished")

        for thread in (TOPIC_A, TOPIC_B):
            nonce = self.h.asks.pending(thread)[0].nonce
            self.assertTrue(self.h.permissions.answer_ask(nonce, allow=True))
        join_turn(self.h.clock, thread_a)
        join_turn(self.h.clock, thread_b)
        self.assertEqual((box_a[0].answer, box_a[0].err), ("ok:allow", None))
        self.assertEqual((box_b[0].answer, box_b[0].err), ("ok:allow", None))
        self.assertEqual(self.h.asks.pending(TOPIC_A), [])
        self.assertEqual(self.h.asks.pending(TOPIC_B), [])

        for slot in (slot_a, slot_b, slot_c):
            slot.give_back()
        self.assertEqual(self.h.slots.free(), 2, "all permits back after the turns (2)")


if __name__ == "__main__":
    unittest.main()
