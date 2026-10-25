"""The active timer is frozen at the TurnResult (ResultFreezer, the first listener of the turn's bus): a listener that runs
after the result and spends time on it cannot make a finished turn hit max_turn_seconds. The turn-level cases are in
tests/test_turn_runner.py; this file holds the bus-order case on its own."""

import unittest
from unittest import mock

from tests.fakes.clock import FakeClock
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core.events_bus import TurnEventBus
from tg_bridge.core.turn_listeners import ResultFreezer
from tg_bridge.core.turn_state import TurnOutcome, TurnState

THREAD = 7
OK = TurnResult(TurnStatus.OK, answer="ok:allow", duration_ms=5, num_turns=1, subtype="success")


class ResultFreezeTests(unittest.TestCase):
    """The active timer stops at the TurnResult itself: a listener that runs after the result and spends time on it (a
    slow feed) cannot make a finished turn hit its cap."""

    def test_a_slow_listener_after_the_result_cannot_cap_a_finished_turn(self) -> None:
        clock = FakeClock()
        state = TurnState(THREAD, clock, max_active_seconds=3, hard_cap_seconds=30)
        state.start()
        seen: list[float | None] = []

        class SlowFeed:
            def on_event(self, event) -> None:
                clock.advance(4)
                seen.append(state.check_caps())

        bus = TurnEventBus([ResultFreezer(state.mark_result_received), SlowFeed()], mock.Mock())
        bus.publish(OK)
        self.assertEqual(seen, [None])
        self.assertEqual(state.finish(OK), TurnOutcome.OK)
