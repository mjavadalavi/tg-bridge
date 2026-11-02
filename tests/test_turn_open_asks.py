"""A turn that starts while one of its session's approval cards is still open (K11). The card was posted while the session
was idle, when a background agent asked; the next turn of the kept session must start in the waiting phase, its active
timer must not advance until the card is answered, and an answer that lands while the turn is being synced must not
leave the turn waiting for nothing.

The permission service, the card, the wait router and the turn state are the real classes on real repositories in a
TempHome; the clock is a FakeClock and the workers run inline. Nothing sleeps and no process starts.
"""

import os
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, RecordingSink, fake_spec
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.claude.backend import ClaudeRules
from tg_bridge.agents.events import ApprovalRequest, Decision
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.core.turn_kit import sync_open_asks
from tg_bridge.core.turn_listeners import WaitRouter
from tg_bridge.core.turn_state import TurnPhase, TurnState
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.rules import RuleRepo
from tg_bridge.telegram.messenger import Messenger

THREAD = 9301
CHAT = -100
CWD = "/work"
NONCE = "0000abcd"
ALL = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})
MAX_ACTIVE = 30.0


class NoTimer:
    """A nudge timer that never fires on its own."""

    def __init__(self, seconds, fire) -> None:
        self.seconds = seconds

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass


class InlineWorkers:
    """Runs each submitted job at once, on the calling thread."""

    def submit(self, fn, *args) -> None:
        fn(*args)


class CountStub:
    """A permission service whose open-ask count is read from a script: each call takes the next value, the last one
    repeats. It stands in for PermissionService where a test needs an answer to land between two reads."""

    def __init__(self, *counts: int) -> None:
        self._counts = list(counts)

    def open_asks(self, binding) -> int:
        if len(self._counts) > 1:
            return self._counts.pop(0)
        return self._counts[0]


def bash_request() -> ApprovalRequest:
    return ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "git status"},
                           summary="Bash", choices=ALL)


class OpenAskBeforeTurnTests(unittest.TestCase):
    """The real permission service: a card is posted while no turn is registered, then a turn starts."""

    def setUp(self) -> None:
        self._home = TempHome()
        home = self._home.__enter__().home
        self.clock = FakeClock()
        self.tg = FakeTelegram(clock=self.clock)
        self.logger = Logger(os.path.join(home, "bridge.log"))
        self.db = Database(Paths.from_home(home))
        self.router = WaitRouter()
        messenger = Messenger(self.tg, self.logger, self.clock)
        cards = ApprovalCards(messenger, self.logger, self.clock, timer_factory=NoTimer)
        self.service = PermissionService(
            asks=AskRepo(self.db, self.clock), rules=RuleRepo(self.db, self.clock), cards=cards, messenger=messenger,
            workers=InlineWorkers(), waiting=self.router, logger=self.logger, nonce_factory=lambda: NONCE)
        backend = FakeBackend(capabilities=FAKE_CAPABILITIES)
        session = backend.open_session(fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        self.binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=CWD, rules=ClaudeRules(), mode="default")
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, session)

    def tearDown(self) -> None:
        self.db.close()
        self._home.__exit__(None, None, None)

    def new_turn(self) -> TurnState:
        turn = TurnState(THREAD, self.clock, max_active_seconds=MAX_ACTIVE, hard_cap_seconds=10_000)
        self.router.register(THREAD, turn)
        return turn

    def test_a_turn_started_while_a_card_is_open_waits_and_its_timer_stands_still(self) -> None:
        self.listener.on_event(bash_request())  # idle: no turn is registered, the card is posted and stays open
        self.assertEqual(self.service.open_asks(self.binding), 1)
        turn = self.new_turn()
        turn.start()
        sync_open_asks(self.router, self.service, self.binding)
        self.assertIs(turn.phase, TurnPhase.WAITING_APPROVAL)
        self.clock.advance(100)
        self.assertEqual(turn.active_seconds(), 0.0)
        self.assertIsNone(turn.check_caps())  # 100 s of waiting do not reach the 30 s active cap
        self.assertTrue(self.service.answer_ask(NONCE, True))
        self.assertIs(turn.phase, TurnPhase.RUNNING)
        self.clock.advance(5)
        self.assertEqual(turn.active_seconds(), 5.0)

    def test_a_turn_with_no_open_card_is_not_touched(self) -> None:
        turn = self.new_turn()
        turn.start()
        sync_open_asks(self.router, self.service, self.binding)
        self.assertIs(turn.phase, TurnPhase.STARTING)
        self.clock.advance(50)
        self.assertEqual(turn.active_seconds(), 50.0)

    def test_a_card_answered_during_the_sync_leaves_the_turn_running(self) -> None:
        """The count reads 1 and then 0: the answer's waiting=False reached the turn before the notice, so the turn must
        not be left waiting. Without the second read, the turn would wait for good."""
        turn = self.new_turn()
        turn.start()
        sync_open_asks(self.router, CountStub(1, 0), self.binding)
        self.assertIs(turn.phase, TurnPhase.RUNNING)
        self.clock.advance(7)
        self.assertEqual(turn.active_seconds(), 7.0)


class OpenAskBeforeStartTests(unittest.TestCase):
    """The turn state alone: a card that arrives before start() must not start the active timer."""

    def test_a_waiting_change_before_start_keeps_the_timer_paused_until_the_answer(self) -> None:
        clock = FakeClock()
        router = WaitRouter()
        turn = TurnState(THREAD, clock, max_active_seconds=MAX_ACTIVE, hard_cap_seconds=10_000)
        router.register(THREAD, turn)
        router.waiting_changed(THREAD, True)  # a background card, while the turn is registered but not started
        self.assertIs(turn.phase, TurnPhase.WAITING_APPROVAL)
        turn.start()
        clock.advance(50)
        self.assertEqual(turn.active_seconds(), 0.0)
        router.waiting_changed(THREAD, False)
        self.assertIs(turn.phase, TurnPhase.RUNNING)
        clock.advance(3)
        self.assertEqual(turn.active_seconds(), 3.0)


if __name__ == "__main__":
    unittest.main()
