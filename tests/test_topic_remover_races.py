"""TopicRemover races (card F2): a removal marks the topic first, so a message delivered during it is dropped, a turn
taken but not yet registered runs no turn, a kept session's release is asked again each round, a refusal clears the mark
at once, a removal during shutdown is refused, and an open terminal session does not refuse the removal.

The App is built over a TempHome with a FakeTelegram; the race test runs a real Dispatcher runner thread that blocks in
the slot wait (a turn taken but not registered). The FakeClock's sleep runs a hook, so the removal's wait meets the
race at a known point. Nothing touches the real state directory, tmux or the Telegram API.
"""

import os
import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.app import App
from tg_bridge.config import Config
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.removals import GRACE_SECONDS, Removals
from tg_bridge.core.terminal_claim import claim_path, take_claim
from tg_bridge.core.topic_queues import Admission
from tg_bridge.core.topic_remover import SHUTDOWN_TEXT, TERMINAL_OPEN_TEXT, TopicRemover
from tg_bridge.core.turn_kit import TurnReply
from tg_bridge.core.turn_runner import active_key
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.paths import Paths
from tg_bridge.store.forget import TopicForget
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.inbox import STATE_DONE, InboxRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

CHAT = -1009902
THREAD = 7
GROUP_KEY = "group_id"
ALLOWED_UID = 42
STOP_WAIT = 15.0


REAL_YIELD_SECONDS = 0.02  # each wait round of the removal yields for real, so the runner thread gets to finish


class HookClock(FakeClock):
    """A FakeClock whose first sleep runs a hook (the removal's wait reaches the race point there). Every sleep yields
    for a moment in real time, so a runner thread that the removal waits for can finish."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        hook, self.hook = self.hook, None
        if hook is not None:
            hook()
        time.sleep(REAL_YIELD_SECONDS)


class FakeSlot:
    """A slot whose permit wait blocks until the test opens the gate: the turn is taken but not registered yet."""

    def __init__(self, gate: threading.Event, entered: threading.Event, chat_id: int, thread_id: int) -> None:
        self._gate = gate
        self._entered = entered
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.gives = 0

    def take(self) -> bool:
        self._entered.set()
        self._gate.wait(5)
        return True

    def give_back(self) -> None:
        self.gives += 1


class FakeSlots:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.slot: FakeSlot | None = None

    def new_slot(self, chat_id: int, thread_id: int) -> FakeSlot:
        self.slot = FakeSlot(self.gate, self.entered, chat_id, thread_id)
        return self.slot


class FakeTurns:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def run(self, **kwargs) -> TurnReply:
        self.runs.append(kwargs)
        return TurnReply(None, "ran", TurnOutcome.ERROR)


class Quiet:
    """A stand-in for the messenger, the answer sender and the title service: only a turn that runs would use them."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, line: str) -> None:
        self.lines.append(line)


class FakeLinger:
    """A kept session: holding until its release is accepted, and a release is accepted only on the release_on-th call
    (a session in a turn refuses it before that)."""

    def __init__(self, release_on: int) -> None:
        self.release_on = release_on
        self.turn = True
        self.calls = 0

    def holding(self, thread_id: int) -> bool:
        return self.turn

    def request_release(self, thread_id: int) -> bool:
        self.calls += 1
        if self.calls >= self.release_on:
            self.turn = False
            return True
        return False


class FakeDispatcher:
    def busy(self, thread_id: int) -> bool:
        return False

    def clear_queue(self, thread_id: int) -> int:
        return 0


class FakeForget:
    def __init__(self) -> None:
        self.forgotten: list[int] = []

    def forget_topic(self, thread_id: int) -> None:
        self.forgotten.append(thread_id)


class FakeKv:
    def get(self, key: str) -> str | None:
        return None


def _remover(*, live, linger, dispatcher, forget, logger, clock, removals, view_dir="") -> TopicRemover:
    return TopicRemover(live=live, linger=linger, dispatcher=dispatcher, forget=forget, client=None, kv=FakeKv(),
                        group_key=GROUP_KEY, view_dir=view_dir, logger=logger, clock=clock, removals=removals)


class UnitRemoverTests(unittest.TestCase):
    """The remover over fakes: the release retry, the refusal, the shutdown and the mark's life."""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.removals = Removals(self.clock)
        self.logger = FakeLogger()
        self.forget = FakeForget()
        self.live = LiveRegistry()

    def _remove(self, linger: FakeLinger) -> str:
        remover = _remover(live=self.live, linger=linger, dispatcher=FakeDispatcher(), forget=self.forget,
                           logger=self.logger, clock=self.clock, removals=self.removals)
        return remover.remove(THREAD, delete_in_telegram=False)

    def test_release_is_asked_again_each_round_until_the_kept_session_leaves_its_turn(self) -> None:
        linger = FakeLinger(release_on=3)
        text = self._remove(linger)
        self.assertEqual(text, f"topic {THREAD} removed (telegram: kept)")
        self.assertEqual(linger.calls, 3, "the first two refused releases were retried")
        self.assertEqual(self.forget.forgotten, [THREAD])

    def test_a_refusal_forgets_nothing_and_clears_the_mark_at_once(self) -> None:
        text = self._remove(FakeLinger(release_on=10**9))
        self.assertTrue(text.startswith(f"topic {THREAD} not removed: its session did not stop"), text)
        self.assertEqual(self.forget.forgotten, [])
        self.assertFalse(self.removals.active(THREAD), "a refused removal clears the mark at once")

    def test_a_removal_during_shutdown_is_refused_and_marks_nothing(self) -> None:
        self.live.begin_shutdown()
        text = self._remove(FakeLinger(release_on=1))
        self.assertEqual(text, f"topic {THREAD} not removed: {SHUTDOWN_TEXT}")
        self.assertEqual(self.forget.forgotten, [])
        self.assertFalse(self.removals.active(THREAD))

    def test_the_mark_is_kept_for_the_grace_after_a_removal_and_then_cleared(self) -> None:
        self._remove(FakeLinger(release_on=1))
        self.assertTrue(self.removals.active(THREAD), "a late update still finds the topic marked")
        self.clock.advance(GRACE_SECONDS - 1)
        self.assertTrue(self.removals.active(THREAD))
        self.clock.advance(2)
        self.assertFalse(self.removals.active(THREAD))


class RaceTests(unittest.TestCase):
    """The message-then-Delete race and the message that arrives during a removal, with a real runner thread."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = HookClock()
        telegram = FakeTelegram(self.clock)
        config = Config({"allow_user_ids": [ALLOWED_UID]}, home=self.home.home, env={})
        self.app = App.build(Paths.from_home(self.home.home), config, telegram=telegram, clock=self.clock, env={})
        self.addCleanup(self.app.close)
        self.app.kv.set(GROUP_KEY, str(CHAT))
        db = self.app.db
        self.sessions = SessionRepo(db, self.clock)
        self.history = HistoryRepo(db, self.clock)
        self.inbox = InboxRepo(db, self.clock)
        self.sessions.ensure(THREAD, f"topic{THREAD}", "claude-cmd", self.home.home)
        self.history.note_session(THREAD, "uuid-7", "hello", self.home.home, "claude-cmd", "claude")
        for update_id in (2001, 2002, 2003, 2004):
            self.inbox.ingest(update_id, {"text": "m"}, thread_id=THREAD, chat_id=CHAT, sender="u")
        self.slots = FakeSlots()
        self.turns = FakeTurns()
        self.removals = Removals(self.clock)
        self.dispatcher = Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=Quiet(), answers=Quiet(), titles=Quiet(),
            client=telegram, live=self.app.live, config=config, logger=FakeLogger(), linger=None, clock=self.clock,
            removals=self.removals)
        self.remover = _remover(
            live=self.app.live, linger=self.app.linger, dispatcher=self.dispatcher,
            forget=TopicForget(db=db, sessions=self.sessions, history=self.history, rules=RuleRepo(db, self.clock),
                               asks=AskRepo(db, self.clock), inbox=self.inbox, kv=self.app.kv),
            logger=FakeLogger(), clock=self.clock, removals=self.removals)

    def _count(self, sql: str) -> int:
        return int(self.app.db.query(sql, (THREAD,))[0][0])

    def _state(self, update_id: int) -> str:
        return self.inbox.get(update_id).state

    def test_message_then_delete_runs_no_turn_and_brings_no_rows_back(self) -> None:
        first = self.dispatcher.deliver(THREAD, f"topic{THREAD}", CHAT, "u", 2001, "first")
        self.assertIs(first.admission, Admission.STARTED)
        self.assertTrue(self.slots.entered.wait(5), "the runner took the first message and waits for its slot")
        queued = self.dispatcher.deliver(THREAD, f"topic{THREAD}", CHAT, "u", 2002, "second")
        self.assertIs(queued.admission, Admission.QUEUED)
        during: list[Admission] = []
        states: dict[int, str] = {}

        def deliver_during_removal() -> None:  # runs inside the removal's first wait round
            during.append(self.dispatcher.deliver(THREAD, f"topic{THREAD}", CHAT, "u", 2003, "third").admission)
            states.update({update_id: self._state(update_id) for update_id in (2002, 2003)})
            self.slots.gate.set()  # the slot is granted now: the runner must see the removal and stop

        self.clock.hook = deliver_during_removal
        text = self.remover.remove(THREAD, delete_in_telegram=False)

        self.assertEqual(text, f"topic {THREAD} removed (telegram: kept)")
        self.assertEqual(during, [Admission.DROPPED], "a message delivered during the removal is dropped")
        self.assertEqual(self.turns.runs, [], "no turn runs for the removed topic")
        self.assertEqual(self.slots.slot.gives, 1, "the permit taken for the dropped turn is given back")
        self.assertEqual(states, {2002: STATE_DONE, 2003: STATE_DONE}, "the cleared and the dropped rows are done")
        self.assertEqual(self._count("SELECT COUNT(*) FROM inbox_queue WHERE thread_id=?"), 0)
        self.assertEqual(self._count("SELECT COUNT(*) FROM sessions WHERE thread_id=?"), 0)
        self.assertEqual(self._count("SELECT COUNT(*) FROM session_history WHERE thread_id=?"), 0)
        self.assertNotIn(active_key(THREAD), {row["k"] for row in self.app.db.query("SELECT k FROM kv")})
        self.assertFalse(self.dispatcher.busy(THREAD))

    def test_a_late_update_in_the_grace_period_is_dropped_and_marked_done(self) -> None:
        self.remover.remove(THREAD, delete_in_telegram=False)
        self.inbox.ingest(2004, {"text": "late"}, thread_id=THREAD, chat_id=CHAT, sender="u")  # arrives after forget
        late = self.dispatcher.deliver(THREAD, f"topic{THREAD}", CHAT, "u", 2004, "late")
        self.assertIs(late.admission, Admission.DROPPED)
        self.assertEqual(self._state(2004), STATE_DONE)
        self.assertEqual(self.turns.runs, [])
        self.assertEqual(self._count("SELECT COUNT(*) FROM sessions WHERE thread_id=?"), 0)
        self.clock.advance(GRACE_SECONDS + 1)
        self.assertFalse(self.dispatcher.removing(THREAD))


class TerminalTests(unittest.TestCase):
    """An open terminal session does not refuse the removal: its claim file is left, its other files go."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        config = Config({"allow_user_ids": [ALLOWED_UID]}, home=self.home.home, env={})
        self.app = App.build(Paths.from_home(self.home.home), config, telegram=FakeTelegram(self.clock),
                             clock=self.clock, env={})
        self.addCleanup(self.app.close)
        self.app.kv.set(GROUP_KEY, str(CHAT))
        self.logger = FakeLogger()

    def test_the_claim_stays_and_the_removal_goes_on(self) -> None:
        view = self.app.paths.view
        self.assertTrue(take_claim(view, THREAD, os.getpid(), self.clock.now()))
        json_path = os.path.join(view, f"{THREAD}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        remover = _remover(live=self.app.live, linger=self.app.linger, dispatcher=FakeDispatcher(),
                           forget=FakeForget(), logger=self.logger, clock=self.clock, removals=Removals(self.clock),
                           view_dir=view)
        text = remover.remove(THREAD, delete_in_telegram=False)
        self.assertEqual(text, f"topic {THREAD} removed (telegram: kept)")
        self.assertTrue(os.path.exists(claim_path(view, THREAD)), "the terminal's claim file is left alone")
        self.assertFalse(os.path.exists(json_path))
        self.assertIn(f"topic {THREAD}: {TERMINAL_OPEN_TEXT}", self.logger.lines)


if __name__ == "__main__":
    unittest.main()
