"""ClaudeSession (print transport): the end of an interrupted turn. An interrupt of a self turn makes the CLI end it with a
result of subtype error_during_execution; that result is TurnStatus.INTERRUPTED, so LingerService keeps the session.
Every other error stays ERROR, and the interrupt's intent never reaches the turn after the one it was aimed at.

The session is not started: no child process runs. Lines go to _on_line as the reader thread would deliver them, and
the interrupt's control request is answered by a stub on the session's channel.
Scenarios: interrupt then error_during_execution is INTERRUPTED; no interrupt keeps ERROR; another error subtype keeps
ERROR after an interrupt; the intent ends with its turn; an interrupt made while idle reaches neither the next user turn
nor a self turn; a failed interrupt leaves no intent; a result that arrives before the interrupt's reply is INTERRUPTED.
"""

import json
import os
import unittest

from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.claude.session import ClaudeSession
from tg_bridge.agents.events import TurnBegan, TurnResult, TurnStatus
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

ERROR_DURING_EXECUTION = {"type": "result", "subtype": "error_during_execution", "is_error": True,
                          "session_id": "sess-1", "num_turns": 1, "duration_ms": 120}
ERROR_MAX_TURNS = {"type": "result", "subtype": "error_max_turns", "is_error": True, "session_id": "sess-1",
                   "errors": ["max turns reached"]}
SUCCESS = {"type": "result", "subtype": "success", "is_error": False, "result": "done", "session_id": "sess-1"}
SELF_TEXT = {"type": "assistant", "session_id": "sess-1", "message": {"content": [{"type": "text", "text": "update"}]}}


def results(sink: RecordingSink) -> list[TurnResult]:
    return [event for event in list(sink.events) if isinstance(event, TurnResult)]


def accepting(session: ClaudeSession) -> None:
    """The interrupt's control request is answered at once, as the CLI does."""
    session._channel.request = lambda subtype, timeout=0.0, **fields: {}


class InterruptedResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        logger = Logger(os.path.join(self.home.state, "bridge.log"))
        launcher = LauncherSpec(name="fake", backend="claude", command="fake", argv=("fake",),
                                config_dir=self.home.home)
        spec = SessionSpec(
            thread_id=7, chat_id=-100, topic="side", launcher=launcher, cwd=self.home.home, mode="default",
            model=None, resume_id=None, new_id=None, allow_rules=("Read",), deny_tools=(), system_rules="RULES",
            env=dict(os.environ), errfile_dir=os.path.join(self.home.state, "turns"),
        )
        self.sink = RecordingSink()
        self.session = ClaudeSession(spec, self.sink, logger=logger, clock=FakeClock())

    def feed(self, event: dict) -> None:
        """One stdout line, delivered as the reader thread would."""
        self.session._on_line(json.dumps(event))

    def test_interrupted_turn_ends_as_interrupted_not_error(self) -> None:
        accepting(self.session)
        self.session.send_user("go")
        self.assertTrue(self.session.interrupt())
        self.feed(ERROR_DURING_EXECUTION)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.INTERRUPTED)
        self.assertIsNone(turn.error)
        self.assertEqual(turn.subtype, "error_during_execution")

    def test_error_without_an_interrupt_stays_error(self) -> None:
        self.session.send_user("go")
        self.feed(ERROR_DURING_EXECUTION)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)
        self.assertEqual(turn.error, "no detail")

    def test_other_error_subtypes_stay_error_after_an_interrupt(self) -> None:
        accepting(self.session)
        self.session.send_user("go")
        self.assertTrue(self.session.interrupt())
        self.feed(ERROR_MAX_TURNS)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)
        self.assertEqual(turn.error, "max turns reached")

    def test_intent_ends_with_its_turn(self) -> None:
        accepting(self.session)
        self.session.send_user("first")
        self.assertTrue(self.session.interrupt())
        self.feed(ERROR_DURING_EXECUTION)
        self.session.send_user("second")
        self.feed(ERROR_DURING_EXECUTION)
        first, second = results(self.sink)
        self.assertEqual(first.status, TurnStatus.INTERRUPTED)
        self.assertEqual(second.status, TurnStatus.ERROR)

    def test_interrupt_while_idle_does_not_reach_the_next_user_turn(self) -> None:
        accepting(self.session)
        self.assertTrue(self.session.interrupt())
        self.session.send_user("next")
        self.feed(ERROR_DURING_EXECUTION)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)

    def test_interrupt_while_idle_does_not_reach_a_self_turn(self) -> None:
        accepting(self.session)
        self.assertTrue(self.session.interrupt())
        self.feed(SELF_TEXT)
        self.assertEqual([e for e in self.sink.events if isinstance(e, TurnBegan)], [TurnBegan(origin="")])
        self.feed(ERROR_DURING_EXECUTION)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)

    def test_failed_interrupt_leaves_no_intent(self) -> None:
        def no_reply(subtype: str, timeout: float = 0.0, **fields: object) -> dict:
            raise RuntimeError("stdin closed")

        self.session._channel.request = no_reply
        self.session.send_user("go")
        self.assertFalse(self.session.interrupt())
        self.feed(ERROR_DURING_EXECUTION)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.ERROR)

    def test_result_arriving_before_the_interrupt_reply_is_interrupted(self) -> None:
        def reply_after_result(subtype: str, timeout: float = 0.0, **fields: object) -> dict:
            self.feed(ERROR_DURING_EXECUTION)  # the CLI may end the turn before it answers
            return {}

        self.session._channel.request = reply_after_result
        self.session.send_user("go")
        self.assertTrue(self.session.interrupt())
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.INTERRUPTED)

    def test_a_successful_turn_after_an_interrupt_is_ok(self) -> None:
        accepting(self.session)
        self.session.send_user("go")
        self.assertTrue(self.session.interrupt())
        self.feed(SUCCESS)
        (turn,) = results(self.sink)
        self.assertEqual(turn.status, TurnStatus.OK)


if __name__ == "__main__":
    unittest.main()
