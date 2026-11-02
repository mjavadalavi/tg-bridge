"""ClaudeSession with a real child process (tests/fakes/fake_claude.py) in the linger scenarios: the print transport
reports the background work still pending (BackgroundWork) and the turns the CLI starts by itself (TurnBegan).
Scenarios: a Bash task counts from its start to its notification and its self turn follows; the user's own turns, the
first one included, never begin a turn; subagent text while idle begins none; the string "true" flag counts as
background; a ScheduleWakeup counts until its scheduled turn starts; a quick background task still brings its self turn;
closing stdin keeps a session with pending work alive (spike S5); stop() while idle ends the process; a permission ask
made while idle is emitted.
"""

import os
import sys
import unittest

import tests.fakes as fakes_package
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.claude.session import ClaudeSession
from tg_bridge.agents.events import ApprovalRequest, BackgroundWork, TextEvent, TurnBegan, TurnResult
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

WAIT = 15.0
FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_claude.py")


def trace(sink: RecordingSink) -> list[tuple]:
    """The linger-relevant events in arrival order: ('work', pending), ('began', origin), ('text', text), ('result', answer)."""
    steps: list[tuple] = []
    for event in list(sink.events):
        if isinstance(event, BackgroundWork):
            steps.append(("work", event.pending))
        elif isinstance(event, TurnBegan):
            steps.append(("began", event.origin))
        elif isinstance(event, TextEvent) and event.parent_id is None:
            steps.append(("text", event.text))
        elif isinstance(event, TurnResult):
            steps.append(("result", event.answer))
    return steps


def began_events(sink: RecordingSink) -> list[TurnBegan]:
    return [event for event in list(sink.events) if isinstance(event, TurnBegan)]


def is_result(answer: str):
    return lambda event: isinstance(event, TurnResult) and event.answer == answer


class LingerSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.flag = os.path.join(self.home.state, "bg.flag")
        self.sessions: list[ClaudeSession] = []

    def tearDown(self) -> None:
        for session in self.sessions:
            session.stop(grace=1)

    def start(self, **fake_env: str) -> tuple[ClaudeSession, RecordingSink]:
        """A started session whose fake CLI runs the given FAKE_* scenario."""
        env = dict(os.environ)
        env.update(fake_env)
        launcher = LauncherSpec(name="fake", backend="claude", command="fake",
                                argv=(sys.executable, "-I", FAKE), config_dir=self.home.home)
        spec = SessionSpec(
            thread_id=7, chat_id=-100, topic="side", launcher=launcher, cwd=self.home.home, mode="default",
            model=None, resume_id=None, new_id=None, allow_rules=("Read",), deny_tools=(), system_rules="RULES",
            env=env, errfile_dir=os.path.join(self.home.state, "turns"),
        )
        sink = RecordingSink()
        session = ClaudeSession(spec, sink, logger=self.logger, clock=self.clock)
        self.sessions.append(session)
        session.start()
        return session, sink

    def touch_flag(self) -> None:
        with open(self.flag, "w", encoding="utf-8"):
            pass

    def test_bash_background_reports_work_then_a_self_turn(self) -> None:
        session, sink = self.start(FAKE_BG="bash", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.assertEqual(trace(sink), [("work", 1), ("text", "STARTED"), ("result", "STARTED")])
        self.assertTrue(session.alive())  # the process waits for its background work
        self.touch_flag()
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        self.assertEqual(trace(sink)[3:], [("work", 0), ("began", "task-notification"), ("text", "FINISHED"),
                                           ("result", "FINISHED")])

    def test_first_turn_never_emits_turn_began(self) -> None:
        session, sink = self.start(FAKE_BG="bash", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.assertEqual(began_events(sink), [])
        self.touch_flag()
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        self.assertEqual(len(began_events(sink)), 1)  # only the self turn began
        session.send_user("again")  # a user's own turn: the CLI answers it, and it does not begin a turn
        self.assertTrue(sink.wait_for(is_result("ok:2")))
        self.assertEqual(len(began_events(sink)), 1)

    def test_subagent_text_while_idle_is_not_a_turn_start(self) -> None:
        session, sink = self.start(FAKE_BG="agent", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.touch_flag()
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        subagent = [e for e in list(sink.events) if isinstance(e, TextEvent) and e.text == "AGENTOK"]
        self.assertEqual(len(subagent), 1)
        self.assertIsNotNone(subagent[0].parent_id)
        began = [i for i, e in enumerate(list(sink.events)) if isinstance(e, TurnBegan)]
        self.assertEqual(len(began), 1)
        self.assertGreater(began[0], list(sink.events).index(subagent[0]))  # the subagent text came first, no turn began

    def test_agent_string_true_counts_as_background(self) -> None:
        session, sink = self.start(FAKE_BG="agent_str", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.assertEqual(trace(sink), [("work", 1), ("text", "STARTED"), ("result", "STARTED")])
        self.touch_flag()
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        self.assertEqual(trace(sink)[3:], [("work", 0), ("began", "task-notification"), ("text", "FINISHED"),
                                           ("result", "FINISHED")])

    def test_wakeup_counts_until_command_lifecycle(self) -> None:
        session, sink = self.start(FAKE_BG="wakeup")  # no flag: the wakeup fires after the default 0.5 s
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("AWAKE")))
        self.assertEqual(trace(sink), [("work", 1), ("text", "STARTED"), ("result", "STARTED"), ("work", 0),
                                       ("began", "scheduled"), ("text", "AWAKE"), ("result", "AWAKE")])

    def test_quick_background_still_brings_a_self_turn(self) -> None:
        session, sink = self.start(FAKE_BG="bash", FAKE_BG_QUICK="1")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        self.assertEqual(trace(sink), [("work", 1), ("text", "STARTED"), ("work", 0), ("result", "STARTED"),
                                       ("began", "task-notification"), ("text", "FINISHED"), ("result", "FINISHED")])

    def test_closing_stdin_does_not_end_a_session_with_pending_work(self) -> None:
        session, sink = self.start(FAKE_BG="bash", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.assertFalse(session.end_gracefully(0.3))  # spike S5: the CLI waits for its background work
        self.assertTrue(session.alive())
        session.stop(grace=2)
        self.assertFalse(session.alive())

    def test_stop_while_idle_ends_the_process(self) -> None:
        session, sink = self.start(FAKE_BG="bash")  # the self turn follows after 0.5 s; nothing is pending after it
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("FINISHED")))
        self.assertTrue(session.alive())
        session.stop(grace=2)
        self.assertFalse(session.alive())

    def test_idle_permission_request_is_emitted(self) -> None:
        session, sink = self.start(FAKE_BG="bash", FAKE_BG_ASK="1", FAKE_BG_FLAG=self.flag, FAKE_BG_SECONDS="30")
        session.send_user("go")
        self.assertTrue(sink.wait_for(is_result("STARTED")))
        self.touch_flag()  # the ask comes only after the first result, whatever the machine's load
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        events = list(sink.events)
        asks = [e for e in events if isinstance(e, ApprovalRequest)]
        self.assertEqual((asks[0].approval_id, asks[0].tool), ("perm_bg", "Bash"))
        self.assertGreater(events.index(asks[0]), events.index(next(e for e in events if isinstance(e, TurnResult))))
        session.answer_permission("perm_bg", True)
        self.assertTrue(sink.wait_for(is_result("FINISHED")))


if __name__ == "__main__":
    unittest.main()
