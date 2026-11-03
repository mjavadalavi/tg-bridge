"""TranscriptTurns on its own (no PTY, no tmux): the turn of one Claude session read from its transcript.

The lines are fed straight to on_transcript, the tail generation 0 being the current one. The one test with a real tail
(the echo net) starts it on an empty transcript and stops it again, so no line is read from the file."""

import os
import time
import unittest

from tests.support import TempHome
from tg_bridge.agents.claude.hookbridge import TURN_ENDED
from tg_bridge.agents.claude.tui_turn import (ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE, INTERRUPTED_SUBTYPE,
                                              USER_ECHO_SECONDS)
from tg_bridge.agents.claude.transcript_turns import TranscriptTurns
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import TurnResult, TurnStatus, UsageEvent

TURN_DURATION_LINE = {"type": "system", "subtype": "turn_duration", "durationMs": 1234, "messageCount": 4}
INTERRUPT_LINE = {"type": "user", "message": {"role": "user", "content": "[Request interrupted by user]"}}
WAIT_SECONDS = 5.0


def echo_line(text: str) -> dict:
    """The typed message as the transcript writes it (the bridge's prompt is turnOrigin 'sdk')."""
    return {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": text}}


def assistant_line(text: str) -> dict:
    message = {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}],
               "usage": {"input_tokens": 3, "output_tokens": 2}}
    return {"type": "assistant", "isSidechain": False, "message": message}


class RecordingClock:
    """sleep() returns at once and records the seconds asked for, so a timer fires as soon as it starts."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class FakeLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: object) -> None:
        self.lines.append(str(msg))


class TranscriptTurnsTests(unittest.TestCase):
    def setUp(self) -> None:
        home = self.enterContext(TempHome()).home
        self.cwd = os.path.join(home, "work")
        os.makedirs(self.cwd)
        self.events: list[object] = []
        self.clock = RecordingClock()
        self.denied: list[tuple[str, bool]] = []
        self.turns = TranscriptTurns(config_dir=os.path.join(home, "config"), cwd=self.cwd, session_id="sid-1",
                                     translator=ClaudeTranslator(cwd=self.cwd), emit=self.events.append,
                                     log=lambda msg: None, logger=FakeLogger(), clock=self.clock,
                                     deny_requests=self.deny)
        self.addCleanup(self.turns.close)

    def test_is_open_only_for_the_current_turn_until_it_ends(self) -> None:
        gen, _primary, _follows = self.turns.begin()
        self.assertTrue(self.turns.is_open(gen))
        self.assertFalse(self.turns.is_open(gen + 1))
        self.turns.on_transcript(0, echo_line("hello"))
        self.turns.on_transcript(0, TURN_DURATION_LINE)
        self.assertFalse(self.turns.is_open(gen))

    def deny(self, reason: str, *, main_only: bool = False) -> list[str]:
        self.denied.append((reason, main_only))
        return []

    def outcomes(self) -> list[object]:
        return [e for e in self.events if isinstance(e, (TurnResult, UsageEvent))]

    def results(self) -> list[TurnResult]:
        return [e for e in self.events if isinstance(e, TurnResult)]

    def test_a_begun_turn_ends_ok_at_turn_duration_after_its_echo(self) -> None:
        gen, _primary, _follows = self.turns.begin()
        self.turns.on_transcript(0, echo_line("hello"))
        self.turns.on_transcript(0, assistant_line("ANSWER"))
        self.turns.on_transcript(0, TURN_DURATION_LINE)

        usage, result = self.outcomes()
        self.assertIsInstance(usage, UsageEvent)
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "ANSWER"))
        self.assertEqual(self.events.index(usage) + 1, self.events.index(result))
        self.assertEqual(gen, 1)

    def test_no_echo_within_thirty_seconds_with_a_tail_ends_the_turn_with_an_error(self) -> None:
        gen, primary, follows = self.turns.begin()
        os.makedirs(os.path.dirname(primary))
        open(primary, "w", encoding="utf-8").close()  # an empty transcript: the tail starts and reads nothing
        self.assertTrue(self.turns.follow_turn(gen, primary, follows))  # a tail follows, so the echo net is armed
        self.turns.watch_echo(gen)

        deadline = time.monotonic() + WAIT_SECONDS
        while not self.results() and time.monotonic() < deadline:
            time.sleep(0.01)
        (result,) = self.results()
        self.assertEqual((result.status, result.subtype, result.error),
                         (TurnStatus.ERROR, ECHO_MISSED_SUBTYPE, ECHO_MISSED_ERROR))
        self.assertIn(USER_ECHO_SECONDS, self.clock.sleeps)

    def foreign_turns(self) -> TranscriptTurns:
        """The attached session's mode: the same session, with foreign_interrupts on."""
        turns = TranscriptTurns(config_dir=self.turns._config_dir, cwd=self.cwd, session_id="sid-1",
                                translator=ClaudeTranslator(cwd=self.cwd), emit=self.events.append,
                                log=lambda msg: None, logger=FakeLogger(), clock=self.clock,
                                deny_requests=self.deny, foreign_interrupts=True)
        self.addCleanup(turns.close)
        return turns

    def test_an_interrupt_line_is_ignored_without_an_esc_of_ours_by_default(self) -> None:
        """The tui transport: an interrupt line counts only while the bridge's own Esc is awaited."""
        self.turns.begin()
        self.turns.on_transcript(0, echo_line("hello"))
        self.turns.on_transcript(0, INTERRUPT_LINE)
        self.assertEqual(self.results(), [])

        self.turns.on_transcript(0, assistant_line("ANSWER"))
        self.turns.on_transcript(0, TURN_DURATION_LINE)
        (result,) = self.results()
        self.assertEqual(result.status, TurnStatus.OK)

    def test_an_interrupt_line_ends_the_turn_as_interrupted_with_foreign_interrupts(self) -> None:
        """The attached session: an Esc or a denial in the terminal ends the turn, without an Esc of the bridge's."""
        turns = self.foreign_turns()
        turns.begin()
        turns.on_transcript(0, echo_line("hello"))
        turns.on_transcript(0, INTERRUPT_LINE)

        (result,) = self.results()
        self.assertEqual((result.status, result.subtype), (TurnStatus.INTERRUPTED, INTERRUPTED_SUBTYPE))
        self.assertEqual(self.denied, [(TURN_ENDED, True)])
        turns.on_transcript(0, TURN_DURATION_LINE)  # the turn is over: its late duration line starts nothing
        self.assertEqual(len(self.results()), 1)

    def test_an_interrupt_line_before_any_turn_is_not_a_turn(self) -> None:
        turns = self.foreign_turns()
        turns.on_transcript(0, INTERRUPT_LINE)
        self.assertEqual(self.results(), [])

    def test_a_stale_generation_is_ignored(self) -> None:
        stale, _, _ = self.turns.begin()
        self.turns.begin()  # a newer turn replaces the first one

        self.turns.finish(stale, error=ECHO_MISSED_ERROR)  # the old turn's echo net fires late
        self.turns.finish(stale, duration=TURN_DURATION_LINE)
        self.assertEqual(self.outcomes(), [])

        self.turns.on_transcript(0, echo_line("second"))
        self.turns.on_transcript(0, TURN_DURATION_LINE)
        (result,) = self.results()
        self.assertEqual(result.status, TurnStatus.OK)


if __name__ == "__main__":
    unittest.main()
