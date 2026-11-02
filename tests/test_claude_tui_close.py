"""ClaudeTuiSession closing (the tui transport): the lines the CLI writes just before it exits are still read. When the TUI
exits by itself (a natural exit), the close drains the transcript with self-turn detection on, so a background update
written just before the exit opens a self turn, its answer and turn_duration produce the TurnResult, and all of that comes
before ProcessExited. An intentional close (end_gracefully, stop) marks the session ending first, so its drain begins no
turn. Once the session is closed, a transcript line is no longer handled.

The tail here never polls on its own (a poll interval of an hour, and no thread started), so the lines are read only by
the final drain of the close. The session is never started: no child process runs."""

import json
import os

from tests.test_claude_tui_session import TuiTestCase, results
from tg_bridge.agents.claude.tail import TranscriptTail
from tg_bridge.agents.events import ProcessExited, TurnBegan, TurnResult, TurnStatus

NEVER_POLLS = 3600.0
USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}


def assistant_line(text: str) -> dict:
    message = {"id": "msg_last", "type": "message", "role": "assistant", "model": "fake",
               "content": [{"type": "text", "text": text}], "usage": USAGE}
    return {"type": "assistant", "isSidechain": False, "message": message}


TURN_DURATION_LINE = {"type": "system", "subtype": "turn_duration", "durationMs": 1234, "messageCount": 4}


class CloseDrainTests(TuiTestCase):
    def install_tail(self, session: object, path: str) -> TranscriptTail:
        """The tail a send would have followed: its lines reach _on_transcript of the current generation. Not started."""
        session._tail_gen += 1
        tail_gen = session._tail_gen
        tail = TranscriptTail(path, lambda line: session._on_transcript(tail_gen, line), logger=self.logger,
                              poll_seconds=NEVER_POLLS)
        session._tail = tail
        return tail

    def open_turn(self, session: object) -> None:
        """The state a send_user leaves behind: one open turn, reset and waiting for its lines."""
        with session._turn_lock:
            session._gen += 1
            session._done = False
            session._turn.reset()

    def test_lines_written_just_before_exit_end_the_turn_before_process_exited(self) -> None:
        sink, session = self.open("text")
        path = os.path.join(self.home.state, "close.jsonl")
        self.install_tail(session, path)
        self.open_turn(session)
        with open(path, "w", encoding="utf-8") as f:  # the CLI writes its answer and turn_duration, then exits
            for line in (assistant_line("LAST ANSWER"), TURN_DURATION_LINE):
                f.write(json.dumps(line) + "\n")

        session._on_exited(0)  # the TUI is gone before any poll read the lines: the close must read them

        (result,) = results(sink)
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "LAST ANSWER"))
        exited = [i for i, e in enumerate(sink.events) if isinstance(e, ProcessExited)]
        self.assertEqual(len(exited), 1)
        self.assertLess(sink.events.index(result), exited[0])

    def test_a_background_update_just_before_a_natural_exit_opens_a_self_turn(self) -> None:
        sink, session = self.open("text")
        path = os.path.join(self.home.state, "ending.jsonl")
        self.install_tail(session, path)  # no open turn: _done stays True, as after a finished turn
        notification = {"type": "user", "turnOrigin": "task_notification",
                        "message": {"role": "user", "content": "<task-notification>done</task-notification>"}}
        with open(path, "w", encoding="utf-8") as f:  # a self turn the CLI starts, its answer, then it exits
            for line in (notification, assistant_line("LATE ANSWER"), TURN_DURATION_LINE):
                f.write(json.dumps(line) + "\n")

        session._on_exited(0)  # a natural exit: the final drain still detects the self turn

        (began,) = [e for e in sink.events if isinstance(e, TurnBegan)]
        (result,) = results(sink)
        self.assertEqual((result.status, result.answer), (TurnStatus.OK, "LATE ANSWER"))
        exited = [i for i, e in enumerate(sink.events) if isinstance(e, ProcessExited)]
        self.assertLess(sink.events.index(began), sink.events.index(result))
        self.assertLess(sink.events.index(result), exited[0])

    def test_an_intentional_close_starts_no_turn_from_a_drained_line(self) -> None:
        sink, session = self.open("text")
        path = os.path.join(self.home.state, "intentional.jsonl")
        self.install_tail(session, path)  # no open turn, as after a finished turn
        notification = {"type": "user", "turnOrigin": "task_notification",
                        "message": {"role": "user", "content": "<task-notification>done</task-notification>"}}
        with open(path, "w", encoding="utf-8") as f:  # what /exit makes the CLI write: a line, its answer, then exit
            for line in (notification, assistant_line("LATE ANSWER"), TURN_DURATION_LINE):
                f.write(json.dumps(line) + "\n")

        self.assertTrue(session.end_gracefully(0.0))  # no process ran: the close is marked, then the session is gone
        session._on_exited(0)

        self.assertEqual([e for e in sink.events if isinstance(e, (TurnBegan, TurnResult))], [])

    def test_a_line_read_after_the_close_is_not_handled(self) -> None:
        sink, session = self.open("text")
        path = os.path.join(self.home.state, "closed.jsonl")
        self.install_tail(session, path)
        self.open_turn(session)
        session._on_exited(0)
        gen = session._gen
        before = list(sink.events)
        session._on_transcript(session._tail_gen, assistant_line("TOO LATE"))  # a stale line, after the close
        session._on_transcript(session._tail_gen, TURN_DURATION_LINE)
        self.assertEqual(sink.events, before)
        self.assertEqual(session._gen, gen)
