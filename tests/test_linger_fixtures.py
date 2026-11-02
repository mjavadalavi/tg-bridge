"""Linger fixtures (C2f): redacted copies of the spike logs that the linger cards test against.

Fixtures: tests/fixtures/claude_lines/linger_print_{bash,monitor,agent,wakeup}.jsonl (print transport, stream-json)
and tests/fixtures/claude_transcripts/linger_tui_{bash,monitor,agent,wakeup}.jsonl (tui transport, transcript lines).
Scenarios: every line of every fixture is one JSON object; no home or scratchpad path is left; no key-shaped or
bearer-shaped value is present; each file has at most 60 lines; each file keeps the start and end markers of its
background scenario (spec 0.1 for print, 0.2 for tui), so the cards can test against the real line shapes.
"""

import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PRINT_DIR = os.path.join(HERE, "fixtures", "claude_lines")
TUI_DIR = os.path.join(HERE, "fixtures", "claude_transcripts")
MAX_LINES = 60
SECRET_RE = re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9_-]{20,}|Bearer\s")
LEAK_MARKERS = ("/home/", "scratchpad")

# File name -> substrings that must appear in the file's raw text (the scenario's start and end markers).
MARKERS = {
    os.path.join(PRINT_DIR, "linger_print_bash.jsonl"): [
        '"subtype": "task_started"', '"backgroundTaskId"', '"subtype": "task_updated"',
        '"subtype": "task_notification"', '"origin": {"kind": "task-notification"'],
    os.path.join(PRINT_DIR, "linger_print_monitor.jsonl"): [
        '"subtype": "task_started"', '"taskId"', '"timeoutMs"', '"subtype": "task_notification"',
        '"origin": {"kind": "task-notification"'],
    os.path.join(PRINT_DIR, "linger_print_agent.jsonl"): [
        '"subtype": "task_started"', '"isAsync": true', '"status": "async_launched"',
        '"parent_tool_use_id": "call_', '"subtype": "task_notification"', '"origin": {"kind": "task-notification"'],
    os.path.join(PRINT_DIR, "linger_print_wakeup.jsonl"): [
        '"is_error": true', '"tool_use_result": {"scheduledFor"', 'Next wakeup scheduled for',
        '"type": "command_lifecycle"', '"state": "started"', '"wakeup_source": "loop_wakeup"',
        '"state": "completed"'],
    os.path.join(TUI_DIR, "linger_tui_bash.jsonl"): [
        '"backgroundTaskId"', '"operation": "enqueue"', '<task-notification>', '<status>completed</status>',
        '"turnOrigin": "task_notification"', '"subtype": "turn_duration"'],
    os.path.join(TUI_DIR, "linger_tui_monitor.jsonl"): [
        '"taskId"', '"operation": "enqueue"', '<task-notification>', '<status>completed</status>',
        '"turnOrigin": "task_notification"', '"subtype": "turn_duration"'],
    os.path.join(TUI_DIR, "linger_tui_agent.jsonl"): [
        '"isAsync": true', '"status": "async_launched"', '"pendingBackgroundAgentCount": 1',
        '<task-notification>', '<status>completed</status>', '"turnOrigin": "task_notification"'],
    os.path.join(TUI_DIR, "linger_tui_wakeup.jsonl"): [
        '"scheduledFor"', '"subtype": "scheduled_task_fire"', '"turnOrigin": "scheduled"', '"isMeta": true',
        '<command-name>/exit</command-name>', '<local-command-caveat>', '<local-command-stdout>'],
}


def _read(path: str) -> str:
    """The whole fixture file as text (UTF-8)."""
    with open(path, encoding="utf-8") as f:
        return f.read()


class LingerFixtureTests(unittest.TestCase):
    def test_every_fixture_line_is_a_json_object(self) -> None:
        for path in MARKERS:
            with self.subTest(path=os.path.basename(path)):
                for number, line in enumerate(_read(path).splitlines(), start=1):
                    value = json.loads(line)
                    self.assertIsInstance(value, dict, f"line {number} of {path} is not a JSON object")

    def test_no_home_or_scratchpad_paths_left(self) -> None:
        for path in MARKERS:
            text = _read(path)
            for leak in LEAK_MARKERS:
                with self.subTest(path=os.path.basename(path), leak=leak):
                    self.assertNotIn(leak, text)

    def test_no_key_or_token_shaped_value(self) -> None:
        for path in MARKERS:
            with self.subTest(path=os.path.basename(path)):
                self.assertIsNone(SECRET_RE.search(_read(path)))

    def test_each_file_is_small(self) -> None:
        for path in MARKERS:
            with self.subTest(path=os.path.basename(path)):
                self.assertLessEqual(len(_read(path).splitlines()), MAX_LINES)

    def test_markers_present(self) -> None:
        for path, markers in MARKERS.items():
            text = _read(path)
            for marker in markers:
                with self.subTest(path=os.path.basename(path), marker=marker):
                    self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
