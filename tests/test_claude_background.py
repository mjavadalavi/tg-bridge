"""BackgroundLedger (tg_bridge/agents/claude/background.py), on its own, without a process.
Scenarios: a print Bash task counts from its start confirmation to its notification, a Monitor counts until its stream
ends, a background Agent counts until it completes, subagent text counts nothing, a ScheduleWakeup counts until its
scheduled turn starts, the same four shapes on the tui transport, a background_tasks_changed snapshot replaces the task
set, a non-terminal notification keeps its task, an error result counts nothing, the string "true" flag and the
tool-use-id fallback work, a call counted once when the task id and the tool-use id differ, the eight C2f fixtures replayed
line by line (each one counts one task or wakeup, then none), and malformed lines never raise."""

import json
import os
import unittest

from tg_bridge.agents.claude.background import BackgroundLedger

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
PRINT_FIXTURES = [os.path.join(FIXTURES, "claude_lines", f"linger_print_{kind}.jsonl")
                  for kind in ("bash", "monitor", "agent", "wakeup")]
TUI_FIXTURES = [os.path.join(FIXTURES, "claude_transcripts", f"linger_tui_{kind}.jsonl")
                for kind in ("bash", "monitor", "agent", "wakeup")]


def replay(path: str) -> list[tuple[int, bool]]:
    """(pending, changed) after each line of a fixture file, in file order."""
    ledger = BackgroundLedger()
    steps = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                changed = ledger.note(json.loads(raw))
                steps.append((ledger.pending(), changed))
    return steps


def tool_use(tool_id: str, name: str, tool_input: dict | None = None) -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}]}}


def print_result(tool_id: str, result: dict, *, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}, "tool_use_result": result}


def tui_result(tool_id: str, result: dict) -> dict:
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tool_id}]},
            "toolUseResult": result}


def task_started(task_id: str, tool_id: str) -> dict:
    return {"type": "system", "subtype": "task_started", "task_id": task_id, "tool_use_id": tool_id}


def task_notification(task_id: str, tool_id: str, status: str) -> dict:
    return {"type": "system", "subtype": "task_notification", "task_id": task_id, "tool_use_id": tool_id,
            "status": status}


def tui_notification(task_id: str, tool_id: str, status: str) -> dict:
    text = (f"<task-notification>\n<task-id>{task_id}</task-id>\n<tool-use-id>{tool_id}</tool-use-id>\n"
            f"<status>{status}</status>")
    return {"type": "user", "message": {"role": "user", "content": text}, "turnOrigin": "task_notification"}


def lifecycle(state: str, scheduled_for: int) -> dict:
    return {"type": "command_lifecycle", "state": state, "wakeup_source": "loop_wakeup",
            "scheduled_for": scheduled_for}


class BackgroundLedgerTests(unittest.TestCase):
    def test_print_bash_counts_one_until_the_notification(self) -> None:
        ledger = BackgroundLedger()
        self.assertFalse(ledger.note(tool_use("t1", "Bash", {"run_in_background": True})))
        self.assertTrue(ledger.note(task_started("b1", "t1")))
        self.assertFalse(ledger.note(print_result("t1", {"backgroundTaskId": "b1"})))  # same task, no change
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(task_notification("b1", "t1", "completed")))
        self.assertEqual(ledger.pending(), 0)

    def test_print_monitor_counts_one_until_stream_ended(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("m1", "Monitor", {"command": "tail -f x"}))
        ledger.note(print_result("m1", {"taskId": "mon1", "timeoutMs": 300000, "persistent": False}))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(task_notification("mon1", "m1", "completed")))
        self.assertEqual(ledger.pending(), 0)

    def test_print_agent_counts_one_until_completed(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("a1", "Agent", {"run_in_background": True}))
        ledger.note(print_result("a1", {"isAsync": True, "status": "async_launched", "agentId": "ag1"}))
        self.assertEqual(ledger.pending(), 1)
        ledger.note(task_notification("ag1", "a1", "completed"))
        self.assertEqual(ledger.pending(), 0)

    def test_print_subagent_text_changes_nothing(self) -> None:
        ledger = BackgroundLedger()
        line = {"type": "assistant", "parent_tool_use_id": "call_1", "agent_id": "ag1",
                "message": {"content": [{"type": "text", "text": "working"}]}}
        self.assertFalse(ledger.note(line))
        self.assertEqual(ledger.pending(), 0)

    def test_print_wakeup_counts_until_command_lifecycle_started(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("w1", "ScheduleWakeup", {"delaySeconds": 60}))
        self.assertTrue(ledger.note(print_result("w1", {"scheduledFor": 1700000000000})))
        self.assertEqual(ledger.pending(), 1)
        self.assertFalse(ledger.note({"type": "command_lifecycle", "state": "completed"}))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(lifecycle("started", 1700000000000)))
        self.assertEqual(ledger.pending(), 0)

    def test_tui_bash_counts_one_until_the_notification(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("t1", "Bash", {"run_in_background": "true"}))
        ledger.note(tui_result("t1", {"backgroundTaskId": "b1"}))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(tui_notification("b1", "t1", "completed")))
        self.assertEqual(ledger.pending(), 0)

    def test_tui_monitor_counts_one_until_the_notification(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("m1", "Monitor", {"command": "tail -f x"}))
        ledger.note(tui_result("m1", {"taskId": "mon1", "timeoutMs": 300000}))
        self.assertEqual(ledger.pending(), 1)
        ledger.note(tui_notification("mon1", "m1", "completed"))
        self.assertEqual(ledger.pending(), 0)

    def test_tui_agent_counts_one_until_the_notification(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("a1", "Agent", {"run_in_background": "true"}))
        ledger.note(tui_result("a1", {"isAsync": True, "status": "async_launched", "agentId": "ag1"}))
        self.assertEqual(ledger.pending(), 1)
        ledger.note(tui_notification("ag1", "a1", "completed"))
        self.assertEqual(ledger.pending(), 0)

    def test_tui_wakeup_counts_until_the_scheduled_fire(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("w1", "ScheduleWakeup", {"delaySeconds": 60}))
        ledger.note(tui_result("w1", {"scheduledFor": 1700000000000}))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note({"type": "system", "subtype": "scheduled_task_fire", "taskId": "s1",
                                     "prompt": "check"}))
        self.assertEqual(ledger.pending(), 0)

    def test_tui_non_terminal_notification_keeps_the_task(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("m1", "Monitor", {"command": "tail -f x"}))
        ledger.note(tui_result("m1", {"taskId": "mon1", "timeoutMs": 300000}))
        self.assertFalse(ledger.note(tui_notification("mon1", "m1", "running")))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(tui_notification("mon1", "m1", "completed")))
        self.assertEqual(ledger.pending(), 0)

    def test_snapshot_replaces_the_task_set(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(task_started("b1", "t1"))
        snapshot = {"type": "system", "subtype": "background_tasks_changed", "tasks": [
            {"task_id": "b1", "task_type": "local_bash"}, {"task_id": "b2", "task_type": "local_agent"}]}
        self.assertTrue(ledger.note(snapshot))
        self.assertEqual(ledger.pending(), 2)
        self.assertTrue(ledger.note({"type": "system", "subtype": "background_tasks_changed", "tasks": []}))
        self.assertEqual(ledger.pending(), 0)

    def test_non_terminal_notification_keeps_the_task(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(task_started("mon1", "m1"))
        self.assertFalse(ledger.note(task_notification("mon1", "m1", "running")))
        self.assertEqual(ledger.pending(), 1)

    def test_error_tool_result_counts_nothing(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("w1", "ScheduleWakeup", {"delaySeconds": 60}))
        self.assertFalse(ledger.note(print_result("w1", {}, is_error=True)))
        self.assertFalse(ledger.note({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "w1", "is_error": True, "content": "noop is required"}]}}))
        self.assertFalse(ledger.note(print_result("w1", {"scheduledFor": 1700000000000}, is_error=True)))
        self.assertEqual(ledger.pending(), 0)

    def test_string_true_agent_fallback_by_tool_use_id(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("a1", "Agent", {"run_in_background": "TRUE"}))
        self.assertTrue(ledger.note(tui_result("a1", {"unrelated": 1})))  # no marker: keyed by the tool-use id
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(tui_notification("", "a1", "completed")))  # only the tool-use id is given
        self.assertEqual(ledger.pending(), 0)

    def test_result_with_another_id_than_task_started_counts_once(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("a1", "Agent", {"run_in_background": True}))
        ledger.note(task_started("task-x", "a1"))
        self.assertFalse(ledger.note(print_result("a1", {"isAsync": True, "status": "async_launched",
                                                         "agentId": "agent-y"})))
        self.assertEqual(ledger.pending(), 1)
        self.assertTrue(ledger.note(task_notification("task-x", "a1", "completed")))
        self.assertEqual(ledger.pending(), 0)

    def test_bare_result_after_task_started_counts_once(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("t1", "Bash", {"run_in_background": True}))
        ledger.note(task_started("b1", "t1"))
        self.assertFalse(ledger.note(print_result("t1", {})))  # the fallback key is the tool-use id, already counted
        self.assertEqual(ledger.pending(), 1)

    def test_fixtures_replay_one_task_then_none(self) -> None:
        for path in PRINT_FIXTURES + TUI_FIXTURES:
            with self.subTest(fixture=os.path.basename(path)):
                steps = replay(path)
                self.assertEqual([pending for pending, changed in steps if changed], [1, 0])
                self.assertEqual(sum(changed for _, changed in steps), 2)
                self.assertEqual(steps[-1][0], 0)

    def test_foreground_bash_without_the_flag_is_not_tracked(self) -> None:
        ledger = BackgroundLedger()
        ledger.note(tool_use("t1", "Bash", {"run_in_background": False}))
        self.assertFalse(ledger.note(tui_result("t1", {"stdout": "x"})))
        self.assertEqual(ledger.pending(), 0)

    def test_bad_shapes_never_raise(self) -> None:
        bad_lines = [
            None, "text", [], {},
            {"type": "assistant", "message": {"content": "not a list"}},
            {"type": "assistant", "message": {"content": [None, 3, {"type": "tool_use", "name": None}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": 5}]},
             "tool_use_result": "string"},
            {"type": "user", "message": {"content": "<task-notification>\n<status>completed</status>"}},
            {"type": "user", "message": {"content": [{"type": "text", "text": None}]}},
            {"type": "system", "subtype": "background_tasks_changed", "tasks": "nope"},
            {"type": "system", "subtype": "background_tasks_changed", "tasks": [None, {"task_id": 4}]},
            {"type": "system", "subtype": "task_notification", "status": "completed"},
            {"type": "system", "subtype": "task_started"},
            {"type": "command_lifecycle", "state": "started", "wakeup_source": "loop_wakeup",
             "scheduled_for": "soon"},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "w1"}]},
             "tool_use_result": {"scheduledFor": "2026"}},
        ]
        ledger = BackgroundLedger()
        for line in bad_lines:
            with self.subTest(line=line):
                self.assertIsInstance(ledger.note(line), bool)
        self.assertEqual(ledger.pending(), 0)


if __name__ == "__main__":
    unittest.main()
