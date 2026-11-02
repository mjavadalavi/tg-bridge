"""BackgroundLedger: the background work a Claude session still has pending (agents/claude/background.py).

A Claude session can leave work running after its turn ends: a background Bash command, a Monitor, a background Agent, or
a ScheduleWakeup. The ledger counts that work from the raw lines of the session, for both transports: print (the
stream-json objects) and tui (the transcript lines). It is pure bookkeeping: no I/O, no clock, no lock. It never raises on
a line it does not know, so a CLI change can only leave the count stale, never crash the reader.

Threads: the ledger holds no lock of its own. Its owner (ClaudeSession and ClaudeTuiSession) calls note() and pending()
under the owner's own lock, exactly as it does for its other turn state.

Lock map: none here. The owner's lock guards the ledger; the ledger never takes a lock.
"""

import re

TERMINAL_STATUSES = frozenset({"completed", "failed", "killed", "stopped", "cancelled", "canceled", "error", "timeout",
                               "expired"})
BACKGROUND_TOOLS = frozenset({"Bash", "Agent", "Task"})   # these take run_in_background
ALWAYS_BACKGROUND_TOOLS = frozenset({"Monitor"})          # a Monitor always runs in the background
WAKEUP_TOOL = "ScheduleWakeup"
LOOP_WAKEUP = "loop_wakeup"                                # the wakeup_source of a scheduled turn
NOTIFICATION_OPEN = "<task-notification>"                  # the first text of a tui task notification
TASK_ID_TAG = re.compile(r"<task-id>(.*?)</task-id>", re.DOTALL)
TOOL_USE_ID_TAG = re.compile(r"<tool-use-id>(.*?)</tool-use-id>", re.DOTALL)
STATUS_TAG = re.compile(r"<status>(.*?)</status>", re.DOTALL)


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None  # JSON numbers only; no bools


def _true(value: object) -> bool:
    """run_in_background is a boolean in the print transport and the string "true" in the tui transport."""
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _blocks(message: object) -> list[dict]:
    """The dict blocks of a message's content list; [] for a string content or anything else."""
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def _text(message: object) -> str:
    """The text of a user message: its string content, or the text of its first text block."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    for block in _blocks(message):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    return ""


class BackgroundLedger:
    """Pending background tasks (by task id, with the tool-use id when known) and scheduled wakeups. Not thread-safe:
    the owning session calls it under its own lock."""

    def __init__(self) -> None:
        self._tasks: dict[str, str] = {}   # task id -> tool-use id ('' when unknown)
        self._wakeups: list[int] = []      # scheduledFor (epoch ms), oldest first
        self._calls: dict[str, str] = {}   # tool-use id -> tool name, for background calls whose result is not seen yet

    def note(self, line: dict) -> bool:
        """Apply one raw line. True when pending() changed. Unknown or malformed lines change nothing."""
        before = self.pending()
        if isinstance(line, dict):
            self._apply(line)
        return self.pending() != before

    def pending(self) -> int:
        return len(self._tasks) + len(self._wakeups)

    def _apply(self, line: dict) -> None:
        kind = line.get("type")
        if kind == "assistant":
            self._calls_in(line.get("message"))
        elif kind == "user":
            self._user(line)
        elif kind == "system":
            self._system(line)
        elif kind == "command_lifecycle":
            self._lifecycle(line)

    def _calls_in(self, message: object) -> None:
        """Remember the background calls of a message (any sidechain too), until their result arrives."""
        for block in _blocks(message):
            name = _str(block.get("name"))
            tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            background = name in ALWAYS_BACKGROUND_TOOLS or (
                name in BACKGROUND_TOOLS and _true(tool_input.get("run_in_background")))
            tool_use_id = _str(block.get("id"))
            if block.get("type") == "tool_use" and tool_use_id and (background or name == WAKEUP_TOOL):
                self._calls[tool_use_id] = name

    def _user(self, line: dict) -> None:
        message = line.get("message")
        result = line.get("tool_use_result") or line.get("toolUseResult")  # print or tui spelling
        result = result if isinstance(result, dict) else {}
        for block in _blocks(message):
            if block.get("type") == "tool_result":
                self._result(block, result)
        text = _text(message)
        if text.lstrip().startswith(NOTIFICATION_OPEN):
            self._tui_notification(text)

    def _result(self, block: dict, result: dict) -> None:
        """The start confirmation of one tool call: which kind of background work it began, if any."""
        tool_use_id = _str(block.get("tool_use_id"))
        called = self._calls.pop(tool_use_id, "") if tool_use_id else ""
        if block.get("is_error") is True:
            return
        task_id = _str(result.get("backgroundTaskId"))  # Bash
        if not task_id and "timeoutMs" in result:
            task_id = _str(result.get("taskId"))  # Monitor
        agent_id = _str(result.get("agentId"))
        is_async = result.get("isAsync") is True or result.get("status") == "async_launched"
        scheduled = _int(result.get("scheduledFor"))
        if task_id:
            self._add_task(task_id, tool_use_id)
        elif agent_id and is_async:
            self._add_task(agent_id, tool_use_id)
        elif scheduled is not None:
            self._wakeups.append(scheduled)
        elif called and called != WAKEUP_TOOL and tool_use_id:
            self._add_task(tool_use_id, tool_use_id)  # a CLI without the result dict: keyed by the tool-use id

    def _system(self, line: dict) -> None:
        subtype = line.get("subtype")
        if subtype == "task_started":
            task_id = _str(line.get("task_id"))
            if task_id:
                self._add_task(task_id, _str(line.get("tool_use_id")))
        elif subtype == "background_tasks_changed":
            self._snapshot(line.get("tasks"))
        elif subtype == "task_notification":
            if _str(line.get("status")) in TERMINAL_STATUSES:
                self._remove(_str(line.get("task_id")), _str(line.get("tool_use_id")))
        elif subtype == "scheduled_task_fire" and self._wakeups:
            del self._wakeups[0]

    def _lifecycle(self, line: dict) -> None:
        """A scheduled turn started: its wakeup is no longer pending."""
        if line.get("state") == "started" and line.get("wakeup_source") == LOOP_WAKEUP and self._wakeups:
            scheduled = _int(line.get("scheduled_for"))
            index = self._wakeups.index(scheduled) if scheduled in self._wakeups else 0
            del self._wakeups[index]

    def _tui_notification(self, text: str) -> None:
        status = STATUS_TAG.search(text)
        if status is None or status.group(1).strip() not in TERMINAL_STATUSES:
            return
        task = TASK_ID_TAG.search(text)
        tool_use = TOOL_USE_ID_TAG.search(text)
        self._remove(task.group(1).strip() if task else "", tool_use.group(1).strip() if tool_use else "")

    def _add_task(self, task_id: str, tool_use_id: str) -> None:
        if tool_use_id and tool_use_id in self._tasks.values():
            return  # this call is already counted under another id (a CLI that names the task differently)
        self._tasks[task_id] = tool_use_id or self._tasks.get(task_id, "")

    def _snapshot(self, tasks: object) -> None:
        """The task set becomes exactly the listed tasks; the tool-use id of a task already known is kept."""
        if not isinstance(tasks, list):
            return
        listed: dict[str, str] = {}
        for task in tasks:
            task_id = _str(task.get("task_id")) if isinstance(task, dict) else ""
            if task_id:
                listed[task_id] = self._tasks.get(task_id, _str(task.get("tool_use_id")))
        self._tasks = listed

    def _remove(self, task_id: str, tool_use_id: str) -> None:
        """Forget the tasks named by either id (a notification may carry only one of them)."""
        for key, value in list(self._tasks.items()):
            if (task_id and key == task_id) or (tool_use_id and value == tool_use_id):
                del self._tasks[key]
