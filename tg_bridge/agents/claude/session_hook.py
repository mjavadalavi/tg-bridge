"""The SessionStart hook of a session that `tg-bridge view` runs in a tmux tab.

A `/clear` or a `/resume` in the tab starts a new Claude Code session id. Claude Code runs a SessionStart hook at that
moment, so with this hook installed through --settings the TUI reports the new id to the bridge. The command is
`tg-bridge hook session-start <thread> <pid>`: the interpreter, -I, this checkout's tg-bridge.py and the words, with the
thread id and the pid of the tab's process last. The hook only builds the command and its settings JSON here; the
bridge side that reads the report is elsewhere.
"""

import json
import shlex
import sys
from collections.abc import Sequence

from tg_bridge.agents.claude.hookbridge import MATCH_ALL, entry_script

SESSION_START_EVENT = "SessionStart"
SESSION_HOOK_WORDS = ("hook", "session-start")
SESSION_HOOK_TIMEOUT_SECONDS = 10


def session_start_command(thread_id: int, pid: int, python: str = sys.executable) -> list[str]:
    """The argv of the SessionStart hook for one tab. -I keeps the user's site packages and PYTHON* variables out, as
    for the permission hook; the entry script puts its own folder on sys.path itself."""
    return [python, "-I", entry_script(), *SESSION_HOOK_WORDS, str(thread_id), str(pid)]


def session_start_settings(command: Sequence[str], timeout_seconds: int = SESSION_HOOK_TIMEOUT_SECONDS) -> str:
    """The compact --settings JSON that installs command as the SessionStart hook for every source (startup, clear,
    resume and compact). The TUI runs the command through a shell, so the argv is quoted with shlex."""
    entry = {"type": "command", "command": shlex.join(command), "timeout": timeout_seconds}
    settings = {"hooks": {SESSION_START_EVENT: [{"matcher": MATCH_ALL, "hooks": [entry]}]}}
    return json.dumps(settings, separators=(",", ":"))
