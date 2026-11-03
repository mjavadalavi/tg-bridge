"""The `tg-bridge hook session-start` client: a SessionStart hook moves the tab's terminal claim to the session it runs.

Claude Code runs this hook when the CLI of a `tg-bridge view` tab starts, clears, resumes or compacts a session. It
reads the hook's JSON from stdin (session_id, source) and sets the session uuid of the live claim that names the tab's
pid (terminal_claim.update_uuid), so the next message from Telegram follows the session the tab is in now. Its stdout
becomes model context for the session and a non-zero exit shows an error in the tab, so the function prints nothing
and always returns 0. Every outcome is one line in the bridge log.
"""

import json
from typing import TextIO

from tg_bridge.core import terminal_claim
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths


def run_session_start_hook(stdin: TextIO, thread: str, pid: str, paths: Paths) -> int:
    """Handles one SessionStart event for the tab of thread, run by the process pid. Always returns 0."""
    logger = Logger(paths.log)
    try:
        data = json.loads(stdin.read())
    except ValueError:
        logger.log(f"terminal: session-start hook of thread {thread} got bad JSON on stdin")
        return 0
    session_id = data.get("session_id") if isinstance(data, dict) else None
    if not isinstance(session_id, str) or not session_id:
        logger.log(f"terminal: session-start hook of thread {thread} got no session_id")
        return 0
    source = data.get("source")
    source = source if isinstance(source, str) else ""
    try:
        thread_id, pid_number = int(thread), int(pid)
    except ValueError:
        logger.log(f"terminal: session-start hook got thread {thread!r} and pid {pid!r}, not numbers")
        return 0
    try:
        moved = terminal_claim.update_uuid(paths.view, thread_id, pid_number, session_id, hooked=True)
    except OSError as exc:
        logger.log(f"terminal: session-start hook cannot update the claim of thread {thread}: {exc!r}")
        return 0
    if moved:
        logger.log(f"terminal: thread {thread} tab session now {session_id} ({source})")
    else:
        logger.log(f"terminal: thread {thread} tab session {session_id} not moved: no live claim of pid {pid}")
    return 0
