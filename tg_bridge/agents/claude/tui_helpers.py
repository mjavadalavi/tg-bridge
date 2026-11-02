"""Helpers of the tui transport (agents/claude/tui_session.py): the child's environment and socket path. Pure functions,
moved out of tui_session to keep it under the size ceiling; nothing here holds state. The per-turn bookkeeping of a
transcript line lives in tui_turn.py.
"""

import os
import tempfile
import uuid
from collections.abc import Mapping

from tg_bridge.agents.claude.hookbridge import ENV_SOCKET

TERM = "xterm-256color"
SOCKET_PATH_MAX_BYTES = 100  # AF_UNIX paths hold 108 bytes; the margin keeps a longer folder name out of trouble


def _child_env(env: Mapping[str, str], socket_path: str) -> dict[str, str]:
    """The child's environment: CLAUDECODE and every CLAUDE_CODE_* variable removed (the TUI would take them as its own
    session's), other settings kept, and the terminal type and the hook socket added."""
    child = {k: v for k, v in env.items() if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")}
    child["TERM"] = TERM
    child[ENV_SOCKET] = socket_path
    return child


def permission_args(mode: str) -> list[str]:
    """The permission flags of the TUI for a topic's mode. The bridge's tui transport and the view's pane both take them
    from here, so a topic opens with the same permissions in either place."""
    return ["--permission-mode", mode, "--allow-dangerously-skip-permissions"]


def _socket_path(errfile_dir: str, thread_id: int) -> str:
    """A per-session socket in the state folder. A path too long for AF_UNIX goes to the temp folder instead."""
    name = f"hook-{thread_id}-{uuid.uuid4().hex[:8]}.sock"
    path = os.path.join(errfile_dir, name)
    if len(os.fsencode(path)) > SOCKET_PATH_MAX_BYTES:
        path = os.path.join(tempfile.gettempdir(), name)
    return path

