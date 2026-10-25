"""Codex rollout -> HandoffEntry list, the source of a cross-backend handoff (DESIGN sections 3-4).

A Codex thread is stored as a rollout file at <codex_home>/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread id>.jsonl.
read_entries keeps what a new agent needs to pick the conversation up: the user's messages, the assistant's messages
(commentary and final answers alike), and one 'tool' line per tool call. A shell call reads as '$ <command>', any other
call as its name plus a short detail. Developer messages, injected context (AGENTS.md, environment_context), image
inputs, reasoning, tool output and every other line type are skipped. Nothing is copied or logged here, and the
functions never raise for a bad or missing file.
"""

import glob
import json
import os
import re
from typing import Any

from tg_bridge.agents.codex.instructions import is_rules_block
from tg_bridge.agents.events import HandoffEntry

SESSIONS_DIR = "sessions"
THREAD_ID = re.compile(r"[A-Za-z0-9-]{1,64}")   # a Codex thread id: safe to put in a glob pattern and a file name
RESPONSE_ITEM = "response_item"
MESSAGE = "message"
CONVERSATION_ROLES = ("user", "assistant")       # the only message roles that carry conversation
TEXT_BLOCKS = ("input_text", "output_text")
INJECTED_PREFIXES = ("# AGENTS.md instructions", "<environment_context>")  # context the bridge did not type
IMAGE_MARKERS = ("<image", "</image")           # the text wrapper Codex puts around an image input
TOOL_CALLS = ("function_call", "custom_tool_call", "local_shell_call")
ROLE_TOOL = "tool"
SHELL_TOOL = "shell"
BRIEF_CUT = 80                                  # a tool line is cut to this many characters
COMMAND_FIELD = re.compile(r'cmd:\s*"((?:[^"\\]|\\.)*)"')   # the command inside a code-mode exec call


def find_rollout(codex_home: str | None, thread_id: str | None) -> str | None:
    """The rollout file of a thread under codex_home, or None. The newest matching file wins."""
    if not codex_home or not thread_id or not THREAD_ID.fullmatch(thread_id):
        return None
    pattern = os.path.join(glob.escape(codex_home), SESSIONS_DIR, "*", "*", "*", f"rollout-*-{thread_id}.jsonl")
    matches = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not matches:
        return None
    return max(matches, key=lambda path: (_mtime(path), path))


def read_entries(path: str | None) -> tuple[HandoffEntry, ...]:
    """The conversation of one rollout in file order. An empty path, a missing file or a directory gives ()."""
    if not path:
        return ()
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ()
    entries: list[HandoffEntry] = []
    for line in lines:
        entries.extend(_line_entries(line))
    return tuple(entries)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _line_entries(line: str) -> list[HandoffEntry]:
    """The entries of one rollout line; [] for anything that is not a conversation message or a tool call."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict) or obj.get("type") != RESPONSE_ITEM:
        return []
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return []
    kind = payload.get("type")
    if kind == MESSAGE:
        return _message_entries(payload)
    if kind in TOOL_CALLS:
        label = _tool_label(kind, payload)
        return [HandoffEntry(role=ROLE_TOOL, text=label)] if label else []
    return []


def _message_entries(payload: dict) -> list[HandoffEntry]:
    """One entry per user or assistant message: its text blocks joined. Developer messages and injected context give []."""
    role = payload.get("role")
    content = payload.get("content")
    if role not in CONVERSATION_ROLES or not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in TEXT_BLOCKS:
            continue
        text = block.get("text")
        if not isinstance(text, str) or text.lstrip().startswith(IMAGE_MARKERS) or is_rules_block(text):
            continue
        texts.append(text.strip())
    text = "\n".join(part for part in texts if part)
    if not text or (role == "user" and text.startswith(INJECTED_PREFIXES)):
        return []
    return [HandoffEntry(role=role, text=text)]


def _tool_label(kind: str, payload: dict) -> str:
    """'$ <command>' for a shell call, else the call's name and a short detail. '' when there is no name."""
    if kind == "local_shell_call":
        name, detail = SHELL_TOOL, payload.get("action")
    else:
        name = payload.get("name")
        detail = payload.get("arguments", payload.get("input"))
    if not isinstance(name, str) or not name:
        return ""
    command = _command_of(detail)
    if command:
        return _brief("$ " + command)
    return _brief(f"{name} {_detail_text(detail)}")


def _command_of(detail: Any) -> str | None:
    """The shell command a call runs, from a dict (JSON arguments, local_shell action) or a code-mode exec input."""
    if isinstance(detail, dict):
        return _join_command(detail.get("command", detail.get("cmd")))
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return _join_command(parsed.get("command", parsed.get("cmd")))
        match = COMMAND_FIELD.search(detail)
        if match:
            return _join_command(_unescape(match.group(1)))
    return None


def _join_command(value: Any) -> str | None:
    """A command given as a string or as an argv list, as one stripped string; None when it is empty."""
    if isinstance(value, list):
        value = " ".join(item for item in value if isinstance(item, str))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _unescape(raw: str) -> str:
    try:
        return json.loads('"' + raw + '"')
    except ValueError:
        return raw


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, (dict, list)):
        return json.dumps(detail, ensure_ascii=False)
    return ""


def _brief(text: str) -> str:
    """One line, whitespace collapsed, cut to BRIEF_CUT characters with an ellipsis."""
    flat = " ".join(text.split())
    return flat if len(flat) <= BRIEF_CUT else flat[:BRIEF_CUT - 1] + "…"
