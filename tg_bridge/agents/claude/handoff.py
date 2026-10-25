"""Claude transcript -> HandoffEntry list, the source of a cross-backend handoff (DESIGN sections 3-4).

A finished Claude session leaves a transcript at <config_dir>/projects/<encoded cwd>/<sid>.jsonl. read_entries keeps
what a new agent needs to pick the conversation up: the user's text, the assistant's text, and one line per tool call
(its label from tools.py). It skips thinking blocks, tool results, meta and system lines, and every line that is not a
JSON object. Nothing is copied or logged here, and the function never raises for a bad or missing file.
"""

import json

from tg_bridge.agents.claude.tools import tool_label
from tg_bridge.agents.events import HandoffEntry

TRANSCRIPT_TYPES = ("user", "assistant")   # the only top-level line types that carry conversation
TEXT_BLOCK = "text"
TOOL_BLOCK = "tool_use"
ROLE_TOOL = "tool"


def read_entries(path: str | None, cwd: str | None = None) -> tuple[HandoffEntry, ...]:
    """The conversation in transcript order. cwd makes tool labels relative to the session's folder, as the feed does.

    An empty path, a missing or unreadable file, or a file with no conversation gives ().
    """
    if not path:
        return ()
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ()
    entries: list[HandoffEntry] = []
    for line in lines:
        entries.extend(_line_entries(line, cwd))
    return tuple(entries)


def _line_entries(line: str, cwd: str | None) -> list[HandoffEntry]:
    """The entries of one transcript line; [] for anything that is not a conversation message."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict) or obj.get("type") not in TRANSCRIPT_TYPES or obj.get("isMeta"):
        return []
    role = obj["type"]
    msg = obj.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        text = content.strip()
        return [HandoffEntry(role=role, text=text)] if text else []
    if not isinstance(content, list):
        return []
    out: list[HandoffEntry] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        entry = _block_entry(block, role, cwd)
        if entry is not None:
            out.append(entry)
    return out


def _block_entry(block: dict, role: str, cwd: str | None) -> HandoffEntry | None:
    """One content block as an entry. Thinking, tool results and other block types give None."""
    kind = block.get("type")
    if kind == TEXT_BLOCK:
        text = block.get("text")
        text = text.strip() if isinstance(text, str) else ""
        return HandoffEntry(role=role, text=text) if text else None
    if kind == TOOL_BLOCK:
        name = block.get("name")
        if not isinstance(name, str) or not name:
            return None
        label = tool_label(name, block.get("input"), cwd)
        return HandoffEntry(role=ROLE_TOOL, text=label) if label else None
    return None
