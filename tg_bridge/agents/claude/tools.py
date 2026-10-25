"""Claude tool labels: the one-line text of a tool call in the progress feed (port of the live _tool_label).

Pure functions, no I/O. The label is built once, in the translator, so every listener that shows ToolStart.label
prints the same text. The file-tool sets are shared with the permission rules (agents/rules.py) and the approval kind.
"""

import os
import urllib.parse
from typing import Any

from tg_bridge.agents.rules import EDIT_TOOLS, FILE_TOOLS  # shared with the permission rule grammar

BASH_COMMAND_CUT = 60   # a Bash call without a description is labelled with this much of its command
MCP_PREFIX = "mcp__"


def rel_path(path: str, cwd: str | None) -> str:
    """The path relative to the session's cwd when it lies inside it, else the path as given."""
    if path and cwd and os.path.isabs(path):
        rel = os.path.relpath(path, cwd)
        if rel != ".." and not rel.startswith(".." + os.sep):
            return rel
    return path or ""


def tool_label(name: str, inp: Any, cwd: str | None) -> str:
    """The short text of one Claude tool call, e.g. 'Read src/a.py', 'Bash Show status', 'srv:tool'."""
    inp = inp if isinstance(inp, dict) else {}
    if name.startswith(MCP_PREFIX):
        srv, _, tool = name[len(MCP_PREFIX):].partition("__")
        return f"{srv}:{tool}" if tool else name
    if name in FILE_TOOLS:
        path = rel_path(str(inp.get("file_path") or inp.get("notebook_path") or ""), cwd)
        return f"{name} {path}" if path else name
    if name == "Bash":
        text = (" ".join(str(inp.get("description") or "").split())
                or " ".join(str(inp.get("command") or "").split())[:BASH_COMMAND_CUT])
        return f"Bash {text}" if text else "Bash"
    if name == "Grep":
        path = rel_path(str(inp.get("path") or ""), cwd)
        return f'Grep "{inp.get("pattern") or ""}"' + (f" in {path}" if path else "")
    if name == "Glob":
        return f"Glob {inp.get('pattern') or ''}".rstrip()
    if name == "WebFetch":
        url = str(inp.get("url") or "")
        return f"WebFetch {urllib.parse.urlparse(url).netloc or url[:60]}".rstrip()
    if name == "WebSearch":
        return f"WebSearch {inp.get('query') or ''}".rstrip()
    if name in ("Agent", "Task"):
        return f"{name} {inp.get('description') or ''}".rstrip()
    if name == "Skill":
        return f"Skill {inp.get('skill') or ''}".rstrip()
    return name
