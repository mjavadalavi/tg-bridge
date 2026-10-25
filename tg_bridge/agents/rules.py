"""RuleGrammar: `Tool(glob)` rules shared by both backends (rule_matches, compound-command refusal, realpath).

A rule is a tool name ('Bash', 'WebFetch') or a tool name with a glob in parentheses ('Bash(git status *)',
'Edit(/home/x/proj/*)'). The grammar is the one the Claude CLI uses for --allowedTools and for the addRules
suggestions, so the Codex backend reuses it unchanged: its mapper turns a request into (tool, input) and the
same matcher decides. Pure functions and frozen constants only; no I/O and no state.
"""

import fnmatch
import json
import os
from typing import Any

from tg_bridge.agents.events import ApprovalRequest

FILE_TOOLS = ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit")
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")  # an 'Edit' rule covers all of these
# Characters that let a Bash command chain, redirect or substitute: a glob like 'git status *' would also match
# 'git status; curl x', so such commands never match a pattern rule (only a bare tool rule covers them).
COMPOUND_CHARS = ";&|`<>\n\r"


def target_path(tool: str, inp: dict[str, Any]) -> str:
    """The file a file tool touches, '' for every other tool."""
    if tool in FILE_TOOLS:
        return inp.get("file_path") or inp.get("notebook_path") or ""
    return ""


def rule_matches(rule: str, tool: str, inp: dict[str, Any] | None) -> bool:
    """'X' matches tool X ('Edit' also covers Write/MultiEdit/NotebookEdit). 'X(pattern)' adds a glob:
    Bash matches the command (a trailing ' *' also matches the bare prefix, so 'git status *' matches
    'git status'), file tools match the realpath of the path, anything else matches json.dumps(input)."""
    inp = inp or {}
    name, sep, pat = rule.partition("(")
    same_tool = name == tool or (name == "Edit" and tool in EDIT_TOOLS)
    if not sep:
        return same_tool
    if not same_tool or not pat.endswith(")"):
        return False
    pat = pat[:-1]
    if tool == "Bash":
        cmd = inp.get("command") or ""
        if any(ch in cmd for ch in COMPOUND_CHARS) or "$(" in cmd:
            return False
        if fnmatch.fnmatchcase(cmd, pat):
            return True
        return pat.endswith(" *") and cmd == pat[:-2]
    if tool in FILE_TOOLS:
        path = target_path(tool, inp)
        if not os.path.isabs(path):  # a relative path has no fixed location: a pattern rule never matches it
            return False
        return fnmatch.fnmatchcase(os.path.realpath(path), pat)  # realpath: '..' and symlinks cannot slip past
    return fnmatch.fnmatchcase(json.dumps(inp, ensure_ascii=False), pat)


class RuleGrammar:
    """The RuleMatcher half that is the same for every backend: matches(rule, request, cwd).

    Stateless; the cwd argument is part of the protocol and is not needed by this grammar (file rules are absolute).
    """

    def matches(self, rule: str, req: ApprovalRequest, cwd: str) -> bool:
        return rule_matches(rule, req.tool, req.input)
