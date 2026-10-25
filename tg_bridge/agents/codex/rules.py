"""CodexRuleMapper: a Codex approval -> (tool, input) for the shared RuleGrammar, and the 'always' rule text.

The 'always' rule uses the same grammar as Claude's saved rules (agents/rules.py), so it keeps working after a launcher
switch and never uses a `cmd:` prefix. A command gets `Bash(<prefix> *)`, where the prefix is the execpolicy amendment
Codex proposes; a single file change gets `Edit(<absolute path>)`. A prefix or path that could match more than it names
(glob characters, parentheses, quotes, compound-command characters) gives no rule, so the Always button is not offered.

Pure functions only: no I/O, no state. The matcher view unwraps the `<shell> -lc '<command>'` wrapper that Codex puts
around every shell command, so a rule saved from `touch a.txt` also matches the wrapped form of the same command.
"""

import re
from typing import Any

from tg_bridge.agents.events import ApprovalRequest, Decision
from tg_bridge.agents.rules import COMPOUND_CHARS, rule_matches

SHELL_WRAPPER = re.compile(r"^(?:\S*/)?(?:zsh|bash|sh) -l?c '([^'\n]*)'$")
PATTERN_UNSAFE = frozenset("*?[]()'\"\\$") | frozenset(COMPOUND_CHARS)  # chars that would widen or break a rule
FILE_TOOL = "Edit"        # the rule name a file change matches under (covers Write/MultiEdit/NotebookEdit too)
BASH_TOOL = "Bash"


def unwrap_shell(command: str) -> str:
    """The inner command of a `zsh -lc '...'` wrapper, else the command unchanged."""
    match = SHELL_WRAPPER.match(command)
    return match.group(1) if match else command


def _safe_text(text: str) -> bool:
    return bool(text.strip()) and not any(ch in PATTERN_UNSAFE for ch in text)


def prefix_text(tokens: Any) -> str | None:
    """The Bash prefix an execpolicy amendment stands for ('touch probe_ok.txt'), or None when it is unusable.

    Each token must be one plain word, so the joined text cannot be widened by a glob or split by a quote.
    """
    if not isinstance(tokens, list) or not tokens:
        return None
    if not all(isinstance(t, str) and t and t.split() == [t] for t in tokens):
        return None
    text = " ".join(tokens)
    return text if _safe_text(text) else None


def always_rule(kind: str, inp: dict[str, Any]) -> str | None:
    """The rule text the Always button saves for one approval, or None when no safe rule exists."""
    if kind == "command":
        prefix = inp.get("prefix")
        return f"{BASH_TOOL}({prefix} *)" if isinstance(prefix, str) and _safe_text(prefix) else None
    if kind == "file":
        paths = inp.get("paths") or []
        if len(paths) != 1 or not isinstance(paths[0], str):
            return None
        path = paths[0]
        if not path.startswith("/") or not _safe_text(path):
            return None
        return f"{FILE_TOOL}({path})"
    return None


class CodexRuleMapper:
    """The Codex half of RuleMatcher: view() feeds RuleGrammar, rule_for() gives the Always text."""

    def view(self, req: ApprovalRequest) -> tuple[str, dict[str, Any]]:
        """The (tool, input) pair that rule_matches() compares a saved rule against."""
        inp = req.input or {}
        if req.kind == "command":
            return BASH_TOOL, {"command": unwrap_shell(str(inp.get("command") or "")), "cwd": inp.get("cwd") or ""}
        if req.kind == "file":
            paths = inp.get("paths") or []
            # several files, or none: no pattern rule can name them, so only a bare 'Edit' rule covers them
            return FILE_TOOL, {"file_path": paths[0] if len(paths) == 1 else ""}
        return req.tool, dict(inp)

    def matches(self, rule: str, req: ApprovalRequest, cwd: str) -> bool:
        tool, inp = self.view(req)
        return rule_matches(rule, tool, inp)

    def rule_for(self, req: ApprovalRequest) -> str | None:
        if Decision.ALLOW_ALWAYS not in req.choices:
            return None
        return always_rule(req.kind, req.input or {})
