"""ClaudeRuleMapper: the 'always' rule and its session permission update from the CLI's permission_suggestions.

Port of the live `_always_rule`. The first addRules suggestion is used as the CLI sent it, with its destination
moved to session. Without one the rule is synthesized: Bash -> first two words + ' *', edit tools -> Edit, anything
else -> the tool name. Pure: the request payload is read from ApprovalRequest.raw (the whole control_request event).
"""

from typing import Any

from tg_bridge.agents.events import ApprovalRequest, Decision
from tg_bridge.agents.rules import EDIT_TOOLS


def always_rule(tool: str, inp: dict[str, Any], suggestions: Any) -> tuple[str, dict[str, Any]]:
    """(rule text, permission update) behind the Always button, for one approval request."""
    for s in suggestions if isinstance(suggestions, list) else []:
        rules = s.get("rules") if isinstance(s, dict) else None
        if isinstance(s, dict) and s.get("type") == "addRules" and isinstance(rules, list) and rules:
            r = rules[0] if isinstance(rules[0], dict) else {}
            name = r.get("toolName") or tool
            content = r.get("ruleContent")
            return (f"{name}({content})" if content else name), dict(s, destination="session")
    if tool == "Bash":
        words = (inp.get("command") or "").split()[:2]
        name, content = "Bash", (" ".join(words) + " *") if words else None
    elif tool in EDIT_TOOLS:
        name, content = "Edit", None
    else:
        name, content = tool, None
    rule = f"{name}({content})" if content else name
    entry = {"toolName": name, "ruleContent": content} if content else {"toolName": name}
    return rule, {"type": "addRules", "rules": [entry], "behavior": "allow", "destination": "session"}


class ClaudeRuleMapper:
    """Implements the per-backend half of RuleMatcher for Claude: rule_for(req) and the updatedPermissions list."""

    def rule_for(self, req: ApprovalRequest) -> str | None:
        if Decision.ALLOW_ALWAYS not in req.choices:
            return None
        return self._always(req)[0]

    def updated_permissions(self, req: ApprovalRequest) -> list[dict[str, Any]]:
        """The updatedPermissions list for a control_response allow on the Always button (empty when not offered)."""
        if Decision.ALLOW_ALWAYS not in req.choices:
            return []
        return [self._always(req)[1]]

    @staticmethod
    def _always(req: ApprovalRequest) -> tuple[str, dict[str, Any]]:
        payload = req.raw.get("request") if isinstance(req.raw, dict) else None
        suggestions = payload.get("permission_suggestions") if isinstance(payload, dict) else None
        return always_rule(req.tool, req.input or {}, suggestions)
