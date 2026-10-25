"""Codex approvals (DESIGN section 4): server requests <-> ApprovalRequest/Decision, and the mode -> policy table.

Pure: no I/O, no clock, no state. It owns four things:
- which server requests are approvals, and how each one becomes an ApprovalRequest (command, file change,
  permissions, MCP elicitation). Choices come from the item's availableDecisions, or the SCHEMA default
  accept/decline/cancel when the field is absent (file requests never carry it);
- the result object answered for each Decision (accept, decline, cancel, a granted permission profile, an
  elicitation action), kept in one place so the session only calls respond();
- serverRequest/resolved -> ApprovalWithdrawn;
- the neutral mode -> approvalPolicy / sandbox table for thread/start, thread/resume and turn/start.
A server request that is not an approval (requestUserInput, dynamic tool calls, auth) is not handled here: the
session answers it with UNSUPPORTED_ERROR_CODE so the turn never waits for an answer that will not come.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tg_bridge.agents.codex.rules import always_rule, prefix_text
from tg_bridge.agents.events import ApprovalRequest, ApprovalWithdrawn, Decision

COMMAND_METHOD = "item/commandExecution/requestApproval"
FILE_METHOD = "item/fileChange/requestApproval"
PERMISSIONS_METHOD = "item/permissions/requestApproval"
ELICITATION_METHOD = "mcpServer/elicitation/request"
RESOLVED_METHOD = "serverRequest/resolved"
APPROVAL_METHODS = frozenset({COMMAND_METHOD, FILE_METHOD, PERMISSIONS_METHOD, ELICITATION_METHOD})
UNSUPPORTED_ERROR_CODE = -32601
UNSUPPORTED_MESSAGE = "unsupported by tg-bridge"
SCHEMA_DECISIONS = frozenset({"accept", "decline", "cancel"})   # the default when availableDecisions is absent
APPROVALS_REVIEWER = "user"          # always sent: a user config with auto_review would answer approvals itself
GRANTABLE_PROFILE_KEYS = ("network", "fileSystem")
SUMMARY_CHARS = 120


@dataclass(frozen=True, slots=True)
class ModePolicy:
    approval_policy: str             # approvalPolicy on thread/start|resume and turn/start
    sandbox: str                     # the 'sandbox' name on thread/start|resume
    sandbox_type: str                # the 'type' of the sandboxPolicy object on turn/start


class UnknownMode(ValueError):
    """A stored mode key that the Codex table does not know."""


# Stored mode keys are the same as Claude's; bypassPermissions is the default for new topics on both backends.
MODE_POLICIES: Mapping[str, ModePolicy] = MappingProxyType({
    "default": ModePolicy("untrusted", "workspace-write", "workspaceWrite"),
    "acceptEdits": ModePolicy("on-request", "workspace-write", "workspaceWrite"),
    "bypassPermissions": ModePolicy("never", "danger-full-access", "dangerFullAccess"),
    "plan": ModePolicy("on-request", "read-only", "readOnly"),
})


def policy(mode: str) -> ModePolicy:
    try:
        return MODE_POLICIES[mode]
    except KeyError:
        raise UnknownMode(f"no Codex policy for mode {mode!r}") from None


def sandbox_policy(sandbox_type: str) -> dict[str, Any]:
    """A fresh sandboxPolicy object for turn/start (the field has more keys than the thread-level name)."""
    if sandbox_type == "workspaceWrite":
        return {"type": "workspaceWrite", "writableRoots": [], "networkAccess": False,
                "excludeTmpdirEnvVar": False, "excludeSlashTmp": False}
    if sandbox_type == "readOnly":
        return {"type": "readOnly", "networkAccess": False}
    return {"type": sandbox_type}


def thread_params(mode: str) -> dict[str, Any]:
    """The policy fields of thread/start and thread/resume for a stored mode."""
    chosen = policy(mode)
    return {"approvalPolicy": chosen.approval_policy, "sandbox": chosen.sandbox,
            "approvalsReviewer": APPROVALS_REVIEWER}


def turn_params(mode: str) -> dict[str, Any]:
    """The policy fields of a turn/start override for a stored mode."""
    chosen = policy(mode)
    return {"approvalPolicy": chosen.approval_policy, "sandboxPolicy": sandbox_policy(chosen.sandbox_type),
            "approvalsReviewer": APPROVALS_REVIEWER}


def is_approval(method: str) -> bool:
    return method in APPROVAL_METHODS


def rpc_id(req: ApprovalRequest) -> int:
    """The JSON-RPC id of the server request this approval came from (kept in raw, never logged)."""
    return int(req.raw["id"])


def approval_id_for(request_id: int) -> str:
    return f"rpc:{request_id}"


def request_from_frame(method: str, params: Mapping[str, Any], request_id: int, *,
                       changes: Sequence[Mapping[str, Any]] = (), cwd: str = "") -> ApprovalRequest | None:
    """ApprovalRequest for an approval server request, or None when the method is not one of ours.

    `changes` is the `changes` list of the fileChange item this file approval refers to (the request itself carries
    no paths); `cwd` is the thread cwd, used to make relative paths absolute.
    """
    if not is_approval(method) or isinstance(request_id, bool) or not isinstance(request_id, int):
        return None
    body = dict(params) if isinstance(params, Mapping) else {}
    raw = {"id": request_id, "method": method, "params": body}
    if method == COMMAND_METHOD:
        return _command(request_id, body, cwd, raw)
    if method == FILE_METHOD:
        return _file(request_id, body, changes, cwd, raw)
    if method == PERMISSIONS_METHOD:
        return _permissions(request_id, body, cwd, raw)
    return _elicitation(request_id, body, raw)


def withdrawn(params: Mapping[str, Any]) -> ApprovalWithdrawn | None:
    """ApprovalWithdrawn for a serverRequest/resolved notification (the request is answered elsewhere or gone)."""
    request_id = params.get("requestId") if isinstance(params, Mapping) else None
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        return None
    return ApprovalWithdrawn(approval_id=approval_id_for(request_id))


def result_for(req: ApprovalRequest, decision: Decision) -> dict[str, Any]:
    """The `result` object to answer the server request with. Raises ValueError for a choice that was not offered."""
    if decision not in req.choices:
        raise ValueError(f"{decision.name} is not offered for this {req.kind} approval")
    if req.kind == "permissions":
        granted = {} if decision is Decision.DENY else _granted(req.input.get("permissions"))
        return {"permissions": granted, "scope": "turn"}
    if req.kind == "elicitation":
        if decision is Decision.DENY:
            return {"action": "decline", "content": None}
        return {"action": "accept", "content": {}}
    if decision is Decision.DENY:
        return {"decision": "cancel" if req.deny_aborts_turn else "decline"}
    # ONCE, TURN and ALWAYS are all a plain accept: the bridge keeps the rule and the turn-wide auto-accept itself
    return {"decision": "accept"}


def _granted(requested: Any) -> dict[str, Any]:
    if not isinstance(requested, Mapping):
        return {}
    return {key: requested[key] for key in GRANTABLE_PROFILE_KEYS if requested.get(key) is not None}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _offered(raw: Any) -> frozenset[str]:
    """Decision names Codex offers. Entries are plain strings or one-key objects (acceptWithExecpolicyAmendment)."""
    if not isinstance(raw, list):
        return SCHEMA_DECISIONS
    names: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, Mapping) and len(item) == 1:
            names.add(next(iter(item)))
    return frozenset(names)


def _choices(offered: frozenset[str], can_always: bool) -> frozenset[Decision]:
    # DENY is always offered: 'decline' when Codex offers it, else 'cancel', which aborts the turn
    choices = {Decision.DENY}
    if "accept" in offered:
        choices |= {Decision.ALLOW_ONCE, Decision.ALLOW_TURN}
        if can_always:
            choices.add(Decision.ALLOW_ALWAYS)
    return frozenset(choices)


def _short(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= SUMMARY_CHARS else first[:SUMMARY_CHARS - 1] + "…"


def _command(request_id: int, params: dict[str, Any], cwd: str, raw: dict[str, Any]) -> ApprovalRequest:
    command = _text(params.get("command"))
    inp = {"command": command, "cwd": _text(params.get("cwd")) or cwd,
           "prefix": prefix_text(params.get("proposedExecpolicyAmendment"))}
    return _decided(request_id, "command", "commandExecution", inp, f"$ {_short(command)}",
                    _offered(params.get("availableDecisions")), raw)


def _file(request_id: int, params: dict[str, Any], changes: Sequence[Mapping[str, Any]], cwd: str,
          raw: dict[str, Any]) -> ApprovalRequest:
    paths = [p for p in (_absolute(c.get("path"), cwd) for c in changes if isinstance(c, Mapping)) if p]
    inp = {"paths": paths, "reason": _text(params.get("reason")), "grantRoot": _text(params.get("grantRoot"))}
    shown = ", ".join(_relative(p, cwd) for p in paths) or "files"
    return _decided(request_id, "file", "fileChange", inp, f"✏️ {shown}",
                    _offered(params.get("availableDecisions")), raw)


def _permissions(request_id: int, params: dict[str, Any], cwd: str, raw: dict[str, Any]) -> ApprovalRequest:
    reason = _text(params.get("reason"))
    requested = params.get("permissions")
    inp = {"permissions": dict(requested) if isinstance(requested, Mapping) else {}, "reason": reason,
           "cwd": _text(params.get("cwd")) or cwd}
    summary = "extra permissions" + (f": {_short(reason)}" if reason else "")
    return ApprovalRequest(approval_id=approval_id_for(request_id), kind="permissions", tool="permissions",
                           input=inp, summary=summary, choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}),
                           raw=raw)


def _elicitation(request_id: int, params: dict[str, Any], raw: dict[str, Any]) -> ApprovalRequest:
    server = _text(params.get("serverName"))
    message = _text(params.get("message"))
    inp = {"server": server, "message": message, "mode": _text(params.get("mode"))}
    return ApprovalRequest(approval_id=approval_id_for(request_id), kind="elicitation", tool="mcpElicitation",
                           input=inp, summary=f"MCP {server or 'server'} asks: {_short(message)}",
                           choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}), raw=raw)


def _decided(request_id: int, kind: str, tool: str, inp: dict[str, Any], summary: str,
             offered: frozenset[str], raw: dict[str, Any]) -> ApprovalRequest:
    rule = always_rule(kind, inp)
    choices = _choices(offered, can_always=rule is not None)
    return ApprovalRequest(approval_id=approval_id_for(request_id), kind=kind, tool=tool, input=inp,
                           summary=summary, choices=choices,
                           always_hint=rule if Decision.ALLOW_ALWAYS in choices else None,
                           deny_aborts_turn="decline" not in offered, raw=raw)


def _absolute(path: Any, cwd: str) -> str | None:
    """Absolute, realpath-normalised path (what file rules are matched against). None for an empty path."""
    if not isinstance(path, str) or not path:
        return None
    joined = os.path.join(cwd, path) if cwd else path
    return os.path.realpath(joined) if os.path.isabs(joined) else joined


def _relative(path: str, cwd: str) -> str:
    prefix = cwd.rstrip("/") + "/" if cwd else ""
    return path[len(prefix):] if prefix and path.startswith(prefix) else path
