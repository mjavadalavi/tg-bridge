"""ClaudeTranslator: one decoded stream-json object -> zero or more AgentEvents (the Claude anti-corruption layer).

Only this module reads Claude's raw JSON shapes (PROTOCOL.md). It is pure: no I/O, no clock, no logging, no state
beyond the cwd used for tool labels. The session handles control_response lines itself (ControlChannel) and never
passes them here. An object this module does not know yields [], and for a control_request that means the session
answers it as unhandled, so the CLI never waits for a reply that will not come.

Mapping:
  system/init                  -> SessionStarted (slash_commands, permissionMode as mode)
  system/api_retry             -> StatusEvent('retry', attempt, max_attempts=max_retries)
  system/status compacting     -> StatusEvent('compacting')
  other system subtypes        -> []   (thinking_tokens, hook_*, session_title_changed)
  assistant text               -> TextEvent (parent_id = parent_tool_use_id; subagent text is never the answer)
  assistant tool_use           -> ToolStart (label from tools.tool_label)
  assistant thinking           -> []
  user tool_result             -> ToolEnd
  result                       -> UsageEvent + TurnResult; [] when resume_reason is set (replay of an old turn)
  control_request can_use_tool -> ApprovalRequest (AskUserQuestion: kind 'question' with its questions)
  control_cancel_request       -> ApprovalWithdrawn
"""

from typing import Any

from tg_bridge.agents.claude.tools import FILE_TOOLS, tool_label
from tg_bridge.agents.events import (PLAN_CHOICES, PLAN_KIND, ApprovalRequest, ApprovalWithdrawn, AgentEvent,
                                     Decision, Question, SessionStarted, StatusEvent, TextEvent, ToolEnd, ToolStart,
                                     TurnResult, TurnStatus, UsageEvent)

NO_CONVERSATION = "No conversation found"  # the CLI's answer to --resume of an id it never persisted
ASK_USER_QUESTION = "AskUserQuestion"
EXIT_PLAN_MODE = "ExitPlanMode"  # a finished plan: a plan card (PLAN_KIND) with no Always and no Bypass button
RESULT_DETAIL_CHARS = 1500  # error text kept on a failed TurnResult (the live file's cut)
TOOL_ERROR_CHARS = 200      # first line of a failed tool result kept as ToolEnd.detail
ALL_DECISIONS = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})


class ClaudeTranslator:
    """Turns one stream-json object into the AgentEvents it means. cwd only shortens file paths in tool labels."""

    def __init__(self, cwd: str | None = None) -> None:
        self._cwd = cwd

    def translate(self, event: dict) -> list[AgentEvent]:
        kind = event.get("type")
        if kind == "system":
            return self._system(event)
        if kind == "assistant":
            return self._assistant(event)
        if kind == "user":
            return self._user(event)
        if kind == "result":
            return self._result(event)
        if kind == "control_request":
            return self._control_request(event)
        if kind == "control_cancel_request":
            return self._cancel(event)
        return []

    # ---- system ---------------------------------------------------------------------------------------

    def _system(self, event: dict) -> list[AgentEvent]:
        sub = event.get("subtype")
        if sub == "init":
            return self._init(event)
        if sub == "api_retry":
            return [StatusEvent(kind="retry", detail=str(event.get("error") or ""),
                                retry_in=_seconds(event.get("retry_delay_ms")),
                                attempt=_int(event.get("attempt")), max_attempts=_int(event.get("max_retries")))]
        if sub == "status" and event.get("status") == "compacting":
            return [StatusEvent(kind="compacting")]
        return []

    def _init(self, event: dict) -> list[AgentEvent]:
        sid = event.get("session_id")
        if not isinstance(sid, str) or not sid:
            return []
        commands = _command_names(event.get("slash_commands"))
        return [SessionStarted(session_id=sid, model=_text_or_none(event.get("model")),
                               mode=_text_or_none(event.get("permissionMode")),
                               cwd=_text_or_none(event.get("cwd")), commands=commands)]

    # ---- assistant and user content -------------------------------------------------------------------

    def _assistant(self, event: dict) -> list[AgentEvent]:
        parent = _text_or_none(event.get("parent_tool_use_id"))
        out: list[AgentEvent] = []
        for block in _blocks(event):
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    out.append(TextEvent(text=text, parent_id=parent))
            elif btype == "tool_use":
                tool_id, name = block.get("id"), block.get("name")
                if not isinstance(tool_id, str) or not isinstance(name, str):
                    continue
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                out.append(ToolStart(tool_id=tool_id, name=name, label=tool_label(name, inp, self._cwd),
                                     input=inp, parent_id=parent))
        return out

    def _user(self, event: dict) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        for block in _blocks(event):
            if block.get("type") != "tool_result" or not isinstance(block.get("tool_use_id"), str):
                continue
            failed = bool(block.get("is_error"))
            detail = _first_line(_content_text(block.get("content")))[:TOOL_ERROR_CHARS] if failed else ""
            out.append(ToolEnd(tool_id=block["tool_use_id"], ok=not failed, detail=detail))
        return out

    # ---- result ---------------------------------------------------------------------------------------

    def _result(self, event: dict) -> list[AgentEvent]:
        if event.get("resume_reason"):  # replay of an old interrupted turn, logged only
            return []
        result = event.get("result")
        result_text = result if isinstance(result, str) else ""
        errors = [str(e) for e in event.get("errors") or []] if isinstance(event.get("errors"), list) else []
        resume_failed = NO_CONVERSATION in result_text or any(NO_CONVERSATION in e for e in errors)
        if event.get("is_error"):
            status, answer = TurnStatus.ERROR, None
            error = (result_text or "; ".join(errors) or "no detail")[:RESULT_DETAIL_CHARS]
        else:
            status, answer, error = TurnStatus.OK, (result_text or None), None
        turn = TurnResult(status=status, answer=answer, duration_ms=_int(event.get("duration_ms")),
                          num_turns=_int(event.get("num_turns")), subtype=str(event.get("subtype") or ""),
                          error=error, resume_failed=resume_failed)
        return [_usage(event), turn]

    # ---- control --------------------------------------------------------------------------------------

    def _control_request(self, event: dict) -> list[AgentEvent]:
        rid, req = event.get("request_id"), event.get("request")
        if not isinstance(rid, str) or not isinstance(req, dict) or req.get("subtype") != "can_use_tool":
            return []
        tool = req.get("tool_name") if isinstance(req.get("tool_name"), str) else "?"
        inp = req.get("input") if isinstance(req.get("input"), dict) else {}
        summary = _text_or_none(req.get("display_name")) or tool
        if tool == ASK_USER_QUESTION:  # answered through answers=..., so the only button is Skip (deny)
            return [ApprovalRequest(approval_id=rid, kind="question", tool=tool, input=inp, summary=summary,
                                    choices=frozenset({Decision.DENY}), questions=_questions(inp.get("questions")),
                                    raw=event)]
        choices = PLAN_CHOICES if tool == EXIT_PLAN_MODE else ALL_DECISIONS
        return [ApprovalRequest(approval_id=rid, kind=_approval_kind(tool), tool=tool, input=inp, summary=summary,
                                choices=choices, raw=event)]

    def _cancel(self, event: dict) -> list[AgentEvent]:
        rid = event.get("request_id")
        return [ApprovalWithdrawn(approval_id=rid)] if isinstance(rid, str) else []


# ---- helpers (pure) -----------------------------------------------------------------------------------------


def _command_names(raw: Any) -> tuple[str, ...]:
    """The slash command names of an init event: a string is a name, an object contributes its 'name' (the live file
    reads both forms); anything else and blank names are dropped."""
    if not isinstance(raw, list):
        return ()
    names = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("name")
        if isinstance(item, str) and item:
            names.append(item)
    return tuple(names)


def _blocks(event: dict) -> list[dict]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _usage(event: dict) -> UsageEvent:
    """Tokens as the live file stores and shows them: input_tokens is the fresh input (the turns row's in_tok), and
    cached_tokens the cache reads and writes, which the footer adds to it."""
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    model_usage = event.get("modelUsage") if isinstance(event.get("modelUsage"), dict) else {}
    cache_read = _int(usage.get("cache_read_input_tokens")) or 0
    cache_write = _int(usage.get("cache_creation_input_tokens")) or 0
    cost = event.get("total_cost_usd")
    cost_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    unknown = bool(event.get("hasUnknownModelCost")) or any(
        not isinstance(v, dict) or v.get("costUSD") is None for v in model_usage.values())
    reported = bool(usage or model_usage or cost_usd is not None)  # a result that reports nothing knows no cost
    windows = [w for w in (_int(v.get("contextWindow")) for v in model_usage.values() if isinstance(v, dict)) if w]
    return UsageEvent(input_tokens=_int(usage.get("input_tokens")) or 0,
                      output_tokens=_int(usage.get("output_tokens")) or 0, cached_tokens=cache_read + cache_write,
                      cost_usd=cost_usd, cost_known=reported and not unknown,
                      context_window=windows[0] if windows else None)


def _questions(raw: Any) -> tuple[Question, ...]:
    if not isinstance(raw, list):
        return ()
    out = []
    for q in raw:
        if not isinstance(q, dict):
            continue
        options = []
        for opt in q.get("options") or []:
            label = opt.get("label") if isinstance(opt, dict) else opt
            options.append(str(label or "?"))
        out.append(Question(text=str(q.get("question") or ""), options=tuple(options),
                            multi=bool(q.get("multiSelect"))))
    return tuple(out)


def _approval_kind(tool: str) -> str:
    if tool == EXIT_PLAN_MODE:
        return PLAN_KIND
    if tool == "Bash":
        return "command"
    if tool in FILE_TOOLS:
        return "file"
    return "tool"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _seconds(milliseconds: Any) -> float | None:
    if isinstance(milliseconds, (int, float)) and not isinstance(milliseconds, bool):
        return milliseconds / 1000
    return None


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
