"""CodexTranslator: one app-server frame -> zero or more AgentEvents (the Codex anti-corruption layer).

Only this module reads Codex's raw JSON-RPC shapes (PROTOCOL-codex.md, DESIGN section 4). It is pure: no I/O, no clock,
no logging. Per turn it keeps only what the mapping needs (buffered deltas, announced tool ids, the last final answer,
the retry count); turn/started clears that state, and one process runs one turn in practice.

Frames it accepts:
  notification  {"method", "params"} (no id)           -> events below
  response      {"id", "result"|"error"} + method=...  -> the caller passes the method of the request it sent
  server request {"id", "method"} (approvals and the rest) -> []: approvals.py owns those
A response is only meaningful with `method`; without it the translator returns [].

Mapping:
  thread/start|resume result     -> SessionStarted (resumed for thread/resume)
  thread/resume error            -> TurnResult(ERROR, resume_failed=True)
  thread/start|turn/start error  -> TurnResult(ERROR, error=message)
  item/agentMessage/delta        -> buffered; nothing emitted
  item/completed agentMessage    -> TextEvent(text, final=phase == final_answer)
  item/started commandExecution  -> ToolStart('$ <command>'); fileChange -> ToolStart('✏️ <paths>') once per item
  item/started mcpToolCall/webSearch -> ToolStart; item/completed of a tool item -> ToolEnd (ok or failure detail)
  thread/tokenUsage/updated      -> UsageEvent for this turn (total minus the baseline frozen at turn/started;
                                    nothing before turn/started, so a resume replay emits nothing; cost unknown)
  error with willRetry true      -> StatusEvent('retry', attempt) or StatusEvent('stream_cut', attempt)
  thread/compacted, contextCompaction item -> StatusEvent('compacting')
  turn/completed                 -> TurnResult (completed|interrupted|failed; answer = last final_answer text)
  everything else (noise, reasoning, plans, unknown items) -> []
"""

from types import MappingProxyType
from typing import Any

from tg_bridge.agents.events import (AgentEvent, SessionStarted, StatusEvent, TextEvent, ToolEnd, ToolStart,
                                     TurnResult, TurnStatus, UsageEvent)

START = "thread/start"
RESUME = "thread/resume"
TURN_START = "turn/start"
FINAL_PHASE = "final_answer"
TOOL_KINDS = frozenset({"commandExecution", "fileChange", "mcpToolCall", "webSearch"})
STREAM_CUT_INFO = frozenset({"responseStreamDisconnected", "responseStreamConnectionFailed"})
TURN_STATUS = MappingProxyType({"completed": TurnStatus.OK, "interrupted": TurnStatus.INTERRUPTED,
                                "failed": TurnStatus.ERROR})
TOOL_ERROR_CHARS = 200  # first line of a failure kept as ToolEnd.detail, as in the Claude translator


class CodexTranslator:
    """Turns one app-server frame into the AgentEvents it means. mode is stamped on SessionStarted, cwd shortens paths."""

    def __init__(self, mode: str | None = None, cwd: str | None = None) -> None:
        self._mode = mode
        self._cwd = cwd
        self._deltas: dict[str, list[str]] = {}  # item id -> agent message deltas not yet completed
        self._announced: set[str] = set()        # tool item ids whose ToolStart was already emitted
        self._final_text: str | None = None      # text of the last final_answer message this turn
        self._attempts = 0                       # model-side retries counted this turn
        self._latest = (0, 0, 0)                 # last tokenUsage.total seen (input, output, cached), any frame
        self._base = (0, 0, 0)                   # the same, frozen at turn/started: the turn's starting point
        self._in_turn = False                    # turn/started seen; usage before it only moves the baseline

    def translate(self, frame: dict, method: str | None = None) -> list[AgentEvent]:
        if "method" in frame:
            if "id" in frame:  # a server request: never ours to answer here
                return []
            params = frame.get("params")
            return self._notification(frame["method"], params if isinstance(params, dict) else {})
        if "id" in frame and method is not None:
            return self._response(method, frame)
        return []

    # ---- responses to our requests --------------------------------------------------------------------

    def _response(self, method: str, frame: dict) -> list[AgentEvent]:
        if "error" in frame:
            message = _error_text(frame.get("error")) or "request failed"
            if method == RESUME:
                return [TurnResult(status=TurnStatus.ERROR, answer=None, error=message, resume_failed=True)]
            if method in (START, TURN_START):
                return [TurnResult(status=TurnStatus.ERROR, answer=None, error=message)]
            return []
        result = frame.get("result")
        if method in (START, RESUME) and isinstance(result, dict):
            thread = result.get("thread")
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id:
                return []
            return [SessionStarted(session_id=thread_id, model=_text(result.get("model")), mode=self._mode,
                                   cwd=_text(result.get("cwd")), resumed=method == RESUME)]
        return []

    # ---- notifications --------------------------------------------------------------------------------

    def _notification(self, method: str, params: dict) -> list[AgentEvent]:
        if method == "turn/started":
            self._reset_turn()
            self._base = self._latest
            self._in_turn = True
            return []
        if method == "item/started":
            return self._item_started(params.get("item"))
        if method == "item/completed":
            return self._item_completed(params.get("item"))
        if method == "item/agentMessage/delta":
            self._buffer(params)
            return []
        if method == "item/fileChange/patchUpdated":
            item = {"id": params.get("itemId"), "type": "fileChange", "changes": params.get("changes")}
            return self._announce_file_change(item)
        if method == "thread/tokenUsage/updated":
            return self._usage(params)
        if method == "error":
            return self._error(params)
        if method == "thread/compacted":
            return [StatusEvent(kind="compacting")]
        if method == "turn/completed":
            return [self._turn_completed(params.get("turn"))]
        return []

    def _reset_turn(self) -> None:
        self._deltas.clear()
        self._announced.clear()
        self._final_text = None
        self._attempts = 0

    def _buffer(self, params: dict) -> None:
        item_id, delta = params.get("itemId"), params.get("delta")
        if isinstance(item_id, str) and isinstance(delta, str):
            self._deltas.setdefault(item_id, []).append(delta)

    # ---- items ----------------------------------------------------------------------------------------

    def _item_started(self, item: Any) -> list[AgentEvent]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return []
        kind = item.get("type")
        if kind == "contextCompaction":
            return [StatusEvent(kind="compacting")]
        if kind == "fileChange":
            return self._announce_file_change(item)
        if kind in TOOL_KINDS:
            return self._announce(item)
        return []

    def _item_completed(self, item: Any) -> list[AgentEvent]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return []
        kind = item.get("type")
        if kind == "agentMessage":
            return self._message_done(item)
        if kind in TOOL_KINDS:
            return self._announce(item) + [_tool_end(item)]
        return []

    def _message_done(self, item: dict) -> list[AgentEvent]:
        item_id = item["id"]
        text = item.get("text") if isinstance(item.get("text"), str) else ""
        text = text or "".join(self._deltas.pop(item_id, []))
        final = item.get("phase") == FINAL_PHASE
        if not text.strip():
            return []
        if final:
            self._final_text = text
        return [TextEvent(text=text, final=final, item_id=item_id)]

    def _announce(self, item: dict) -> list[AgentEvent]:
        """ToolStart for a tool item, once per id. A file change without paths waits for its patchUpdated."""
        item_id = item["id"]
        if item_id in self._announced:
            return []
        self._announced.add(item_id)
        kind = item.get("type")
        if kind == "commandExecution":
            command = _command(item)
            return [ToolStart(tool_id=item_id, name="commandExecution", label="$ " + command,
                              input={"command": command, "cwd": _text(item.get("cwd")) or ""})]
        if kind == "fileChange":
            return [ToolStart(tool_id=item_id, name="fileChange", label=_file_label(item, self._cwd),
                              input={"paths": _paths(item)})]
        if kind == "mcpToolCall":
            server, tool = _text(item.get("server")) or "?", _text(item.get("tool")) or "?"
            return [ToolStart(tool_id=item_id, name=f"mcp:{server}.{tool}", label=f"🔌 {server}.{tool}",
                              input={"server": server, "tool": tool})]
        query = _text(item.get("query")) or ""
        return [ToolStart(tool_id=item_id, name="webSearch", label="🔎 " + query, input={"query": query})]

    def _announce_file_change(self, item: dict) -> list[AgentEvent]:
        """A file change is announced once its paths are known (patchUpdated may carry them after item/started)."""
        if not _paths(item):
            return []
        return self._announce(item)

    # ---- usage, errors, completion --------------------------------------------------------------------

    def _usage(self, params: dict) -> list[AgentEvent]:
        usage = params.get("tokenUsage")
        if not isinstance(usage, dict) or not isinstance(usage.get("total"), dict):
            return []
        total = usage["total"]
        self._latest = (_int(total.get("inputTokens")) or 0, _int(total.get("outputTokens")) or 0,
                        _int(total.get("cachedInputTokens")) or 0)
        if not self._in_turn:  # a resume replays the whole thread's total before turn/started: baseline only
            return []
        input_base, output_base, cached_base = self._base
        input_now, output_now, cached_now = self._latest
        return [UsageEvent(input_tokens=input_now - input_base, output_tokens=output_now - output_base,
                           cached_tokens=cached_now - cached_base, cost_usd=None, cost_known=False,
                           context_window=_int(usage.get("modelContextWindow")))]

    def _error(self, params: dict) -> list[AgentEvent]:
        if params.get("willRetry") is not True:  # a fatal error is reported by turn/completed instead
            return []
        self._attempts += 1
        error = params.get("error")
        detail = _error_text(error)
        kind = "stream_cut" if _error_info(error) in STREAM_CUT_INFO else "retry"
        return [StatusEvent(kind=kind, detail=detail, attempt=self._attempts)]

    def _turn_completed(self, turn: Any) -> TurnResult:
        turn = turn if isinstance(turn, dict) else {}
        status = turn.get("status")
        message = _error_text(turn.get("error")) or None
        if status not in TURN_STATUS and message is None:
            message = f"unexpected turn status: {status}"
        return TurnResult(status=TURN_STATUS.get(status, TurnStatus.ERROR),
                          answer=self._final_text or _final_in_items(turn.get("items")),
                          duration_ms=_int(turn.get("durationMs")), subtype=str(status or ""), error=message)


# ---- helpers (pure) -----------------------------------------------------------------------------------------


def _tool_end(item: dict) -> ToolEnd:
    status, code, error = item.get("status"), item.get("exitCode"), item.get("error")
    failed_exit = _int(code) not in (None, 0)
    if status not in ("failed", "declined") and not failed_exit and not error:
        return ToolEnd(tool_id=item["id"], ok=True)
    message = _error_text(error)
    if message:
        detail = message
    elif failed_exit:
        detail = f"exit {code}"
    else:
        detail = str(status or "failed")
    first_line = (detail.splitlines() or [""])[0]
    return ToolEnd(tool_id=item["id"], ok=False, detail=first_line[:TOOL_ERROR_CHARS])


def _command(item: dict) -> str:
    actions = item.get("commandActions")
    if isinstance(actions, list) and actions and isinstance(actions[0], dict) and _text(actions[0].get("command")):
        return actions[0]["command"]
    return _text(item.get("command")) or ""


def _paths(item: dict) -> list[str]:
    changes = item.get("changes")
    if not isinstance(changes, list):
        return []
    return [p for p in (_text(c.get("path")) for c in changes if isinstance(c, dict)) if p]


def _file_label(item: dict, cwd: str | None) -> str:
    paths = [_relative(p, cwd) for p in _paths(item)]
    return "✏️ " + (", ".join(paths) if paths else "files")


def _relative(path: str, cwd: str | None) -> str:
    prefix = (cwd or "").rstrip("/") + "/"
    return path[len(prefix):] if cwd and path.startswith(prefix) else path


def _final_in_items(items: Any) -> str | None:
    """The final answer from a turn/completed summary (its items hold only the final agentMessage)."""
    if not isinstance(items, list):
        return None
    for item in reversed(items):
        if isinstance(item, dict) and item.get("type") == "agentMessage" and item.get("phase") == FINAL_PHASE:
            text = _text(item.get("text"))
            if text:
                return text
    return None


def _error_info(error: Any) -> str | None:
    """codexErrorInfo is a plain name or a one-key object such as {"responseStreamDisconnected": {...}}."""
    info = error.get("codexErrorInfo") if isinstance(error, dict) else None
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and len(info) == 1:
        return next(iter(info))
    return None


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return _text(error.get("message")) or ""
    return _text(error) or ""


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
