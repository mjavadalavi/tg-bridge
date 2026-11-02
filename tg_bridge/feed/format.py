"""Feed text: number, path, time and header formatting (pure functions, no I/O).

Port of the formatting half of the live file's C12 (_fmt_k, _rel, _mmss, _turn_footer, _final_header, the running
header of Turn._header) and of C13 _short. The inputs are the normalized events of agents/events.py, so the same
footer comes out for either backend. Two deliberate differences from the live file:

- the cost is '—' when the backend did not report one (UsageEvent.cost_usd is None); a reported cost that is not
  known to be exact gets a '*', as before;
- stream cut-offs get their own header text, '📡 stream cut, retrying n'.

Token counts: UsageEvent.input_tokens is the fresh input and cached_tokens the cache reads and writes; the footer
shows their sum as '↑' (the live file's footer did the same with the three Claude usage fields).
"""

import os
import re

from tg_bridge.agents.events import (
    AgentEvent,
    StatusEvent,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnResult,
    TurnStatus,
    UsageEvent,
)

SEP = " · "
UNKNOWN = "?"
WAITING_TEXT = "⚙️ waiting for the model…"
NO_COST = "—"
OUTCOME_OK = "ok"
OUTCOME_STOPPED = "stopped"
OUTCOME_CAPPED = "capped"
OUTCOME_INTERRUPTED = "interrupted"
LINE_LABEL_CHARS = 200  # a tool label cut in the readable feed
LINE_ERROR_CHARS = 200  # a tool failure or a turn error cut in the readable feed
REPLY_INDENT = "  ↳ "  # a message from a subagent (the event has a parent id)
NO_DETAIL = "tool failed"
# Terminal escapes and control characters that must not reach the viewer's terminal: CSI (ESC [ ... final, or its C1
# form), OSC (ESC ] ... ended by BEL, ESC \, or the end of the text), other ESC sequences and a lone ESC, then the C0
# and C1 control characters. Tab and newline are kept. Limit: the payload of DCS, APC, PM and SOS strings and of the
# C1 OSC form stays as visible text (it cannot drive the terminal), and a stray ESC also takes the letter after it.
TERMINAL_JUNK = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\?|\Z)"
    r"|\x1b[ -/]*[0-~]?"
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]"
)


def fmt_k(n: int) -> str:
    """1234 -> '1.2k'; numbers under 1000 stay as they are."""
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def mmss(secs: float) -> str:
    """Elapsed seconds as m:ss."""
    secs = int(secs)
    return f"{secs // 60}:{secs % 60:02d}"


def short(text: str, n: int) -> str:
    """The text cut to n characters, the last one replaced by an ellipsis when it is cut."""
    return text if len(text) <= n else text[: n - 1] + "…"


def rel(path: str, cwd: str) -> str:
    """The path relative to the session's cwd when it lies inside it, else the path as given."""
    if path and cwd and os.path.isabs(path):
        rel_path = os.path.relpath(path, cwd)
        if rel_path != ".." and not rel_path.startswith(".." + os.sep):
            return rel_path
    return path or ""


def status_text(ev: StatusEvent) -> str | None:
    """The header note for a status event, or None when the event does not change the header."""
    if ev.kind == "compacting":
        return "🗜 compacting"
    if ev.kind == "retry":
        attempt = UNKNOWN if ev.attempt is None else str(ev.attempt)
        if ev.max_attempts is None:
            return f"🔁 API retry {attempt}"
        return f"🔁 API retry {attempt}/{ev.max_attempts}"
    if ev.kind == "stream_cut":
        attempt = UNKNOWN if ev.attempt is None else str(ev.attempt)
        return f"📡 stream cut, retrying {attempt}"
    return None


def running_header(tools: int, elapsed_s: float, idle_s: float, *, ping_seconds: float,
                   parked: bool, state: str | None) -> str:
    """The first line of an open tool-log message: count, elapsed time, idle time (only past the ping
    threshold and not while a permission card is open), then the current state note."""
    parts = [f"⚙️ working · {tools} tools · {mmss(elapsed_s)}"]
    if idle_s >= ping_seconds and not parked:
        parts.append(f"⏳ idle {int(idle_s // 60)} min")
    if state:
        parts.append(state)
    return SEP.join(parts)


def waiting_header(state: str | None) -> str:
    """The first line of a tool-log message that has no tool yet: the turn is waiting for the model, with the current
    state note (API retry, stream cut, compacting) after it when there is one."""
    return SEP.join([WAITING_TEXT] + ([state] if state else []))


def footer(result: TurnResult, usage: UsageEvent | None, tools: int, wall: float) -> str:
    """The summary of a turn that answered: tool calls, duration, tokens in and out, cost, turns."""
    usage = usage or UsageEvent()
    secs = int(result.duration_ms // 1000) if isinstance(result.duration_ms, (int, float)) else int(wall)
    dur = f"{secs // 60}m{secs % 60:02d}s" if secs >= 60 else f"{secs}s"
    up = usage.input_tokens + usage.cached_tokens
    down = usage.output_tokens
    if usage.cost_usd is None:
        cost = NO_COST
    else:
        cost = f"${usage.cost_usd:.2f}" + ("" if usage.cost_known else "*")
    return SEP.join([f"✅ {tools} tools", dur, f"↑{fmt_k(up)} ↓{fmt_k(down)}", cost,
                     f"{int(result.num_turns or 0)} turns"])


def final_header(outcome: str, *, result: TurnResult | None, usage: UsageEvent | None,
                 tools: int, wall: float, limit_hit: float = 0) -> str:
    """The last header of a turn's tool log: the footer when it answered, else why the turn ended.

    outcome is one of 'ok', 'stopped', 'capped', 'interrupted'; anything else reads as a failed turn.
    """
    if outcome == OUTCOME_OK and result is not None:
        return footer(result, usage, tools, wall)
    if outcome == OUTCOME_STOPPED:
        return f"⏹ stopped after {tools} tools — session kept"
    if outcome == OUTCOME_CAPPED:
        return f"⏱ stopped after {limit_hit}s of work · {tools} tools"
    if outcome == OUTCOME_INTERRUPTED:
        return f"⏹ interrupted by a bridge restart · {tools} tools"
    return f"❌ turn failed · {tools} tools"


def strip_terminal_junk(text: str) -> str:
    """The text without terminal escape sequences and control characters (tab and newline stay)."""
    return TERMINAL_JUNK.sub("", text)


def plain_line(event: AgentEvent, cwd: str) -> str | None:
    """One line of the readable terminal feed for an event, or None when the feed does not show it.

    Shown: the text of an assistant message (indented when a subagent wrote it), a tool start, a failed tool end, the
    result of the turn, the token usage and a status note. cwd is the session's working folder; the labels the
    backends send are already one line, so no event needs it yet. Every piece of text is passed through
    strip_terminal_junk first, so the agent's output cannot send escape codes to the viewer's terminal.
    """
    if isinstance(event, TextEvent):
        text = strip_terminal_junk(event.text).strip()
        if not text:
            return None
        return (REPLY_INDENT if event.parent_id else "") + text
    if isinstance(event, ToolStart):
        prefix = REPLY_INDENT if event.parent_id else ""
        return f"{prefix}🔧 {short(strip_terminal_junk(event.label), LINE_LABEL_CHARS)}"
    if isinstance(event, ToolEnd):
        if event.ok:
            return None
        return f"❌ {short(strip_terminal_junk(event.detail or NO_DETAIL), LINE_ERROR_CHARS)}"
    if isinstance(event, TurnResult):
        return _result_line(event)
    if isinstance(event, UsageEvent):
        up = event.input_tokens + event.cached_tokens
        return f"🔢 tokens {fmt_k(up)} in · {fmt_k(event.output_tokens)} out"
    if isinstance(event, StatusEvent):
        return status_text(event)
    return None


def _result_line(result: TurnResult) -> str:
    """The turn's last line: done, interrupted or the error, with the duration when the backend gave one."""
    if result.status is TurnStatus.OK:
        line = "✅ done"
    elif result.status is TurnStatus.INTERRUPTED:
        line = "⏹ interrupted"
    else:
        line = f"⚠ error: {short(strip_terminal_junk(result.error or result.subtype or 'unknown'), LINE_ERROR_CHARS)}"
    if isinstance(result.duration_ms, (int, float)):
        line = SEP.join([line, mmss(result.duration_ms / 1000)])
    return line
