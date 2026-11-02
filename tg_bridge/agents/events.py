"""Normalized agent events (DESIGN section 3): the anti-corruption layer between raw CLI output and the bridge.

Only the backend translators (agents/claude/translate.py, agents/codex/translate.py) build these objects; every
consumer (feed, permissions, usage, session recorder, answer collector) reads nothing else. All dataclasses are
frozen, so an event can be handed to several listeners without any of them changing it.
"""

from dataclasses import dataclass, field
from enum import Enum


class TurnStatus(Enum):
    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class SessionStarted:
    """Claude: system/init. Codex: the thread/start or thread/resume response."""

    session_id: str
    model: str | None = None
    mode: str | None = None          # neutral mode name (see base.ModeChoice)
    cwd: str | None = None
    commands: tuple[str, ...] = ()   # Claude slash_commands; Codex ()
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class TextEvent:
    """One whole assistant message. The translator aggregates deltas before emitting it."""

    text: str
    final: bool = False              # Codex phase == final_answer; Claude: unknown until TurnResult -> False
    item_id: str | None = None
    parent_id: str | None = None     # Claude subagent text (parent_tool_use_id); never collected as the answer


@dataclass(frozen=True, slots=True)
class ToolStart:
    tool_id: str
    name: str                        # backend tool name (Bash, Edit, commandExecution, fileChange, mcp:<server>.<tool>)
    label: str                       # one-line label already formatted by the backend
    input: dict = field(default_factory=dict)
    parent_id: str | None = None     # Claude subagent tool call: the feed indents it under its parent and counts it


@dataclass(frozen=True, slots=True)
class ToolEnd:
    tool_id: str
    ok: bool
    detail: str = ""                 # short error text / exit code when not ok


@dataclass(frozen=True, slots=True)
class StatusEvent:
    kind: str                        # 'compacting' | 'retry' | 'stream_cut' | 'warning' | 'subagent'
    detail: str = ""
    retry_in: float | None = None
    attempt: int | None = None       # Claude api_retry attempt; Codex: counted by the translator
    max_attempts: int | None = None  # Claude api_retry max_retries; Codex None


@dataclass(frozen=True, slots=True)
class HandoffEntry:
    """Not an AgentEvent: read from a finished session's transcript or rollout for a cross-backend handoff."""

    role: str                        # 'user' | 'assistant' | 'tool'
    text: str                        # tool: the one-line label


class Decision(Enum):
    ALLOW_ONCE = "once"
    ALLOW_ALWAYS = "always"
    ALLOW_TURN = "turn"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    options: tuple[str, ...]
    multi: bool = False


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    approval_id: str                 # bridge-unique key, stored in asks.request_id (Claude request_id; Codex 'rpc:<id>')
    kind: str                        # 'tool' | 'command' | 'file' | 'question' | 'permissions' | 'elicitation'
    tool: str                        # tool name for rules and labels
    input: dict                      # what the card shows (command, path, diff summary)
    summary: str                     # one-line card title
    choices: frozenset[Decision]     # buttons to show (the card hides missing ones)
    always_hint: str | None = None   # human text for the always button ('Bash(git status *)' for both backends)
    questions: tuple[Question, ...] = ()
    deny_aborts_turn: bool = False   # Codex 'cancel' only -> the card warns
    raw: dict = field(default_factory=dict, compare=False)  # backend payload kept for answer(); never logged


@dataclass(frozen=True, slots=True)
class ApprovalWithdrawn:
    approval_id: str


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Cumulative for the turn; the last one wins. input_tokens is the fresh input as the backend reports it (the
    turns row stores it as in_tok); cached_tokens is the cache reads and writes, which the footer adds to it."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    cost_known: bool = False
    context_window: int | None = None


@dataclass(frozen=True, slots=True)
class TurnResult:
    status: TurnStatus
    answer: str | None
    duration_ms: int | None = None
    num_turns: int | None = None
    subtype: str = ""                # Claude result.subtype; Codex turn.status
    error: str | None = None
    resume_failed: bool = False      # 'No conversation found' / thread/resume error


@dataclass(frozen=True, slots=True)
class ProcessExited:
    returncode: int | None
    stderr_tail: str = ""


@dataclass(frozen=True, slots=True)
class BackgroundWork:
    """Claude only: the number of background tasks and scheduled wakeups the session is waiting for. Emitted on every
    change, from the session's own threads. Never shown by the feed; the SessionRouter keeps the latest count."""

    pending: int


@dataclass(frozen=True, slots=True)
class TurnBegan:
    """Claude only: the CLI began a turn by itself (a task notification or a wakeup), with no user message."""

    origin: str = ""                 # 'task-notification' (print) | 'task_notification' (tui) | 'scheduled' | 'human' | '' ; logged only


AgentEvent = (SessionStarted | TextEvent | ToolStart | ToolEnd | StatusEvent
              | ApprovalRequest | ApprovalWithdrawn | UsageEvent | TurnResult | ProcessExited
              | BackgroundWork | TurnBegan)
