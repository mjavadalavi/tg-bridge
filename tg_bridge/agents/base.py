"""Backend interface (DESIGN section 3): AgentBackend / AgentSession ABCs and their value objects.

A backend adapter (agents/claude/*, agents/codex/*) implements these. The core never imports an adapter: it asks
the BackendRegistry (agents/registry.py) for the backend a LauncherSpec names, opens a session with a SessionSpec,
and receives AgentEvents through the EventSink. The sink runs on the session's reader thread and must not block.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tg_bridge.agents.events import AgentEvent, ApprovalRequest, Decision, HandoffEntry
from tg_bridge.config import DEFAULT_TUI_SIZE, Config, LauncherSpec

EventSink = Callable[[AgentEvent], None]   # called on the session reader thread; must not block


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Only the flags that differ between the two backends (both interrupt, resume, approve and remember)."""

    live_mode: bool                  # a mode change applies inside the running turn
    live_model: bool
    questions: bool                  # AskUserQuestion-like cards
    cost_usd: bool
    images: bool                     # native image input (else the '[photo saved for you at <path>]' text)
    preassigned_id: bool             # Claude --session-id; Codex False (the id is learned from thread/start)
    linger: bool = False             # the session may be kept open after a turn while background work is pending
                                     # (needs the BackgroundWork and TurnBegan events)


@dataclass(frozen=True, slots=True)
class ModeChoice:
    key: str                         # stored in sessions.mode: 'default' | 'acceptEdits' | 'bypassPermissions' | 'plan'
    label: str
    note: str = ""                   # e.g. Codex: 'applies from next message'


@dataclass(frozen=True, slots=True)
class DiskSession:
    """One session the CLI left on disk for a folder: its id, a short title and the file's modification time."""

    uuid: str
    title: str
    modified: float                  # seconds since the epoch


@dataclass(frozen=True, slots=True)
class SessionSpec:
    thread_id: int
    chat_id: int
    topic: str
    launcher: LauncherSpec           # config.LauncherSpec(name, backend, command, argv, config_dir)
    cwd: str
    mode: str
    model: str | None
    resume_id: str | None            # existing session/thread id, or None
    new_id: str | None               # pre-assigned uuid when capabilities.preassigned_id
    allow_rules: tuple[str, ...]     # topic allowed_tools + perm_rules; empty in bypass
    deny_tools: tuple[str, ...]      # 'KillShell' + topic deny_tools
    system_rules: str                # formatted RULES text (+ a pending handoff digest on a fresh session)
    env: dict[str, str]
    errfile_dir: str
    view_dir: str = ""              # tui sessions mirror their raw output to <view_dir>/<thread>.pty; '' = none
    tui_size: tuple[int, int] = DEFAULT_TUI_SIZE  # (rows, cols) of the TUI's pseudo-terminal


class RuleMatcher(Protocol):
    """Implemented by agents/rules.py RuleGrammar plus a per-backend mapper."""

    def matches(self, rule: str, req: ApprovalRequest, cwd: str) -> bool: ...

    def rule_for(self, req: ApprovalRequest) -> str | None:
        """Rule text saved on the 'always' button (the same grammar for both backends)."""
        ...


class AgentSession(ABC):
    """One running agent process or connection, bound to one topic turn."""

    session_id: str | None

    @abstractmethod
    def start(self) -> None:
        """Spawn, handshake, open or resume. SessionStarted is emitted once the id is known: the Codex adapter emits it
        here or on its reader thread, the Claude adapter only with the first user message (system/init). Callers must
        not wait for it before send_user()."""

    @abstractmethod
    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None: ...

    @abstractmethod
    def answer(self, req: ApprovalRequest, decision: Decision, *,
               updated_input: dict | None = None, answers: dict | None = None,
               deny_message: str | None = None) -> None:
        """Answer an approval. deny_message is the reason a Deny carries to the model when the bridge (not the owner)
        refused the request; None means the owner's own Deny. Backends without a deny text ignore it."""

    @abstractmethod
    def interrupt(self) -> bool:
        """False if unsupported; the caller then escalates to stop()."""

    @abstractmethod
    def set_mode(self, mode: str) -> bool:
        """False means not live: the mode applies from the next turn."""

    @abstractmethod
    def set_model(self, model: str) -> bool: ...

    @abstractmethod
    def end_gracefully(self, timeout: float) -> bool:
        """Close the input so the CLI finishes by itself, with no interrupt, and wait up to timeout seconds for it to
        exit. True when it has exited. The turn runner calls this after a normal end, before stop()."""

    @abstractmethod
    def stop(self, grace: float = 5.0) -> int | None:
        """interrupt, then EOF, then the SIGINT/SIGTERM/SIGKILL ladder, then a process-group sweep. Returns the process's
        return code, or None when there is no process (or it has no code yet)."""

    @abstractmethod
    def alive(self) -> bool: ...

    @abstractmethod
    def stderr_tail(self, n: int = 20) -> str: ...


class AgentBackend(ABC):
    """Factory and adapter for one CLI. Stateless apart from its own configuration."""

    name: str                        # 'claude' | 'codex'
    capabilities: Capabilities
    rules: RuleMatcher

    @abstractmethod
    def modes(self) -> tuple[ModeChoice, ...]: ...

    @abstractmethod
    def models(self, cfg: Config) -> tuple[str, ...]:
        """config 'models'[backend] when set, else the built-in list."""

    @abstractmethod
    def open_session(self, spec: SessionSpec, sink: EventSink) -> AgentSession: ...

    def capabilities_for(self, spec: LauncherSpec) -> Capabilities:
        """The flags of one launcher. They can differ per launcher, e.g. Claude's tui transport has no live mode or
        model switch; the default is the backend's own flags."""
        return self.capabilities

    def one_shot(self, prompt: str, timeout: float) -> str | None:
        """Used for topic titles. The default None means unsupported."""
        return None

    def resume_hint(self, session_id: str, cwd: str) -> str:
        """Text shown by /status."""
        return ""

    def carry_history(self, from_spec: LauncherSpec, to_spec: LauncherSpec, src_cwd: str, sid: str,
                      dst_cwd: str | None = None) -> bool:
        """Same-backend launcher switch. Claude copies the transcript; the default copies nothing."""
        return False

    def handoff_entries(self, session_id: str, cwd: str, spec: LauncherSpec) -> tuple[HandoffEntry, ...]:
        """Cross-backend handoff source. The default () means there is nothing to carry."""
        return ()

    def list_sessions(self, spec: LauncherSpec, cwd: str, limit: int,
                      exclude: frozenset[str] = frozenset()) -> tuple[DiskSession, ...]:
        """The sessions on disk for this folder, newest first, at most limit of them, leaving out the ids in exclude.
        The default () means the backend cannot list them."""
        return ()

    def has_transcript(self, spec: LauncherSpec, cwd: str, sid: str) -> bool:
        """True when this launcher's own folder holds the transcript of session sid for cwd. The default False means the
        backend keeps no transcript a launcher could be asked about."""
        return False
