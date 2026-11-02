"""ClaudeBackend: the Claude adapter of AgentBackend (capabilities, modes, models, sessions, titles, /status hint).

Everything that is specific to the Claude CLI and is not inside a session lives here: the capability flags, the
permission modes the /mode picker offers, the rule matcher (the shared grammar plus the Claude 'always' mapper), and
the one-shot call that titles a topic. Sessions are built by ClaudeSession (print) or ClaudeTuiSession (tui);
this class never parses a stream line.

The one-shot title call runs the launcher named by TG_TITLE_LAUNCHER (a cheap model seat), in its own process group,
and the whole group is killed when the call times out, as the live file does. The argv is passed in by the caller
(agents/defaults.py), so this module reads no environment variable itself.
"""

import os
import signal
import subprocess
from collections.abc import Sequence

from tg_bridge.agents.base import AgentBackend, AgentSession, Capabilities, DiskSession, EventSink, ModeChoice, SessionSpec
from tg_bridge.agents.claude import history, sessions_index
from tg_bridge.agents.claude.handoff import read_entries
from tg_bridge.agents.claude.rules import ClaudeRuleMapper
from tg_bridge.agents.claude.session import ClaudeSession
from tg_bridge.agents.claude.tui_session import ClaudeTuiSession
from tg_bridge.agents.events import ApprovalRequest, HandoffEntry
from tg_bridge.agents.rules import RuleGrammar
from tg_bridge.clock import Clock
from tg_bridge.config import BACKEND_CLAUDE, TRANSPORT_TUI, Config, LauncherSpec
from tg_bridge.logging_ import Logger

CLAUDE_CAPABILITIES = Capabilities(
    live_mode=True,            # set_permission_mode applies inside the running turn
    live_model=True,           # set_model applies inside the running turn
    questions=True,            # AskUserQuestion cards
    cost_usd=True,             # total_cost_usd on the result
    images=False,              # photos reach Claude as a '[photo saved ...]' text line
    preassigned_id=True,       # --session-id lets the bridge store the id before the process starts
    linger=True,               # stream-json reports background work and the self turns it starts (C3b)
)
# The tui transport drives the interactive CLI through a PTY, so some print-only features do not exist there.
CLAUDE_TUI_CAPABILITIES = Capabilities(
    live_mode=False,           # no set_permission_mode in the TUI: the mode applies from the next turn, via the
                               # CLI flag
    live_model=False,          # typing /model in the TUI saves the user's default model, so the model applies from the
                               # next turn, through the CLI flag instead
    questions=False,           # AskUserQuestion is disallowed in tui, so no question cards
    cost_usd=False,            # the transcript carries no total_cost_usd, so there is no cost to show
    images=False,              # photos reach Claude as a '[photo saved ...]' text line, as in print
    preassigned_id=True,       # --session-id works in the TUI, so the id is still stored before the process starts
    linger=True,               # the transcript tail reports background work and the self turns it starts (C4)
)
# Labels are the /mode picker's button texts, in the live file's order (MODE_PICK).
CLAUDE_MODES = (
    ModeChoice(key="bypassPermissions", label="bypass"),
    ModeChoice(key="acceptEdits", label="acceptEdits"),
    ModeChoice(key="default", label="default"),
    ModeChoice(key="plan", label="plan"),
)
TITLE_FLAGS = ("--max-turns", "1", "--no-session-persistence")  # the title call writes no transcript for /resume
TITLE_MAX_CHARS = 60       # the title is cut to this many characters, as the live file does
TITLE_REAP_WAIT = 5.0      # seconds to wait for a killed title process to be reaped


class ClaudeRules:
    """RuleMatcher for Claude: the shared rule grammar for matching, the Claude mapper for the 'always' rule."""

    def __init__(self) -> None:
        self._grammar = RuleGrammar()
        self._mapper = ClaudeRuleMapper()

    def matches(self, rule: str, req: ApprovalRequest, cwd: str) -> bool:
        return self._grammar.matches(rule, req, cwd)

    def rule_for(self, req: ApprovalRequest) -> str | None:
        return self._mapper.rule_for(req)

    def updated_permissions(self, req: ApprovalRequest) -> list[dict]:
        """The updatedPermissions list for an 'always' answer; empty when the request does not offer it."""
        return self._mapper.updated_permissions(req)


class ClaudeBackend(AgentBackend):
    """Stateless apart from its logger, clock and title argv. One instance serves every Claude topic."""

    name = BACKEND_CLAUDE
    capabilities = CLAUDE_CAPABILITIES

    def __init__(self, *, logger: Logger, clock: Clock, title_argv: Sequence[str] = ()) -> None:
        self._logger = logger
        self._clock = clock
        self._title_argv = tuple(title_argv)   # empty: no auto-titles
        self.rules = ClaudeRules()

    def capabilities_for(self, spec: LauncherSpec) -> Capabilities:
        """The tui transport has its own flags; every print launcher has the print flags."""
        return CLAUDE_TUI_CAPABILITIES if spec.transport == TRANSPORT_TUI else CLAUDE_CAPABILITIES

    def modes(self) -> tuple[ModeChoice, ...]:
        return CLAUDE_MODES

    def models(self, cfg: Config) -> tuple[str, ...]:
        """The config 'models' list, else the built-in choices (Config.models does both)."""
        return cfg.models()

    def open_session(self, spec: SessionSpec, sink: EventSink) -> AgentSession:
        """The tui transport runs the interactive TUI through a PTY; every other launcher runs the print session."""
        if spec.launcher.transport == TRANSPORT_TUI:
            return ClaudeTuiSession(spec, sink, logger=self._logger, clock=self._clock)
        return ClaudeSession(spec, sink, logger=self._logger, clock=self._clock)

    def one_shot(self, prompt: str, timeout: float) -> str | None:
        """Run the title launcher once with the prompt and return one cleaned line, or None on any failure.

        The process gets its own group. On timeout the whole group is killed, so nothing it started survives.
        """
        if not self._title_argv:
            return None
        argv = [*self._title_argv, "-p", prompt, *TITLE_FLAGS]
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                    start_new_session=True)
        except OSError as exc:
            self._logger.log(f"title: could not start the title launcher: {exc!r}")
            return None
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid)
            _reap(proc)
            self._logger.log("title: timed out, the title process group was killed")
            return None
        title = " ".join((out or "").split())[:TITLE_MAX_CHARS].strip()
        return title or None

    def resume_hint(self, session_id: str, cwd: str) -> str:
        return f"claude --resume {session_id}"

    def carry_history(self, from_spec: LauncherSpec, to_spec: LauncherSpec, src_cwd: str, sid: str,
                      dst_cwd: str | None = None) -> bool:
        """Copy the transcript from one Claude launcher's config folder into the other's (thinking stripped)."""
        return history.carry_history(from_spec.config_dir, to_spec.config_dir, src_cwd, sid, dst_cwd, self._logger)

    def handoff_entries(self, session_id: str, cwd: str, spec: LauncherSpec) -> tuple[HandoffEntry, ...]:
        """The conversation of a finished Claude session, read from its transcript (empty when there is none)."""
        path = history.transcript_path(spec.config_dir, cwd, session_id)
        return read_entries(path, cwd)

    def list_sessions(self, spec: LauncherSpec, cwd: str, limit: int,
                      exclude: frozenset[str] = frozenset()) -> tuple[DiskSession, ...]:
        """The Claude sessions on disk for this cwd under the launcher's config folder, newest first, without the ids in
        exclude."""
        return sessions_index.list_sessions(spec.config_dir, cwd, limit, exclude)

    def has_transcript(self, spec: LauncherSpec, cwd: str, sid: str) -> bool:
        """True when the launcher's config folder holds the transcript of session sid under this cwd's own folder. Other
        folders of the config dir do not count: claude --resume only looks in the cwd's folder."""
        return history.cwd_transcript_path(spec.config_dir, cwd, sid) is not None


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reap(proc: subprocess.Popen[str]) -> None:
    try:
        proc.communicate(timeout=TITLE_REAP_WAIT)
    except Exception:  # reaping is best effort; the group is already dead
        pass
