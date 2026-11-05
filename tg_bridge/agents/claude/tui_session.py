"""ClaudeTuiSession: one interactive Claude TUI in a pseudo-terminal, bound to one topic turn (the tui transport).

Some services refuse `claude -p` but accept the interactive TUI. The TUI runs on a PTY (PtyProcess) and writes its
transcript while a turn runs, so the turn is read from the transcript by TranscriptTurns (transcript_turns.py): every line
goes to the same ClaudeTranslator as the print transport, and the turn ends at the first system/turn_duration after the
message was typed. An interrupt (Esc) writes none, so the session ends that turn itself. This class owns the PTY, the
permission hook and the view mirror; the turn state is self._turns.

The tail lives as long as the session, not only as long as a turn (see TranscriptTurns). A closing session (_ending)
starts no such turn.

Permissions come over the PermissionRequest hook (HookServer). While a request is pending no key is typed, because a
key would answer the TUI's own dialog, which is drawn at the same time.

build_tui_args() is the TUI argv: the print flags minus the stream-json ones. AskUserQuestion is disallowed, since
answering it through the TUI is not verified (questions=False in CLAUDE_TUI_CAPABILITIES).

With spec.view_dir set, every raw chunk of the TUI output is also written to <view_dir>/<thread>.pty (PtyMirror), which the
`tg-bridge view` windows replay. The PTY's size is spec.tui_size. When the mirror starts a new file, the TUI is asked to
redraw (PtyProcess.redraw), so the new file gets the whole screen and not only the updates after the cut.
"""

import os
import re
import threading

from tg_bridge.agents.base import AgentSession, EventSink, SessionSpec
from tg_bridge.agents.claude.hookbridge import (TURN_ENDED, HookServer, hook_command, hook_settings, permission_allow,
                                                permission_deny)
from tg_bridge.agents.claude.memory_settings import with_memory
from tg_bridge.agents.claude.protocol import DENY_MESSAGE
from tg_bridge.agents.claude.session import BYPASS_MODE, INTERRUPT_TIMEOUT, KILL_SWITCH_TOOL
# INTERRUPT_FALLBACK_SECONDS and TRANSCRIPT_WAIT_SECONDS now live in transcript_turns; they stay importable from here.
from tg_bridge.agents.claude.transcript_turns import (INTERRUPT_FALLBACK_SECONDS, TRANSCRIPT_WAIT_SECONDS,
                                                      TranscriptTurns)
from tg_bridge.agents.claude.tui_helpers import _child_env, _socket_path, permission_args
from tg_bridge.agents.claude.translate import NO_CONVERSATION, ClaudeTranslator
from tg_bridge.agents.claude.tui_gate import wait_for_hooks
from tg_bridge.agents.claude.trust_dialog import prompt_showing, startup
from tg_bridge.agents.claude.tui_turn import STARTUP_DIALOG_ERROR
from tg_bridge.agents.events import ApprovalRequest, ApprovalWithdrawn, Decision, SessionStarted
from tg_bridge.agents.pty_mirror import PtyMirror
from tg_bridge.agents.pty_process import CLOSE_WRITE_TIMEOUT_SECONDS, PtyProcess
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

ESC_GAP_SECONDS = 0.5           # Esc, a pause, then /exit, so the TUI has left its menu
EXIT_COMMAND = "/exit"
HOOK_DENY_SECONDS = 0.5         # after a pending request is denied at stop, the TUI gets this long before the first key
DEFAULT_PROMPT_TIMEOUT = 45.0
DEFAULT_HOOK_TIMEOUT_SECONDS = 21600
_WHITESPACE = re.compile(r"\s+")


def _turn_state(name: str) -> property:
    """A read-write property onto self._turns. The turn state lives in TranscriptTurns; these names keep the session's view
    of it, which the tests use to open and inspect a turn."""
    return property(lambda self: getattr(self._turns, name), lambda self, value: setattr(self._turns, name, value))


def build_tui_args(spec: SessionSpec, settings: str) -> list[str]:
    """The argv of the interactive TUI for spec. settings is the --settings JSON that installs the permission hook; with
    spec.memory_dir it also carries the topic's autoMemoryDirectory. There is one --settings flag either way."""
    deny = ",".join(dict.fromkeys((KILL_SWITCH_TOOL, "AskUserQuestion", *spec.deny_tools)))
    args = [*spec.launcher.argv, *permission_args(spec.mode), "--disallowedTools", deny,
            "--append-system-prompt", spec.system_rules, "--name", f"tg:{spec.topic}"]
    allowed = ",".join(rule for rule in spec.allow_rules if rule)
    if allowed and spec.mode != BYPASS_MODE:  # bypass ignores allow lists, as the print transport does
        args += ["--allowedTools", allowed]
    if spec.resume_id:
        args += ["--resume", spec.resume_id]
    elif spec.new_id:
        args += ["--session-id", spec.new_id]
    if spec.model:
        args += ["--model", spec.model]
    return [*args, "--settings", with_memory(settings, spec.memory_dir)]


class ClaudeTuiSession(PtyProcess, AgentSession):
    """One interactive claude TUI bound to one topic turn. start() spawns it and waits for the prompt; send_user() types
    the message and follows the transcript to the end of the turn; end_gracefully() types /exit."""

    # The turn state is in self._turns (TranscriptTurns); these are the session's names for it.
    _turn_lock = _turn_state("_turn_lock")
    _gen = _turn_state("_gen")
    _done = _turn_state("_done")
    _tail = _turn_state("_tail")
    _tail_gen = _turn_state("_tail_gen")
    _turn = _turn_state("_turn")

    def __init__(self, spec: SessionSpec, sink: EventSink, *, logger: Logger, clock: Clock,
                 translator: ClaudeTranslator | None = None, prompt_timeout: float = DEFAULT_PROMPT_TIMEOUT,
                 hook_timeout_seconds: int = DEFAULT_HOOK_TIMEOUT_SECONDS) -> None:
        self.session_id = spec.resume_id or spec.new_id
        socket_path = _socket_path(spec.errfile_dir, spec.thread_id)
        rows, cols = spec.tui_size
        super().__init__(name=str(spec.thread_id), cwd=spec.cwd, env=_child_env(spec.env, socket_path),
                         errfile_dir=spec.errfile_dir, sink=sink, logger=logger, clock=clock, rows=rows, cols=cols)
        self._spec = spec
        mirror = os.path.join(spec.view_dir, f"{spec.thread_id}.pty")
        self._mirror = (PtyMirror(mirror, logger, on_rotate=self.redraw) if spec.view_dir else None)  # the view's raw copy
        self._translator = translator if translator is not None else ClaudeTranslator(cwd=spec.cwd)
        self._prompt_timeout = prompt_timeout
        self._hook_timeout_seconds = hook_timeout_seconds
        self._server = HookServer(socket_path, self._on_hook_request, logger=logger)
        self._mode_lock = threading.Lock()  # guards _mode, which the hook thread reads
        self._mode = spec.mode
        self._turns = TranscriptTurns(config_dir=spec.launcher.config_dir, cwd=spec.cwd, session_id=self.session_id,
                                      translator=self._translator, emit=self._emit, log=self._log, logger=logger,
                                      clock=clock, deny_requests=self._deny_requests)
        self._startup_problem: str | None = None   # why the startup left no prompt; written by start(), read by send_user

    # ---- PtyProcess hooks -----------------------------------------------------------------------------

    def _argv(self) -> list[str]:
        return build_tui_args(self._spec, hook_settings(hook_command(), self._hook_timeout_seconds))

    def _on_output(self, text: str) -> None:
        """Reader thread: every raw chunk goes to the view mirror, when the session has one."""
        if self._mirror is not None:
            self._mirror.write(text)

    def _on_spawned(self) -> None:
        """Reports the session and runs the startup (trust_dialog.startup): the startup dialogs are answered and the prompt is
        waited for. A problem is kept and logged: send_user types nothing while it stands. A dead TUI is seen by the turn
        runner, as it is for print."""
        if self.session_id is None:
            self._log("no session id: the turn cannot follow a transcript")
        else:
            self._emit(SessionStarted(session_id=self.session_id, model=self._spec.model, mode=self._spec.mode,
                                      cwd=self._spec.cwd, resumed=bool(self._spec.resume_id)))
        self._startup_problem = startup(self, self._clock, self._prompt_timeout, self._log)

    def _graceful_close(self) -> None:
        """First step of stop(): Esc leaves any menu, then /exit ends the TUI, each write short-bounded. stop() has already
        marked the session as ending, so no self turn starts from the lines this writes."""
        if self.press_esc(CLOSE_WRITE_TIMEOUT_SECONDS):
            self._clock.sleep(ESC_GAP_SECONDS)
            self.type_text(EXIT_COMMAND, CLOSE_WRITE_TIMEOUT_SECONDS)

    def _on_exited(self, returncode: int | None) -> None:
        self._close_side_channels()
        super()._on_exited(returncode)

    def stderr_tail(self, n: int = 20) -> str:
        """The screen tail. The TUI draws spaces as cursor moves, so a phrase can come out without its spaces. A resume
        failure is then named in full, so the turn runner's NO_CONVERSATION check still sees it."""
        tail = super().stderr_tail(n)
        if _WHITESPACE.sub("", NO_CONVERSATION) in _WHITESPACE.sub("", tail) and NO_CONVERSATION not in tail:
            return f"{NO_CONVERSATION}\n{tail}"
        return tail

    # ---- lifecycle ------------------------------------------------------------------------------------

    def start(self) -> None:
        """Opens the permission socket, then starts the TUI. A failed spawn closes the socket again."""
        if not self._turns.open_unless_closed(self._server.start):  # a stop() came first: nothing is opened or spawned
            return
        if self._mirror is not None:  # before the spawn, so the TUI's first output is not lost
            self._mirror.start()
        try:
            super().start()
        except BaseException:
            self._close_side_channels()
            raise

    def stop(self, grace: float = 5.0) -> int | None:
        self._turns.mark_ending()
        try:
            self._deny_pending()  # before the first key: a key would answer the TUI's own dialog
            return super().stop(grace)
        finally:
            self._close_side_channels()

    def end_gracefully(self, timeout: float) -> bool:
        """Types /exit and waits up to timeout seconds. True when the TUI has exited (or never ran)."""
        self._turns.mark_ending()
        proc = self._proc
        if proc is None or not self.alive():
            return True
        self._deny_pending()
        self.type_text(EXIT_COMMAND, CLOSE_WRITE_TIMEOUT_SECONDS)
        return self._wait(proc, timeout)

    def _deny_pending(self) -> None:
        """Before a stop or an end: every pending request is denied, a background subagent's too, since no turn
        continues. Only the interrupt path (TranscriptTurns.finish) passes main_only, so a subagent's ask survives an Esc."""
        if self._deny_requests(TURN_ENDED):  # before a key: the TUI gets HOOK_DENY_SECONDS to take the deny
            self._clock.sleep(HOOK_DENY_SECONDS)

    def _deny_requests(self, message: str, *, main_only: bool = False) -> list[str]:
        """Denies the pending permission requests and withdraws each one, so its card closes and a later tap cannot save
        a rule for a request that is already denied. Emitted with no lock held."""
        ids = self._server.cancel_all(message, main_only=main_only)
        for request_id in ids:
            self._emit(ApprovalWithdrawn(approval_id=request_id))
        return ids

    def _close_side_channels(self) -> None:
        """Stops following the transcript (TranscriptTurns.close, which drains the tail first) and denies every pending
        permission request. Safe to call more than once."""
        self._turns.close()
        self._server.close()
        if self._mirror is not None:
            self._mirror.close()

    # ---- AgentSession ---------------------------------------------------------------------------------

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        if images:
            raise ValueError("Claude sessions take text only: capabilities.images is False")
        wait_for_hooks(self._server, self._clock, self._deny_requests, self._log)
        gen, primary, follows = self._turns.begin()
        if self._startup_problem is not None:  # nothing is typed into a dialog the bridge does not know
            if not prompt_showing(self):
                self._log(f"message not typed: {self._startup_problem}")
                self._turns.finish(gen, error=STARTUP_DIALOG_ERROR)
                return
            self._startup_problem = None  # the dialog was answered (in the view) and the prompt shows now
        armed = self._turns.follow_turn(gen, primary, follows)
        if not self.type_text(text):
            self._log("the message was not typed: the TUI is gone")
            return
        if armed:
            self._turns.watch_echo(gen)

    def answer(self, req: ApprovalRequest, decision: Decision, *,
               updated_input: dict | None = None, answers: dict | None = None,
               updated_permissions: list | None = None, deny_message: str | None = None) -> None:
        """Answers the hook of a pending request. updated_permissions is accepted for the signature but not sent: the
        TUI's own rules are the bridge's, so an 'always' answer is a plain allow here."""
        if decision is Decision.DENY:
            response = permission_deny(deny_message or DENY_MESSAGE)
        else:
            inp = updated_input
            if answers is not None:
                inp = dict(req.input if inp is None else inp, answers=answers)
            response = permission_allow(inp)
        if not self._server.answer(req.approval_id, response):
            self._log(f"answer for an unknown or closed request {req.approval_id}")

    def interrupt(self, timeout: float = INTERRUPT_TIMEOUT) -> bool:
        """Esc. The TUI reacts at once, so timeout is not waited for. The interrupt line of the transcript, or else the
        fallback after INTERRUPT_FALLBACK_SECONDS, ends the turn with INTERRUPTED. Nothing is emitted outside a turn."""
        return self._turns.interrupt(self.press_esc)

    def set_mode(self, mode: str) -> bool:
        """Remembers the mode. Only bypass is live: the hook then allows every request, the TUI never asks in bypass
        either. Any other mode applies from the next turn, through the CLI flag."""
        with self._mode_lock:
            self._mode = mode
        return mode == BYPASS_MODE

    def set_model(self, model: str) -> bool:
        """Always False: typing /model would save the user's default model, so the model applies from the next turn."""
        return False

    # ---- permission hook (server thread) --------------------------------------------------------------

    def _on_hook_request(self, request_id: str, payload: dict) -> None:
        with self._mode_lock:
            bypass = self._mode == BYPASS_MODE
        if bypass:
            self._server.answer(request_id, permission_allow())
            return
        event = {"type": "control_request", "request_id": request_id,
                 "request": {"subtype": "can_use_tool", "tool_name": payload.get("tool_name"),
                             "input": payload.get("tool_input") or {},
                             "permission_suggestions": payload.get("permission_suggestions")}}
        try:
            events = self._translator.translate(event)
        except Exception as exc:  # the TUI waits for an answer, so a failure is a deny
            self._log(f"permission request: {exc!r}")
            events = []
        if not events:
            self._server.answer(request_id, permission_deny(DENY_MESSAGE))
            return
        for item in events:
            self._emit(item)

    # ---- transcript (delegated) -----------------------------------------------------------------------

    def _on_transcript(self, tail_gen: int, line: dict) -> None:
        """One transcript line of the current turns (TranscriptTurns.on_transcript). The tail thread calls it, and the
        tests feed lines through it."""
        self._turns.on_transcript(tail_gen, line)
