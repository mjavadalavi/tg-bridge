"""ClaudeTuiSession: one interactive Claude TUI in a pseudo-terminal, bound to one topic turn (the tui transport).

Some services refuse `claude -p` but accept the interactive TUI. The TUI runs on a PTY (PtyProcess) and writes its
transcript while a turn runs, so the turn is read from the transcript: TranscriptTail follows the session's .jsonl and
every line goes to the same ClaudeTranslator as the print transport. The turn ends at the first system/turn_duration
after the message was typed; an interrupt (Esc) writes none, so the session ends that turn itself.

The tail lives as long as the session, not only as long as a turn: after a turn it keeps reading, so background work
the CLI reports (BackgroundLedger, BackgroundWork) and a turn the CLI starts by itself (TurnBegan, then a new generation
with its own TurnResult) are both seen. A closing session (_ending) starts no such turn.

Threads: the tail thread delivers every transcript line (_on_transcript), the PTY reader delivers the output and the exit,
and the caller's thread types keys. Lock map: _turn_lock guards the generations, the tail, the ledger and TuiTurn; no event
is emitted and no tail is stopped while it is held.

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
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.background import BackgroundLedger
from tg_bridge.agents.claude.hookbridge import (TURN_ENDED, HookServer, hook_command, hook_settings,
                                                permission_allow, permission_deny)
from tg_bridge.agents.claude.protocol import DENY_MESSAGE
from tg_bridge.agents.claude.session import BYPASS_MODE, INTERRUPT_TIMEOUT, KILL_SWITCH_TOOL
from tg_bridge.agents.claude.tail import TranscriptTail
from tg_bridge.agents.claude.tui_helpers import _child_env, _socket_path, permission_args
from tg_bridge.agents.claude.translate import NO_CONVERSATION, ClaudeTranslator
from tg_bridge.agents.claude.tui_gate import wait_for_hooks
from tg_bridge.agents.claude.trust_dialog import prompt_showing, startup
from tg_bridge.agents.claude.tui_turn import (ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE, INTERRUPTED_SUBTYPE,
                                                STARTUP_DIALOG_ERROR, TURN_DURATION, USER_ECHO_SECONDS, TuiTurn,
                                                is_interrupt, turn_start_origin)
from tg_bridge.agents.events import (ApprovalRequest, ApprovalWithdrawn, BackgroundWork, Decision, SessionStarted,
                                     TurnBegan, TurnResult, TurnStatus, UsageEvent)
from tg_bridge.agents.pty_mirror import PtyMirror
from tg_bridge.agents.pty_process import CLOSE_WRITE_TIMEOUT_SECONDS, PtyProcess
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

ESC_GAP_SECONDS = 0.5           # Esc, a pause, then /exit, so the TUI has left its menu
EXIT_COMMAND = "/exit"
TRANSCRIPT_WAIT_SECONDS = 15.0  # the transcript must exist this soon after the message is typed
HOOK_DENY_SECONDS = 0.5         # after a pending request is denied at stop, the TUI gets this long before the first key
INTERRUPT_FALLBACK_SECONDS = 3.0  # without an interrupt line in the transcript, the session ends the turn itself
DEFAULT_PROMPT_TIMEOUT = 45.0
DEFAULT_HOOK_TIMEOUT_SECONDS = 21600
_WHITESPACE = re.compile(r"\s+")


def build_tui_args(spec: SessionSpec, settings: str) -> list[str]:
    """The argv of the interactive TUI for spec. settings is the --settings JSON that installs the permission hook."""
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
    return [*args, "--settings", settings]


class ClaudeTuiSession(PtyProcess, AgentSession):
    """One interactive claude TUI bound to one topic turn. start() spawns it and waits for the prompt; send_user() types
    the message and follows the transcript to the end of the turn; end_gracefully() types /exit."""

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
        self._turn_lock = threading.Lock()  # guards everything below; never held while emitting or stopping a tail
        self._mode = spec.mode
        self._closed = False
        self._gen = 0                       # the turn generation: a line from an older turn's tail is ignored
        self._done = True                   # no turn is running yet
        self._tail: TranscriptTail | None = None
        self._tail_gen = 0                  # the tail generation: a line from a replaced tail is ignored
        self._ending = False                # set by an intentional close (stop, end_gracefully): no self turn starts from here on
        self._before: tuple[str | None, int] = (None, 0)  # the transcript found before the send, and its size
        self._turn = TuiTurn()                     # the turn's usage, text and interrupt state; guarded by _turn_lock
        self._ledger = BackgroundLedger()          # the background work still pending; guarded by _turn_lock
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
        with self._turn_lock:
            if self._closed:  # a stop() came first: nothing is opened or spawned
                return
            self._server.start()
        if self._mirror is not None:  # before the spawn, so the TUI's first output is not lost
            self._mirror.start()
        try:
            super().start()
        except BaseException:
            self._close_side_channels()
            raise

    def stop(self, grace: float = 5.0) -> int | None:
        self._mark_ending()
        try:
            self._deny_pending()  # before the first key: a key would answer the TUI's own dialog
            return super().stop(grace)
        finally:
            self._close_side_channels()

    def end_gracefully(self, timeout: float) -> bool:
        """Types /exit and waits up to timeout seconds. True when the TUI has exited (or never ran)."""
        self._mark_ending()
        proc = self._proc
        if proc is None or not self.alive():
            return True
        self._deny_pending()
        self.type_text(EXIT_COMMAND, CLOSE_WRITE_TIMEOUT_SECONDS)
        return self._wait(proc, timeout)

    def _mark_ending(self) -> None:
        """An intentional close (stop, end_gracefully) begins: a transcript line that would start a turn is no longer a
        turn. A natural exit does not call this, so its final drain still sees a background update."""
        with self._turn_lock:
            self._ending = True

    def _deny_pending(self) -> None:
        """Before a stop or an end: every pending request is denied, a background subagent's too, since no turn
        continues. Only the interrupt path (_finish) passes main_only, so a subagent's ask survives an Esc."""
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
        """Stops following the transcript and denies every pending permission request. Safe to call more than once.

        The tail is stopped first: its final drain delivers the lines the CLI wrote before it exited (the last answer,
        turn_duration, and a background update written just before a natural exit), so transcript handling is still on
        while it runs; an intentional close marked _ending before this, so its drain starts no self turn. Only then is the
        session marked closed, which rejects the lines read after this point. The drain runs without _turn_lock held."""
        with self._turn_lock:
            tail = self._tail
        if tail is not None:
            tail.stop()
        with self._turn_lock:
            self._closed = True
            current, self._tail = self._tail, None
        if current is not None and current is not tail:  # a follow that started during the drain
            current.stop()
        self._server.close()
        if self._mirror is not None:
            self._mirror.close()

    # ---- AgentSession ---------------------------------------------------------------------------------

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        if images:
            raise ValueError("Claude sessions take text only: capabilities.images is False")
        wait_for_hooks(self._server, self._clock, self._deny_requests, self._log)
        primary = self._transcript_path()
        before = None
        if self.session_id:
            before = history.transcript_path(self._spec.launcher.config_dir, self._spec.cwd, self.session_id)
        with self._turn_lock:
            follows = self._tail is not None  # a live session keeps the tail of its earlier turns
            self._gen += 1
            gen = self._gen
            self._done = False
            self._turn.reset()
            self._before = (before, TranscriptTail.size_of(before) if before else 0)
        if self._startup_problem is not None:  # nothing is typed into a dialog the bridge does not know
            if not prompt_showing(self):
                self._log(f"message not typed: {self._startup_problem}")
                self._finish(gen, error=STARTUP_DIALOG_ERROR)
                return
            self._startup_problem = None  # the dialog was answered (in the view) and the prompt shows now
        if primary is not None and not follows:
            self._follow(gen, primary, TranscriptTail.size_of(primary))  # the turn's lines start after what is there
            threading.Thread(target=self._locate, args=(gen, primary), name="transcript-locate", daemon=True).start()
        with self._turn_lock:  # the echo net is armed only where this turn's transcript is followed
            armed = self._tail is not None and self._gen == gen
        if not self.type_text(text):
            self._log("the message was not typed: the TUI is gone")
            return
        if armed:
            threading.Thread(target=self._echo_watch, args=(gen,), name="user-echo", daemon=True).start()

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
        with self._turn_lock:
            gen = self._gen
            armed = not self._done
            self._turn.interrupting = self._turn.interrupting or armed
        sent = self.press_esc()
        if armed:
            threading.Thread(target=self._interrupt_fallback, args=(gen,), name="interrupt-fallback",
                             daemon=True).start()
        return sent

    def set_mode(self, mode: str) -> bool:
        """Remembers the mode. Only bypass is live: the hook then allows every request, the TUI never asks in bypass
        either. Any other mode applies from the next turn, through the CLI flag."""
        with self._turn_lock:
            self._mode = mode
        return mode == BYPASS_MODE

    def set_model(self, model: str) -> bool:
        """Always False: typing /model would save the user's default model, so the model applies from the next turn."""
        return False

    # ---- permission hook (server thread) --------------------------------------------------------------

    def _on_hook_request(self, request_id: str, payload: dict) -> None:
        with self._turn_lock:
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

    # ---- transcript (tail thread) ---------------------------------------------------------------------

    def _transcript_path(self) -> str | None:
        if not self.session_id:
            return None
        folder = history.encoded_cwd(os.path.realpath(self._spec.cwd))
        return os.path.join(self._spec.launcher.config_dir, "projects", folder, self.session_id + ".jsonl")

    def _follow(self, gen: int, path: str, offset: int) -> None:
        """Replaces the tail with one reading path from offset. Each tail has its own generation, so the lines of a
        replaced tail are ignored."""
        with self._turn_lock:
            if gen != self._gen or self._done or self._closed:
                return
            self._tail_gen += 1
            tail_gen = self._tail_gen
        tail = TranscriptTail(path, lambda line: self._on_transcript(tail_gen, line), logger=self._logger,
                              start_offset=offset)
        with self._turn_lock:
            if tail_gen != self._tail_gen or self._closed:  # a newer follow or a close came first
                return
            old, self._tail = self._tail, tail
        tail.start()
        if old is not None:
            old.stop()

    def _locate(self, gen: int, primary: str) -> None:
        """Fifteen seconds after the send: when the transcript is still not where it was expected, it is looked up with
        history.transcript_path and followed from there. The lookup happens once and is logged either way."""
        self._clock.sleep(TRANSCRIPT_WAIT_SECONDS)
        with self._turn_lock:
            if gen != self._gen or self._done or self._closed:
                return
            before_path, before_size = self._before
        if os.path.isfile(primary):
            return
        found = history.transcript_path(self._spec.launcher.config_dir, self._spec.cwd, self.session_id or "")
        if found is None or found == primary:
            self._log(f"no transcript within {TRANSCRIPT_WAIT_SECONDS:g} s; the turn may not end on its own")
            return
        self._log(f"transcript found at {found}, following it")
        # A file that existed before the send is read from its size at that time; a new one from its start.
        self._follow(gen, found, before_size if found == before_path else 0)

    def _on_transcript(self, tail_gen: int, line: dict) -> None:
        """Tail thread: one transcript line. A line that starts a turn while none is open (a task notification, a
        wakeup, or the bridge's own message seen late) opens a new generation, unless the session is ending."""
        origin = turn_start_origin(line)
        with self._turn_lock:
            if tail_gen != self._tail_gen or self._closed:
                return
            changed = self._ledger.note(line)
            pending = self._ledger.pending()
            began = self._done and not self._ending and origin is not None
            if began:
                self._gen += 1
                self._done = False
                self._turn.reset()
            gen = self._gen
        if changed:
            self._emit(BackgroundWork(pending=pending))
        if began:
            self._emit(TurnBegan(origin=origin))
        if line.get("type") == "system" and line.get("subtype") == TURN_DURATION:
            self._finish(gen, duration=line)
            return
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
            self._turn.note(line)
            interrupted = self._turn.interrupting and is_interrupt(line)
        try:
            events = self._translator.translate(line)
        except Exception as exc:  # one bad line must not stop the turn
            self._log(f"translate: {exc!r}")
            events = []
        self._emit_turn(gen, events)
        if interrupted:
            self._finish(gen, interrupted=True)

    def _emit_turn(self, gen: int, events: list) -> None:
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
        for event in events:
            if not isinstance(event, (SessionStarted, TurnResult, UsageEvent)):  # the session reports these itself
                self._emit(event)

    def _interrupt_fallback(self, gen: int) -> None:
        self._clock.sleep(INTERRUPT_FALLBACK_SECONDS)
        self._finish(gen, interrupted=True)

    def _echo_watch(self, gen: int) -> None:
        """Safety net for a typed message: when no line of this turn's message is read within USER_ECHO_SECONDS, the keys
        went to a dialog of the TUI, and the turn ends with an error instead of waiting for a turn_duration that never comes.
        _finish does nothing when the message was read, or when the generation is no longer this turn."""
        self._clock.sleep(USER_ECHO_SECONDS)
        self._finish(gen, error=ECHO_MISSED_ERROR)

    def _finish(self, gen: int, *, duration: dict | None = None, interrupted: bool = False,
                error: str | None = None) -> None:
        """Ends the turn once: an open permission request is closed first, then the usage and the TurnResult. The tail
        keeps running: it belongs to the session, and closes with it. An error ends it only while its message is unread."""
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
            if error is not None and (self._turn.echoed or self._closed):  # read after all, or the session is gone
                return
            self._done = True
            self._turn.interrupting = False
            usage = self._turn.usage_event()
            if interrupted:
                result = TurnResult(status=TurnStatus.INTERRUPTED, answer=None, subtype=INTERRUPTED_SUBTYPE)
            elif error is not None:
                result = TurnResult(status=TurnStatus.ERROR, answer=None, subtype=ECHO_MISSED_SUBTYPE, error=error)
            else:
                result = self._turn.result(duration)
        if interrupted:  # the main turn's requests still open after an Esc are denied before it is reported as ended;
            self._deny_requests(TURN_ENDED, main_only=True)  # a background subagent's ask stays open
        self._emit(usage)
        self._emit(result)

