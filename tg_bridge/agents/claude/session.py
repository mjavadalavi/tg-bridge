"""ClaudeSession: one headless Claude CLI bound to one topic turn, driven over stream-json stdio.

Port of the process and protocol part of the live TopicSession (C11). build_args() is a pure function: for the same
inputs it returns the same argv as the live build_args. The session parses each line only far enough to route it:
the session id is learned here, control_response lines go to the ControlChannel, and every other line goes to the
ClaudeTranslator, which is the one place that turns raw objects into AgentEvents.

Threads: the reader thread (ProcessSession) calls _on_line; answer(), interrupt(), set_mode(), set_model() and
send_user() are called from other threads. A control request made on the reader thread is refused (ControlChannel),
and set_mode()/set_model()/interrupt() report that as False instead of raising.

Background work (linger): every raw line also goes through BackgroundLedger, which counts the background tasks and
wakeups still pending. _track() decides which BackgroundWork and TurnBegan events a line produces; they are emitted before
the translated events of the same line. A turn the CLI starts by itself (a task notification, a wakeup, or a main
session's text while no turn is open) is reported as TurnBegan; the user's own turns never are.

Interrupts: interrupt() marks the current turn (_interrupt_intent). That turn's error_during_execution result is then
reported as INTERRUPTED, so a self turn keeps the session (linger).
Lock map: _state_lock guards _ledger, _turn_open, _removed and _interrupt_intent. Nothing is emitted or written while it
is held.
"""

import json
import threading
from dataclasses import replace

from tg_bridge.agents.base import AgentSession, EventSink, SessionSpec
from tg_bridge.agents.claude import protocol
from tg_bridge.agents.claude.background import BackgroundLedger
from tg_bridge.agents.claude.memory_settings import with_memory
from tg_bridge.agents.claude.protocol import ControlChannel
from tg_bridge.agents.claude.rules import ClaudeRuleMapper
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import (AgentEvent, ApprovalRequest, BackgroundWork, Decision, TurnBegan, TurnResult,
                                     TurnStatus)
from tg_bridge.agents.process import ProcessSession
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

KILL_SWITCH_TOOL = "KillShell"        # the CLI refuses these even in bypass mode
BYPASS_MODE = "bypassPermissions"
INTERRUPT_TIMEOUT = 10.0
ERROR_DURING_EXECUTION = "error_during_execution"  # the CLI's end of a turn that an interrupt stopped


def _opens_turn(event: dict) -> bool:
    """True for a line that starts a turn the CLI began by itself: a system init, the start of a scheduled turn, or a
    main session's assistant line. A subagent line (parent_tool_use_id set) never starts one."""
    kind = event.get("type")
    if kind == "system":
        return event.get("subtype") == "init"
    if kind == "command_lifecycle":
        return event.get("state") == "started"
    return kind == "assistant" and event.get("parent_tool_use_id") is None


def _origin(event: dict, removed: bool) -> str:
    """Why a self turn began, for the log: a scheduled turn, a task notification, or unknown (empty)."""
    if event.get("type") == "command_lifecycle":
        return "scheduled"
    return "task-notification" if removed else ""


def _as_interrupted(event: AgentEvent) -> AgentEvent:
    """The end of a turn that was interrupted on purpose: error_during_execution is INTERRUPTED, not an error."""
    if isinstance(event, TurnResult) and event.status is TurnStatus.ERROR and event.subtype == ERROR_DURING_EXECUTION:
        return replace(event, status=TurnStatus.INTERRUPTED, error=None)
    return event


def build_args(spec: SessionSpec) -> list[str]:
    """The claude argv for spec: the live TopicSession.build_args flags, plus --settings with the topic's own memory
    folder when spec.memory_dir is set."""
    deny = ",".join(dict.fromkeys((KILL_SWITCH_TOOL, *spec.deny_tools)))
    args = [*spec.launcher.argv, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--permission-prompt-tool", "stdio", "--permission-mode", spec.mode,
            "--allow-dangerously-skip-permissions", "--disallowedTools", deny,
            "--append-system-prompt", spec.system_rules, "--name", f"tg:{spec.topic}",
            "--forward-subagent-text"]
    allowed = ",".join(rule for rule in spec.allow_rules if rule)
    if allowed and spec.mode != BYPASS_MODE:  # bypass ignores allow lists, as the live file does
        args += ["--allowedTools", allowed]
    if spec.resume_id:
        args += ["--resume", spec.resume_id]
    elif spec.new_id:
        args += ["--session-id", spec.new_id]
    if spec.model:
        args += ["--model", spec.model]
    if spec.memory_dir:
        args += ["--settings", with_memory("", spec.memory_dir)]
    return args


class ClaudeSession(ProcessSession, AgentSession):
    """One claude process. start() spawns it and sends initialize; the session id is learned from the stream."""

    def __init__(self, spec: SessionSpec, sink: EventSink, *, logger: Logger, clock: Clock,
                 translator: ClaudeTranslator | None = None,
                 initialize_timeout: float = protocol.INITIALIZE_TIMEOUT) -> None:
        # the error file is named by thread id, as in the live file: <thread>-<time>.err
        super().__init__(name=str(spec.thread_id), cwd=spec.cwd, env=spec.env, errfile_dir=spec.errfile_dir,
                         sink=sink, logger=logger, clock=clock)
        self._spec = spec
        self._translator = translator if translator is not None else ClaudeTranslator(cwd=spec.cwd)
        self._initialize_timeout = initialize_timeout
        self._channel = ControlChannel(self.write, self.on_reader_thread)
        self.session_id: str | None = None
        self._state_lock = threading.Lock()
        self._ledger = BackgroundLedger()
        self._turn_open = False    # a user or self turn is running (set by send_user, cleared by the result)
        self._removed = False      # the ledger removed a pending task since the last turn began or send_user
        self._interrupt_intent = False  # interrupt() was asked for the current turn (cleared when it ends or the next begins)

    # ---- ProcessSession hooks -------------------------------------------------------------------------

    def _argv(self) -> list[str]:
        return build_args(self._spec)

    def _on_spawned(self) -> None:
        """Handshake. A timeout or a dead process is logged and the turn goes on, as in the live file."""
        try:
            self._channel.initialize(self._initialize_timeout)
        except TimeoutError:
            self._log("initialize timed out, continuing")
        except RuntimeError as e:  # the CLI exited before replying, e.g. --resume of an unknown id
            self._log(f"initialize failed, continuing: {e!r}")

    def _on_exited(self, returncode: int | None) -> None:
        self._channel.fail_all()  # wake every waiter; their replies stay None
        super()._on_exited(returncode)

    def _on_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        self._learn_session_id(event)
        kind = event.get("type")
        try:
            work, began, interrupted = self._track(event)
        except Exception as e:  # a bookkeeping failure must not end the stream
            self._log(f"background ledger: {e!r}")
            work, began, interrupted = None, None, False
        if work is not None:
            self._emit(work)
        if began is not None:
            self._emit(began)
        if kind == "control_response":
            self._channel.deliver(event)
            return
        try:
            events = self._translator.translate(event)
        except Exception as e:  # a translator failure must not end the stream or leave the CLI waiting
            self._log(f"translate: {e!r}")
            events = []
        if interrupted:
            events = [_as_interrupted(item) for item in events]
        for item in events:
            self._emit(item)
        if kind == "control_request" and not events:
            self.write(protocol.unhandled_response(event))

    def _track(self, event: dict) -> tuple[BackgroundWork | None, TurnBegan | None, bool]:
        """Ledger and turn bookkeeping for one raw line, under _state_lock. Returns the events to emit before the
        translated ones, and whether this line ends an interrupted turn. A result clears the turn flag before the
        result's own events are emitted, so a send_user made from a TurnResult listener is not overwritten."""
        with self._state_lock:
            before = self._ledger.pending()
            changed = self._ledger.note(event)
            pending = self._ledger.pending()
            if pending < before:
                self._removed = True
            work = BackgroundWork(pending=pending) if changed else None
            began = None
            if not self._turn_open and _opens_turn(event):
                self._turn_open = True
                began = TurnBegan(origin=_origin(event, self._removed))
                self._removed = False
                self._interrupt_intent = False  # a new turn never inherits an interrupt aimed at an older one
            interrupted = False
            if event.get("type") == "result" and not event.get("resume_reason"):
                self._turn_open = False
                interrupted, self._interrupt_intent = self._interrupt_intent, False
        return work, began, interrupted

    def _learn_session_id(self, event: dict) -> None:
        sid = event.get("session_id")
        kind = event.get("type")
        if sid and (self.session_id is None or kind == "result"
                    or (kind == "system" and event.get("subtype") == "init")):
            self.session_id = sid

    # ---- AgentSession ---------------------------------------------------------------------------------

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        if images:
            raise ValueError("Claude sessions take text only: capabilities.images is False")
        with self._state_lock:
            self._turn_open = True  # before the frame goes out: the CLI's answer must find the turn already open
            self._removed = False
            self._interrupt_intent = False
        self.write(protocol.user_message(text, self.session_id))

    def answer(self, req: ApprovalRequest, decision: Decision, *,
               updated_input: dict | None = None, answers: dict | None = None,
               updated_permissions: list | None = None, deny_message: str | None = None) -> None:
        """Answer a can_use_tool request. ALLOW_ALWAYS needs updated_permissions from the rule mapper; without
        it the answer is a plain allow. answers (AskUserQuestion) are merged into the tool input. A Deny carries
        deny_message to the CLI (the owner's text when None); the model reads it."""
        if decision is Decision.DENY:
            if deny_message is None:
                self.answer_permission(req.approval_id, False)
            else:
                self.answer_permission(req.approval_id, False, message=deny_message)
            return
        if decision is Decision.ALLOW_ALWAYS and updated_permissions is None:
            updated_permissions = ClaudeRuleMapper().updated_permissions(req)  # derived from the request itself
        inp = req.input if updated_input is None else updated_input
        if answers is not None:
            inp = dict(inp, answers=answers)
        self.answer_permission(req.approval_id, True, updated_input=inp, updated_permissions=updated_permissions)

    def answer_permission(self, request_id: str, allow: bool, updated_input: dict | None = None,
                          message: str = protocol.DENY_MESSAGE, updated_permissions: list | None = None) -> bool:
        """The live answer_permission: one control_response frame, allow or deny."""
        if allow:
            frame = protocol.permission_allow(request_id, updated_input, updated_permissions)
        else:
            frame = protocol.permission_deny(request_id, message)
        return self.write(frame)

    def interrupt(self, timeout: float = INTERRUPT_TIMEOUT) -> bool:
        with self._state_lock:  # set before the request: the CLI's result may arrive before its control_response
            self._interrupt_intent = True
        try:
            self._channel.request("interrupt", timeout)
            return True
        except Exception as e:  # no reply, dead stdin, or called on the reader thread
            with self._state_lock:
                self._interrupt_intent = False
            self._log(f"interrupt failed: {e!r}")
            return False

    def set_mode(self, mode: str) -> bool:
        return self._control_ok("set_permission_mode", mode=mode)

    def set_model(self, model: str) -> bool:
        return self._control_ok("set_model", model=model)

    def stop(self, grace: float = 5.0) -> int | None:
        """Ask the CLI to interrupt first (bounded by grace), then the ProcessSession ladder."""
        if self.alive():
            self.interrupt(timeout=grace)
        return super().stop(grace)

    def _control_ok(self, subtype: str, **fields: object) -> bool:
        try:
            self._channel.request(subtype, **fields)
            return True
        except Exception as e:
            self._log(f"{subtype} failed: {e!r}")
            return False
