"""CodexSession: one `codex app-server` process bound to one turn (DESIGN section 4).

Handshake and thread open happen in start(): initialize (with the optOutNotificationMethods list, retried once without
it when the server refuses), initialized, then thread/start or thread/resume. The SessionStarted event goes to the sink
before start() returns, so the id is stored before any turn/start is sent. send_user() sends turn/start. Replies to our
requests go through CodexTranslator, so the normalized events are produced in one place. Approval server requests go
through approvals.py, and every other server request is answered with -32601 by JsonRpcPeer, with one status line.

Threads: the reader thread (ProcessSession) parses each line and hands it to the JsonRpcPeer. Callbacks run there and
only emit events or write an answer. start(), send_user() and answer() run on the caller's thread and may wait for a
reply. set_mode() and set_model() store the override for the next turn, so they return False (not live).

Ending a turn: interrupt() sends turn/interrupt from the caller's thread and waits for the turn to end. stop() closes
stdin and runs the kill ladder. When the interrupt request fails or the turn does not end in time, interrupt() stops
the process and closes the turn as interrupted. Each turn gets at most one TurnResult.
"""

import json
import threading
import time
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from tg_bridge import __version__
from tg_bridge.agents.base import AgentSession, EventSink, SessionSpec
from tg_bridge.agents.codex import approvals
from tg_bridge.agents.codex.instructions import wrap_rules
from tg_bridge.agents.codex.rpc import JsonRpcPeer, PeerClosed, RpcError
from tg_bridge.agents.codex.translate import RESUME, START, TURN_START, CodexTranslator
from tg_bridge.agents.events import (ApprovalRequest, Decision, SessionStarted, StatusEvent, TurnResult,
                                     TurnStatus)
from tg_bridge.agents.process import ProcessSession
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

OPT_OUT_METHODS = ("mcpServer/startupStatus/updated", "remoteControl/status/changed", "hook/started",
                   "hook/completed", "account/rateLimits/updated")
REQUEST_TIMEOUT = 30.0          # seconds for each request; a silent server must not hold the turn forever
LOCAL_REPLY_ID = 0              # CodexTranslator reads only the reply shape, so a reply built here carries a dummy id
TURN_INTERRUPT = "turn/interrupt"
INTERRUPT_TIMEOUT = 10.0        # seconds for the turn/interrupt reply and the turn's end together (DESIGN section 4)
INTERRUPT_GRACE = 2.0           # stop() grace when the interrupt fallback has to stop the process


class CodexSession(ProcessSession, AgentSession):
    """One app-server process. start() spawns it and opens the thread; send_user() runs one turn."""

    def __init__(self, spec: SessionSpec, sink: EventSink, *, logger: Logger, clock: Clock,
                 translator: CodexTranslator | None = None, request_timeout: float = REQUEST_TIMEOUT) -> None:
        env = dict(spec.env)
        if spec.launcher.config_dir:
            # the launcher's folder wins over an inherited CODEX_HOME: rollouts for resume and handoff live there
            env["CODEX_HOME"] = spec.launcher.config_dir
        # the error file is named by thread id, as for Claude: <thread>-<time>.err
        super().__init__(name=str(spec.thread_id), cwd=spec.cwd, env=env, errfile_dir=spec.errfile_dir,
                         sink=sink, logger=logger, clock=clock)
        self._spec = spec
        self._translator = translator if translator is not None else CodexTranslator(mode=spec.mode, cwd=spec.cwd)
        self._timeout = request_timeout
        self._peer = JsonRpcPeer(self.write, self.on_reader_thread, report=self._log)
        self._peer.on_notification(self._on_notification)
        for method in approvals.APPROVAL_METHODS:
            self._peer.on_request(method, partial(self._on_approval, method))
        self._changes: dict[str, list] = {}      # fileChange item id -> changes, for the approval that follows it
        self._mode = spec.mode                   # overrides from set_mode() apply to the next turn
        self._model = spec.model
        self._failed = False                     # the handshake or the thread open failed: nothing more is sent
        self._turn_id: str | None = None
        self._closing = False                    # stop() was called: nothing more is sent
        self._turn_done = threading.Event()      # set once the turn's TurnResult was emitted
        self._turn_lock = threading.Lock()       # claims the single TurnResult of this session
        self.session_id: str | None = None

    # ---- ProcessSession hooks -------------------------------------------------------------------------

    def _argv(self) -> list[str]:
        return [*self._spec.launcher.argv, "app-server"]

    def _on_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            frame = json.loads(line)
        except ValueError:
            return
        if not isinstance(frame, dict):
            return
        method = frame.get("method")
        if isinstance(method, str) and "id" in frame and not approvals.is_approval(method):
            # JsonRpcPeer answers this with -32601; the user sees one line and the turn goes on
            self._publish([StatusEvent(kind="warning", detail=f"{method} is not supported by tg-bridge")])
        self._peer.handle_line(frame)

    def _on_exited(self, returncode: int | None) -> None:
        self._peer.close()  # fail any request still waiting for a reply from this process
        super()._on_exited(returncode)

    # ---- AgentSession ---------------------------------------------------------------------------------

    def start(self) -> None:
        """Spawn, initialize and open or resume the thread. A failure becomes a TurnResult(ERROR), not an exception."""
        try:
            super().start()
        except OSError as e:  # the launcher is missing or not executable
            self._fail(f"codex could not start: {e}")
            return
        try:
            self._initialize()
        except (RpcError, PeerClosed, TimeoutError, RuntimeError) as e:
            self._fail(f"codex handshake failed: {e}")
            return
        if not self._open_thread():
            self._failed = True

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        if self._failed or self._closing or self.session_id is None:
            self._log("send_user skipped: the session did not start or was stopped")
            return
        items: list[dict[str, Any]] = []
        if self._spec.system_rules:
            # the rules go with every turn: a resumed thread drops developerInstructions (DESIGN section 4)
            items.append({"type": "text", "text": wrap_rules(self._spec.system_rules), "text_elements": []})
        items.append({"type": "text", "text": text, "text_elements": []})
        items += [{"type": "localImage", "path": path} for path in images]
        params: dict[str, Any] = {"threadId": self.session_id, "input": items, **approvals.turn_params(self._mode)}
        if self._model:
            params["model"] = self._model
        with self._turn_lock:
            self._turn_done.clear()  # a new turn: its TurnResult is the next one emitted
            self._turn_id = None     # interrupt() must not use the id of the previous turn
        turn = self._exchange(TURN_START, params)
        if isinstance(turn, Mapping) and isinstance(turn.get("turn"), Mapping):
            self._turn_id = turn["turn"].get("id")

    def answer(self, req: ApprovalRequest, decision: Decision, *,
               updated_input: dict | None = None, answers: dict | None = None,
               deny_message: str | None = None) -> None:
        """Answer an approval. Codex asks no questions, takes no edited input and has no deny text, so answers,
        updated_input and deny_message are not used. An invalid choice or a request that is no longer open is logged
        and dropped."""
        try:
            result = approvals.result_for(req, decision)
        except ValueError as e:
            self._log(f"answer refused: {e}")
            return
        if not self._peer.respond(approvals.rpc_id(req), result):
            self._log(f"answer for {req.approval_id} dropped: the request is no longer open")

    def interrupt(self) -> bool:
        """Ask the running turn to stop, then wait for its TurnResult.

        turn/interrupt is sent from the caller's thread. One INTERRUPT_TIMEOUT deadline covers the reply and the turn's
        end together. If the reply fails, or the turn has not ended by the deadline, the process is stopped and the
        turn is closed as interrupted. Returns True once the turn has ended. False when there is nothing to interrupt
        (no turn running, or the session was never started or already stopped) and when called on the reader thread,
        where waiting for a reply would deadlock.
        """
        if self.on_reader_thread():
            self._log("interrupt refused on the reader thread")
            return False
        thread_id, turn_id = self.session_id, self._turn_id
        if self._turn_done.is_set() or self._closing or not self.alive() or thread_id is None or turn_id is None:
            return False
        deadline = time.monotonic() + INTERRUPT_TIMEOUT
        try:
            self._peer.request(TURN_INTERRUPT, {"threadId": thread_id, "turnId": turn_id},
                               timeout=INTERRUPT_TIMEOUT)
        except (RpcError, PeerClosed, TimeoutError) as e:
            self._log(f"turn/interrupt failed: {e}")
            self._stop_after_interrupt("interrupt request failed, the process was stopped")
            return True
        if not self._turn_done.wait(max(deadline - time.monotonic(), 0.0)):
            self._log("the turn did not end after turn/interrupt")
            self._stop_after_interrupt("interrupt timed out, the process was stopped")
        return True

    def end_gracefully(self, timeout: float) -> bool:
        self._closing = True
        return super().end_gracefully(timeout)

    def stop(self, grace: float = 5.0) -> int | None:
        """Close stdin (the app-server sees EOF), wait grace seconds, then the kill ladder and a group sweep."""
        self._closing = True
        return super().stop(grace)

    def _stop_after_interrupt(self, reason: str) -> None:
        """The fallback of interrupt(): stop the process, then close the turn as interrupted if it is still open."""
        self.stop(grace=INTERRUPT_GRACE)
        self._end_turn(TurnResult(status=TurnStatus.INTERRUPTED, answer=None, error=reason))

    def set_mode(self, mode: str) -> bool:
        approvals.policy(mode)  # raises UnknownMode for a key Codex does not know
        self._mode = mode
        return False  # applies from the next turn

    def set_model(self, model: str) -> bool:
        self._model = model
        return False  # applies from the next turn

    # ---- handshake and thread -------------------------------------------------------------------------

    def _initialize(self) -> None:
        """initialize, then initialized. A refusal that may be the optOut list is retried once without it."""
        try:
            self._peer.request("initialize", self._init_params(opt_out=True), timeout=self._timeout)
        except RpcError as e:
            self._log(f"initialize refused, retrying without optOutNotificationMethods: {e.message}")
            self._peer.request("initialize", self._init_params(opt_out=False), timeout=self._timeout)
        self._peer.notify("initialized")

    def _init_params(self, *, opt_out: bool) -> dict[str, Any]:
        capabilities: dict[str, Any] = {"experimentalApi": False}
        if opt_out:
            capabilities["optOutNotificationMethods"] = list(OPT_OUT_METHODS)
        return {"clientInfo": {"name": "tg-bridge", "title": "tg-bridge", "version": __version__},
                "capabilities": capabilities}

    def _open_thread(self) -> bool:
        """thread/resume when the spec names a thread, else thread/start. True when SessionStarted was emitted."""
        spec = self._spec
        if spec.resume_id:
            params = {"threadId": spec.resume_id, "excludeTurns": True, **self._thread_params()}
            return self._exchange(RESUME, params) is not None
        return self._exchange(START, self._thread_params()) is not None

    def _thread_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"cwd": self._spec.cwd, **approvals.thread_params(self._mode)}
        if self._model:
            params["model"] = self._model
        return params

    # ---- replies and events ---------------------------------------------------------------------------

    def _exchange(self, method: str, params: dict[str, Any]) -> Any:
        """One request. The reply goes through the translator, which yields SessionStarted, a resume_failed
        TurnResult or an error TurnResult. Returns the result object, or None when the request failed."""
        try:
            result = self._peer.request(method, params, timeout=self._timeout)
        except RpcError as e:
            error = {"code": e.code, "message": e.message}
            self._publish(self._translator.translate({"id": LOCAL_REPLY_ID, "error": error}, method=method))
            return None
        except (PeerClosed, TimeoutError) as e:
            self._fail(f"{method} failed: {e}")
            return None
        self._publish(self._translator.translate({"id": LOCAL_REPLY_ID, "result": result}, method=method))
        return result

    def _fail(self, message: str) -> None:
        self._failed = True
        self._log(message)
        self._publish([TurnResult(status=TurnStatus.ERROR, answer=None, error=message)])

    def _on_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == approvals.RESOLVED_METHOD:
            gone = approvals.withdrawn(params)
            if gone is not None:
                self._publish([gone])
            return
        self._track_file_change(method, params)
        self._publish(self._translator.translate({"method": method, "params": params}))

    def _track_file_change(self, method: str, params: Mapping[str, Any]) -> None:
        """Remember the changes of a fileChange item, because its approval request carries no paths."""
        item = params.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "fileChange" or not isinstance(item.get("id"), str):
            return
        if method == "item/started":
            self._changes[item["id"]] = list(item.get("changes") or [])
        elif method == "item/completed":
            self._changes.pop(item["id"], None)

    def _on_approval(self, method: str, request_id: int, params: dict[str, Any]) -> None:
        item_id = params.get("itemId")
        changes: Sequence[Mapping[str, Any]] = self._changes.get(item_id, ()) if isinstance(item_id, str) else ()
        req = approvals.request_from_frame(method, params, request_id, changes=changes, cwd=self._spec.cwd)
        if req is None:
            self._peer.error(request_id, approvals.UNSUPPORTED_ERROR_CODE, approvals.UNSUPPORTED_MESSAGE)
            return
        self._publish([req])

    def _publish(self, events: Sequence[Any]) -> None:
        for event in events:
            if isinstance(event, SessionStarted):
                self.session_id = event.session_id
            if isinstance(event, TurnResult):
                self._end_turn(event)
                continue
            self._emit(event)

    def _end_turn(self, result: TurnResult) -> None:
        """Emit the session's TurnResult, once. The claim is taken under a lock; the sink is called outside it."""
        with self._turn_lock:
            if self._turn_done.is_set():
                self._log(f"second TurnResult ({result.status.value}) dropped")
                return
            self._turn_done.set()
        self._emit(result)
