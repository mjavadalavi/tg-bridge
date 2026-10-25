"""Fake `codex app-server` over stdio JSON-RPC (no real codex is spawned). Test use only.

The launcher wrapper tests/fakes/launchers/codex runs this file with the argv a real launch gets, so the bridge's
`app-server` argument arrives here and is ignored. Message shapes come from scratchpad codex-probe/run1.jsonl and
run2.jsonl (codex-cli 0.162.1) and the 0.164.0 recheck (fix2 codex_live). Fields the bridge never reads are left out. The fake never executes commands or
writes files; it only speaks the protocol.

Environment:
  FAKE_CODEX_MODE       one of MODES below (default normal)
  FAKE_CODEX_LOG        file that receives every inbound frame as one JSON line {"in": frame}; unset means no log
  FAKE_CODEX_PIDFILE    grandchild mode: the pid of the sleeping child is written here
  FAKE_CODEX_ANSWER     text of the final answer (default "done")

Modes:
  normal            commentary, final answer, token usage, turn/completed completed
  command_approval  command approval offered as accept / acceptWithExecpolicyAmendment / cancel (no decline, as live);
                    accept -> completed, decline -> the turn goes on to the answer, cancel -> turn interrupted
  decline_offered   as command_approval, but availableDecisions also offers decline and acceptForSession
  file_approval     item/fileChange/requestApproval with NO availableDecisions (as live); same decision rules
  interrupt         command approval, then the command runs until turn/interrupt (its item never completes)
  resume_missing    thread/resume answers with an error; turns run normally
  fail              turn/completed with status failed and an error object
  retry             one `error` notification with willRetry true (responseStreamDisconnected), then the answer
  unknown_request   sends item/tool/requestUserInput and expects -32601 from the bridge, then answers
  stubborn          normal turn; after stdin EOF it ignores the EOF and sleeps until SIGTERM
  grandchild        normal turn; also starts a `sleep 300` child in the same process group
  reject_optout     initialize with capabilities.optOutNotificationMethods is refused (-32600); without it, normal

Exit: rc 0 on stdin EOF, except stubborn. Interrupts and approvals that are pending when EOF arrives end the process
with rc 0 and no turn/completed, as live.
"""

import itertools
import json
import os
import subprocess
import sys
import time

ANSWER_DEFAULT = "done"
THREAD_ID = "fake-thread-1"
CONTEXT_WINDOW = 258400
COMMAND = "/usr/bin/zsh -lc 'touch probe_ok.txt'"
UNSUPPORTED_CODE = -32601
INVALID_REQUEST_CODE = -32600
DELTA_CHUNK = 8
MODES = (
    "normal",
    "command_approval",
    "decline_offered",
    "file_approval",
    "interrupt",
    "resume_missing",
    "fail",
    "retry",
    "unknown_request",
    "stubborn",
    "grandchild",
    "reject_optout",
)
ACCEPTS = ("accept", "acceptForSession")
COMMAND_DECISIONS = ["accept", {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["touch", "probe_ok.txt"]}},
                     "cancel"]
PENDING_DECISIONS = ["accept", "acceptForSession", "decline", "cancel"]


class Interrupted(Exception):
    """turn/interrupt arrived while the fake was waiting. The turn ends as interrupted."""


class Closed(Exception):
    """stdin reached EOF while the fake was waiting. The process exits."""


def now_ms() -> int:
    return int(time.time() * 1000)


def verdict(reply: dict) -> str:
    """accept | decline | cancel | error, from the bridge's reply to a server request."""
    result = reply.get("result")
    decision = result.get("decision") if isinstance(result, dict) else None
    if isinstance(decision, dict):  # acceptWithExecpolicyAmendment and similar objects accept as well
        return "accept"
    return decision if isinstance(decision, str) else "error"


def park() -> None:
    while True:
        time.sleep(0.1)


class Fake:
    """One app-server connection. run() returns on stdin EOF; each turn is handled inside turn/start."""

    def __init__(self, mode: str, log_path: str, answer: str) -> None:
        self.mode = mode
        self.log_path = log_path
        self.answer = answer
        self.server_ids = itertools.count(0)
        self.item_count = 0
        self.turn_count = 0
        self.thread_id = THREAD_ID
        self.turn_id = ""
        self.children: list[subprocess.Popen] = []

    # ---- wire ---------------------------------------------------------------------------------------------------

    def send(self, frame: dict) -> None:
        sys.stdout.write(json.dumps(frame, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"method": method, "params": params, "emittedAtMs": now_ms()})

    def result(self, request_id: int, result: dict) -> None:
        self.send({"id": request_id, "result": result})

    def error(self, request_id: int, code: int, message: str) -> None:
        self.send({"id": request_id, "error": {"code": code, "message": message}})

    def read(self) -> dict | None:
        line = sys.stdin.readline()
        if not line:
            return None
        frame = json.loads(line)
        self.log(frame)
        return frame

    def log(self, frame: dict) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"in": frame}, ensure_ascii=False) + "\n")

    # ---- dispatch -----------------------------------------------------------------------------------------------

    def run(self) -> None:
        try:
            while True:
                frame = self.read()
                if frame is None:
                    return
                self.dispatch(frame)
        except Closed:
            return

    def dispatch(self, frame: dict) -> None:
        method = frame.get("method")
        if method is None or "id" not in frame:  # a reply nobody waits for, or a notification such as initialized
            return
        params = frame.get("params") or {}
        handlers = {
            "initialize": self.on_initialize,
            "thread/start": self.on_thread_start,
            "thread/resume": self.on_thread_resume,
            "turn/start": self.on_turn_start,
            "turn/interrupt": self.on_turn_interrupt,
        }
        handler = handlers.get(method)
        if handler is None:
            self.error(frame["id"], UNSUPPORTED_CODE, f"unsupported by fake codex: {method}")
        else:
            handler(frame["id"], params)

    def on_initialize(self, request_id: int, params: dict) -> None:
        capabilities = params.get("capabilities") or {}
        if self.mode == "reject_optout" and "optOutNotificationMethods" in capabilities:
            self.error(request_id, INVALID_REQUEST_CODE, "optOutNotificationMethods is not accepted")
            return
        self.result(request_id, {"userAgent": "fake-codex/0.164.0 (fake)", "codexHome": "/fake/codex-home",
                                 "platformFamily": "unix", "platformOs": "linux"})

    def on_thread_start(self, request_id: int, params: dict) -> None:
        self.result(request_id, self.thread_payload(THREAD_ID, params))

    def on_thread_resume(self, request_id: int, params: dict) -> None:
        thread_id = params.get("threadId", THREAD_ID)
        if self.mode == "resume_missing":
            self.error(request_id, INVALID_REQUEST_CODE, f"no rollout found for thread id {thread_id}")
            return
        self.result(request_id, self.thread_payload(thread_id, params))

    def on_turn_start(self, request_id: int, params: dict) -> None:
        self.turn_count += 1
        self.turn_id = f"fake-turn-{self.turn_count}"
        self.thread_id = params.get("threadId", self.thread_id)
        self.result(request_id, {"turn": self.turn_payload("inProgress", [], None)})
        self.run_turn()

    def on_turn_interrupt(self, request_id: int, params: dict) -> None:
        self.result(request_id, {})  # no turn is running here; the answer is still the protocol reply

    def thread_payload(self, thread_id: str, params: dict) -> dict:
        return {
            "thread": {"id": thread_id, "sessionId": thread_id, "ephemeral": bool(params.get("ephemeral")),
                       "status": {"type": "idle"}, "cwd": params.get("cwd"), "cliVersion": "0.164.0",
                       "originator": "fake-codex", "source": "vscode", "turns": []},
            "model": "gpt-fake",
            "modelProvider": "fake",
            "cwd": params.get("cwd"),
            "approvalPolicy": params.get("approvalPolicy"),
            "approvalsReviewer": params.get("approvalsReviewer", "user"),
            "sandbox": {"type": "workspaceWrite", "networkAccess": False},
            "reasoningEffort": "high",
        }

    # ---- turn ---------------------------------------------------------------------------------------------------

    def run_turn(self) -> None:
        self.notify("turn/started", {"threadId": self.thread_id, "turn": self.turn_payload("inProgress", [], None)})
        try:
            status, error, final = getattr(self, "script_" + self.mode)()
        except Interrupted:
            status, error, final = "interrupted", None, None
        items = [final] if final is not None and status == "completed" else []
        self.notify("turn/completed", {"threadId": self.thread_id,
                                       "turn": self.turn_payload(status, items, error, completed=True)})

    def turn_payload(self, status: str, items: list, error: dict | None, completed: bool = False) -> dict:
        return {"id": self.turn_id, "rootTurnId": self.turn_id, "items": items,
                "itemsView": "summary" if completed else "notLoaded", "status": status, "error": error,
                "startedAt": None, "completedAt": None, "durationMs": None}

    def script_normal(self) -> tuple:
        self.message("Working on it.\n", "commentary")
        return "completed", None, self.final_answer()

    script_resume_missing = script_normal
    script_stubborn = script_normal

    def script_grandchild(self) -> tuple:
        child = subprocess.Popen(["sleep", "300"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        self.children.append(child)
        pidfile = os.environ.get("FAKE_CODEX_PIDFILE")
        if pidfile:
            with open(pidfile, "w", encoding="utf-8") as handle:
                handle.write(str(child.pid))
        return self.script_normal()

    def script_fail(self) -> tuple:
        error = {"message": "fake failure", "codexErrorInfo": "other", "additionalDetails": None}
        return "failed", error, None

    def script_retry(self) -> tuple:
        error = {"message": "stream disconnected before completion",
                 "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": None}},
                 "additionalDetails": None}
        self.notify("error", {"error": error, "willRetry": True, "threadId": self.thread_id, "turnId": self.turn_id})
        return "completed", None, self.final_answer()

    def script_unknown_request(self) -> tuple:
        self.server_request("item/tool/requestUserInput",
                            {"threadId": self.thread_id, "turnId": self.turn_id, "itemId": self.next_item_id(),
                             "questions": []})
        return "completed", None, self.final_answer()

    def script_command_approval(self) -> tuple:
        return self.command_flow(COMMAND_DECISIONS, keep_running=False)

    def script_decline_offered(self) -> tuple:
        return self.command_flow(PENDING_DECISIONS, keep_running=False)

    def script_interrupt(self) -> tuple:
        return self.command_flow(COMMAND_DECISIONS, keep_running=True)

    def script_file_approval(self) -> tuple:
        item_id = self.next_item_id()
        self.message("I'll write the file now.\n", "commentary")
        change = {"path": "probe_ok.txt", "kind": {"type": "add", "move_path": None}, "diff": "hello\n"}
        started = {"type": "fileChange", "id": item_id, "changes": [change], "status": "inProgress"}
        self.notify("item/started", {"item": started, "threadId": self.thread_id, "turnId": self.turn_id,
                                     "startedAtMs": now_ms()})
        reply = self.server_request("item/fileChange/requestApproval",
                                    {"grantRoot": None, "itemId": item_id, "reason": None,
                                     "startedAtMs": now_ms(), "threadId": self.thread_id, "turnId": self.turn_id})
        decision = verdict(reply)
        if decision == "cancel":
            self.item_done(started, "declined")
            raise Interrupted()
        self.item_done(started, "completed" if decision in ACCEPTS else "declined")
        return "completed", None, self.final_answer()

    def command_flow(self, decisions: list, keep_running: bool) -> tuple:
        item_id = self.next_item_id()
        self.message("I'll run the command now.\n", "commentary")
        started = {"type": "commandExecution", "id": item_id, "pluginId": None, "scriptPath": None,
                   "command": COMMAND, "cwd": os.getcwd(), "processId": None, "source": "agent",
                   "status": "inProgress", "commandActions": [{"type": "unknown", "command": "touch probe_ok.txt"}],
                   "aggregatedOutput": None, "exitCode": None, "durationMs": None}
        self.notify("item/started", {"item": started, "threadId": self.thread_id, "turnId": self.turn_id,
                                     "startedAtMs": now_ms()})
        reply = self.server_request("item/commandExecution/requestApproval",
                                    {"kind": "command", "threadId": self.thread_id, "turnId": self.turn_id,
                                     "itemId": item_id, "startedAtMs": now_ms(), "environmentId": "local", "reason": None,
                                     "command": COMMAND, "cwd": os.getcwd(),
                                     "commandActions": started["commandActions"],
                                     "proposedExecpolicyAmendment": ["touch", "probe_ok.txt"],
                                     "availableDecisions": decisions})
        decision = verdict(reply)
        if decision == "cancel":
            self.item_done(started, "declined")
            raise Interrupted()
        if decision not in ACCEPTS:
            self.item_done(started, "declined")
            return "completed", None, self.final_answer()
        if keep_running:
            self.await_reply(None)  # the command never finishes; only turn/interrupt ends the turn
        self.item_done(started, "completed", exit_code=0)
        return "completed", None, self.final_answer()

    # ---- items and server requests -----------------------------------------------------------------------------

    def next_item_id(self) -> str:
        self.item_count += 1
        return f"fake-item-{self.item_count}"

    def item_done(self, item: dict, status: str, exit_code: int | None = None) -> None:
        done = dict(item, status=status)
        if exit_code is not None:
            done["exitCode"] = exit_code
        self.notify("item/completed", {"item": done, "threadId": self.thread_id, "turnId": self.turn_id,
                                       "completedAtMs": now_ms()})

    def message(self, text: str, phase: str) -> dict:
        item_id = self.next_item_id()
        empty = {"type": "agentMessage", "id": item_id, "text": "", "phase": phase, "memoryCitation": None,
                 "delivery": None, "questions": None}
        self.notify("item/started", {"item": empty, "threadId": self.thread_id, "turnId": self.turn_id,
                                     "startedAtMs": now_ms()})
        for start in range(0, len(text), DELTA_CHUNK):
            self.notify("item/agentMessage/delta", {"threadId": self.thread_id, "turnId": self.turn_id,
                                                    "itemId": item_id, "delta": text[start:start + DELTA_CHUNK]})
        done = dict(empty, text=text)
        self.notify("item/completed", {"item": done, "threadId": self.thread_id, "turnId": self.turn_id,
                                       "completedAtMs": now_ms()})
        return done

    def final_answer(self) -> dict:
        final = self.message(self.answer, "final_answer")
        usage = {"inputTokens": 19078, "cachedInputTokens": 0, "cacheWriteInputTokens": 0, "outputTokens": 44,
                 "reasoningOutputTokens": 0, "totalTokens": 19122}
        self.notify("thread/tokenUsage/updated", {
            "threadId": self.thread_id, "turnId": self.turn_id,
            "tokenUsage": {"total": usage, "last": usage, "modelContextWindow": CONTEXT_WINDOW},
        })
        return final

    def server_request(self, method: str, params: dict) -> dict:
        server_id = next(self.server_ids)
        self.send({"id": server_id, "method": method, "params": params})
        try:
            reply = self.await_reply(server_id)
        except Interrupted:
            self.resolved(server_id)
            raise
        self.resolved(server_id)
        return reply

    def resolved(self, server_id: int) -> None:
        self.notify("serverRequest/resolved", {"threadId": self.thread_id, "requestId": server_id})

    def await_reply(self, server_id: int | None) -> dict:
        """Read frames until the reply to server_id arrives. turn/interrupt ends the wait; EOF closes the process."""
        while True:
            frame = self.read()
            if frame is None:
                raise Closed()
            if "method" not in frame and frame.get("id") == server_id:
                return frame
            if frame.get("method") == "turn/interrupt" and "id" in frame:
                self.result(frame["id"], {})
                raise Interrupted()
            if "method" in frame and "id" in frame:  # a second request during a turn cannot start another turn
                self.error(frame["id"], UNSUPPORTED_CODE, "turn in progress")


def main() -> int:
    mode = os.environ.get("FAKE_CODEX_MODE", "normal")
    if mode not in MODES:
        sys.stderr.write(f"fake_codex: unknown FAKE_CODEX_MODE {mode!r}\n")
        return 2
    fake = Fake(mode, os.environ.get("FAKE_CODEX_LOG", ""), os.environ.get("FAKE_CODEX_ANSWER", ANSWER_DEFAULT))
    fake.run()
    if mode == "stubborn":
        park()
    return 0


if __name__ == "__main__":
    sys.exit(main())
