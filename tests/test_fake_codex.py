"""tests/fakes/fake_codex.py driven through JsonRpcPeer, one scenario per mode.

Each test starts the fake through its launcher wrapper (tests/fakes/launchers/codex) in its own process group, with a
reader thread that feeds the peer line by line, as the real session does. The scenarios assert the message sequence
the bridge will depend on: text answers, approvals (accept, decline, cancel, offered decisions), interrupts while a
command runs and while an approval is pending, resume errors, failed and retried turns, -32601 for unsupported server
requests, EOF exit codes, and the two process-lifetime modes (stubborn, grandchild).
"""

import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from tests.support import TempHome
from tg_bridge.agents.codex.rpc import JsonRpcPeer, RpcError

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "fakes", "launchers", "codex")
WAIT = 10.0
SERVER_METHODS = (
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
)
PROMPT = "Run the shell command: touch probe_ok.txt  Then reply with the single word done."
ALL_MODES = (
    "normal", "command_approval", "decline_offered", "file_approval", "interrupt", "resume_missing", "fail",
    "retry", "unknown_request", "stubborn", "grandchild",
)


def alive(pid: int) -> bool:
    """True while pid runs. A zombie (exited, not yet reaped) counts as gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            state = handle.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


class Driver:
    """One fake process plus a JsonRpcPeer on its pipes. Events are (kind, method, request_id, params) tuples."""

    def __init__(self, mode: str, home: str, cwd: str) -> None:
        self.mode = mode
        self.cwd = cwd
        self.log_path = os.path.join(home, f"frames-{mode}.jsonl")
        self.pid_path = os.path.join(home, f"grandchild-{mode}.pid")
        self.events: queue.Queue = queue.Queue()
        self.reports: list[str] = []
        self.thread_id = ""
        self.turn_id = ""
        self.write_lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.stderr = open(os.path.join(home, f"stderr-{mode}.txt"), "w", encoding="utf-8")
        env = dict(os.environ, FAKE_CODEX_MODE=mode, FAKE_CODEX_LOG=self.log_path, FAKE_CODEX_PIDFILE=self.pid_path)
        self.proc = subprocess.Popen([LAUNCHER, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, text=True, bufsize=1, env=env, start_new_session=True)
        self.peer = JsonRpcPeer(self._write, lambda: threading.current_thread() is self.reader,
                                report=self.reports.append)
        self.peer.on_notification(self._on_notification)
        for method in SERVER_METHODS:
            self.peer.on_request(method, self._on_request_for(method))
        self.reader = threading.Thread(target=self._read_loop, name=f"fake-codex-reader-{mode}", daemon=True)
        self.reader.start()

    # ---- pipes ------------------------------------------------------------------------------------------------

    def _write(self, frame: dict) -> bool:
        line = json.dumps(frame) + "\n"
        with self.write_lock:
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return False
        return True

    def _read_loop(self) -> None:
        for line in self.proc.stdout:
            self.peer.handle_line(json.loads(line))
        self.peer.close()

    def _on_notification(self, method: str, params: dict) -> None:
        self.events.put(("notification", method, None, params))

    def _on_request_for(self, method: str):
        def handler(request_id: int, params: dict) -> None:
            self.events.put(("request", method, request_id, params))
        return handler

    # ---- protocol steps ---------------------------------------------------------------------------------------

    def handshake(self) -> dict:
        result = self.peer.request("initialize", {"clientInfo": {"name": "tg-bridge-test", "title": None,
                                                                  "version": "0.0.1"},
                                                  "capabilities": {"experimentalApi": False,
                                                                   "requestAttestation": False}})
        self.peer.notify("initialized")
        return result

    def start_thread(self, approval: str = "untrusted") -> dict:
        result = self.peer.request("thread/start", {"cwd": self.cwd, "approvalPolicy": approval,
                                                    "approvalsReviewer": "user", "sandbox": "workspace-write",
                                                    "ephemeral": False})
        self.thread_id = result["thread"]["id"]
        return result

    def resume_thread(self, thread_id: str) -> dict:
        return self.peer.request("thread/resume", {"threadId": thread_id, "cwd": self.cwd,
                                                   "approvalPolicy": "untrusted", "sandbox": "workspace-write",
                                                   "excludeTurns": True})

    def start_turn(self, text: str = PROMPT) -> dict:
        result = self.peer.request("turn/start", {"threadId": self.thread_id,
                                                  "input": [{"type": "text", "text": text, "text_elements": []}]})
        self.turn_id = result["turn"]["id"]
        return result

    def interrupt(self) -> dict:
        return self.peer.request("turn/interrupt", {"threadId": self.thread_id, "turnId": self.turn_id})

    def wait_for(self, method: str, kind: str = "notification", timeout: float = WAIT) -> list[tuple]:
        """Return every event up to and including the first (kind, method) event; fail after timeout."""
        deadline = time.monotonic() + timeout
        seen: list[tuple] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out waiting for {kind} {method}; saw {[e[1] for e in seen]}")
            try:
                event = self.events.get(timeout=remaining)
            except queue.Empty:
                continue
            seen.append(event)
            if event[0] == kind and event[1] == method:
                return seen

    def inbound(self) -> list[dict]:
        """Every frame the fake received, in order (from FAKE_CODEX_LOG)."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as handle:
            return [json.loads(line)["in"] for line in handle if line.strip()]

    def inbound_methods(self) -> list[str]:
        return [frame["method"] for frame in self.inbound() if "method" in frame]

    def close_stdin(self) -> None:
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def teardown(self) -> None:
        try:  # the group can outlive the leader (grandchild mode), so kill it even when the leader has exited
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=WAIT)
        if self.reader is not None:
            self.reader.join(timeout=WAIT)
        for pipe in (self.proc.stdin, self.proc.stdout):
            pipe.close()
        self.stderr.close()


def methods_of(events: list[tuple]) -> list[str]:
    return [event[1] for event in events]


def params_of(event: tuple) -> dict:
    return event[3]


def item_completions(events: list[tuple], item_type: str) -> list[dict]:
    return [params_of(e)["item"] for e in events if e[1] == "item/completed"
            and params_of(e).get("item", {}).get("type") == item_type]


class FakeCodexCase(unittest.TestCase):
    """Shared setup: a temp HOME (isolation only, the fake never reads it), a temp cwd, and driver cleanup."""

    def setUp(self) -> None:
        home = TempHome()
        home.__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        self.home = home.home
        self.cwd = tempfile.mkdtemp(prefix="fake-codex-cwd-", dir=self.home)

    def driver(self, mode: str, handshake: bool = True) -> Driver:
        drv = Driver(mode, self.home, self.cwd)
        self.addCleanup(drv.teardown)
        if handshake:
            drv.handshake()
        return drv

    def final_turn(self, drv: Driver) -> tuple[list[tuple], dict]:
        events = drv.wait_for("turn/completed")
        return events, params_of(events[-1])["turn"]


class TextTurns(FakeCodexCase):
    def test_normal_turn_answers_and_exits_zero_on_eof(self) -> None:
        drv = self.driver("normal")
        thread = drv.start_thread()
        self.assertEqual(thread["thread"]["id"], "fake-thread-1")
        self.assertEqual(thread["model"], "gpt-fake")
        turn = drv.start_turn()
        self.assertEqual(turn["turn"]["status"], "inProgress")
        events, done = self.final_turn(drv)
        self.assertIn("item/agentMessage/delta", methods_of(events))
        self.assertIn("thread/tokenUsage/updated", methods_of(events))
        self.assertEqual(done["status"], "completed")
        self.assertIsNone(done["error"])
        self.assertEqual(done["items"][0]["phase"], "final_answer")
        self.assertEqual(done["items"][0]["text"], "done")
        commentary = [i for i in item_completions(events, "agentMessage") if i["phase"] == "commentary"]
        self.assertEqual(len(commentary), 1)
        drv.close_stdin()
        self.assertEqual(drv.proc.wait(timeout=WAIT), 0)
        self.assertEqual(drv.inbound_methods(), ["initialize", "initialized", "thread/start", "turn/start"])

    def test_final_answer_text_comes_from_env(self) -> None:
        drv = self.driver("normal")
        drv.start_thread()
        drv.start_turn()
        self.assertEqual(self.final_turn(drv)[1]["items"][0]["text"], "done")

    def test_unknown_method_gets_minus_32601(self) -> None:
        drv = self.driver("normal")
        with self.assertRaises(RpcError) as raised:
            drv.peer.request("nonexistent/method", {})
        self.assertEqual(raised.exception.code, -32601)

    def test_thread_resume_keeps_the_requested_id(self) -> None:
        drv = self.driver("normal")
        result = drv.resume_thread("fake-thread-9")
        self.assertEqual(result["thread"]["id"], "fake-thread-9")
        self.assertIn("thread/resume", drv.inbound_methods())

    def test_resume_missing_turns_into_an_error_reply(self) -> None:
        drv = self.driver("resume_missing")
        with self.assertRaises(RpcError) as raised:
            drv.resume_thread("fake-thread-gone")
        self.assertEqual(raised.exception.code, -32600)
        self.assertIn("fake-thread-gone", raised.exception.message)
        drv.thread_id = "fake-thread-gone"
        drv.start_turn()
        self.assertEqual(self.final_turn(drv)[1]["status"], "completed")

    def test_fail_mode_ends_the_turn_as_failed_with_an_error(self) -> None:
        drv = self.driver("fail")
        drv.start_thread()
        drv.start_turn()
        _, done = self.final_turn(drv)
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error"]["message"], "fake failure")
        self.assertEqual(done["items"], [])

    def test_retry_mode_sends_one_retry_error_then_answers(self) -> None:
        drv = self.driver("retry")
        drv.start_thread()
        drv.start_turn()
        events, done = self.final_turn(drv)
        errors = [params_of(e) for e in events if e[1] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["willRetry"])
        self.assertEqual(errors[0]["error"]["codexErrorInfo"], {"responseStreamDisconnected": {"httpStatusCode": None}})
        self.assertEqual(errors[0]["turnId"], drv.turn_id)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["items"][0]["text"], "done")

    def test_unknown_server_request_is_refused_with_minus_32601(self) -> None:
        drv = self.driver("unknown_request")
        drv.start_thread()
        drv.start_turn()
        events, done = self.final_turn(drv)
        self.assertNotIn("item/tool/requestUserInput", methods_of(events))  # no handler, so the peer answers
        refusals = [f for f in drv.inbound() if "error" in f and "id" in f]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["error"]["code"], -32601)
        self.assertEqual(refusals[0]["id"], 0)
        self.assertEqual(done["status"], "completed")


class CommandApprovals(FakeCodexCase):
    def test_command_accept_runs_the_command_and_completes(self) -> None:
        drv = self.driver("command_approval")
        drv.start_thread()
        drv.start_turn()
        request = drv.wait_for("item/commandExecution/requestApproval", kind="request")[-1]
        offered = params_of(request)["availableDecisions"]
        self.assertIn("accept", offered)
        self.assertIn("cancel", offered)
        self.assertNotIn("decline", offered)
        self.assertEqual(params_of(request)["proposedExecpolicyAmendment"], ["touch", "probe_ok.txt"])
        drv.peer.respond(request[2], {"decision": "accept"})
        events, done = self.final_turn(drv)
        self.assertIn("serverRequest/resolved", methods_of(events))
        command = item_completions(events, "commandExecution")
        self.assertEqual(command[0]["status"], "completed")
        self.assertEqual(command[0]["exitCode"], 0)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["items"][0]["text"], "done")

    def test_command_decline_lets_the_turn_go_on(self) -> None:
        drv = self.driver("command_approval")
        drv.start_thread()
        drv.start_turn()
        request = drv.wait_for("item/commandExecution/requestApproval", kind="request")
        drv.peer.respond(request[-1][2], {"decision": "decline"})
        events, done = self.final_turn(drv)
        self.assertEqual(item_completions(events, "commandExecution")[0]["status"], "declined")
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["items"][0]["text"], "done")

    def test_command_cancel_aborts_the_turn(self) -> None:
        drv = self.driver("command_approval")
        drv.start_thread()
        drv.start_turn()
        request = drv.wait_for("item/commandExecution/requestApproval", kind="request")
        drv.peer.respond(request[-1][2], {"decision": "cancel"})
        events, done = self.final_turn(drv)
        self.assertEqual(item_completions(events, "commandExecution")[0]["status"], "declined")
        self.assertEqual(done["status"], "interrupted")
        self.assertEqual(done["items"], [])
        finals = [i for i in item_completions(events, "agentMessage") if i["phase"] == "final_answer"]
        self.assertEqual(finals, [])

    def test_decline_offered_mode_lists_decline_and_accept_for_session(self) -> None:
        drv = self.driver("decline_offered")
        drv.start_thread()
        drv.start_turn()
        request = drv.wait_for("item/commandExecution/requestApproval", kind="request")
        offered = params_of(request[-1])["availableDecisions"]
        self.assertIn("decline", offered)
        self.assertIn("acceptForSession", offered)
        drv.peer.respond(request[-1][2], {"decision": "decline"})
        _, done = self.final_turn(drv)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["items"][0]["text"], "done")


class FileApprovals(FakeCodexCase):
    def test_file_change_request_has_no_available_decisions_and_accept_completes(self) -> None:
        drv = self.driver("file_approval")
        drv.start_thread()
        drv.start_turn()
        seen = drv.wait_for("item/fileChange/requestApproval", kind="request")
        request = seen[-1]
        self.assertNotIn("availableDecisions", params_of(request))
        opened = [params_of(e)["item"] for e in seen
                  if e[1] == "item/started" and params_of(e)["item"]["type"] == "fileChange"]
        self.assertEqual(params_of(request)["itemId"], opened[0]["id"])
        drv.peer.respond(request[2], {"decision": "accept"})
        events, done = self.final_turn(drv)
        changed = item_completions(events, "fileChange")
        self.assertEqual(changed[0]["status"], "completed")
        self.assertEqual(changed[0]["changes"][0]["path"], "probe_ok.txt")
        self.assertEqual(done["status"], "completed")

    def test_file_change_cancel_aborts_the_turn(self) -> None:
        drv = self.driver("file_approval")
        drv.start_thread()
        drv.start_turn()
        request = drv.wait_for("item/fileChange/requestApproval", kind="request")[-1]
        drv.peer.respond(request[2], {"decision": "cancel"})
        events, done = self.final_turn(drv)
        self.assertEqual(item_completions(events, "fileChange")[0]["status"], "declined")
        self.assertEqual(done["status"], "interrupted")


class Interrupts(FakeCodexCase):
    def test_interrupt_stops_a_running_command_and_the_item_never_completes(self) -> None:
        drv = self.driver("interrupt")
        drv.start_thread()
        drv.start_turn("Run the shell command: sleep 30  then reply done.")
        request = drv.wait_for("item/commandExecution/requestApproval", kind="request")[-1]
        drv.peer.respond(request[2], {"decision": "accept"})
        drv.wait_for("serverRequest/resolved")
        self.assertEqual(drv.interrupt(), {})
        events, done = self.final_turn(drv)
        self.assertEqual(done["status"], "interrupted")
        self.assertEqual(item_completions(events, "commandExecution"), [])
        self.assertEqual(drv.inbound_methods().count("turn/interrupt"), 1)

    def test_interrupt_while_approval_is_pending_resolves_it_then_interrupts(self) -> None:
        drv = self.driver("interrupt")
        drv.start_thread()
        drv.start_turn("Run the shell command: sleep 30  then reply done.")
        drv.wait_for("item/commandExecution/requestApproval", kind="request")
        self.assertEqual(drv.interrupt(), {})
        events, done = self.final_turn(drv)
        methods = methods_of(events)
        self.assertLess(methods.index("serverRequest/resolved"), methods.index("turn/completed"))
        self.assertEqual(done["status"], "interrupted")


class Lifetimes(FakeCodexCase):
    def test_stubborn_ignores_eof_until_sigterm(self) -> None:
        drv = self.driver("stubborn")
        drv.start_thread()
        drv.start_turn()
        self.assertEqual(self.final_turn(drv)[1]["status"], "completed")
        drv.close_stdin()
        with self.assertRaises(subprocess.TimeoutExpired):
            drv.proc.wait(timeout=0.5)
        drv.proc.terminate()
        self.assertEqual(drv.proc.wait(timeout=WAIT), -signal.SIGTERM)

    def test_grandchild_stays_in_the_group_until_the_group_is_killed(self) -> None:
        drv = self.driver("grandchild")
        drv.start_thread()
        drv.start_turn()
        self.assertEqual(self.final_turn(drv)[1]["status"], "completed")
        with open(drv.pid_path, encoding="utf-8") as handle:
            child_pid = int(handle.read())
        self.assertTrue(alive(child_pid))
        drv.close_stdin()
        self.assertEqual(drv.proc.wait(timeout=WAIT), 0)
        self.assertTrue(alive(child_pid))  # the parent is gone, the child is not
        os.killpg(drv.proc.pid, signal.SIGKILL)  # what the bridge's kill ladder does to the whole group
        deadline = time.monotonic() + WAIT
        while alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(child_pid))

    def test_every_mode_but_stubborn_exits_zero_on_eof(self) -> None:
        for mode in ALL_MODES:
            if mode == "stubborn":
                continue
            with self.subTest(mode=mode):
                drv = self.driver(mode)
                drv.close_stdin()
                self.assertEqual(drv.proc.wait(timeout=WAIT), 0)

    def test_unknown_mode_is_refused_with_rc_2(self) -> None:
        env = dict(os.environ, FAKE_CODEX_MODE="no-such-mode")
        result = subprocess.run([LAUNCHER, "app-server"], input="", capture_output=True, text=True,
                                env=env, timeout=WAIT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no-such-mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
