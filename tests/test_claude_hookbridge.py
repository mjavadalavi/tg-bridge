"""The permission bridge (plan section 5): decision shapes, the socket server, the hook client and the hook settings.

The server and the client talk over a real Unix socket. The socket path must stay short (AF_UNIX allows 108 bytes), so
each test folder is made under /tmp rather than under the scratchpad or the test home. The raw-socket tests send bytes
the client never would, to check the server's refusals. OneShotBridge stands in for a bridge that misbehaves.
"""

import io
import json
import os
import queue
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Mapping
from typing import Any
from unittest import mock

from tg_bridge.agents.claude.hookbridge import (
    ENV_SOCKET,
    HookServer,
    hook_command,
    hook_settings,
    permission_allow,
    permission_deny,
    run_hook_client,
)
from tg_bridge.logging_ import Logger

WAIT = 5.0
TURN_ENDED = "the turn ended"
NOT_REACHABLE_PREFIX = "tg-bridge is not reachable: "
UNREADABLE = "tg-bridge could not read the hook request"
NESTED_DEPTH = 100_000  # far deeper than the JSON parser's recursion limit
ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tg-bridge.py")
PAYLOAD: dict[str, Any] = {
    "session_id": "s-1",
    "transcript_path": "/x/t.jsonl",
    "cwd": "/work",
    "permission_mode": "default",
    "hook_event_name": "PermissionRequest",
    "tool_name": "Bash",
    "tool_input": {"command": "ls"},
    "permission_suggestions": [],
}


def _decision(output: Mapping[str, Any]) -> dict[str, Any]:
    return output["hookSpecificOutput"]["decision"]


def _run_client(stdin_text: str, env: Mapping[str, str], timeout: float | None = None) -> tuple[int, dict]:
    """Runs the hook client on the given stdin text and returns (status, the one JSON line it printed)."""
    out = io.StringIO()
    status = run_hook_client(io.StringIO(stdin_text), out, env, timeout)
    return status, json.loads(out.getvalue())


def _short_folder() -> str:
    """A fresh folder under /tmp, short enough for an AF_UNIX path. The caller removes it."""
    return tempfile.mkdtemp(prefix="tgb-hook-", dir="/tmp")


def _request_threads() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name == "hook-request"}


def _new_request_threads(before: set[threading.Thread]) -> set[threading.Thread]:
    """Waits until the server has started a connection thread that was not in before, then returns those threads."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        new = _request_threads() - before
        if new:
            return new
        time.sleep(0.01)
    raise AssertionError("the server started no connection thread")


class ShapeTests(unittest.TestCase):
    def test_allow_is_the_documented_output(self) -> None:
        self.assertEqual(permission_allow(), {"hookSpecificOutput": {
            "hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}})

    def test_allow_with_an_edited_input_carries_it_in_the_decision(self) -> None:
        output = permission_allow({"command": "ls -la"})
        self.assertEqual(_decision(output), {"behavior": "allow", "updatedInput": {"command": "ls -la"}})

    def test_an_empty_edited_input_is_still_sent(self) -> None:
        self.assertEqual(_decision(permission_allow({})), {"behavior": "allow", "updatedInput": {}})

    def test_deny_carries_the_message_and_no_permission_updates(self) -> None:
        output = permission_deny("not now")
        self.assertEqual(output["hookSpecificOutput"]["hookEventName"], "PermissionRequest")
        self.assertEqual(_decision(output), {"behavior": "deny", "message": "not now"})


class HookSettingsTests(unittest.TestCase):
    def test_settings_install_the_command_for_every_tool_and_round_trip(self) -> None:
        command = ["/usr/bin/python3", "-I", "/opt/my dir/tg-bridge.py", "hook", "permission"]
        data = json.loads(hook_settings(command, 600))
        entry = data["hooks"]["PermissionRequest"][0]
        self.assertEqual(entry["matcher"], "*")
        self.assertEqual(entry["hooks"], [{"type": "command", "command": shlex.join(command), "timeout": 600}])
        self.assertEqual(shlex.split(entry["hooks"][0]["command"]), command)

    def test_settings_are_compact_json_on_one_line(self) -> None:
        settings = hook_settings(["python3", "x.py"], 30)
        self.assertEqual(settings, json.dumps(json.loads(settings), separators=(",", ":")))
        self.assertNotIn("\n", settings)


class HookCommandTests(unittest.TestCase):
    def test_the_command_runs_the_entry_script_with_the_hook_words_last(self) -> None:
        command = hook_command("/usr/bin/python3")
        self.assertEqual(command[:2], ["/usr/bin/python3", "-I"])
        self.assertEqual(command[-2:], ["hook", "permission"])
        self.assertEqual(len(command), 5)

    def test_the_entry_script_is_this_checkouts_tg_bridge_py(self) -> None:
        entry = hook_command("/usr/bin/python3")[2]
        self.assertTrue(os.path.isabs(entry))
        self.assertTrue(os.path.isfile(entry))
        self.assertEqual(os.path.realpath(entry), os.path.realpath(ENTRY))

    def test_the_default_interpreter_is_the_running_one(self) -> None:
        self.assertEqual(hook_command()[0], sys.executable)


class ServerTestCase(unittest.TestCase):
    """A HookServer that queues every request it is handed, on a socket in a fresh short-path folder."""

    def setUp(self) -> None:
        self.dir = _short_folder()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.logger = Logger(os.path.join(self.dir, "bridge.log"))
        self.path = os.path.join(self.dir, "run", "hook.sock")
        self.requests: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.server = self._server(self.path, lambda rid, payload: self.requests.put((rid, payload)))

    def _server(self, path: str, on_request: Callable[[str, dict], None]) -> HookServer:
        server = HookServer(path, on_request, logger=self.logger)
        server.start()
        self.addCleanup(server.close)
        return server

    def _next_request(self) -> tuple[str, dict]:
        return self.requests.get(timeout=WAIT)

    def _client(self, payload: object) -> tuple[threading.Thread, io.StringIO]:
        """Runs the hook client in a thread, as the TUI waits on its hook process, against this test's socket."""
        out = io.StringIO()
        thread = threading.Thread(target=run_hook_client,
                                  args=(io.StringIO(json.dumps(payload)), out, {ENV_SOCKET: self.path}),
                                  daemon=True)
        thread.start()
        return thread, out

    @staticmethod
    def _finish(thread: threading.Thread, out: io.StringIO) -> dict:
        thread.join(WAIT)
        if thread.is_alive():
            raise AssertionError("the hook client did not finish")
        return json.loads(out.getvalue())

    def _raw_exchange(self, data: bytes) -> dict:
        """Sends raw bytes to the socket and returns the one JSON line that comes back."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(WAIT)
            sock.connect(self.path)
            sock.sendall(data)
            reply = b""
            while b"\n" not in reply:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                reply += chunk
        return json.loads(reply)


class RoundTripTests(ServerTestCase):
    def test_allow_reaches_the_client_as_the_bridge_sent_it(self) -> None:
        thread, out = self._client(PAYLOAD)
        request_id, payload = self._next_request()
        self.assertEqual(payload, PAYLOAD)
        self.assertRegex(request_id, r"^[0-9a-f]{32}$")
        self.assertTrue(self.server.answer(request_id, permission_allow()))
        self.assertEqual(self._finish(thread, out), permission_allow())

    def test_allow_with_an_updated_input_reaches_the_client(self) -> None:
        thread, out = self._client(PAYLOAD)
        request_id, _ = self._next_request()
        self.server.answer(request_id, permission_allow({"command": "ls -la"}))
        self.assertEqual(self._finish(thread, out), permission_allow({"command": "ls -la"}))

    def test_deny_reaches_the_client_with_its_message(self) -> None:
        thread, out = self._client(PAYLOAD)
        request_id, _ = self._next_request()
        self.server.answer(request_id, permission_deny("the owner said no"))
        self.assertEqual(self._finish(thread, out), permission_deny("the owner said no"))

    def test_pending_count_follows_the_request_until_it_is_answered(self) -> None:
        thread, out = self._client(PAYLOAD)
        request_id, _ = self._next_request()
        self.assertEqual(self.server.pending_count(), 1)
        self.server.answer(request_id, permission_allow())
        self._finish(thread, out)
        self.assertEqual(self.server.pending_count(), 0)

    def test_each_request_gets_its_own_id(self) -> None:
        clients = [self._client(PAYLOAD) for _ in range(2)]
        ids = {self._next_request()[0] for _ in range(2)}
        self.assertEqual(len(ids), 2)
        self.assertEqual(set(self.server.cancel_all("done")), ids)
        for thread, out in clients:
            self._finish(thread, out)

    def test_cancel_all_returns_the_ids_it_denied_and_each_client_gets_the_deny(self) -> None:
        clients = [self._client(PAYLOAD) for _ in range(2)]
        ids = [self._next_request()[0] for _ in range(2)]
        cancelled = self.server.cancel_all("the turn ended")
        self.assertEqual(sorted(cancelled), sorted(ids))
        self.assertEqual(self.server.pending_count(), 0)
        for thread, out in clients:
            self.assertEqual(self._finish(thread, out), permission_deny("the turn ended"))

    def test_main_only_cancel_leaves_a_request_from_an_agent_open(self) -> None:
        main = self._client(PAYLOAD)
        main_id = self._next_request()[0]
        agent = self._client({**PAYLOAD, "agent_id": "agent-1"})
        agent_id = self._next_request()[0]
        self.assertEqual(self.server.cancel_all("done", main_only=True), [main_id])
        self.assertEqual(self._finish(*main), permission_deny("done"))
        self.assertEqual(self.server.pending_count(), 1)
        self.assertTrue(self.server.answer(agent_id, permission_allow()))
        self.assertEqual(self._finish(*agent), permission_allow())

    def test_answering_an_unknown_id_is_refused(self) -> None:
        self.assertFalse(self.server.answer("0" * 32, permission_allow()))

    def test_answering_twice_is_refused_the_second_time(self) -> None:
        thread, out = self._client(PAYLOAD)
        request_id, _ = self._next_request()
        self.assertTrue(self.server.answer(request_id, permission_allow()))
        self.assertFalse(self.server.answer(request_id, permission_deny("late")))
        self.assertEqual(self._finish(thread, out), permission_allow())

    def test_a_handler_failure_is_answered_with_a_deny_at_once(self) -> None:
        def failing(request_id: str, payload: dict) -> None:
            raise RuntimeError("boom")

        server = self._server(os.path.join(self.dir, "fail.sock"), failing)
        status, output = _run_client(json.dumps(PAYLOAD), {ENV_SOCKET: server.path})
        self.assertEqual(status, 0)
        self.assertEqual(_decision(output), {
            "behavior": "deny", "message": "tg-bridge could not handle the permission request"})
        self.assertEqual(server.pending_count(), 0)

    def test_input_the_bridge_cannot_read_is_denied_without_a_callback(self) -> None:
        self.assertEqual(self._raw_exchange(b"not json\n"), permission_deny(UNREADABLE))
        self.assertEqual(self._raw_exchange(b"[1, 2]\n"), permission_deny(UNREADABLE))
        self.assertTrue(self.requests.empty())
        self.assertEqual(self.server.pending_count(), 0)

    def test_a_nested_payload_is_denied_and_the_server_keeps_serving(self) -> None:
        deep = b"[" * NESTED_DEPTH + b"]" * NESTED_DEPTH + b"\n"
        self.assertEqual(self._raw_exchange(deep), permission_deny(UNREADABLE))
        thread, out = self._client(PAYLOAD)
        request_id, payload = self._next_request()
        self.assertEqual(payload, PAYLOAD)
        self.server.answer(request_id, permission_allow())
        self.assertEqual(self._finish(thread, out), permission_allow())


class LifecycleTests(ServerTestCase):
    def test_close_denies_what_is_pending_and_removes_the_socket(self) -> None:
        thread, out = self._client(PAYLOAD)
        self._next_request()
        self.server.close()
        self.assertEqual(self._finish(thread, out), permission_deny(TURN_ENDED))
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self.server.pending_count(), 0)

    def test_close_is_idempotent(self) -> None:
        self.server.close()
        self.server.close()
        self.assertFalse(os.path.exists(self.path))

    def test_close_before_start_binds_nothing_and_a_later_start_stays_closed(self) -> None:
        path = os.path.join(self.dir, "late", "hook.sock")
        server = HookServer(path, lambda rid, payload: None, logger=self.logger)
        server.close()
        server.start()
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(os.path.dirname(path)))

    def test_close_after_a_failed_start_removes_the_bound_file(self) -> None:
        path = os.path.join(self.dir, "failed", "hook.sock")
        server = HookServer(path, lambda rid, payload: None, logger=self.logger)
        with mock.patch("tg_bridge.agents.claude.hookbridge.os.chmod", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                server.start()
        self.assertTrue(os.path.exists(path))  # bound before the failure, so the file is there
        server.close()
        self.assertFalse(os.path.exists(path))

    def test_the_client_leaving_drops_its_pending_request(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.path)
            sock.sendall(json.dumps(PAYLOAD).encode("utf-8") + b"\n")
            request_id, _ = self._next_request()
            self.assertEqual(self.server.pending_count(), 1)
        deadline = time.monotonic() + WAIT
        while self.server.pending_count() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.server.pending_count(), 0)
        self.assertFalse(self.server.answer(request_id, permission_allow()))

    def test_a_silent_client_is_closed_by_close_and_its_thread_ends(self) -> None:
        before = _request_threads()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(WAIT)
            sock.connect(self.path)
            threads = _new_request_threads(before)
            self.server.close()
            data = sock.makefile("rb").read()  # the deny line, then end-of-file once the server closed the socket
        self.assertEqual(json.loads(data), permission_deny(UNREADABLE))
        for thread in threads:
            thread.join(WAIT)
            self.assertFalse(thread.is_alive())

    def test_a_partial_line_is_denied_once_the_read_times_out(self) -> None:
        with mock.patch("tg_bridge.agents.claude.hookbridge.REQUEST_READ_SECONDS", 0.2):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(self.path)
                sock.sendall(b'{"tool_name": ')  # no newline, and the client never closes
                sock.settimeout(WAIT)
                reply = sock.makefile("rb").readline()
        self.assertEqual(json.loads(reply), permission_deny(UNREADABLE))

    def test_start_makes_a_private_folder_and_socket(self) -> None:
        nested = os.path.join(self.dir, "a", "b", "hook.sock")
        self._server(nested, lambda rid, payload: None)
        self.assertTrue(stat.S_ISSOCK(os.stat(nested).st_mode))
        self.assertEqual(stat.S_IMODE(os.stat(nested).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(nested)).st_mode), 0o700)

    def test_start_replaces_a_stale_file_at_the_path(self) -> None:
        stale = os.path.join(self.dir, "stale.sock")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("left over by a crash")
        self._server(stale, lambda rid, payload: None)
        self.assertTrue(stat.S_ISSOCK(os.stat(stale).st_mode))


class OneShotBridge:
    """A stand-in bridge: it takes one connection, records the line it reads, then sends reply (or nothing) and hangs
    up, or with hold=True keeps the connection open until close(), so the client can time out."""

    def __init__(self, path: str, reply: bytes | None, hold: bool = False) -> None:
        self.path = path
        self.received: list[bytes] = []
        self._release = threading.Event()
        self._hold = hold
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen()
        self._thread = threading.Thread(target=self._serve, args=(reply,), daemon=True)
        self._thread.start()

    def _serve(self, reply: bytes | None) -> None:
        conn, _ = self._sock.accept()
        with conn:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            self.received.append(data)
            if reply is not None:
                conn.sendall(reply)
            if self._hold:
                self._release.wait(WAIT)

    def close(self) -> None:
        self._release.set()
        self._thread.join(WAIT)
        self._sock.close()


class ClientTests(unittest.TestCase):
    """Every way the hook side can fail prints one deny line and returns 0, so the TUI never waits on its own dialog."""

    def setUp(self) -> None:
        self.dir = _short_folder()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "hook.sock")
        self.stdin = json.dumps(PAYLOAD)

    def _assert_unreachable(self, status: int, output: dict, reason: str) -> None:
        self.assertEqual(status, 0)
        self.assertEqual(_decision(output), {"behavior": "deny", "message": NOT_REACHABLE_PREFIX + reason})

    def test_no_socket_in_the_environment_denies(self) -> None:
        status, output = _run_client(self.stdin, {})
        self._assert_unreachable(status, output, f"no {ENV_SOCKET} in the environment")

    def test_a_missing_socket_file_denies_with_the_system_reason(self) -> None:
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        self.assertEqual(status, 0)
        self.assertEqual(_decision(output)["behavior"], "deny")
        self.assertEqual(_decision(output)["message"], NOT_REACHABLE_PREFIX + "No such file or directory")

    def test_a_socket_path_that_refuses_connections_denies(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:  # a plain file is not a listening socket
            handle.write("")
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        self.assertEqual(status, 0)
        self.assertEqual(_decision(output)["behavior"], "deny")
        self.assertTrue(_decision(output)["message"].startswith(NOT_REACHABLE_PREFIX))

    def test_stdin_that_is_not_json_denies_without_connecting(self) -> None:
        status, output = _run_client("{not json", {ENV_SOCKET: self.path})
        self._assert_unreachable(status, output, "the hook input is not JSON")

    def test_stdin_that_is_not_an_object_denies(self) -> None:
        status, output = _run_client("[1, 2]", {ENV_SOCKET: self.path})
        self._assert_unreachable(status, output, "the hook input is not a JSON object")

    def test_a_bridge_that_hangs_up_without_answering_denies(self) -> None:
        bridge = OneShotBridge(self.path, reply=None)
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        bridge.close()
        self._assert_unreachable(status, output, "no answer from the bridge")

    def test_a_bridge_that_never_answers_times_out_into_a_deny(self) -> None:
        bridge = OneShotBridge(self.path, reply=None, hold=True)
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path}, timeout=0.2)
        bridge.close()
        self._assert_unreachable(status, output, "no answer from the bridge")

    def test_a_bridge_that_answers_with_junk_denies(self) -> None:
        bridge = OneShotBridge(self.path, reply=b"oops\n")
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        bridge.close()
        self._assert_unreachable(status, output, "the bridge answer is not JSON")

    def test_stdin_nested_too_deep_denies_instead_of_crashing(self) -> None:
        deep = "[" * NESTED_DEPTH + "]" * NESTED_DEPTH
        status, output = _run_client(deep, {ENV_SOCKET: self.path})
        self._assert_unreachable(status, output, "the hook input is nested too deeply")

    def test_a_bridge_answer_nested_too_deep_denies(self) -> None:
        bridge = OneShotBridge(self.path, reply=("[" * NESTED_DEPTH + "]" * NESTED_DEPTH + "\n").encode("utf-8"))
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        bridge.close()
        self._assert_unreachable(status, output, "the bridge answer is nested too deeply")

    def test_a_bridge_that_answers_with_a_list_denies(self) -> None:
        bridge = OneShotBridge(self.path, reply=b"[]\n")
        status, output = _run_client(self.stdin, {ENV_SOCKET: self.path})
        bridge.close()
        self._assert_unreachable(status, output, "the bridge answer is not a JSON object")

    def test_the_input_goes_over_as_one_compact_line_and_the_answer_is_relayed(self) -> None:
        answer = permission_allow({"command": "ls"})
        bridge = OneShotBridge(self.path, reply=(json.dumps(answer) + "\n").encode("utf-8"))
        status, output = _run_client(json.dumps(PAYLOAD, indent=2), {ENV_SOCKET: self.path})
        bridge.close()
        self.assertEqual((status, output), (0, answer))
        self.assertEqual(bridge.received, [json.dumps(PAYLOAD, separators=(",", ":")).encode("utf-8") + b"\n"])


class EntryScriptTests(ServerTestCase):
    def test_the_real_hook_command_round_trips_through_the_entry_script(self) -> None:
        def answer_next() -> None:
            request_id, _ = self._next_request()
            self.server.answer(request_id, permission_allow({"command": "pwd"}))

        answerer = threading.Thread(target=answer_next, daemon=True)
        answerer.start()
        env = {ENV_SOCKET: self.path, "PATH": os.environ.get("PATH", "")}
        proc = subprocess.run(hook_command(), input=json.dumps(PAYLOAD), capture_output=True, text=True, env=env,
                              timeout=60, check=False)
        answerer.join(WAIT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), permission_allow({"command": "pwd"}))


if __name__ == "__main__":
    unittest.main()
