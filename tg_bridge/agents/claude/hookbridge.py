"""The permission bridge of the TUI transport (plan section 5): one Unix socket per session and the hook command that
talks to it.

Claude Code's interactive TUI asks the user before it runs a tool. With a PermissionRequest hook installed through
--settings, the TUI runs our command instead and waits for the decision on its stdout. The command is
`tg-bridge hook permission`: it reads the hook's JSON from stdin, sends it over the session's socket, and prints the
decision the bridge sends back. HookServer is the bridge side. It keeps each request pending until the bridge answers
it (a Telegram card) or the turn ends, so the TUI waits on the bridge and never on its own dialog.

Every failure on the hook side becomes a deny, never silence: a hook that prints nothing leaves the TUI's dialog open
with nobody there to press a key. Only the decision shapes below are emitted. An updatedPermissions field was not
verified against the live TUI, and an unexpected field could leave its dialog hanging, so it is never sent.
"""

import json
import os
import select
import shlex
import socket
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

from tg_bridge.logging_ import Logger

ENV_SOCKET = "TG_BRIDGE_HOOK_SOCK"
HOOK_EVENT = "PermissionRequest"
MATCH_ALL = "*"
HOOK_WORDS = ("hook", "permission")
ENTRY_SCRIPT = "tg-bridge.py"
MAX_LINE_BYTES = 1 << 20
READ_CHUNK = 65536
REQUEST_READ_SECONDS = 10  # how long a new connection may take to send its request line
DIR_MODE = 0o700
SOCKET_MODE = 0o600
POLL_SECONDS = 0.5
TURN_ENDED = "the turn ended"
BAD_PAYLOAD = "tg-bridge could not read the hook request"
HANDLER_FAILED = "tg-bridge could not handle the permission request"
NOT_REACHABLE = "tg-bridge is not reachable: {reason}"


def permission_allow(updated_input: dict | None = None) -> dict:
    """The hook output that allows the tool call. updated_input, when given, replaces the tool input (for an edited
    command)."""
    decision: dict = {"behavior": "allow"}
    if updated_input is not None:
        decision["updatedInput"] = updated_input
    return _output(decision)


def permission_deny(message: str) -> dict:
    """The hook output that refuses the tool call. The TUI shows the message to the model."""
    return _output({"behavior": "deny", "message": message})


def _output(decision: dict) -> dict:
    return {"hookSpecificOutput": {"hookEventName": HOOK_EVENT, "decision": decision}}


def hook_settings(command: Sequence[str], timeout_seconds: int) -> str:
    """The compact --settings JSON that installs command as the PermissionRequest hook for every tool. The TUI runs the
    command through a shell, so the argv is quoted with shlex."""
    entry = {"type": "command", "command": shlex.join(command), "timeout": timeout_seconds}
    settings = {"hooks": {HOOK_EVENT: [{"matcher": MATCH_ALL, "hooks": [entry]}]}}
    return json.dumps(settings, separators=(",", ":"))


def hook_command(python: str = sys.executable) -> list[str]:
    """The argv of the hook: the interpreter, -I, this checkout's tg-bridge.py and the `hook permission` words. -I keeps
    the user's site packages and PYTHON* variables out; the entry script puts its own folder on sys.path itself."""
    return [python, "-I", _entry_script(), *HOOK_WORDS]


def _entry_script() -> str:
    """tg-bridge.py in the checkout that holds the tg_bridge package. This file is
    tg_bridge/agents/claude/hookbridge.py, so the checkout is four directories up."""
    here = os.path.realpath(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(root, ENTRY_SCRIPT)


class _Pending:
    """One request waiting for its answer. The connection thread waits on event; answer() sets response before event.
    agent is True when the hook payload names the agent that asked (a subagent), not the main turn."""

    def __init__(self, agent: bool = False) -> None:
        self.event = threading.Event()
        self.response: dict | None = None
        self.agent = agent


class HookServer:
    """The bridge side of one session's permission socket. Every method is safe to call from any thread.

    on_request(request_id, payload) runs on the connection's own thread, so it may block on the bridge's work; the
    request stays pending until answer() gets the id, or cancel_all() or close() denies it. An exception from on_request
    is logged and answered with a deny, so the TUI never waits for a reply that will not come.
    """

    def __init__(self, path: str, on_request: Callable[[str, dict], None], *, logger: Logger) -> None:
        self.path = path
        self._on_request = on_request
        self._logger = logger
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._conns: set[socket.socket] = set()  # every accepted connection until its thread is done
        self._sock: socket.socket | None = None
        self._closed = False

    def start(self) -> None:
        """Binds the socket and starts accepting. The folder is made 0700, a stale file at the path is replaced and the
        socket file is 0600, so only this user can reach the bridge. After close() it binds nothing."""
        with self._lock:  # held across the bind, so a close() cannot land between the check and the socket
            if self._closed:
                return
            os.makedirs(os.path.dirname(self.path), mode=DIR_MODE, exist_ok=True)
            if os.path.lexists(self.path):
                os.unlink(self.path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(self.path)
                os.chmod(self.path, SOCKET_MODE)
                sock.listen()
            except OSError:
                sock.close()
                raise
            self._sock = sock
        threading.Thread(target=self._accept_loop, args=(sock,), name="hook-accept", daemon=True).start()

    def _accept_loop(self, sock: socket.socket) -> None:
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return  # close() shut the listening socket down
            with self._lock:  # tracked before its thread starts, so close() reaches a connection that sends nothing
                if self._closed:
                    conn.close()
                    return
                self._conns.add(conn)
            threading.Thread(target=self._serve, args=(conn,), name="hook-request", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.settimeout(REQUEST_READ_SECONDS)  # a client that never sends its line cannot hold this thread
                payload = _read_payload(conn)
                if payload is None:
                    self._reply(conn, permission_deny(BAD_PAYLOAD))
                    return
                request_id = uuid.uuid4().hex
                # agent_id in the real TUI's hook payload is an unverified assumption (the tests send it by hand). When it
                # is absent, an interrupt cancels every request, and each one is still withdrawn.
                pending = _Pending(agent=bool(payload.get("agent_id")))
                with self._lock:
                    closed = self._closed
                    if not closed:
                        self._pending[request_id] = pending
                if closed:
                    self._reply(conn, permission_deny(TURN_ENDED))
                    return
                try:
                    self._on_request(request_id, payload)
                except Exception as exc:  # the TUI is waiting for an answer, so a failure is a deny, not silence
                    self._logger.log(f"hook: on_request failed for {request_id}: {exc!r}")
                    self.answer(request_id, permission_deny(HANDLER_FAILED))
                response = self._wait(conn, request_id, pending)
                if response is not None:
                    self._reply(conn, response)
        finally:
            with self._lock:
                self._conns.discard(conn)

    def _wait(self, conn: socket.socket, request_id: str, pending: _Pending) -> dict | None:
        """Blocks until the request is answered. None when the client left first: the request is then dropped."""
        while not pending.event.wait(POLL_SECONDS):
            if _peer_gone(conn):
                with self._lock:
                    if self._pending.pop(request_id, None) is not None:
                        self._logger.log(f"hook: client left before the answer for {request_id}")
                        return None
                break  # answered in the meantime: answer() sets the response under the same lock
        return pending.response

    def _reply(self, conn: socket.socket, response: dict) -> None:
        line = json.dumps(response, separators=(",", ":")) + "\n"
        try:
            conn.sendall(line.encode("utf-8"))
        except OSError as exc:
            self._logger.log(f"hook: could not send the answer: {exc}")

    def answer(self, request_id: str, response: dict) -> bool:
        """Sends the hook output for a pending request. False when the id is unknown or was already answered."""
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                return False
            pending.response = response
            pending.event.set()
            return True

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def cancel_all(self, message: str, *, main_only: bool = False) -> list[str]:
        """Denies the pending requests with message and returns their ids, in the order they came. main_only leaves
        open the requests whose payload names an agent (a background subagent), so only the main turn's are denied."""
        response = permission_deny(message)
        with self._lock:
            cancelled = [rid for rid, pending in self._pending.items() if not (main_only and pending.agent)]
            waiting = [self._pending.pop(rid) for rid in cancelled]
            for pending in waiting:
                pending.response = response
                pending.event.set()
        return cancelled

    def close(self) -> None:
        """Stops accepting, denies what is still pending, wakes every connection still reading its request and removes
        the socket file. A connection's own thread closes its socket once its deny is sent: closing it from here would
        race that deny. Safe whether or not start() ran, including after a start() that failed once the file was bound.
        Calling it twice is harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sock, self._sock = self._sock, None
            conns = list(self._conns)
        self.cancel_all(TURN_ENDED)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RD)  # the blocked read returns at once; the write side stays open for the deny
            except OSError:  # its thread already finished and closed it
                pass
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # wakes the accept loop on Linux, where close() alone does not
            except OSError:
                pass
            sock.close()
        try:
            os.unlink(self.path)
        except OSError:  # no file: start() never bound it, or it is already gone
            pass


def _read_line(conn: socket.socket) -> bytes | None:
    """The bytes up to the first newline, without it. None when the peer closes first, the line is over MAX_LINE_BYTES
    or the read fails."""
    data = bytearray()
    try:
        while b"\n" not in data:
            chunk = conn.recv(READ_CHUNK)
            if not chunk:
                return None
            data += chunk
            if len(data) > MAX_LINE_BYTES:
                return None
    except OSError:
        return None
    return bytes(data.split(b"\n", 1)[0])


def _read_payload(conn: socket.socket) -> dict | None:
    """The one JSON object the hook client sends. None when the line is missing, not JSON or not an object."""
    line = _read_line(conn)
    if line is None:
        return None
    try:
        value = json.loads(line)
    except (ValueError, RecursionError):  # RecursionError: nested deeper than the parser allows
        return None
    return value if isinstance(value, dict) else None


def _peer_gone(conn: socket.socket) -> bool:
    """True when the client closed its end. The client sends one line and then only waits, so a readable socket that
    holds no data is an end-of-file."""
    try:
        readable, _, _ = select.select([conn], [], [], 0)
        return bool(readable) and conn.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class _HookClientError(Exception):
    """The bridge could not be reached or did not answer. The text is the reason shown to the model."""


def run_hook_client(stdin: TextIO, stdout: TextIO, env: Mapping[str, str], timeout: float | None = None) -> int:
    """`tg-bridge hook permission`: forwards the hook's JSON to the session socket and prints the decision on one line.

    The answer is the bridge's own output. Any failure (no socket in env, nothing listening, bad JSON, no answer)
    prints a deny that names the reason instead, so the TUI never waits on its own dialog. Always returns 0.
    """
    try:
        response = _ask_bridge(_read_input(stdin), env.get(ENV_SOCKET, ""), timeout)
    except _HookClientError as exc:
        response = permission_deny(NOT_REACHABLE.format(reason=exc))
    stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    stdout.flush()
    return 0


def _read_input(stdin: TextIO) -> str:
    try:
        return stdin.read()
    except (OSError, ValueError) as exc:
        raise _HookClientError(f"cannot read the hook input ({exc})") from None


def _decode(text: str | bytes, what: str) -> object:
    """The JSON value in text. Input nested too deep for the parser is refused like input that is not JSON."""
    try:
        return json.loads(text)
    except RecursionError:
        raise _HookClientError(f"{what} is nested too deeply") from None
    except ValueError:
        raise _HookClientError(f"{what} is not JSON") from None


def _ask_bridge(payload_text: str, path: str, timeout: float | None) -> dict:
    """Sends the hook input as one compact line to path and returns the decoded answer."""
    if not path:
        raise _HookClientError(f"no {ENV_SOCKET} in the environment")
    payload = _decode(payload_text, "the hook input")
    if not isinstance(payload, dict):
        raise _HookClientError("the hook input is not a JSON object")
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with sock:
        sock.settimeout(timeout)
        try:
            sock.connect(path)
            sock.sendall(line.encode("utf-8"))
        except OSError as exc:
            raise _HookClientError(exc.strerror or type(exc).__name__) from None
        reply = _read_line(sock)
    if reply is None:
        raise _HookClientError("no answer from the bridge")
    answer = _decode(reply, "the bridge answer")
    if not isinstance(answer, dict):
        raise _HookClientError("the bridge answer is not a JSON object")
    return answer
