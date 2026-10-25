"""JsonRpcPeer: JSON-RPC over the codex app-server stdio pipe (int ids, pending calls, server requests, -32601).

Wire format (scratchpad PROTOCOL-codex.md A.0): objects with no "jsonrpc" field, one per line. Our requests carry int
ids counted from 1. The server's requests carry its own int ids and must be answered with respond() or error(). A
message with a method and no id is a notification. A message with an id and a result or error answers one of ours.
Every notification's emittedAtMs is dropped before the callback sees the params.

The peer never reads stdout itself. The owning session's reader thread hands each parsed line to handle_line(), which
runs the callbacks in arrival order on that thread. request() refuses to run on the reader thread for the same reason as
ControlChannel: the reply can only be read by that thread, so waiting there would hang until the timeout.

Each server request is answered at most once: respond() and error() claim the id under the lock first. A handler that
raises gets an internal error reply unless it had already answered. Unregistered server methods get -32601 at once.
"""

import itertools
import threading
from collections.abc import Callable, Mapping
from typing import Any

Message = dict[str, Any]
WriteFn = Callable[[Message], bool]
NotificationHandler = Callable[[str, dict[str, Any]], None]
RequestHandler = Callable[[int, dict[str, Any]], None]
ReportFn = Callable[[str], None]

DEFAULT_TIMEOUT = 30.0
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603
UNSUPPORTED_MESSAGE = "unsupported"
EMITTED_AT_KEY = "emittedAtMs"


def _ignore(_message: str) -> None:
    return None


class RpcError(RuntimeError):
    """The server answered one of our requests with an error object."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"rpc error {code}: {message}")
        self.code = code
        self.message = message


class PeerClosed(RuntimeError):
    """The peer is closed or stdin is gone, so the call cannot be sent or will never get a reply."""


class _Slot:
    """One pending call. The reader thread sets result or error and the event; close() sets closed and the event."""

    __slots__ = ("event", "result", "error", "closed")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: RpcError | None = None
        self.closed = False


def _params_of(message: Mapping[str, Any]) -> dict[str, Any]:
    """The params object without emittedAtMs. Anything that is not an object becomes an empty dict."""
    params = message.get("params")
    if not isinstance(params, dict):
        return {}
    return {key: value for key, value in params.items() if key != EMITTED_AT_KEY}


class JsonRpcPeer:
    """One end of a JSON-RPC connection.

    write_line takes one frame dict and returns False when stdin is closed (ProcessSession.write fits). is_reader_thread
    tells request() whether it runs on the reader thread. report receives one line for each callback that failed.
    All state is instance-owned. The lock covers dictionaries and flags only, never a write or a wait.
    """

    def __init__(self, write_line: WriteFn, is_reader_thread: Callable[[], bool], *,
                 report: ReportFn = _ignore) -> None:
        self._write = write_line
        self._is_reader_thread = is_reader_thread
        self._report = report
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, _Slot] = {}
        self._open_requests: set[int] = set()
        self._request_handlers: dict[str, RequestHandler] = {}
        self._notification_handler: NotificationHandler | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    # ---- registration ------------------------------------------------------------------------------------------

    def on_notification(self, handler: NotificationHandler) -> None:
        """Set the one callback for every notification: handler(method, params)."""
        with self._lock:
            self._notification_handler = handler

    def on_request(self, method: str, handler: RequestHandler) -> None:
        """Route server requests for method to handler(request_id, params). The handler must answer, now or later."""
        with self._lock:
            self._request_handlers[method] = handler

    # ---- outgoing ----------------------------------------------------------------------------------------------

    def request(self, method: str, params: Mapping[str, Any] | None = None,
                timeout: float = DEFAULT_TIMEOUT) -> Any:
        """Send a request and return its result.

        Raises RuntimeError on the reader thread. Raises PeerClosed when the peer is closed, stdin is gone, or close()
        comes before the reply. Raises RpcError when the server answers with an error. Raises TimeoutError when no
        reply comes within timeout seconds; a late reply is then ignored.
        """
        if self._is_reader_thread():
            raise RuntimeError(f"{method}: request() on the reader thread would wait for itself")
        slot = _Slot()
        with self._lock:
            if self._closed:
                raise PeerClosed(f"{method}: peer closed")
            request_id = next(self._ids)
            self._pending[request_id] = slot
        try:
            if not self._write({"id": request_id, "method": method, "params": dict(params or {})}):
                raise PeerClosed(f"{method}: stdin closed")
            if not slot.event.wait(timeout):
                raise TimeoutError(f"{method} timed out after {timeout}s")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        if slot.closed:
            raise PeerClosed(f"{method}: peer closed before reply")
        if slot.error is not None:
            raise slot.error
        return slot.result

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> bool:
        """Send a notification (no reply expected). False when the peer is closed or stdin is gone."""
        frame: Message = {"method": method}
        if params is not None:
            frame["params"] = dict(params)
        return self._send(frame)

    def respond(self, request_id: int, result: Any) -> bool:
        """Answer a server request. False when that id is not open (never seen, or already answered)."""
        if not self._claim(request_id):
            return False
        return self._send({"id": request_id, "result": result})

    def error(self, request_id: int, code: int, message: str) -> bool:
        """Answer a server request with an error object. Same claim rule as respond()."""
        if not self._claim(request_id):
            return False
        return self._send({"id": request_id, "error": {"code": code, "message": message}})

    def close(self) -> None:
        """Fail every pending request and forget open server requests. Safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._pending.values())
            self._open_requests.clear()
            for slot in slots:
                slot.closed = True
                slot.event.set()

    # ---- incoming (reader thread) ------------------------------------------------------------------------------

    def handle_line(self, message: Mapping[str, Any]) -> None:
        """Route one parsed message from stdout. Messages that fit no shape are ignored."""
        if not isinstance(message, Mapping):
            return
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                self._on_server_request(message["id"], method, _params_of(message))
            else:
                self._on_notification(method, _params_of(message))
        elif "id" in message:
            self._on_reply(message)

    def _on_notification(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            handler = self._notification_handler
        if handler is None:
            return
        try:
            handler(method, params)
        except Exception as exc:  # the reader thread must survive a bad callback
            self._report(f"notification {method} handler failed: {exc!r}")

    def _on_server_request(self, request_id: Any, method: str, params: dict[str, Any]) -> None:
        if type(request_id) is not int:
            self._report(f"server request {method} has a non-integer id; ignored")
            return
        with self._lock:
            handler = self._request_handlers.get(method)
            if handler is not None and not self._closed:
                self._open_requests.add(request_id)
        if handler is None:
            self._send({"id": request_id, "error": {"code": METHOD_NOT_FOUND, "message": UNSUPPORTED_MESSAGE}})
            return
        try:
            handler(request_id, params)
        except Exception as exc:
            self._report(f"server request {method} handler failed: {exc!r}")
            self.error(request_id, INTERNAL_ERROR, "handler failed")

    def _on_reply(self, message: Mapping[str, Any]) -> None:
        reply_id = message["id"]
        if type(reply_id) is not int:
            return
        with self._lock:
            slot = self._pending.get(reply_id)
            if slot is None or slot.closed or slot.event.is_set():
                return
            error = message.get("error")
            if error is not None:
                code = error.get("code") if isinstance(error, Mapping) else None
                text = error.get("message") if isinstance(error, Mapping) else error
                slot.error = RpcError(code if type(code) is int else INTERNAL_ERROR, str(text))
            else:
                slot.result = message.get("result")
            slot.event.set()

    # ---- helpers -----------------------------------------------------------------------------------------------

    def _claim(self, request_id: int) -> bool:
        with self._lock:
            if request_id not in self._open_requests:
                return False
            self._open_requests.discard(request_id)
        return True

    def _send(self, frame: Message) -> bool:
        with self._lock:
            if self._closed:
                return False
        return self._write(frame)
