"""Claude stream-json frames and ControlChannel (request_id correlation, timeout, reader-thread guard).

Frame builders are pure functions that return the dict a session writes as one JSON line. Their key order and values
match the live TopicSession byte for byte (tests/test_claude_protocol.py checks this against the live file), so the
CLI sees exactly what it saw before the split. See scratchpad PROTOCOL.md for the wire format.

ControlChannel sends control_request frames and matches control_response frames back to the waiting caller by
request_id. It never reads stdout itself: the owning session's reader thread passes each control_response to
deliver(). A caller on the reader thread would wait for a reply that only that thread can read, so request() refuses
to run there instead of deadlocking until the timeout.
"""

import itertools
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

Frame = dict[str, Any]
WriteFn = Callable[[Frame], bool]
DENY_MESSAGE = "denied by owner from Telegram"
UNHANDLED_MESSAGE = "unhandled by tg-bridge"
DEFAULT_TIMEOUT = 15.0
INITIALIZE_TIMEOUT = 10.0


def _random_suffix() -> str:
    return os.urandom(4).hex()


# ---- frame builders (pure) -----------------------------------------------------------------------------------------


def control_request(request_id: str, subtype: str, **fields: Any) -> Frame:
    """Host -> CLI control request. Fields follow the subtype, in the order given."""
    return {"type": "control_request", "request_id": request_id, "request": {"subtype": subtype, **fields}}


def initialize_request(request_id: str) -> Frame:
    return control_request(request_id, "initialize", hooks=None)


def interrupt_request(request_id: str) -> Frame:
    return control_request(request_id, "interrupt")


def set_permission_mode_request(request_id: str, mode: str) -> Frame:
    return control_request(request_id, "set_permission_mode", mode=mode)


def set_model_request(request_id: str, model: str) -> Frame:
    return control_request(request_id, "set_model", model=model)


def user_message(text: str, session_id: str | None) -> Frame:
    """A user turn. Before the session id is known the CLI accepts an empty string."""
    return {"type": "user", "message": {"role": "user", "content": text},
            "parent_tool_use_id": None, "session_id": session_id or ""}


def _success(request_id: str, response: Mapping[str, Any]) -> Frame:
    return {"type": "control_response",
            "response": {"subtype": "success", "request_id": request_id, "response": dict(response)}}


def permission_allow(request_id: str, updated_input: Mapping[str, Any] | None = None,
                     updated_permissions: list[Any] | None = None) -> Frame:
    """Answer a can_use_tool request with allow. updatedInput defaults to {} when the caller passes None."""
    response: dict[str, Any] = {"behavior": "allow",
                                "updatedInput": dict(updated_input) if updated_input is not None else {}}
    if updated_permissions:
        response["updatedPermissions"] = updated_permissions
    return _success(request_id, response)


def permission_deny(request_id: str, message: str = DENY_MESSAGE) -> Frame:
    return _success(request_id, {"behavior": "deny", "message": message})


def control_error(request_id: str, error: str) -> Frame:
    """An error reply to any control request the bridge does not handle."""
    return {"type": "control_response",
            "response": {"subtype": "error", "request_id": request_id, "error": error}}


def unhandled_response(event: Mapping[str, Any]) -> Frame:
    """Reply to a CLI control_request the bridge cannot answer: can_use_tool is denied, anything else gets an error.

    Keeps the CLI from waiting forever for a reply that will never come.
    """
    request_id = event.get("request_id")
    if (event.get("request") or {}).get("subtype") == "can_use_tool":
        return permission_deny(request_id)
    return control_error(request_id, UNHANDLED_MESSAGE)


# ---- ControlChannel ------------------------------------------------------------------------------------------------


class _Slot:
    """One waiting request: the event is set by deliver() or fail_all(); response stays None when the stream ended."""

    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None


class ControlChannel:
    """Correlates control requests with their replies for one session.

    write is the session's stdin writer (returns False when stdin is closed). on_reader_thread tells request() whether
    it is running on the reader thread. suffix supplies the random part of a request id (tests inject a fixed one).
    All state is instance-owned; the lock is held only for dictionary access, never while writing or waiting.
    """

    def __init__(self, write: WriteFn, on_reader_thread: Callable[[], bool], *,
                 suffix: Callable[[], str] = _random_suffix) -> None:
        self._write = write
        self._on_reader_thread = on_reader_thread
        self._suffix = suffix
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._pending: dict[str, _Slot] = {}

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def request(self, subtype: str, timeout: float = DEFAULT_TIMEOUT, **fields: Any) -> dict[str, Any]:
        """Send a control request and return the reply's inner response dict.

        Raises RuntimeError when called on the reader thread, when stdin is closed, when the stream ended before a
        reply, or when the CLI answers with an error. Raises TimeoutError when no reply comes within timeout seconds.
        """
        if self._on_reader_thread():
            raise RuntimeError(f"{subtype}: request() on the reader thread would wait for itself")
        slot = _Slot()
        with self._lock:
            request_id = f"req_{next(self._counter)}_{self._suffix()}"
            self._pending[request_id] = slot
        try:
            if not self._write(control_request(request_id, subtype, **fields)):
                raise RuntimeError(f"{subtype}: stdin closed")
            if not slot.event.wait(timeout):
                raise TimeoutError(f"{subtype} timed out after {timeout}s")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        response = slot.response
        if response is None:
            raise RuntimeError(f"{subtype}: session ended before reply")
        if response.get("subtype") == "error":
            raise RuntimeError(str(response.get("error") or "control error"))
        return response.get("response") or {}

    def initialize(self, timeout: float = INITIALIZE_TIMEOUT) -> dict[str, Any]:
        """The handshake request. Errors propagate; the caller decides whether to continue without it."""
        return self.request("initialize", timeout, hooks=None)

    def deliver(self, event: Mapping[str, Any]) -> bool:
        """Reader thread: hand a control_response to the waiting request. False when nothing is waiting for it."""
        response = event.get("response")
        if not isinstance(response, dict):
            return False
        with self._lock:
            slot = self._pending.get(response.get("request_id"))
        if slot is None or slot.event.is_set():
            return False
        slot.response = response
        slot.event.set()
        return True

    def fail_all(self) -> None:
        """Reader thread, when the stream ends: wake every waiter. Their response stays None."""
        with self._lock:
            slots = list(self._pending.values())
        for slot in slots:
            slot.event.set()
