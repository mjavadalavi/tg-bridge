"""FakeTelegram: an in-memory TelegramClient that records calls and can script failures. No network.

It implements the same call(method, feed=False, **params) as HttpTelegramClient, so services are built with it in
tests. It models the outcome the real client gives its caller:

- a scripted 429 is absorbed like the real client does: the fake records it, waits retry_after on the injected
  clock (when there is one) and the call then succeeds;
- a scripted 400 raises TelegramBadRequest at once, with no retry;
- any other scripted code raises TelegramError (the outcome after the client's three attempts).

Successful calls return Telegram-shaped results: sendMessage, sendDocument, sendPhoto and editMessageText return a
message dict with a fresh message_id, createForumTopic returns a topic dict, and other methods return True. The
helpers read back what was sent: texts(thread), buttons(thread) and edits(message_id).
"""

import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from tg_bridge.clock import Clock
from tg_bridge.telegram.client import TelegramBadRequest, TelegramError
from tg_bridge.telegram.ratelimit import DEFAULT_FLOOD_WAIT

FIRST_MESSAGE_ID = 1000
MESSAGE_METHODS = ("sendMessage", "sendDocument", "sendPhoto")
BAD_REQUEST = 400
TOO_MANY_REQUESTS = 429


@dataclass(frozen=True)
class Call:
    """One recorded call: the method name, its parameters, and whether it was marked as a feed call."""

    method: str
    params: dict[str, Any]
    feed: bool


@dataclass(frozen=True)
class Failure:
    """A scripted failure waiting for the next call of its method."""

    code: int
    retry_after: int | None = None
    description: str = ""


class FakeTelegram:
    """A thread-safe fake of the Bot API. Locks guard the records; clock sleeps happen outside the lock."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: list[Call] = []
        self._failures: dict[str, deque[Failure]] = {}
        self._flood_waits: list[tuple[str, int]] = []
        self._next_id = FIRST_MESSAGE_ID
        self._messages: dict[int, dict[str, Any]] = {}
        self._edits: dict[int, list[str]] = {}
        self._results: dict[str, Any] = {}

    @property
    def calls(self) -> tuple[Call, ...]:
        """Every call so far, in order, including calls that then failed."""
        with self._lock:
            return tuple(self._calls)

    @property
    def flood_waits(self) -> tuple[tuple[str, int], ...]:
        """(method, retry_after seconds) for every scripted 429, in order."""
        with self._lock:
            return tuple(self._flood_waits)

    def fail_next(
        self, method: str, code: int, retry_after: int | None = None, description: str = ""
    ) -> None:
        """Script the next call of method to fail with code. Several calls to fail_next queue up in order."""
        with self._lock:
            self._failures.setdefault(method, deque()).append(Failure(code, retry_after, description))

    def call(self, method: str, feed: bool = False, **params: Any) -> Any:
        with self._lock:
            self._calls.append(Call(method, dict(params), feed))
            failure = self._failures[method].popleft() if self._failures.get(method) else None
        if failure is not None:
            self._apply_failure(method, failure)
        return self._result(method, params)

    def script_result(self, method: str, value: Any) -> None:
        """Make every later call of method return value instead of its default result (for example getFile)."""
        with self._lock:
            self._results[method] = value

    def texts(self, thread_id: int | str) -> list[str]:
        """The text of every sendMessage that went to thread_id, oldest first."""
        return [str(c.params.get("text", "")) for c in self._sent_to(thread_id)]

    def buttons(self, thread_id: int | str) -> list[list[list[dict[str, Any]]]]:
        """For every sendMessage in thread_id that carried an inline keyboard: its rows of buttons, oldest first."""
        keyboards = []
        for sent in self._sent_to(thread_id):
            markup = sent.params.get("reply_markup")
            if markup is None:
                continue
            decoded = json.loads(markup) if isinstance(markup, str) else markup
            keyboards.append(decoded.get("inline_keyboard", []))
        return keyboards

    def edits(self, message_id: int) -> list[str]:
        """The texts that editMessageText applied to message_id, oldest first."""
        with self._lock:
            return list(self._edits.get(message_id, []))

    def _sent_to(self, thread_id: int | str) -> list[Call]:
        with self._lock:
            return [
                c
                for c in self._calls
                if c.method == "sendMessage" and str(c.params.get("message_thread_id")) == str(thread_id)
            ]

    def _apply_failure(self, method: str, failure: Failure) -> None:
        if failure.code == TOO_MANY_REQUESTS:
            seconds = failure.retry_after or DEFAULT_FLOOD_WAIT
            with self._lock:
                self._flood_waits.append((method, seconds))
            if self._clock is not None:
                self._clock.sleep(seconds)
            return
        if failure.code == BAD_REQUEST:
            raise TelegramBadRequest(method, BAD_REQUEST, failure.description)
        raise TelegramError(method, failure.code, failure.description, failure.retry_after)

    def _result(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            if method in self._results:
                return self._results[method]
        if method in MESSAGE_METHODS:
            return self._new_message(params)
        if method == "editMessageText":
            return self._edit_message(params)
        if method == "createForumTopic":
            with self._lock:
                self._next_id += 1
                return {"message_thread_id": self._next_id, "name": params.get("name", "")}
        return True

    def _new_message(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            message = {
                "message_id": self._next_id,
                "chat": {"id": params.get("chat_id")},
                "message_thread_id": params.get("message_thread_id"),
                "text": params.get("text", ""),
            }
            if "reply_markup" in params:
                message["reply_markup"] = params["reply_markup"]
            self._messages[self._next_id] = message
            return dict(message)

    def _edit_message(self, params: dict[str, Any]) -> dict[str, Any]:
        message_id = int(params.get("message_id", 0))
        with self._lock:
            self._edits.setdefault(message_id, []).append(str(params.get("text", "")))
            message = self._messages.setdefault(
                message_id, {"message_id": message_id, "chat": {"id": params.get("chat_id")}}
            )
            message["text"] = str(params.get("text", ""))
            return dict(message)
