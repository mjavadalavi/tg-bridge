"""FakeUploader: an in-memory DocumentSender that records uploads and can script a failure. No network.

send_doc_bytes records every upload (bytes, name, mime, caption, method, field) and returns a message dict. fail_next()
makes the next upload raise TelegramError, the same outcome Uploader gives after Telegram refuses a request.
send_file records the path only; the byte path is covered by test_upload against a loopback stub.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from tg_bridge.telegram.client import TelegramError

FIRST_MESSAGE_ID = 2000


@dataclass(frozen=True)
class Upload:
    """One recorded upload of in-memory bytes."""

    chat_id: int | str
    thread_id: int | str
    data: bytes
    filename: str
    mime: str
    caption: str
    method: str
    field_name: str


class FakeUploader:
    """A thread-safe DocumentSender double. Failures are queued with fail_next and consumed by the next uploads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._uploads: list[Upload] = []
        self._files: list[tuple[int | str, int | str, str, str]] = []
        self._failures: deque[str] = deque()
        self._next_id = FIRST_MESSAGE_ID

    @property
    def uploads(self) -> tuple[Upload, ...]:
        """Every byte upload so far, in order, including the ones that then failed."""
        with self._lock:
            return tuple(self._uploads)

    @property
    def files(self) -> tuple[tuple[int | str, int | str, str, str], ...]:
        """(chat_id, thread_id, path, caption) for every send_file call, in order."""
        with self._lock:
            return tuple(self._files)

    def fail_next(self, description: str = "upload refused") -> None:
        """Make the next send_doc_bytes raise TelegramError with this description."""
        with self._lock:
            self._failures.append(description)

    def send_doc_bytes(
        self,
        chat_id: int | str,
        thread_id: int | str,
        data: bytes,
        filename: str,
        mime: str,
        caption: str = "",
        method: str = "sendDocument",
        field_name: str = "document",
    ) -> dict[str, Any]:
        with self._lock:
            self._uploads.append(Upload(chat_id, thread_id, data, filename, mime, caption, method, field_name))
            failure = self._failures.popleft() if self._failures else None
            self._next_id += 1
            message_id = self._next_id
        if failure is not None:
            raise TelegramError(method, 500, failure)
        return {"message_id": message_id, "chat": {"id": chat_id}, "message_thread_id": thread_id}

    def send_file(self, chat_id: int | str, thread_id: int | str, path: str, caption: str = "") -> None:
        with self._lock:
            self._files.append((chat_id, thread_id, path, caption))
