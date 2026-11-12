"""AnswerSender: a turn's final answer as HTML messages, a plain-text resend on a 400, or one .md document (C08).

Port of the live file's send_answer() and _send_answer_doc() (C08). The rules are the live ones:

- an answer longer than doc_threshold, or one that does not fit ANSWER_MAX_MESSAGES HTML messages, goes out as one
  answer-<timestamp>.md document with its first 900 characters as the caption;
- otherwise each chunk is sent as Telegram HTML; an HTTP 400 (TelegramBadRequest) resends that chunk as plain text;
  any other failure is logged and the remaining chunks are dropped (the old behaviour);
- when the document upload fails, the answer is sent as plain text through Messenger.send_text instead.

Collaborators are injected: the JSON client for sendMessage, a DocumentSender for uploads, the Messenger for the
text fallback, the Logger, and the Clock (the document name is stamped from it, so tests are deterministic).
"""

import datetime
from collections.abc import Callable

from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramBadRequest, TelegramClient
from tg_bridge.telegram.messenger import Messenger
from tg_bridge.telegram.render import _answer_chunks, md_to_html
from tg_bridge.telegram.upload import DocumentSender

DOC_THRESHOLD = 9000  # characters; a longer answer always goes out as a document
CAPTION_CHARS = 900
PLAIN_LIMIT = 4096  # characters of a chunk resent as plain text
ERROR_TEXT_LIMIT = 150
EMPTY_ANSWER = "(empty)"
DOCUMENT_NAME = "answer-%Y%m%d-%H%M%S.md"
DOCUMENT_MIME = "text/markdown"


class AnswerSender:
    """Sends one turn's final answer to a topic. Holds no state between answers."""

    def __init__(
        self,
        client: TelegramClient,
        documents: DocumentSender,
        messenger: Messenger,
        logger: Logger,
        clock: Clock,
        on_answer: Callable[[int | str, int | str, str], None] | None = None,
    ) -> None:
        self._client = client
        self._on_answer = on_answer  # told of every answer once it was sent (driving mode speaks it)
        self._documents = documents
        self._messenger = messenger
        self._logger = logger
        self._clock = clock

    def send_answer(
        self,
        chat_id: int | str,
        thread_id: int | str,
        text: str,
        doc_threshold: int = DOC_THRESHOLD,
    ) -> None:
        """The answer to a topic. Never raises: every failure is logged and the send stops there."""
        body = (text or "").strip() or EMPTY_ANSWER
        chunks = _answer_chunks(body) if len(body) <= doc_threshold else None
        if chunks is None:
            self._send_document(chat_id, thread_id, body)
        else:
            self._send_html(chat_id, thread_id, chunks)
        if self._on_answer is not None:
            try:
                self._on_answer(chat_id, thread_id, body)
            except Exception as exc:  # the listener never breaks the answer
                self._logger.log(f"answer listener: {str(exc)[:ERROR_TEXT_LIMIT]}")

    def _send_html(self, chat_id: int | str, thread_id: int | str, chunks: list[str]) -> None:
        for md in chunks:
            try:
                self._client.call(
                    "sendMessage",
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=md_to_html(md),
                    parse_mode="HTML",
                )
            except TelegramBadRequest:
                self._logger.log("answer HTML refused (400), sending plain text")
                if not self._send_plain(chat_id, thread_id, md):
                    return
            except Exception as exc:
                self._logger.log(f"answer send failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
                return

    def _send_plain(self, chat_id: int | str, thread_id: int | str, md: str) -> bool:
        """One chunk as plain text. False when that send failed too (logged)."""
        try:
            self._client.call("sendMessage", chat_id=chat_id, message_thread_id=thread_id, text=md[:PLAIN_LIMIT])
        except Exception as exc:
            self._logger.log(f"answer send failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
            return False
        return True

    def _send_document(self, chat_id: int | str, thread_id: int | str, body: str) -> None:
        """The whole answer as one .md document. If the upload fails, the answer goes out as plain text."""
        name = datetime.datetime.fromtimestamp(self._clock.now()).strftime(DOCUMENT_NAME)
        caption = body[:CAPTION_CHARS] + ("…" if len(body) > CAPTION_CHARS else "")
        try:
            self._documents.send_doc_bytes(
                chat_id, thread_id, body.encode("utf-8"), name, DOCUMENT_MIME, caption=caption
            )
        except Exception as exc:
            self._logger.log(f"answer document failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
            self._messenger.send_text(chat_id, thread_id, body)
