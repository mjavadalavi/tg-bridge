"""Messenger facade: send_text, send_buttons, edit_text, delete, rename_topic over a TelegramClient (C07).

Port of send_text, send_buttons and rename_topic of tg-bridge.py (C07). Callers say what to send; the facade owns the
chunking, the fallbacks and the rename throttle. Failure policy follows the old helpers:

- send_text and send_buttons log the failure and return (the old helpers swallowed it); a failing chunk stops the rest.
- rename_topic logs and returns; the throttle state is set before the call, so a failed rename still waits its turn.
- edit_text and delete raise TelegramError: their callers (feed, cards) decide what a failed edit means.

Rename state lives on the instance and is guarded by rename_lock, which is never held across a Telegram call.
"""

import threading
from typing import Any

from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.keyboards import CallbackData, keyboard_json

SEND_CHUNK = 3800  # characters per sendMessage; Telegram's limit is 4096 after formatting
SEND_MAX_CHUNKS = 4  # more than this and the rest is replaced by TRUNCATED_NOTE
TRUNCATED_NOTE = "…(truncated)"
EMPTY_TEXT = "(empty)"
EDIT_TEXT_LIMIT = 4096
TOPIC_NAME_LIMIT = 128
RENAME_INTERVAL_SECONDS = 15.0  # a topic is renamed at most once per this many seconds, unless forced
ERROR_TEXT_LIMIT = 150


def chunk_text(text: str) -> list[str]:
    """The text as sendMessage-sized pieces: stripped, '(empty)' when blank, at most SEND_MAX_CHUNKS plus a note."""
    rest = text.strip() or EMPTY_TEXT
    chunks = [rest[i : i + SEND_CHUNK] for i in range(0, len(rest), SEND_CHUNK)]
    if len(chunks) > SEND_MAX_CHUNKS:
        chunks = chunks[:SEND_MAX_CHUNKS] + [TRUNCATED_NOTE]
    return chunks


class Messenger:
    """Telegram send helpers for one bot. Every collaborator is injected; the rename throttle is per instance."""

    def __init__(self, client: TelegramClient, logger: Logger, clock: Clock) -> None:
        self._client = client
        self._logger = logger
        self._clock = clock
        self._rename_lock = threading.Lock()
        self._renamed_at: float | None = None  # clock.monotonic() of the last rename attempt; None before the first
        self._renamed_name = ""

    def send_text(self, chat_id: int | str, thread_id: int | str, text: str) -> None:
        """Plain text to a topic, in chunks. A failed chunk is logged and the remaining chunks are not sent."""
        for chunk in chunk_text(text):
            try:
                self._client.call("sendMessage", chat_id=chat_id, message_thread_id=thread_id, text=chunk)
            except Exception as exc:
                self._logger.log(f"sendMessage failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
                return

    def send_buttons(
        self,
        chat_id: int | str,
        thread_id: int | str,
        text: str,
        rows: list[list[tuple[str, str | CallbackData]]],
    ) -> dict[str, Any] | None:
        """A message with an inline keyboard. Returns the sent message, or None when the send failed (logged).

        The keyboard is built before the send, so an oversized callback raises CallbackDataError instead of being
        logged as a failed send.
        """
        markup = keyboard_json(rows)
        try:
            return self._client.call(
                "sendMessage", chat_id=chat_id, message_thread_id=thread_id, text=text, reply_markup=markup
            )
        except Exception as exc:
            self._logger.log(f"sendButtons failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
            return None

    def edit_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        reply_markup: str | None = None,
        parse_mode: str | None = None,
        feed: bool = False,
    ) -> dict[str, Any]:
        """Edit a message's text (and optionally its keyboard). Raises TelegramError when Telegram refuses the edit.

        reply_markup is a ready JSON string, such as EMPTY_KEYBOARD to remove the buttons. feed=True marks the live
        feed's edit so it takes its share of the send budget.
        """
        params: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text[:EDIT_TEXT_LIMIT]}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        if parse_mode is not None:
            params["parse_mode"] = parse_mode
        return self._client.call("editMessageText", feed=feed, **params)

    def delete(self, chat_id: int | str, message_id: int) -> None:
        """Delete one message. Raises TelegramError when Telegram refuses the deletion."""
        self._client.call("deleteMessage", chat_id=chat_id, message_id=message_id)

    def rename_topic(self, chat_id: int | str, thread_id: int | str, name: str, force: bool = False) -> None:
        """Rename a forum topic. Unless forced, the same name is never sent twice and topics are renamed at most once
        per RENAME_INTERVAL_SECONDS. Failures are logged, never raised."""
        now = self._clock.monotonic()
        with self._rename_lock:
            if not force:
                if name == self._renamed_name:
                    return
                if self._renamed_at is not None and now - self._renamed_at < RENAME_INTERVAL_SECONDS:
                    return
            self._renamed_at, self._renamed_name = now, name
        try:
            self._client.call(
                "editForumTopic", chat_id=chat_id, message_thread_id=thread_id, name=name[:TOPIC_NAME_LIMIT]
            )
        except Exception as exc:
            self._logger.log(f"rename_topic {thread_id} -> {name!r}: {exc}")
