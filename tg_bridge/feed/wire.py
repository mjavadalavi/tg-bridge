"""FeedWire: the Telegram calls of one turn feed, with the live C12 rules for a failed call.

Port of Turn._call and _lift_keyboard of tg-bridge.py (C12). Every call is marked feed=True, so it takes the feed's
share of the send budget. HTML the server refuses (HTTP 400) goes out again as plain text. 'message is not modified'
is no change (None). 'message to edit not found' and 'can't be edited' raise MessageGone, so the caller can start a new
message with the same lines. Any other failure is raised for the caller to count.
"""

from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramClient, TelegramError
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD

BAD_REQUEST = 400
PARSE_MODE = "HTML"
ERROR_TEXT_LIMIT = 120


class MessageGone(Exception):
    """An edit found its message deleted (or too old to edit): the log starts a new message."""


def description(exc: BaseException) -> str:
    """The lower-cased description Telegram gave with a failure ('' for anything that is not a Telegram error)."""
    return (exc.description or "").lower() if isinstance(exc, TelegramError) else ""


def is_bad_request(exc: BaseException) -> bool:
    return isinstance(exc, TelegramError) and exc.code == BAD_REQUEST


class FeedWire:
    """The sends and edits of one feed, in one topic of one chat."""

    def __init__(self, client: TelegramClient, logger: Logger, chat_id: int | str, thread_id: int) -> None:
        self._client = client
        self._logger = logger
        self._chat_id = chat_id
        self._thread_id = thread_id

    def call(self, method: str, html_text: str, plain: str, markup: str | None, **ids: object) -> dict | None:
        """One sendMessage (no ids) or editMessageText (ids carries message_id). Returns the result, or None when an
        edit changed nothing. Raises MessageGone or the failure itself."""
        params: dict[str, object] = dict(ids, chat_id=self._chat_id, text=html_text, parse_mode=PARSE_MODE)
        if method == "sendMessage":
            params["message_thread_id"] = self._thread_id
        if markup is not None:
            params["reply_markup"] = markup
        try:
            return self._client.call(method, feed=True, **params)
        except TelegramError as exc:
            early = self._soft_outcome(exc)
            if early is not _UNHANDLED:
                return early
            if exc.code != BAD_REQUEST:
                raise
            self._logger.log(f"feed {self._thread_id}: HTML refused ({(exc.description or '')[:80]}), sending plain text")
        del params["parse_mode"]
        params["text"] = plain
        try:
            return self._client.call(method, feed=True, **params)
        except TelegramError as exc:
            early = self._soft_outcome(exc)
            if early is not _UNHANDLED:
                return early
            raise

    def lift_keyboard(self, mid: int) -> None:
        """Once, best effort, when the feed gives up: a log message left with its Stop button loses it."""
        try:
            self._client.call(
                "editMessageReplyMarkup", chat_id=self._chat_id, message_id=mid, reply_markup=EMPTY_KEYBOARD
            )
        except Exception as exc:
            self._logger.log(f"feed {self._thread_id}: could not remove the Stop button: {str(exc)[:ERROR_TEXT_LIMIT]}")

    @staticmethod
    def _soft_outcome(exc: TelegramError) -> object:
        """None for 'not modified', raises MessageGone for a vanished message, else _UNHANDLED."""
        desc = description(exc)
        if "not modified" in desc:
            return None
        if "to edit not found" in desc or "can't be edited" in desc:
            raise MessageGone() from exc
        return _UNHANDLED


_UNHANDLED = object()
