"""CallbackRouter: the one entry point for inline-keyboard taps (C20 handle_callback).

A callback_query is checked against the allow-list first (a tap from anyone else gets 'not allowed' and nothing
changes), then its prefix picks the handler: a plain prefix (pa, now, st, ...) through on(), and a picker tap
(cm:<thread>:<kind>:<arg>) through the second level, on_picker(), keyed by kind. A prefix or kind nobody registered
is logged and ignored, as the live file does. Every handler runs inside one try/except that logs the failure, so one
bad tap never reaches the poll loop.

A handler gets a Tap. Its replies are the only Telegram calls a tap makes: the answer to the tap goes out at once
(Telegram wants it fast), and every message write (an edit, a new picker message) is queued on the ordered out-worker,
so the poll thread never waits for the send budget. Routing reads the registries only; they are filled once at start-up
and never change while taps arrive, so the router needs no lock.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.bot.commands.base import Submitter
from tg_bridge.config import Config
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.keyboards import PICKER_PREFIX, SEPARATOR, CallbackData, keyboard_json
from tg_bridge.telegram.messenger import Messenger

NOT_ALLOWED_TEXT = "not allowed"
PICKER_FIELDS = 4  # cm:<thread>:<kind>:<arg>: the last field keeps any ':' (a model name such as qwen3:8b)
ERROR_TEXT_CHARS = 120
ANSWER_METHOD = "answerCallbackQuery"


class TapReplies:
    """The Telegram side of a tap: the answer goes straight to the client, message writes go to the out-worker."""

    def __init__(self, client: TelegramClient, messenger: Messenger, out: Submitter) -> None:
        self._client = client
        self._messenger = messenger
        self._out = out

    def answer(self, callback_id: str, text: str) -> None:
        self._client.call(ANSWER_METHOD, callback_query_id=callback_id, text=text)

    def edit(self, chat_id: int | str, message_id: int, text: str, rows: list[list[Any]]) -> None:
        self._out.submit(self._messenger.edit_text, chat_id, message_id, text, keyboard_json(rows))

    def send_buttons(self, chat_id: int | str, thread: int, text: str, rows: list[list[Any]]) -> None:
        self._out.submit(self._messenger.send_buttons, chat_id, thread, text, rows)


@dataclass(frozen=True)
class Tap:
    """One tap: the callback's prefix and arguments, and the message it was pressed on.

    args are the ':'-separated fields after the prefix, as text. Handlers check their own arguments; the router only
    guarantees the shape of a picker tap (see CallbackRouter.dispatch). The methods are the handler's only replies.
    """

    callback_id: str
    prefix: str
    args: tuple[str, ...]
    chat_id: int | str | None
    message_id: int | None
    message_text: str
    replies: TapReplies

    def toast(self, text: str) -> None:
        """The short answer shown to the tapper (answerCallbackQuery)."""
        self.replies.answer(self.callback_id, text)

    def edit(self, text: str, rows: list[list[Any]] | None = None) -> None:
        """Replaces the text of the tapped message and its keyboard (none when rows is empty or None)."""
        if self.chat_id is None or self.message_id is None:
            return
        self.replies.edit(self.chat_id, self.message_id, text, rows or [])

    def send_buttons(self, thread: int, text: str, rows: list[list[Any]]) -> None:
        """A new message with a keyboard in the topic thread, in the chat the tap came from."""
        if self.chat_id is None:
            return
        self.replies.send_buttons(self.chat_id, thread, text, rows)


TapHandler = Callable[[Tap], None]


class CallbackRouter:
    """Allow-list, prefix registry and picker-kind registry for inline-keyboard taps. One per App."""

    def __init__(self, *, config: Config, client: TelegramClient, messenger: Messenger, out: Submitter,
                 logger: Logger) -> None:
        self._allowed = tuple(config.allow_user_ids())
        self._replies = TapReplies(client, messenger, out)
        self._logger = logger
        self._prefixes: dict[str, TapHandler] = {}
        self._pickers: dict[str, TapHandler] = {}

    def on(self, prefix: str, handler: TapHandler) -> None:
        """Registers the handler of a plain prefix. ValueError for an empty or ':'-containing prefix, the picker prefix
        (it has its own registry), or a prefix that is already registered."""
        if not prefix or SEPARATOR in prefix or prefix == PICKER_PREFIX:
            raise ValueError(f"invalid callback prefix: {prefix!r}")
        if prefix in self._prefixes:
            raise ValueError(f"callback prefix {prefix!r} is already registered")
        self._prefixes[prefix] = handler

    def on_picker(self, kind: str, handler: TapHandler) -> None:
        """Registers the handler of a picker kind (cm:<thread>:<kind>:<arg>). ValueError on a duplicate kind."""
        if not kind or SEPARATOR in kind:
            raise ValueError(f"invalid picker kind: {kind!r}")
        if kind in self._pickers:
            raise ValueError(f"picker kind {kind!r} is already registered")
        self._pickers[kind] = handler

    def dispatch(self, query: Mapping[str, Any]) -> None:
        """Handles one callback_query. Never raises: a refusal, an unknown prefix or a failing handler is logged."""
        callback_id = str(query.get("id") or "")
        try:
            tapper = (query.get("from") or {}).get("id")
            if tapper not in self._allowed:
                self._logger.log(f"callback refused from {tapper}")
                self._replies.answer(callback_id, NOT_ALLOWED_TEXT)
                return
            data = CallbackData.decode(str(query.get("data") or ""))
            if data.prefix == PICKER_PREFIX:
                self._route_picker(query, callback_id)
                return
            handler = self._prefixes.get(data.prefix)
            if handler is None:
                self._logger.log(f"callback with unknown prefix {data.prefix!r} ignored")
                return
            handler(self._tap(query, callback_id, data.prefix, tuple(str(a) for a in data.args)))
        except Exception as exc:
            self._logger.log(f"answerCallbackQuery: {str(exc)[:ERROR_TEXT_CHARS]}")

    def _route_picker(self, query: Mapping[str, Any], callback_id: str) -> None:
        """cm:<thread>:<kind>:<arg>: the thread must be a number and the arg present; the kind picks the handler."""
        data = CallbackData.decode(str(query.get("data") or ""), PICKER_FIELDS)
        args = tuple(str(a) for a in data.args)
        if len(args) != PICKER_FIELDS - 1 or not args[0].isdigit() or not args[2]:
            self._logger.log(f"malformed picker callback ignored: {str(query.get('data'))[:ERROR_TEXT_CHARS]!r}")
            return
        handler = self._pickers.get(args[1])
        if handler is None:
            self._logger.log(f"picker kind {args[1]!r} is not handled; tap ignored")
            return
        handler(self._tap(query, callback_id, PICKER_PREFIX, args))

    def _tap(self, query: Mapping[str, Any], callback_id: str, prefix: str, args: tuple[str, ...]) -> Tap:
        message = query.get("message") or {}
        chat = message.get("chat") or {}
        return Tap(
            callback_id=callback_id,
            prefix=prefix,
            args=args,
            chat_id=chat.get("id"),
            message_id=message.get("message_id"),
            message_text=str(message.get("text") or ""),
            replies=self._replies,
        )
