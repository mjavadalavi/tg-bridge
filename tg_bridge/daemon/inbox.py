"""InboxService: the durable inbox of the daemon (C22): ingest, process, replay, and the spool drain.

Port of the live file's ingest_update, inbox_done, process_update, replay_inbox and drain_spool (tg-bridge.py:3331-3420,
3472-3491, 3495-3580). The rules that make a restart lose nothing:

- ingest_update stores the update as pending before the offset moves (the offset is the last update_id stored), so a
  crash can re-deliver an update but never drop one. It returns False for a re-delivery, and the poller then skips it.
- process_update marks the row done after its work, unless the message was deferred (a queue place or a media worker
  owns the row now). A handler that raises is logged, and the row is still finished, as in the live file.
- replay_inbox runs at start-up: finished rows older than seven days are pruned, then every pending row is processed in
  update_id order. A callback tap older than CALLBACK_REPLAY_SECONDS is stale (its message has moved on) and is finished
  without running, and so is a payload that is not a JSON object.
- drain_spool relays the {topic, text} files that other local tools drop into the spool folder.

Every collaborator is injected. The service holds no state of its own; the chat-learning writes go to kv.
"""

import datetime
import json
import os
from collections.abc import Mapping
from typing import Any, Protocol

from tg_bridge.bot.inbound import Disposition
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.messenger import Messenger

CALLBACK_REPLAY_SECONDS = 600  # a callback tap older than this at start-up is stale and is not run
OFFSET_KEY = "offset"  # kv key: the last update_id stored (the poller asks Telegram for the next one)
GROUP_KEY = "group_id"  # kv key: the forum group the bridge answers in, learned from its first message
CHAT_KEY = "chat:{chat_id}"  # kv key: the title (or type) of a chat, learned from its messages
SPOOL_SUFFIX = ".json"
DATA_LOG_LIMIT = 20  # characters of a callback_data shown in the log
ERROR_TEXT_LIMIT = 200


class MessageSink(Protocol):
    """What the service needs from the message handler (bot.inbound.MessageHandler)."""

    def handle(self, message: Mapping[str, Any], update_id: int | None = None) -> Disposition:
        """Handles one message. DEFERRED means the row stays pending until its owner marks it done."""
        ...


class CallbackSink(Protocol):
    """What the service needs from the callback router (bot.callbacks.router.CallbackRouter)."""

    def dispatch(self, query: Mapping[str, Any]) -> None:
        """Handles one callback_query. Never raises."""
        ...


def _decode(payload: str) -> dict[str, Any] | None:
    """The update stored as JSON text, or None when it is not a JSON object."""
    try:
        update = json.loads(payload)
    except ValueError:
        return None
    return update if isinstance(update, dict) else None


def _age_seconds(created: str, now: float) -> float:
    """Seconds between a stored UTC stamp and now. An unreadable stamp counts as infinitely old (stale)."""
    try:
        return now - datetime.datetime.fromisoformat(created).timestamp()
    except ValueError:
        return float("inf")


class InboxService:
    """Ingest, process and replay the Telegram updates; relay the spool files. One per App."""

    def __init__(
        self,
        *,
        inbox: InboxRepo,
        kv: KvRepo,
        sessions: SessionRepo,
        messenger: Messenger,
        messages: MessageSink,
        callbacks: CallbackSink,
        spool_dir: str,
        clock: Clock,
        logger: Logger,
    ) -> None:
        self._inbox = inbox
        self._kv = kv
        self._sessions = sessions
        self._messenger = messenger
        self._messages = messages
        self._callbacks = callbacks
        self._spool_dir = spool_dir
        self._clock = clock
        self._logger = logger

    def ingest_update(self, update: Mapping[str, Any]) -> bool:
        """Stores one getUpdates update as pending, then moves the offset to it. False when it was already stored."""
        message = update.get("message") or update.get("channel_post") or {}
        query = update.get("callback_query") or {}
        source = message or query.get("message") or {}
        sender = message.get("from") or query.get("from") or {}
        update_id = int(update["update_id"])
        stored = self._inbox.ingest(
            update_id,
            update,
            thread_id=source.get("message_thread_id"),
            chat_id=(source.get("chat") or {}).get("id"),
            sender=str(sender.get("id") or ""),
        )
        self._kv.set(OFFSET_KEY, str(update_id))
        return stored

    def inbox_done(self, update_id: int | None) -> None:
        """Marks one row done. None is a no-op: the message had no inbox row (for example a queued pane message)."""
        if update_id is not None:
            self._inbox.done(update_id)

    def process_update(self, update: Mapping[str, Any]) -> None:
        """Handles one stored update: the message handler or the callback router, then the row is finished.

        Each handler has its own try/except. A message that was deferred keeps its row pending.
        """
        update_id = int(update["update_id"])
        query = update.get("callback_query")
        message = update.get("message") or update.get("channel_post")
        if query:
            self._logger.log(f"callback data={str(query.get('data') or '')[:DATA_LOG_LIMIT]}")
        deferred = False
        if message:
            self._learn_chat(message)
            try:
                deferred = self._messages.handle(message, update_id=update_id) is Disposition.DEFERRED
            except Exception as exc:
                self._logger.log(f"handle: {str(exc)[:ERROR_TEXT_LIMIT]}")
        if query:
            try:
                self._callbacks.dispatch(query)
            except Exception as exc:
                self._logger.log(f"callback: {str(exc)[:ERROR_TEXT_LIMIT]}")
        if not deferred:
            self.inbox_done(update_id)

    def replay_inbox(self) -> None:
        """Start-up: prune old finished rows, then process every pending row in update_id order (see module doc)."""
        self._inbox.prune()
        pending = self._inbox.pending()
        self._logger.log(f"replay: {len(pending)} pending update(s)")
        now = self._clock.now()
        for record in pending:
            age = _age_seconds(record.created, now)
            update = _decode(record.payload)
            if update is None or (update.get("callback_query") and age > CALLBACK_REPLAY_SECONDS):
                self._logger.log(f"replay: update {record.update_id} skipped (age {age:.0f}s)")
                self._inbox.done(record.update_id)
                continue
            self._logger.log(f"replay: update {record.update_id}")
            self.process_update(update)

    def drain_spool(self) -> None:
        """Relays every {topic, text} file of the spool folder to its topic, then deletes the file.

        A file whose topic is unknown, or whose group is not learned yet, stays in place for the next pass. A bad file
        is logged and left alone.
        """
        if not os.path.isdir(self._spool_dir):
            return
        for name in sorted(os.listdir(self._spool_dir)):
            if not name.endswith(SPOOL_SUFFIX):
                continue
            path = os.path.join(self._spool_dir, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    item = json.load(handle)
                topic = item.get("topic")
                thread = self._sessions.thread_of(topic)
                group = self._kv.get(GROUP_KEY)
                if thread is None or not group:
                    self._logger.log(f"spool {name}: unknown topic {topic!r} — left in place")
                    continue
                self._messenger.send_text(int(group), thread, item.get("text", ""))
                os.remove(path)
            except Exception as exc:
                self._logger.log(f"spool {name}: {str(exc)[:ERROR_TEXT_LIMIT]}")

    def _learn_chat(self, message: Mapping[str, Any]) -> None:
        """Remembers the chat's title in kv, and the forum group's id, from any message. Never raises."""
        try:
            chat = message.get("chat") or {}
            if chat.get("id") is not None:
                title = chat.get("title") or chat.get("type") or ""
                self._kv.set(CHAT_KEY.format(chat_id=chat["id"]), str(title))
                if chat.get("type") == "group" or chat.get("is_forum"):
                    self._kv.set(GROUP_KEY, str(chat["id"]))
        except Exception as exc:
            self._logger.log(f"chat learn: {str(exc)[:ERROR_TEXT_LIMIT]}")
