"""MessageHandler: what the bot does with one inbound Telegram message (C18 handle_message and _continue_message).

Port of the live file's handle_message and _continue_message (tg-bridge.py:2288-2336, 3189-3305). The order is the live
one, and every step returns at the first match:

1. handle(): the poll-thread guards. Allowlist (refused and ignored messages are logged), forum_topic_created learning
   (the name goes to kv, and the folder suggestion runs on its own pool), the General-topic guard, and the media split.
   A media message goes to the media worker: in a plain session topic its queue place is reserved first, so the order
   of messages is kept.
2. continue_message(): an awaited 'Other' answer comes first. Then topic -> session resolution: an existing sessions row
   wins; otherwise a spawn alias starts a live seat, an unknown topic gets the hint, and a configured or default topic
   creates its row (and applies the folder kept by the first-message folder card). Then pane topics type the text into
   their tmux pane, 'auto on' and 'auto off' switch the topic's permission mode, bridge commands run now (never queued),
   and everything else is queued behind the topic's running turn.

Disposition says what the caller does next: DONE (the update is finished) or DEFERRED (a queue place or a media worker
owns it now; the dispatcher or the media job marks its inbox row done). Replies are sent through the ordered out-worker,
so the poll thread never waits for the Telegram send budget. Every collaborator is injected, and no module state exists.
"""

import enum
import threading
from collections.abc import Mapping
from typing import Any, Protocol

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.keep_guard import WHAT_MODE, KeepGuard, applies
from tg_bridge.config import BACKEND_CLAUDE, Config
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.media import MediaOutcome, MediaService
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService
from tg_bridge.core.topic_backend import topic_capabilities
from tg_bridge.core.topic_queues import QUEUE_MAX, Admission
from tg_bridge.core.topics import TopicKind, TopicResolver, guard_text
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRecord, SessionRepo
from tg_bridge.telegram.keyboards import CallbackData
from tg_bridge.telegram.messenger import Messenger

AUTO_ON = "auto on"
AUTO_OFF = "auto off"
DEFAULT_OFF_MODE = "default"  # 'auto off' returns to the configured default, or here to 'default' when that is bypass
PANE_TYPE = "pane"  # the config topic type of a tmux pane binding
UNPAIRED_LAUNCHER = "pane"  # the launcher a config topic without one records
AUTO_ON_TEXT = (
    '🤖 full auto — this topic will not ask for permission anymore (read, write, shell, web). '
    'Send "auto off" to restore the approval buttons.'
)
AUTO_OFF_TEXT = "🛡 approvals restored — sensitive tools will ask via buttons again."
AUTO_OFF_NEXT_TEXT = ("🛡 approvals restored from your next message — this launcher cannot change a running turn "
                      "(tap Stop to end it now).")  # a running turn whose CLI cannot switch its mode live
PANE_SENT_TEXT = "✅ → {where}"
PANE_FAILED_TEXT = "⚠️ {info}"
PANE_LABEL = "pane {pane}"
SPAWNED_TEXT = "✅ spawned {launcher} in tmux window '{name}' ({pane}) — your message is its first prompt."
QUEUED_TEXT = "📥 queued — position {position} after the current turn."
QUEUED_TERMINAL_TEXT = ("📥 queued — position {position}; the topic's terminal session (tg-bridge view) is busy "
                        "or has text typed in it.")
QUEUE_FULL_TEXT = "📥 queue full ({limit}) — wait for the current turn to finish."
REFUSED_LOG = "REFUSED message from {user} ({uid}) — allowlist not provisioned"
REMOVING_LOG = "late message for a topic being removed: dropped (thread {thread})"
IGNORED_LOG = "IGNORED message from {user} ({uid}) — not allow-listed"
NOW_LABEL = "⏭ Now"
NEXT_LABEL = "⏩ Next"
DROP_LABEL = "❌ Drop"


class Disposition(enum.Enum):
    """What the caller does with the update once handling returns."""

    DONE = "done"  # answered, refused or ignored here: the update is finished
    DEFERRED = "deferred"  # a queue place or a media worker owns it now; that owner marks the inbox row done


class Seats(Protocol):
    """The tmux side (daemon/panes.Panes). Only these members are used, so tests pass a stub."""

    @property
    def spawn_cwd(self) -> str: ...

    def type_to_pane(self, pane: str, text: str) -> tuple[bool, str]: ...

    def spawn_seat(self, command: str, session_name: str, topic: str, sender: str,
                   text: str) -> tuple[bool, str, str | None]: ...


def sender_name(frm: Mapping[str, Any]) -> str:
    """The name the rules text and the queue show for a sender: first name, else username, else the id."""
    return str(frm.get("first_name") or frm.get("username") or frm.get("id") or "")


def queue_rows(thread: int, nid: str) -> list[list[tuple[str, CallbackData]]]:
    """The Now / Next / Drop buttons of a queued message."""
    return [[
        (NOW_LABEL, CallbackData("now", (thread, nid))),
        (NEXT_LABEL, CallbackData("jump", (thread, nid))),
        (DROP_LABEL, CallbackData("drop", (thread, nid))),
    ]]


class MessageHandler:
    """Handles one inbound message. Built once by the App; holds no state beyond its injected collaborators."""

    def __init__(
        self,
        *,
        config: Config,
        topics: TopicResolver,
        sessions: SessionRepo,
        kv: KvRepo,
        inbox: InboxRepo,
        dispatcher: Dispatcher,
        permissions: PermissionService,
        live: LiveRegistry,
        cwd: CwdService,
        commands: CommandRegistry,
        command_deps: CommandDeps,
        backends: BackendRegistry,
        messenger: Messenger,
        out: WorkerPool,
        media: MediaService,
        media_pool: WorkerPool,
        seats: Seats,
        logger: Logger,
        keep: KeepGuard | None = None,
    ) -> None:
        self._config = config
        self._topics = topics
        self._sessions = sessions
        self._kv = kv
        self._inbox = inbox
        self._dispatcher = dispatcher
        self._permissions = permissions
        self._live = live
        self._cwd = cwd
        self._commands = commands
        self._deps = command_deps
        self._backends = backends
        self._messenger = messenger
        self._out = out
        self._media = media
        self._media_pool = media_pool
        self._seats = seats
        self._logger = logger
        self._keep = keep  # None: 'auto on' and 'auto off' apply at once, as before the confirm card (section 5 D1)

    # --- poll-thread guards

    def handle(self, message: Mapping[str, Any], update_id: int | None = None) -> Disposition:
        """The guards of one message, then either the media split or continue_message. Never waits for a turn."""
        chat = message.get("chat") or {}
        frm = message.get("from") or {}
        uid, chat_id = frm.get("id"), chat.get("id")
        if uid is None or chat_id is None:
            return Disposition.DONE
        thread = message.get("message_thread_id")
        created = message.get("forum_topic_created")
        if created:  # learned wherever it appears, before the allowlist, as the live file does
            self._topics.learn(thread, created.get("name", ""))
        allowed = self._config.allow_user_ids()
        if not allowed:  # the allowlist is provisioned by hand in config.json, never learned from a first message
            self._logger.log(REFUSED_LOG.format(user=frm.get("username"), uid=uid))
            return Disposition.DONE
        if uid not in allowed:
            self._logger.log(IGNORED_LOG.format(user=frm.get("username"), uid=uid))
            return Disposition.DONE
        if created:
            self._cwd.suggest_for_new_topic(self._config, chat_id, thread, created.get("name", ""))
            return Disposition.DONE
        text = (message.get("text") or message.get("caption") or "").strip()
        has_media = bool(message.get("photo") or message.get("voice") or message.get("audio"))
        if not text and not has_media:
            return Disposition.DONE
        guidance = guard_text(chat.get("type"), thread)
        if guidance is not None:
            self._say(chat_id, thread, guidance)
            return Disposition.DONE
        if has_media:
            return self._media_message(message, text, update_id)
        return self.continue_message(message, text, update_id)

    def _media_message(self, message: Mapping[str, Any], text: str, update_id: int | None) -> Disposition:
        """A photo, voice or audio message: its queue place is reserved first in a plain session topic, then the media
        worker downloads it and continues the message. Any other topic lets the worker resolve it the usual way."""
        chat_id = (message.get("chat") or {}).get("id")
        thread = message.get("message_thread_id")
        topic = self._topics.queued_session_name(thread)
        if topic is None:
            self._media_pool.submit(self._media_job, message, text, update_id, None)
            return Disposition.DEFERRED
        delivery = self._dispatcher.reserve(thread, topic, chat_id, sender_name(message.get("from") or {}), update_id)
        if delivery.admission is Admission.FULL:
            self._say(chat_id, thread, QUEUE_FULL_TEXT.format(limit=QUEUE_MAX))
            return Disposition.DONE
        self._media_pool.submit(self._media_job, message, text, update_id, delivery.nid)
        return Disposition.DEFERRED

    def _media_job(self, message: Mapping[str, Any], text: str, update_id: int | None,
                   reserved: str | None) -> None:
        """Worker body: download and transcribe, then continue the message. The queue place and the inbox row are
        released here unless the message was queued (then its dispatcher owns both)."""
        thread = message.get("message_thread_id")
        try:
            outcome: MediaOutcome | None = self._media.fetch(message, text, self._native_images(thread))
        except Exception as exc:  # the media layer reports its own failures; anything else must not leave a place held
            self._logger.log(f"media for thread {thread} failed: {str(exc)[:200]}")
            outcome = None
        if outcome is not None:
            if self.continue_message(message, outcome.text, update_id, reserved, outcome.images) is Disposition.DEFERRED:
                return
        self._dispatcher.unreserve(thread, reserved)
        if update_id is not None:
            self._inbox.done(update_id)

    # --- the message after the guards

    def continue_message(self, message: Mapping[str, Any], text: str, update_id: int | None = None,
                         reserved: str | None = None, images: tuple[str, ...] = ()) -> Disposition:
        """Everything after the media step. reserved is the queue place of a media message, if it has one."""
        chat_id = (message.get("chat") or {}).get("id")
        thread = message.get("message_thread_id")
        if self._permissions.record_text(thread, text):  # the text answers an 'Other' question of a card
            return Disposition.DONE
        sender = sender_name(message.get("from") or {})
        row = self._sessions.get(thread)
        if row is None:
            if self._dispatcher.removing(thread):  # a late update must not re-create a topic being removed
                self._drop_removing(thread, update_id)
                return Disposition.DONE
            row = self._new_topic(message, thread, chat_id, sender, text)
            if row is None:  # a spawn seat or an unknown topic answered the message
                return Disposition.DONE
        pane = self._pane_of(row.name)
        if pane is not None:
            self._type_into_pane(chat_id, thread, row.name, pane, text)
            return Disposition.DONE
        if self._auto_mode(chat_id, thread, text):
            return Disposition.DONE
        if self._commands.dispatch(self._deps, chat_id, thread, row.name, text):  # commands run now, never queued
            return Disposition.DONE
        return self._enqueue(chat_id, thread, row.name, sender, update_id, text, reserved, images)

    def _drop_removing(self, thread: int, update_id: int | None) -> None:
        """The Dispatcher's drop for a removed topic: the inbox row is marked done, so the update never replays."""
        if update_id is not None:
            self._inbox.done(update_id)
        self._logger.log(REMOVING_LOG.format(thread=thread))

    def _new_topic(self, message: Mapping[str, Any], thread: int, chat_id: Any, sender: str,
                   text: str) -> SessionRecord | None:
        """A topic with no sessions row: spawn a seat, say it is unknown, or create its row from the config topic.
        None when the message was answered here (a spawn or an unknown topic)."""
        target = self._topics.resolve(thread, message.get("reply_to_message"))
        if target.kind is TopicKind.UNKNOWN:
            self._say(chat_id, thread, self._topics.unknown_text(target))
            return None
        if target.kind is TopicKind.SPAWN:
            self._spawn(chat_id, thread, target.name, target.spawn_command, target.spawn_name, sender, text)
            return None
        if target.kind in (TopicKind.CONFIGURED, TopicKind.DEFAULT):
            self._sessions.ensure(thread, target.name, target.config.get("launcher", UNPAIRED_LAUNCHER),
                                  target.config.get("cwd", self._config.home), self._config.default_mode())
            self._cwd.apply_pending(thread)  # the folder kept by the first-message folder card
        return self._sessions.get(thread)

    def _spawn(self, chat_id: Any, thread: int, name: str, command: str, seat_name: str, sender: str,
               text: str) -> None:
        """Opens a live seat for a spawn topic (alias.task). Blocks on tmux, as the live file did, for the window's prompt."""
        ok, info, pane = self._seats.spawn_seat(command, seat_name, name, sender, text)
        launcher = command.rsplit("/", 1)[-1]
        if pane:
            self._kv.set(f"pane:{name}", pane)
            self._sessions.ensure(thread, name, launcher, self._seats.spawn_cwd, "")
        if ok and pane:
            reply = SPAWNED_TEXT.format(launcher=launcher, name=seat_name, pane=pane)
        else:
            reply = PANE_FAILED_TEXT.format(info=info)
        self._say(chat_id, thread, reply)
        self._logger.log(f"spawn topic {name}: {info}")

    def _pane_of(self, name: str) -> str | None:
        """The tmux pane a topic types into: its config pane (type 'pane') or the kv binding a spawn left, else None."""
        topic = self._config.topics().get(name) or {}
        pane = topic.get("pane") if topic.get("type") == PANE_TYPE else None
        return pane or self._kv.get(f"pane:{name}")

    def _type_into_pane(self, chat_id: Any, thread: int, name: str, pane: str, text: str) -> None:
        ok, info = self._seats.type_to_pane(pane, text)
        topic = self._config.topics().get(name) or {}
        if ok:
            where = topic.get("label") or PANE_LABEL.format(pane=pane)
            self._say(chat_id, thread, PANE_SENT_TEXT.format(where=where))
        else:
            self._say(chat_id, thread, PANE_FAILED_TEXT.format(info=info))
        self._logger.log(f"pane topic {name}: {info}")

    def _auto_mode(self, chat_id: Any, thread: int, text: str) -> bool:
        """'auto on' and 'auto off' (exact words, any case): the topic's stored mode changes at once, and a running turn
        switches through PermissionService.set_live_mode on a thread of its own, since the CLI may take a while."""
        word = text.strip().lower()
        if word not in (AUTO_ON, AUTO_OFF):
            return False
        off_mode = self._config.default_mode()
        if off_mode == BYPASS_MODE:
            off_mode = DEFAULT_OFF_MODE
        mode = BYPASS_MODE if word == AUTO_ON else off_mode
        if self._keep is not None:
            card = self._keep.hold(thread, WHAT_MODE, applies(lambda: self._store_auto(thread, mode, word)))
            if card is not None:  # the change waits for the confirm card (section 5 D1)
                self._out.submit(self._messenger.send_buttons, chat_id, thread, card.text, card.rows)
                return True
        self._store_auto(thread, mode, word)
        self._say(chat_id, thread, self._auto_reply(thread, word))
        self._logger.log(f"{word} thread {thread}")
        return True

    def _store_auto(self, thread: int, mode: str, word: str) -> None:
        """Stores the topic's mode for 'auto on' or 'auto off' and sends it to a running turn on a thread of its own."""
        self._sessions.set_mode(thread, mode)
        if self._live.running(thread) is not None:
            threading.Thread(target=self._permissions.set_live_mode, args=(thread, mode),
                             kwargs={"by": word}, daemon=True).start()

    def _auto_reply(self, thread: int, word: str) -> str:
        """The reply to 'auto on' / 'auto off'. 'auto off' during a turn whose CLI cannot change its mode live (a tui
        turn started in bypass) says the approvals come back from the next message, not now."""
        if word == AUTO_ON:
            return AUTO_ON_TEXT
        row = self._sessions.get(thread)
        if self._live.running(thread) is not None and row is not None:
            if not topic_capabilities(self._config, self._backends, row, row.name).live_mode:
                return AUTO_OFF_NEXT_TEXT
        return AUTO_OFF_TEXT

    def _enqueue(self, chat_id: Any, thread: int, name: str, sender: str, update_id: int | None, text: str,
                 reserved: str | None, images: tuple[str, ...]) -> Disposition:
        """Queues the text behind a running turn, or starts the topic's runner. Notices carry the Now/Next/Drop buttons."""
        delivery = self._dispatcher.deliver(thread, name, chat_id, sender, update_id, text, reserved, images)
        if delivery.admission is Admission.FULL:
            self._say(chat_id, thread, QUEUE_FULL_TEXT.format(limit=QUEUE_MAX))
            return Disposition.DONE
        if delivery.notify and delivery.nid is not None:
            text = QUEUED_TERMINAL_TEXT if self._dispatcher.waiting_terminal(thread) else QUEUED_TEXT
            self._out.submit(self._messenger.send_buttons, chat_id, thread,
                             text.format(position=delivery.position), queue_rows(thread, delivery.nid))
        return Disposition.DEFERRED

    # --- helpers

    def _native_images(self, thread: int) -> bool:
        """Whether the topic's turns take a photo as an image (the backend of its resolved launcher, the one TurnRunner
        uses). A topic with no row yet is a Claude topic, as it will be created."""
        row = self._sessions.get(thread)
        if row is None:
            return self._backends.get(BACKEND_CLAUDE).capabilities.images
        return topic_capabilities(self._config, self._backends, row, row.name).images

    def _say(self, chat_id: Any, thread: int, text: str) -> None:
        """A plain reply to the topic, through the ordered out-worker."""
        self._out.submit(self._messenger.send_text, chat_id, thread, text)
