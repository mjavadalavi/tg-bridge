"""TopicResolver: which session a Telegram topic (forum thread) belongs to (C18 topic resolution).

Port of the topic lookup in the live file's _continue_message and _session_name, and of the General-topic guard in
handle_message. The order is the live one:

1. A sessions row for the thread wins: its name is the session name.
2. Otherwise the name is learned: kv 'thread:<id>' (written from a forum_topic_created message), else the
   forum_topic_created name of the message this one replies to (which is then learned into kv).
3. A non-empty config topic with that name is used.
4. Else a spawn alias: 'alias.task' with a non-blank task and a spawn entry for the alias.
5. Else config default_topic, which makes any fresh topic its own session.
6. Else the topic is unknown; unknown_text() lists the known names.

The resolver never creates session rows and never sends a message: the caller acts on the TopicTarget it gets.
Its only write is the kv learning. Config is read through its accessors, so nothing is cached here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tg_bridge.config import Config
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo

GENERAL_TOPIC_THREAD = 1  # the General topic of a forum group has thread id 1
GENERAL_TEXT = "This is the general board — open a topic and message me there."
NO_TOPIC_TEXT = 'Add me to the "My AIs" forum group and talk in a topic.'
UNKNOWN_TOPIC_HINT = (
    "To spawn a new live seat, name the topic like work.mytask — a session starts in a new tmux window "
    "and your first message becomes its first prompt."
)


class TopicKind(Enum):
    """How a thread was resolved. Only SESSION and the three config kinds lead to a turn."""

    SESSION = "session"  # the thread already has a sessions row
    CONFIGURED = "configured"  # a config topic with this name
    SPAWN = "spawn"  # a spawn alias: 'alias.task'
    DEFAULT = "default"  # config default_topic
    UNKNOWN = "unknown"  # nothing matched; reply with unknown_text()


@dataclass(frozen=True)
class TopicTarget:
    """The result of resolve(). `config` is the topic entry a new row is created from (CONFIGURED and DEFAULT)."""

    thread_id: int
    kind: TopicKind
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    spawn_command: str = ""
    spawn_name: str = ""


def guard_text(chat_type: str | None, thread_id: int | None) -> str | None:
    """The reply for a message that cannot belong to a session, or None when the message may continue.

    The General topic (thread 1, or no thread) of a group gets the board text; a private chat without a thread gets
    the add-me text. Private chats in a topic continue.
    """
    if thread_id in (None, GENERAL_TOPIC_THREAD) and chat_type != "private":
        return GENERAL_TEXT
    if thread_id is None:
        return NO_TOPIC_TEXT
    return None


class TopicResolver:
    """Maps a thread to its session. Every collaborator is injected; holds no state of its own."""

    def __init__(self, *, config: Config, kv: KvRepo, sessions: SessionRepo, logger: Logger) -> None:
        self._config = config
        self._kv = kv
        self._sessions = sessions
        self._logger = logger

    def learn(self, thread_id: int, name: str) -> None:
        """Remember the name of a topic from its forum_topic_created message."""
        self._kv.set(f"thread:{thread_id}", name)

    def resolve(self, thread_id: int, reply_to: Mapping[str, Any] | None = None) -> TopicTarget:
        """The session a thread belongs to. reply_to is the message this one replies to, when it has one."""
        row = self._sessions.get(thread_id)
        if row is not None:
            return TopicTarget(thread_id, TopicKind.SESSION, row.name)
        name = self._learned_name(thread_id, reply_to)
        topic = self._config.topics().get(name)
        if topic:
            return self._resolved(TopicTarget(thread_id, TopicKind.CONFIGURED, name, config=dict(topic)))
        alias, sep, task = name.partition(".")
        spawn_command = self._config.spawn().get(alias) if sep and task.strip() else None
        if spawn_command:
            return self._resolved(TopicTarget(thread_id, TopicKind.SPAWN, name,
                                              spawn_command=str(spawn_command), spawn_name=task.strip()))
        default = self._config.default_topic()
        if default:
            return self._resolved(TopicTarget(thread_id, TopicKind.DEFAULT, name, config=default))
        self._logger.log(f"unresolved topic: thread {thread_id} — name learned so far: {name!r}")
        return TopicTarget(thread_id, TopicKind.UNKNOWN, name)

    def queued_session_name(self, thread_id: int) -> str | None:
        """The session name of a plain headless topic (the only kind that queues), else None.

        Pane topics (config type 'pane' or 'pane' key, or a kv pane:<name> binding) are not queued. One row read, no
        network.
        """
        row = self._sessions.get(thread_id)
        if row is None:
            return None
        topic = self._config.topics().get(row.name) or {}
        if topic.get("type") == "pane" or topic.get("pane") or self._kv.get(f"pane:{row.name}"):
            return None
        return row.name

    def unknown_text(self, target: TopicTarget) -> str:
        """The reply to an UNKNOWN topic: the name, the known topic and spawn names, and how to spawn a seat."""
        known = ", ".join(sorted(set(list(self._config.topics()) + list(self._config.spawn()))))
        return (f'Unknown topic "{target.name or target.thread_id}" (thread {target.thread_id}).\n'
                f"Known: {known}\n{UNKNOWN_TOPIC_HINT}")

    def _learned_name(self, thread_id: int, reply_to: Mapping[str, Any] | None) -> str:
        """kv 'thread:<id>' first; else the reply's forum_topic_created name, which is learned into kv."""
        name = self._kv.get(f"thread:{thread_id}") or ""
        if name:
            return name
        created = (reply_to or {}).get("forum_topic_created") or {}
        name = created.get("name") or ""
        if name:
            self.learn(thread_id, name)
        return name

    def _resolved(self, target: TopicTarget) -> TopicTarget:
        self._logger.log(f"topic resolved: thread {target.thread_id} -> {target.name}")
        return target
