"""TopicProbe (card D2): finds the topics the owner deleted in Telegram and removes them with TopicRemover.

Telegram never tells a bot that a topic was deleted, so the probe asks. editForumTopic is called on each idle topic
with PROBE_ICON, an icon id that is not a valid custom emoji, so the call can never change a topic. A live topic
answers 400 DOCUMENT_INVALID (alive). A deleted topic answers 400 TOPIC_ID_INVALID (or 'message thread not found'), and
the topic is then removed by TopicRemover with delete_in_telegram=False, because Telegram already deleted it.

A topic is probed only when it is idle: no running turn, no kept session, no live terminal claim and no daemon turn
mark. Another 400 error logs one line with the thread id and the round goes on with the next topic. A failure without a
400 code (connection failure, server error, flood wait given up) logs one line and ends the round; the next round tries
again. Calls are CALL_SPACING_SECONDS apart on the injected clock.

start() runs a round every interval seconds on a daemon thread, the first one FIRST_PROBE_SECONDS after the start;
A round removes nothing when more than half of its answering topics (and at least MASS_GONE_MIN) look deleted at once:
that points at a wrong group, and the removals are logged as not made. stop() ends the thread (App.shutdown_sessions calls it). An interval of 0 (config topic_probe_seconds) starts no thread.
"""

import threading
from dataclasses import dataclass, field

from tg_bridge.clock import Clock
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.terminal_claim import live_claim, turn_marked
from tg_bridge.core.topic_remover import TopicRemover, is_topic_gone
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.client import BAD_REQUEST, TelegramClient, TelegramError

FIRST_PROBE_SECONDS = 60.0
CALL_SPACING_SECONDS = 1.0
STOP_JOIN_SECONDS = 5.0
ERROR_TEXT_LIMIT = 150
PROBE_METHOD = "editForumTopic"
PROBE_ICON = "1"  # not a valid custom emoji id: Telegram refuses it, so a live topic never gets a new icon
ALIVE_MARKER = "document_invalid"  # the 400 description of a live topic that got the probe icon
UNEXPECTED_SUCCESS_LOG = "topic probe: unexpected success (the probe icon was accepted); topics kept"
DELETED_LOG = "topic {thread} ({name}) was deleted in Telegram: removed"
TOPIC_ERROR_LOG = "topic probe: thread {thread} failed ({error}); next topic probed"
SKIPPED_LOG = "topic probe: thread {thread} failed ({error}); round skipped"
ROUND_ERROR_LOG = "topic probe: {error}"
MASS_GONE_MIN = 3  # a mass-gone round needs at least this many deleted topics ...
MASS_GONE_LOG = "probe: {gone} of {probed} topics look deleted — not removing (wrong group?)"  # ... and over half of them


@dataclass
class _Round:
    """One round's answers: the topics Telegram says are gone (thread, name), and how many topics answered at all."""

    gone: list[tuple[int, str]] = field(default_factory=list)
    answered: int = 0


class TopicProbe:
    """Probes the idle topics of the sessions table and removes the ones Telegram deleted."""

    def __init__(self, *, sessions: SessionRepo, kv: KvRepo, live: LiveRegistry, linger: LingerService,
                 remover: TopicRemover, client: TelegramClient, group_key: str, view_dir: str, logger: Logger,
                 clock: Clock, interval: float, first_seconds: float = FIRST_PROBE_SECONDS) -> None:
        self._sessions = sessions
        self._kv = kv
        self._live = live
        self._linger = linger
        self._remover = remover
        self._client = client
        self._group_key = group_key  # the kv key that holds the forum group id
        self._view_dir = view_dir  # '' when views are off: then there are no claims or turn marks to check
        self._logger = logger
        self._clock = clock
        self._interval = interval
        self._first = first_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._success_logged = False  # the unexpected-success line is logged once

    @property
    def started(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        """Starts the round thread. Nothing starts when the interval is 0, or when the thread already runs."""
        if self._interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="topic-probe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ends the round thread after its current step; a round in progress is not waited for beyond STOP_JOIN_SECONDS."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(STOP_JOIN_SECONDS)

    def run_once(self) -> None:
        """One round over the sessions table. Nothing is removed when shutdown stops the round, or when most of the
        topics look deleted at once (a wrong group: see _remove_gone). The removals come after the whole scan."""
        group = self._kv.get(self._group_key)
        if not group:
            return
        found = _Round()
        if self._scan(int(group), found):
            self._remove_gone(found)

    def _scan(self, chat_id: int, found: "_Round") -> bool:
        """Probes the idle topics and sorts the answers. False when shutdown stopped the scan. A failure that is not a
        400 ends the scan, and the topics found so far are still handled."""
        spaced = False
        for record in self._sessions.all_records():
            if self._stop.is_set() or self._live.is_shutting_down():
                return False
            thread = record.thread_id
            if not self._idle(thread):
                continue
            if spaced:
                self._clock.sleep(CALL_SPACING_SECONDS)
            spaced = True
            try:
                self._client.call(PROBE_METHOD, chat_id=chat_id, message_thread_id=thread,
                                  icon_custom_emoji_id=PROBE_ICON)
            except TelegramError as exc:
                if exc.code != BAD_REQUEST:  # no Telegram answer (connection, server, flood): the scan ends
                    self._logger.log(SKIPPED_LOG.format(thread=thread, error=str(exc)[:ERROR_TEXT_LIMIT]))
                    return True
                found.answered += 1
                if is_topic_gone(exc):
                    found.gone.append((thread, record.name))
                elif ALIVE_MARKER not in (exc.description or "").lower():
                    self._logger.log(TOPIC_ERROR_LOG.format(thread=thread, error=str(exc)[:ERROR_TEXT_LIMIT]))
                continue
            except Exception as exc:  # not a Telegram answer at all (connection failure): the scan ends
                self._logger.log(SKIPPED_LOG.format(thread=thread, error=str(exc)[:ERROR_TEXT_LIMIT]))
                return True
            found.answered += 1
            if not self._success_logged:
                self._success_logged = True
                self._logger.log(UNEXPECTED_SUCCESS_LOG)
        return True

    def _remove_gone(self, found: "_Round") -> None:
        """Removes the topics found deleted. When at least MASS_GONE_MIN topics are gone and they are more than half of
        the topics that answered, the group is probably not the bridge's group: nothing is removed, and that is logged."""
        if not found.gone:
            return
        if len(found.gone) >= MASS_GONE_MIN and 2 * len(found.gone) > found.answered:
            self._logger.log(MASS_GONE_LOG.format(gone=len(found.gone), probed=found.answered))
            return
        for thread, name in found.gone:
            self._logger.log(DELETED_LOG.format(thread=thread, name=name))
            self._remover.remove(thread, delete_in_telegram=False)

    def _idle(self, thread: int) -> bool:
        """True when the topic has no running turn, no kept session, no live claim and no daemon turn mark."""
        if self._live.running(thread) is not None or self._live.lingering(thread) is not None:
            return False
        if self._linger.holding(thread):
            return False
        if not self._view_dir:
            return True
        return live_claim(self._view_dir, thread) is None and not turn_marked(self._view_dir, thread)

    def _loop(self) -> None:
        delay = self._first
        while not self._stop.wait(delay):
            delay = self._interval
            try:
                self.run_once()
            except Exception as exc:  # a failing round is logged; the next one still runs
                self._logger.log(ROUND_ERROR_LOG.format(error=str(exc)[:ERROR_TEXT_LIMIT]))
