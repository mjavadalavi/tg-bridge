"""TopicRemover: forgets a topic completely (the daemon side of deleting a topic).

remove(thread_id, delete_in_telegram) runs in this order:
0. a daemon that is shutting down refuses ('the bridge is shutting down'). Otherwise the topic is marked as being removed
   (Removals, shared with the Dispatcher): from now on its messages are dropped and no turn starts for it;
1. the queued messages of the topic are dropped (Dispatcher.clear_queue), their inbox rows are marked done, so they do
   not replay at the next start;
2. the topic's session is stopped and released the way /stop does it. Every round of the wait, a running turn (or a self
   turn) is marked as stopped on purpose and its session is stopped, and an idle kept session is asked to close
   (LingerService.request_release); a kept session in a turn refuses the release until the turn ends, so the request is
   repeated each round. The wait (bounded, STOP_WAIT_SECONDS) ends when the topic's runner is idle (it covers a turn
   that was taken but not yet registered), no turn is live and no session is kept. Otherwise the removal is refused:
   nothing is forgotten and the text says why;
3. when delete_in_telegram: deleteForumTopic in the forum group. A 400 'TOPIC_ID_INVALID' or 'message thread not found'
   means the topic is already gone and counts as removed. Any other error refuses the removal (nothing is forgotten);
4. TopicForget.forget_topic (one DB transaction; turns and other topics are kept). The mark stays for a grace period, so
   a late update cannot re-create the topic's row;
5. the view files of the topic are removed, except its .lock file (a flock file is never unlinked while it may be held)
   and, while a terminal session is open on the topic, its .term claim: that window closes with the topic by itself.

A refusal clears the mark at once. It returns one line for the logs, 'topic <id> removed (telegram: deleted|already
gone|kept)', or the refusal text.
Threads: the caller's (a topic removal is requested from a callback, a probe or the view). The wait sleeps on the injected
clock, so the tests run it without real time.
"""

import os

from tg_bridge.clock import Clock
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.removals import Removals
from tg_bridge.core.terminal_claim import CLAIM_SUFFIX, LOCK_SUFFIX, live_claim
from tg_bridge.logging_ import Logger
from tg_bridge.store.forget import TopicForget
from tg_bridge.store.kv import KvRepo
from tg_bridge.telegram.client import TelegramClient, TelegramError

STOP_WAIT_SECONDS = 15.0  # how long a removal waits for the turn and the kept session to end
STOP_POLL_SECONDS = 0.2
SHUTDOWN_TEXT = "the bridge is shutting down"
BUSY_TEXT = "a removal of this topic is already running"
TERMINAL_OPEN_TEXT = "terminal session open: its window closes with the topic"
BAD_REQUEST = 400
TOPIC_GONE_MARKERS = ("topic_id_invalid", "message thread not found")
DELETED = "deleted"
ALREADY_GONE = "already gone"
KEPT = "kept"
REQUEST_SUFFIX = ".delete"  # the view's delete request (card D3): written by view-delete, taken by DeleteRequests
RESULT_SUFFIX = ".deleted"  # its one-line answer, read and removed by view-delete
ACK_SUFFIX = ".deleting"  # the daemon's sign that it has taken the request and is working on it (removed when answered)


class _Refused(RuntimeError):
    """A step failed in a way that must not forget the topic; the message is the reason shown to the caller."""


def is_topic_gone(exc: TelegramError) -> bool:
    if exc.code != BAD_REQUEST:
        return False
    description = (exc.description or "").lower()
    return any(marker in description for marker in TOPIC_GONE_MARKERS)


def remove_view_files(view_dir: str, thread_id: int, keep_claim: bool = False) -> int:
    """Removes the topic's files in the view folder (<thread>.json, .feed, .pty, .term, .turn, ...) and keeps <thread>.lock
    and the delete request and answer files (.delete, .deleting, .deleted); keep_claim also keeps the .term claim of an open terminal
    session. Returns how many files were removed. An empty view_dir (views off) removes nothing."""
    if not view_dir:
        return 0
    try:
        names = os.listdir(view_dir)
    except FileNotFoundError:
        return 0
    prefix = f"{thread_id}."
    kept = (LOCK_SUFFIX, REQUEST_SUFFIX, ACK_SUFFIX, RESULT_SUFFIX, *((CLAIM_SUFFIX,) if keep_claim else ()))
    removed = 0
    for name in names:
        if not name.startswith(prefix) or name.endswith(kept):
            continue
        try:
            os.remove(os.path.join(view_dir, name))
        except FileNotFoundError:
            continue
        removed += 1
    return removed


class TopicRemover:
    """Removes one topic: its session, queue, Telegram topic (optional), DB rows, kv keys and view files."""

    def __init__(self, *, live: LiveRegistry, linger: LingerService, dispatcher: Dispatcher, forget: TopicForget,
                 client: TelegramClient, kv: KvRepo, group_key: str, view_dir: str, logger: Logger,
                 clock: Clock, removals: Removals) -> None:
        self._live = live
        self._linger = linger
        self._dispatcher = dispatcher
        self._forget = forget
        self._client = client
        self._kv = kv
        self._group_key = group_key  # the kv key that holds the forum group id (the bridge learns it)
        self._view_dir = view_dir  # '' when views are off
        self._logger = logger
        self._clock = clock
        self._removals = removals  # the marks shared with the Dispatcher (the same instance)

    def remove(self, thread_id: int, delete_in_telegram: bool) -> str:
        """Forgets the topic. Returns the log line for a removal, or the reason it was refused (nothing forgotten)."""
        if self._live.is_shutting_down():
            return self._refuse(thread_id, SHUTDOWN_TEXT)
        if not self._removals.begin(thread_id):
            return self._refuse(thread_id, BUSY_TEXT)
        try:
            return self._remove_marked(thread_id, delete_in_telegram)
        except BaseException:  # an unexpected failure must not leave the topic marked for good
            self._removals.cancel(thread_id)
            raise

    def _remove_marked(self, thread_id: int, delete_in_telegram: bool) -> str:
        self._dispatcher.clear_queue(thread_id)  # after the mark: no message delivered from now on can start a turn
        if not self._stop_session(thread_id):
            return self._refused(thread_id, "its session did not stop")
        try:
            outcome = self._telegram_outcome(thread_id, delete_in_telegram)
        except _Refused as exc:
            return self._refused(thread_id, str(exc))
        open_terminal = self._terminal_open(thread_id)
        self._forget.forget_topic(thread_id)
        self._removals.finish(thread_id)  # the mark stays for the grace period
        remove_view_files(self._view_dir, thread_id, keep_claim=open_terminal)
        line = f"topic {thread_id} removed (telegram: {outcome})"
        self._logger.log(line)
        return line

    def _stop_session(self, thread_id: int) -> bool:
        """Stops the topic's running turn, or releases its idle kept session, the way /stop does it. Each round asks again,
        since a kept session in a turn or a self turn refuses the release until it ends. True once the runner is idle and
        nothing runs or is kept there; False when that did not happen within STOP_WAIT_SECONDS."""
        deadline = self._clock.monotonic() + STOP_WAIT_SECONDS
        while True:
            running = self._live.request_abort(thread_id, stop_tap=True)
            if running is not None:
                running.session.stop()
            else:
                self._linger.request_release(thread_id)  # False when nothing is kept, or the kept session is busy
            if not self._busy(thread_id):
                return True
            if self._clock.monotonic() >= deadline:
                return False
            self._clock.sleep(STOP_POLL_SECONDS)

    def _busy(self, thread_id: int) -> bool:
        return (self._dispatcher.busy(thread_id) or self._live.running(thread_id) is not None
                or self._live.lingering(thread_id) is not None or self._linger.holding(thread_id))

    def _terminal_open(self, thread_id: int) -> bool:
        """True when a terminal session (tg-bridge view) holds the topic. It does not refuse the removal: the owner asked
        for it, and the terminal window closes with the topic."""
        if not self._view_dir:
            return False
        try:
            claim = live_claim(self._view_dir, thread_id)
        except OSError as exc:  # the claim is best-effort here: the removal goes on, its file is removed as before
            self._logger.log(f"topic {thread_id}: terminal claim check failed: {str(exc)[:120]}")
            return False
        if claim is not None:
            self._logger.log(f"topic {thread_id}: {TERMINAL_OPEN_TEXT}")
        return claim is not None

    def _telegram_outcome(self, thread_id: int, delete_in_telegram: bool) -> str:
        """DELETED or ALREADY_GONE after a successful deleteForumTopic, KEPT when no Telegram call was asked for.
        Raises _Refused for any other failure."""
        if not delete_in_telegram:
            return KEPT
        group = self._kv.get(self._group_key)
        if not group:
            raise _Refused("no forum group is known yet, so the Telegram topic cannot be deleted")
        try:
            self._client.call("deleteForumTopic", chat_id=int(group), message_thread_id=thread_id)
        except Exception as exc:  # any failure refuses the removal; a network error is as final as an API one
            if isinstance(exc, TelegramError) and is_topic_gone(exc):
                return ALREADY_GONE
            raise _Refused(f"the Telegram deletion failed: {str(exc)[:200]}") from exc
        return DELETED

    def _refused(self, thread_id: int, reason: str) -> str:
        self._removals.cancel(thread_id)  # a refused removal clears its mark at once
        return self._refuse(thread_id, reason)

    def _refuse(self, thread_id: int, reason: str) -> str:
        line = f"topic {thread_id} not removed: {reason}"
        self._logger.log(line)
        return line
