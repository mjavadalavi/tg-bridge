"""Poller: the daemon's getUpdates loop, with the spool drain, the offset and the outage notice (C22).

Port of the poll loop of the live cmd_daemon and of _poll_failed / _poll_ok (tg-bridge.py:3533-3600, 3635-3665). One pass
(poll_once) does:

1. drain the spool folder;
2. getUpdates from the stored offset, with the configured long-poll timeout. A failure is logged, counted as an outage
   and retried after POLL_RETRY_SECONDS;
3. for each update: ingest it (stored and offset moved), and process it only when the ingest was new.

Outage tracking: the first failures are logged, and when Telegram is reachable again after OUTAGE_NOTICE_SECONDS, one
notice goes to the most recently active topic (SessionRepo.most_recent_thread) of the learned group. A notice is skipped
when no topic or group is known yet.

A SystemExit raised by a signal handler during the blocking getUpdates call is not an Exception, so it leaves the loop
and the daemon's shutdown runs. The Poller is used by the poll thread only, so its outage state needs no lock.
"""

import datetime
import zoneinfo
from typing import Any

from tg_bridge.clock import Clock
from tg_bridge.config import Config
from tg_bridge.daemon.inbox import GROUP_KEY, OFFSET_KEY, InboxService
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.messenger import Messenger

POLL_RETRY_SECONDS = 15  # pause after a failed getUpdates or a failed pass
OUTAGE_LOG_AFTER = 3  # consecutive failures before the outage is logged
OUTAGE_NOTICE_SECONDS = 120  # an outage shorter than this is not announced in the topic
OUTAGE_TIMEZONE = "Asia/Tehran"
OUTAGE_LOG = "telegram unreachable — polling keeps retrying"
RECOVERED_LOG = "telegram reachable again after {seconds}s"
NOTICE_SKIPPED_LOG = "outage notice skipped: no topic or group learned yet"
NOTICE_TEXT = (
    "⚠️ Telegram was unreachable {start}–{end} (Asia/Tehran), back now. "
    "Messages sent meanwhile are being processed."
)
STAMP_FORMAT = "%H:%M"


class Poller:
    """Runs the poll loop on the daemon's main thread. Every collaborator is injected."""

    def __init__(
        self,
        *,
        telegram: TelegramClient,
        inbox: InboxService,
        kv: KvRepo,
        sessions: SessionRepo,
        messenger: Messenger,
        config: Config,
        clock: Clock,
        logger: Logger,
    ) -> None:
        self._telegram = telegram
        self._inbox = inbox
        self._kv = kv
        self._sessions = sessions
        self._messenger = messenger
        self._config = config
        self._clock = clock
        self._logger = logger
        self._failures = 0  # consecutive failed getUpdates calls
        self._outage_since = 0.0  # clock.now() of the first failure of the current outage

    def run(self) -> None:
        """The loop of the daemon. It only ends through an exception that is not an Exception (a stop signal)."""
        while True:
            self.poll_once()

    def poll_once(self) -> None:
        """One pass: spool, getUpdates, ingest and process. A failing pass is logged and retried after a pause."""
        try:
            self._inbox.drain_spool()
            updates = self._get_updates()
            if updates is None:
                return
            for update in updates:
                if self._inbox.ingest_update(update):  # stored and acknowledged before any work starts
                    self._inbox.process_update(update)
        except Exception as exc:
            self._logger.log(f"poll loop: {exc}")
            self._clock.sleep(POLL_RETRY_SECONDS)

    def _get_updates(self) -> list[Any] | None:
        """One getUpdates call from the stored offset. None when it failed (already logged and paused)."""
        params: dict[str, Any] = {"timeout": self._config.poll_seconds()}
        offset = self._kv.get(OFFSET_KEY)
        if offset:
            params["offset"] = int(offset) + 1
        try:
            updates = self._telegram.call("getUpdates", **params)
        except Exception as exc:
            self._logger.log(f"poll loop: {exc}")
            self._poll_failed()
            self._clock.sleep(POLL_RETRY_SECONDS)
            return None
        if not isinstance(updates, list):
            raise ValueError("getUpdates did not return a list")
        self._poll_ok()
        return updates

    def _poll_failed(self) -> None:
        if self._failures == 0:
            self._outage_since = self._clock.now()
        self._failures += 1
        if self._failures == OUTAGE_LOG_AFTER:
            self._logger.log(OUTAGE_LOG)

    def _poll_ok(self) -> None:
        failures, since = self._failures, self._outage_since
        if failures == 0:
            return
        self._failures = 0
        if failures < OUTAGE_LOG_AFTER:
            return
        now = self._clock.now()
        self._logger.log(RECOVERED_LOG.format(seconds=int(now - since)))
        if now - since >= OUTAGE_NOTICE_SECONDS:
            self._notify_outage(since, now)

    def _notify_outage(self, since: float, now: float) -> None:
        thread = self._sessions.most_recent_thread()
        group = self._kv.get(GROUP_KEY)
        if thread is None or not group:
            self._logger.log(NOTICE_SKIPPED_LOG)
            return
        zone = zoneinfo.ZoneInfo(OUTAGE_TIMEZONE)
        start = datetime.datetime.fromtimestamp(since, zone).strftime(STAMP_FORMAT)
        end = datetime.datetime.fromtimestamp(now, zone).strftime(STAMP_FORMAT)
        self._messenger.send_text(int(group), thread, NOTICE_TEXT.format(start=start, end=end))
