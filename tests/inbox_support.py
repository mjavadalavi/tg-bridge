"""Shared harness of the inbox and poller tests (STEP 51): a real migrated Database with the repositories the inbox uses,
an InboxService and a Messenger over FakeTelegram and FakeClock, recording stubs for the message handler and the
callback router, and builders for Telegram-shaped updates. Everything lives in the TempHome the test opened.
"""

import os
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.bot.inbound import Disposition
from tg_bridge.daemon.inbox import InboxService
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.messenger import Messenger

CHAT = -100
GROUP_CHAT: dict[str, Any] = {"id": CHAT, "type": "supergroup", "is_forum": True}
THREAD = 9101
USER = 42


def message_update(update_id: int, message_id: int, thread: int | None = THREAD, text: str = "hi",
                   chat: dict[str, Any] | None = None) -> dict[str, Any]:
    """A text message from the allow-listed user, in a forum topic of the group unless told otherwise."""
    message: dict[str, Any] = {"message_id": message_id, "chat": dict(chat or GROUP_CHAT),
                               "from": {"id": USER, "first_name": "T"}, "text": text}
    if thread is not None:
        message["message_thread_id"] = thread
    return {"update_id": update_id, "message": message}


def callback_update(update_id: int, data: str = "pa:abcd") -> dict[str, Any]:
    """A callback tap from the allow-listed user."""
    return {"update_id": update_id,
            "callback_query": {"id": f"cb{update_id}", "data": data, "from": {"id": USER}}}


class RecordingMessages:
    """Stands in for the message handler: records (message_id, update_id) and answers with a scripted outcome."""

    def __init__(self) -> None:
        self.seen: list[tuple[Any, Any]] = []
        self.outcomes: dict[Any, Disposition | BaseException] = {}  # by message_id; DONE when not scripted

    def handle(self, message: dict[str, Any], update_id: int | None = None) -> Disposition:
        message_id = message.get("message_id")
        self.seen.append((message_id, update_id))
        outcome = self.outcomes.get(message_id, Disposition.DONE)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingCallbacks:
    """Stands in for the callback router: records the query ids it was given; raises `fail` when set."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.fail: BaseException | None = None

    def dispatch(self, query: dict[str, Any]) -> None:
        self.queries.append(str(query.get("id")))
        if self.fail is not None:
            raise self.fail


class InboxHarness:
    """The inbox wired over real repositories. kv_class lets a test watch the kv writes (the offset)."""

    def __init__(self, home: TempHome, kv_class: type[KvRepo] = KvRepo) -> None:
        self.home = home
        self.paths = Paths.from_home(home.home)
        self.clock = FakeClock()
        self.allowed: list[int] = [USER]  # the allowlist the service reads; a test may change it
        self.db = Database(self.paths)
        self.kv = kv_class(self.db)
        self.sessions = SessionRepo(self.db, self.clock)
        self.inbox = InboxRepo(self.db, self.clock)
        self.tg = FakeTelegram(clock=self.clock)
        self.log_path = os.path.join(home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.messages = RecordingMessages()
        self.callbacks = RecordingCallbacks()
        self.service = InboxService(
            inbox=self.inbox,
            kv=self.kv,
            sessions=self.sessions,
            messenger=self.messenger,
            messages=self.messages,
            callbacks=self.callbacks,
            spool_dir=self.paths.spool,
            clock=self.clock,
            logger=self.logger,
            allowed_user_ids=lambda: self.allowed,
        )

    def close(self) -> None:
        self.db.close()

    def state(self, update_id: int) -> str | None:
        """The inbox state of one update ('pending' or 'done'), or None when it has no row."""
        record = self.inbox.get(update_id)
        return record.state if record else None

    def log_text(self) -> str:
        """Everything the bridge logged so far."""
        try:
            with open(self.log_path, encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            return ""
