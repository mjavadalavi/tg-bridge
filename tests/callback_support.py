"""Shared harness of the callback tests (STEP 49): the router wired with every tap handler over real repositories,
a FakeTelegram, a FakeClock and a TempHome. No subprocess starts: sessions are StubSession objects, the permission
service and the launcher switch are recording stubs, and the dispatcher's runner thread is never started (its queue
is driven directly). The out-worker is real: every message write of a tap is awaited with join() before asserting.
"""

import json
import os
import threading
import unittest
from dataclasses import dataclass, field
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.callbacks.approvals import ApprovalTaps
from tg_bridge.bot.callbacks.handlers import QueueTaps, StatusTaps, StopTap
from tg_bridge.bot.callbacks.picker_taps import PickerTaps
from tg_bridge.bot.callbacks.router import CallbackRouter
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.info import register_info_commands
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.config import Config
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo, InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.messenger import Messenger

THREAD = 9601
CHAT = -1009601
MID = 777
ALLOWED_UID = 42
CALLBACK_ID = "cbq-test"
TOPIC = "CB"
CLAUDE_CMD = "wrap run"
OTHER_CMD = "plain run"
MODELS = ["opus", "sonnet", "haiku", "default", "qwen3:8b"]
U1 = "1f2e3d4c-bbbb-4ccc-8ddd-000000000001"
U2 = "1f2e3d4c-bbbb-4ccc-8ddd-000000000002"


class StubSession:
    """Stands in for an AgentSession: records stop() and set_model(); stop() runs an optional hook first."""

    def __init__(self) -> None:
        self.stops = 0
        self.stopped = threading.Event()
        self.models: list[str] = []
        self.on_stop = None

    def stop(self, grace: float = 5.0) -> None:
        self.stops += 1
        if self.on_stop is not None:
            self.on_stop()
        self.stopped.set()

    def halt_turn(self) -> None:  # the ⏹ and Now taps end the turn with halt_turn, which is stop() for this stand-in
        self.stop()

    def set_model(self, model: str) -> bool:
        self.models.append(model)
        return True


class StubPermissions:
    """Records every PermissionService call a tap makes. Each answer returns `result`, so a test can expire an ask."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.result = True
        self.live_modes: list[tuple[int, str, str]] = []

    def answer_ask(self, nonce: str, allow: bool, by: str = "owner") -> bool:
        self.calls.append(("answer_ask", nonce, allow))
        return self.result

    def answer_always(self, nonce: str, by: str = "owner") -> bool:
        self.calls.append(("answer_always", nonce))
        return self.result

    def answer_bypass(self, nonce: str, by: str = "owner") -> bool:
        self.calls.append(("answer_bypass", nonce))
        return self.result

    def answer_question(self, nonce: str, qi: int, oi: int) -> bool:
        self.calls.append(("answer_question", nonce, qi, oi))
        return self.result

    def answer_other(self, nonce: str, qi: int) -> bool:
        self.calls.append(("answer_other", nonce, qi))
        return self.result

    def set_live_mode(self, thread_id: int, mode: str, by: str = "owner") -> bool:
        self.live_modes.append((thread_id, mode, by))
        return True


@dataclass
class StubLauncherSwitch:
    """Records the launcher change a /launcher tap asks for and answers with a fixed (kind, reply)."""

    kind: str = "changed"
    reply: str = "🚀 launcher: changed"
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def set(self, thread: int, name: str, live: object | None, new: str) -> tuple[str, str]:
        self.calls.append((thread, name, live, new))
        return self.kind, self.reply


class NoThread:
    """A thread stand-in for the dispatcher: start() does nothing, so a queued message stays queued."""

    def start(self) -> None:
        return None


def no_start(**kwargs: object) -> NoThread:
    return NoThread()


class CallbackCase(unittest.TestCase):
    """The router with every tap handler registered, over real repositories under a TempHome."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        self.dir_a = os.path.join(self.project, "cwd-a")
        self.dir_b = os.path.join(self.project, "cwd-b")
        for path in (self.dir_a, self.dir_b):
            os.makedirs(path, exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(self._config_data(), home=self.base, env={})
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.rules = RuleRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.live = LiveRegistry()
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.dispatcher = Dispatcher(
            turns=object(), slots=object(), inbox=InboxRepo(self.db, self.clock), messenger=self.messenger,
            answers=object(), titles=object(), client=self.telegram, live=self.live, config=self.config,
            logger=self.logger, start_thread=no_start, clock=self.clock)
        self.stops = SessionCommands(config=self.config, sessions=self.sessions, history=self.history,
                                     live=self.live, backends=self.backends, messenger=self.messenger,
                                     clock=self.clock, logger=self.logger, stop_wait=0.6,
                                     handoff=HandoffService(self.kv, self.backends, self.logger))
        self.commands = CommandRegistry()
        self.stops.register(self.commands)
        register_info_commands(self.commands)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=TurnRepo(self.db), rules=self.rules, kv=self.kv,
            live=self.live, queues=TopicQueues(), backends=self.backends, messenger=self.messenger, out=self.out)
        self.permissions = StubPermissions()
        self.launchers = StubLauncherSwitch()
        self.folders = CwdService(
            sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger, live=self.live,
            search_pool=WorkerPool("search", 1, self.logger), out_pool=self.out, logger=self.logger,
            clock=self.clock, handoff=HandoffService(self.kv, self.backends, self.logger),
            finder=lambda term, roots, deadline: [])
        self.router = CallbackRouter(config=self.config, client=self.telegram, messenger=self.messenger,
                                     out=self.out, logger=self.logger)
        self.register_all()
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "bypassPermissions")

    def _config_data(self) -> dict:
        return {
            "allow_user_ids": [ALLOWED_UID],
            "models": list(MODELS),
            "cwd_shortcuts": [self.dir_a, self.dir_b],
            "launchers": {
                "wrap": {"cmd": CLAUDE_CMD, "config_dir": os.path.join(self.project, "cfg-a")},
                "plain": {"cmd": OTHER_CMD, "config_dir": os.path.join(self.project, "cfg-b")},
            },
            "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": self.project}},
        }

    def register_all(self) -> None:
        """Every tap handler of the step, as the App registers them (the order does not matter to the router)."""
        QueueTaps(dispatcher=self.dispatcher, live=self.live).register(self.router)
        StopTap(stops=self.stops).register(self.router)
        StatusTaps(deps=self.deps, registry=self.commands, stops=self.stops).register(self.router)
        PickerTaps(config=self.config, sessions=self.sessions, live=self.live, backends=self.backends,
                   permissions=self.permissions, launchers=self.launchers, stops=self.stops, rules=self.rules,
                   folders=self.folders, logger=self.logger).register(self.router)
        ApprovalTaps(permissions=self.permissions).register(self.router)

    # --- driving taps and reading what they did

    def query(self, data: str, tapper: int = ALLOWED_UID, mid: int | None = MID, text: str = "card text") -> dict:
        message: dict[str, Any] = {"chat": {"id": CHAT}, "text": text, "message_thread_id": THREAD}
        if mid is not None:
            message["message_id"] = mid
        return {"id": CALLBACK_ID, "from": {"id": tapper}, "data": data, "message": message}

    def tap(self, data: str, **kwargs: Any) -> None:
        """Dispatches one callback and waits until the out-worker has written what the tap queued."""
        self.router.dispatch(self.query(data, **kwargs))
        self.assertTrue(self.out.join(5), "the tap's writes did not go out")

    def calls(self, method: str) -> list[dict[str, Any]]:
        return [c.params for c in self.telegram.calls if c.method == method]

    def answers(self) -> list[str]:
        return [str(p.get("text")) for p in self.calls("answerCallbackQuery")]

    def edits(self, mid: int = MID) -> list[dict[str, Any]]:
        return [p for p in self.calls("editMessageText") if p.get("message_id") == mid]

    def sent_texts(self) -> list[str]:
        return [str(p.get("text", "")) for p in self.calls("sendMessage") if str(p.get("message_thread_id")) == str(THREAD)]

    def sent_buttons(self) -> list[dict[str, Any]]:
        return [p for p in self.calls("sendMessage") if p.get("reply_markup") is not None
                and str(p.get("message_thread_id")) == str(THREAD)]

    def callbacks_of(self, markup: str) -> list[str]:
        return [b["callback_data"] for row in json.loads(markup)["inline_keyboard"] for b in row]

    def labels_of(self, markup: str) -> list[str]:
        return [b["text"] for row in json.loads(markup)["inline_keyboard"] for b in row]

    def start_turn(self, session: StubSession | None = None) -> StubSession:
        """A running turn on THREAD: the session and a feed token are registered like a real turn does."""
        session = session or StubSession()
        self.feed = object()
        self.assertTrue(self.live.begin_turn(THREAD, session, self.feed))
        return session

    def end_turn(self, session: StubSession) -> None:
        self.live.end_turn(THREAD, session, self.feed)

    def logged(self, fragment: str) -> bool:
        with open(self.logger.path, encoding="utf-8") as handle:
            return fragment in handle.read()
