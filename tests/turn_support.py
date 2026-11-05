"""Shared fixtures for the turn runner tests: a harness with every service a TurnRunner needs, recording doubles for
the feed and the workers, a FakeBackend that records what the database held when a session opened, and two helpers
that wait for a turn to end while the FakeClock moves (a dead process or a cap is only seen when time passes).

Used by tests/test_turn_runner.py (FakeBackend) and tests/test_turn_runner_e2e.py (real child processes).
"""

import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from tests.fakes.backend import FakeBackend
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.fakes.uploads import FakeUploader
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.config import Config
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.self_turn import SelfTurns
from tg_bridge.core.slots import SlotPool
from tg_bridge.core.turn_listeners import WaitRouter
from tg_bridge.core.turn_runner import TurnRunner
from tg_bridge.core.view_recorder import ViewFactory
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.messenger import Messenger

THREAD = 9101
CHAT = -100
TOPIC = "side"
DEFAULT_CONFIG: dict[str, Any] = {
    "launchers": {"fake": "fake-cmd", "alt": "alt-cmd", "tlauncher": "topic-cmd"},
    "topics": {TOPIC: {"launcher": "topic-cmd"}},
    "max_turn_seconds": 1800,
    "hard_cap_seconds": 21600,
}


class RecordingTimer:
    """A nudge timer that never fires on its own; the tests do not need the nudge."""

    def __init__(self, seconds: float, fire: Callable[[], None]) -> None:
        self.seconds = seconds

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass


class InlineWorkers:
    """Runs each submitted job at once, on the calling thread."""

    def submit(self, fn: Callable[..., object], *args: object) -> None:
        fn(*args)


class RecordingFeed:
    """A TurnProgress that keeps what it was given: the events in order, the start, the close and the summaries."""

    def __init__(self) -> None:
        self.events: list[object] = []
        self.started = False
        self.closed = False
        self.headers: list[str] = []

    def start(self) -> None:
        self.started = True

    def on_event(self, event: object) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True

    def finish(self, header: str) -> None:
        self.headers.append(header)


class FeedRecorder:
    """The feed factory of the tests: one RecordingFeed per attempt, kept in creation order."""

    def __init__(self) -> None:
        self.feeds: list[RecordingFeed] = []

    def __call__(self, chat_id: int | str, thread_id: int, parked: Callable[[], bool]) -> RecordingFeed:
        feed = RecordingFeed()
        self.feeds.append(feed)
        return feed

    @property
    def last(self) -> RecordingFeed:
        return self.feeds[-1]


class ProbingBackend(FakeBackend):
    """A FakeBackend that records the topic's stored session id at the moment a session is opened (before any spawn)."""

    probe: Callable[[], str] | None = None
    probed: str | None = None

    def open_session(self, spec, sink):  # type: ignore[override]
        if self.probe is not None:
            self.probed = self.probe()
        return super().open_session(spec, sink)


class TurnHarness:
    """Every service a TurnRunner is built with, on real repositories in a migrated database under `home`."""

    def __init__(self, home: str, config_data: Mapping[str, Any] | None = None,
                 base_env: Mapping[str, str] | None = None, poll_seconds: float = 0.005,
                 clock: FakeClock | None = None) -> None:
        self.home = home
        self.clock = clock if clock is not None else FakeClock()
        self.log_path = os.path.join(home, "bridge.log")
        self.logger = Logger(self.log_path)
        self.tg = FakeTelegram(clock=self.clock)
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.db = Database(Paths.from_home(home))
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.rules = RuleRepo(self.db, self.clock)
        self.turns = TurnRepo(self.db)
        self.kv = KvRepo(self.db)
        self.asks = AskRepo(self.db, self.clock)
        self.live = LiveRegistry()
        self.waits = WaitRouter()
        cards = ApprovalCards(self.messenger, self.logger, self.clock, timer_factory=RecordingTimer)
        self.permissions = PermissionService(
            asks=self.asks, rules=self.rules, cards=cards, messenger=self.messenger, workers=InlineWorkers(),
            waiting=self.waits, logger=self.logger)
        self.slots = SlotPool(2, self.messenger, self.live.is_shutting_down, poll_seconds=0.01)
        self.feeds = FeedRecorder()
        self.config = Config(dict(DEFAULT_CONFIG, **(config_data or {})), home=home, env={})
        self.errfile_dir = os.path.join(home, "errs")
        os.makedirs(self.errfile_dir, exist_ok=True)
        self.base_env = dict(base_env or {})
        self.poll_seconds = poll_seconds

    def set_config(self, **overrides: Any) -> None:
        """Replaces the config with the defaults plus these keys. Call before registry_for()/runner()."""
        self.config = Config(dict(DEFAULT_CONFIG, **overrides), home=self.home, env={})

    def registry_for(self, backend: FakeBackend | None = None, *, name: str = "claude") -> BackendRegistry:
        """A registry whose `name` backend is the given fake (or any backend when None: the caller registers it)."""
        registry = BackendRegistry(self.config, self.logger, self.clock)
        if backend is not None:
            registry.register(name, lambda config, logger, clock: backend)
        return registry

    def runner(self, registry: BackendRegistry, views: ViewFactory | None = None, view_dir: str = "",
               linger: LingerService | None = None, memory_dir: str = "") -> TurnRunner:
        return TurnRunner(
            config=self.config, registry=registry, live=self.live, permissions=self.permissions, waits=self.waits,
            sessions=self.sessions, history=self.history, rules=self.rules, turns=self.turns, kv=self.kv,
            handoff=HandoffService(self.kv, registry, self.logger), feeds=self.feeds, clock=self.clock, logger=self.logger, base_env=self.base_env,
            errfile_dir=self.errfile_dir, poll_seconds=self.poll_seconds, views=views, view_dir=view_dir,
            linger=linger, memory_dir=memory_dir)

    def linger(self, registry: BackendRegistry) -> LingerService:
        """A LingerService built from the config as it is now, with a real SelfTurns (its 🔔 replies go to the fake
        Telegram). Build it again after a set_config(): the service reads the config it was given."""
        handoff = HandoffService(self.kv, registry, self.logger)
        self_turns = SelfTurns(
            config=self.config, clock=self.clock, logger=self.logger, feeds=self.feeds, views=None,
            permissions=self.permissions, handoff=handoff, waits=self.waits, live=self.live, turns=self.turns,
            answers=AnswerSender(self.tg, FakeUploader(), self.messenger, self.logger, self.clock),
            messenger=self.messenger, poll_seconds=self.poll_seconds)
        return LingerService(
            config=self.config, sessions=self.sessions, kv=self.kv, live=self.live, permissions=self.permissions,
            self_turns=self_turns, clock=self.clock, logger=self.logger, poll_seconds=self.poll_seconds)

    def log_lines(self) -> list[str]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read().splitlines()


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    """True once predicate() holds, False after timeout seconds. Never advances the FakeClock."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def join_turn(clock: FakeClock, thread: threading.Thread, timeout: float = 20.0) -> None:
    """Waits for a turn thread to end while the FakeClock moves on, so a dead process or a cap is noticed. Fails the
    test (AssertionError) when the turn does not end within timeout real seconds."""
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        clock.advance(1.0)
        thread.join(0.005)
        if time.monotonic() > deadline:
            raise AssertionError("the turn did not end")


def run_async(runner: TurnRunner, **kwargs: Any) -> tuple[threading.Thread, list[Any]]:
    """Runs one turn on a thread. The returned list gets the TurnReply when the turn ends."""
    box: list[Any] = []
    thread = threading.Thread(target=lambda: box.append(runner.run(**kwargs)), daemon=True)
    thread.start()
    return thread, box
