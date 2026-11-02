"""Shared harness of the inbound tests (STEP 50): a MessageHandler wired over real repositories, a real Dispatcher whose
runner thread is never started (its queue is driven directly), a real CwdService and CommandRegistry, and recording
stubs for the parts that would start a process or call tmux: the permission service, the media service and the tmux
seats. Replies go through a real out-worker, so every test joins it before it reads what was sent.
"""

import os
import threading
import unittest
from dataclasses import dataclass, field
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import Command, CommandContext, CommandDeps, CommandRegistry
from tg_bridge.bot.inbound import MessageHandler
from tg_bridge.config import Config
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.media import MediaOutcome
from tg_bridge.core.removals import Removals
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.topics import TopicResolver
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo, InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.messenger import Messenger

CHAT = -1009701
THREAD = 9701
OTHER_THREAD = 9702
ALLOWED_UID = 42
STRANGER_UID = 7
CLAUDE_CMD = "wrap run"
SPAWN_CMD = "/opt/bin/budget-claude"
SPAWN_LAUNCHER = "budget-claude"
TIMEOUT = 5
TOPIC = "side"
PANE_TOPIC = "seat"
PANE = "%7"
WAIT_PANE = "%9"
SPAWN_CWD = "/spawn/cwd"
TUI_TOPIC = "deck"
TUI_CMD = "tui run"  # the command of a tui launcher in config, which a session row names


class StartRecorder:
    """Stands in for threading.Thread in the dispatcher: records each runner start, never runs it."""

    def __init__(self) -> None:
        self.started: list[tuple[int, str]] = []

    def __call__(self, **kwargs: Any) -> "_Start":
        return _Start(self, kwargs["args"])


class _Start:
    def __init__(self, owner: StartRecorder, args: tuple[int, str]) -> None:
        self._owner = owner
        self._args = args

    def start(self) -> None:
        self._owner.started.append(self._args)


class StubPermissions:
    """Records the awaited-text check and the live mode switches. record_text answers with `awaited`."""

    def __init__(self) -> None:
        self.awaited = False
        self.texts: list[tuple[int, str]] = []
        self.live_modes: list[tuple[int, str, str]] = []
        self.switched = threading.Event()

    def record_text(self, thread_id: int, text: str) -> bool:
        self.texts.append((thread_id, text))
        return self.awaited

    def set_live_mode(self, thread_id: int, mode: str, by: str = "owner") -> bool:
        self.live_modes.append((thread_id, mode, by))
        self.switched.set()
        return True


@dataclass
class StubMedia:
    """Records every fetch; answers with `outcome` (None: the download failed and the user was told)."""

    outcome: MediaOutcome | None = MediaOutcome(text="heard you", images=())
    calls: list[tuple[str, bool]] = field(default_factory=list)
    error: Exception | None = None

    def fetch(self, message: dict, text: str, native_images: bool) -> MediaOutcome | None:
        self.calls.append((text, native_images))
        if self.error is not None:
            raise self.error
        return self.outcome


class StubSeats:
    """The tmux side: records every call. Answers with the scripted results."""

    spawn_cwd = SPAWN_CWD

    def __init__(self) -> None:
        self.typed: list[tuple[str, str]] = []
        self.spawned: list[tuple[str, str, str, str, str]] = []
        self.type_result: tuple[bool, str] = (True, f"typed into pane {PANE}")
        self.spawn_result: tuple[bool, str, str | None] = (True, f"typed into pane {WAIT_PANE}", WAIT_PANE)

    def type_to_pane(self, pane: str, text: str) -> tuple[bool, str]:
        self.typed.append((pane, text))
        return self.type_result

    def spawn_seat(self, command: str, session_name: str, topic: str, sender: str,
                   text: str) -> tuple[bool, str, str | None]:
        self.spawned.append((command, session_name, topic, sender, text))
        return self.spawn_result


class StubSession:
    """A live agent session stand-in for LiveRegistry.begin_turn."""


def pong(ctx: CommandContext, args: str) -> None:
    ctx.say(f"pong {args}".strip())


class InboundCase(unittest.TestCase):
    """The handler over real repositories under a TempHome. Subclasses use self.handle() and the readers below."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        self.dir_a = os.path.join(self.project, "cwd-a")
        os.makedirs(self.dir_a, exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(self._config_data(), home=self.base, env={})
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.inbox = InboxRepo(self.db, self.clock)
        self.live = LiveRegistry()
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.media_pool = WorkerPool("media", 2, self.logger)
        self.starts = StartRecorder()
        self.removals = Removals(self.clock)  # shared with the dispatcher, as the App shares it with TopicRemover
        self.dispatcher = Dispatcher(
            turns=object(), slots=object(), inbox=self.inbox, messenger=self.messenger, answers=object(),
            titles=object(), client=self.telegram, live=self.live, config=self.config, logger=self.logger,
            start_thread=self.starts, removals=self.removals)
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.topics = TopicResolver(config=self.config, kv=self.kv, sessions=self.sessions, logger=self.logger)
        self.permissions = StubPermissions()
        self.media = StubMedia()
        self.seats = StubSeats()
        self.folders = CwdService(
            sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger, live=self.live,
            search_pool=WorkerPool("search", 1, self.logger), out_pool=self.out, logger=self.logger,
            clock=self.clock, handoff=HandoffService(self.kv, self.backends, self.logger),
            finder=lambda term, roots, deadline: [])
        self.commands = CommandRegistry()
        self.commands.register(Command("ping", "answer pong", pong))
        self.rules = RuleRepo(self.db, self.clock)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=TurnRepo(self.db), rules=self.rules, kv=self.kv,
            live=self.live, queues=TopicQueues(), backends=self.backends, messenger=self.messenger, out=self.out)
        self.handler = MessageHandler(
            config=self.config, topics=self.topics, sessions=self.sessions, kv=self.kv, inbox=self.inbox,
            dispatcher=self.dispatcher, permissions=self.permissions, live=self.live, cwd=self.folders,
            commands=self.commands, command_deps=self.deps, backends=self.backends, messenger=self.messenger,
            out=self.out, media=self.media, media_pool=self.media_pool, seats=self.seats, logger=self.logger)

    def _config_data(self) -> dict:
        return {
            "allow_user_ids": [ALLOWED_UID],
            "topics": {
                TOPIC: {"launcher": CLAUDE_CMD, "cwd": self.project},
                PANE_TOPIC: {"type": "pane", "pane": PANE, "label": "seat-label"},
            },
            "launchers": {"tui": {"cmd": TUI_CMD, "config_dir": os.path.join(self.project, "cfg-tui"),
                                  "transport": "tui"}},
            "spawn": {"budget": SPAWN_CMD},
            "agent_launchers": {"codex": {"backend": "codex", "command": "codex",
                                          "config_dir": os.path.join(self.project, "cfg-codex")}},
        }

    # --- driving the handler

    def message(self, text: str = "", thread: int | None = THREAD, uid: int = ALLOWED_UID,
                chat_type: str = "supergroup", **extra: Any) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "message_id": 1, "chat": {"id": CHAT, "type": chat_type},
            "from": {"id": uid, "first_name": "Tester", "username": "tester"},
            "text": text,
        }
        if thread is not None:
            msg["message_thread_id"] = thread
        msg.update(extra)
        return msg

    def handle(self, msg: dict[str, Any], update_id: int | None = None) -> Any:
        """Runs handle() and waits until the out-worker and the media worker have finished what they queued."""
        disposition = self.handler.handle(msg, update_id)
        self.settle()
        return disposition

    def settle(self) -> None:
        self.assertTrue(self.out.join(TIMEOUT), "replies did not go out")
        self.assertTrue(self.media_pool.join(TIMEOUT), "media jobs did not finish")

    def name_topic(self, thread: int, name: str) -> None:
        """What a forum_topic_created message would have learned for the thread."""
        self.kv.set(f"thread:{thread}", name)

    def start_turn(self, thread: int) -> None:
        self.assertTrue(self.live.begin_turn(thread, StubSession(), object()))

    # --- reading what was done

    def sent(self, thread: int = THREAD) -> list[str]:
        return self.telegram.texts(thread)

    def keyboards(self, thread: int = THREAD) -> list[list[list[dict[str, Any]]]]:
        return self.telegram.buttons(thread)

    def row(self, thread: int = THREAD) -> Any:
        return self.sessions.get(thread)

    def queued(self, thread: int = THREAD) -> list[Any]:
        return list(self.dispatcher.queue_snapshot(thread))

    def logged(self, fragment: str) -> bool:
        with open(self.logger.path, encoding="utf-8") as handle:
            return fragment in handle.read()
