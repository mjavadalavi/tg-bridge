"""App: the composition root (DESIGN sections 2 and 5). Builds every service once and connects it by constructor injection.

App.build() is the only place where services are created and wired. Nothing is looked up from a module global, and the
App the caller holds is the only process-wide object (cli.main() in STEP 54). Shutdown is the LiveRegistry flag: once
begin_shutdown() sets it, no turn can start, and the services see it through LiveRegistry.is_shutting_down.

Test hook: TG_BRIDGE_FAKE_TG=1 makes the Telegram client a FakeTelegram (tests/fakes/telegram.py) when the paths are not
the real user's home. On the real home the variable is ignored with one log line, so a stray variable can never blind
the real daemon. The hook is off when a client is passed in explicitly.
"""

import os
import pwd
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.agents.claude.attached_session import pane_idle, pane_reason
from tg_bridge.agents.defaults import build_default_registry, title_launcher_from_env
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.callbacks.approvals import ApprovalTaps
from tg_bridge.bot.callbacks.handlers import QueueTaps, StatusTaps, StopChoiceTaps, StopTap
from tg_bridge.bot.callbacks.drive_taps import DriveTaps
from tg_bridge.bot.callbacks.keep_taps import KeepTaps
from tg_bridge.bot.callbacks.picker_taps import PickerTaps
from tg_bridge.bot.callbacks.router import CallbackRouter
from tg_bridge.bot.callbacks.terminal_taps import TerminalTaps
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.driving import DriveCommand
from tg_bridge.bot.commands.info import register_info_commands
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.bot.commands.settings import SettingsCommands
from tg_bridge.bot.inbound import MessageHandler
from tg_bridge.bot.keep_guard import KeepGuard
from tg_bridge.bot.terminal_guard import TerminalGuard
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.config import BACKEND_CLAUDE, Config
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.driving import DrivingMode, DrivingVoice, drive_note
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.launcher_switch import LauncherSwitch
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.media import MediaService, SonioxTranscriber
from tg_bridge.core.removals import Removals
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.plan_approval import PlanFollowUp
from tg_bridge.core.self_turn import SelfTurns
from tg_bridge.core.slots import SLOT_COUNT, SlotPool
from tg_bridge.core.speech import Speech
from tg_bridge.core.terminal_claim import clear_turn_marks
from tg_bridge.core.titles import TitleService
from tg_bridge.core.topics import TopicResolver
from tg_bridge.core.turn_listeners import WaitRouter
from tg_bridge.core.delete_requests import DeleteRequests
from tg_bridge.core.topic_probe import TopicProbe
from tg_bridge.core.topic_remover import TopicRemover
from tg_bridge.core.turn_runner import TurnRunner
from tg_bridge.core.view_recorder import ViewRecorder, mark_stale_views
from tg_bridge.core.workers import WorkerPool
from tg_bridge.daemon.inbox import GROUP_KEY, InboxService
from tg_bridge.daemon.lifecycle import Lifecycle
from tg_bridge.daemon.panes import Panes
from tg_bridge.daemon.poller import Poller
from tg_bridge.feed.feed import TurnFeed
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.forget import TopicForget
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo, InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.client import HttpTelegramClient, TelegramClient
from tg_bridge.telegram.messenger import Messenger
from tg_bridge.telegram.proxy import OpenerCache, proxy_url
from tg_bridge.telegram.ratelimit import RateGate
from tg_bridge.telegram.upload import Uploader

FAKE_TELEGRAM_ENV = "TG_BRIDGE_FAKE_TG"
FAKE_TELEGRAM_IGNORED_LOG = "{env} ignored: HOME is the real home"
OUT_THREADS = 1       # the ordered out-worker: every reply and edit goes through it
MEDIA_THREADS = 2     # photo and voice downloads
SEARCH_THREADS = 1    # /cwd folder searches
TITLE_THREADS = 1     # one-time topic titles, one at a time


def real_home() -> str:
    """The home directory of the current user from the password database (not from HOME). Empty when unknown."""
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return ""


def fake_telegram_requested(paths: Paths, env: Mapping[str, str], log: Callable[[str], None]) -> bool:
    """True when TG_BRIDGE_FAKE_TG=1 asks for the fake client and HOME is not the real home.

    Fails closed: when the real home cannot be read, the hook is refused. A refusal on the real home is logged once.
    """
    if env.get(FAKE_TELEGRAM_ENV) != "1":
        return False
    home = os.path.realpath(paths.home)
    real = real_home()
    if not real or home == os.path.realpath(real):
        log(FAKE_TELEGRAM_IGNORED_LOG.format(env=FAKE_TELEGRAM_ENV))
        return False
    return True


def ensure_view_dir(path: str, logger: Logger) -> bool:
    """Creates the view folder (mode 0700). When it cannot be created the daemon runs with the views off: one warning
    line in the log, and no error."""
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError as exc:
        logger.log(f"warning: view folder {path} cannot be created, views are off: {exc!r}")
        return False
    return True


def _new_fake_telegram(clock: Clock) -> TelegramClient:
    # Imported here, so the production package loads the test fake only when the guarded hook is on.
    from tests.fakes.telegram import FakeTelegram
    return FakeTelegram(clock)



# the /help text and the setMyCommands menu, in the order of the live BRIDGE_COMMANDS (tg-bridge.py)
LIVE_MENU_ORDER = ("status", "stop", "mode", "model", "launcher", "new", "clear", "resume", "cwd", "queue", "cost",
                   "rules", "forget", "drive", "help")

class _QueueView:
    """What CommandDeps.queues needs (snapshot): a view of the queues the Dispatcher owns."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatcher = dispatcher

    def snapshot(self, thread_id: int) -> tuple[Any, ...]:
        return self._dispatcher.queue_snapshot(thread_id)


@dataclass(frozen=True)
class App:
    """The wired services the daemon, the CLI and the tests use. Built only by App.build()."""

    paths: Paths
    config: Config
    logger: Logger
    clock: Clock
    db: Database
    client: TelegramClient
    registry: BackendRegistry
    kv: KvRepo
    sessions: SessionRepo
    live: LiveRegistry
    permissions: PermissionService
    dispatcher: Dispatcher
    commands: CommandRegistry
    handler: MessageHandler
    router: CallbackRouter
    inbox: InboxService
    poller: Poller
    lifecycle: Lifecycle
    panes: Panes
    messenger: Messenger
    uploader: Uploader
    linger: LingerService  # the one kept-session service, shared by the turn runner and the dispatcher
    remover: TopicRemover  # forgets a topic completely (session, queue, rows, kv keys, view files, Telegram)
    probe: TopicProbe  # finds the topics deleted in Telegram and removes them (started by the daemon, see cli)
    delete_requests: DeleteRequests  # answers the view's delete requests (view-delete), started by the daemon
    topics: TopicResolver  # topic resolution and the thread name learning

    @classmethod
    def build(cls, paths: Paths, config: Config, telegram: TelegramClient | None = None,
              clock: Clock | None = None, registry: BackendRegistry | None = None,
              env: Mapping[str, str] | None = None, sweep_stale_views: bool = False) -> "App":
        """Creates every service once and wires it. env (os.environ when omitted) supplies the child environment,
        the proxy settings and TG_TITLE_LAUNCHER; the caller owns the App and must close() it. sweep_stale_views is
        True only at the daemon's start-up (it clears the 'running' state a killed daemon left in the view files, so
        a short command must never pass it). A view folder that cannot be created turns the views off, not the App."""
        environ: Mapping[str, str] = os.environ if env is None else env
        clock = SystemClock() if clock is None else clock
        logger = Logger(paths.log)
        os.makedirs(paths.turns, mode=0o700, exist_ok=True)
        views_on = ensure_view_dir(paths.view, logger)
        if views_on and sweep_stale_views:
            mark_stale_views(paths.view, clock, logger)
            clear_turn_marks(paths.view, logger)
        db = Database(paths)
        try:
            return cls._wire(paths, config, logger, clock, db, environ, telegram, registry, views_on)
        except BaseException:
            db.close()
            raise

    @classmethod
    def _wire(cls, paths: Paths, config: Config, logger: Logger, clock: Clock, db: Database,
              environ: Mapping[str, str], telegram: TelegramClient | None,
              registry: BackendRegistry | None, views_on: bool) -> "App":
        openers = OpenerCache(lambda: proxy_url(paths, environ, logger.log))
        if telegram is None and fake_telegram_requested(paths, environ, logger.log):
            telegram = _new_fake_telegram(clock)
        if telegram is None:
            telegram = HttpTelegramClient(paths, clock, logger, RateGate(clock), openers)
        if registry is None:
            registry = build_default_registry(config, logger, clock,
                                              title_launcher=title_launcher_from_env(environ))

        # repositories and the shared pools and registries
        sessions = SessionRepo(db, clock)
        history = HistoryRepo(db, clock)
        kv = KvRepo(db)
        rules = RuleRepo(db, clock)
        turns = TurnRepo(db)
        asks = AskRepo(db, clock)
        inbox_rows = InboxRepo(db, clock)
        cwd_paths = CwdRepo(db)
        out = WorkerPool("out", OUT_THREADS, logger)
        media_pool = WorkerPool("media", MEDIA_THREADS, logger)
        search_pool = WorkerPool("search", SEARCH_THREADS, logger)
        title_pool = WorkerPool("title", TITLE_THREADS, logger)
        messenger = Messenger(telegram, logger, clock)
        live = LiveRegistry()
        waits = WaitRouter()

        # turns: permissions, the turn runner, the dispatcher
        handoff = HandoffService(kv, registry, logger)
        uploader = Uploader(paths, openers, logger)
        driving = DrivingMode(kv, clock)  # driving mode: answers and cards are also spoken
        voice = DrivingVoice(driving, Speech(clock, logger, os.path.join(paths.state, "speech")), uploader, messenger,
                             WorkerPool("voice", 1, logger), clock, logger)
        cards = ApprovalCards(messenger, logger, clock, on_posted=voice.card)
        def plan_approved(thread_id: int, chat_id: int, mode: str) -> None:  # the dispatcher is built below
            PlanFollowUp(sessions, dispatcher, messenger, logger)(thread_id, chat_id, mode)

        permissions = PermissionService(asks=asks, rules=rules, cards=cards, messenger=messenger, workers=out,
                                        waiting=waits, logger=logger, plan_approved=plan_approved)
        slots = SlotPool(SLOT_COUNT, messenger, live.is_shutting_down)

        def feeds(chat_id: int | str, thread_id: int, parked: Callable[[], bool]) -> TurnFeed:
            return TurnFeed(telegram, logger, clock, chat_id, thread_id, parked=parked)

        def views(thread_id: int, topic: str, transport: str, launcher: str) -> ViewRecorder:
            return ViewRecorder(paths.view, thread_id, topic, transport, launcher, clock, logger)

        answers = AnswerSender(telegram, uploader, messenger, logger, clock, on_answer=voice.answer)
        self_turns = SelfTurns(
            config=config, clock=clock, logger=logger, feeds=feeds, views=views if views_on else None,
            permissions=permissions, handoff=handoff, waits=waits, live=live, turns=turns, answers=answers,
            messenger=messenger)
        linger = LingerService(
            config=config, sessions=sessions, kv=kv, live=live, permissions=permissions, self_turns=self_turns,
            clock=clock, logger=logger)
        runner = TurnRunner(
            config=config, registry=registry, live=live, permissions=permissions, waits=waits, sessions=sessions,
            history=history, rules=rules, turns=turns, kv=kv, handoff=handoff, feeds=feeds, clock=clock,
            logger=logger, base_env=dict(environ), errfile_dir=paths.turns,
            views=views if views_on else None, view_dir=paths.view if views_on else "", linger=linger,
            memory_dir=paths.memory, note=lambda: drive_note(driving))
        titles = TitleService(backend=registry.get(BACKEND_CLAUDE), sessions=sessions, messenger=messenger,
                              pool=title_pool, logger=logger)
        removals = Removals(clock)  # the marks of the topics being removed: the dispatcher and the remover share them
        # the terminal guard: built before the dispatcher, whose before_turn hook settles a deferred change first
        terminal = TerminalGuard(view_dir=paths.view if views_on else "", live=live, clock=clock)
        dispatcher = Dispatcher(
            turns=runner, slots=slots, inbox=inbox_rows, messenger=messenger, answers=answers, titles=titles,
            client=telegram, live=live, config=config, logger=logger, linger=linger,
            view_dir=paths.view if views_on else "", removals=removals,
            pane_idle=lambda claim: pane_idle(claim.socket, claim.pane),  # a terminal claim's turns wait for its pane
            pane_reason=lambda claim: pane_reason(claim.socket, claim.pane), before_turn=terminal.before_turn)
        remover = TopicRemover(
            live=live, linger=linger, dispatcher=dispatcher,
            forget=TopicForget(db=db, sessions=sessions, history=history, rules=rules, asks=asks, inbox=inbox_rows,
                               kv=kv),
            client=telegram, kv=kv, group_key=GROUP_KEY, view_dir=paths.view if views_on else "", logger=logger,
            clock=clock, removals=removals)
        probe = TopicProbe(
            sessions=sessions, kv=kv, live=live, linger=linger, remover=remover, client=telegram, group_key=GROUP_KEY,
            view_dir=paths.view if views_on else "", logger=logger, clock=clock,
            interval=config.topic_probe_seconds())
        delete_requests = DeleteRequests(view_dir=paths.view, sessions=sessions, remover=remover, logger=logger)

        # bot: topics, folders, launchers, commands and taps
        topics = TopicResolver(config=config, kv=kv, sessions=sessions, logger=logger)
        cwd = CwdService(sessions=sessions, kv=kv, paths=cwd_paths, messenger=messenger, live=live,
                         search_pool=search_pool, out_pool=out, logger=logger, clock=clock, handoff=handoff)
        launchers = LauncherSwitch(config, sessions, history, handoff, registry)
        keep = KeepGuard(linger=linger, live=live)  # one guard: its cards are answered by the taps below
        stops = SessionCommands(config=config, sessions=sessions, history=history, live=live, backends=registry,
                                messenger=messenger, clock=clock, logger=logger, handoff=handoff, linger=linger,
                                keep=keep, terminal=terminal)
        settings = SettingsCommands(config=config, sessions=sessions, live=live, backends=registry,
                                    permissions=permissions, launchers=launchers, cwd=cwd, logger=logger, keep=keep)
        commands = CommandRegistry()
        register_info_commands(commands)
        stops.register(commands)
        settings.register(commands)
        DriveCommand(driving, logger).register(commands)
        commands.set_menu_order(LIVE_MENU_ORDER)
        deps = CommandDeps(config=config, sessions=sessions, turns=turns, rules=rules, kv=kv, live=live,
                           queues=_QueueView(dispatcher), backends=registry, messenger=messenger, out=out,
                           linger=linger, view_dir=paths.view if views_on else "")
        router = CallbackRouter(config=config, client=telegram, messenger=messenger, out=out, logger=logger)
        for tap in (ApprovalTaps(permissions=permissions),
                    QueueTaps(dispatcher=dispatcher, live=live, linger=linger),
                    StopTap(stops=stops),
                    StopChoiceTaps(stops=stops),
                    StatusTaps(deps=deps, registry=commands, stops=stops),
                    PickerTaps(config=config, sessions=sessions, live=live, backends=registry,
                               permissions=permissions, launchers=launchers, stops=stops, rules=rules,
                               folders=cwd, logger=logger, keep=keep),
                    KeepTaps(keep=keep),
                    DriveTaps(mode=driving, logger=logger),
                    TerminalTaps(terminal=terminal)):
            tap.register(router)
        panes = Panes(clock=clock)
        media = MediaService(paths, telegram, messenger, openers, SonioxTranscriber(paths, openers, clock), clock)
        handler = MessageHandler(
            config=config, topics=topics, sessions=sessions, kv=kv, inbox=inbox_rows, dispatcher=dispatcher,
            permissions=permissions, live=live, cwd=cwd, commands=commands, command_deps=deps, backends=registry,
            messenger=messenger, out=out, media=media, media_pool=media_pool, seats=panes, logger=logger, keep=keep)

        # daemon: inbox, poller, lifecycle
        inbox = InboxService(inbox=inbox_rows, kv=kv, sessions=sessions, messenger=messenger, messages=handler,
                             callbacks=router, spool_dir=paths.spool, clock=clock, logger=logger,
                             allowed_user_ids=config.allow_user_ids)
        poller = Poller(telegram=telegram, inbox=inbox, kv=kv, sessions=sessions, messenger=messenger,
                        config=config, clock=clock, logger=logger)
        lifecycle = Lifecycle(live=live, permissions=permissions, kv=kv, messenger=messenger, telegram=telegram,
                              commands=commands, clock=clock, logger=logger)
        return cls(paths=paths, config=config, logger=logger, clock=clock, db=db, client=telegram,
                   registry=registry, kv=kv, sessions=sessions, live=live, permissions=permissions,
                   dispatcher=dispatcher, commands=commands, handler=handler, router=router, inbox=inbox,
                   poller=poller, lifecycle=lifecycle, panes=panes, messenger=messenger, uploader=uploader,
                   linger=linger, remover=remover, probe=probe, delete_requests=delete_requests, topics=topics)

    def shutdown_sessions(self) -> int:
        """Stops every live session (no new turn can start from here on). Returns how many were signalled."""
        self.delete_requests.stop()  # no view delete starts after this; one in progress ends at its next step
        self.probe.stop()  # no removal round starts after this; a round in progress ends at its next step
        return self.lifecycle.shutdown_sessions()

    def close(self) -> None:
        """Closes the database. Call once, after shutdown_sessions()."""
        self.db.close()
