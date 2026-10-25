"""Queue, stop and /status-shortcut taps (C20 now/jump/drop, st, and the cm: status kinds).

- now / jump / drop: the buttons under a queued message (callback now:<thread>:<nid> and so on). Now moves the item to
  the front and interrupts the running turn on a thread of its own; Next only moves it; Drop removes it.
- st: the ⏹ Stop button on a progress message. It calls SessionCommands.stop_turn, the same method /stop calls.
- cm:<thread>:<kind>:<arg> status kinds: st (Stop, the same method again), nw (stop and start fresh), sn (New) and sc
  (Cost), which run the typed command through the CommandRegistry, and sm / so / sl, which send the mode, model or
  launcher picker as a new message.

Every reply is made through the Tap, so nothing here waits on Telegram.
"""

import threading

from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.info import launcher_state
from tg_bridge.bot.commands.session import STOP_BUTTON_KIND, SessionCommands
from tg_bridge.bot.pickers import STATUS_COST, STATUS_LAUNCHER, STATUS_MODE, STATUS_MODEL, STATUS_NEW, STATUS_STOP
from tg_bridge.bot.pickers import launcher_picker, mode_picker, model_picker
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.topic_backend import topic_backend, topic_models
from tg_bridge.store.sessions import SessionRecord

NOT_QUEUED_TEXT = "not queued anymore"
NOW_TOAST = "Interrupting — your item runs now ⏭"
NOW_NOTE = "📥 ⏭ will run now — current turn interrupted"
JUMP_TOAST = "Moved to front ✅"
JUMP_NOTE = "📥 ⏩ runs next after the current turn"
DROP_TOAST = "Dropped 🗑"
DROP_NOTE = "📥 ❌ dropped from the queue"
STOP_TOAST = "Stopping ⏹"
NOTHING_RUNNING_TEXT = "nothing is running in this topic"
NO_SESSION_TEXT = "no session in this topic"
NEW_TOAST = "🆕 fresh session next"
COST_TOAST = "💰"
MODE_TOAST = "🛡 mode"
MODEL_TOAST = "🧠 model"
LAUNCHER_TOAST = "🚀 launcher"
STOP_PREFIX = "st"


def _thread_arg(tap: Tap, count: int) -> int | None:
    """The thread of a tap whose arguments are count fields with a numeric thread first, or None when malformed."""
    if len(tap.args) != count or not tap.args[0].isdigit():
        return None
    return int(tap.args[0])


class QueueTaps:
    """The now / jump / drop buttons of a queued message. The queue itself is the Dispatcher's."""

    def __init__(self, *, dispatcher: Dispatcher, live: LiveRegistry) -> None:
        self._dispatcher = dispatcher
        self._live = live

    def register(self, router: CallbackRouter) -> None:
        router.on("now", self._now)
        router.on("jump", self._jump)
        router.on("drop", self._drop)

    def _item(self, tap: Tap) -> tuple[int, str] | None:
        thread = _thread_arg(tap, 2)
        return None if thread is None or not tap.args[1] else (thread, tap.args[1])

    def _now(self, tap: Tap) -> None:
        item = self._item(tap)
        if item is None:
            return
        thread, nid = item
        if not self._dispatcher.to_front(thread, nid):
            tap.toast(NOT_QUEUED_TEXT)
            return
        running = self._live.request_abort(thread, stop_tap=False)
        if running is not None:  # stop() can take ~20 s: the tap answers at once, the stop runs on its own thread
            threading.Thread(target=running.session.stop, daemon=True).start()
        tap.toast(NOW_TOAST)
        tap.edit(NOW_NOTE)

    def _jump(self, tap: Tap) -> None:
        item = self._item(tap)
        if item is None:
            return
        if not self._dispatcher.to_front(*item):
            tap.toast(NOT_QUEUED_TEXT)
            return
        tap.toast(JUMP_TOAST)
        tap.edit(JUMP_NOTE)

    def _drop(self, tap: Tap) -> None:
        item = self._item(tap)
        if item is None:
            return
        if self._dispatcher.drop(*item) is None:
            tap.toast(NOT_QUEUED_TEXT)
            return
        tap.toast(DROP_TOAST)
        tap.edit(DROP_NOTE)


class StopTap:
    """The ⏹ Stop button of a progress message (callback st:<thread>): the /stop path, nothing else."""

    def __init__(self, *, stops: SessionCommands) -> None:
        self._stops = stops

    def register(self, router: CallbackRouter) -> None:
        router.on(STOP_PREFIX, self._stop)

    def _stop(self, tap: Tap) -> None:
        thread = _thread_arg(tap, 1)
        if thread is None:
            return
        stopped = self._stops.stop_turn(thread)
        tap.toast(STOP_TOAST if stopped else NOTHING_RUNNING_TEXT)


class StatusTaps:
    """The cm: kinds of the /status buttons and of the stop-and-new button. Each reuses a command or a picker."""

    def __init__(self, *, deps: CommandDeps, registry: CommandRegistry, stops: SessionCommands) -> None:
        self._deps = deps
        self._registry = registry
        self._stops = stops

    def register(self, router: CallbackRouter) -> None:
        router.on_picker(STATUS_STOP, self._stop)
        router.on_picker(STOP_BUTTON_KIND, self._stop_and_new)
        router.on_picker(STATUS_NEW, self._new)
        router.on_picker(STATUS_COST, self._cost)
        router.on_picker(STATUS_MODE, self._mode)
        router.on_picker(STATUS_MODEL, self._model)
        router.on_picker(STATUS_LAUNCHER, self._launcher)

    def _open(self, tap: Tap) -> tuple[int, SessionRecord] | None:
        """The thread and its session row, or None after the reply: no chat (nothing to answer in) or no session."""
        thread = _thread_arg(tap, 3)
        if thread is None or tap.chat_id is None:
            return None
        row = self._deps.sessions.get(thread)
        if row is None:
            tap.toast(NO_SESSION_TEXT)
            return None
        return thread, row

    def _backend(self, row: SessionRecord):
        """The backend the topic's turns run on now, resolved from its launcher (see topic_backend)."""
        return topic_backend(self._deps.config, self._deps.backends, row, row.name)

    def _stop(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        stopped = self._stops.stop_turn(opened[0])
        tap.toast(STOP_TOAST if stopped else NOTHING_RUNNING_TEXT)

    def _stop_and_new(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, _ = opened
        tap.toast(self._stops.stop_and_new(thread, tap.chat_id, tap.message_id))

    def _new(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        self._registry.dispatch(self._deps, tap.chat_id, thread, row.name, "/new")
        tap.toast(NEW_TOAST)

    def _cost(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        self._registry.dispatch(self._deps, tap.chat_id, thread, row.name, "/cost")
        tap.toast(COST_TOAST)

    def _mode(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        backend = self._backend(row)
        picker = mode_picker(thread, backend.modes(), row.mode or self._deps.config.default_mode(),
                             backend.capabilities.live_mode)
        tap.send_buttons(thread, picker.text, picker.rows)
        tap.toast(MODE_TOAST)

    def _model(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        backend, models = topic_models(self._deps.config, self._deps.backends, row, row.name)
        picker = model_picker(thread, models, row.model or None, backend.capabilities.live_model)
        tap.send_buttons(thread, picker.text, picker.rows)
        tap.toast(MODEL_TOAST)

    def _launcher(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        config = self._deps.config
        name, command = launcher_state(config, row, row.name)
        picker = launcher_picker(thread, list(config.launcher_specs().values()), name, command)
        tap.send_buttons(thread, picker.text, picker.rows)
        tap.toast(LAUNCHER_TOAST)
