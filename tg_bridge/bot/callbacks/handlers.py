"""Queue, stop and /status-shortcut taps (C20 now/jump/drop, st, and the cm: status kinds).

- now / jump / drop: the buttons under a queued message (callback now:<thread>:<nid> and so on). Now moves the item to
  the front and interrupts the running turn on a thread of its own; during a background update it interrupts that
  update only and the item runs on the same session (section 5 D2). Next only moves it; Drop removes it.
- st: the ⏹ Stop button on a progress message. It calls SessionCommands.stop_turn, the same method /stop calls. While a
  background update runs, the topic gets the choice card instead (sq: this update only, or close the session).
- cm:<thread>:<kind>:<arg> status kinds: st (Stop, the same method again), nw (stop and start fresh), sn (New) and sc
  (Cost), which run the typed command through the CommandRegistry, and sm / so / sl, which send the mode, model or
  launcher picker as a new message.

Every reply is made through the Tap, so nothing here waits on Telegram.
"""

import threading

from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.info import launcher_state
from tg_bridge.bot.commands.session import (
    STOP_BUTTON_KIND,
    UPDATE_CLOSED_TEXT,
    UPDATE_GONE_TEXT,
    UPDATE_INTERRUPTED_TEXT,
    SessionCommands,
)
from tg_bridge.bot.commands.stop_types import StopKind, StopResult
from tg_bridge.bot.pickers import (
    CHOICE_CLOSE,
    CHOICE_UPDATE,
    STATUS_COST,
    STATUS_LAUNCHER,
    STATUS_MODE,
    STATUS_MODEL,
    STATUS_NEW,
    STATUS_STOP,
    STOP_CHOICE_KIND,
    launcher_picker,
    mode_picker,
    model_picker,
    stop_choice_picker,
)
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.topic_backend import topic_backend, topic_capabilities, topic_models
from tg_bridge.store.sessions import SessionRecord

NOT_QUEUED_TEXT = "not queued anymore"
NOW_TOAST = "Interrupting — your item runs now ⏭"
NOW_NOTE = "📥 ⏭ will run now — current turn interrupted"
JUMP_TOAST = "Moved to front ✅"
JUMP_NOTE = "📥 ⏩ runs next after the current turn"
DROP_TOAST = "Dropped 🗑"
DROP_NOTE = "📥 ❌ dropped from the queue"
STOP_TOAST = "Stopping ⏹"
CHOOSE_TOAST = "Choose what to stop ⏹"
NOTHING_RUNNING_TEXT = "nothing is running in this topic"
NO_SESSION_TEXT = "no session in this topic"
NEW_TOAST = "🆕 fresh session next"
COST_TOAST = "💰"
MODE_TOAST = "🛡 mode"
MODEL_TOAST = "🧠 model"
LAUNCHER_TOAST = "🚀 launcher"
STOP_PREFIX = "st"
INTERRUPT_TOAST = "Interrupting this update ⏹"
CHOICE_EXPIRED_TOAST = "this choice has expired"


def _thread_arg(tap: Tap, count: int) -> int | None:
    """The thread of a tap whose arguments are count fields with a numeric thread first, or None when malformed."""
    if len(tap.args) != count or not tap.args[0].isdigit():
        return None
    return int(tap.args[0])


class QueueTaps:
    """The now / jump / drop buttons of a queued message. The queue itself is the Dispatcher's."""

    def __init__(self, *, dispatcher: Dispatcher, live: LiveRegistry, linger: LingerService | None = None) -> None:
        self._dispatcher = dispatcher
        self._live = live
        self._linger = linger  # None: no session is kept, so a running turn is always stopped whole

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
        # a background update is interrupted alone (section 5 D2): the item then runs on the same kept session
        interrupted = self._linger is not None and self._linger.request_interrupt(thread)
        running = None if interrupted else self._live.request_abort(thread, stop_tap=False)
        if running is not None:  # stop() can take ~20 s: the tap answers at once, the stop runs on its own thread
            threading.Thread(target=running.session.halt_turn, daemon=True).start()
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


def _choice_arg(arg: str) -> tuple[str, int, int] | None:
    """The choice, hold id and self turn number of a choice-card tap (u:<hold_id>:<self_seq>), or None when malformed."""
    parts = arg.split(":")
    if len(parts) != 3 or parts[0] not in (CHOICE_UPDATE, CHOICE_CLOSE):
        return None
    if not parts[1].isdigit() or not parts[2].isdigit():
        return None
    return parts[0], int(parts[1]), int(parts[2])


def _stop_or_ask(tap: Tap, stops: SessionCommands, thread: int) -> None:
    """The shared stop of the ⏹ buttons: stops at once, or posts the choice card while a background update runs."""
    result: StopResult = stops.stop_turn(thread)
    if result.kind is StopKind.CHOOSE:
        picker = stop_choice_picker(thread, result.hold_id, result.self_seq)
        tap.send_buttons(thread, picker.text, picker.rows)
        tap.toast(CHOOSE_TOAST)
        return
    tap.toast(STOP_TOAST if result else NOTHING_RUNNING_TEXT)


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
        _stop_or_ask(tap, self._stops, thread)


class StopChoiceTaps:
    """The buttons of the choice card (picker kind sq, arg u or c, then the hold id and self turn it was posted for):
    this update only, or close the session (D2). A tap acts only while that same self turn still runs (K9); otherwise it
    is expired and changes nothing."""

    def __init__(self, *, stops: SessionCommands) -> None:
        self._stops = stops

    def register(self, router: CallbackRouter) -> None:
        router.on_picker(STOP_CHOICE_KIND, self._choose)

    def _choose(self, tap: Tap) -> None:
        thread = _thread_arg(tap, 3)
        if thread is None:
            return
        choice = _choice_arg(tap.args[2])
        if choice is None or not self._stops.choice_current(thread, choice[1], choice[2]):
            tap.toast(CHOICE_EXPIRED_TOAST)
            return
        if choice[0] == CHOICE_UPDATE:
            if self._stops.interrupt_update(thread):
                tap.toast(INTERRUPT_TOAST)
                tap.edit(UPDATE_INTERRUPTED_TEXT)
            else:
                tap.toast(UPDATE_GONE_TEXT)
                tap.edit(UPDATE_GONE_TEXT)
        else:
            if self._stops.stop_now(thread):
                tap.toast(STOP_TOAST)
                tap.edit(UPDATE_CLOSED_TEXT)
            else:
                tap.toast(NOTHING_RUNNING_TEXT)
                tap.edit(UPDATE_GONE_TEXT)


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
        _stop_or_ask(tap, self._stops, opened[0])

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
        caps = topic_capabilities(self._deps.config, self._deps.backends, row, row.name)
        picker = mode_picker(thread, backend.modes(), row.mode or self._deps.config.default_mode(), caps.live_mode)
        tap.send_buttons(thread, picker.text, picker.rows)
        tap.toast(MODE_TOAST)

    def _model(self, tap: Tap) -> None:
        opened = self._open(tap)
        if opened is None:
            return
        thread, row = opened
        _, models = topic_models(self._deps.config, self._deps.backends, row, row.name)
        caps = topic_capabilities(self._deps.config, self._deps.backends, row, row.name)
        picker = model_picker(thread, models, row.model or None, caps.live_model)
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
