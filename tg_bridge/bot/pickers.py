"""Picker builders (STEP 46): the model, mode, launcher, resume, rules and status keyboards with their text (C19).

Port of the picker half of the live file: model_picker, mode_picker, launcher_picker, resume_picker, rules_picker and
status_rows. The builders are pure. They take the values they show (the backend's modes, the config's launchers, the
history rows) as arguments, read no database and send nothing, so a command and a tap handler build the same markup.

Callbacks are cm:<thread>:<kind>:<arg> (keyboards.CallbackData). A model option whose callback would not fit the
64-byte limit is left out of its keyboard, so one oversized model name cannot break the whole picker; the typed
/model <name> still works for it. Mode, launcher, resume and rule buttons carry an index or a row id, never a name.

Backend differences are shown, not hidden: a launcher of a non-Claude backend carries a backend tag, and the mode and
model pickers say NOT_LIVE_NOTE when the backend cannot change them inside a running turn (Capabilities.live_*).
"""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from tg_bridge.agents.base import ModeChoice
from tg_bridge.config import BACKEND_CLAUDE, LauncherSpec
from tg_bridge.store.rules import RuleRecord
from tg_bridge.store.sessions import HistoryRecord
from tg_bridge.telegram.keyboards import (
    PICKER_PREFIX,
    CallbackData,
    CallbackDataError,
    keyboard_json,
    picker_rows,
)

ButtonRows = list[list[tuple[str, str | CallbackData]]]

KIND_MODEL = "mo"
KIND_MODE = "md"
KIND_LAUNCHER = "la"
KIND_RESUME = "rs"
KIND_RULE_DELETE = "rf"
KIND_RULE_ALL = "ra"
STATUS_STOP = "st"
STATUS_NEW = "sn"
STATUS_MODE = "sm"
STATUS_MODEL = "so"
STATUS_COST = "sc"
STATUS_LAUNCHER = "sl"
STOP_CHOICE_KIND = "sq"         # the D2 card: stop the background update only, or close the session
CHOICE_UPDATE = "u"
CHOICE_CLOSE = "c"

DEFAULT_MODEL = "default"
NOT_LIVE_NOTE = "applies from your next message"
RESUME_LIMIT = 10                 # the /resume picker shows the newest sessions of the topic
RULES_PER_ROW = 4
OWNER_TIMEZONE = "Asia/Tehran"    # session start times on the buttons are shown on the owner's clock
NO_RULES_TEXT = "no always-allow rules in this topic."
RULES_HINT = "remove one with its 🗑 button, /forget <number> or /forget all"
RESUME_LIVE_TEXT = "↩️ the next message resumes the ✓ session"
RESUME_FRESH_TEXT = "↩️ the next message starts a fresh session"
RESUME_AWAY_TEXT = "⚠ = made by another launcher: tapping it switches the launcher to that one"
STOP_CHOICE_TEXT = "⏹ a background update is running. Stop what?"
STOP_CHOICE_UPDATE_BUTTON = "⏹ this update only"
STOP_CHOICE_CLOSE_BUTTON = "🛑 close the session"


@dataclass(frozen=True)
class Picker:
    """A picker message: its text and its inline keyboard rows."""

    text: str
    rows: ButtonRows

    def markup(self) -> str:
        """The reply_markup JSON Telegram expects for these rows."""
        return keyboard_json(self.rows)


def _cb(thread: int, kind: str, arg: str | int) -> CallbackData:
    """A picker callback, checked against the 64-byte limit when the picker is built."""
    data = CallbackData(PICKER_PREFIX, (str(thread), kind, arg))
    data.encode()
    return data


def _fits(thread: int, kind: str, arg: str) -> bool:
    """True when the callback for this option can be sent (a model name may be too long for Telegram)."""
    try:
        _cb(thread, kind, arg)
    except CallbackDataError:
        return False
    return True


def _with_note(text: str, live: bool) -> str:
    """The text, with the not-live note on its own line when the backend applies the change from the next message."""
    return text if live else f"{text}\n{NOT_LIVE_NOTE}"


def started_label(started: str) -> str:
    """A session's start as MM-DD HH:MM on the owner's clock; '--' when the stored value is not a timestamp."""
    try:
        moment = datetime.datetime.fromisoformat(started).astimezone(ZoneInfo(OWNER_TIMEZONE))
    except (TypeError, ValueError):
        return "--"
    return moment.strftime("%m-%d %H:%M")


def model_picker(thread: int, models: Sequence[str], current: str | None, live: bool) -> Picker:
    """The /model picker: one button per model name, the current one marked ✓ ('default' when nothing is set)."""
    shown = current or DEFAULT_MODEL
    options = [(model, model, model) for model in models if _fits(thread, KIND_MODEL, model)]
    text = _with_note(f"🧠 model: {shown}", live)
    return Picker(text, picker_rows(thread, KIND_MODEL, options, shown))


def mode_picker(thread: int, modes: Sequence[ModeChoice], current: str, live: bool) -> Picker:
    """The /mode picker: one button per mode of the topic's backend, callback md:<index> into that same list."""
    options = [(mode.label, mode.key, index) for index, mode in enumerate(modes)]
    text = _with_note(f"🛡 mode: {current}", live)
    return Picker(text, picker_rows(thread, KIND_MODE, options, current))


def _launcher_label(spec: LauncherSpec) -> str:
    """The button text of a launcher: its name, plus the backend tag for a non-Claude launcher."""
    if spec.backend == BACKEND_CLAUDE:
        return spec.name
    return f"{spec.name} · {spec.backend}"


def launcher_picker(
    thread: int, launchers: Sequence[LauncherSpec], current_name: str | None, current_command: str = ""
) -> Picker:
    """The /launcher picker: one button per launcher (callback la:<index>), the one in use marked ✓.

    current_name is the launcher the topic runs (None when it runs an unnamed command). The text falls back to
    current_command, then to 'none', so a topic whose launcher is not in config still shows what it runs.
    """
    options = [(_launcher_label(spec), spec.name, index) for index, spec in enumerate(launchers)]
    text = f"🚀 launcher: {current_name or current_command or 'none'}"
    return Picker(text, picker_rows(thread, KIND_LAUNCHER, options, current_name))


def resume_picker(
    thread: int,
    history: Sequence[HistoryRecord],
    current_uuid: str,
    current_command: str,
    names: Mapping[str, str],
) -> Picker:
    """The /resume picker: the topic's newest sessions (RESUME_LIMIT), the one in use marked ✓.

    history is newest first, as the command sorts it. Every button names its session's launcher (names maps launcher
    command -> launcher name). A session made by another launcher is marked ⚠ instead, and tapping it switches the
    launcher. Each button carries its session_history id (cm:<thread>:rs:<id>).
    """
    rows: ButtonRows = []
    away_any = False
    for record in list(history)[:RESUME_LIMIT]:
        what = f"{record.first_text}…" if record.first_text else record.uuid[:8]
        mark = "✓ " if current_uuid and record.uuid == current_uuid else ""
        away = bool(record.launcher) and record.launcher != current_command
        away_any = away_any or away
        name = names.get(record.launcher, "")
        if away:
            tag = f"⚠ {name or 'other launcher'} · "
        else:
            tag = f"{name} · " if name else ""
        label = f"{mark}{tag}{started_label(record.started)} · {what}"
        rows.append([(label, _cb(thread, KIND_RESUME, record.id))])
    text = RESUME_LIVE_TEXT if current_uuid else RESUME_FRESH_TEXT
    if away_any:
        text += "\n" + RESUME_AWAY_TEXT
    return Picker(text, rows)


def rules_picker(thread: int, rules: Sequence[RuleRecord]) -> Picker:
    """The /rules list: one line per rule, a 🗑 button per rule (its number, rf) and a 🗑 all button (ra).

    With no rules the text is the plain notice and there is no keyboard.
    """
    if not rules:
        return Picker(NO_RULES_TEXT, [])
    lines = [f"{rule.id}. {rule.rule} (used {rule.hits}×)" for rule in rules]
    text = "\n".join(lines) + "\n" + RULES_HINT
    buttons = [(f"🗑 {rule.id}", _cb(thread, KIND_RULE_DELETE, rule.id)) for rule in rules]
    rows: ButtonRows = [buttons[i : i + RULES_PER_ROW] for i in range(0, len(buttons), RULES_PER_ROW)]
    rows.append([("🗑 all", _cb(thread, KIND_RULE_ALL, 0))])
    return Picker(text, rows)


def stop_choice_picker(thread: int, hold_id: int, self_seq: int) -> Picker:
    """The card that asks which stop a running background update gets (section 5 D2): this update only, or the session.
    Each button names the hold and the self turn it was posted for (K9), so a tap after that update ended expires."""
    update = _cb(thread, STOP_CHOICE_KIND, _choice(CHOICE_UPDATE, hold_id, self_seq))
    close = _cb(thread, STOP_CHOICE_KIND, _choice(CHOICE_CLOSE, hold_id, self_seq))
    rows: ButtonRows = [[(STOP_CHOICE_UPDATE_BUTTON, update), (STOP_CHOICE_CLOSE_BUTTON, close)]]
    return Picker(STOP_CHOICE_TEXT, rows)


def _choice(choice: str, hold_id: int, self_seq: int) -> str:
    """The argument of a stop-choice button: the choice, the hold id and the self turn number (u:<hold>:<seq>)."""
    return f"{choice}:{hold_id}:{self_seq}"


def status_rows(thread: int, running: bool) -> ButtonRows:
    """The shortcut buttons under /status. Stop shows only while a turn runs; the others reuse a command or a picker."""
    stop: ButtonRows = [[("⏹ Stop", _cb(thread, STATUS_STOP, 0))]] if running else []
    return stop + [
        [("🆕 New", _cb(thread, STATUS_NEW, 0)), ("🛡 Mode", _cb(thread, STATUS_MODE, 0))],
        [("🧠 Model", _cb(thread, STATUS_MODEL, 0)), ("💰 Cost", _cb(thread, STATUS_COST, 0))],
        [("🚀 Launcher", _cb(thread, STATUS_LAUNCHER, 0))],
    ]
