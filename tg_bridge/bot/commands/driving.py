"""/drive: driving mode (core/driving.py). Every answer and approval card is also spoken, as a voice note in its topic.

/drive alone posts the panel: the mode's state and voice, with buttons to turn it on or off and to pick the female or
the male voice (bot/callbacks/drive_taps.py edits the panel on each tap). The words on | off | female | male | status
(or روشن خاموش زن مرد وضعیت) do the same without the panel's buttons.
"""

from types import MappingProxyType

from tg_bridge.bot.commands.base import ButtonRows, Command, CommandContext, CommandRegistry
from tg_bridge.core.driving import DrivingMode
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.keyboards import CallbackData

DRIVE_PREFIX = "dr"  # dr:<action>: the panel's buttons
ON = "on"
OFF = "off"
FEMALE = "female"
MALE = "male"
STATUS = "status"
DRIVE_HELP = "driving mode: answers also come as voice notes; shows the on/off and voice buttons"
WORDS = MappingProxyType({ON: ON, "روشن": ON, OFF: OFF, "خاموش": OFF, FEMALE: FEMALE, "زن": FEMALE, MALE: MALE,
                          "مرد": MALE, STATUS: STATUS, "وضعیت": STATUS})
VOICE_LABELS = MappingProxyType({FEMALE: "👩 female", MALE: "👨 male"})
PANEL_ON = ("🚗 Driving mode: ON ({minutes} minutes left) · voice: {voice}\n"
            "Every answer also comes as a voice note, right after it in the same topic.")
PANEL_OFF = "🚗 Driving mode: OFF · voice: {voice}"
TURN_ON_LABEL = "▶️ Turn on"
TURN_OFF_LABEL = "⏹ Turn off"
PICKED_MARK = " ✓"
UNKNOWN_TEXT = "usage: /drive [on|off|female|male|status]"


def panel_text(mode: DrivingMode) -> str:
    voice = VOICE_LABELS[mode.voice()]
    return PANEL_ON.format(minutes=mode.minutes_left(), voice=voice) if mode.active() else PANEL_OFF.format(voice=voice)


def panel_rows(mode: DrivingMode) -> ButtonRows:
    switch = (TURN_OFF_LABEL, CallbackData(DRIVE_PREFIX, (OFF,))) if mode.active() else \
        (TURN_ON_LABEL, CallbackData(DRIVE_PREFIX, (ON,)))
    voices = [(label + (PICKED_MARK if mode.voice() == voice else ""), CallbackData(DRIVE_PREFIX, (voice,)))
              for voice, label in VOICE_LABELS.items()]
    return [[switch], voices]


def apply(mode: DrivingMode, action: str) -> str | None:
    """Applies one action of the panel or the command. None when it worked, else the text that says why not."""
    if action in VOICE_LABELS:
        mode.set_voice(action)
    elif action == OFF:
        mode.turn_off()
    elif action == ON:
        mode.turn_on()
    elif action != STATUS:
        return UNKNOWN_TEXT
    return None


class DriveCommand:
    """The /drive command over one DrivingMode."""

    def __init__(self, mode: DrivingMode, logger: Logger) -> None:
        self._mode = mode
        self._logger = logger

    def register(self, registry: CommandRegistry) -> None:
        registry.register(Command("drive", DRIVE_HELP, self._run))

    def _run(self, ctx: CommandContext, args: str) -> None:
        word = args.strip().lower()
        failed = apply(self._mode, WORDS.get(word, word)) if word else None
        if failed is not None:
            ctx.say(failed)
            return
        ctx.say_buttons(panel_text(self._mode), panel_rows(self._mode))
