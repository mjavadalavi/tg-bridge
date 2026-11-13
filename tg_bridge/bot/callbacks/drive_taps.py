"""DriveTaps: the buttons of the /drive panel (dr:on, dr:off, dr:female, dr:male).

A tap applies its action to the DrivingMode (bot/commands/driving.apply) and edits the panel to the new state, with
its buttons, so the panel always shows what is on. An unknown action is told in the toast and the panel stays as it is.
"""

from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.commands.driving import DRIVE_PREFIX, VOICE_LABELS, apply, panel_rows, panel_text
from tg_bridge.core.driving import DrivingMode
from tg_bridge.logging_ import Logger

DONE_TEXT = "🚗 done"


class DriveTaps:
    """The dr prefix: one handler for every button of the panel."""

    def __init__(self, *, mode: DrivingMode, logger: Logger) -> None:
        self._mode = mode
        self._logger = logger

    def register(self, router: CallbackRouter) -> None:
        router.on(DRIVE_PREFIX, self._tap)

    def _tap(self, tap: Tap) -> None:
        action = tap.args[0] if tap.args else ""
        failed = apply(self._mode, action)
        if failed is not None:
            tap.toast(failed)
            return
        tap.toast(VOICE_LABELS.get(action, DONE_TEXT))
        tap.edit(panel_text(self._mode), panel_rows(self._mode))
