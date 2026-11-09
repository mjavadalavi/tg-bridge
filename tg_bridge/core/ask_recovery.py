"""Start-up recovery of permission asks: every ask still pending in the database was never answered and its turn is
gone (the bridge restarted). Each one is voided, and its card, when it has one, is closed with a restart line.

PermissionService.recover_cards calls recover_asks once at start-up, before any turn runs.
"""

from typing import Any

from tg_bridge.logging_ import Logger
from tg_bridge.store.asks import VOID, AskRepo
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD
from tg_bridge.telegram.messenger import Messenger

VOID_RESTART_TEXT = "\n\n⌛ void — bridge restarted"
DETAIL_KEYS = ("command", "file_path", "notebook_path", "path")
DETAIL_CHARS = 600
ERROR_TEXT_LIMIT = 150


def _detail(inp: dict[str, Any]) -> str:
    """The text a recovered card shows under its title: the command or path, else the input as JSON."""
    for key in DETAIL_KEYS:
        value = inp.get(key)
        if isinstance(value, str) and value:
            return value[:DETAIL_CHARS]
    return str(inp)[:DETAIL_CHARS]


def recover_asks(asks: AskRepo, messenger: Messenger, logger: Logger) -> int:
    """Voids every pending ask and closes its card. Returns how many asks were voided."""
    count = 0
    for record in asks.pending():
        if not asks.answer(record.nonce, VOID):
            continue
        count += 1
        if record.message_id is None:
            continue
        text = f"🔐 {record.tool_name}\n{_detail(record.tool_input())}{VOID_RESTART_TEXT}"
        try:
            messenger.edit_text(record.chat_id, record.message_id, text, reply_markup=EMPTY_KEYBOARD)
        except Exception as exc:
            logger.log(f"recover card {record.nonce}: {str(exc)[:ERROR_TEXT_LIMIT]}")
    return count
