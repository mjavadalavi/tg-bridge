"""Two actions the typed commands and the picker taps share on a running session: the model switch and the background
call. Both live here once, so a change to either reaches /model and the model picker alike."""

import threading
from collections.abc import Callable

from tg_bridge.agents.base import AgentSession
from tg_bridge.logging_ import Logger

SET_MODEL_ERROR_CHARS = 120


def spawn(target: Callable[..., object], *args: object, **kwargs: object) -> None:
    """Runs a call that may wait for the CLI on a daemon thread, so the poll thread never waits for it."""
    threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()


def apply_model(session: AgentSession, model: str, logger: Logger) -> None:
    """Switches a running session to model. A failure is logged; the stored value still applies from the next turn."""
    try:
        session.set_model(model)
    except Exception as exc:
        logger.log(f"set_model {model} failed: {str(exc)[:SET_MODEL_ERROR_CHARS]}")
