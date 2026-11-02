"""`tg-bridge view-delete <thread>` (card D3): the view's Delete topic request, answered by the daemon.

request_topic_delete writes <view>/<thread>.delete atomically, after removing any old <thread>.deleting and <thread>.deleted.
Then it polls every POLL_STEP seconds, in two phases:
- no <thread>.deleting (the daemon's acknowledgement) within ACK_SECONDS: 'the bridge daemon did not answer'. The request
  file is removed, so the daemon does not act on it later;
- once the acknowledgement is seen: up to ANSWER_SECONDS for <thread>.deleted (the daemon's one-line answer). When that
  runs out, the removal is still going on in the daemon: the text says so and does not claim a failure.
The answer file is removed once read. show_in_tmux puts the answer on the tmux status line for a few seconds; it is only
called inside tmux.
"""

import os
import subprocess
from typing import Final

from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.core.delete_requests import (FAILED_PREFIX, ack_path, remove_quietly, request_path, result_path,
                                            write_atomic)

ACK_SECONDS: Final[float] = 10.0  # how long the daemon may take to acknowledge a request
ANSWER_SECONDS: Final[float] = 120.0  # how long a taken request may take to finish (stop, Telegram retries)
POLL_STEP: Final[float] = 1.0
DISPLAY_MILLISECONDS: Final[str] = "5000"
TIMEOUT_TEXT: Final[str] = "the bridge daemon did not answer — is it running?"
STILL_TEXT: Final[str] = "still deleting in the background — check Telegram"
REQUEST_TEXT: Final[str] = "requested {stamp}"
WRITE_FAILED_TEXT: Final[str] = "could not write the delete request: {error}"


def _take_answer(path: str) -> str | None:
    """The answer's one line, with its file removed; None while the daemon has not answered."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except FileNotFoundError:
        return None
    remove_quietly(path)
    return text


def request_topic_delete(view_dir: str, thread: int, clock: Clock) -> tuple[bool, str]:
    """Asks the daemon to delete the topic. Returns (ok, the one-line result). ok is False for a refusal, a failure
    and a missing daemon. A removal the daemon took but did not finish in time is ok (nothing has failed yet)."""
    answer = result_path(view_dir, thread)
    request = request_path(view_dir, thread)
    for stale in (answer, ack_path(view_dir, thread)):  # old files must not be taken as this request's
        remove_quietly(stale)
    try:
        write_atomic(request, REQUEST_TEXT.format(stamp=utc_stamp(clock.now())))
    except OSError as exc:
        return False, WRITE_FAILED_TEXT.format(error=exc)
    deadline = clock.monotonic() + ACK_SECONDS
    acknowledged = False
    while True:
        text = _take_answer(answer)
        if text is not None:
            return not text.startswith(FAILED_PREFIX), text
        if not acknowledged and os.path.exists(ack_path(view_dir, thread)):
            acknowledged = True
            deadline = clock.monotonic() + ANSWER_SECONDS
        if clock.monotonic() >= deadline:
            if acknowledged:
                return True, STILL_TEXT
            remove_quietly(request)  # nobody took it: the daemon must not act on it later
            return False, TIMEOUT_TEXT
        clock.sleep(POLL_STEP)


def show_in_tmux(text: str) -> None:
    """Shows the text in the tmux client for DISPLAY_MILLISECONDS. The text is one argument, never a shell line."""
    try:
        subprocess.run(["tmux", "display-message", "-d", DISPLAY_MILLISECONDS, text], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass  # no tmux on this machine: the printed line is the answer
