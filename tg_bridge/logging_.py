"""Logger(path): append + truncate (C03); injectable, no module global.

Port of the live file's log() (C03): each line gets a UTC stamp, and when the file grows past 200 KB only its last
300 lines are kept. Each Logger owns its path and its lock, so nothing is shared through a module global. Every
message goes through redact() before it is written, so a bot token, an api_key= value or a user:password proxy
credential in an exception text never reaches bridge.log. Write errors are swallowed, as the live file did.
"""

import datetime
import os
import re
import threading

MAX_BYTES = 200_000
KEEP_LINES = 300
STAMP_FORMAT = "%m-%d %H:%M:%SZ"
REDACTED = "<redacted>"

BOT_TOKEN = re.compile(r"\d+:[A-Za-z0-9_-]{30,}")
API_KEY = re.compile(r"(api_key=)[^\s&\"',;]+", re.IGNORECASE)
URL_CREDENTIALS = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@]+@")


def redact(text: str) -> str:
    """Mask bot tokens, api_key= values and user:password credentials in proxy URLs."""
    text = URL_CREDENTIALS.sub(lambda m: m.group(1) + REDACTED + "@", text)
    text = BOT_TOKEN.sub(REDACTED, text)
    return API_KEY.sub(lambda m: m.group(1) + REDACTED, text)


class Logger:
    """Appends redacted, stamped lines to one file. Safe to call from several threads."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def log(self, msg: object) -> None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(STAMP_FORMAT)
        line = f"{stamp} {redact(str(msg))}\n"
        with self._lock:
            try:
                self._truncate_if_big()
                with open(self.path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(line)
            except OSError:
                pass

    def _truncate_if_big(self) -> None:
        if not os.path.exists(self.path) or os.path.getsize(self.path) <= MAX_BYTES:
            return
        with open(self.path, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-KEEP_LINES:]
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(tail)
