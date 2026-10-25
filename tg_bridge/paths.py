"""Paths dataclass derived from HOME at construction (not at import).

Every file the bridge reads or writes lives under one state directory, `<home>/.local/state/tg-bridge`. The
live file computed these names at import time from HOME; here they are computed when a Paths object is built,
so a test builds one under its own TempHome and nothing is shared between tests.
"""

import os
from dataclasses import dataclass

STATE_SUBDIR = os.path.join(".local", "state", "tg-bridge")


@dataclass(frozen=True)
class Paths:
    """Absolute locations of the bridge's files and folders for one home directory."""

    home: str
    state: str
    token: str
    soniox_key: str
    config: str
    db: str
    spool: str
    inbox: str
    log: str
    lock: str
    turns: str

    @classmethod
    def from_home(cls, home: str) -> "Paths":
        """The paths for the given home directory. Nothing is created on disk here."""
        state = os.path.join(home, STATE_SUBDIR)
        return cls(
            home=home,
            state=state,
            token=os.path.join(state, "token"),
            soniox_key=os.path.join(state, "soniox-key"),
            config=os.path.join(state, "config.json"),
            db=os.path.join(state, "tg-bridge.db"),
            spool=os.path.join(state, "spool"),
            inbox=os.path.join(state, "inbox"),
            log=os.path.join(state, "bridge.log"),
            lock=os.path.join(state, "lock"),
            turns=os.path.join(state, "turns"),
        )

    @classmethod
    def from_environment(cls) -> "Paths":
        """The paths for the current user's home directory (HOME, as expanduser reads it)."""
        return cls.from_home(os.path.expanduser("~"))
