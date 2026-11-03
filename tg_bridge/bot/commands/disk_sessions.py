"""disk_sessions: the Claude sessions on disk for a topic's folder, remembered for the /resume picker (card H3c).

Moved out of session.py to keep it under the module size limit. Each session found is recorded in the history, so the
picker lists it next to the sessions the bridge ran itself.
"""

import os

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.pickers import RESUME_LIMIT
from tg_bridge.clock import utc_stamp
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherSpec
from tg_bridge.store.sessions import HistoryRepo


def remember_disk_sessions(config: Config, backends: BackendRegistry, history: HistoryRepo, thread: int, cwd: str,
                           current_command: str, taken: set[str]) -> dict[str, str]:
    """Records the sessions on disk for the topic's folder, one listing per Claude config folder.

    Launchers that share a config folder are listed once. Their sessions are attributed to the current launcher when
    it uses that folder, else to the first launcher in config order that does. The ids in taken (sessions another
    topic is bound to) are left out of the listing before its limit applies, so they do not use up its slots.
    Returns the modification stamp of each session found, keyed by its id (the newest copy when a folder is listed
    twice).
    """
    exclude = frozenset(taken)
    owners: dict[str, LauncherSpec] = {}
    for spec in config.launcher_specs().values():
        if spec.backend != BACKEND_CLAUDE or not spec.config_dir:
            continue
        key = os.path.realpath(spec.config_dir)
        if key not in owners or spec.command == current_command:
            owners[key] = spec
    found: dict[str, str] = {}
    for spec in owners.values():
        for disk in backends.get(spec.backend).list_sessions(spec, cwd, RESUME_LIMIT, exclude):
            stamp = utc_stamp(disk.modified)
            history.remember_disk(thread, disk.uuid, started=stamp, first_text=disk.title, cwd=cwd,
                                  launcher=spec.command, backend=spec.backend)
            found[disk.uuid] = max(found.get(disk.uuid, ""), stamp)
    return found
