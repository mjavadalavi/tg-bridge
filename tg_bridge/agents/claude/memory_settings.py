"""The per-topic auto-memory folder of a Claude session, as a value in its --settings JSON.

Claude Code keeps one auto-memory per working folder (cwd). Several Telegram topics share the cwd ~/workspace, so their
memories mixed in one folder. The `autoMemoryDirectory` setting points a session at a folder of its own, one per topic,
and it is honoured when it comes from --settings. with_memory() merges it into the --settings JSON the session already
gets (the permission hook, or nothing), so the topic's working folder (cwd) stays as it is.
"""

import json

MEMORY_SETTING = "autoMemoryDirectory"


def with_memory(settings: str, memory_dir: str) -> str:
    """The compact --settings JSON of settings with autoMemoryDirectory set to memory_dir. settings is the JSON already
    given ('' = none) and keeps every key it has. A memory_dir of '' returns settings unchanged."""
    if not memory_dir:
        return settings
    merged = json.loads(settings) if settings else {}
    merged[MEMORY_SETTING] = memory_dir
    return json.dumps(merged, separators=(",", ":"))
