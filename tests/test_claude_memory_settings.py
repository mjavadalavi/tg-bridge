"""with_memory (the autoMemoryDirectory value of a session's --settings JSON): it is merged into empty settings and into
an existing hook JSON, which keeps its hook, and an empty folder leaves the JSON unchanged. Pure functions, no process.
"""

import json
import unittest

from tg_bridge.agents.claude.hookbridge import hook_settings
from tg_bridge.agents.claude.memory_settings import MEMORY_SETTING, with_memory

MEMORY = "/state/memory/7"


class WithMemoryTests(unittest.TestCase):
    def test_the_setting_is_the_name_claude_code_reads(self) -> None:
        self.assertEqual(MEMORY_SETTING, "autoMemoryDirectory")

    def test_empty_settings_become_the_memory_setting_alone_in_compact_form(self) -> None:
        self.assertEqual(with_memory("", MEMORY), '{"autoMemoryDirectory":"/state/memory/7"}')

    def test_an_existing_hook_json_keeps_its_hook_and_gains_the_memory_setting(self) -> None:
        hook = hook_settings(["python3", "-I", "x.py", "hook", "permission"], 600)
        merged = with_memory(hook, MEMORY)
        self.assertEqual(json.loads(merged)["hooks"], json.loads(hook)["hooks"])
        self.assertEqual(json.loads(merged)[MEMORY_SETTING], MEMORY)
        self.assertEqual(merged, json.dumps({**json.loads(hook), MEMORY_SETTING: MEMORY}, separators=(",", ":")))

    def test_an_empty_folder_leaves_the_settings_unchanged(self) -> None:
        self.assertEqual(with_memory("", ""), "")
        hook = hook_settings(["python3", "x.py"], 30)
        self.assertEqual(with_memory(hook, ""), hook)

    def test_a_new_folder_replaces_an_earlier_memory_setting(self) -> None:
        self.assertEqual(with_memory(with_memory("", "/a"), "/b"), '{"autoMemoryDirectory":"/b"}')


if __name__ == "__main__":
    unittest.main()
