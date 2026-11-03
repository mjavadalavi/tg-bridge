"""The SessionStart hook command and its --settings JSON (session follow): the argv, the entry script it names and the
settings shape Claude Code reads. Nothing here runs a process; the JSON is parsed back and checked field by field.
"""

import json
import os
import shlex
import unittest

from tg_bridge.agents.claude.session_hook import (
    SESSION_START_EVENT,
    session_start_command,
    session_start_settings,
)

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tg-bridge.py")


class SessionStartHookTests(unittest.TestCase):
    def test_the_command_is_this_interpreter_isolated_the_entry_script_and_the_words_with_thread_and_pid(self) -> None:
        command = session_start_command(42, 7001, "/usr/bin/python3")
        self.assertEqual(command[:2], ["/usr/bin/python3", "-I"])
        self.assertEqual(os.path.realpath(command[2]), os.path.realpath(ENTRY))
        self.assertEqual(command[3:], ["hook", "session-start", "42", "7001"])

    def test_the_settings_install_one_session_start_hook_for_every_source(self) -> None:
        command = session_start_command(42, 7001, "/usr/bin/python3")
        settings = json.loads(session_start_settings(command))
        self.assertEqual(list(settings), ["hooks"])
        self.assertEqual(list(settings["hooks"]), [SESSION_START_EVENT])
        groups = settings["hooks"][SESSION_START_EVENT]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["matcher"], "*")
        self.assertEqual(len(groups[0]["hooks"]), 1)
        entry = groups[0]["hooks"][0]
        self.assertEqual(entry["type"], "command")
        self.assertEqual(entry["timeout"], 10)
        self.assertEqual(shlex.split(entry["command"]), command)
