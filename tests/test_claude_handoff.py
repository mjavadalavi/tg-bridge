"""Claude handoff reader (agents/claude/handoff.py) and ClaudeBackend.handoff_entries (DESIGN STEP 21).

Fixture: tests/fixtures/claude_transcripts/handoff_session.jsonl, built from the transcript shapes in PROTOCOL.md.
Scenarios: order preserved; user and assistant text kept; tool calls become 'tool' entries labelled by tools.py;
thinking, tool_result text, meta and system lines, non-JSON and non-object lines, and empty text are absent; a
missing transcript, a malformed session id or an empty config folder gives (); the backend finds the transcript
under the cwd's encoded folder and, failing that, in any other folder.
"""

import json
import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.claude.handoff import read_entries
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.claude.tools import tool_label
from tg_bridge.agents.events import HandoffEntry
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_transcripts",
                       "handoff_session.jsonl")
CWD = "/home/x/proj"
SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000002"

# (role, text) in transcript order, as read_entries(FIXTURE, CWD) must return them.
EXPECTED = [
    ("user", "Fix the login bug in src/auth.py"),
    ("assistant", "I will read the file first."),
    ("tool", "Read src/auth.py"),
    ("tool", "Bash Run the tests"),
    ("assistant", "Tests pass."),
    ("user", "Thanks, now commit it"),
    ("user", "ünïcode ✓"),
    ("tool", 'Grep "login" in src'),
    ("tool", "srv:tool"),
]


def _as_pairs(entries) -> list[tuple[str, str]]:
    return [(entry.role, entry.text) for entry in entries]


class ReadEntriesTests(unittest.TestCase):
    def test_fixture_yields_conversation_in_transcript_order(self) -> None:
        entries = read_entries(FIXTURE, CWD)
        self.assertIsInstance(entries, tuple)
        self.assertTrue(all(isinstance(entry, HandoffEntry) for entry in entries))
        self.assertEqual(_as_pairs(entries), EXPECTED)

    def test_tool_labels_match_tools_py(self) -> None:
        tool_texts = [entry.text for entry in read_entries(FIXTURE, CWD) if entry.role == "tool"]
        self.assertEqual(tool_texts, [
            tool_label("Read", {"file_path": "/home/x/proj/src/auth.py"}, CWD),
            tool_label("Bash", {"command": "pytest -q", "description": "Run the tests"}, CWD),
            tool_label("Grep", {"pattern": "login", "path": "/home/x/proj/src"}, CWD),
            tool_label("mcp__srv__tool", {}, CWD),
        ])

    def test_cwd_none_keeps_absolute_tool_paths(self) -> None:
        texts = [entry.text for entry in read_entries(FIXTURE) if entry.role == "tool"]
        self.assertIn("Read /home/x/proj/src/auth.py", texts)

    def test_thinking_tool_results_and_meta_text_are_absent(self) -> None:
        blob = json.dumps(_as_pairs(read_entries(FIXTURE, CWD)), ensure_ascii=False)
        for forbidden in ("private reasoning", "more private", "only thinking", "def login", "ignore me",
                          "context compacted", "Login fix", "not json", "1, 2, 3"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, blob)

    def test_blank_text_and_nameless_tool_use_are_skipped(self) -> None:
        texts = [entry.text for entry in read_entries(FIXTURE, CWD)]
        self.assertNotIn("", texts)
        self.assertNotIn("   ", texts)
        self.assertEqual(len([t for t in texts if t.startswith("Bash")]), 1)  # the nameless tool_use adds nothing

    def test_missing_or_empty_path_gives_empty_tuple(self) -> None:
        with TempHome() as home:
            self.assertEqual(read_entries(os.path.join(home.home, "nope.jsonl"), CWD), ())
        self.assertEqual(read_entries(None, CWD), ())
        self.assertEqual(read_entries("", CWD), ())

    def test_directory_path_gives_empty_tuple(self) -> None:
        with TempHome() as home:
            self.assertEqual(read_entries(home.home, CWD), ())

    def test_bad_lines_do_not_stop_the_read(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.home, "t.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"type": "user", "message": {"content": "first"}}\n')
                handle.write("{broken json\n")
                handle.write('"just a string"\n')
                handle.write('{"type": "user", "message": {"content": 42}}\n')
                handle.write('{"type": "user", "message": {"content": "last"}}')  # no trailing newline
            self.assertEqual(_as_pairs(read_entries(path, CWD)), [("user", "first"), ("user", "last")])

    def test_invalid_utf8_bytes_are_replaced_not_raised(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.home, "bad.jsonl")
            with open(path, "wb") as handle:
                handle.write(b'{"type": "user", "message": {"content": "ok \xff end"}}\n')
            entries = read_entries(path, CWD)
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].text.startswith("ok "))


class ClaudeBackendHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.real_home = os.path.realpath(self.home.home)
        self.config_dir = os.path.join(self.real_home, "cfg-claude")
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.backend = ClaudeBackend(logger=self.logger, clock=FakeClock())
        self.spec = LauncherSpec(name="claude", backend="claude", command="claude", argv=("claude",),
                                 config_dir=self.config_dir)

    def put_fixture(self, folder: str, sid: str = SID) -> str:
        path = os.path.join(self.config_dir, "projects", folder, sid + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return path

    def test_reads_the_transcript_under_the_cwd_folder(self) -> None:
        self.put_fixture(encoded_cwd(CWD))
        self.assertEqual(_as_pairs(self.backend.handoff_entries(SID, CWD, self.spec)), EXPECTED)

    def test_falls_back_to_a_scan_of_every_folder(self) -> None:
        self.put_fixture("-some-other-folder")
        self.assertEqual(_as_pairs(self.backend.handoff_entries(SID, CWD, self.spec)), EXPECTED)

    def test_missing_transcript_gives_empty_tuple(self) -> None:
        self.assertEqual(self.backend.handoff_entries(SID, CWD, self.spec), ())

    def test_malformed_session_id_gives_empty_tuple(self) -> None:
        self.put_fixture(encoded_cwd(CWD), sid="../escape")
        self.assertEqual(self.backend.handoff_entries("../escape", CWD, self.spec), ())

    def test_empty_config_folder_gives_empty_tuple(self) -> None:
        self.put_fixture(encoded_cwd(CWD))
        blank = LauncherSpec(name="c", backend="claude", command="claude", argv=("claude",), config_dir="")
        self.assertEqual(self.backend.handoff_entries(SID, CWD, blank), ())


if __name__ == "__main__":
    unittest.main()
