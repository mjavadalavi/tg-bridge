"""LauncherSwitch (STEP 40): /launcher and its taps settle the session.

Scenarios: the same launcher says so and changes nothing; a running turn refuses the change; a launcher sharing the
folder keeps the id; another Claude folder gets a copy (thinking stripped, history row adopted); a missing transcript
gives the 'not found' text; a topic with no history gets the fresh-session text; claude -> codex stores a handoff and
clears the id; codex -> claude reads the rollout fixture; claude -> codex -> claude keeps sessions.launcher a Claude
command; an unknown name is refused.
"""

import json
import os
import shutil
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.handoff import read_entries as read_claude_entries
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.codex.handoff import read_entries as read_codex_entries
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.config import Config
from tg_bridge.core.handoff import HEADER, HandoffService, build_digest, handoff_key
from tg_bridge.core.launcher_switch import (
    BUSY_TEXT,
    CONTINUED_TEXT,
    COPIED_TEXT,
    FRESH_TEXT,
    HANDOFF_TEXT,
    KIND_BUSY,
    KIND_CHANGED,
    KIND_SAME,
    NOT_FOUND_TEXT,
    SAME_TEXT,
    LauncherSwitch,
)
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CLAUDE_TRANSCRIPT = os.path.join(FIXTURES, "claude_transcripts", "handoff_session.jsonl")
CODEX_ROLLOUT = os.path.join(FIXTURES, "codex_rollout.jsonl")
THREAD = 4040
TOPIC = "LX"
SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000002"
CODEX_THREAD = "01a16f09-bcb0-70b3-a1ff-fefc0cc5402c"
ROLLOUT_NAME = "rollout-2026-10-24T17-51-12-" + CODEX_THREAD + ".jsonl"
LIVE_BUSY = object()   # any non-None value means a turn runs (the switch refuses while a turn is live)
CLAUDE_CMD = "wrap run"


def layout(home: str) -> dict[str, str]:
    """The folders of one test home: two Claude config folders, the Codex home and the topic's project folder."""
    return {
        "cfg_a": os.path.join(home, "claude-a"),
        "cfg_b": os.path.join(home, "claude-b"),
        "codex": os.path.join(home, "codex-home"),
        "project": os.path.join(home, "project"),
    }


def config_data(paths: dict[str, str]) -> dict:
    """The config both files read: 'launchers' (Claude, with folders), 'agent_launchers' (Codex) and the topic."""
    return {
        "launchers": {
            "wrap": {"cmd": CLAUDE_CMD, "config_dir": paths["cfg_a"]},
            "plain": {"cmd": "plain", "config_dir": paths["cfg_b"]},
            "shared": {"cmd": "shared-cmd", "config_dir": paths["cfg_a"]},
        },
        "agent_launchers": {"codex": {"backend": "codex", "command": "codex", "config_dir": paths["codex"]}},
        "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": paths["project"]}},
    }


def clear_transcripts(paths: dict[str, str]) -> None:
    """Removes every Claude transcript of a home, so each scenario starts with exactly the ones it writes."""
    for key in ("cfg_a", "cfg_b"):
        shutil.rmtree(os.path.join(paths[key], "projects"), ignore_errors=True)


def transcript_with_thinking() -> str:
    """A two-line Claude transcript whose assistant message holds a thinking block next to the text."""
    user = {"type": "user", "sessionId": SID, "cwd": "", "message": {"role": "user", "content": "hello there"}}
    assistant = {"type": "assistant", "sessionId": SID, "message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "private", "signature": "sig"},
        {"type": "text", "text": "hi back"}]}}
    return json.dumps(user) + "\n" + json.dumps(assistant) + "\n"


class SwitchCase(unittest.TestCase):
    """A real config, registry and repositories in a migrated database under a TempHome."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.paths = layout(self.base)
        os.makedirs(self.paths["project"], exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(config_data(self.paths), home=self.base, env={})
        self.registry = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.handoff = HandoffService(self.kv, self.registry, self.logger)
        self.switch = LauncherSwitch(self.config, self.sessions, self.history, self.handoff, self.registry)
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.paths["project"], "bypassPermissions")

    def set_session(self, uuid: str = "", pick: str = "", backend: str = "claude") -> None:
        self.sessions.set_uuid(THREAD, uuid)
        self.sessions.set_launcher_pick(THREAD, pick)
        self.sessions.set_backend(THREAD, backend)
        if uuid:
            self.history.note_session(THREAD, uuid, "earlier question", self.paths["project"], CLAUDE_CMD, "claude")

    def put_claude_transcript(self, config_dir: str, sid: str, source: str | None = None) -> str:
        folder = os.path.join(config_dir, "projects", encoded_cwd(os.path.realpath(self.paths["project"])))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, sid + ".jsonl")
        if source is None:
            with open(CLAUDE_TRANSCRIPT, encoding="utf-8") as handle:
                content = handle.read()
        else:
            content = source
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def put_rollout(self) -> str:
        day = os.path.join(self.paths["codex"], "sessions", "2026", "10", "24")
        os.makedirs(day, exist_ok=True)
        path = os.path.join(day, ROLLOUT_NAME)
        with open(CODEX_ROLLOUT, encoding="utf-8") as handle:
            content = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def row(self):
        return self.sessions.get(THREAD)

    def read_file(self, path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def change(self, new: str, live: object | None = None) -> tuple[str, str]:
        return self.switch.set(THREAD, TOPIC, live, new)


class SameLauncherTests(SwitchCase):
    def test_same_launcher_says_so_and_stores_nothing(self) -> None:
        self.set_session(SID, pick="")
        kind, reply = self.change("wrap")
        self.assertEqual((kind, reply), (KIND_SAME, SAME_TEXT.format(name="wrap")))
        self.assertEqual(reply, "🚀 launcher: wrap — already the launcher, nothing changed")
        self.assertEqual(self.row().uuid, SID)
        self.assertEqual(self.row().launcher_pick, "")

    def test_same_as_stored_pick_also_counts(self) -> None:
        self.set_session(SID, pick="plain")
        self.assertEqual(self.change("plain")[0], KIND_SAME)


class BusyTests(SwitchCase):
    def test_running_turn_refuses_the_change(self) -> None:
        self.set_session(SID)
        kind, reply = self.change("plain", live=LIVE_BUSY)
        self.assertEqual((kind, reply), (KIND_BUSY, BUSY_TEXT))
        self.assertEqual(reply, "a turn is running — send /stop first, then try again.")
        self.assertEqual(self.row().launcher_pick, "")
        self.assertEqual(self.row().uuid, SID)

    def test_same_launcher_is_checked_before_busy(self) -> None:
        self.set_session(SID)
        self.assertEqual(self.change("wrap", live=LIVE_BUSY)[0], KIND_SAME)


class SameBackendTests(SwitchCase):
    def test_shared_folder_keeps_the_id(self) -> None:
        """Ported labels:
        4a: same folder says so ('🚀 launcher: lb — continuing the same conversation')
        4b: same folder keeps uuid, nothing moved to prev_uuid
        4c: same folder still records the new launcher for /resume
        4d: same folder: the transcript is not rewritten
        """
        self.set_session(SID)
        transcript = self.put_claude_transcript(self.paths["cfg_a"], SID)
        before = self.read_file(transcript)
        kind, reply = self.change("shared")
        self.assertEqual((kind, reply), (KIND_CHANGED, CONTINUED_TEXT.format(name="shared")))
        row = self.row()
        self.assertEqual((row.uuid, row.prev_uuid, row.launcher_pick), (SID, "", "shared"))
        self.assertEqual(self.history.find(THREAD, SID).launcher, "shared-cmd")
        self.assertEqual(self.read_file(transcript), before)

    def test_other_folder_gets_a_copy_without_thinking(self) -> None:
        """Ported labels:
        3a: switching to a folder-separate launcher copies and says so
        3b: the transcript is in the new launcher folder under the same cwd folder
        3i: the uuid is kept, no new session
        3j: the pick is stored
        3k: session_history records the new launcher command for the uuid
        """
        self.set_session(SID)
        self.put_claude_transcript(self.paths["cfg_a"], SID, transcript_with_thinking())
        kind, reply = self.change("plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, COPIED_TEXT.format(name="plain")))
        self.assertEqual(self.row().uuid, SID)
        copied = os.path.join(self.paths["cfg_b"], "projects", encoded_cwd(os.path.realpath(self.paths["project"])),
                              SID + ".jsonl")
        with open(copied, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("hi back", text)
        self.assertNotIn("thinking", text)
        self.assertEqual(self.history.find(THREAD, SID).launcher, "plain")
        self.assertEqual((self.row().prev_uuid, self.row().launcher_pick), ("", "plain"))

    def test_transcript_in_another_folder_is_found_and_copied(self) -> None:
        """Ported labels:
        6a: a transcript in another folder is found and copied
        6b: the copy lands under the cwd folder of the new launcher
        """
        self.set_session(SID)
        other = os.path.join(self.paths["cfg_a"], "projects", "some-other-folder")
        os.makedirs(other)
        with open(os.path.join(other, SID + ".jsonl"), "w", encoding="utf-8") as handle:
            handle.write(transcript_with_thinking())
        kind, reply = self.change("plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, COPIED_TEXT.format(name="plain")))
        copied = os.path.join(self.paths["cfg_b"], "projects", encoded_cwd(os.path.realpath(self.paths["project"])),
                              SID + ".jsonl")
        self.assertTrue(os.path.isfile(copied))

    def test_a_failed_copy_falls_back_to_a_fresh_session(self) -> None:
        """Ported labels:
        10c: a failed copy falls back to a fresh session
        """
        self.set_session(SID)
        self.put_claude_transcript(self.paths["cfg_a"], SID)
        os.makedirs(self.paths["cfg_b"])
        with open(os.path.join(self.paths["cfg_b"], "projects"), "w", encoding="utf-8") as handle:
            handle.write("x")  # a file where the copy's folder must go
        kind, reply = self.change("plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, NOT_FOUND_TEXT.format(name="plain")))
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", SID))

    def test_missing_transcript_starts_fresh_and_moves_the_id(self) -> None:
        """Ported labels:
        5a: missing source says so ('🚀 launcher: lb — previous history not found, the next message starts a fresh session')
        5b: fresh session, the old uuid kept in prev_uuid for /resume
        5c: nothing was written on a missing source
        """
        self.set_session(SID)
        kind, reply = self.change("plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, NOT_FOUND_TEXT.format(name="plain")))
        self.assertEqual(reply, "🚀 launcher: plain — previous history not found, the next message starts a fresh session")
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", SID))
        self.assertFalse(os.path.exists(os.path.join(self.paths["cfg_b"], "projects")))

    def test_topic_without_history_says_the_fresh_text(self) -> None:
        """Ported labels:
        5d: a topic with no session says the old text
        """
        self.set_session("")
        kind, reply = self.change("plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, FRESH_TEXT.format(name="plain")))
        self.assertEqual(reply, "🚀 launcher: plain — the next message starts a fresh session")
        self.assertEqual(self.row().launcher_pick, "plain")

    def test_unmatched_stored_command_counts_as_no_launcher(self) -> None:
        # a topic whose stored command matches no visible launcher is unconfigured (live launcher_state: name None),
        # so the switch does not copy or continue its history even though its folder would match
        alt = THREAD + 1
        self.sessions.ensure(alt, "side", "altclaude --x", self.paths["project"])
        self.sessions.set_uuid(alt, SID)
        self.history.note_session(alt, SID, "earlier question", self.paths["project"], "altclaude --x", "claude")
        self.put_claude_transcript(os.path.join(self.base, ".claude"), SID)
        kind, reply = self.switch.set(alt, "side", None, "plain")
        self.assertEqual((kind, reply), (KIND_CHANGED, NOT_FOUND_TEXT.format(name="plain")))
        self.assertEqual((self.sessions.get(alt).uuid, self.sessions.get(alt).prev_uuid), ("", SID))

    def test_string_and_dict_entries_with_one_folder_share_the_session(self) -> None:
        """Ported labels:
        8i: a string and a dict entry with the same folder share the session
        """
        data = {"launchers": {"x": "cmd x", "y": {"cmd": "cmd y", "config_dir": "~/.claude"}},
                "topics": {TOPIC: {"launcher": "cmd x", "cwd": self.paths["project"]}}}
        switch = LauncherSwitch(Config(data, home=self.base, env={}), self.sessions, self.history, self.handoff,
                                self.registry)
        self.set_session(SID)
        kind, reply = switch.set(THREAD, TOPIC, None, "y")
        self.assertEqual((kind, reply), (KIND_CHANGED, CONTINUED_TEXT.format(name="y")))
        self.assertEqual(reply, "🚀 launcher: y — continuing the same conversation")
        self.assertEqual(self.row().uuid, SID)

    def test_unknown_launcher_name_is_refused(self) -> None:
        self.set_session(SID)
        with self.assertRaises(ValueError):
            self.change("nope")
        self.assertEqual(self.row().uuid, SID)


class CrossBackendTests(SwitchCase):
    def test_claude_to_codex_stores_a_handoff_and_clears_the_id(self) -> None:
        self.set_session(SID)
        transcript = self.put_claude_transcript(self.paths["cfg_a"], SID)
        kind, reply = self.change("codex")
        self.assertEqual((kind, reply), (KIND_CHANGED, HANDOFF_TEXT.format(name="codex")))
        self.assertEqual(reply, "🚀 codex — continuing with a summary of the previous conversation")
        digest = self.kv.get(handoff_key(THREAD))
        expected = build_digest(read_claude_entries(transcript, self.paths["project"]))
        self.assertTrue(expected.startswith(HEADER))
        self.assertEqual(digest, expected)
        row = self.row()
        self.assertEqual((row.uuid, row.prev_uuid, row.backend, row.launcher_pick),
                         ("", SID, "codex", "codex"))
        self.assertEqual(row.launcher, CLAUDE_CMD)

    def test_a_quick_switch_back_before_any_message_keeps_the_pending_digest(self) -> None:
        self.set_session(SID)
        self.put_claude_transcript(self.paths["cfg_a"], SID)
        self.change("codex")
        digest = self.kv.get(handoff_key(THREAD))
        self.assertTrue(digest.startswith(HEADER))
        kind, reply = self.change("wrap")
        self.assertEqual((kind, reply), (KIND_CHANGED, HANDOFF_TEXT.format(name="wrap")))
        self.assertEqual(self.kv.get(handoff_key(THREAD)), digest)
        row = self.row()
        self.assertEqual((row.uuid, row.backend), ("", "claude"))

    def test_claude_to_codex_without_history_says_fresh_and_stores_nothing(self) -> None:
        self.set_session("")
        kind, reply = self.change("codex")
        self.assertEqual((kind, reply), (KIND_CHANGED, FRESH_TEXT.format(name="codex")))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        self.assertEqual(self.row().backend, "codex")

    def test_claude_to_codex_with_missing_transcript_says_not_found(self) -> None:
        self.set_session(SID)
        kind, reply = self.change("codex")
        self.assertEqual((kind, reply), (KIND_CHANGED, NOT_FOUND_TEXT.format(name="codex")))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", SID))

    def test_codex_to_claude_reads_the_rollout_fixture(self) -> None:
        self.set_session(CODEX_THREAD, pick="codex", backend="codex")
        self.put_rollout()
        kind, reply = self.change("wrap")
        self.assertEqual((kind, reply), (KIND_CHANGED, HANDOFF_TEXT.format(name="wrap")))
        digest = self.kv.get(handoff_key(THREAD))
        self.assertEqual(digest, build_digest(read_codex_entries(CODEX_ROLLOUT)))
        self.assertNotIn("SCRUBBED developer", digest)
        row = self.row()
        self.assertEqual((row.uuid, row.prev_uuid, row.backend, row.launcher_pick),
                         ("", CODEX_THREAD, "claude", "wrap"))

    def test_claude_codex_claude_keeps_sessions_launcher_a_claude_command(self) -> None:
        self.set_session(SID)
        self.put_claude_transcript(self.paths["cfg_a"], SID)
        self.assertEqual(self.change("codex")[0], KIND_CHANGED)
        self.sessions.set_uuid(THREAD, CODEX_THREAD)   # a codex turn ran and recorded its thread id
        self.put_rollout()
        self.assertEqual(self.change("wrap")[0], KIND_CHANGED)
        row = self.row()
        self.assertEqual(row.launcher, CLAUDE_CMD)
        self.assertEqual((row.backend, row.launcher_pick), ("claude", "wrap"))
        self.assertEqual(self.kv.get(handoff_key(THREAD)), build_digest(read_codex_entries(CODEX_ROLLOUT)))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
