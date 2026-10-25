"""HandoffService (STEP 37): build_digest keeps the newest entries within the cap, prepare/take store and take the
digest in kv handoff:<thread>, and the digest never carries thinking text.

Scenarios: cap respected with the newest entry kept; oldest-to-newest order in the output; an oversize newest entry is
cut to the cap; empty, blank and unknown-role entries give ''; prepare gives False and no kv row when nothing is
carried (and clears a stale row); take is one-shot, also under four racing threads; a backend session that was never
started (no session id) is not read; the log holds the digest length but never its text; the real Claude reader keeps
user and assistant text and tool lines but no thinking block.
"""

import os
import re
import threading
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.events import HandoffEntry
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherSpec
from tg_bridge.core.handoff import DIGEST_CAP, HEADER, OMITTED, HandoffService, build_digest, handoff_key
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_transcripts",
                       "handoff_session.jsonl")
THREAD = 42
SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000002"
CWD = "/home/x/proj"


def user(text: str) -> HandoffEntry:
    return HandoffEntry(role="user", text=text)


def agent(text: str) -> HandoffEntry:
    return HandoffEntry(role="assistant", text=text)


def tool(text: str) -> HandoffEntry:
    return HandoffEntry(role="tool", text=text)


class StubBackend:
    """Duck-typed backend: returns fixed entries and records each call."""

    def __init__(self, entries: tuple[HandoffEntry, ...] = ()) -> None:
        self.entries = entries
        self.calls: list[tuple[str, str]] = []

    def handoff_entries(self, session_id: str, cwd: str, spec: LauncherSpec) -> tuple[HandoffEntry, ...]:
        self.calls.append((session_id, cwd))
        return self.entries


class StubRegistry:
    """Duck-typed BackendRegistry: every name maps to one stub backend."""

    def __init__(self, backend: object) -> None:
        self.backend = backend

    def get(self, name: str) -> object:
        return self.backend


SPEC = LauncherSpec(name="plain", backend=BACKEND_CLAUDE, command="plain", argv=("plain",), config_dir="/nope")


class BuildDigestTests(unittest.TestCase):
    def test_all_entries_fit_in_order_oldest_first_without_marker(self) -> None:
        digest = build_digest([user("first question"), agent("first answer"), tool("$ pytest -q"), user("thanks")])
        self.assertEqual(
            digest,
            HEADER + "User: first question\nAgent: first answer\nTool: $ pytest -q\nUser: thanks",
        )
        self.assertNotIn(OMITTED, digest)

    def test_labels_map_roles(self) -> None:
        digest = build_digest([user("u"), agent("a"), tool("t")])
        self.assertIn("User: u\nAgent: a\nTool: t", digest)

    def test_cap_respected_and_newest_entries_kept(self) -> None:
        entries = [user(f"old message {i} " + "x" * 300) for i in range(60)]
        entries.append(agent("NEWEST answer"))
        digest = build_digest(entries, cap=2000)
        self.assertLessEqual(len(digest), 2000)
        self.assertTrue(digest.startswith(HEADER + OMITTED + "\n"))
        self.assertTrue(digest.endswith("Agent: NEWEST answer"))
        self.assertNotIn("old message 0 ", digest)

    def test_kept_entries_are_a_contiguous_newest_suffix(self) -> None:
        entries = [user(f"msg-{i:02d} " + "y" * 150) for i in range(40)]
        digest = build_digest(entries, cap=1500)
        kept = [int(re.search(r"msg-(\d+)", part).group(1)) for part in digest.split("\n")[1:]]
        self.assertGreater(len(kept), 1)
        self.assertEqual(kept, list(range(kept[0], 40)))
        self.assertEqual(kept[-1], 39)

    def test_default_cap_is_twelve_thousand(self) -> None:
        self.assertEqual(DIGEST_CAP, 12000)
        digest = build_digest([user("z" * 20000)])
        self.assertLessEqual(len(digest), 12000)

    def test_oversize_newest_entry_is_cut_to_the_cap(self) -> None:
        digest = build_digest([user("old short"), agent("A" * 500 + " END")], cap=200)
        self.assertLessEqual(len(digest), 200)
        self.assertTrue(digest.startswith(HEADER + OMITTED + "\nAgent: AAA"))
        self.assertTrue(digest.endswith("…"))
        self.assertNotIn("old short", digest)
        self.assertNotIn("END", digest)

    def test_oversize_single_entry_fills_the_cap_exactly(self) -> None:
        digest = build_digest([agent("B" * 5000)], cap=300)
        self.assertEqual(len(digest), 300)
        self.assertTrue(digest.startswith(HEADER + OMITTED + "\nAgent: BBB"))

    def test_nothing_fits_when_cap_leaves_no_room(self) -> None:
        self.assertEqual(build_digest([user("hello " * 200)], cap=len(HEADER) + 10), "")

    def test_empty_blank_and_unknown_role_entries_give_empty_string(self) -> None:
        self.assertEqual(build_digest([]), "")
        self.assertEqual(build_digest([user(""), agent("   \n "), HandoffEntry(role="thinking", text="hmm")]), "")

    def test_blank_entries_are_skipped_among_real_ones(self) -> None:
        digest = build_digest([user("  real  "), agent(""), tool("$ ls")])
        self.assertEqual(digest, HEADER + "User: real\nTool: $ ls")


class HandoffServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.db = Database(Paths.from_home(self.home.home))
        self.addCleanup(self.db.close)
        self.kv = KvRepo(self.db)
        self.log_path = os.path.join(self.home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.backend = StubBackend((user("Fix the login bug"), agent("On it"), tool("$ pytest -q")))
        self.service = HandoffService(self.kv, StubRegistry(self.backend), self.logger)

    def test_prepare_stores_digest_in_kv_and_returns_true(self) -> None:
        self.assertTrue(self.service.prepare(THREAD, "claude", SPEC, SID, CWD))
        stored = self.kv.get(handoff_key(THREAD))
        self.assertEqual(stored, build_digest(self.backend.entries))
        self.assertEqual(self.backend.calls, [(SID, CWD)])

    def test_prepare_with_no_entries_returns_false_and_writes_no_row(self) -> None:
        self.backend.entries = ()
        self.assertFalse(self.service.prepare(THREAD, "claude", SPEC, SID, CWD))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))

    def test_prepare_with_no_entries_clears_a_stale_digest(self) -> None:
        self.kv.set(handoff_key(THREAD), HEADER + "User: stale")
        self.backend.entries = ()
        self.assertFalse(self.service.prepare(THREAD, "claude", SPEC, SID, CWD))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))

    def test_prepare_without_session_id_does_not_read_the_backend(self) -> None:
        self.assertFalse(self.service.prepare(THREAD, "claude", SPEC, None, CWD))
        self.assertFalse(self.service.prepare(THREAD, "claude", SPEC, "", CWD))
        self.assertEqual(self.backend.calls, [])
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))

    def test_prepare_without_session_id_keeps_a_pending_digest(self) -> None:
        self.kv.set(handoff_key(THREAD), HEADER + "User: pending")
        self.assertTrue(self.service.prepare(THREAD, "claude", SPEC, None, CWD))
        self.assertEqual(self.kv.get(handoff_key(THREAD)), HEADER + "User: pending")
        self.assertEqual(self.backend.calls, [])

    def test_take_returns_digest_once_then_none(self) -> None:
        self.service.prepare(THREAD, "claude", SPEC, SID, CWD)
        expected = build_digest(self.backend.entries)
        self.assertEqual(self.service.take(THREAD), expected)
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        self.assertIsNone(self.service.take(THREAD))

    def test_take_without_prepare_is_none(self) -> None:
        self.assertIsNone(self.service.take(THREAD))

    def test_digests_are_per_thread(self) -> None:
        self.service.prepare(THREAD, "claude", SPEC, SID, CWD)
        self.assertIsNone(self.service.take(THREAD + 1))
        self.assertIsNotNone(self.service.take(THREAD))

    def test_take_is_one_shot_under_racing_threads(self) -> None:
        self.service.prepare(THREAD, "claude", SPEC, SID, CWD)
        results: list[str | None] = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            value = self.service.take(THREAD)
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(1 for value in results if value is not None), 1)

    def test_log_holds_length_but_never_the_digest_text(self) -> None:
        self.service.prepare(THREAD, "claude", SPEC, SID, CWD)
        digest = build_digest(self.backend.entries)
        with open(self.log_path, encoding="utf-8") as handle:
            logged = handle.read()
        self.assertIn(f"digest of {len(digest)} chars", logged)
        self.assertNotIn("Fix the login bug", logged)
        self.assertNotIn("pytest -q", logged)


class ClaudeTranscriptHandoffTests(unittest.TestCase):
    """The real Claude reader behind a real registry: the digest has the conversation and no thinking text."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.db = Database(Paths.from_home(self.home.home))
        self.addCleanup(self.db.close)
        self.kv = KvRepo(self.db)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config_dir = os.path.join(os.path.realpath(self.home.home), "cfg-claude")
        folder = os.path.join(self.config_dir, "projects", history.encoded_cwd(CWD))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as src, \
                open(os.path.join(folder, SID + ".jsonl"), "w", encoding="utf-8") as dst:
            dst.write(src.read())
        registry = BackendRegistry(Config({}, home=self.home.home), self.logger, FakeClock())
        registry.register(BACKEND_CLAUDE, lambda cfg, log, clk: ClaudeBackend(logger=log, clock=clk))
        self.spec = LauncherSpec(name="wrap", backend=BACKEND_CLAUDE, command="wrap run",
                                 argv=("wrap", "run"), config_dir=self.config_dir)
        self.service = HandoffService(self.kv, registry, self.logger)

    def test_digest_keeps_conversation_and_drops_thinking(self) -> None:
        self.assertTrue(self.service.prepare(THREAD, BACKEND_CLAUDE, self.spec, SID, CWD))
        digest = self.service.take(THREAD)
        self.assertIsNotNone(digest)
        self.assertIn("User: Fix the login bug in src/auth.py", digest)
        self.assertIn("Agent: I will read the file first.", digest)
        self.assertIn("Agent: Tests pass.", digest)
        self.assertNotIn("private reasoning", digest)
        self.assertNotIn("more private", digest)
        self.assertNotIn("sig", digest.split())


if __name__ == "__main__":
    unittest.main()
