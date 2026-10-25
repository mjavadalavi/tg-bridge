"""Claude transcript carry-over (agents/claude/history.py and ClaudeBackend.carry_history).

Scenarios: the transcript is found under the cwd's encoded folder and, failing that, by a scan of every folder; a
malformed session id finds nothing; thinking blocks are stripped and a message left empty gets a placeholder text;
an existing destination is replaced only when it is shorter; a failing write leaves no temp file; a failure is logged
and never raised.
"""

import json
import os
import unittest
from unittest import mock

from tests.support import TempHome
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.claude.history import (
    carry_history,
    copy_transcript,
    encoded_cwd,
    strip_thinking_line,
    transcript_path,
)
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger
from tests.fakes.clock import FakeClock

SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000001"


def sample_lines() -> list[str]:
    """A transcript with plain text, thinking, redacted thinking, a non-JSON line and an unchanged unicode line."""
    return [
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi there"}}) + "\n",
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "private", "signature": "s"},
            {"type": "text", "text": "hello"}]}}) + "\n",
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "redacted_thinking", "data": "opaque"}]}}) + "\n",
        "not json at all\n",
        json.dumps({"type": "user", "message": {"role": "user", "content": "ünïcode ✓"}}, ensure_ascii=False) + "\n",
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "last", "signature": "s"}]}}),  # no trailing newline
    ]


class HistoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.real_home = os.path.realpath(self.home.home)
        self.cwd = os.path.join(self.real_home, "work", "my proj")
        os.makedirs(self.cwd)
        self.cfg_a = os.path.join(self.real_home, "cfg-a")
        self.cfg_b = os.path.join(self.real_home, "cfg-b")
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))

    def put_transcript(self, config_dir: str, folder: str, sid: str = SID, text: str | None = None) -> str:
        path = os.path.join(config_dir, "projects", folder, sid + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text if text is not None else "".join(sample_lines()))
        return path

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def log_text(self) -> str:
        try:
            with open(self.logger.path, encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return ""


class TranscriptPathTests(HistoryTestCase):
    def test_path_found_by_encoded_cwd(self) -> None:
        """Ported labels:
        1d: the exact encoded-cwd folder is found
        """
        path = self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        self.assertEqual(transcript_path(self.cfg_a, self.cwd, SID), path)

    def test_path_found_by_fallback_scan_of_every_folder(self) -> None:
        """Ported labels:
        1e: with no folder for the cwd, any folder holding the uuid is found
        """
        path = self.put_transcript(self.cfg_a, "some-other-folder")
        self.assertEqual(transcript_path(self.cfg_a, self.cwd, SID), path)

    def test_cwd_folder_wins_over_the_scan(self) -> None:
        self.put_transcript(self.cfg_a, "aaa-other")
        own = self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        self.assertEqual(transcript_path(self.cfg_a, self.cwd, SID), own)

    def test_bad_session_id_returns_none(self) -> None:
        """Ported labels:
        1g: a uuid with path parts is refused
        """
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd), sid="bad")
        for sid in ("", None, "../" + SID, "a b", "x" * 65, SID + "/..", "uué"):
            with self.subTest(sid=sid):
                self.assertIsNone(transcript_path(self.cfg_a, self.cwd, sid))

    def test_missing_transcript_or_folder_returns_none(self) -> None:
        """Ported labels:
        1f: a uuid that is nowhere gives None
        """
        self.assertIsNone(transcript_path(self.cfg_a, self.cwd, SID))
        self.assertIsNone(transcript_path("", self.cwd, SID))
        self.assertIsNone(transcript_path(self.cfg_a, "", SID))

    def test_encoded_cwd_replaces_every_non_alphanumeric_character(self) -> None:
        """Ported labels:
        1a: every non [A-Za-z0-9] char becomes -
        1c: non-ASCII letter is replaced
        """
        self.assertEqual(encoded_cwd("/home/u/my.proj_1"), "-home-u-my-proj-1")
        self.assertEqual(encoded_cwd("/a b/é"), "-a-b--")
        self.assertEqual(encoded_cwd("/x/Été"), "-x--t-")

    def test_dashes_and_digits_are_kept_as_they_are(self) -> None:
        """Ported labels:
        1b: dashes and digits kept as-is
        """
        self.assertEqual(encoded_cwd("/home/u/my-app/v2"), "-home-u-my-app-v2")


class StripThinkingTests(unittest.TestCase):
    def test_thinking_and_redacted_thinking_are_removed_text_kept(self) -> None:
        """Ported labels:
        2a: thinking dropped, text kept, newline kept
        """
        line = json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x"}, {"type": "text", "text": "keep"},
            {"type": "redacted_thinking", "data": "y"}]}}) + "\n"
        out = json.loads(strip_thinking_line(line))
        self.assertEqual(out["message"]["content"], [{"type": "text", "text": "keep"}])
        self.assertTrue(strip_thinking_line(line).endswith("\n"))

    def test_message_left_empty_gets_the_placeholder_text(self) -> None:
        """Ported labels:
        2b: an emptied message gets a text placeholder
        """
        line = json.dumps({"message": {"content": [{"type": "thinking", "thinking": "x"}]}})
        out = json.loads(strip_thinking_line(line))
        self.assertEqual(out["message"]["content"], [{"type": "text", "text": "(…)"}])

    def test_non_dict_block_is_kept_and_thinking_block_dropped(self) -> None:
        """Ported labels:
        2g: a non-dict block is kept and a thinking block dropped
        """
        line = '{"message": {"content": [{"type": "thinking"}, 5]}}\n'
        self.assertEqual(strip_thinking_line(line), '{"message": {"content": [5]}}\n')

    def test_line_without_newline_stays_without_newline(self) -> None:
        line = json.dumps({"message": {"content": [{"type": "thinking", "thinking": "x"}]}})
        self.assertFalse(strip_thinking_line(line).endswith("\n"))

    def test_lines_with_nothing_to_remove_come_back_unchanged(self) -> None:
        """Ported labels:
        2c: a non-JSON line is unchanged
        2d: a line with nothing to remove is byte-identical
        2e: a string content is left alone
        2f: a JSON value that is not an object is unchanged
        """
        for line in ("not json\n", json.dumps({"message": {"content": "plain"}}) + "\n",
                     json.dumps({"message": {"content": [{"type": "text", "text": "a"}]}}, indent=2) + "\n",
                     json.dumps([1, 2]) + "\n"):
            with self.subTest(line=line):
                self.assertEqual(strip_thinking_line(line), line)


class CopyTranscriptTests(HistoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.src = os.path.join(self.real_home, "src.jsonl")
        with open(self.src, "w", encoding="utf-8", newline="") as handle:
            handle.write("".join(sample_lines()))
        self.dst = os.path.join(self.real_home, "out", "projects", "f", SID + ".jsonl")

    def test_copy_strips_thinking_and_creates_the_folder(self) -> None:
        """Ported labels:
        3c: same number of lines
        3f: no thinking block survives in the copy
        3g: the text block of that message is kept
        """
        self.assertTrue(copy_transcript(self.src, self.dst))
        text = self.read_bytes(self.dst).decode("utf-8")
        self.assertNotIn("private", text)
        self.assertNotIn("opaque", text)
        self.assertIn("hello", text)
        self.assertIn("ünïcode ✓", text)
        self.assertEqual(text.count("\n"), 5)

    def test_copy_keeps_other_lines_and_leaves_the_source_alone(self) -> None:
        """Ported labels:
        3d: the non-JSON line is copied unchanged
        3e: lines without thinking are unchanged
        3h: the emptied message got a placeholder
        3l: the source transcript is untouched
        """
        before = self.read_bytes(self.src)
        self.assertTrue(copy_transcript(self.src, self.dst))
        out = self.read_bytes(self.dst).decode("utf-8").splitlines(keepends=True)
        src_lines = sample_lines()
        self.assertEqual(len(out), len(src_lines))
        self.assertEqual(out[3], src_lines[3])
        self.assertEqual(out[0], src_lines[0])
        self.assertEqual(out[4], src_lines[4])
        self.assertEqual(json.loads(out[5])["message"]["content"], [{"type": "text", "text": "(…)"}])
        self.assertEqual(self.read_bytes(self.src), before)

    def test_shorter_existing_destination_is_replaced(self) -> None:
        """Ported labels:
        7d: a shorter existing destination is replaced by the full copy
        7e: no temp file after a replace
        """
        os.makedirs(os.path.dirname(self.dst))
        with open(self.dst, "w", encoding="utf-8") as handle:
            handle.write("short\n")
        self.assertTrue(copy_transcript(self.src, self.dst))
        self.assertNotEqual(self.read_bytes(self.dst), b"short\n")
        self.assertIn(b"hello", self.read_bytes(self.dst))
        self.assertEqual([n for n in os.listdir(os.path.dirname(self.dst)) if n.endswith(".tmp")], [])

    def test_longer_existing_destination_is_kept(self) -> None:
        """Ported labels:
        7a: a longer existing destination is kept as it is
        7b: a kept longer copy still counts as carried over
        7c: no temp file is left behind (kept case)
        """
        os.makedirs(os.path.dirname(self.dst))
        longer = b"x" * 10_000
        with open(self.dst, "wb") as handle:
            handle.write(longer)
        self.assertTrue(copy_transcript(self.src, self.dst))
        self.assertEqual(self.read_bytes(self.dst), longer)
        self.assertEqual([n for n in os.listdir(os.path.dirname(self.dst)) if n.endswith(".tmp")], [])

    def test_failing_write_leaves_no_tmp_and_raises(self) -> None:
        with mock.patch("tg_bridge.agents.claude.history.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                copy_transcript(self.src, self.dst)
        leftovers = [name for name in os.listdir(os.path.dirname(self.dst)) if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertFalse(os.path.exists(self.dst))

    def test_missing_source_raises_oserror_and_leaves_no_tmp(self) -> None:
        with self.assertRaises(OSError):
            copy_transcript(os.path.join(self.real_home, "nope.jsonl"), self.dst)
        self.assertFalse(os.path.exists(self.dst + "." + str(os.getpid()) + ".tmp"))


class CarryHistoryTests(HistoryTestCase):
    def test_copies_into_the_destination_config_folder_under_the_cwd(self) -> None:
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        self.assertTrue(carry_history(self.cfg_a, self.cfg_b, self.cwd, SID, None, self.logger))
        dst = os.path.join(self.cfg_b, "projects", encoded_cwd(self.cwd), SID + ".jsonl")
        self.assertIn(b"hello", self.read_bytes(dst))
        self.assertNotIn(b"private", self.read_bytes(dst))

    def test_dst_cwd_chooses_the_destination_folder(self) -> None:
        other = os.path.join(self.real_home, "elsewhere")
        os.makedirs(other)
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        self.assertTrue(carry_history(self.cfg_a, self.cfg_b, self.cwd, SID, other, self.logger))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg_b, "projects", encoded_cwd(other), SID + ".jsonl")))

    def test_missing_source_returns_false_quietly(self) -> None:
        self.assertFalse(carry_history(self.cfg_a, self.cfg_b, self.cwd, SID, None, self.logger))
        self.assertFalse(os.path.exists(os.path.join(self.cfg_b, "projects")))

    def test_bad_session_id_returns_false(self) -> None:
        """Ported labels:
        10a: a bad uuid gives False, no raise
        """
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        self.assertFalse(carry_history(self.cfg_a, self.cfg_b, self.cwd, "../" + SID, None, self.logger))

    def test_destination_that_cannot_hold_the_folder_returns_false(self) -> None:
        """Ported labels:
        10b: a folder that cannot be created gives False, no raise
        """
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        os.makedirs(self.cfg_b)
        with open(os.path.join(self.cfg_b, "projects"), "w", encoding="utf-8") as handle:
            handle.write("x")  # a file where the projects folder must go
        self.assertFalse(carry_history(self.cfg_a, self.cfg_b, self.cwd, SID, None, self.logger))

    def test_copy_failure_is_logged_and_not_raised(self) -> None:
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        with mock.patch("tg_bridge.agents.claude.history.copy_transcript", side_effect=OSError("boom")):
            self.assertFalse(carry_history(self.cfg_a, self.cfg_b, self.cwd, SID, None, self.logger))
        self.assertIn("history copy for " + SID + " failed: boom", self.log_text())


class ClaudeBackendCarryTests(HistoryTestCase):
    def spec(self, name: str, config_dir: str) -> LauncherSpec:
        return LauncherSpec(name=name, backend="claude", command=name, argv=(name,), config_dir=config_dir)

    def test_backend_uses_each_launchers_config_dir(self) -> None:
        self.put_transcript(self.cfg_a, encoded_cwd(self.cwd))
        backend = ClaudeBackend(logger=self.logger, clock=FakeClock())
        ok = backend.carry_history(self.spec("a", self.cfg_a), self.spec("b", self.cfg_b), self.cwd, SID)
        self.assertTrue(ok)
        self.assertTrue(os.path.isfile(os.path.join(self.cfg_b, "projects", encoded_cwd(self.cwd), SID + ".jsonl")))

    def test_backend_missing_source_is_false(self) -> None:
        backend = ClaudeBackend(logger=self.logger, clock=FakeClock())
        self.assertFalse(backend.carry_history(self.spec("a", self.cfg_a), self.spec("b", self.cfg_b),
                                               self.cwd, SID, dst_cwd=self.cwd))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
