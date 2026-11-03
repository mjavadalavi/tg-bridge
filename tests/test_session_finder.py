"""session_finder.find_relocated on temp dirs: the newest transcript written after the paste whose tail holds the echo of the
pasted text. The file is excluded when it is the one already followed, and only the newest MAX_FILES files and the tail of
each are read."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from tg_bridge.agents.claude.session_finder import MAX_FILES, TAIL_BYTES, find_in_session, find_relocated

OLD = "11111111-2222-3333-4444-555555555555"
NEW = "66666666-7777-8888-9999-000000000000"
TEXT = "please summarise the ledger for March"


def echo(text: str) -> dict:
    return {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": text}}


def answer(text: str) -> dict:
    return {"type": "assistant", "message": {"id": "m", "role": "assistant", "content": [{"type": "text", "text": text}]}}


def stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FinderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_dir = os.path.join(self.tmp.name, "config")
        self.folder = os.path.join(self.config_dir, "projects", "-work")
        os.makedirs(self.folder)
        self.clock = 1_000_000.0  # the paste time: files get mtimes around it

    def transcript(self, sid: str, lines: list[dict | str], mtime_offset: float = 1.0, folder: str | None = None) -> str:
        path = os.path.join(folder or self.folder, sid + ".jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
        os.utime(path, (self.clock + mtime_offset, self.clock + mtime_offset))
        return path

    def find(self, text: str = TEXT, exclude: str = ""):
        return find_relocated(self.config_dir, self.clock, text, exclude=exclude)


class FindRelocatedTests(FinderCase):
    def test_the_echo_in_a_newer_transcript_is_found_with_its_stem_and_line_offset(self) -> None:
        path = self.transcript(NEW, [answer("earlier"), echo(TEXT), answer("ANSWER")])
        found = self.find()
        self.assertIsNotNone(found)
        self.assertEqual((found.path, found.session_id), (path, NEW))
        with open(path, "rb") as handle:
            handle.seek(found.offset)
            self.assertEqual(json.loads(handle.readline())["message"]["content"], TEXT, "offset is not the echo line")

    def test_a_transcript_not_modified_after_the_paste_is_not_looked_at(self) -> None:
        self.transcript(NEW, [echo(TEXT)], mtime_offset=-5.0)
        self.assertIsNone(self.find())

    def test_the_excluded_transcript_is_skipped_even_when_it_holds_the_echo(self) -> None:
        old = self.transcript(OLD, [echo(TEXT)])
        self.assertIsNone(self.find(exclude=old))

    def test_a_transcript_without_the_echo_is_not_a_match(self) -> None:
        self.transcript(NEW, [answer(TEXT), {"type": "user", "isMeta": True, "message": {"content": TEXT}}])
        self.assertIsNone(self.find(), "an assistant line or a meta line counts as the echo")

    def test_a_task_notification_with_the_text_is_not_an_echo(self) -> None:
        self.transcript(NEW, [{"type": "user", "turnOrigin": "task_notification", "message": {"content": TEXT}}])
        self.assertIsNone(self.find())

    def test_only_the_first_sixty_characters_of_the_text_are_matched(self) -> None:
        head = "x" * 50 + " " + "y" * 9  # sixty characters
        self.transcript(NEW, [echo(head + " ZZZ, the transcript cut the rest")])
        self.assertIsNotNone(self.find(head + " " + "w" * 40), "a 60 character prefix should be enough")
        self.assertIsNone(self.find("q" * 60), "a different start must not match")

    def test_the_stripped_text_is_matched_and_blank_text_finds_nothing(self) -> None:
        self.transcript(NEW, [echo(TEXT)])
        self.assertIsNotNone(self.find("   " + TEXT + "\n"))
        self.assertIsNone(self.find("   \n"))

    def test_a_slash_command_with_arguments_is_found_by_its_stored_command_tags(self) -> None:
        stored = ("<command-name>/review-pr</command-name>\n<command-message>review-pr</command-message>\n"
                  "<command-args>12</command-args>")
        path = self.transcript(NEW, [answer("earlier"), echo(stored)])
        found = self.find("/review-pr 12")
        self.assertIsNotNone(found)
        self.assertEqual((found.path, found.session_id), (path, NEW))
        self.assertIsNone(self.find("/review-pr 13"), "another argument is another command")

    def test_the_newest_matching_transcript_wins(self) -> None:
        self.transcript(OLD, [echo(TEXT)], mtime_offset=1.0)
        self.transcript(NEW, [echo(TEXT)], mtime_offset=2.0)
        self.assertEqual(self.find().session_id, NEW)

    def test_transcripts_of_other_folders_are_searched_too(self) -> None:
        other = os.path.join(self.config_dir, "projects", "-other-place")
        os.makedirs(other)
        self.transcript(NEW, [echo(TEXT)], folder=other)
        self.assertEqual(self.find().session_id, NEW)

    def test_a_missing_config_folder_or_projects_folder_finds_nothing(self) -> None:
        self.assertIsNone(find_relocated(os.path.join(self.tmp.name, "absent"), self.clock, TEXT))
        self.assertIsNone(find_relocated("", self.clock, TEXT))

    def test_only_the_newest_max_files_transcripts_are_read(self) -> None:
        self.transcript(OLD, [echo(TEXT)], mtime_offset=1.0)  # the oldest of the MAX_FILES + 1 candidates
        for index in range(MAX_FILES):
            self.transcript(f"aaaaaaaa-0000-0000-0000-{index:012d}", [answer("noise")], mtime_offset=10.0 + index)
        self.assertIsNone(self.find(), "a transcript beyond the newest MAX_FILES was read")

    def test_an_echo_older_than_the_tail_is_not_read(self) -> None:
        filler = answer("x" * 1000)
        self.transcript(NEW, [echo(TEXT)] + [filler] * (TAIL_BYTES // 500))  # the echo is further back than the tail
        self.assertIsNone(self.find())

    def test_an_enqueue_line_with_the_text_is_found_as_the_echo(self) -> None:
        queued = {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-11-02T10:00:00Z",
            "sessionId": NEW,
            "content": TEXT,
        }
        path = self.transcript(NEW, [answer("earlier"), queued, answer("ANSWER")])
        found = self.find()
        self.assertIsNotNone(found, "the CLI's enqueue line holds the pasted text")
        self.assertEqual((found.path, found.session_id), (path, NEW))
        with open(path, "rb") as handle:
            handle.seek(found.offset)
            self.assertEqual(json.loads(handle.readline())["content"], TEXT, "offset is not the enqueue line")

    def test_a_queued_command_attachment_with_the_text_is_found_as_the_echo(self) -> None:
        attachment = {"type": "attachment", "attachment": {"type": "queued_command", "prompt": TEXT}}
        path = self.transcript(NEW, [answer("earlier"), attachment, answer("ANSWER")])
        found = self.find()
        self.assertIsNotNone(found, "the attachment of a message delivered mid-turn holds the pasted text")
        self.assertEqual((found.path, found.session_id), (path, NEW))
        with open(path, "rb") as handle:
            handle.seek(found.offset)
            line = json.loads(handle.readline())
            self.assertEqual(line["attachment"]["prompt"], TEXT, "offset is not the attachment")

    def test_a_queued_command_attachment_with_another_prompt_is_not_a_match(self) -> None:
        other = {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "a different question"}}
        self.transcript(NEW, [other])
        self.assertIsNone(self.find())

    def test_an_enqueue_line_without_content_or_a_dequeue_is_not_a_match(self) -> None:
        enqueue_without_content = {"type": "queue-operation", "operation": "enqueue", "sessionId": NEW}
        dequeue_with_text = {"type": "queue-operation", "operation": "dequeue", "sessionId": NEW, "content": TEXT}
        self.transcript(NEW, [enqueue_without_content, dequeue_with_text])
        self.assertIsNone(self.find(), "a queue line that is not an enqueue with the text counts as the echo")

    def test_a_partial_last_line_is_not_matched_until_its_newline_is_written(self) -> None:
        path = os.path.join(self.folder, NEW + ".jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(echo(TEXT)))  # no newline yet
        os.utime(path, (self.clock + 1, self.clock + 1))
        self.assertIsNone(self.find())

    def test_an_echo_line_older_than_the_paste_is_ignored(self) -> None:
        old_line = dict(echo(TEXT), timestamp=stamp(self.clock - 3600))  # an hour before the paste, in a live file
        self.transcript(NEW, [old_line])
        self.assertIsNone(self.find(), "a short message matched an old line of a recently modified transcript")

    def test_an_echo_line_after_the_paste_is_found(self) -> None:
        path = self.transcript(NEW, [dict(echo(TEXT), timestamp=stamp(self.clock + 60))])
        found = self.find()
        self.assertIsNotNone(found)
        self.assertEqual((found.path, found.session_id, found.offset), (path, NEW, 0))

    def test_an_echo_line_without_a_timestamp_is_still_found(self) -> None:
        self.transcript(NEW, [echo(TEXT)])
        self.assertEqual(self.find().session_id, NEW)

    def test_a_malformed_line_is_skipped(self) -> None:
        self.transcript(NEW, ["{not json", echo(TEXT)])
        self.assertEqual(self.find().session_id, NEW)

    def test_the_scan_is_read_only(self) -> None:
        path = self.transcript(NEW, [echo(TEXT)])
        with open(path, "rb") as handle:
            before = handle.read()
        self.find()
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(sorted(os.listdir(self.folder)), [NEW + ".jsonl"])


class FindInSessionTests(FinderCase):
    def find_session(self, session_id: str, text: str = TEXT):
        return find_in_session(self.config_dir, session_id, self.clock, text)

    def test_find_in_session_reads_only_that_session(self) -> None:
        other = os.path.join(self.config_dir, "projects", "-other-place")
        os.makedirs(other)
        self.transcript(OLD, [echo(TEXT)], folder=self.folder)
        new_path = self.transcript(NEW, [echo(TEXT)], folder=other)
        found = self.find_session(NEW)
        self.assertIsNotNone(found)
        self.assertEqual((found.path, found.session_id), (new_path, NEW), "another session's transcript was read")
        self.assertEqual(self.find_session(OLD).session_id, OLD)

    def test_find_in_session_reads_the_newest_copy(self) -> None:
        other = os.path.join(self.config_dir, "projects", "-other-place")  # sorts before "-work"
        os.makedirs(other)
        self.transcript(NEW, [answer("earlier")], mtime_offset=1.0, folder=other)  # older copy, no echo
        new_path = self.transcript(NEW, [echo(TEXT)], mtime_offset=2.0, folder=self.folder)  # the copy the tab writes
        found = self.find_session(NEW)
        self.assertIsNotNone(found, "the older copy that sorts first hid the newer copy")
        self.assertEqual((found.path, found.session_id), (new_path, NEW))

    def test_find_in_session_is_none_without_the_file_or_the_echo(self) -> None:
        self.assertIsNone(self.find_session(NEW), "no transcript of the session exists")
        self.assertIsNone(find_in_session(os.path.join(self.tmp.name, "absent"), NEW, self.clock, TEXT))
        self.transcript(NEW, [answer(TEXT)])
        self.assertIsNone(self.find_session(NEW), "an assistant line counts as the echo")

    def test_find_in_session_rejects_a_path_like_id(self) -> None:
        self.transcript(NEW, [echo(TEXT)])
        for session_id in ["", "../-work/" + NEW, "..", NEW + "/..", "-work/" + NEW, "a\\" + NEW]:
            self.assertIsNone(self.find_session(session_id), f"path-like session id {session_id!r} was read")

    def test_find_in_session_applies_the_timestamp_rule(self) -> None:
        old_line = dict(echo(TEXT), timestamp=stamp(self.clock - 3600))  # an hour before the paste
        self.transcript(NEW, [old_line])
        self.assertIsNone(self.find_session(NEW), "an echo line older than the paste was matched")


if __name__ == "__main__":
    unittest.main()
