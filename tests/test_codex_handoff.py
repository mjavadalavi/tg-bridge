"""Codex rollout handoff reader (agents/codex/handoff.py) and CodexBackend.handoff_entries (DESIGN STEP 29).

Fixture: tests/fixtures/codex_rollout.jsonl, a scrubbed copy of a real probe rollout (63 lines, thread
01a16f09-bcb0-70b3-a1ff-fefc0cc5402c). Scenarios: only real user and assistant texts and tool labels come out, in file
order; developer messages, injected AGENTS.md / environment_context, image inputs, reasoning and tool output are absent;
unknown or malformed lines are ignored; a missing rollout, a malformed thread id or an empty codex home gives ().
"""

import json
import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.codex.backend import CodexBackend
from tg_bridge.agents.codex.handoff import BRIEF_CUT, find_rollout, read_entries
from tg_bridge.agents.codex.instructions import wrap_rules
from tg_bridge.agents.events import HandoffEntry
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "codex_rollout.jsonl")
THREAD = "01a16f09-bcb0-70b3-a1ff-fefc0cc5402c"
ROLLOUT_NAME = "rollout-2026-10-24T17-51-12-" + THREAD + ".jsonl"
DAY_DIR = os.path.join("sessions", "2026", "10", "24")

# The apply_patch call is code-mode JavaScript with no command field, so its label is the name plus the cut detail.
APPLY_PATCH_LABEL = r'exec text(await tools.apply_patch("*** Begin Patch\n*** Add File: fc_probe.txt\…'

# (role, text) in file order, as read_entries(FIXTURE) must return them.
FIXTURE_EXPECTED = [
    ("user", "Run this shell command and nothing else: sleep 30"),
    ("tool", "$ sleep 30"),
    ("user", "Run this shell command and nothing else: sleep 30"),
    ("tool", "$ sleep 30"),
    ("user", "Use apply_patch to create a new file named fc_probe.txt containing the single line hello. "
             "Then, in your final reply, name the color of the attached 1x1 image in one word."),
    ("assistant", "I’ll create the file with the requested line."),
    ("tool", APPLY_PATCH_LABEL),
    ("assistant", "Green"),
    ("user", "Run this shell command and nothing else: sleep 30"),
    ("tool", "$ sleep 30"),
    ("user", "Quote every probe marker code (PROBE-MARKER-...) that is in your developer instructions, "
             "comma separated. If there are none, reply NONE."),
    ("assistant", "NONE"),
]


def _pairs(entries) -> list[tuple[str, str]]:
    return [(entry.role, entry.text) for entry in entries]


def _line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _message(role: str, *texts: str, kind: str = "input_text") -> dict:
    return {"timestamp": "2026-10-24T10:00:00Z", "type": "response_item",
            "payload": {"type": "message", "role": role,
                        "content": [{"type": kind, "text": text} for text in texts]}}


def _call(payload: dict) -> dict:
    return {"timestamp": "2026-10-24T10:00:01Z", "type": "response_item", "payload": payload}


class ReadEntriesFixtureTests(unittest.TestCase):
    def test_fixture_yields_real_conversation_in_file_order(self) -> None:
        entries = read_entries(FIXTURE)
        self.assertIsInstance(entries, tuple)
        self.assertTrue(all(isinstance(entry, HandoffEntry) for entry in entries))
        self.assertEqual(_pairs(entries), FIXTURE_EXPECTED)

    def test_skipped_content_is_absent_from_the_output(self) -> None:
        blob = json.dumps(_pairs(read_entries(FIXTURE)), ensure_ascii=False)
        for forbidden in ("SCRUBBED developer", "# AGENTS.md instructions", "skills_instructions",
                          "turn_aborted", "aborted by user", "Script completed", "<image", "data:image",
                          "You are `/root`"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, blob)

    def test_missing_fixture_path_gives_empty_tuple(self) -> None:
        with TempHome() as home:
            self.assertEqual(read_entries(os.path.join(home.home, "nope.jsonl")), ())
        self.assertEqual(read_entries(None), ())
        self.assertEqual(read_entries(""), ())

    def test_directory_path_gives_empty_tuple(self) -> None:
        with TempHome() as home:
            self.assertEqual(read_entries(home.home), ())


class ReadEntriesSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.path = os.path.join(self.home.home, "rollout.jsonl")

    def write(self, *lines: str) -> tuple[HandoffEntry, ...]:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        return read_entries(self.path)

    def test_user_and_assistant_texts_are_kept_in_order(self) -> None:
        entries = self.write(_line(_message("user", "first")), _line(_message("assistant", "second", kind="output_text")),
                             _line(_message("user", "third")))
        self.assertEqual(_pairs(entries), [("user", "first"), ("assistant", "second"), ("user", "third")])

    def test_developer_messages_are_skipped(self) -> None:
        entries = self.write(_line(_message("developer", "rules for the model")), _line(_message("user", "hi")))
        self.assertEqual(_pairs(entries), [("user", "hi")])

    def test_injected_context_messages_are_skipped(self) -> None:
        entries = self.write(
            _line(_message("user", "# AGENTS.md instructions\n\n<INSTRUCTIONS>\nrules\n</INSTRUCTIONS>")),
            _line(_message("user", "<environment_context>\n  <cwd>/x</cwd>\n</environment_context>")),
            _line(_message("user", "real question")))
        self.assertEqual(_pairs(entries), [("user", "real question")])

    def test_image_markers_are_dropped_and_text_blocks_joined(self) -> None:
        entries = self.write(_line({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "look at this"},
            {"type": "input_text", "text": '<image name=[Image #1] path="/tmp/a.png">'},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "input_text", "text": "</image>"},
            {"type": "input_text", "text": "and tell me"},
        ]}}))
        self.assertEqual(_pairs(entries), [("user", "look at this\nand tell me")])

    def test_blank_and_non_text_messages_are_skipped(self) -> None:
        entries = self.write(_line(_message("user", "   ")),
                             _line({"type": "response_item", "payload": {"type": "message", "role": "user", "content": "plain"}}),
                             _line({"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": []}}))
        self.assertEqual(entries, ())

    def test_turn_rules_block_is_not_part_of_the_user_text(self) -> None:
        entries = self.write(_line({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": wrap_rules("RULES: never run rm -rf")},
            {"type": "input_text", "text": "what is in the repo?"},
        ]}}))
        self.assertEqual(_pairs(entries), [("user", "what is in the repo?")])

    def test_unknown_and_malformed_lines_are_ignored(self) -> None:
        entries = self.write(
            _line(_message("user", "before")),
            "{broken json\n",
            '"just a string"\n',
            "[1, 2, 3]\n",
            _line({"type": "event_msg", "payload": {"type": "item_completed"}}),
            _line({"type": "future_line_type", "payload": {"type": "message", "role": "user", "content": []}}),
            _line({"type": "response_item", "payload": "not a dict"}),
            _line({"type": "response_item", "payload": {"type": "reasoning", "summary": []}}),
            _line({"type": "response_item", "payload": {"type": "custom_tool_call_output", "output": "out"}}),
            _line({"type": "response_item", "payload": {"type": "function_call_output", "output": "out"}}),
            _line(_message("user", "after")))
        self.assertEqual(_pairs(entries), [("user", "before"), ("user", "after")])

    def test_invalid_utf8_bytes_are_replaced_not_raised(self) -> None:
        with open(self.path, "wb") as handle:
            handle.write(b'{"type": "response_item", "payload": {"type": "message", "role": "user", '
                         b'"content": [{"type": "input_text", "text": "ok \xff end"}]}}\n')
        entries = read_entries(self.path)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].text.startswith("ok "))

    def test_shell_calls_read_as_dollar_command(self) -> None:
        entries = self.write(
            _line(_call({"type": "function_call", "name": "shell",
                         "arguments": json.dumps({"command": ["bash", "-lc", "git status"]})})),
            _line(_call({"type": "local_shell_call", "status": "completed",
                         "action": {"type": "exec", "command": ["ls", "-la"]}})),
            _line(_call({"type": "custom_tool_call", "name": "exec",
                         "input": 'await tools.exec_command({cmd:"echo \\"hi\\"",yield_time_ms:1000});'})))
        self.assertEqual(_pairs(entries), [("tool", "$ bash -lc git status"), ("tool", "$ ls -la"),
                                           ("tool", '$ echo "hi"')])

    def test_other_tool_calls_read_as_name_and_short_detail(self) -> None:
        entries = self.write(
            _line(_call({"type": "function_call", "name": "update_plan", "arguments": '{"step": 1}'})),
            _line(_call({"type": "function_call", "name": "web_search"})),
            _line(_call({"type": "custom_tool_call", "name": "exec", "input": "text(await tools.apply_patch(\"x\"));"})))
        self.assertEqual(_pairs(entries), [("tool", 'update_plan {"step": 1}'), ("tool", "web_search"),
                                           ("tool", 'exec text(await tools.apply_patch("x"));')])

    def test_nameless_tool_call_is_skipped(self) -> None:
        self.assertEqual(self.write(_line(_call({"type": "function_call", "arguments": "{}"}))), ())

    def test_long_tool_detail_is_cut_to_one_short_line(self) -> None:
        long_text = "word\n" * 100
        entries = self.write(_line(_call({"type": "custom_tool_call", "name": "exec", "input": long_text})))
        self.assertEqual(len(entries), 1)
        self.assertLessEqual(len(entries[0].text), BRIEF_CUT)
        self.assertNotIn("\n", entries[0].text)
        self.assertTrue(entries[0].text.endswith("…"))


class FindRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.codex_home = os.path.join(os.path.realpath(self.home.home), ".codex")

    def put(self, name: str, day: str = DAY_DIR, body: str = "") -> str:
        folder = os.path.join(self.codex_home, day)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def test_finds_the_rollout_of_the_thread(self) -> None:
        path = self.put(ROLLOUT_NAME)
        self.put("rollout-2026-10-24T09-00-00-99999999-aaaa-bbbb-cccc-dddddddddddd.jsonl")
        self.assertEqual(find_rollout(self.codex_home, THREAD), path)

    def test_newest_match_wins_when_a_thread_has_several_rollouts(self) -> None:
        older = self.put(ROLLOUT_NAME, day=os.path.join("sessions", "2026", "10", "23"))
        newer = self.put(ROLLOUT_NAME)
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        self.assertEqual(find_rollout(self.codex_home, THREAD), newer)

    def test_missing_rollout_gives_none(self) -> None:
        self.assertIsNone(find_rollout(self.codex_home, THREAD))
        self.assertIsNone(find_rollout(os.path.join(self.codex_home, "absent"), THREAD))

    def test_malformed_or_empty_thread_id_gives_none(self) -> None:
        self.put(ROLLOUT_NAME)
        for bad in ("", None, "../" + THREAD, "*", THREAD + "/x", "a" * 65):
            with self.subTest(bad=bad):
                self.assertIsNone(find_rollout(self.codex_home, bad))

    def test_empty_codex_home_gives_none(self) -> None:
        self.assertIsNone(find_rollout("", THREAD))
        self.assertIsNone(find_rollout(None, THREAD))

    def test_codex_home_with_glob_characters_is_taken_literally(self) -> None:
        odd_home = os.path.join(self.home.home, "home[1]*")
        folder = os.path.join(odd_home, DAY_DIR)
        os.makedirs(folder)
        path = os.path.join(folder, ROLLOUT_NAME)
        open(path, "w", encoding="utf-8").close()
        self.assertEqual(find_rollout(odd_home, THREAD), path)
        self.assertIsNone(find_rollout(os.path.join(self.home.home, "home[1]x"), THREAD))

    def test_directory_named_like_a_rollout_is_not_a_match(self) -> None:
        os.makedirs(os.path.join(self.codex_home, DAY_DIR, ROLLOUT_NAME))
        self.assertIsNone(find_rollout(self.codex_home, THREAD))


class CodexBackendHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.real_home = os.path.realpath(self.home.home)
        self.codex_home = os.path.join(self.real_home, ".codex")
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.backend = CodexBackend(logger=self.logger, clock=FakeClock())
        self.spec = LauncherSpec(name="codex", backend="codex", command="codex", argv=("codex",),
                                 config_dir=self.codex_home)

    def put_fixture(self, thread: str = THREAD) -> str:
        folder = os.path.join(self.codex_home, DAY_DIR)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "rollout-2026-10-24T17-51-12-" + thread + ".jsonl")
        with open(FIXTURE, encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return path

    def test_reads_the_rollout_under_the_launcher_codex_home(self) -> None:
        self.put_fixture()
        self.assertEqual(_pairs(self.backend.handoff_entries(THREAD, "/any/cwd", self.spec)), FIXTURE_EXPECTED)

    def test_missing_rollout_gives_empty_tuple(self) -> None:
        self.assertEqual(self.backend.handoff_entries(THREAD, "/any/cwd", self.spec), ())

    def test_other_thread_id_gives_empty_tuple(self) -> None:
        self.put_fixture()
        self.assertEqual(self.backend.handoff_entries("00000000-0000-0000-0000-000000000000", "/c", self.spec), ())

    def test_malformed_thread_id_gives_empty_tuple(self) -> None:
        self.put_fixture()
        self.assertEqual(self.backend.handoff_entries("../" + THREAD, "/c", self.spec), ())

    def test_empty_config_dir_gives_empty_tuple(self) -> None:
        self.put_fixture()
        blank = LauncherSpec(name="codex", backend="codex", command="codex", argv=("codex",), config_dir="")
        self.assertEqual(self.backend.handoff_entries(THREAD, "/c", blank), ())


if __name__ == "__main__":
    unittest.main()
