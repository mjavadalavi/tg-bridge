"""ClaudeTranslator (agents/claude/translate.py) on recorded stream-json lines, and the wiring into ClaudeSession.

Fixtures live in tests/fixtures/claude_lines/. They are built from the verified shapes in PROTOCOL.md, and
turn_feed_fake_run.jsonl is the real stdout of tests/fakes/fake_claude.py (FAKE_FEED=1 + permission ask).

Scenarios: session/init and noise lines; api_retry with attempt and max; compacting; subagent text never the answer
and subagent tools carry parent_id; tool results and their error detail; result usage and cost_known; error results;
resume failure and resume_reason replays; can_use_tool (command, file, AskUserQuestion); cancel; unknown objects.
"""

import json
import os
import sys
import unittest

import tests.fakes as fakes_package
from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.claude.session import ClaudeSession
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import (ApprovalRequest, ApprovalWithdrawn, Decision, Question, SessionStarted,
                                     StatusEvent, TextEvent, ToolEnd, ToolStart, TurnResult, TurnStatus, UsageEvent)
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_lines")
FAKE = os.path.join(os.path.dirname(os.path.abspath(fakes_package.__file__)), "fake_claude.py")
CWD = "/home/x/proj"
ALL_DECISIONS = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})


def lines(name: str) -> list[dict]:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def translate_all(objects: list[dict], cwd: str | None = CWD) -> list:
    translator = ClaudeTranslator(cwd=cwd)
    out: list = []
    for obj in objects:
        out.extend(translator.translate(obj))
    return out


def by_type(events: list, kind: type) -> list:
    return [e for e in events if isinstance(e, kind)]


class SessionStartTests(unittest.TestCase):
    def test_init_gives_session_started_with_commands_and_mode(self) -> None:
        events = translate_all(lines("session_start.jsonl"))
        self.assertEqual(by_type(events, SessionStarted),
                         [SessionStarted(session_id="sess-abc", model="claude-sonnet-x", mode="acceptEdits",
                                         cwd="/home/x/proj", commands=("compact", "review"))])

    def test_hooks_title_and_thinking_tokens_are_ignored(self) -> None:
        objects = [o for o in lines("session_start.jsonl") if o.get("subtype") != "init"]
        self.assertEqual(translate_all(objects), [])

    def test_init_without_session_id_gives_nothing(self) -> None:
        self.assertEqual(translate_all([{"type": "system", "subtype": "init", "slash_commands": ["x"]}]), [])

    def test_init_with_odd_commands_keeps_only_strings(self) -> None:
        obj = {"type": "system", "subtype": "init", "session_id": "s", "slash_commands": ["a", 3, None]}
        self.assertEqual(translate_all([obj]), [SessionStarted(session_id="s", commands=("a",))])
        obj["slash_commands"] = "not a list"
        self.assertEqual(translate_all([obj]), [SessionStarted(session_id="s")])

    def test_init_takes_the_name_of_object_entries_as_live_does(self) -> None:
        obj = {"type": "system", "subtype": "init", "session_id": "s",
               "slash_commands": ["a", {"name": "compact", "description": "x"}, {"name": ""}, {}, None]}
        self.assertEqual(translate_all([obj]), [SessionStarted(session_id="s", commands=("a", "compact"))])


class StatusLineTests(unittest.TestCase):
    def test_api_retry_gives_retry_with_attempt_and_max_attempts(self) -> None:
        retry = translate_all(lines("status_lines.jsonl")[1:2])
        self.assertEqual(retry, [StatusEvent(kind="retry", detail="rate_limit", retry_in=8.0, attempt=2,
                                             max_attempts=10)])

    def test_api_retry_without_counts_leaves_them_none(self) -> None:
        retry = translate_all(lines("status_lines.jsonl")[2:3])
        self.assertEqual(retry, [StatusEvent(kind="retry", detail="overloaded")])
        self.assertIsNone(retry[0].attempt)
        self.assertIsNone(retry[0].max_attempts)
        self.assertIsNone(retry[0].retry_in)

    def test_compacting_gives_compacting(self) -> None:
        self.assertEqual(translate_all(lines("status_lines.jsonl")[0:1]), [StatusEvent(kind="compacting")])

    def test_other_status_values_give_nothing(self) -> None:
        self.assertEqual(translate_all(lines("status_lines.jsonl")[3:4]), [])


class TurnFeedTests(unittest.TestCase):
    """The real stdout of fake_claude with a Read error, an api_retry, a Bash with a subagent Glob and text."""

    def setUp(self) -> None:
        self.events = translate_all(lines("turn_feed_fake_run.jsonl"))

    def test_whole_turn_in_order(self) -> None:
        self.assertEqual([type(e).__name__ for e in self.events], [
            "SessionStarted", "TextEvent", "ToolStart", "ToolEnd", "StatusEvent", "ToolStart", "ToolStart",
            "TextEvent", "TextEvent", "ApprovalRequest", "UsageEvent", "TurnResult"])

    def test_text_and_tool_start_carry_labels_and_parents(self) -> None:
        texts = by_type(self.events, TextEvent)
        self.assertEqual([(t.text, t.parent_id) for t in texts], [
            ("Let me check.", None), ("SUBAGENT-SECRET", "tu_2"), ("ok:allow", None)])
        starts = by_type(self.events, ToolStart)
        self.assertEqual([(s.tool_id, s.name, s.label, s.parent_id) for s in starts], [
            ("tu_1", "Read", "Read x.txt", None),
            ("tu_2", "Bash", "Bash List files", None),
            ("tu_3", "Glob", "Glob **/*.py", "tu_2"),
        ])
        self.assertEqual(starts[1].input, {"command": "ls\n-la", "description": "List files"})

    def test_failed_tool_result_gives_first_line_as_detail(self) -> None:
        self.assertEqual(by_type(self.events, ToolEnd), [ToolEnd(tool_id="tu_1", ok=False,
                                                                 detail="File does not exist.")])

    def test_retry_in_the_turn_is_counted(self) -> None:
        self.assertEqual(by_type(self.events, StatusEvent), [
            StatusEvent(kind="retry", detail="rate_limit", retry_in=8.0, attempt=1, max_attempts=10)])

    def test_subagent_text_is_never_final(self) -> None:
        self.assertTrue(all(not t.final for t in by_type(self.events, TextEvent)))

    def test_permission_ask_of_the_turn(self) -> None:
        (req,) = by_type(self.events, ApprovalRequest)
        self.assertEqual(req.approval_id, "perm_1")
        self.assertEqual((req.kind, req.tool, req.summary), ("command", "Bash", "Bash"))
        self.assertEqual(req.input, {"command": "echo hi", "description": "Say hi"})

    def test_result_without_usage_gives_empty_usage_and_ok_turn(self) -> None:
        self.assertEqual(by_type(self.events, UsageEvent), [UsageEvent()])
        self.assertEqual(by_type(self.events, TurnResult), [TurnResult(
            status=TurnStatus.OK, answer="ok:allow", subtype="success")])


class ResultTests(unittest.TestCase):
    def test_ok_result_gives_usage_as_the_footer_counts_it(self) -> None:
        events = translate_all(lines("result_ok.jsonl"))
        self.assertEqual(by_type(events, UsageEvent), [UsageEvent(
            input_tokens=10, output_tokens=55, cached_tokens=200 + 3000, cost_usd=0.0123,
            cost_known=True, context_window=200000)])
        self.assertEqual(by_type(events, TurnResult), [TurnResult(
            status=TurnStatus.OK, answer="Done.", duration_ms=4120, num_turns=2, subtype="success")])

    def test_thinking_block_is_not_text(self) -> None:
        texts = by_type(translate_all(lines("result_ok.jsonl")), TextEvent)
        self.assertEqual(texts, [TextEvent(text="Done.")])

    def test_cache_reads_stay_out_of_the_fresh_input(self) -> None:
        (usage,) = by_type(translate_all(lines("result_cache_no_total.jsonl")), UsageEvent)
        self.assertEqual((usage.input_tokens, usage.cached_tokens), (12, 80000))

    def test_missing_total_cost_with_no_unknown_model_is_cost_known(self) -> None:
        (usage,) = by_type(translate_all(lines("result_cache_no_total.jsonl")), UsageEvent)
        self.assertEqual((usage.cost_usd, usage.cost_known), (None, True))

    def test_unknown_model_cost_gives_cost_not_known(self) -> None:
        (usage,) = by_type(translate_all(lines("result_unknown_cost.jsonl")), UsageEvent)
        self.assertEqual((usage.cost_usd, usage.cost_known, usage.context_window), (0.5, False, None))

    def test_error_result_joins_errors_and_keeps_subtype(self) -> None:
        (turn,) = by_type(translate_all(lines("result_error.jsonl")), TurnResult)
        self.assertEqual(turn, TurnResult(status=TurnStatus.ERROR, answer=None, duration_ms=90, num_turns=1,
                                          subtype="error_during_execution", error="boom one; boom two"))

    def test_no_conversation_is_flagged_as_resume_failed(self) -> None:
        (turn,) = by_type(translate_all(lines("result_resume_failed.jsonl")), TurnResult)
        self.assertTrue(turn.resume_failed)
        self.assertEqual(turn.status, TurnStatus.ERROR)

    def test_no_conversation_in_errors_is_flagged_too(self) -> None:
        obj = {"type": "result", "subtype": "error_during_execution", "is_error": True,
               "errors": ["No conversation found with session ID: z"]}
        (turn,) = by_type(translate_all([obj]), TurnResult)
        self.assertTrue(turn.resume_failed)

    def test_replayed_result_with_resume_reason_is_dropped(self) -> None:
        self.assertEqual(translate_all(lines("result_resume_reason.jsonl")), [])

    def test_success_without_text_has_no_answer(self) -> None:
        (turn,) = by_type(translate_all([{"type": "result", "subtype": "success", "result": ""}]), TurnResult)
        self.assertIsNone(turn.answer)
        self.assertEqual(turn.status, TurnStatus.OK)

    def test_long_error_text_is_cut(self) -> None:
        obj = {"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "x" * 5000}
        (turn,) = by_type(translate_all([obj]), TurnResult)
        self.assertEqual(len(turn.error), 1500)


class ControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.objects = lines("permission_ask.jsonl")

    def test_bash_ask_keeps_input_and_offers_every_choice(self) -> None:
        (req,) = translate_all(self.objects[0:1])
        self.assertIsInstance(req, ApprovalRequest)
        self.assertEqual((req.approval_id, req.kind, req.tool, req.summary), ("req-1", "command", "Bash", "Bash"))
        self.assertEqual(req.input, {"command": "touch probe_ok.txt", "description": "touch probe_ok.txt"})
        self.assertEqual(req.choices, ALL_DECISIONS)
        self.assertEqual(req.questions, ())
        self.assertFalse(req.deny_aborts_turn)
        self.assertIsNone(req.always_hint)
        self.assertIs(req.raw, self.objects[0])

    def test_ask_user_question_gives_questions_and_only_skip(self) -> None:
        (req,) = translate_all(self.objects[1:2])
        self.assertEqual((req.kind, req.tool), ("question", "AskUserQuestion"))
        self.assertEqual(req.choices, frozenset({Decision.DENY}))
        self.assertEqual(req.questions, (Question(text="Which database?", options=("Postgres", "SQLite"),
                                                  multi=False),))

    def test_ask_user_question_without_questions_keeps_an_empty_tuple(self) -> None:
        obj = {"type": "control_request", "request_id": "q1",
               "request": {"subtype": "can_use_tool", "tool_name": "AskUserQuestion", "input": {}}}
        (req,) = translate_all([obj])
        self.assertEqual(req.questions, ())

    def test_edit_ask_is_a_file_request(self) -> None:
        (req,) = translate_all(self.objects[2:3])
        self.assertEqual((req.kind, req.tool), ("file", "Edit"))

    def test_other_control_subtypes_give_nothing_so_the_session_answers_unhandled(self) -> None:
        """Ported labels:
        I: unsupported subtype -> error control_response
        """
        self.assertEqual(translate_all(self.objects[3:4]), [])

    def test_cancel_withdraws_the_ask(self) -> None:
        self.assertEqual(translate_all(self.objects[4:5]), [ApprovalWithdrawn(approval_id="req-1")])

    def test_control_response_is_not_translated(self) -> None:
        self.assertEqual(translate_all(self.objects[5:6]), [])

    def test_ask_without_request_id_gives_nothing(self) -> None:
        obj = {"type": "control_request", "request": {"subtype": "can_use_tool", "tool_name": "Bash"}}
        self.assertEqual(translate_all([obj]), [])


class RobustnessTests(unittest.TestCase):
    def test_unknown_types_and_broken_shapes_give_nothing(self) -> None:
        objects = [
            {"type": "weird"}, {}, {"type": "assistant"}, {"type": "assistant", "message": None},
            {"type": "assistant", "message": {"content": "text"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "   "}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": 5, "name": "Read"}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": None}]}},
            {"type": "system", "subtype": "thinking_tokens", "tokens": 3},
            {"type": "control_request", "request_id": "r", "request": "not a dict"},
        ]
        self.assertEqual(translate_all(objects), [])

    def test_tool_use_without_input_gets_an_empty_dict(self) -> None:
        obj = {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": "Glob"}]}}
        (start,) = translate_all([obj])
        self.assertEqual((start.label, start.input), ("Glob", {}))

    def test_tool_result_with_list_content_gives_first_nonblank_line(self) -> None:
        obj = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t", "is_error": True,
                                                        "content": [{"type": "text", "text": "\n  \nfirst\nsecond"}]}]}}
        self.assertEqual(translate_all([obj]), [ToolEnd(tool_id="t", ok=False, detail="first")])

    def test_successful_tool_result_has_no_detail(self) -> None:
        obj = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t", "content": "fine"}]}}
        self.assertEqual(translate_all([obj]), [ToolEnd(tool_id="t", ok=True)])

    def test_labels_are_relative_to_the_cwd_given(self) -> None:
        obj = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "/home/x/proj/a.py"}}]}}
        self.assertEqual(translate_all([obj], cwd="/home/x/proj")[0].label, "Read a.py")
        self.assertEqual(translate_all([obj], cwd=None)[0].label, "Read /home/x/proj/a.py")


class SessionWiringTests(unittest.TestCase):
    """A ClaudeSession built without a translator uses ClaudeTranslator: real child, real events, real answer."""

    def test_turn_with_a_permission_ask_is_answered_and_finishes(self) -> None:
        with TempHome() as home:
            logger = Logger(os.path.join(home.state, "bridge.log"))
            launcher = LauncherSpec(name="fake", backend="claude", command="fake",
                                    argv=(sys.executable, "-I", FAKE), config_dir=home.home)
            spec = SessionSpec(
                thread_id=7, chat_id=-100, topic="side", launcher=launcher, cwd=home.home, mode="default",
                model=None, resume_id=None, new_id=None, allow_rules=("Read",), deny_tools=(),
                system_rules="RULES", env=dict(os.environ, FAKE_FEED="1"),
                errfile_dir=os.path.join(home.state, "turns"))
            sink = RecordingSink()
            session = ClaudeSession(spec, sink, logger=logger, clock=FakeClock())
            try:
                session.start()
                session.send_user("hello")
                self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
                (req,) = by_type(sink.events, ApprovalRequest)
                session.answer(req, Decision.ALLOW_ONCE)
                self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult)))
            finally:
                session.stop(grace=1)
        self.assertEqual(by_type(sink.events, SessionStarted)[0].session_id, "sess-123")
        self.assertEqual(session.session_id, "sess-123")
        self.assertIn(ToolEnd(tool_id="tu_1", ok=False, detail="File does not exist."), sink.events)
        # the first TurnResult is the answer; the one after it comes from stop()'s interrupt
        self.assertEqual(by_type(sink.events, TurnResult)[0], TurnResult(
            status=TurnStatus.OK, answer="ok:allow", subtype="success"))


if __name__ == "__main__":
    unittest.main()
