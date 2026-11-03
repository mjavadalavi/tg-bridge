"""TuiTurn, the per-turn bookkeeping of the tui transport (tg_bridge/agents/claude/tui_turn.py), on its own, without a PTY.
Scenarios: a message written as several lines counts its usage once and its text blocks in order, the answer is the
last message's text, an API error line gives an ERROR result with the (cut) detail and resume_failed only for the
no-conversation text, duration_ms comes from the turn_duration line, usage fields that are not JSON ints count as 0,
reset() clears the turn, is_interrupt() finds the interrupt mark in a user line only, and turn_start_origin() tells
which transcript lines start a turn (a human prompt, a task notification, a scheduled wakeup) and which do not. is_echo()
tells which lines show that the typed message reached the transcript, and the ledger marks the turn echoed on those lines
and on any assistant line, never on a metadata line."""

import unittest

from tg_bridge.agents.claude.translate import NO_CONVERSATION, RESULT_DETAIL_CHARS
from tg_bridge.agents.claude.tui_turn import (INTERRUPTED_SUBTYPE, TURN_DURATION, TuiTurn, echo_needle, is_echo,
                                              is_echo_of, is_interrupt, is_queued, turn_start_origin)
from tg_bridge.agents.events import TurnStatus


def assistant(message_id: str, *, texts: tuple[str, ...] = (), usage: dict | None = None, uuid: str = "u") -> dict:
    message: dict = {"id": message_id, "content": [{"type": "text", "text": t} for t in texts]}
    if usage is not None:
        message["usage"] = usage
    return {"type": "assistant", "uuid": uuid, "message": message}


class TuiTurnTests(unittest.TestCase):
    def test_lines_of_one_message_id_count_usage_once_and_keep_text_blocks_in_order(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("first",), usage={"input_tokens": 10, "output_tokens": 1}))
        turn.note(assistant("m1", texts=("second",), usage={"input_tokens": 10, "output_tokens": 7}))
        turn.note(assistant("m2", texts=("answer",), usage={"input_tokens": 3, "output_tokens": 2,
                                                           "cache_read_input_tokens": 5,
                                                           "cache_creation_input_tokens": 4}))
        usage = turn.usage_event()
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.cached_tokens), (13, 9, 9))
        self.assertFalse(usage.cost_known)
        result = turn.result(None)
        self.assertEqual(result.status, TurnStatus.OK)
        self.assertEqual(result.answer, "answer")  # the last message's blocks only
        self.assertIsNone(result.duration_ms)

    def test_the_answer_joins_the_blocks_of_the_last_message_that_has_text(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("one",)))
        turn.note(assistant("m1", texts=("two",)))
        turn.note(assistant("m2", texts=("   ",)))  # blank text is no answer
        self.assertEqual(turn.result(None).answer, "one\ntwo")

    def test_no_text_at_all_gives_no_answer(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", usage={"input_tokens": 1}))
        self.assertIsNone(turn.result({"durationMs": 9}).answer)

    def test_duration_comes_from_the_turn_duration_line(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("ok",)))
        result = turn.result({"type": "system", "subtype": TURN_DURATION, "durationMs": 1234})
        self.assertEqual(result.duration_ms, 1234)
        self.assertEqual(result.subtype, TURN_DURATION)

    def test_a_bool_or_a_string_usage_value_counts_as_zero(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", usage={"input_tokens": True, "output_tokens": "5", "cache_read_input_tokens": 2}))
        usage = turn.usage_event()
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.cached_tokens), (0, 0, 2))

    def test_an_api_error_line_gives_an_error_result_with_its_text(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("rate limited",)) | {"isApiErrorMessage": True})
        result = turn.result({"durationMs": 5})
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertEqual(result.error, "rate limited")
        self.assertEqual(result.subtype, TURN_DURATION)
        self.assertEqual(result.duration_ms, 5)
        self.assertFalse(result.resume_failed)

    def test_an_error_without_text_says_no_detail(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1") | {"isApiErrorMessage": True})
        self.assertEqual(turn.result(None).error, "no detail")

    def test_the_error_detail_is_cut_to_the_result_limit(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("x" * (RESULT_DETAIL_CHARS + 50),)) | {"isApiErrorMessage": True})
        self.assertEqual(len(turn.result(None).error), RESULT_DETAIL_CHARS)

    def test_the_no_conversation_text_marks_a_failed_resume(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=(NO_CONVERSATION,)) | {"isApiErrorMessage": True})
        self.assertTrue(turn.result(None).resume_failed)

    def test_a_later_ok_line_replaces_an_earlier_error_line(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("broken",)) | {"isApiErrorMessage": True})
        turn.note(assistant("m2", texts=("fine",)))
        self.assertEqual(turn.result(None).status, TurnStatus.OK)

    def test_non_assistant_lines_are_not_noted(self) -> None:
        turn = TuiTurn()
        turn.note({"type": "user", "message": {"id": "m1", "usage": {"input_tokens": 99}}})
        self.assertEqual(turn.usage_event().input_tokens, 0)
        self.assertIsNone(turn.result(None).answer)

    def test_reset_clears_the_turn_and_the_interrupting_flag(self) -> None:
        turn = TuiTurn()
        turn.note(assistant("m1", texts=("old",), usage={"input_tokens": 4}) | {"isApiErrorMessage": True})
        turn.interrupting = True
        turn.reset()
        self.assertFalse(turn.interrupting)
        self.assertEqual(turn.usage_event().input_tokens, 0)
        result = turn.result(None)
        self.assertEqual(result.status, TurnStatus.OK)
        self.assertIsNone(result.answer)

    def test_interrupt_result_subtype_constant(self) -> None:
        self.assertEqual(INTERRUPTED_SUBTYPE, "interrupted")

    def test_is_interrupt_finds_the_mark_in_a_user_line_only(self) -> None:
        mark = {"type": "user", "message": {"content": "[Request interrupted by user]"}}
        self.assertTrue(is_interrupt(mark))
        self.assertFalse(is_interrupt({"type": "assistant", "message": {"content": "[Request interrupted by user]"}}))
        self.assertFalse(is_interrupt({"type": "user", "message": {"content": "hello"}}))


class TurnStartOriginTests(unittest.TestCase):
    def test_the_origin_of_each_kind_of_turn_start(self) -> None:
        notification = {"type": "user", "message": {"role": "user", "content": "<task-notification>x</task-notification>"}}
        cases = [
            ({"type": "user", "turnOrigin": "human", "origin": {"kind": "human"},
              "message": {"role": "user", "content": "hi"}}, "human"),
            ({"type": "user", "turnOrigin": "task_notification", **notification}, "task_notification"),
            ({**notification, "origin": {"kind": "task-notification"}}, "task_notification"),
            ({"type": "user", "isMeta": True, "turnOrigin": "scheduled",
              "message": {"role": "user", "content": "Reply with AWAKE."}}, "scheduled"),
        ]
        for line, origin in cases:
            with self.subTest(origin=origin):
                self.assertEqual(turn_start_origin(line), origin)

    def test_lines_that_do_not_start_a_turn_give_none(self) -> None:
        cases = [
            ("tool_result list", {"type": "user", "turnOrigin": "human", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "hi", "is_error": False}]}}),
            ("local command, no turnOrigin", {"type": "user", "message": {"role": "user", "content": "<command-name>/exit"}}),
            ("assistant line", {"type": "assistant", "turnOrigin": "scheduled", "message": {"id": "m", "content": []}}),
            ("scheduled_task_fire", {"type": "system", "subtype": "scheduled_task_fire", "taskId": "w"}),
            ("unknown turnOrigin", {"type": "user", "turnOrigin": "queued", "message": {"role": "user", "content": "x"}}),
            ("origin that is not an object", {"type": "user", "origin": "task-notification"}),
        ]
        for name, line in cases:
            with self.subTest(case=name):
                self.assertIsNone(turn_start_origin(line))


if __name__ == "__main__":
    unittest.main()


SDK_PROMPT = {"type": "user", "turnOrigin": "sdk", "entrypoint": "sdk-cli", "message": {"role": "user", "content": "hi"}}
METADATA_LINES = [{"type": "last-prompt", "lastPrompt": "hi"}, {"type": "cost-state", "costUsd": 0},
                  {"type": "custom-title", "customTitle": "t"}, {"type": "agent-name", "agentName": "a"},
                  {"type": "attachment", "attachment": {}}]


class EchoTests(unittest.TestCase):
    def test_a_typed_prompt_is_an_echo_in_each_shape_the_tui_writes(self) -> None:
        human = {"type": "user", "turnOrigin": "human", "origin": {"kind": "human"},
                 "message": {"role": "user", "content": "hi"}}
        no_origin = {"type": "user", "message": {"role": "user", "content": "hi"}}
        listed = {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": [
            {"type": "text", "text": "hi"}]}}
        for name, line in (("sdk", SDK_PROMPT), ("human", human), ("no origin", no_origin), ("text list", listed)):
            with self.subTest(name):
                self.assertTrue(is_echo(line))

    def test_a_tool_result_a_meta_line_and_a_self_turn_are_not_echoes(self) -> None:
        tool_result = {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}}
        meta = dict(SDK_PROMPT, isMeta=True)
        scheduled = dict(SDK_PROMPT, turnOrigin="scheduled")
        notification = {"type": "user", "turnOrigin": "task_notification", "origin": {"kind": "task-notification"},
                        "message": {"role": "user", "content": "<task-notification/>"}}
        blank = dict(SDK_PROMPT, message={"role": "user", "content": "   "})
        for name, line in (("tool_result", tool_result), ("isMeta", meta), ("scheduled", scheduled),
                           ("task notification", notification), ("blank", blank),
                           ("assistant", assistant("m1", texts=("hi",)))):
            with self.subTest(name):
                self.assertFalse(is_echo(line))
        for line in METADATA_LINES:
            with self.subTest(line["type"]):
                self.assertFalse(is_echo(line))

    def test_an_sdk_echo_marks_the_turn_echoed(self) -> None:
        turn = TuiTurn()
        turn.reset()
        turn.note(SDK_PROMPT)
        self.assertTrue(turn.echoed)

    def test_an_assistant_line_marks_the_turn_echoed_because_the_cli_is_working(self) -> None:
        turn = TuiTurn()
        turn.reset()
        turn.note(assistant("m1", texts=("working",)))
        self.assertTrue(turn.echoed)
        self.assertEqual(turn.result(None).answer, "working")

    def test_only_metadata_lines_leave_the_turn_unechoed(self) -> None:
        turn = TuiTurn()
        turn.reset()
        for line in METADATA_LINES:
            turn.note(line)
        self.assertFalse(turn.echoed)

    def test_an_enqueue_line_marks_the_turn_queued_but_not_echoed(self) -> None:
        enqueue = {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-11-02T10:00:00Z",
                   "sessionId": "s1", "content": "hi"}
        self.assertTrue(is_queued(enqueue))
        turn = TuiTurn()
        turn.reset()
        turn.note(enqueue)
        self.assertTrue(turn.queued)
        self.assertFalse(turn.echoed)

    def test_a_dequeue_line_marks_nothing(self) -> None:
        dequeue = {"type": "queue-operation", "operation": "dequeue", "timestamp": "2026-11-02T10:00:05Z",
                   "sessionId": "s1"}
        self.assertFalse(is_queued(dequeue))
        turn = TuiTurn()
        turn.reset()
        turn.note(dequeue)
        self.assertFalse(turn.queued)
        self.assertFalse(turn.echoed)

    def test_reset_clears_queued(self) -> None:
        turn = TuiTurn()
        turn.note({"type": "queue-operation", "operation": "enqueue", "sessionId": "s1"})
        self.assertTrue(turn.queued)
        turn.reset()
        self.assertFalse(turn.queued)

    def test_a_strict_turn_ignores_assistant_lines_before_its_echo(self) -> None:
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.note(assistant("m0", texts=("earlier answer",), usage={"input_tokens": 7}) | {"isApiErrorMessage": True})
        self.assertFalse(turn.echoed)
        self.assertTrue(turn.waiting_for_echo)
        self.assertEqual(turn.usage_event().input_tokens, 0)
        self.assertIsNone(turn.result(None).answer)
        self.assertEqual(turn.result(None).status, TurnStatus.OK)  # the earlier turn's error is not this turn's
        turn.note(SDK_PROMPT)
        self.assertTrue(turn.echoed)
        self.assertFalse(turn.waiting_for_echo)
        turn.note(assistant("m1", texts=("later answer",), usage={"input_tokens": 3}))
        self.assertEqual(turn.usage_event().input_tokens, 3)
        self.assertEqual(turn.result(None).answer, "later answer")

    def test_a_strict_turn_is_echoed_only_by_its_user_line(self) -> None:
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.note(assistant("m1", texts=("working",)))
        self.assertFalse(turn.echoed)
        for line in METADATA_LINES:
            turn.note(line)
        self.assertFalse(turn.echoed)
        turn.note(SDK_PROMPT)
        self.assertTrue(turn.echoed)

    def test_strict_survives_reset(self) -> None:
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.note(SDK_PROMPT)
        turn.note(assistant("m1", texts=("first turn",)))
        turn.reset()
        self.assertTrue(turn.strict)
        self.assertTrue(turn.waiting_for_echo)
        turn.note(assistant("m2", texts=("earlier turn, read late",)))
        self.assertFalse(turn.echoed)
        self.assertIsNone(turn.result(None).answer)

    def test_a_non_strict_turn_keeps_an_assistant_line_as_its_echo(self) -> None:
        turn = TuiTurn()
        turn.reset()
        self.assertFalse(turn.strict)
        turn.note(assistant("m1", texts=("working",)))
        self.assertTrue(turn.echoed)
        self.assertFalse(turn.waiting_for_echo)
        self.assertEqual(turn.result(None).answer, "working")

TEXT = "please summarise the ledger for March"
INTERRUPT_LINE = {"type": "user", "message": {"role": "user", "content": "[Request interrupted by user]"}}


def owner_prompt(text: str) -> dict:
    """A prompt the owner typed into the tab (an interactive prompt: turnOrigin 'human')."""
    return {"type": "user", "turnOrigin": "human", "message": {"role": "user", "content": text}}


def queued_command(prompt: str) -> dict:
    """The attachment the CLI writes when it delivers a message pasted while a turn was busy."""
    return {"type": "attachment", "attachment": {"type": "queued_command", "prompt": prompt, "commandMode": "prompt"}}


def command_line(command: str, args: str) -> dict:
    """The user line the CLI stores for a slash command with arguments: the command and its args as tags."""
    content = (f"<command-name>{command}</command-name>\n<command-message>{command[1:]}</command-message>\n"
               f"<command-args>{args}</command-args>")
    return owner_prompt(content)


def enqueue(content: str) -> dict:
    return {"type": "queue-operation", "operation": "enqueue", "sessionId": "s1", "content": content}


class AttachedEchoTests(unittest.TestCase):
    """An attached turn with its pasted text (expect): only the lines holding that text are its echo or its queue."""

    def turn(self) -> TuiTurn:
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.expect(TEXT)
        return turn

    def test_another_prompt_of_the_owner_and_an_interrupt_line_are_not_the_echo(self) -> None:
        turn = self.turn()
        turn.note(owner_prompt("what is in the ledger?"))
        turn.note(INTERRUPT_LINE)
        self.assertFalse(turn.echoed)
        self.assertTrue(turn.waiting_for_echo)
        turn.note(owner_prompt(TEXT + ", and the totals"))
        self.assertTrue(turn.echoed)

    def test_a_foreign_enqueue_does_not_mark_the_turn_queued(self) -> None:
        turn = self.turn()
        turn.note(enqueue("<task-notification><task-id>b1</task-id></task-notification>"))
        turn.note(enqueue("a question typed in the tab"))
        self.assertFalse(turn.queued)
        self.assertFalse(turn.echoed)
        turn.note(enqueue(TEXT))
        self.assertTrue(turn.queued)
        self.assertFalse(turn.echoed, "a queue line is not the echo: the message has not started its turn")

    def test_the_queued_command_attachment_holding_the_text_is_the_echo(self) -> None:
        turn = self.turn()
        turn.note(queued_command("a different prompt"))
        self.assertFalse(turn.echoed)
        turn.note(queued_command(TEXT))
        self.assertTrue(turn.echoed)
        self.assertFalse(turn.waiting_for_echo)
        turn.note(assistant("m1", texts=("answer",)))
        self.assertEqual(turn.result(None).answer, "answer")

    def test_a_blank_text_keeps_the_strict_rule_of_any_user_line(self) -> None:
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.expect("   ")
        turn.note(owner_prompt("anything"))
        self.assertTrue(turn.echoed)

    def test_reset_clears_the_text_and_expect_after_it_names_the_next_turn(self) -> None:
        turn = self.turn()
        turn.reset()
        turn.note(owner_prompt("anything"))
        self.assertTrue(turn.echoed, "the text of the earlier turn is still held after reset")
        turn = TuiTurn(strict=True)
        turn.reset()
        turn.expect("the next message, long enough to be named by its start")
        turn.note(owner_prompt(TEXT))
        self.assertFalse(turn.echoed)

    def test_a_non_strict_turn_ignores_the_text(self) -> None:
        turn = TuiTurn()
        turn.reset()
        turn.expect(TEXT)
        turn.note(owner_prompt("anything"))
        self.assertTrue(turn.echoed)
        turn = TuiTurn()
        turn.reset()
        turn.expect(TEXT)
        turn.note(assistant("m1", texts=("working",)))
        self.assertTrue(turn.echoed)

    def test_is_echo_of_accepts_the_user_line_the_attachment_and_the_enqueue_with_the_text(self) -> None:
        needle = echo_needle(TEXT)
        self.assertTrue(is_echo_of(owner_prompt(TEXT), needle))
        self.assertTrue(is_echo_of(queued_command(TEXT), needle))
        self.assertTrue(is_echo_of(enqueue(TEXT), needle))
        self.assertFalse(is_echo_of(owner_prompt("another prompt"), needle))
        self.assertFalse(is_echo_of(INTERRUPT_LINE, needle))
        self.assertFalse(is_echo_of(queued_command("another prompt"), needle))
        self.assertFalse(is_echo_of(owner_prompt(TEXT), ""), "a blank needle matches nothing")

    def test_is_echo_of_matches_a_slash_command_with_arguments_by_its_stored_tags(self) -> None:
        tagged = command_line("/review-pr", "12")
        self.assertTrue(is_echo_of(tagged, echo_needle("/review-pr 12")))
        self.assertFalse(is_echo_of(tagged, echo_needle("/review-pr 13")), "another argument is another command")
        self.assertFalse(is_echo_of(command_line("/model", "sonnet"), echo_needle("/review-pr 12")))
        self.assertTrue(is_echo_of(command_line("/model", ""), echo_needle("/model")), "no arguments still matches")
        self.assertTrue(is_echo_of(command_line("/model", "sonnet"), echo_needle("/model")))
        self.assertFalse(is_echo_of(tagged, echo_needle("review-pr 12")), "a plain needle is matched as before")

    def test_echo_needle_is_the_first_sixty_characters_of_the_stripped_text(self) -> None:
        self.assertEqual(echo_needle("  " + TEXT + "  "), TEXT)
        self.assertEqual(echo_needle("x" * 100), "x" * 60)
        self.assertEqual(echo_needle(" \n "), "")
