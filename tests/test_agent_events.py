"""Normalized event model and backend base: frozen value objects, field defaults, and AgentBackend defaults."""

import dataclasses
import unittest

from tests.fakes.backend import FakeRules, FakeSession
from tg_bridge.agents.base import AgentBackend, Capabilities, ModeChoice, SessionSpec
from tg_bridge.agents.events import (
    ApprovalRequest,
    ApprovalWithdrawn,
    Decision,
    HandoffEntry,
    ProcessExited,
    Question,
    SessionStarted,
    StatusEvent,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnResult,
    TurnStatus,
    UsageEvent,
)
from tg_bridge.config import Config, LauncherSpec


def _request(**overrides) -> ApprovalRequest:
    values = {
        "approval_id": "a1",
        "kind": "command",
        "tool": "Bash",
        "input": {"command": "ls"},
        "summary": "$ ls",
        "choices": frozenset({Decision.ALLOW_ONCE}),
    }
    values.update(overrides)
    return ApprovalRequest(**values)


class FrozenValueObjectTests(unittest.TestCase):
    def _instances(self):
        launcher = LauncherSpec(name="l", backend="claude", command="c", argv=("c",), config_dir="/x")
        return [
            SessionStarted(session_id="s"),
            TextEvent(text="t"),
            ToolStart(tool_id="t", name="Bash", label="$ ls"),
            ToolEnd(tool_id="t", ok=True),
            StatusEvent(kind="retry"),
            HandoffEntry(role="user", text="hi"),
            Question(text="q", options=("a",)),
            _request(),
            ApprovalWithdrawn(approval_id="a1"),
            UsageEvent(input_tokens=1),
            TurnResult(status=TurnStatus.OK, answer="x"),
            ProcessExited(returncode=0),
            Capabilities(live_mode=True, live_model=True, questions=False, cost_usd=False, images=True,
                         preassigned_id=True),
            ModeChoice(key="default", label="Default"),
            SessionSpec(thread_id=1, chat_id=2, topic="t", launcher=launcher, cwd="/", mode="default", model=None,
                        resume_id=None, new_id=None, allow_rules=(), deny_tools=(), system_rules="", env={},
                        errfile_dir="/"),
        ]

    def test_every_value_object_is_frozen(self) -> None:
        for obj in self._instances():
            first_field = dataclasses.fields(obj)[0].name
            with self.subTest(type=type(obj).__name__):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(obj, first_field, "changed")

    def test_every_value_object_is_a_dataclass_with_slots(self) -> None:
        for obj in self._instances():
            with self.subTest(type=type(obj).__name__):
                self.assertTrue(dataclasses.is_dataclass(obj))
                self.assertFalse(hasattr(obj, "__dict__"))


class EventDefaultsTests(unittest.TestCase):
    def test_session_started_defaults(self) -> None:
        started = SessionStarted(session_id="abc")
        self.assertEqual((started.model, started.mode, started.cwd, started.commands, started.resumed),
                         (None, None, None, (), False))

    def test_text_event_defaults(self) -> None:
        event = TextEvent(text="hi")
        self.assertFalse(event.final)
        self.assertIsNone(event.item_id)
        self.assertIsNone(event.parent_id)

    def test_status_event_attempt_fields_default_to_none(self) -> None:
        event = StatusEvent(kind="retry", attempt=2, max_attempts=10)
        self.assertEqual((event.attempt, event.max_attempts, event.retry_in), (2, 10, None))
        self.assertIsNone(StatusEvent(kind="stream_cut").max_attempts)

    def test_turn_result_defaults(self) -> None:
        result = TurnResult(status=TurnStatus.INTERRUPTED, answer=None)
        self.assertEqual((result.subtype, result.error, result.resume_failed), ("", None, False))
        self.assertIsNone(result.duration_ms)
        self.assertIsNone(result.num_turns)

    def test_usage_defaults_to_zero_and_unknown_cost(self) -> None:
        usage = UsageEvent()
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.cached_tokens), (0, 0, 0))
        self.assertIsNone(usage.cost_usd)
        self.assertFalse(usage.cost_known)

    def test_tool_start_input_defaults_to_an_empty_dict_per_instance(self) -> None:
        first = ToolStart(tool_id="1", name="Bash", label="a")
        second = ToolStart(tool_id="2", name="Bash", label="b")
        self.assertEqual(first.input, {})
        self.assertIsNot(first.input, second.input)

    def test_approval_raw_payload_is_not_compared(self) -> None:
        self.assertEqual(_request(raw={"x": 1}), _request(raw={"x": 2}))

    def test_handoff_entry_holds_role_and_text(self) -> None:
        entry = HandoffEntry(role="tool", text="$ git status")
        self.assertEqual((entry.role, entry.text), ("tool", "$ git status"))


class _MinimalBackend(AgentBackend):
    """The smallest concrete backend, to check the non-abstract defaults of AgentBackend."""

    name = "minimal"

    def __init__(self) -> None:
        self.capabilities = Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False,
                                         images=False, preassigned_id=False)
        self.rules = FakeRules()

    def modes(self):
        return ()

    def models(self, cfg: Config):
        return ()

    def open_session(self, spec, sink):
        raise NotImplementedError


class AgentBackendDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _MinimalBackend()
        self.launcher = LauncherSpec(name="l", backend="minimal", command="m", argv=("m",), config_dir="/x")

    def test_carry_history_defaults_to_false(self) -> None:
        self.assertIs(self.backend.carry_history(self.launcher, self.launcher, "/src", "sid"), False)

    def test_handoff_entries_default_to_empty_tuple(self) -> None:
        self.assertEqual(self.backend.handoff_entries("sid", "/src", self.launcher), ())

    def test_one_shot_and_resume_hint_defaults(self) -> None:
        self.assertIsNone(self.backend.one_shot("title", 5.0))
        self.assertEqual(self.backend.resume_hint("sid", "/src"), "")

    def test_fake_session_is_an_agent_session(self) -> None:
        from tg_bridge.agents.base import AgentSession

        self.assertTrue(issubclass(FakeSession, AgentSession))


if __name__ == "__main__":
    unittest.main()
