"""TurnEventBus: delivery order, one listener's failure isolated from the others, failures logged.

The turn feed, permission, usage and session listeners all sit on one bus, so a crash in one of them must never stop
the turn. Nothing here touches the network or a real state directory: the logger writes into a TempHome.
"""

import os
import re
import unittest

from tests.support import TempHome
from tg_bridge.agents.events import ProcessExited, TextEvent, ToolEnd, TurnResult, TurnStatus
from tg_bridge.core.events_bus import EventListener, TurnEventBus
from tg_bridge.logging_ import Logger


class Recorder:
    """A listener that remembers every event it was given, in order."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[object] = []

    def on_event(self, event: object) -> None:
        self.seen.append(event)


class BrokenListener:
    """A listener that raises on every event whose type is in `fail_on`."""

    def __init__(self, fail_on: tuple[type, ...]) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def on_event(self, event: object) -> None:
        self.calls += 1
        if isinstance(event, self.fail_on):
            raise RuntimeError("listener exploded")


def _log_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


class DeliveryTests(unittest.TestCase):
    """A) every listener gets every event, in listener order and event order."""

    def test_each_listener_receives_each_event_in_order(self) -> None:
        with TempHome() as home:
            logger = Logger(os.path.join(home.state, "bridge.log"))
            first, second = Recorder("first"), Recorder("second")
            bus = TurnEventBus([first, second], logger)
            events = [TextEvent(text="a"), ToolEnd(tool_id="t1", ok=True),
                      TurnResult(status=TurnStatus.OK, answer="a")]
            for ev in events:
                bus.publish(ev)
        self.assertEqual(first.seen, events)
        self.assertEqual(second.seen, events)

    def test_listener_order_is_the_order_given(self) -> None:
        """A) the calls happen in the order the listeners were passed in (feed before permissions, etc.)."""
        calls: list[str] = []

        class Named:
            def __init__(self, name: str) -> None:
                self.name = name

            def on_event(self, event: object) -> None:
                calls.append(self.name)

        with TempHome() as home:
            bus = TurnEventBus([Named("feed"), Named("permissions"), Named("usage")],
                               Logger(os.path.join(home.state, "bridge.log")))
            bus.publish(ProcessExited(returncode=0))
        self.assertEqual(calls, ["feed", "permissions", "usage"])

    def test_publish_with_no_listeners_is_a_no_op(self) -> None:
        with TempHome() as home:
            bus = TurnEventBus([], Logger(os.path.join(home.state, "bridge.log")))
            bus.publish(TextEvent(text="nobody listens"))
            self.assertEqual(bus.listeners, ())

    def test_listeners_property_returns_a_snapshot(self) -> None:
        """A) the bus copies the listeners it was given; later changes to the caller's list do not reach it."""
        with TempHome() as home:
            given: list[EventListener] = [Recorder("only")]
            bus = TurnEventBus(given, Logger(os.path.join(home.state, "bridge.log")))
            given.append(Recorder("late"))
            self.assertEqual(len(bus.listeners), 1)


class FailureIsolationTests(unittest.TestCase):
    """B) a listener that raises is logged and skipped for that event; the rest of the turn goes on."""

    def test_failing_listener_does_not_stop_later_listeners(self) -> None:
        """B) listener exception: the listeners after it still get the same event."""
        with TempHome() as home:
            logger = Logger(os.path.join(home.state, "bridge.log"))
            broken = BrokenListener(fail_on=(TextEvent,))
            after = Recorder("after")
            bus = TurnEventBus([broken, after], logger)
            ev = TextEvent(text="hello")
            bus.publish(ev)
        self.assertEqual(after.seen, [ev])
        self.assertEqual(broken.calls, 1)

    def test_failing_listener_does_not_stop_earlier_listeners_or_publish(self) -> None:
        """B) the listeners before the failing one already got the event, and publish() returns normally."""
        with TempHome() as home:
            before = Recorder("before")
            broken = BrokenListener(fail_on=(TextEvent,))
            bus = TurnEventBus([before, broken], Logger(os.path.join(home.state, "bridge.log")))
            bus.publish(TextEvent(text="x"))
        self.assertEqual(len(before.seen), 1)

    def test_failing_listener_still_gets_the_next_event(self) -> None:
        """B) the failure is per event: the same listener receives the following events."""
        with TempHome() as home:
            broken = BrokenListener(fail_on=(TextEvent,))
            bus = TurnEventBus([broken], Logger(os.path.join(home.state, "bridge.log")))
            bus.publish(TextEvent(text="one"))
            bus.publish(ToolEnd(tool_id="t", ok=True))
        self.assertEqual(broken.calls, 2)

    def test_failure_is_logged_with_listener_and_event_names(self) -> None:
        """B) the log names the listener class and the event type, and carries the exception text."""
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            bus = TurnEventBus([BrokenListener(fail_on=(TextEvent,))], Logger(path))
            bus.publish(TextEvent(text="x"))
            lines = _log_lines(path)
        self.assertEqual(len(lines), 1)
        self.assertIn("BrokenListener", lines[0])
        self.assertIn("TextEvent", lines[0])
        self.assertIn("listener exploded", lines[0])
        self.assertRegex(lines[0], r"^\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z ")

    def test_healthy_turn_logs_nothing(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            bus = TurnEventBus([Recorder("ok")], Logger(path))
            bus.publish(TextEvent(text="fine"))
            self.assertEqual(_log_lines(path), [])

    def test_non_exception_base_errors_are_not_swallowed(self) -> None:
        """B) only Exception is caught: a KeyboardInterrupt from a listener propagates to the caller."""

        class Interrupting:
            def on_event(self, event: object) -> None:
                raise KeyboardInterrupt

        with TempHome() as home:
            bus = TurnEventBus([Interrupting()], Logger(os.path.join(home.state, "bridge.log")))
            with self.assertRaises(KeyboardInterrupt):
                bus.publish(TextEvent(text="x"))

    def test_failure_text_is_redacted_by_the_logger(self) -> None:
        """B) a token inside a listener's exception text never reaches bridge.log."""
        token = "123456789:AAEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K1x"

        class Leaky:
            def on_event(self, event: object) -> None:
                raise RuntimeError(f"bad url https://api.telegram.org/bot{token}/getMe")

        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            TurnEventBus([Leaky()], Logger(path)).publish(TextEvent(text="x"))
            text = "\n".join(_log_lines(path))
        self.assertNotIn(token, text)
        self.assertIn("<redacted>", text)


if __name__ == "__main__":
    unittest.main()
