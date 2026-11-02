"""TurnRunner with a view factory (the ViewFactory wiring of core/turn_runner.py): every attempt gets its own listener built
from the topic's thread, name, transport and launcher, and that listener receives the turn's events. An end-to-end case
writes the real view files through ViewRecorder. Runs on FakeBackend (tests/fakes/backend.py), no process is started.
"""

import json
import os
import unittest
from unittest import mock

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, RecordingFeed, TurnHarness
from tg_bridge.agents.events import TextEvent, TurnResult, TurnStatus
from tg_bridge.core.view_recorder import ViewRecorder

OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")


class RecordingView(RecordingFeed):
    """A recording listener that also takes the view's end call (ViewRecorder.finish)."""

    def __init__(self) -> None:
        super().__init__()
        self.finished = False

    def finish(self) -> None:
        self.finished = True


class ViewWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.created: list[tuple[int, str, str, str, RecordingFeed]] = []

    def backend(self, turns: int) -> FakeBackend:
        return FakeBackend([[TextEvent(text=f"hi {n}"), OK] for n in range(turns)], capabilities=FAKE_CAPABILITIES)

    def recording_factory(self, thread_id: int, topic: str, transport: str, launcher: str) -> RecordingView:
        listener = RecordingView()
        self.created.append((thread_id, topic, transport, launcher, listener))
        return listener

    def turn(self, backend: FakeBackend, views=None):
        runner = self.h.runner(self.h.registry_for(backend), views=views)
        return runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")

    def test_the_factory_gets_the_topic_and_its_transport_and_launcher(self) -> None:
        reply = self.turn(self.backend(1), views=self.recording_factory)
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(len(self.created), 1)
        thread_id, topic, transport, launcher, _ = self.created[0]
        self.assertEqual((thread_id, topic, transport), (THREAD, TOPIC, "print"))
        self.assertTrue(launcher, "the launcher name must be passed")

    def test_the_listener_receives_the_turn_events_and_its_result(self) -> None:
        self.turn(self.backend(1), views=self.recording_factory)
        listener = self.created[0][4]
        self.assertIn(TextEvent(text="hi 0"), listener.events)
        self.assertTrue(any(isinstance(event, TurnResult) for event in listener.events))
        self.assertTrue(listener.finished, "the attempt's cleanup ends the view")

    def test_each_turn_gets_its_own_listener(self) -> None:
        backend = self.backend(2)
        self.turn(backend, views=self.recording_factory)
        self.turn(backend, views=self.recording_factory)
        self.assertEqual(len(self.created), 2)
        self.assertIsNot(self.created[0][4], self.created[1][4])

    def test_without_a_factory_no_listener_is_made_and_the_turn_runs(self) -> None:
        reply = self.turn(self.backend(1), views=None)
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(self.created, [])


class ViewDirSpecTests(unittest.TestCase):
    """The spec a session is opened with carries the view folder (empty when there is none) and the configured tui_size."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def spec_of_one_turn(self, view_dir: str):
        backend = FakeBackend([[TextEvent(text="x"), OK]], capabilities=FAKE_CAPABILITIES)
        runner = self.h.runner(self.h.registry_for(backend), view_dir=view_dir)
        runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(len(backend.specs), 1)
        return backend.specs[0]

    def test_the_spec_carries_the_view_folder_and_the_configured_tui_size(self) -> None:
        self.h.set_config(tui_size="120x40")
        spec = self.spec_of_one_turn("/state/view")
        self.assertEqual((spec.view_dir, spec.tui_size), ("/state/view", (40, 120)))

    def test_without_a_view_folder_the_spec_has_none_and_the_default_size(self) -> None:
        spec = self.spec_of_one_turn("")
        self.assertEqual((spec.view_dir, spec.tui_size), ("", (50, 200)))


class ViewFilesEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.view_dir = os.path.join(self.home.home, "view")
        os.makedirs(self.view_dir)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def test_a_turn_leaves_an_idle_meta_file_and_a_feed_ending_in_the_result(self) -> None:
        def factory(thread_id: int, topic: str, transport: str, launcher: str) -> ViewRecorder:
            return ViewRecorder(self.view_dir, thread_id, topic, transport, launcher, self.h.clock, self.h.logger)

        backend = FakeBackend([[TextEvent(text="the answer"), OK]], capabilities=FAKE_CAPABILITIES)
        runner = self.h.runner(self.h.registry_for(backend), views=factory)
        runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        with open(os.path.join(self.view_dir, f"{THREAD}.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual((meta["topic"], meta["transport"], meta["state"]), (TOPIC, "print", "idle"))
        with open(os.path.join(self.view_dir, f"{THREAD}.feed"), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[1:], ["the answer", "✅ done · 0:00"])


class BrokenViewTests(unittest.TestCase):
    """A view that cannot be built, or that fails to write, costs the view only: the turn still runs, and a launch that
    failed (no end event at all) leaves the topic idle, not running."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.view_dir = os.path.join(self.home.home, "view")
        os.makedirs(self.view_dir)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def real_factory(self, launcher: str = "claude-main"):
        def factory(thread_id: int, topic: str, transport: str, name: str) -> ViewRecorder:
            return ViewRecorder(self.view_dir, thread_id, topic, transport, launcher, self.h.clock, self.h.logger)
        return factory

    def turn(self, backend: FakeBackend, views):
        runner = self.h.runner(self.h.registry_for(backend), views=views)
        return runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")

    def meta(self) -> dict:
        with open(os.path.join(self.view_dir, f"{THREAD}.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_a_factory_that_raises_costs_only_the_view(self) -> None:
        def factory(thread_id: int, topic: str, transport: str, launcher: str) -> ViewRecorder:
            raise ValueError("cannot build the view")

        backend = FakeBackend([[TextEvent(text="hi"), OK]], capabilities=FAKE_CAPABILITIES)
        reply = self.turn(backend, factory)
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(len(backend.specs), 1, "the turn ran")
        with open(self.h.log_path, encoding="utf-8") as handle:
            self.assertIn("recorder not built, the turn runs without it", handle.read())

    def test_a_launch_that_fails_before_any_event_leaves_the_topic_idle(self) -> None:
        backend = FakeBackend([[OK]], capabilities=FAKE_CAPABILITIES)
        with mock.patch.object(backend, "open_session", side_effect=RuntimeError("no such binary")):
            reply = self.turn(backend, self.real_factory())
        self.assertIn("launcher failed", reply.err or "")
        self.assertEqual(self.meta()["state"], "idle")

    def test_a_launcher_name_with_a_lone_surrogate_still_runs_and_writes_the_view(self) -> None:
        backend = FakeBackend([[TextEvent(text="hi"), OK]], capabilities=FAKE_CAPABILITIES)
        reply = self.turn(backend, self.real_factory(launcher="bad\ud800"))
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(self.meta()["state"], "idle")

    def test_a_wiring_error_after_the_view_opens_still_leaves_the_topic_idle(self) -> None:
        """The view is opened before the bus, the end gate and the spec are built: a failure there must still finish it."""
        backend = FakeBackend([[TextEvent(text="hi"), OK]], capabilities=FAKE_CAPABILITIES)
        with mock.patch("tg_bridge.core.turn_kit.EndGate", side_effect=RuntimeError("wiring broke")):
            reply = self.turn(backend, self.real_factory())
        self.assertIn("launcher failed", reply.err or "")
        self.assertIn("wiring broke", reply.err or "")
        self.assertEqual(self.meta()["state"], "idle")


if __name__ == "__main__":
    unittest.main()
