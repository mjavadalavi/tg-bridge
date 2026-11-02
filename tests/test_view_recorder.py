"""ViewRecorder (core/view_recorder.py): the per-topic feed and meta files of `tg-bridge view`.

Covers the header and the running state at construction, the lines per event, the idle state at the end of a turn, the
feed cap (the newest half is kept, cut at a line), atomic writes, the OSError tolerance (logged once, never raised) and
the lock when several reader threads publish at once.
"""

import json
import os
import threading
import unittest
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.events import ProcessExited, TextEvent, ToolEnd, ToolStart, TurnResult, TurnStatus
from tg_bridge.core import view_recorder
from tg_bridge.core.view_recorder import ViewRecorder
from tg_bridge.logging_ import Logger

THREAD = 4242
TOPIC = "Launch notes"
LAUNCHER = "claude-main"


class RecorderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view_dir = os.path.join(self.home.home, "view")
        os.makedirs(self.view_dir)
        self.clock = FakeClock()
        self.log_path = os.path.join(self.home.home, "bridge.log")
        self.logger = Logger(self.log_path)

    def recorder(self, view_dir: str | None = None, transport: str = "print") -> ViewRecorder:
        return ViewRecorder(self.view_dir if view_dir is None else view_dir, THREAD, TOPIC, transport, LAUNCHER,
                            self.clock, self.logger)

    def meta(self) -> dict:
        with open(os.path.join(self.view_dir, f"{THREAD}.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def feed_lines(self) -> list[str]:
        path = os.path.join(self.view_dir, f"{THREAD}.feed")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()

    def log_lines(self) -> list[str]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read().splitlines()


class ConstructionTests(RecorderCase):
    def test_construction_writes_running_meta_and_the_header(self) -> None:
        self.clock = FakeClock(start=1_700_000_000.0)
        self.recorder(transport="tui")
        meta = self.meta()
        self.assertEqual(meta, {"topic": TOPIC, "transport": "tui", "launcher": LAUNCHER, "state": "running",
                                "updated": 1_700_000_000.0})
        lines = self.feed_lines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("── 20"), lines[0])  # the local date and time
        self.assertTrue(lines[0].endswith(f" · {LAUNCHER} ──"), lines[0])

    def test_no_temp_file_is_left_behind(self) -> None:
        recorder = self.recorder()
        recorder.on_event(TurnResult(TurnStatus.OK, answer="x"))
        self.assertEqual([name for name in os.listdir(self.view_dir) if name.endswith(".tmp")], [])


class EventTests(RecorderCase):
    def test_shown_events_are_appended_as_lines_and_hidden_ones_are_not(self) -> None:
        recorder = self.recorder()
        recorder.on_event(ToolStart(tool_id="a", name="Bash", label="Bash: ls"))
        recorder.on_event(ToolEnd(tool_id="a", ok=True))
        recorder.on_event(TextEvent(text="listed"))
        self.assertEqual(self.feed_lines()[1:], ["🔧 Bash: ls", "listed"])

    def test_the_turn_result_marks_the_topic_idle(self) -> None:
        recorder = self.recorder()
        self.assertEqual(self.meta()["state"], "running")
        recorder.on_event(TurnResult(TurnStatus.OK, answer="done", duration_ms=2_000))
        self.assertEqual(self.meta()["state"], "idle")
        self.assertEqual(self.feed_lines()[-1], "✅ done · 0:02")

    def test_a_process_exit_also_marks_the_topic_idle(self) -> None:
        recorder = self.recorder()
        recorder.on_event(ProcessExited(returncode=1))
        self.assertEqual(self.meta()["state"], "idle")

    def test_a_text_event_keeps_the_running_state(self) -> None:
        recorder = self.recorder()
        recorder.on_event(TextEvent(text="working"))
        self.assertEqual(self.meta()["state"], "running")


class CapTests(RecorderCase):
    def test_past_the_cap_the_newest_half_is_kept_at_a_line_boundary(self) -> None:
        with mock.patch.object(view_recorder, "FEED_MAX_BYTES", 400):
            recorder = self.recorder()
            for number in range(60):
                recorder.on_event(TextEvent(text=f"line {number:03d} " + "-" * 20))
        size = os.path.getsize(os.path.join(self.view_dir, f"{THREAD}.feed"))
        self.assertLessEqual(size, 400)
        lines = self.feed_lines()
        self.assertEqual(lines[-1], "line 059 " + "-" * 20)  # the newest line is kept
        self.assertTrue(all(line.startswith("line ") and line.endswith("-" * 20) for line in lines),
                        lines)  # each kept line is whole, none cut in half
        self.assertGreater(len(lines), 5)

    def test_under_the_cap_nothing_is_dropped(self) -> None:
        with mock.patch.object(view_recorder, "FEED_MAX_BYTES", 10_000):
            recorder = self.recorder()
            for number in range(20):
                recorder.on_event(TextEvent(text=f"m{number}"))
        self.assertEqual(len(self.feed_lines()), 21)  # the header plus 20 lines

    def test_the_default_cap_is_one_mebibyte(self) -> None:
        self.assertEqual(view_recorder.FEED_MAX_BYTES, 1024 * 1024)


class FailureTests(RecorderCase):
    def test_a_missing_folder_never_raises_and_is_logged_once(self) -> None:
        missing = os.path.join(self.home.home, "no", "such", "folder")
        recorder = self.recorder(view_dir=missing)
        for number in range(5):
            recorder.on_event(TextEvent(text=f"m{number}"))
        recorder.on_event(TurnResult(TurnStatus.OK, answer="x"))
        logged = [line for line in self.log_lines() if "view: cannot write" in line]
        self.assertEqual(len(logged), 1, self.log_lines())

    def test_a_failing_write_is_swallowed(self) -> None:
        recorder = self.recorder()
        with mock.patch.object(view_recorder, "_append", side_effect=OSError("disk full")):
            recorder.on_event(TextEvent(text="lost"))  # must not raise
        self.assertEqual(len([line for line in self.log_lines() if "disk full" in line]), 1)


class ThreadTests(RecorderCase):
    def test_events_published_from_several_threads_are_all_written_whole(self) -> None:
        recorder = self.recorder()
        workers = 4
        per_worker = 40

        def publish(worker: int) -> None:
            for number in range(per_worker):
                recorder.on_event(TextEvent(text=f"w{worker}-{number}"))

        threads = [threading.Thread(target=publish, args=(worker,)) for worker in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = self.feed_lines()[1:]
        self.assertEqual(len(lines), workers * per_worker)
        self.assertEqual(len(set(lines)), workers * per_worker)


class FinishTests(RecorderCase):
    def test_finish_marks_a_running_topic_idle(self) -> None:
        recorder = self.recorder()
        recorder.finish()
        self.assertEqual(self.meta()["state"], "idle")

    def test_finish_writes_nothing_once_the_topic_is_idle(self) -> None:
        recorder = self.recorder()
        recorder.on_event(TurnResult(TurnStatus.OK, answer="done"))
        updated = self.meta()["updated"]
        self.clock.advance(5.0)
        recorder.finish()
        self.assertEqual(self.meta()["updated"], updated)


class SurrogateTests(RecorderCase):
    def test_a_lone_surrogate_in_the_launcher_name_leaves_an_ascii_meta_file(self) -> None:
        recorder = ViewRecorder(self.view_dir, THREAD, TOPIC, "print", "bad\ud800launcher", self.clock, self.logger)
        recorder.on_event(TurnResult(TurnStatus.OK, answer="x"))
        with open(os.path.join(self.view_dir, f"{THREAD}.json"), "rb") as handle:
            raw = handle.read()
        raw.decode("ascii")  # the surrogate is written as a JSON escape, so the file is plain ASCII
        self.assertEqual(self.meta()["launcher"], "bad\ud800launcher")
        self.assertEqual(self.meta()["state"], "idle")

    def test_a_lone_surrogate_in_the_feed_is_written_as_a_question_mark(self) -> None:
        recorder = self.recorder()
        recorder.on_event(TextEvent(text="a\ud800b"))
        self.assertEqual(self.feed_lines()[-1], "a?b")


class WriteErrorTests(RecorderCase):
    def test_a_value_error_in_a_write_is_logged_and_never_raised(self) -> None:
        recorder = self.recorder()
        with mock.patch.object(view_recorder, "_append", side_effect=ValueError("bad bytes")):
            recorder.on_event(TextEvent(text="lost"))  # must not raise
        self.assertEqual(len([line for line in self.log_lines() if "bad bytes" in line]), 1)


class StaleViewTests(RecorderCase):
    """The start-up sweep: a meta file that still says running is set to idle; the other files are left as they are."""

    def write_meta(self, thread: int, state: str, launcher: str = LAUNCHER) -> None:
        with open(os.path.join(self.view_dir, f"{thread}.json"), "w", encoding="utf-8") as handle:
            json.dump({"topic": TOPIC, "transport": "tui", "launcher": launcher, "state": state, "updated": 100.0},
                      handle)

    def stored(self, thread: int) -> dict:
        with open(os.path.join(self.view_dir, f"{thread}.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_running_topics_become_idle_and_the_others_are_kept(self) -> None:
        self.write_meta(1, "running")
        self.write_meta(2, "idle")
        self.write_meta(3, "running", launcher="codex-main")
        with open(os.path.join(self.view_dir, f"{3}.feed"), "w", encoding="utf-8") as handle:
            handle.write("old line\n")
        changed = view_recorder.mark_stale_views(self.view_dir, self.clock, self.logger)
        self.assertEqual(changed, 2)
        self.assertEqual(self.stored(1)["state"], "idle")
        self.assertEqual(self.stored(3), {"topic": TOPIC, "transport": "tui", "launcher": "codex-main",
                                          "state": "idle", "updated": self.clock.now()})
        self.assertEqual(self.stored(2)["updated"], 100.0)  # an idle file is not rewritten
        with open(os.path.join(self.view_dir, "3.feed"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "old line\n")
        self.assertIn("2 topic(s) left running", self.log_lines()[-1])

    def test_a_bad_file_is_logged_and_skipped(self) -> None:
        with open(os.path.join(self.view_dir, "4.json"), "w", encoding="utf-8") as handle:
            handle.write("not json")
        with open(os.path.join(self.view_dir, "5.json"), "w", encoding="utf-8") as handle:
            handle.write("[1, 2]")
        self.write_meta(6, "running")
        changed = view_recorder.mark_stale_views(self.view_dir, self.clock, self.logger)
        self.assertEqual(changed, 1)
        self.assertEqual(self.stored(6)["state"], "idle")
        self.assertTrue(any("4.json" in line for line in self.log_lines()), self.log_lines())

    def test_a_missing_folder_returns_zero_and_is_logged(self) -> None:
        missing = os.path.join(self.home.home, "no-view")
        self.assertEqual(view_recorder.mark_stale_views(missing, self.clock, self.logger), 0)
        self.assertEqual(len([line for line in self.log_lines() if "cannot list" in line]), 1)


class LingeringStateTests(RecorderCase):
    """The kept session of a topic shows lingering between its turns; the start-up sweep resets it like running."""

    def test_set_state_lingering_writes_meta(self) -> None:
        recorder = self.recorder(transport="tui")
        recorder.on_event(TurnResult(TurnStatus.OK, answer="done"))
        self.clock.advance(3.0)
        recorder.set_state(view_recorder.STATE_LINGERING)
        self.assertEqual(self.meta(), {"topic": TOPIC, "transport": "tui", "launcher": LAUNCHER,
                                       "state": "lingering", "updated": self.clock.now()})

    def test_finish_keeps_lingering(self) -> None:
        recorder = self.recorder()
        recorder.set_state(view_recorder.STATE_LINGERING)
        recorder.finish()
        self.assertEqual(self.meta()["state"], "lingering")

    def test_stale_lingering_is_reset_at_start_up(self) -> None:
        with open(os.path.join(self.view_dir, f"{THREAD}.json"), "w", encoding="utf-8") as handle:
            json.dump({"topic": TOPIC, "transport": "tui", "launcher": LAUNCHER, "state": "lingering",
                       "updated": 100.0}, handle)
        changed = view_recorder.mark_stale_views(self.view_dir, self.clock, self.logger)
        self.assertEqual(changed, 1)
        self.assertEqual(self.meta()["state"], "idle")


if __name__ == "__main__":
    unittest.main()
