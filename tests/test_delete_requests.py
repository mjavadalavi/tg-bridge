"""DeleteRequests (card D3): the daemon's watcher for `view-delete`, and the remover's view cleanup that keeps the
request and answer files.

The watcher is driven with run_once() and a stub remover, so each answer text is exact. One test runs the real
TopicRemover of an App over a FakeTelegram, to check that the answer file survives the removal of the topic's files.
Nothing touches the real state directory or the real Telegram API.
"""

import os
import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.app import App
from tg_bridge.config import Config
from tg_bridge.core.delete_requests import (DELETED_TEXT, GONE_TEXT, DeleteRequests, ack_path, request_path,
                                            request_thread, result_path)
from tg_bridge.core.topic_remover import remove_view_files
from tg_bridge.paths import Paths

CHAT = -1009905
THREAD = 7
NAME = "alpha"
GROUP_KEY = "group_id"
DELETED_LINE = f"topic {THREAD} removed (telegram: deleted)"
GONE_LINE = f"topic {THREAD} removed (telegram: already gone)"
REFUSED_LINE = f"topic {THREAD} not removed: the Telegram deletion failed: boom"


class Recorder:
    """A logger that keeps its lines in a list."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, line: str) -> None:
        self.lines.append(line)


class StubSessions:
    """Only get() is used: the topic's name."""

    def __init__(self, names: dict[int, str]) -> None:
        self._names = names

    def get(self, thread_id: int):
        name = self._names.get(thread_id)
        return None if name is None else type("Record", (), {"name": name})()


class StubRemover:
    """Answers with the scripted line (or raises the scripted exception) and records each call."""

    def __init__(self, line: str = DELETED_LINE, raises: Exception | None = None) -> None:
        self.line = line
        self.raises = raises
        self.calls: list[tuple[int, bool]] = []

    def remove(self, thread_id: int, delete_in_telegram: bool) -> str:
        self.calls.append((thread_id, delete_in_telegram))
        if self.raises is not None:
            raise self.raises
        return self.line


class WatcherCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view)
        self.logger = Recorder()

    def watcher(self, remover: object, names: dict[int, str] | None = None, interval: float = 2.0) -> DeleteRequests:
        sessions = StubSessions({THREAD: NAME} if names is None else names)
        return DeleteRequests(view_dir=self.view, sessions=sessions, remover=remover, logger=self.logger,
                              interval=interval)

    def request(self, thread: int = THREAD) -> str:
        path = request_path(self.view, thread)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("requested now\n")
        return path

    def answer(self, thread: int = THREAD) -> str:
        with open(result_path(self.view, thread), encoding="utf-8") as handle:
            return handle.read()


class HandlingTests(WatcherCase):
    def test_a_request_is_handled_once_and_its_answer_is_written(self) -> None:
        remover = StubRemover(DELETED_LINE)
        watcher = self.watcher(remover)
        self.request()
        watcher.run_once()
        watcher.run_once()  # the request file is gone: nothing is handled a second time
        self.assertEqual(remover.calls, [(THREAD, True)])
        self.assertFalse(os.path.exists(request_path(self.view, THREAD)))
        self.assertEqual(self.answer(), DELETED_TEXT.format(name=NAME) + "\n")

    def test_a_topic_already_gone_in_telegram_says_so(self) -> None:
        self.request()
        self.watcher(StubRemover(GONE_LINE)).run_once()
        self.assertEqual(self.answer(), GONE_TEXT + "\n")

    def test_a_refused_removal_is_a_could_not_delete_answer_with_the_reason(self) -> None:
        self.request()
        self.watcher(StubRemover(REFUSED_LINE)).run_once()
        self.assertEqual(self.answer(), "could not delete: the Telegram deletion failed: boom\n")

    def test_a_remover_that_raises_still_answers_so_the_cli_does_not_wait(self) -> None:
        self.request()
        self.watcher(StubRemover(raises=RuntimeError("db locked"))).run_once()
        self.assertEqual(self.answer(), "could not delete: db locked\n")
        self.assertTrue(any("db locked" in line for line in self.logger.lines))

    def test_an_unknown_topic_name_is_shown_by_its_thread_number(self) -> None:
        self.request(THREAD)
        self.watcher(StubRemover(DELETED_LINE), names={}).run_once()
        self.assertEqual(self.answer(), DELETED_TEXT.format(name=str(THREAD)) + "\n")

    def test_names_outside_the_request_shape_are_ignored_and_left_alone(self) -> None:
        names = ("x.delete", "7.deleted", "7.deleting", "07a.delete", "-3.delete", "7.delete.tmp", "delete")
        for name in names:
            with open(os.path.join(self.view, name), "w", encoding="utf-8") as handle:
                handle.write("requested\n")
        remover = StubRemover()
        self.watcher(remover).run_once()
        self.assertEqual(remover.calls, [])
        self.assertEqual(sorted(os.listdir(self.view)), sorted(names))

    def test_non_canonical_request_names_are_removed_and_logged_never_acted_on(self) -> None:
        names = ("0.delete", "0012.delete", "00.delete")
        for name in names:
            with open(os.path.join(self.view, name), "w", encoding="utf-8") as handle:
                handle.write("requested\n")
        remover = StubRemover()
        self.watcher(remover).run_once()
        self.assertEqual(remover.calls, [])
        self.assertEqual(os.listdir(self.view), [])
        for name in names:
            self.assertIn(f"view delete: {name} is not a canonical request name, removed", self.logger.lines)

    def test_canonical_request_names_are_the_positive_numbers_written_without_leading_zeros(self) -> None:
        self.assertEqual(request_thread("12.delete"), 12)
        for name in ("0.delete", "0012.delete", "-3.delete", "x.delete", "12.deleted"):
            self.assertIsNone(request_thread(name))

    def test_the_daemon_acknowledges_before_it_works_and_clears_the_ack_after_the_answer(self) -> None:
        view = self.view

        class Probe(StubRemover):
            def remove(self, thread_id: int, delete_in_telegram: bool) -> str:
                self.acked = os.path.exists(ack_path(view, thread_id))
                return super().remove(thread_id, delete_in_telegram)

        remover = Probe(DELETED_LINE)
        self.request()
        self.watcher(remover).run_once()
        self.assertTrue(remover.acked, "the .deleting file must exist while the removal runs")
        self.assertFalse(os.path.exists(ack_path(self.view, THREAD)))
        self.assertTrue(os.path.exists(result_path(self.view, THREAD)))

    def test_a_missing_view_folder_has_no_requests(self) -> None:
        remover = StubRemover()
        DeleteRequests(view_dir=os.path.join(self.view, "missing"), sessions=StubSessions({}), remover=remover,
                       logger=self.logger).run_once()
        self.assertEqual(remover.calls, [])


class ThreadTests(WatcherCase):
    def test_start_with_interval_zero_starts_no_thread(self) -> None:
        watcher = self.watcher(StubRemover(), interval=0)
        before = threading.active_count()
        watcher.start()
        self.assertEqual(threading.active_count(), before)
        watcher.stop()

    def test_the_thread_answers_a_request_and_stops_on_shutdown(self) -> None:
        remover = StubRemover()
        watcher = self.watcher(remover, interval=0.01)
        watcher.start()
        self.request()
        deadline = time.monotonic() + 5
        while not os.path.exists(result_path(self.view, THREAD)) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(remover.calls, [(THREAD, True)])
        watcher.stop()
        self.assertFalse(watcher._thread.is_alive())
        self.request(thread=8)  # after the shutdown nothing takes it
        time.sleep(0.05)
        self.assertTrue(os.path.exists(request_path(self.view, 8)))
        self.assertEqual(remover.calls, [(THREAD, True)])


class RemoverCleanupTests(WatcherCase):
    def test_the_view_cleanup_removes_the_topic_files_and_keeps_the_request_and_answer(self) -> None:
        for name in ("5.json", "5.feed", "5.lock", "5.delete", "5.deleting", "5.deleted", "50.json"):
            with open(os.path.join(self.view, name), "w", encoding="utf-8") as handle:
                handle.write("x")
        self.assertEqual(remove_view_files(self.view, 5), 2)
        self.assertEqual(sorted(os.listdir(self.view)),
                         ["5.delete", "5.deleted", "5.deleting", "5.lock", "50.json"])

    def test_the_answer_survives_a_real_removal_of_the_topic(self) -> None:
        home = self.enterContext(TempHome())
        clock = FakeClock()
        telegram = FakeTelegram(clock)
        app = App.build(Paths.from_home(home.home), Config({"allow_user_ids": [42]}, home=home.home, env={}),
                        telegram=telegram, clock=clock, env={})
        self.addCleanup(app.close)
        app.kv.set(GROUP_KEY, str(CHAT))
        app.sessions.ensure(THREAD, NAME, "claude-cmd", home.home)
        with open(os.path.join(app.paths.view, f"{THREAD}.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(request_path(app.paths.view, THREAD), "w", encoding="utf-8") as handle:
            handle.write("requested now\n")
        DeleteRequests(view_dir=app.paths.view, sessions=app.sessions, remover=app.remover,
                       logger=self.logger).run_once()
        self.assertIsNone(app.sessions.get(THREAD))
        self.assertFalse(os.path.exists(os.path.join(app.paths.view, f"{THREAD}.json")))
        with open(result_path(app.paths.view, THREAD), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), DELETED_TEXT.format(name=NAME) + "\n")
        self.assertIn("deleteForumTopic", [call.method for call in telegram.calls])


if __name__ == "__main__":
    unittest.main()
