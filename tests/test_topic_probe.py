"""TopicProbe (card D2): the daemon finds topics deleted in Telegram and removes them.

The App is built over a TempHome with a FakeTelegram and a FakeClock, so the real repositories, LiveRegistry, linger
service and TopicRemover run. Nothing touches the real state directory or the real Telegram API. The probe's thread
test uses real (tiny) seconds, because the thread waits on its stop event, not on the clock.
"""

import os
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.app import App
from tg_bridge.config import Config
from tg_bridge.core.terminal_claim import begin_turn as begin_daemon_turn
from tg_bridge.core.terminal_claim import take_claim
from tg_bridge.core.topic_probe import CALL_SPACING_SECONDS, PROBE_ICON, TopicProbe
from tg_bridge.paths import Paths

CHAT = -1009902
ALLOWED_UID = 42
GROUP_KEY = "group_id"
DELETED = "Bad Request: TOPIC_ID_INVALID"
ALIVE = "Bad Request: DOCUMENT_INVALID"
GONE_THREAD = 7
LIVE_THREAD = 8
NEXT_THREAD = 9
BUSY_RUNNING = 11
BUSY_LINGER = 12
BUSY_CLAIM = 13
BUSY_MARK = 14
PROBE_METHOD = "editForumTopic"


class Recorder:
    """A logger that keeps its lines in a list."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, line: str) -> None:
        self.lines.append(line)


class RouteClient:
    """Fails the probe of the given threads with the scripted (code, description), then passes every call through."""

    def __init__(self, inner: FakeTelegram, errors: dict[int, tuple[int, str]] | None = None) -> None:
        self.inner = inner
        self.errors = dict(errors or {})

    def call(self, method, feed=False, **params):
        thread = params.get("message_thread_id")
        if method == PROBE_METHOD and thread in self.errors:
            code, description = self.errors.pop(thread)
            self.inner.fail_next(method, code, description=description)
        return self.inner.call(method, feed, **params)


class DownClient:
    """A client whose connection is down: every call raises a plain connection error, not a Telegram answer."""

    def call(self, method, feed=False, **params):
        raise ConnectionError("connection reset by peer")


def _config(home: str, **extra) -> Config:
    return Config({"allow_user_ids": [ALLOWED_UID], **extra}, home=home, env={})


class ProbeCase(unittest.TestCase):
    """An App with the group learned and a few topics with session rows."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        self.telegram = FakeTelegram(self.clock)
        self.app = App.build(Paths.from_home(self.home.home), _config(self.home.home), telegram=self.telegram,
                             clock=self.clock, env={})
        self.addCleanup(self.app.close)
        self.app.kv.set(GROUP_KEY, str(CHAT))
        self.logger = Recorder()
        self.probe = self.make_probe(interval=600)

    def make_probe(self, interval: float, first_seconds: float = 60.0, client=None) -> TopicProbe:
        app = self.app
        return TopicProbe(sessions=app.sessions, kv=app.kv, live=app.live, linger=app.linger, remover=app.remover,
                          client=client or self.telegram, group_key=GROUP_KEY, view_dir=app.paths.view,
                          logger=self.logger, clock=self.clock, interval=interval, first_seconds=first_seconds)

    def topic(self, thread: int, name: str) -> None:
        self.app.sessions.ensure(thread, name, "claude-cmd", self.home.home)

    def probe_calls(self) -> list[dict]:
        return [c.params for c in self.telegram.calls if c.method == PROBE_METHOD]

    def probe_threads(self) -> list[int]:
        return [params["message_thread_id"] for params in self.probe_calls()]


class RunOnceTests(ProbeCase):
    def test_a_deleted_topic_is_removed_and_nothing_is_deleted_again_in_telegram(self) -> None:
        self.topic(GONE_THREAD, "gone")
        self.telegram.fail_next(PROBE_METHOD, 400, description=DELETED)
        self.probe.run_once()
        self.assertIsNone(self.app.sessions.get(GONE_THREAD))
        self.assertIn("topic 7 (gone) was deleted in Telegram: removed", self.logger.lines)
        self.assertFalse([c for c in self.telegram.calls if c.method == "deleteForumTopic"])

    def test_message_thread_not_found_counts_as_deleted_too(self) -> None:
        self.topic(GONE_THREAD, "gone")
        self.telegram.fail_next(PROBE_METHOD, 400, description="Bad Request: message thread not found")
        self.probe.run_once()
        self.assertIsNone(self.app.sessions.get(GONE_THREAD))

    def test_a_live_topic_answering_document_invalid_is_kept(self) -> None:
        self.topic(LIVE_THREAD, "live")
        self.telegram.fail_next(PROBE_METHOD, 400, description=ALIVE)
        self.probe.run_once()
        self.assertEqual(self.probe_threads(), [LIVE_THREAD])
        self.assertIsNotNone(self.app.sessions.get(LIVE_THREAD))
        self.assertFalse([line for line in self.logger.lines if "deleted" in line or "failed" in line])

    def test_every_probe_sends_the_probe_icon_and_never_a_stored_one(self) -> None:
        self.topic(GONE_THREAD, "gone")
        self.topic(LIVE_THREAD, "live")
        self.app.kv.set(f"icon:{LIVE_THREAD}", "5368324170671202286")  # an old key is never read any more
        self.probe.run_once()
        self.assertEqual(PROBE_ICON, "1")
        self.assertEqual(self.probe_calls(), [
            {"chat_id": CHAT, "message_thread_id": GONE_THREAD, "icon_custom_emoji_id": "1"},
            {"chat_id": CHAT, "message_thread_id": LIVE_THREAD, "icon_custom_emoji_id": "1"},
        ])

    def test_busy_and_claimed_topics_are_not_probed(self) -> None:
        for thread in (BUSY_RUNNING, BUSY_LINGER, BUSY_CLAIM, BUSY_MARK, LIVE_THREAD):
            self.topic(thread, f"t{thread}")
        feed = object()
        self.assertTrue(self.app.live.begin_turn(BUSY_RUNNING, object(), feed))
        self.assertTrue(self.app.live.begin_linger(BUSY_LINGER, object(), limit=3))
        self.assertTrue(take_claim(self.app.paths.view, BUSY_CLAIM, os.getpid(), self.clock.now()))
        self.assertTrue(begin_daemon_turn(self.app.paths.view, BUSY_MARK, self.clock))
        self.probe.run_once()
        self.assertEqual(self.probe_threads(), [LIVE_THREAD])

    def test_another_400_skips_only_that_topic_and_the_round_goes_on(self) -> None:
        self.topic(LIVE_THREAD, "other")
        self.topic(NEXT_THREAD, "gone")
        self.probe = self.make_probe(interval=600, client=RouteClient(self.telegram, {
            LIVE_THREAD: (400, "Bad Request: chat not found"),
            NEXT_THREAD: (400, DELETED),
        }))
        self.probe.run_once()
        self.assertEqual(self.probe_threads(), [LIVE_THREAD, NEXT_THREAD])
        self.assertIsNotNone(self.app.sessions.get(LIVE_THREAD))
        self.assertIsNone(self.app.sessions.get(NEXT_THREAD))
        self.assertEqual([line for line in self.logger.lines if "chat not found" in line],
                         [f"topic probe: thread {LIVE_THREAD} failed (editForumTopic failed (code 400): "
                          "Bad Request: chat not found); next topic probed"])

    def test_a_server_error_ends_the_round(self) -> None:
        self.topic(LIVE_THREAD, "first")
        self.topic(NEXT_THREAD, "gone")
        self.probe = self.make_probe(interval=600, client=RouteClient(self.telegram, {
            LIVE_THREAD: (500, "Internal Server Error"),
            NEXT_THREAD: (400, DELETED),
        }))
        self.probe.run_once()
        self.assertEqual(self.probe_threads(), [LIVE_THREAD])
        self.assertIsNotNone(self.app.sessions.get(NEXT_THREAD))
        self.assertEqual(len([line for line in self.logger.lines if "round skipped" in line]), 1)

    def test_a_connection_failure_ends_the_round(self) -> None:
        self.topic(LIVE_THREAD, "first")
        self.topic(NEXT_THREAD, "gone")
        self.probe = self.make_probe(interval=600, client=DownClient())
        self.probe.run_once()
        self.assertIsNotNone(self.app.sessions.get(NEXT_THREAD))
        self.assertEqual(len([line for line in self.logger.lines if "round skipped" in line]), 1)

    def test_an_unexpected_success_keeps_the_topic_and_is_logged_once(self) -> None:
        self.topic(LIVE_THREAD, "live")
        self.topic(NEXT_THREAD, "other")
        self.probe.run_once()
        self.probe.run_once()
        self.assertEqual(len(self.probe_calls()), 4)
        self.assertIsNotNone(self.app.sessions.get(LIVE_THREAD))
        self.assertEqual([line for line in self.logger.lines if "unexpected success" in line],
                         ["topic probe: unexpected success (the probe icon was accepted); topics kept"])

    def test_calls_are_spaced_one_second_apart(self) -> None:
        self.topic(GONE_THREAD, "gone")
        self.topic(LIVE_THREAD, "live")
        self.probe.run_once()
        self.assertEqual(self.clock.sleeps, [CALL_SPACING_SECONDS])

    def test_no_group_known_means_no_probe(self) -> None:
        self.topic(LIVE_THREAD, "live")
        self.app.kv.delete(GROUP_KEY)
        self.probe.run_once()
        self.assertEqual(self.probe_threads(), [])


class MassGoneTests(ProbeCase):
    """A round that finds most of its topics gone is taken as a wrong group: nothing is removed."""

    def gone_topics(self, threads: list[int]) -> None:
        for thread in threads:
            self.topic(thread, f"t{thread}")
            self.telegram.fail_next(PROBE_METHOD, 400, description=DELETED)

    def test_three_of_three_gone_removes_nothing_and_logs_the_guard(self) -> None:
        self.gone_topics([21, 22, 23])
        self.probe.run_once()
        self.assertEqual([self.app.sessions.get(t) is not None for t in (21, 22, 23)], [True, True, True])
        self.assertIn("probe: 3 of 3 topics look deleted — not removing (wrong group?)", self.logger.lines)
        self.assertFalse([c for c in self.telegram.calls if c.method == "deleteForumTopic"])

    def test_exactly_half_gone_is_still_removed(self) -> None:
        self.gone_topics([21, 22, 23])
        self.topic(24, "alive")
        self.topic(25, "alive2")
        self.topic(26, "alive3")
        self.probe.run_once()
        self.assertEqual([self.app.sessions.get(t) is not None for t in (21, 22, 23)], [False, False, False])
        self.assertFalse([line for line in self.logger.lines if "not removing" in line])

    def test_two_of_two_gone_is_removed_because_the_minimum_is_three(self) -> None:
        self.gone_topics([21, 22])
        self.probe.run_once()
        self.assertIsNone(self.app.sessions.get(21))
        self.assertIsNone(self.app.sessions.get(22))
        self.assertFalse([line for line in self.logger.lines if "not removing" in line])

    def test_three_of_five_gone_is_not_removed(self) -> None:
        self.gone_topics([21, 22, 23])
        self.topic(24, "alive")
        self.topic(25, "alive2")
        self.probe.run_once()
        self.assertIsNotNone(self.app.sessions.get(21))
        self.assertIn("probe: 3 of 5 topics look deleted — not removing (wrong group?)", self.logger.lines)


class ThreadTests(ProbeCase):
    def test_interval_zero_starts_no_thread_and_the_config_default_is_600(self) -> None:
        probe = self.make_probe(interval=0)
        probe.start()
        self.assertFalse(probe.started)
        self.assertEqual(_config(self.home.home).topic_probe_seconds(), 600)
        self.assertEqual(_config(self.home.home, topic_probe_seconds=0).topic_probe_seconds(), 0)

    def test_the_thread_runs_rounds_and_stops_on_shutdown(self) -> None:
        self.topic(LIVE_THREAD, "live")
        probe = self.make_probe(interval=0.01, first_seconds=0.01)
        probe.start()
        self.assertTrue(probe.started)
        deadline = time.monotonic() + 5
        while not self.probe_threads() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.probe_threads())
        probe.stop()
        self.assertFalse(probe._thread.is_alive())
        count = len(self.probe_threads())
        time.sleep(0.05)
        self.assertEqual(len(self.probe_threads()), count)

    def test_the_app_shutdown_stops_the_probe_thread(self) -> None:
        probe = self.app.probe
        self.assertIsInstance(probe, TopicProbe)
        self.app.shutdown_sessions()
        self.assertTrue(probe._stop.is_set())


if __name__ == "__main__":
    unittest.main()
