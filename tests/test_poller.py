"""Poller (STEP 51): port of the poll-loop, offset and outage scenarios of the live cmd_daemon, _poll_failed and _poll_ok
(tg-bridge.py:3533-3665), with the spool drain before each pass.

getUpdates answers come from a script (updates, or an error to raise), every other Bot API call goes to FakeTelegram.
Time moves only on FakeClock: each failed pass sleeps 15 s on it, so a 120 s outage is eight scripted failures.
"""

import unittest
from collections import deque
from typing import Any

from tests.fakes.clock import FakeClock
from tests.inbox_support import CHAT, GROUP_CHAT, THREAD, InboxHarness, message_update
from tests.support import TempHome
from tg_bridge.clock import utc_stamp
from tg_bridge.config import Config
from tg_bridge.daemon.poller import OUTAGE_NOTICE_SECONDS, POLL_RETRY_SECONDS, Poller
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.client import TelegramError

GROUP_ID = str(GROUP_CHAT["id"])
OTHER_THREAD = 9102


class ScriptedPoll:
    """A TelegramClient: getUpdates pops the next scripted answer (an update list, or an exception to raise).

    An empty script answers [] (a long poll with nothing new). Every other method is delegated to the FakeTelegram.
    `methods_before_poll` holds, for each getUpdates, the methods the FakeTelegram had already recorded, so an ordering
    test can see what went out before the long poll started.
    """

    def __init__(self, harness: InboxHarness) -> None:
        self._fake = harness.tg
        self.script: deque[Any] = deque()
        self.get_updates_params: list[dict[str, Any]] = []
        self.methods_before_poll: list[list[str]] = []

    def push(self, *answers: Any) -> None:
        self.script.extend(answers)

    def call(self, method: str, feed: bool = False, **params: Any) -> Any:
        if method != "getUpdates":
            return self._fake.call(method, feed=feed, **params)
        self.methods_before_poll.append([c.method for c in self._fake.calls])
        self.get_updates_params.append(dict(params))
        answer = self.script.popleft() if self.script else []
        if isinstance(answer, BaseException):
            raise answer
        return answer


def down() -> TelegramError:
    return TelegramError("getUpdates", None, "network down")


class PollCase(unittest.TestCase):
    """Builds the harness, a scripted client and a Poller under a TempHome for each test."""

    def setUp(self) -> None:
        self._home = TempHome()
        self._home.__enter__()
        self.h = InboxHarness(self._home)
        self.client = ScriptedPoll(self.h)
        self.config = Config({"poll_seconds": 50, "allow_user_ids": [42]}, home=self._home.home)
        self.poller = Poller(
            telegram=self.client,
            inbox=self.h.service,
            kv=self.h.kv,
            sessions=self.h.sessions,
            messenger=self.h.messenger,
            config=self.config,
            clock=self.h.clock,
            logger=self.h.logger,
        )

    def tearDown(self) -> None:
        self.h.close()
        self._home.__exit__(None, None, None)

    def _outage(self, polls: int) -> None:
        """Runs `polls` failing passes, each of which sleeps the retry pause on the clock."""
        self.client.push(*[down() for _ in range(polls)])
        for _ in range(polls):
            self.poller.poll_once()

    def _learn_group(self) -> None:
        self.h.kv.set("group_id", GROUP_ID)

    def _topic_active_at(self, thread: int, name: str, when: float) -> None:
        """A session row whose last_active is `when` (TurnRepo.insert stamps it)."""
        self.h.sessions.ensure(thread, name, "claude", "")
        stamp = utc_stamp(when)
        TurnRepo(self.h.db).insert(thread, started=stamp, ended=stamp, status="ok")


class PollLoopTests(PollCase):
    def test_first_poll_has_no_offset_and_uses_poll_seconds(self) -> None:
        """Without a stored offset the first getUpdates asks for everything, with the configured long-poll timeout."""
        self.poller.poll_once()
        self.assertEqual(self.client.get_updates_params, [{"timeout": 50}])

    def test_next_poll_asks_after_the_last_stored_update(self) -> None:
        """The offset is the last stored update_id, and the next call asks from one past it."""
        self.client.push([message_update(10, 1), message_update(11, 2)])
        self.poller.poll_once()
        self.poller.poll_once()
        self.assertEqual(self.client.get_updates_params[1], {"timeout": 50, "offset": 12})
        self.assertEqual(self.h.kv.get("offset"), "11")

    def test_each_update_processed_once_even_when_redelivered(self) -> None:
        """An update that comes back in a later batch is stored already, so it is not processed a second time."""
        self.client.push([message_update(10, 1), message_update(11, 2)], [message_update(11, 2), message_update(12, 3)])
        self.poller.poll_once()
        self.poller.poll_once()
        self.assertEqual(self.h.messages.seen, [(1, 10), (2, 11), (3, 12)])
        self.assertTrue(all(self.h.state(u) == "done" for u in (10, 11, 12)))

    def test_failed_poll_is_logged_paused_and_retried(self) -> None:
        """A failing getUpdates is logged as 'poll loop:', followed by the 15 s pause; the next pass delivers."""
        self.client.push(down(), [message_update(10, 1)])
        self.poller.poll_once()
        self.assertEqual(self.h.clock.sleeps, [POLL_RETRY_SECONDS])
        self.assertIn("poll loop: getUpdates failed", self.h.log_text())
        self.poller.poll_once()
        self.assertEqual(self.h.messages.seen, [(1, 10)])

    def test_stop_signal_during_poll_is_not_swallowed(self) -> None:
        """A SystemExit from a signal handler during the blocking call leaves the loop (no retry, no pause)."""
        self.client.push(SystemExit(0))
        with self.assertRaises(SystemExit):
            self.poller.poll_once()
        self.assertEqual(self.h.clock.sleeps, [])
        self.assertNotIn("poll loop", self.h.log_text())

    def test_run_keeps_polling_until_stop(self) -> None:
        """run() loops over poll_once: two batches are processed, then a stop signal ends the loop."""
        self.client.push([message_update(10, 1)], [message_update(11, 2)], SystemExit(0))
        with self.assertRaises(SystemExit):
            self.poller.run()
        self.assertEqual(self.h.messages.seen, [(1, 10), (2, 11)])

    def test_spool_drained_before_each_getupdates(self) -> None:
        """The spool folder is relayed first, then getUpdates is asked, within the same pass."""
        self._learn_group()
        self.h.sessions.ensure(THREAD, "side", "claude", "")
        with open(f"{self.h.paths.spool}/s.json", "w", encoding="utf-8") as handle:
            handle.write('{"topic": "side", "text": "relayed"}')
        self.poller.poll_once()
        self.assertEqual(self.client.methods_before_poll, [["sendMessage"]])
        self.assertEqual(self.h.tg.texts(THREAD), ["relayed"])


class OutageTests(PollCase):
    def test_three_failures_log_unreachable_and_recovery_logs_duration(self) -> None:
        """'telegram unreachable' is logged once, at the third failure; recovery logs the seconds since the first."""
        self._outage(3)
        self.assertEqual(self.h.log_text().count("telegram unreachable"), 1)
        self.client.push([])
        self.poller.poll_once()
        self.assertIn("telegram reachable again after 45s", self.h.log_text())

    def test_short_outage_is_not_announced(self) -> None:
        """An outage shorter than OUTAGE_NOTICE_SECONDS is logged but no message is sent to the topic.

        Ported labels:
            3f: short outage (<120 s) -> no notice
        """
        self._learn_group()
        self._topic_active_at(THREAD, "side", self.h.clock.now())
        self._outage(7)  # 105 s
        self.client.push([])
        self.poller.poll_once()
        self.assertIn("telegram reachable again after 105s", self.h.log_text())
        self.assertEqual(self.h.tg.texts(THREAD), [])

    def test_long_outage_notice_goes_to_most_recent_topic_once(self) -> None:
        """An outage of OUTAGE_NOTICE_SECONDS or more gets one notice, in the topic with the latest last_active.

        3c: the notice gives the outage window as HH:MM–HH:MM (Asia/Tehran) and says it is back now.

        Ported labels:
            3a: long outage -> exactly one unreachable notice (got 1)
            3b: notice went to the most recently active topic
            3c: notice text has HH:MM–HH:MM (Asia/Tehran) (got '⚠️ Telegram was unreachable 19:47–19:52 (Asia/Tehran), back now. Messages sent meanwhile are being processed.')
        """
        self._learn_group()
        start = self.h.clock.now()
        self._topic_active_at(THREAD, "side", start)
        self._topic_active_at(OTHER_THREAD, "B", start + 60)
        self.h.clock.advance(60)
        self._outage(OUTAGE_NOTICE_SECONDS // POLL_RETRY_SECONDS)  # exactly 120 s
        self.client.push([])
        self.poller.poll_once()
        self.assertEqual(self.h.tg.texts(THREAD), [])
        notices = self.h.tg.texts(OTHER_THREAD)
        self.assertEqual(len(notices), 1)
        self.assertTrue(notices[0].startswith("⚠️ Telegram was unreachable "))
        self.assertIn("(Asia/Tehran), back now.", notices[0])
        self.assertRegex(notices[0], r"unreachable \d\d:\d\d\u2013\d\d:\d\d \(Asia/Tehran\), back now\.")

    def test_each_long_outage_gets_its_own_notice(self) -> None:
        """Two separate long outages give two notices; the second outage starts its own clock."""
        self._learn_group()
        self._topic_active_at(THREAD, "side", self.h.clock.now())
        polls = OUTAGE_NOTICE_SECONDS // POLL_RETRY_SECONDS
        self.client.push(*([down() for _ in range(polls)] + [[]] + [down() for _ in range(polls)] + [[]]))
        for _ in range(2 * polls + 2):
            self.poller.poll_once()
        self.assertEqual(len(self.h.tg.texts(THREAD)), 2)

    def test_notice_skipped_when_no_group_learned(self) -> None:
        """Without a learned group the notice is skipped and logged; the outage still ends normally."""
        self._topic_active_at(THREAD, "side", self.h.clock.now())
        self._outage(OUTAGE_NOTICE_SECONDS // POLL_RETRY_SECONDS)
        self.client.push([])
        self.poller.poll_once()
        self.assertIn("outage notice skipped: no topic or group learned yet", self.h.log_text())
        self.assertEqual(self.h.tg.calls, ())

    def test_notice_skipped_when_no_topic_exists(self) -> None:
        """With a group but no topic at all, there is nowhere to post the notice; it is skipped and logged."""
        self._learn_group()
        self._outage(OUTAGE_NOTICE_SECONDS // POLL_RETRY_SECONDS)
        self.client.push([])
        self.poller.poll_once()
        self.assertIn("outage notice skipped", self.h.log_text())
        self.assertEqual(self.h.tg.calls, ())


    def test_a_success_resets_the_failure_counter(self) -> None:
        """3d: a success resets the failure count, so two failures after it do not add up to the three that log
        'telegram unreachable'.

        Ported labels:
            3d: counter reset after success
        """
        self._outage(2)
        self.client.push([])
        self.poller.poll_once()
        self._outage(2)
        self.assertEqual(self.h.log_text().count("telegram unreachable"), 0)

    def test_a_short_outage_still_resets_the_counter(self) -> None:
        """3g: a short outage (under the notice time) still resets the count: the next three failures log a second
        'telegram unreachable', and two do not.

        Ported labels:
            3g: short outage still resets the counter
        """
        self._outage(3)
        self.assertEqual(self.h.log_text().count("telegram unreachable"), 1)
        self.client.push([])
        self.poller.poll_once()
        self._outage(2)
        self.assertEqual(self.h.log_text().count("telegram unreachable"), 1)

    def test_a_later_success_after_the_notice_sends_nothing(self) -> None:
        """3e: once the outage notice went out, a later successful pass sends nothing more.

        Ported labels:
            3e: a later success sends nothing
        """
        self._learn_group()
        self._topic_active_at(THREAD, "side", self.h.clock.now())
        self._outage(OUTAGE_NOTICE_SECONDS // POLL_RETRY_SECONDS)
        self.client.push([], [])
        self.poller.poll_once()
        self.assertEqual(len(self.h.tg.texts(THREAD)), 1)
        self.poller.poll_once()
        self.assertEqual(len(self.h.tg.texts(THREAD)), 1)

    def test_fewer_than_three_failures_send_no_notice_even_when_the_outage_is_long(self) -> None:
        """3h: two failed passes are below the three that count as an outage, so no notice, even when the clock
        says the outage lasted more than the notice time.

        Ported labels:
            3h: fewer than 3 failures -> no notice even when long
        """
        self._learn_group()
        self._topic_active_at(THREAD, "side", self.h.clock.now())
        self._outage(2)
        self.h.clock.advance(OUTAGE_NOTICE_SECONDS * 2)
        self.client.push([])
        self.poller.poll_once()
        self.assertEqual(self.h.tg.texts(THREAD), [])
        self.assertNotIn("outage notice", self.h.log_text())

if __name__ == "__main__":
    unittest.main()
