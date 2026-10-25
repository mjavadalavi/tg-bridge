"""titles: port of the title scenarios of fakecli/test_topic_session.py (7: default title after a failed call) and of
fakecli/test_workers.py (title worker: one job at a time, after the answer), plus the forced rename and the one-title
rule of the live title_session.

Everything runs on FakeTelegram and FakeClock inside a TempHome, with a real Database for the sessions table. The
backend is a recording stub with a one_shot method (the TitleService only calls that), so no launcher process starts.
"""

import os
import threading
import unittest

from tests.fakes.backend import FakeBackend
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.core.titles import TITLE_TIMEOUT_SECONDS, TitleService, default_title, title_prompt
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.messenger import Messenger

CHAT = -1001
THREAD = 9401
NAME = "side"
FIRST = "please check the invoice totals"
ANSWER = "The invoice totals match."


class StubTitleBackend:
    """Records each one_shot call. `reply` is returned, or `error` is raised when set."""

    def __init__(self, reply: str | None = "Invoice totals check", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, float]] = []
        self.threads: list[str] = []
        self._lock = threading.Lock()

    def one_shot(self, prompt: str, timeout: float) -> str | None:
        with self._lock:
            self.calls.append((prompt, timeout))
            self.threads.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        return self.reply


class TitleCase(unittest.TestCase):
    """A TempHome with a migrated Database, a sessions row per test topic, and a TitleService over the stub."""

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock(1000.0)
        self.tg = FakeTelegram(clock=self.clock)
        self.log_path = os.path.join(self._home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.db = Database(Paths.from_home(self.home))
        self.sessions = SessionRepo(self.db, self.clock)
        self.sessions.ensure(THREAD, NAME, "claude --alpha", self.home)
        self.pool = WorkerPool("title", 1, self.logger)
        self.backend = StubTitleBackend()
        self.service = self._service(self.backend)

    def tearDown(self) -> None:
        self.pool.join(5)
        self.db.close()
        self._home.__exit__(None, None, None)

    def _service(self, backend) -> TitleService:
        return TitleService(backend=backend, sessions=self.sessions, messenger=self.messenger, pool=self.pool,
                            logger=self.logger)

    def stored_title(self, thread_id: int = THREAD) -> str:
        row = self.sessions.get(thread_id)
        return row.title if row else ""

    def renames(self) -> list[str]:
        return [str(c.params["name"]) for c in self.tg.calls if c.method == "editForumTopic"]

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read()


class ModelTitleTests(TitleCase):
    def test_model_title_is_stored_as_name_and_title_and_renamed_forced(self) -> None:
        full = self.service.title(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertEqual(full, "side · Invoice totals check")
        self.assertEqual(self.stored_title(), "side · Invoice totals check")
        self.assertEqual(self.renames(), ["side · Invoice totals check"])
        call = [c for c in self.tg.calls if c.method == "editForumTopic"][0]
        self.assertEqual(call.params["chat_id"], CHAT)
        self.assertEqual(call.params["message_thread_id"], THREAD)

    def test_prompt_and_timeout_go_to_the_backend(self) -> None:
        long_msg = "x" * 400
        self.service.title(CHAT, THREAD, NAME, long_msg, ANSWER)
        prompt, timeout = self.backend.calls[0]
        self.assertEqual(prompt, title_prompt(long_msg, ANSWER))
        self.assertIn("x" * 300, prompt)
        self.assertNotIn("x" * 301, prompt)
        self.assertEqual(timeout, TITLE_TIMEOUT_SECONDS)
        self.assertEqual(TITLE_TIMEOUT_SECONDS, 120.0)

    def test_prompt_text_is_the_live_prompt(self) -> None:
        self.assertEqual(
            title_prompt("hi", "hello"),
            "Reply with ONLY a 3-6 word title for a conversation that starts with:\nhi\n\n"
            "and whose first answer begins:\nhello\n"
            "The title names the WORK (what the session is for). No quotes, no punctuation at the end.",
        )

    def test_a_topic_with_a_title_is_never_titled_again(self) -> None:
        self.assertIsNotNone(self.service.title(CHAT, THREAD, NAME, FIRST, ANSWER))
        self.assertIsNone(self.service.title(CHAT, THREAD, NAME, "second question", ANSWER))
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(len(self.renames()), 1)

    def test_an_existing_title_skips_the_call_and_the_rename(self) -> None:
        self.sessions.set_title(THREAD, "side · already set")
        self.assertIsNone(self.service.title(CHAT, THREAD, NAME, FIRST, ANSWER))
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.renames(), [])
        self.assertEqual(self.stored_title(), "side · already set")

    def test_a_missing_session_row_skips_the_call(self) -> None:
        self.assertIsNone(self.service.title(CHAT, 555, NAME, FIRST, ANSWER))
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.renames(), [])

    def test_forced_rename_ignores_the_fifteen_second_throttle(self) -> None:
        self.messenger.rename_topic(CHAT, THREAD, "side · earlier name")
        self.service.title(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertEqual(self.renames(), ["side · earlier name", "side · Invoice totals check"])


class DefaultTitleTests(TitleCase):
    def test_no_model_title_uses_the_first_message_in_one_line(self) -> None:
        """Ported labels:
        default title used
        title stored in sessions row
        """
        self.backend.reply = None
        self.service.title(CHAT, THREAD, NAME, "  hello   there\nworld  ", ANSWER)
        self.assertEqual(self.stored_title(), "side · hello there world")
        self.assertEqual(self.renames(), ["side · hello there world"])

    def test_default_title_is_cut_to_forty_characters(self) -> None:
        self.backend.reply = ""
        self.service.title(CHAT, THREAD, NAME, "a" * 60, ANSWER)
        self.assertEqual(self.stored_title(), "side · " + "a" * 40)

    def test_blank_first_message_falls_back_to_the_session_name(self) -> None:
        self.backend.reply = None
        self.service.title(CHAT, THREAD, NAME, "   ", ANSWER)
        self.assertEqual(self.stored_title(), "side · side")

    def test_a_raising_backend_gives_the_default_title_and_logs(self) -> None:
        self.backend.error = RuntimeError("launcher exploded")
        full = self.service.title(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertEqual(full, "side · please check the invoice totals")
        self.assertIn("title: the title call failed", self.log_text())

    def test_a_backend_without_one_shot_uses_the_default_title(self) -> None:
        service = self._service(FakeBackend())  # the AgentBackend default: one_shot returns None
        service.title(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertEqual(self.stored_title(), "side · please check the invoice totals")

    def test_default_title_helper(self) -> None:
        self.assertEqual(default_title("  one   two ", "N"), "one two")
        self.assertEqual(default_title("", "N"), "N")
        self.assertEqual(default_title("z" * 41, "N"), "z" * 40)


class WorkerTests(TitleCase):
    def test_submit_runs_the_title_on_the_title_worker_thread(self) -> None:
        """Ported labels:
            2a: title job finished"""
        self.service.submit(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertTrue(self.pool.join(5))
        self.assertEqual(self.stored_title(), "side · Invoice totals check")
        self.assertEqual(len(self.backend.threads), 1)
        self.assertNotEqual(self.backend.threads[0], threading.current_thread().name)
        self.assertTrue(self.backend.threads[0].startswith("tg-title-"))

    def test_jobs_run_one_at_a_time_and_each_topic_gets_its_own_title(self) -> None:
        other = 9402
        self.backend.reply = None  # default titles, so each topic shows its own first message
        self.sessions.ensure(other, "CHEAP", "claude --alpha", self.home)
        self.service.submit(CHAT, THREAD, NAME, "first topic", ANSWER)
        self.service.submit(CHAT, other, "CHEAP", "second topic", ANSWER)
        self.assertTrue(self.pool.join(5))
        self.assertEqual(self.stored_title(THREAD), "side · first topic")
        self.assertEqual(self.stored_title(other), "CHEAP · second topic")
        self.assertEqual(len(self.backend.calls), 2)

    def test_a_failing_job_is_logged_and_the_pool_keeps_working(self) -> None:
        self.sessions.ensure(9403, "BROKEN", "claude --alpha", self.home)
        self.pool.submit(lambda: 1 / 0)
        self.service.submit(CHAT, THREAD, NAME, FIRST, ANSWER)
        self.assertTrue(self.pool.join(5))
        self.assertEqual(self.stored_title(), "side · Invoice totals check")
        self.assertIn("title job failed", self.log_text())


if __name__ == "__main__":
    unittest.main()
