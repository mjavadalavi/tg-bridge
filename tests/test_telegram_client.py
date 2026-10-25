"""HttpTelegramClient and FakeTelegram: port of the C04 call behaviour against a local http.server stub.

Scenarios: a 429 waits retry_after on the FakeClock and then succeeds; five 429s are retried and the sixth gives up
into the three-attempt rule; a 400 raises TelegramBadRequest at once; other failures are tried three times five
seconds apart; the token is re-read on every call and never reaches bridge.log; a feed edit takes the feed's share
of the send budget. The last class self-tests FakeTelegram, the double every later step uses.
"""

import json
import os
import threading
import unittest
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.client import (
    FLOOD_RETRIES,
    HttpTelegramClient,
    TelegramBadRequest,
    TelegramError,
)
from tg_bridge.telegram.proxy import OpenerCache
from tg_bridge.telegram.ratelimit import RateGate

FAKE_TOKEN = "123456789:AAEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K1x"
OTHER_TOKEN = "987654321:BBEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K2y"
OK_BODY = {"ok": True, "result": {"message_id": 7, "chat": {"id": -1}}}
RETRY_PAUSE = 5.0


def _flood(retry_after: int | None) -> tuple[int, dict[str, Any]]:
    body: dict[str, Any] = {"ok": False, "error_code": 429, "description": "Too Many Requests"}
    if retry_after is not None:
        body["parameters"] = {"retry_after": retry_after}
    return 429, body


def _failure(code: int, description: str = "boom") -> tuple[int, dict[str, Any]]:
    return code, {"ok": False, "error_code": code, "description": description}


class _StubHandler(BaseHTTPRequestHandler):
    """Answers every request from the server's script, one scripted response per request."""

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._answer()

    def _answer(self) -> None:
        server = self.server
        assert isinstance(server, StubTelegramServer)
        server.paths.append(self.path)
        status, body = server.next_response()
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log."""


class StubTelegramServer(ThreadingHTTPServer):
    """A loopback Bot API stub. Responses are scripted FIFO; once the script runs out it answers 200 OK."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.paths: list[str] = []
        self._script: deque[tuple[int, dict[str, Any]]] = deque()
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def script(self, *responses: tuple[int, dict[str, Any]]) -> None:
        with self._lock:
            self._script.extend(responses)

    def next_response(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return self._script.popleft() if self._script else (200, OK_BODY)


class _DirectOpeners(OpenerCache):
    """An OpenerCache with no proxy at all, so the loopback stub is reached directly whatever the environment says."""

    @staticmethod
    def _build(proxy: str | None) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _ClientCase(unittest.TestCase):
    """Builds a client against a stub server, with a temp HOME holding the token file."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        self._write_token(FAKE_TOKEN)
        self.clock = FakeClock()
        self.logger = Logger(self.paths.log)
        self.rate = RateGate(self.clock)
        self.server = StubTelegramServer()
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        self.addCleanup(self._stop_server, thread)
        self.client = self.make_client(self.clock, self.rate)

    def _stop_server(self, thread: threading.Thread) -> None:
        self.server.shutdown()
        self.server.server_close()
        thread.join(timeout=5)

    def make_client(self, clock: Clock, rate: RateGate) -> HttpTelegramClient:
        return HttpTelegramClient(
            self.paths,
            clock,
            self.logger,
            rate,
            _DirectOpeners(lambda: None),
            api_base=self.server.url,
        )

    def _write_token(self, token: str) -> None:
        fd = os.open(self.paths.token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")

    def log_text(self) -> str:
        if not os.path.exists(self.paths.log):
            return ""
        with open(self.paths.log, encoding="utf-8") as f:
            return f.read()


class RetryAfterTests(_ClientCase):
    def test_429_on_an_edit_waits_retry_after_then_retries(self) -> None:
        """Ported labels:
        L: an edit waits the 429 retry_after (1 s) and retries
        """
        self.server.script(_flood(1))
        self.client.call("editMessageText", chat_id=-1, message_id=5, text="x")
        self.assertEqual(len(self.server.paths), 2)
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_429_then_success_honours_retry_after_on_the_clock(self) -> None:
        """Ported labels:
            A: 429 retried once, then ok (2 calls)
            A: slept the server retry_after 30 s ([30])"""
        self.server.script(_flood(1))
        result = self.client.call("sendMessage", chat_id=-1, message_thread_id=2, text="hi")
        self.assertEqual(result, OK_BODY["result"])
        self.assertEqual(len(self.server.paths), 2)
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_missing_retry_after_waits_the_default_flood_wait(self) -> None:
        self.server.script(_flood(None))
        self.client.call("sendMessage", chat_id=-1, text="hi")
        self.assertEqual(self.clock.sleeps, [30.0])

    def test_flood_retries_give_up_after_five_waits_then_follow_the_three_attempt_rule(self) -> None:
        self.server.script(*[_flood(1)] * (FLOOD_RETRIES + 3))
        with self.assertRaises(TelegramError) as ctx:
            self.client.call("sendMessage", chat_id=-1, text="hi")
        self.assertEqual(ctx.exception.code, 429)
        self.assertEqual(len(self.server.paths), FLOOD_RETRIES + 3)
        self.assertEqual(self.clock.sleeps, [1.0] * FLOOD_RETRIES + [RETRY_PAUSE, RETRY_PAUSE])


class BadRequestTests(_ClientCase):
    def test_400_raises_bad_request_at_once(self) -> None:
        """Ported labels:
            H: one attempt, no sleep, 400 raised (1, [])"""
        self.server.script(_failure(400, "Bad Request: can't parse entities"))
        with self.assertRaises(TelegramBadRequest) as ctx:
            self.client.call("sendMessage", chat_id=-1, text="<b>")
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("can't parse entities", ctx.exception.description)
        self.assertEqual(len(self.server.paths), 1)
        self.assertEqual(self.clock.sleeps, [])

    def test_bad_request_is_still_a_telegram_error(self) -> None:
        self.assertTrue(issubclass(TelegramBadRequest, TelegramError))


class OtherFailureTests(_ClientCase):
    def test_three_failures_raise_after_two_five_second_pauses(self) -> None:
        self.server.script(_failure(500), _failure(500), _failure(500))
        with self.assertRaises(TelegramError) as ctx:
            self.client.call("getUpdates", offset=0)
        self.assertEqual(ctx.exception.code, 500)
        self.assertEqual(len(self.server.paths), 3)
        self.assertEqual(self.clock.sleeps, [RETRY_PAUSE, RETRY_PAUSE])

    def test_one_failure_then_success_returns_the_result(self) -> None:
        self.server.script(_failure(500))
        result = self.client.call("getUpdates", offset=0)
        self.assertEqual(result, OK_BODY["result"])
        self.assertEqual(self.clock.sleeps, [RETRY_PAUSE])

    def test_ok_false_on_http_200_counts_as_a_failure(self) -> None:
        self.server.script((200, {"ok": False, "description": "Forbidden"}), (200, {"ok": False}), (200, {"ok": False}))
        with self.assertRaises(TelegramError) as ctx:
            self.client.call("getMe")
        self.assertIsNone(ctx.exception.code)
        self.assertEqual(len(self.server.paths), 3)

    def test_connection_refused_is_retried_then_raised(self) -> None:
        dead = StubTelegramServer()
        port = dead.server_address[1]
        dead.server_close()
        client = HttpTelegramClient(
            self.paths, self.clock, self.logger, self.rate, _DirectOpeners(lambda: None),
            api_base=f"http://127.0.0.1:{port}",
        )
        with self.assertRaises(TelegramError):
            client.call("getMe")
        self.assertEqual(self.clock.sleeps, [RETRY_PAUSE, RETRY_PAUSE])


class TokenTests(_ClientCase):
    def test_token_is_re_read_on_every_call(self) -> None:
        self.client.call("getMe")
        self._write_token(OTHER_TOKEN)
        self.client.call("getMe")
        self.assertEqual(self.server.paths[0], f"/bot{FAKE_TOKEN}/getMe")
        self.assertEqual(self.server.paths[1], f"/bot{OTHER_TOKEN}/getMe")

    def test_missing_token_file_raises_before_any_request(self) -> None:
        os.remove(self.paths.token)
        with self.assertRaises(OSError):
            self.client.call("getMe")
        self.assertEqual(self.server.paths, [])

    def test_token_never_reaches_the_log(self) -> None:
        self.server.script(_flood(1), _failure(500), _failure(500), _failure(500))
        with self.assertRaises(TelegramError):
            self.client.call("sendMessage", chat_id=-1, text="hi")
        self.client.call("getMe")
        log = self.log_text()
        self.assertIn("flood wait", log)
        self.assertIn("attempt", log)
        self.assertNotIn(FAKE_TOKEN, log)
        self.assertNotIn(FAKE_TOKEN.split(":", 1)[1], log)


class FeedShareTests(_ClientCase):
    def test_feed_edit_takes_a_slot_from_the_feed_share(self) -> None:
        """
        Ported labels:
        T: a feed edit takes one feed slot
        T: a feed edit waits when the feed share is used up
        """
        rate = RateGate(self.clock, group_per_min=2, feed_per_min=1)
        client = self.make_client(self.clock, rate)
        client.call("editMessageText", feed=True, chat_id=-1, message_id=5, text="a")
        client.call("editMessageText", feed=True, chat_id=-1, message_id=5, text="b")
        self.assertEqual(self.clock.sleeps, [60.0])

    def test_plain_edit_and_other_methods_take_no_slot(self) -> None:
        """
        Ported labels:
        T: a non-feed edit takes no slot
        """
        rate = RateGate(self.clock, group_per_min=1, feed_per_min=1)
        client = self.make_client(self.clock, rate)
        for _ in range(3):
            client.call("editMessageText", chat_id=-1, message_id=5, text="a")
            client.call("getMe")
        self.assertEqual(self.clock.sleeps, [])

    def test_send_message_takes_a_group_slot(self) -> None:
        rate = RateGate(self.clock, group_per_min=1, feed_per_min=1)
        client = self.make_client(self.clock, rate)
        client.call("sendMessage", chat_id=-1, text="a")
        client.call("sendMessage", chat_id=-1, text="b")
        self.assertEqual(self.clock.sleeps, [60.0])


class FakeTelegramTests(unittest.TestCase):
    """Self-tests of the double: what it records, returns and scripts."""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.fake = FakeTelegram(self.clock)

    def test_send_returns_incrementing_message_ids(self) -> None:
        first = self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="one")
        second = self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="two")
        self.assertEqual(second["message_id"], first["message_id"] + 1)

    def test_calls_record_method_params_and_feed_flag(self) -> None:
        self.fake.call("editMessageText", feed=True, chat_id=-1, message_id=9, text="x")
        self.fake.call("getMe")
        self.assertEqual(
            [(c.method, c.params, c.feed) for c in self.fake.calls],
            [("editMessageText", {"chat_id": -1, "message_id": 9, "text": "x"}, True), ("getMe", {}, False)],
        )

    def test_texts_and_buttons_filter_by_thread(self) -> None:
        markup = json.dumps({"inline_keyboard": [[{"text": "Yes", "callback_data": "pa:1"}]]})
        self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="card", reply_markup=markup)
        self.fake.call("sendMessage", chat_id=-1, message_thread_id=4, text="elsewhere")
        self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="plain")
        self.assertEqual(self.fake.texts(3), ["card", "plain"])
        self.assertEqual(self.fake.texts("4"), ["elsewhere"])
        self.assertEqual(self.fake.buttons(3), [[[{"text": "Yes", "callback_data": "pa:1"}]]])
        self.assertEqual(self.fake.buttons(4), [])

    def test_edits_are_kept_per_message(self) -> None:
        sent = self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="draft")
        self.fake.call("editMessageText", chat_id=-1, message_id=sent["message_id"], text="v1")
        self.fake.call("editMessageText", chat_id=-1, message_id=sent["message_id"], text="v2")
        self.assertEqual(self.fake.edits(sent["message_id"]), ["v1", "v2"])
        self.assertEqual(self.fake.edits(1), [])

    def test_scripted_429_is_absorbed_and_waits_on_the_clock(self) -> None:
        self.fake.fail_next("sendMessage", code=429, retry_after=1)
        result = self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="hi")
        self.assertIn("message_id", result)
        self.assertEqual(self.fake.flood_waits, (("sendMessage", 1),))
        self.assertEqual(self.clock.sleeps, [1.0])
        self.assertEqual(len(self.fake.calls), 1)

    def test_scripted_400_raises_and_is_recorded(self) -> None:
        self.fake.fail_next("sendMessage", code=400, description="Bad Request: can't parse entities")
        with self.assertRaises(TelegramBadRequest):
            self.fake.call("sendMessage", chat_id=-1, message_thread_id=3, text="<b>")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.fake.texts(3), ["<b>"])

    def test_other_scripted_codes_raise_telegram_error(self) -> None:
        self.fake.fail_next("getUpdates", code=500)
        with self.assertRaises(TelegramError) as ctx:
            self.fake.call("getUpdates", offset=0)
        self.assertEqual(ctx.exception.code, 500)

    def test_scripted_failures_queue_in_order_per_method(self) -> None:
        self.fake.fail_next("sendMessage", code=400)
        self.fake.fail_next("sendMessage", code=429, retry_after=2)
        with self.assertRaises(TelegramBadRequest):
            self.fake.call("sendMessage", chat_id=-1, text="a")
        self.fake.call("sendMessage", chat_id=-1, text="b")
        self.assertEqual(self.fake.flood_waits, (("sendMessage", 2),))
        self.fake.call("sendMessage", chat_id=-1, text="c")
        self.assertEqual(self.fake.flood_waits, (("sendMessage", 2),))


if __name__ == "__main__":
    unittest.main()
