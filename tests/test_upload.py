"""upload: the multipart body, send_doc_bytes and send_file (C23), against a loopback Bot API stub.

Uploads never go through FakeTelegram (a multipart body cannot pass through TelegramClient.call), so these tests send
real HTTP requests to a ThreadingHTTPServer on 127.0.0.1 and read back the path, content type and body of each one.
Scenarios: the photo 400 falls back to a document (the old send_file rule), a file over 50 MB is refused before any
request, a missing file is refused, captions are capped at 1024 characters, the token is re-read per request and never
appears in an exception or in bridge.log.
"""

import json
import os
import re
import threading
import unittest
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.support import TempHome
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.client import TelegramError
from tg_bridge.telegram.proxy import OpenerCache
from tg_bridge.telegram.upload import (
    CAPTION_LIMIT,
    UPLOAD_LIMIT_BYTES,
    UploadRefused,
    Uploader,
    build_multipart,
    mime_for,
)

FAKE_TOKEN = "123456789:AAEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K1x"
OTHER_TOKEN = "987654321:BBEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K2y"
OK_BODY: dict[str, Any] = {"ok": True, "result": {"message_id": 5, "chat": {"id": -1}}}
CHAT = -1001
THREAD = 9301


class _UploadHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        server = self.server
        assert isinstance(server, UploadStub)
        server.requests.append((self.path, self.headers.get("Content-Type", ""), body))
        status, payload = server.next_response()
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log."""


class UploadStub(ThreadingHTTPServer):
    """A loopback Bot API stub: records every POST, answers scripted responses FIFO, then 200 OK."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _UploadHandler)
        self.requests: list[tuple[str, str, bytes]] = []
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
    """An OpenerCache with no proxy, so the loopback stub is reached whatever the environment says."""

    @staticmethod
    def _build(proxy: str | None) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _failure(code: int, description: str) -> tuple[int, dict[str, Any]]:
    return code, {"ok": False, "error_code": code, "description": description}


class UploadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        os.makedirs(self.paths.state, exist_ok=True)
        self._write_token(FAKE_TOKEN)
        self.logger = Logger(self.paths.log)
        self.server = UploadStub()
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        self.addCleanup(self._stop, thread)
        self.uploader = Uploader(self.paths, _DirectOpeners(lambda: None), self.logger, api_base=self.server.url)

    def _stop(self, thread: threading.Thread) -> None:
        self.server.shutdown()
        self.server.server_close()
        thread.join(5)

    def _write_token(self, token: str) -> None:
        with open(self.paths.token, "w", encoding="utf-8") as f:
            f.write(token + "\n")

    def _log(self) -> str:
        if not os.path.exists(self.paths.log):
            return ""
        with open(self.paths.log, encoding="utf-8") as f:
            return f.read()

    def _write_file(self, name: str, data: bytes) -> str:
        path = os.path.join(self.home.home, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _paths_seen(self) -> list[str]:
        return [path for path, _, _ in self.server.requests]


class MultipartBodyTests(unittest.TestCase):
    """The pure part: build_multipart lays out the form fields, then the file, then the closing boundary."""

    def test_fields_then_file_then_closing_boundary(self) -> None:
        """the body carries every field, the file bytes unchanged, and ends with the closing boundary"""
        body = build_multipart(
            "BND",
            [("chat_id", "-1001"), ("message_thread_id", "9301"), ("caption", "hi there")],
            "document",
            "a.md",
            "text/markdown",
            b"# raw\r\nbytes",
        )
        text = body.decode("utf-8", errors="replace")
        self.assertIn('name="chat_id"\r\n\r\n-1001\r\n', text)
        self.assertIn('name="message_thread_id"\r\n\r\n9301\r\n', text)
        self.assertIn('name="caption"\r\n\r\nhi there\r\n', text)
        self.assertIn('name="document"; filename="a.md"\r\nContent-Type: text/markdown\r\n\r\n', text)
        self.assertIn(b"# raw\r\nbytes", body)
        self.assertTrue(body.endswith(b"\r\n--BND--\r\n"))
        self.assertLess(text.index('name="caption"'), text.index('name="document"'))

    def test_quotes_and_line_breaks_in_a_file_name_are_replaced(self) -> None:
        """a file name cannot break out of its quoted header"""
        body = build_multipart("B", [], "document", 'x"y\r\nz.txt', "text/plain", b"")
        self.assertIn(b'filename="x_y__z.txt"', body)

    def test_mime_by_extension(self) -> None:
        """the media types the live file used, and octet-stream for everything else"""
        self.assertEqual(mime_for("/a/b.PNG"), "image/png")
        self.assertEqual(mime_for("/a/b.jpeg"), "image/jpeg")
        self.assertEqual(mime_for("/a/b.webp"), "image/webp")
        self.assertEqual(mime_for("/a/b.txt"), "application/octet-stream")
        self.assertEqual(mime_for("/a/noext"), "application/octet-stream")


class SendDocBytesTests(UploadTestCase):
    def test_posts_one_multipart_request_to_the_method_and_returns_the_result(self) -> None:
        """one POST to /bot<token>/sendDocument with a multipart boundary; the result dict comes back"""
        result = self.uploader.send_doc_bytes(CHAT, THREAD, b"payload", "a.md", "text/markdown", caption="cap")
        self.assertEqual(result, OK_BODY["result"])
        self.assertEqual(len(self.server.requests), 1)
        path, content_type, body = self.server.requests[0]
        self.assertEqual(path, f"/bot{FAKE_TOKEN}/sendDocument")
        match = re.fullmatch(r"multipart/form-data; boundary=(----tgbridge[0-9a-f]{16})", content_type)
        self.assertIsNotNone(match, content_type)
        self.assertIn(b"payload", body)
        self.assertIn(b'name="caption"\r\n\r\ncap\r\n', body)
        self.assertIn(b'name="message_thread_id"\r\n\r\n9301\r\n', body)

    def test_caption_is_cut_to_the_telegram_limit(self) -> None:
        """a caption longer than 1024 characters is sent cut to 1024"""
        self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown", caption="c" * 2000)
        body = self.server.requests[0][2]
        self.assertIn(b"c" * CAPTION_LIMIT + b"\r\n", body)
        self.assertNotIn(b"c" * (CAPTION_LIMIT + 1), body)

    def test_no_caption_field_when_caption_is_empty(self) -> None:
        """an empty caption is left out of the form"""
        self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")
        self.assertNotIn(b'name="caption"', self.server.requests[0][2])

    def test_telegram_refusal_raises_telegram_error_without_the_token(self) -> None:
        """ok:false and HTTP 400 both raise TelegramError with the description; the token never appears"""
        self.server.script(_failure(400, "Bad Request: can't parse"))
        with self.assertRaises(TelegramError) as ctx:
            self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("can't parse", ctx.exception.description)
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_ok_false_on_http_200_raises(self) -> None:
        """a 200 answer with ok:false is a failure too"""
        self.server.script((200, {"ok": False, "description": "chat not found"}))
        with self.assertRaisesRegex(TelegramError, "chat not found"):
            self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")

    def test_token_is_read_again_for_every_request(self) -> None:
        """a rotated token takes effect on the next upload, without a restart"""
        self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")
        self._write_token(OTHER_TOKEN)
        self.uploader.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")
        self.assertEqual(self._paths_seen(), [f"/bot{FAKE_TOKEN}/sendDocument", f"/bot{OTHER_TOKEN}/sendDocument"])

    def test_network_failure_does_not_show_the_token(self) -> None:
        """a connection that cannot be made raises, and its text carries no token"""
        dead = Uploader(self.paths, _DirectOpeners(lambda: None), self.logger, api_base="http://127.0.0.1:1")
        with self.assertRaises(urllib.error.URLError) as ctx:
            dead.send_doc_bytes(CHAT, THREAD, b"x", "a.md", "text/markdown")
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))


class SendFileTests(UploadTestCase):
    def test_image_goes_out_as_a_photo(self) -> None:
        """a .png is sent with sendPhoto and the photo field, as image/png"""
        path = self._write_file("pic.png", b"\x89PNG data")
        self.uploader.send_file(CHAT, THREAD, path, caption="look")
        self.assertEqual(self._paths_seen(), [f"/bot{FAKE_TOKEN}/sendPhoto"])
        body = self.server.requests[0][2]
        self.assertIn(b'name="photo"; filename="pic.png"\r\nContent-Type: image/png', body)
        self.assertIn(b"\x89PNG data", body)
        self.assertIn(b'name="caption"\r\n\r\nlook\r\n', body)

    def test_refused_photo_is_sent_again_as_a_document(self) -> None:
        """photo 400 -> the same file goes as a document, and the refusal is logged"""
        path = self._write_file("pic.jpg", b"jpegbytes")
        self.server.script(_failure(400, "Bad Request: PHOTO_INVALID_DIMENSIONS"))
        self.uploader.send_file(CHAT, THREAD, path)
        self.assertEqual(
            self._paths_seen(), [f"/bot{FAKE_TOKEN}/sendPhoto", f"/bot{FAKE_TOKEN}/sendDocument"]
        )
        self.assertIn(b'name="document"; filename="pic.jpg"', self.server.requests[1][2])
        log = self._log()
        self.assertIn("sendPhoto failed, retrying as document: sendPhoto failed (code 400)", log)
        self.assertNotIn(FAKE_TOKEN, log)

    def test_failing_document_after_a_refused_photo_propagates(self) -> None:
        """if the document fallback fails too, the error reaches the caller"""
        path = self._write_file("pic.gif", b"gif")
        self.server.script(_failure(400, "photo refused"), _failure(413, "too large"))
        with self.assertRaises(TelegramError):
            self.uploader.send_file(CHAT, THREAD, path)

    def test_other_file_goes_out_as_a_document_once(self) -> None:
        """a non-image is sent with sendDocument only, as application/octet-stream"""
        path = self._write_file("notes.txt", b"plain")
        self.uploader.send_file(CHAT, THREAD, path)
        self.assertEqual(self._paths_seen(), [f"/bot{FAKE_TOKEN}/sendDocument"])
        self.assertIn(b"Content-Type: application/octet-stream", self.server.requests[0][2])

    def test_file_over_fifty_megabytes_is_refused_before_any_request(self) -> None:
        """a file larger than the 50 MB Bot API cap is refused, and nothing is sent"""
        path = os.path.join(self.home.home, "huge.bin")
        with open(path, "wb") as f:
            f.truncate(UPLOAD_LIMIT_BYTES + 1)  # a sparse file: no real 50 MB is written
        with self.assertRaisesRegex(UploadRefused, "50 MB"):
            self.uploader.send_file(CHAT, THREAD, path)
        self.assertEqual(self.server.requests, [])

    def test_missing_file_is_refused_before_any_request(self) -> None:
        """a path that is not a file is refused, and nothing is sent"""
        with self.assertRaisesRegex(UploadRefused, "no such file"):
            self.uploader.send_file(CHAT, THREAD, os.path.join(self.home.home, "absent.png"))
        self.assertEqual(self.server.requests, [])


if __name__ == "__main__":
    unittest.main()
