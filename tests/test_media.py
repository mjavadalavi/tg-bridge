"""media: MediaService and SonioxTranscriber (C05, C17) against a loopback stub and FakeTelegram. No network.

Ports the media scenarios of the old fakecli suite as the media service owns them: a photo is saved 0600 and handed
over as an image or as the old text note, a voice note is transcribed and echoed (600 characters), an empty transcript,
a missing Soniox key and a failed transcription each produce the old topic text, and a failed download never leaks
the bot token. The dispatcher and queue scenarios of test_restart_media (placeholder timeout, late result queued at
the back) belong to STEP 44 and STEP 50, not to this module, so they are not ported here.

Downloads and Soniox traffic hit a ThreadingHTTPServer on 127.0.0.1 that records every request and answers scripted
responses; getFile and the topic messages go through FakeTelegram. The Soniox poll pauses use FakeClock.
"""

import contextlib
import json
import os
import stat
import threading
import unittest
import urllib.request
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.core.media import (
    MSG_EMPTY,
    MSG_NO_KEY,
    POLL_LIMIT,
    MediaOutcome,
    MediaService,
    SonioxTranscriber,
    Transcriber,
    TranscriberUnavailable,
    TranscriptionFailed,
)
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.messenger import Messenger
from tg_bridge.telegram.proxy import OpenerCache

FAKE_TOKEN = "123456789:AAEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K1x"
FAKE_KEY = "sk-test-soniox-key-0123456789"
CHAT = -1001
THREAD = 9301
PHOTO_BYTES = b"\x89PNG-fake-photo-bytes"
VOICE_BYTES = b"OggS-fake-voice-bytes"
PHOTO_PATH = "photos/file_7.jpg"
VOICE_PATH = "voice/file_9.oga"


@dataclass(frozen=True)
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


def j(payload: Any) -> bytes:
    return json.dumps(payload).encode()


class _StubHandler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        server = self.server
        assert isinstance(server, MediaStub)
        server.requests.append(Recorded(self.command, self.path, dict(self.headers.items()), body))
        status, payload = server.respond(self.command, self.path)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log."""


class MediaStub(ThreadingHTTPServer):
    """Records every request and answers scripted (status, body) pairs per method and path, then repeats the last one.

    A path without a script answers 404, so a test that forgot a route fails loudly.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.requests: list[Recorded] = []
        self._routes: dict[tuple[str, str], deque[tuple[int, bytes]]] = {}
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def route(self, method: str, path: str, *responses: tuple[int, bytes]) -> None:
        with self._lock:
            self._routes.setdefault((method, path), deque()).extend(responses)

    def respond(self, method: str, path: str) -> tuple[int, bytes]:
        with self._lock:
            queue = self._routes.get((method, path))
            if not queue:
                return 404, j({"ok": False, "description": "not scripted"})
            return queue.popleft() if len(queue) > 1 else queue[0]

    def calls(self, method: str, path: str) -> list[Recorded]:
        with self._lock:
            return [r for r in self.requests if r.method == method and r.path == path]


@contextlib.contextmanager
def running_stub() -> Iterator[MediaStub]:
    server = MediaStub()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class _DirectOpeners(OpenerCache):
    """An OpenerCache with no proxy, so the loopback stub is reached directly whatever the environment says."""

    @staticmethod
    def _build(proxy: str | None) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class StubTranscriber:
    """A Transcriber that returns a scripted transcript, or raises a scripted error, and records the paths it saw."""

    def __init__(self, text: str = "", error: BaseException | None = None) -> None:
        self.text = text
        self.error = error
        self.paths: list[str] = []

    def transcribe(self, path: str) -> str:
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        return self.text


def photo_message(sizes: tuple[str, ...] = ("small-id", "big-id")) -> dict[str, Any]:
    return {
        "chat": {"id": CHAT, "type": "supergroup"},
        "message_thread_id": THREAD,
        "photo": [{"file_id": file_id, "width": 90 * (i + 1)} for i, file_id in enumerate(sizes)],
    }


def voice_message(field_name: str = "voice") -> dict[str, Any]:
    return {
        "chat": {"id": CHAT, "type": "supergroup"},
        "message_thread_id": THREAD,
        field_name: {"file_id": "voice-id", "duration": 3},
    }


class _MediaCase(unittest.TestCase):
    """A temp HOME with the token file, a stub server, FakeTelegram for getFile and topic texts, and a MediaService."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        with open(self.paths.token, "w", encoding="utf-8") as f:
            f.write(FAKE_TOKEN + "\n")
        self.clock = FakeClock()
        self.stub = self.enterContext(running_stub())
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(
            self.telegram, Logger(os.path.join(self.paths.state, "bridge.log")), self.clock
        )
        self.openers = _DirectOpeners(lambda: None)
        self.transcriber = StubTranscriber()
        self.media = self._service(self.transcriber)

    def _service(self, transcriber: Transcriber) -> MediaService:
        return MediaService(
            self.paths,
            self.telegram,
            self.messenger,
            self.openers,
            transcriber,
            self.clock,
            api_base=self.stub.url,
        )

    def serve(self, file_path: str, data: bytes) -> None:
        """Make getFile answer file_path and the stub serve data at that Telegram file URL."""
        self.telegram.script_result("getFile", {"file_path": file_path})
        self.stub.route("GET", f"/file/bot{FAKE_TOKEN}/{file_path}", (200, data))

    def inbox_files(self) -> list[str]:
        if not os.path.isdir(self.paths.inbox):
            return []
        return sorted(os.listdir(self.paths.inbox))

    def topic_texts(self) -> list[str]:
        return self.telegram.texts(THREAD)

    def write_soniox_key(self) -> None:
        with open(self.paths.soniox_key, "w", encoding="utf-8") as f:
            f.write(FAKE_KEY + "\n")


class PhotoTests(_MediaCase):
    def test_photo_saved_mode_0600_and_handed_over_as_note_for_text_backend(self) -> None:
        self.serve(PHOTO_PATH, PHOTO_BYTES)
        outcome = self.media.fetch(photo_message(), "what is this?", native_images=False)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.images, ())
        files = self.inbox_files()
        self.assertEqual(len(files), 1)
        dest = os.path.join(self.paths.inbox, files[0])
        self.assertEqual(outcome.text, f"what is this?\n[photo saved for you at {dest}]")
        self.assertEqual(stat.S_IMODE(os.stat(dest).st_mode), 0o600)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), PHOTO_BYTES)

    def test_biggest_photo_size_is_the_one_downloaded(self) -> None:
        self.serve(PHOTO_PATH, PHOTO_BYTES)
        self.media.fetch(photo_message(("small-id", "mid-id", "big-id")), "", native_images=False)
        requested = [c.params["file_id"] for c in self.telegram.calls if c.method == "getFile"]
        self.assertEqual(requested, ["big-id"])

    def test_photo_with_native_images_passes_the_path_and_no_note(self) -> None:
        self.serve(PHOTO_PATH, PHOTO_BYTES)
        outcome = self.media.fetch(photo_message(), "  look at this  ", native_images=True)
        assert outcome is not None
        dest = os.path.join(self.paths.inbox, self.inbox_files()[0])
        self.assertEqual(outcome, MediaOutcome(text="look at this", images=(dest,)))

    def test_photo_without_caption_for_text_backend_is_only_the_note(self) -> None:
        self.serve(PHOTO_PATH, PHOTO_BYTES)
        outcome = self.media.fetch(photo_message(), "", native_images=False)
        assert outcome is not None
        dest = os.path.join(self.paths.inbox, self.inbox_files()[0])
        self.assertEqual(outcome.text, f"[photo saved for you at {dest}]")

    def test_photo_without_caption_for_native_backend_has_empty_text(self) -> None:
        self.serve(PHOTO_PATH, PHOTO_BYTES)
        outcome = self.media.fetch(photo_message(), "", native_images=True)
        assert outcome is not None
        self.assertEqual(outcome.text, "")
        self.assertEqual(len(outcome.images), 1)

    def test_download_404_tells_topic_and_leaves_no_file(self) -> None:
        self.telegram.script_result("getFile", {"file_path": PHOTO_PATH})
        outcome = self.media.fetch(photo_message(), "", native_images=False)
        self.assertIsNone(outcome)
        self.assertEqual(self.inbox_files(), [])
        texts = self.topic_texts()
        self.assertEqual(len(texts), 1)
        self.assertTrue(texts[0].startswith("photo download failed: "), texts)

    def test_getfile_refusal_tells_topic_with_short_text(self) -> None:
        self.telegram.fail_next("getFile", 400, description="x" * 300)
        outcome = self.media.fetch(photo_message(), "", native_images=False)
        self.assertIsNone(outcome)
        texts = self.topic_texts()
        self.assertEqual(len(texts), 1)
        self.assertTrue(texts[0].startswith("photo download failed: "))
        self.assertLessEqual(len(texts[0]), len("photo download failed: ") + 120)

    def test_bot_token_never_reaches_the_topic_text(self) -> None:
        self.telegram.script_result("getFile", {"file_path": PHOTO_PATH})
        self.media.fetch(photo_message(), "", native_images=False)  # the stub answers 404 for the file URL
        for text in self.topic_texts():
            self.assertNotIn(FAKE_TOKEN, text)
            self.assertNotIn("123456789:", text)


class VoiceTests(_MediaCase):
    def test_voice_transcribed_echoed_and_put_before_caption(self) -> None:
        """Ported labels:
            1d: 🎙 echo was sent with the transcript"""
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.text = "remind me at five"
        outcome = self.media.fetch(voice_message(), "please", native_images=False)
        self.assertEqual(outcome, MediaOutcome(text="remind me at five please"))
        self.assertEqual(self.topic_texts(), ["🎙 remind me at five"])
        self.assertEqual(len(self.transcriber.paths), 1)
        with open(self.transcriber.paths[0], "rb") as f:
            self.assertEqual(f.read(), VOICE_BYTES)

    def test_voice_without_caption_is_just_the_transcript(self) -> None:
        """Ported labels:
            1c: run_turn_v2 got the transcript (got ['hello there'])"""
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.text = "hello there"
        outcome = self.media.fetch(voice_message(), "", native_images=False)
        self.assertEqual(outcome, MediaOutcome(text="hello there"))

    def test_audio_file_is_handled_like_a_voice_note(self) -> None:
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.text = "from an audio file"
        outcome = self.media.fetch(voice_message("audio"), "", native_images=False)
        self.assertEqual(outcome, MediaOutcome(text="from an audio file"))

    def test_echo_is_capped_at_600_characters(self) -> None:
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.text = "a" * 1000
        outcome = self.media.fetch(voice_message(), "", native_images=False)
        assert outcome is not None
        self.assertEqual(self.topic_texts(), ["🎙 " + "a" * 600])
        self.assertEqual(outcome.text, "a" * 1000)

    def test_empty_transcript_tells_topic_and_stops(self) -> None:
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.text = ""
        self.assertIsNone(self.media.fetch(voice_message(), "", native_images=False))
        self.assertEqual(self.topic_texts(), [MSG_EMPTY])

    def test_transcriber_failure_tells_topic_with_its_text(self) -> None:
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.transcriber.error = TranscriptionFailed("bad audio")
        self.assertIsNone(self.media.fetch(voice_message(), "", native_images=False))
        self.assertEqual(self.topic_texts(), ["🎙 transcription failed: bad audio"])

    def test_voice_download_failure_is_reported_as_transcription_failure(self) -> None:
        self.telegram.script_result("getFile", {"file_path": VOICE_PATH})
        self.assertIsNone(self.media.fetch(voice_message(), "", native_images=False))
        texts = self.topic_texts()
        self.assertEqual(len(texts), 1)
        self.assertTrue(texts[0].startswith("🎙 transcription failed: "), texts)
        self.assertNotIn(FAKE_TOKEN, texts[0])

    def test_missing_soniox_key_tells_topic_and_keeps_the_download(self) -> None:
        service = self._service(SonioxTranscriber(self.paths, self.openers, self.clock, self.stub.url))
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.assertIsNone(service.fetch(voice_message(), "", native_images=False))
        self.assertEqual(self.topic_texts(), [MSG_NO_KEY])
        self.assertEqual(len(self.inbox_files()), 1)
        self.assertTrue(all(r.path.startswith("/file/") for r in self.stub.requests))

    def test_plain_message_without_media_is_text_only_and_touches_nothing(self) -> None:
        outcome = self.media.fetch({"chat": {"id": CHAT}, "message_thread_id": THREAD}, "hi", native_images=False)
        self.assertEqual(outcome, MediaOutcome(text="hi"))
        self.assertEqual(self.telegram.calls, ())


class SonioxTests(_MediaCase):
    """SonioxTranscriber against the stub: the upload, the job, the poll loop, the tokens and the cleanup."""

    def setUp(self) -> None:
        super().setUp()
        self.write_soniox_key()
        self.audio_path = os.path.join(self.home.home, "voice.oga")
        with open(self.audio_path, "wb") as f:
            f.write(VOICE_BYTES)
        self.soniox = SonioxTranscriber(self.paths, self.openers, self.clock, self.stub.url)

    def script_happy_path(self, *statuses: str) -> None:
        self.stub.route("POST", "/v1/files", (200, j({"id": "f1"})))
        self.stub.route("POST", "/v1/transcriptions", (200, j({"id": "t1"})))
        self.stub.route(
            "GET",
            "/v1/transcriptions/t1",
            *[(200, j({"status": status})) for status in statuses],
        )
        self.stub.route(
            "GET",
            "/v1/transcriptions/t1/transcript",
            (200, j({"tokens": [{"text": "Hello"}, {"text": " world"}]})),
        )
        self.stub.route("DELETE", "/v1/transcriptions/t1", (200, b"{}"))
        self.stub.route("DELETE", "/v1/files/f1", (200, b"{}"))

    def test_full_flow_returns_joined_tokens_and_cleans_up(self) -> None:
        self.script_happy_path("queued", "processing", "completed")
        self.assertEqual(self.soniox.transcribe(self.audio_path), "Hello world")
        upload = self.stub.calls("POST", "/v1/files")[0]
        self.assertTrue(upload.headers["Content-Type"].startswith("multipart/form-data; boundary="))
        self.assertIn(b'filename="voice.ogg"', upload.body)
        self.assertIn(b"Content-Type: audio/ogg", upload.body)
        self.assertIn(VOICE_BYTES, upload.body)
        self.assertEqual(upload.headers["Authorization"], f"Bearer {FAKE_KEY}")
        job = json.loads(self.stub.calls("POST", "/v1/transcriptions")[0].body)
        self.assertEqual(job, {"model": "stt-async-v5", "file_id": "f1"})
        self.assertEqual(self.clock.sleeps, [1.0, 1.0])
        order = [r.path for r in self.stub.requests if r.method == "DELETE"]
        self.assertEqual(order, ["/v1/transcriptions/t1", "/v1/files/f1"])

    def test_error_status_raises_with_soniox_message_and_still_cleans_up(self) -> None:
        self.stub.route("POST", "/v1/files", (200, j({"id": "f1"})))
        self.stub.route("POST", "/v1/transcriptions", (200, j({"id": "t1"})))
        self.stub.route("GET", "/v1/transcriptions/t1", (200, j({"status": "error", "error_message": "bad audio"})))
        self.stub.route("DELETE", "/v1/transcriptions/t1", (200, b"{}"))
        self.stub.route("DELETE", "/v1/files/f1", (200, b"{}"))
        with self.assertRaises(TranscriptionFailed) as raised:
            self.soniox.transcribe(self.audio_path)
        self.assertEqual(str(raised.exception), "bad audio")
        self.assertEqual(len(self.stub.calls("DELETE", "/v1/transcriptions/t1")), 1)
        self.assertEqual(len(self.stub.calls("DELETE", "/v1/files/f1")), 1)

    def test_poll_limit_times_out_and_cleans_up(self) -> None:
        self.script_happy_path("queued")
        with self.assertRaises(TranscriptionFailed) as raised:
            self.soniox.transcribe(self.audio_path)
        self.assertEqual(str(raised.exception), "timed out waiting for transcript")
        self.assertEqual(len(self.stub.calls("GET", "/v1/transcriptions/t1")), POLL_LIMIT)
        self.assertEqual(len(self.clock.sleeps), POLL_LIMIT)
        self.assertEqual(len(self.stub.calls("DELETE", "/v1/files/f1")), 1)

    def test_job_refusal_still_deletes_the_uploaded_file(self) -> None:
        self.stub.route("POST", "/v1/files", (200, j({"id": "f1"})))
        self.stub.route("POST", "/v1/transcriptions", (500, j({"error": "down"})))
        self.stub.route("DELETE", "/v1/files/f1", (200, b"{}"))
        with self.assertRaises(TranscriptionFailed):
            self.soniox.transcribe(self.audio_path)
        self.assertEqual(len(self.stub.calls("DELETE", "/v1/files/f1")), 1)
        self.assertEqual(self.stub.calls("DELETE", "/v1/transcriptions/t1"), [])

    def test_upload_refusal_raises_without_the_key_in_the_message(self) -> None:
        self.stub.route("POST", "/v1/files", (401, j({"message": "unauthorized"})))
        with self.assertRaises(TranscriptionFailed) as raised:
            self.soniox.transcribe(self.audio_path)
        self.assertNotIn(FAKE_KEY, str(raised.exception))
        self.assertEqual(self.stub.calls("DELETE", "/v1/files/f1"), [])

    def test_missing_key_file_is_unavailable_and_sends_nothing(self) -> None:
        os.remove(self.paths.soniox_key)
        with self.assertRaises(TranscriberUnavailable):
            self.soniox.transcribe(self.audio_path)
        self.assertEqual(self.stub.requests, [])

    def test_full_media_flow_through_the_real_transcriber(self) -> None:
        self.serve(VOICE_PATH, VOICE_BYTES)
        self.script_happy_path("completed")
        service = self._service(self.soniox)
        outcome = service.fetch(voice_message(), "later", native_images=False)
        self.assertEqual(outcome, MediaOutcome(text="Hello world later"))
        self.assertEqual(self.topic_texts(), ["🎙 Hello world"])


if __name__ == "__main__":
    unittest.main()
