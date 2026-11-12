"""MediaService (getFile/download) + Transcriber Protocol + SonioxTranscriber (C05, C17).

Port of the live file's _fetch_media (C17) and soniox_transcribe (C05). A message with a photo, voice note,
audio file or document is turned into what the session should receive:

- a photo is downloaded into the inbox (mode 0600). A backend with native image input gets the path as an image;
  any other backend gets the old text note '[photo saved for you at <path>]' appended to the message text;
- a voice note or audio file is downloaded, transcribed by the injected Transcriber, echoed to the topic (600 chars)
  and its text is put in front of the caption, as the live file did;
- a document (a file sent as a Telegram document, e.g. a PDF) is downloaded into the inbox under its original name and
  always becomes the text note '[file <name> saved for you at <path>]', even for a backend with native images. A file
  over the Bot API's 20 MB download limit is not downloaded: the topic is told its size instead.

Naming: every download is created with tempfile.mkstemp in the inbox, so the name is unique and never truncates an
existing file: '<unix seconds>-<random>-<safe name>'. The safe name keeps Unicode word characters (Persian stays
readable), turns every other character into '_', never contains a path separator, and keeps the extension within
NAME_LIMIT characters. A name that is only dots falls back to the Telegram basename. The note shows a cleaned copy of
the sender's name instead: whitespace (newlines included) collapses to single spaces, '[' and ']' become '(' and ')',
and the name is cut to NAME_LIMIT characters, so a file name can never close the note or add lines to the message.

fetch() returns None when the user has already been told why nothing will run (an error text was sent through the
Messenger). It returns a MediaOutcome otherwise; an outcome may have empty text when the only content is an image.

SonioxTranscriber reads its key from Paths.soniox_key on every call, uploads the audio, starts an async transcription,
polls it (one second apart, at most POLL_LIMIT times) and joins the tokens. Its finally block deletes the job and the
uploaded file even when the transcription fails. Its errors never show the key: the key is masked in every message.
"""

import contextlib
import json
import os
import re
import secrets
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tg_bridge.clock import Clock
from tg_bridge.paths import Paths
from tg_bridge.telegram.client import API_BASE, TelegramClient, TelegramError
from tg_bridge.telegram.messenger import Messenger
from tg_bridge.telegram.proxy import OpenerCache
from tg_bridge.telegram.upload import build_multipart

SONIOX_API = "https://api.soniox.com"
SONIOX_MODEL = "stt-async-v5"
SONIOX_UPLOAD_NAME = "voice.ogg"
SONIOX_UPLOAD_MIME = "audio/ogg"
SONIOX_TIMEOUT = 120  # seconds for one Soniox request
SONIOX_DELETE_TIMEOUT = 30  # seconds for each cleanup request
POLL_LIMIT = 60  # status checks before the transcription counts as timed out; voice notes are short
POLL_SECONDS = 1.0
DOWNLOAD_TIMEOUT = 60  # seconds for one Telegram file download
VOICE_ATTEMPTS = 3  # a voice note's download and its transcription are each tried this many times
VOICE_PAUSES = (3.0, 8.0)  # seconds before the 2nd and the 3rd try
BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # the Bot API's getFile serves files up to 20 MB
NAME_LIMIT = 100  # characters of a document's original name kept in the inbox file name
REDACTED = "<redacted>"
PHOTO_ERROR_LIMIT = 120
FILE_ERROR_LIMIT = 120
VOICE_ERROR_LIMIT = 150
ECHO_LIMIT = 600  # characters of a transcript echoed back to the topic
PHOTO_NOTE = "[photo saved for you at {path}]"
FILE_NOTE = "[file {name} saved for you at {path}]"
MSG_PHOTO_FAILED = "photo download failed: {error}"
MSG_FILE_TOO_LARGE = ("📎 {name} is {size} MB: Telegram lets a bot download files up to 20 MB only. "
                      "Send it smaller, or put it on disk and send its path.")
MSG_FILE_FAILED = "📎 file download failed: {error}"
MSG_TRANSCRIPTION_FAILED = "🎙 transcription failed after 3 tries: {error}"
MSG_ECHO = "🎙 {heard}"
MSG_EMPTY = "🎙 voice came through but transcribed to nothing — try again or type it."
MSG_NO_KEY = "🎙 voice support needs a Soniox key in ~/.local/state/tg-bridge/soniox-key"


class TranscriberUnavailable(RuntimeError):
    """No transcription key is configured, so voice notes cannot be transcribed."""


class TranscriptionFailed(RuntimeError):
    """The transcription service refused, failed, or did not finish in time. The message is safe to show."""


class DownloadFailed(RuntimeError):
    """A Telegram file could not be downloaded. The message never contains the bot token."""


@runtime_checkable
class Transcriber(Protocol):
    """Voice or audio file in, transcript text out. Implemented by SonioxTranscriber and test stubs."""

    def transcribe(self, path: str) -> str:
        """The transcript of the file at path (may be empty). Raises TranscriberUnavailable when not configured."""
        ...


@dataclass(frozen=True, slots=True)
class MediaOutcome:
    """What a message's media adds to the session: the text to run, and local image paths for native image input."""

    text: str
    images: tuple[str, ...] = ()


def _short(exc: BaseException, limit: int) -> str:
    return str(exc)[:limit]


def _masked(text: str, secret: str) -> str:
    """The text with every occurrence of secret replaced, so a credential never reaches a message or an exception."""
    return text.replace(secret, REDACTED) if secret else text


def _safe_name(name: str) -> str:
    """The inbox file name for a sent file name: its base name with every character that is not a Unicode word
    character, '.', '-' or a ZWNJ replaced by '_'. The extension is kept and the stem is trimmed so the result is at
    most NAME_LIMIT characters. No path separator survives, so the saved file stays inside the inbox. A name with
    nothing but dots (such as '..') gives '' and the caller falls back to the Telegram basename."""
    base = re.sub(r"[^\w.\-\u200c]", "_", os.path.basename(name))
    if not base.strip("."):
        return ""
    stem, ext = os.path.splitext(base)
    if len(ext) >= NAME_LIMIT:
        stem, ext = base, ""
    return stem[: NAME_LIMIT - len(ext)] + ext


def _note_name(name: str) -> str:
    """The sender's file name as it may appear inside FILE_NOTE: one line, no brackets, at most NAME_LIMIT characters.
    An empty result becomes 'file'."""
    cleaned = re.sub(r"\s+", " ", name).replace("[", "(").replace("]", ")").strip()[:NAME_LIMIT].rstrip()
    return cleaned or "file"


def _discard(path: str) -> None:
    """Remove a partial download; a file that is already gone is fine."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


class SonioxTranscriber:
    """The Soniox async REST flow: upload, transcribe, poll, fetch tokens, then delete the job and the file.

    Collaborators are injected: Paths (the key file), OpenerCache (the egress proxy), Clock (the poll pauses).
    api_base exists so tests can point the transcriber at a local stub server.
    """

    def __init__(self, paths: Paths, openers: OpenerCache, clock: Clock, api_base: str = SONIOX_API) -> None:
        self._paths = paths
        self._openers = openers
        self._clock = clock
        self._api_base = api_base.rstrip("/")

    def transcribe(self, path: str) -> str:
        key = self._read_key()
        with open(path, "rb") as f:
            audio = f.read()
        boundary = "----tgbridge" + secrets.token_hex(8)
        body = build_multipart(boundary, (), "file", SONIOX_UPLOAD_NAME, SONIOX_UPLOAD_MIME, audio)
        uploaded = self._request(
            "POST", "/v1/files", key, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        file_id = str(uploaded["id"])
        job_id: str | None = None
        try:
            job = self._request(
                "POST",
                "/v1/transcriptions",
                key,
                json.dumps({"model": SONIOX_MODEL, "file_id": file_id}).encode(),
                {"Content-Type": "application/json"},
            )
            job_id = str(job["id"])
            self._wait(job_id, key)
            transcript = self._request("GET", f"/v1/transcriptions/{job_id}/transcript", key)
            tokens = transcript.get("tokens") or []
            return "".join(str(t.get("text", "")) for t in tokens).strip()
        finally:
            self._cleanup(job_id, file_id, key)

    def _read_key(self) -> str:
        try:
            with open(self._paths.soniox_key, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise TranscriberUnavailable("no Soniox key file") from None

    def _wait(self, job_id: str, key: str) -> None:
        for _ in range(POLL_LIMIT):
            status = self._request("GET", f"/v1/transcriptions/{job_id}", key)
            if status.get("status") == "completed":
                return
            if status.get("status") == "error":
                raise TranscriptionFailed(str(status.get("error_message") or "soniox error"))
            self._clock.sleep(POLL_SECONDS)
        raise TranscriptionFailed("timed out waiting for transcript")

    def _cleanup(self, job_id: str | None, file_id: str, key: str) -> None:
        targets = [f"/v1/files/{file_id}"]
        if job_id is not None:
            targets.insert(0, f"/v1/transcriptions/{job_id}")
        for path in targets:
            try:
                self._request("DELETE", path, key, timeout=SONIOX_DELETE_TIMEOUT)
            except Exception:
                pass  # cleanup is best effort, as in the live file

    def _request(
        self,
        method: str,
        path: str,
        key: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = SONIOX_TIMEOUT,
    ) -> dict[str, Any]:
        request_headers = {"Authorization": f"Bearer {key}", **(headers or {})}
        request = urllib.request.Request(
            self._api_base + path, data=data, headers=request_headers, method=method
        )
        try:
            with self._openers.get().open(request, timeout=timeout) as response:
                payload = json.load(response) if method != "DELETE" else {}
        except urllib.error.HTTPError as exc:
            exc.close()  # an HTTP error keeps its socket open until it is closed
            raise TranscriptionFailed(_masked(_short(exc, VOICE_ERROR_LIMIT), key)) from None
        except Exception as exc:
            raise TranscriptionFailed(_masked(_short(exc, VOICE_ERROR_LIMIT), key)) from None
        return payload if isinstance(payload, dict) else {}


class MediaService:
    """Downloads and transcribes the media of one message. Every collaborator is injected, nothing is global."""

    def __init__(
        self,
        paths: Paths,
        client: TelegramClient,
        messenger: Messenger,
        openers: OpenerCache,
        transcriber: Transcriber,
        clock: Clock,
        api_base: str = API_BASE,
    ) -> None:
        self._paths = paths
        self._client = client
        self._messenger = messenger
        self._openers = openers
        self._transcriber = transcriber
        self._clock = clock
        self._api_base = api_base.rstrip("/")

    def fetch(self, message: dict[str, Any], text: str, native_images: bool) -> MediaOutcome | None:
        """The text and images a message's media adds, or None when an error text was already sent to the topic.

        native_images is the backend's capabilities.images: True passes a photo as an image path, False appends the
        '[photo saved for you at <path>]' note to the text. An outcome with empty text is valid for a photo alone.
        """
        chat_id = (message.get("chat") or {}).get("id")
        thread = message.get("message_thread_id")
        images: tuple[str, ...] = ()
        if message.get("photo"):
            saved = self._photo(message["photo"][-1]["file_id"], chat_id, thread)
            if saved is None:
                return None
            if native_images:
                images = (saved,)
            else:
                text = (text + "\n" + PHOTO_NOTE.format(path=saved)).strip()
        document = message.get("document")
        if document:
            name = str(document.get("file_name") or "file")
            size = document.get("file_size")
            if size is not None and size > BOT_DOWNLOAD_LIMIT:
                self._messenger.send_text(chat_id, thread, MSG_FILE_TOO_LARGE.format(
                    name=name, size=f"{size / (1024 * 1024):.1f}"))
                return None
            saved = self._file(str(document["file_id"]), name, chat_id, thread)
            if saved is None:
                return None
            text = (text + "\n" + FILE_NOTE.format(name=_note_name(name), path=saved)).strip()
        media = message.get("voice") or message.get("audio")
        if media:
            heard = self._voice(str(media["file_id"]), chat_id, thread)
            if heard is None:
                return None
            text = (heard + ((" " + text) if text else "")).strip()
        return MediaOutcome(text=text.strip(), images=images)

    def _photo(self, file_id: str, chat_id: Any, thread: Any) -> str | None:
        try:
            return self._download(file_id)
        except Exception as exc:
            self._messenger.send_text(chat_id, thread, MSG_PHOTO_FAILED.format(error=_short(exc, PHOTO_ERROR_LIMIT)))
            return None

    def _file(self, file_id: str, name: str, chat_id: Any, thread: Any) -> str | None:
        try:
            return self._download(file_id, name)
        except Exception as exc:
            self._messenger.send_text(chat_id, thread, MSG_FILE_FAILED.format(error=_short(exc, FILE_ERROR_LIMIT)))
            return None

    def _voice(self, file_id: str, chat_id: Any, thread: Any) -> str | None:
        try:
            path = self._tries(self._download, file_id)
            heard = self._tries(self._transcriber.transcribe, path)
        except TranscriberUnavailable:
            self._messenger.send_text(chat_id, thread, MSG_NO_KEY)
            return None
        except Exception as exc:
            self._messenger.send_text(chat_id, thread, MSG_TRANSCRIPTION_FAILED.format(error=_short(exc, VOICE_ERROR_LIMIT)))
            return None
        if not heard:
            self._messenger.send_text(chat_id, thread, MSG_EMPTY)
            return None
        self._messenger.send_text(chat_id, thread, MSG_ECHO.format(heard=heard[:ECHO_LIMIT]))
        return heard

    def _tries(self, step: Callable[[str], str], arg: str) -> str:
        """step(arg), tried VOICE_ATTEMPTS times with VOICE_PAUSES between: Soniox and the Telegram file server drop a
        request now and then. A missing Soniox key is not tried again (TranscriberUnavailable); the last failure is
        raised."""
        for attempt in range(VOICE_ATTEMPTS):
            if attempt:
                self._clock.sleep(VOICE_PAUSES[min(attempt - 1, len(VOICE_PAUSES) - 1)])
            try:
                return step(arg)
            except TranscriberUnavailable:
                raise
            except Exception:
                if attempt == VOICE_ATTEMPTS - 1:
                    raise
        raise AssertionError("unreachable")

    def _download(self, file_id: str, name: str = "") -> str:
        """Fetch a Telegram file into the inbox and return its local path. The file is created with mkstemp, so its
        name is unique (two downloads in the same second never share a path) and its mode is 0600. A given name (a
        document's original file name) is made safe and kept in the inbox file name; photos and voice notes keep their
        Telegram basename as the suffix. A failed download removes its partial file."""
        info = self._client.call("getFile", file_id=file_id)
        if not isinstance(info, dict) or not info.get("file_path"):
            raise TelegramError("getFile", None, "no file_path in the answer")
        file_path = str(info["file_path"])
        token = self._read_token()
        url = f"{self._api_base}/file/bot{token}/{file_path}"
        os.makedirs(self._paths.inbox, mode=0o700, exist_ok=True)
        leaf = _safe_name(name) or os.path.basename(file_path)
        fd, dest = tempfile.mkstemp(dir=self._paths.inbox, prefix=f"{int(self._clock.now())}-", suffix=f"-{leaf}")
        try:
            with os.fdopen(fd, "wb") as out, self._openers.get().open(url, timeout=DOWNLOAD_TIMEOUT) as response:
                shutil.copyfileobj(response, out)
        except urllib.error.HTTPError as exc:
            exc.close()  # an HTTP error keeps its socket open until it is closed
            _discard(dest)
            raise DownloadFailed(_masked(_short(exc, PHOTO_ERROR_LIMIT), token)) from None
        except Exception as exc:
            _discard(dest)
            raise DownloadFailed(_masked(_short(exc, PHOTO_ERROR_LIMIT), token)) from None
        os.chmod(dest, 0o600)
        return dest

    def _read_token(self) -> str:
        with open(self._paths.token, encoding="utf-8") as f:
            return f.read().strip()
