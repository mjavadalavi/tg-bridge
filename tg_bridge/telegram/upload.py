"""Multipart uploads: send_doc_bytes and send_file (C23), the bodies shared by the CLI and the answer documents.

Port of the live file's send_doc_bytes() and send_file() (C23). Uploads bypass the JSON client on purpose, as they did
in the live file: a multipart body cannot go through TelegramClient.call. They keep the rest of the contract though:
the token is re-read from Paths.token for every request, the request goes through the shared egress proxy
(OpenerCache), and a failure never shows the request URL, so the token cannot reach an exception text or the log.

send_file refuses a missing file and anything over the 50 MB Bot API upload cap with UploadRefused before any request.
An image goes out as a photo; when Telegram refuses the photo the same file is sent again as a document.
"""

import json
import os
import secrets
import urllib.error
import urllib.request
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.client import API_BASE, TelegramError
from tg_bridge.telegram.proxy import OpenerCache

UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024  # the Bot API upload cap
UPLOAD_TIMEOUT = 120  # seconds for one multipart request
CAPTION_LIMIT = 1024  # characters Telegram keeps of a caption
DESCRIPTION_LIMIT = 200
PHOTO_LOG_LIMIT = 150
PHOTO_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
MIME_TYPES = MappingProxyType({
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
})
DEFAULT_MIME = "application/octet-stream"


class UploadRefused(ValueError):
    """A local refusal, before anything was sent: no such file, or larger than UPLOAD_LIMIT_BYTES."""


@runtime_checkable
class DocumentSender(Protocol):
    """What the answer sender and the CLI need from an uploader. Implemented by Uploader and FakeUploader."""

    def send_doc_bytes(
        self,
        chat_id: int | str,
        thread_id: int | str,
        data: bytes,
        filename: str,
        mime: str,
        caption: str = "",
        method: str = "sendDocument",
        field_name: str = "document",
    ) -> dict[str, Any]:
        """One upload of in-memory bytes into a topic. Returns the sent message."""
        ...

    def send_file(self, chat_id: int | str, thread_id: int | str, path: str, caption: str = "") -> None:
        """A local file into a topic: a photo for images (with a document fallback), a document otherwise."""
        ...


def build_multipart(
    boundary: str,
    form: Sequence[tuple[str, str]],
    field_name: str,
    filename: str,
    mime: str,
    data: bytes,
) -> bytes:
    """The multipart/form-data body: the plain form fields first, then the file under field_name. Pure."""
    parts = []
    for name, value in form:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; '
        f'filename="{_safe_filename(filename)}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


def _safe_filename(filename: str) -> str:
    """The file name as it may sit inside a quoted header: quotes and line breaks become underscores."""
    return "".join("_" if ch in '"\r\n' else ch for ch in filename)


def mime_for(path: str) -> str:
    """The media type sent with a local file, by extension."""
    return MIME_TYPES.get(os.path.splitext(path)[1].lower(), DEFAULT_MIME)


class Uploader:
    """The real uploader: one urllib multipart POST per call, through the shared egress proxy.

    Collaborators are injected: Paths (token file), OpenerCache (proxy), Logger (photo fallback line). api_base
    exists so tests can point the uploader at a local stub server.
    """

    def __init__(self, paths: Paths, openers: OpenerCache, logger: Logger, api_base: str = API_BASE) -> None:
        self._paths = paths
        self._openers = openers
        self._logger = logger
        self._api_base = api_base.rstrip("/")

    def send_doc_bytes(
        self,
        chat_id: int | str,
        thread_id: int | str,
        data: bytes,
        filename: str,
        mime: str,
        caption: str = "",
        method: str = "sendDocument",
        field_name: str = "document",
    ) -> dict[str, Any]:
        """One multipart upload into a topic. Raises TelegramError when Telegram refuses it, and the urllib errors
        when the network fails. There is no retry here: callers decide what a failed upload means."""
        boundary = "----tgbridge" + secrets.token_hex(8)
        form = [("chat_id", str(chat_id)), ("message_thread_id", str(thread_id))]
        if caption:
            form.append(("caption", caption[:CAPTION_LIMIT]))
        body = build_multipart(boundary, form, field_name, filename, mime, data)
        return self._post(method, body, boundary)

    def send_file(self, chat_id: int | str, thread_id: int | str, path: str, caption: str = "") -> None:
        """Photo or file into a topic. Images ride sendPhoto (inline preview), everything else sendDocument. A
        refused photo is logged and the same file is sent as a document. Refuses a missing or oversized file."""
        if not os.path.isfile(path):
            raise UploadRefused(f"no such file {path!r}")
        if os.path.getsize(path) > UPLOAD_LIMIT_BYTES:
            raise UploadRefused("file exceeds the Bot API 50 MB upload limit")
        with open(path, "rb") as f:
            data = f.read()
        name, mime = os.path.basename(path), mime_for(path)
        if os.path.splitext(path)[1].lower() not in PHOTO_EXTENSIONS:
            self.send_doc_bytes(chat_id, thread_id, data, name, mime, caption, "sendDocument", "document")
            return
        try:
            self.send_doc_bytes(chat_id, thread_id, data, name, mime, caption, "sendPhoto", "photo")
        except Exception as exc:
            self._logger.log(f"sendPhoto failed, retrying as document: {str(exc)[:PHOTO_LOG_LIMIT]}")
            self.send_doc_bytes(chat_id, thread_id, data, name, mime, caption, "sendDocument", "document")

    def _post(self, method: str, body: bytes, boundary: str) -> dict[str, Any]:
        url = f"{self._api_base}/bot{self._read_token()}/{method}"
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        try:
            with self._openers.get().open(request, timeout=UPLOAD_TIMEOUT) as response:
                payload = _json_object(response.read())
        except urllib.error.HTTPError as exc:
            payload = _json_object(exc.read())
            raise TelegramError(method, exc.code, str(payload.get("description") or "")[:DESCRIPTION_LIMIT]) from None
        if not payload.get("ok"):
            raise TelegramError(method, None, str(payload)[:DESCRIPTION_LIMIT])
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def _read_token(self) -> str:
        with open(self._paths.token, encoding="utf-8") as f:
            return f.read().strip()


def _json_object(raw: bytes | None) -> dict[str, Any]:
    """The JSON object in a Telegram response body, or an empty dict when there is none."""
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
