"""Driving mode: while it is on, every answer and every approval card is also spoken, as a voice note in its topic.

/drive turns it on (bot/commands/driving.py). It turns itself off DRIVE_SECONDS later, so a forgotten mode does not
keep speaking. The owner picks a female or a male voice once; the pick is kept for every later drive.

- Where: the voice note goes to the same topic as the answer, right after it, so nothing has to be switched to.
- What: while the mode is on, every user turn's text gets DRIVE_NOTE, which asks the model to end its answer with one
  line that starts with SPOKEN_MARK and says the answer in a few plain spoken sentences. That line is what is spoken.
  An answer without it (a background update, a model that skipped it) is spoken from its first sentences, with code,
  paths and links taken out. An approval card is spoken as one short line; it is answered with its buttons, as always.
- Failures: the voice is made by core/speech.Speech, which retries; a refused upload is retried UPLOAD_ATTEMPTS times.
  When no voice could be sent, a short text says so in the topic. A deleted topic is not retried.
  Nothing here ever raises into the turn: the voice runs on its own one-thread pool, after the answer was sent.
"""

import re
from types import MappingProxyType
from typing import Protocol

from tg_bridge.agents.events import PLAN_KIND, ApprovalRequest
from tg_bridge.clock import Clock
from tg_bridge.core.speech import FEMALE_VOICE, MALE_VOICE, Speech
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.telegram.client import TelegramError

DRIVE_SECONDS = 4 * 3600
KEY_UNTIL = "driving:until"    # epoch seconds the mode ends at; no key = off
KEY_VOICE = "driving:voice"    # "female" | "male"
VOICES = MappingProxyType({"female": FEMALE_VOICE, "male": MALE_VOICE})
DEFAULT_VOICE = "female"
SPOKEN_MARK = "🔊"
SPOKEN_CHARS = 900             # the most a spoken line may say; the rest is cut at a sentence end
FALLBACK_CHARS = 450           # how much of an answer without a spoken line is read
UPLOAD_ATTEMPTS = 3
UPLOAD_PAUSE = 5.0
ERROR_CHARS = 200
THREAD_GONE = "thread not found"
DRIVE_NOTE = (
    "\n\n[tg-bridge: the owner is driving and will only hear your reply as a voice note. End your final message with "
    "one line that starts with 🔊 and says the reply in 1-3 short spoken sentences, in the language of "
    "the owner's message: plain words only, no code, file paths, links, tables or symbols.]")
NO_VOICE_TEXT = "🔇 the voice of this answer could not be made."
CARD_SPEECH = "یک درخواست تأیید منتظر توست."
PLAN_SPEECH = "برنامه آماده است و منتظر تأیید توست."


class _Upload(Protocol):
    def send_doc_bytes(self, chat_id: int | str, thread_id: int | str, data: bytes, filename: str, mime: str,
                       caption: str = "", method: str = "sendDocument", field_name: str = "document") -> object: ...


class _Text(Protocol):
    def send_text(self, chat_id: int | str, thread_id: int | str, text: str) -> object: ...


class _Pool(Protocol):
    def submit(self, fn: object, *args: object) -> None: ...


class DrivingMode:
    """The mode's state, kept in kv so a restart keeps it: on until a time, and the voice pick."""

    def __init__(self, kv: KvRepo, clock: Clock) -> None:
        self._kv = kv
        self._clock = clock

    def turn_on(self) -> None:
        """Turns the mode on for DRIVE_SECONDS from now."""
        self._kv.set(KEY_UNTIL, str(int(self._clock.now() + DRIVE_SECONDS)))

    def turn_off(self) -> None:
        self._kv.delete(KEY_UNTIL)

    def active(self) -> bool:
        until = self._kv.get(KEY_UNTIL)
        return until is not None and until.isdigit() and int(until) > self._clock.now()

    def minutes_left(self) -> int:
        until = self._kv.get(KEY_UNTIL) or "0"
        return max(0, int((int(until) - self._clock.now()) // 60)) if until.isdigit() else 0

    def voice(self) -> str:
        """'female' or 'male'."""
        picked = self._kv.get(KEY_VOICE)
        return picked if picked in VOICES else DEFAULT_VOICE

    def set_voice(self, voice: str) -> None:
        if voice not in VOICES:
            raise ValueError(f"unknown voice {voice!r}")
        self._kv.set(KEY_VOICE, voice)


class DrivingVoice:
    """Speaks answers and cards while the mode is on. answer() and card() only queue the work on pool."""

    def __init__(self, mode: DrivingMode, speech: Speech, uploader: _Upload, messenger: _Text, pool: _Pool,
                 clock: Clock, logger: Logger) -> None:
        self._mode = mode
        self._speech = speech
        self._uploader = uploader
        self._messenger = messenger
        self._pool = pool
        self._clock = clock
        self._logger = logger

    def answer(self, chat_id: int | str, thread_id: int | str, text: str) -> None:
        if self._wanted():
            self._pool.submit(self._speak, chat_id, int(thread_id), spoken_text(text))

    def card(self, chat_id: int | str, thread_id: int, request: ApprovalRequest) -> None:
        if self._wanted():
            line = PLAN_SPEECH if request.kind == PLAN_KIND else CARD_SPEECH
            self._pool.submit(self._speak, chat_id, thread_id, line)

    def _wanted(self) -> bool:
        try:
            return self._mode.active()
        except Exception as exc:  # never into the turn
            self._logger.log(f"driving: {str(exc)[:ERROR_CHARS]}")
            return False

    def _speak(self, chat_id: int | str, thread_id: int, words: str) -> None:
        try:
            if not self._mode.active():  # turned off (or ended) while this voice waited in the queue
                return
            data = self._speech.synth(words, VOICES[self._mode.voice()]) if words else None
            if not self._mode.active():  # turned off while the voice was being made
                return
            if (data is None or not self._send(chat_id, thread_id, data)) and self._mode.active():
                self._messenger.send_text(chat_id, thread_id, NO_VOICE_TEXT)
        except Exception as exc:
            self._logger.log(f"driving voice for thread {thread_id} failed: {str(exc)[:ERROR_CHARS]}")

    def _send(self, chat_id: int | str, thread_id: int, data: bytes) -> bool:
        """True when the voice went out, or when it is no longer wanted (mode off, topic deleted)."""
        for attempt in range(UPLOAD_ATTEMPTS):
            if attempt:
                self._clock.sleep(UPLOAD_PAUSE)
                if not self._mode.active():  # turned off during the pause: dropped, and nothing to tell
                    return True
            try:
                self._uploader.send_doc_bytes(chat_id, thread_id, data, "voice.ogg", "audio/ogg", "", "sendVoice",
                                              "voice")
                return True
            except TelegramError as exc:
                self._logger.log(f"driving voice upload {attempt + 1}/{UPLOAD_ATTEMPTS}: {str(exc)[:ERROR_CHARS]}")
                if THREAD_GONE in str(exc).lower() or THREAD_GONE in getattr(exc, "description", "").lower():
                    return True  # the topic was deleted: there is nowhere to speak, or to say so
            except OSError as exc:
                self._logger.log(f"driving voice upload {attempt + 1}/{UPLOAD_ATTEMPTS}: {str(exc)[:ERROR_CHARS]}")
        return False


def drive_note(mode: DrivingMode) -> str:
    """What a user turn's text gets while the mode is on ('' when it is off)."""
    return DRIVE_NOTE if mode.active() else ""


def spoken_text(answer: str) -> str:
    """The words to speak for an answer: its last SPOKEN_MARK line, else its first sentences without code or links."""
    marked = [line.strip() for line in answer.splitlines() if line.strip().startswith(SPOKEN_MARK)]
    if marked:
        return _cut(marked[-1][len(SPOKEN_MARK):].strip(" :"), SPOKEN_CHARS)
    plain = re.sub(r"```.*?```", " ", answer, flags=re.S)
    plain = re.sub(r"`[^`]*`|https?://\S+", " ", plain)
    plain = re.sub(r"[*_#>|~\[\]]", " ", plain)
    return _cut(re.sub(r"\s+", " ", plain).strip(), FALLBACK_CHARS)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(mark) for mark in (".", "!", "؟", "?", "\n"))
    return head[:end + 1] if end > limit // 3 else head
