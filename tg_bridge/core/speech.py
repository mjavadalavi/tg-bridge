"""Text to a Telegram voice note: Microsoft's online voices (the edge-tts CLI), then ffmpeg to OGG/Opus.

Speech.synth(text) returns the OGG bytes, or None when every attempt failed. The online service is unofficial and
sometimes drops a request, so a failed attempt is retried ATTEMPTS times with a growing pause. Both tools run as child
processes (the bridge itself has no third-party package): edge-tts reads the text from a file, so a text that starts
with a dash or is very long is still taken as text, and writes an MP3 that ffmpeg turns into the mono Opus file
Telegram plays as a voice note. Each attempt works in its own temporary folder, removed at its end.
"""

import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence

from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

FEMALE_VOICE = "fa-IR-DilaraNeural"  # the owner picks one with /drive female|male (core/driving.py stores it)
MALE_VOICE = "fa-IR-FaridNeural"
EDGE_TTS = "edge-tts"         # found on PATH (the unit's PATH has ~/.local/bin, where `uv tool install` puts it)
FFMPEG = "ffmpeg"
ATTEMPTS = 3
PAUSES = (3.0, 8.0)           # seconds before the 2nd and the 3rd attempt
SYNTH_TIMEOUT = 90            # seconds for one edge-tts run
ENCODE_TIMEOUT = 60           # seconds for one ffmpeg run
ERROR_CHARS = 200

Run = Callable[..., subprocess.CompletedProcess]


class Speech:
    """Synthesizes voice notes. run is subprocess.run (tests pass a fake); workdir holds the per-attempt folders."""

    def __init__(self, clock: Clock, logger: Logger, workdir: str | None = None, run: Run = subprocess.run) -> None:
        self._clock = clock
        self._logger = logger
        self._workdir = workdir
        self._run = run

    def synth(self, text: str, voice: str = FEMALE_VOICE) -> bytes | None:
        """The voice note of text in voice, or None when all ATTEMPTS failed (each failure is logged)."""
        for attempt in range(ATTEMPTS):
            if attempt:
                self._clock.sleep(PAUSES[min(attempt - 1, len(PAUSES) - 1)])
            try:
                return self._once(text, voice)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self._logger.log(f"speech attempt {attempt + 1}/{ATTEMPTS} failed: {_detail(exc)}")
        return None

    def _once(self, text: str, voice: str) -> bytes:
        if self._workdir:
            os.makedirs(self._workdir, mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._workdir) as folder:
            source, mp3, ogg = (os.path.join(folder, name) for name in ("text.txt", "speech.mp3", "voice.ogg"))
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(text)
            self._call([EDGE_TTS, "--voice", voice, "--file", source, "--write-media", mp3], SYNTH_TIMEOUT)
            self._call([FFMPEG, "-loglevel", "error", "-y", "-i", mp3, "-c:a", "libopus", "-b:a", "32k", "-ac", "1",
                        ogg], ENCODE_TIMEOUT)
            with open(ogg, "rb") as handle:
                data = handle.read()
        if not data:
            raise ValueError("the voice file is empty")
        return data

    def _call(self, argv: Sequence[str], timeout: float) -> None:
        self._run(list(argv), check=True, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _detail(exc: BaseException) -> str:
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes) and stderr.strip():
        return stderr.decode("utf-8", "replace").strip().splitlines()[-1][:ERROR_CHARS]
    return str(exc)[:ERROR_CHARS]
