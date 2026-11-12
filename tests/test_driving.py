"""Driving mode (core/driving.py, core/speech.py, bot/commands/driving.py): state, spoken text, voice notes, /drive."""

import os
import subprocess
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tg_bridge.agents.events import PLAN_KIND, ApprovalRequest, Decision
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.callbacks.drive_taps import DriveTaps
from tg_bridge.bot.commands.driving import TURN_OFF_LABEL, TURN_ON_LABEL, DriveCommand
from tg_bridge.core.driving import (CARD_SPEECH, DRIVE_NOTE, DRIVE_SECONDS, DRIVE_TOPIC_NAME, NO_VOICE_TEXT,
                                    PLAN_SPEECH, UPLOAD_ATTEMPTS, DrivingMode, DrivingVoice, drive_note,
                                    spoken_text)
from tg_bridge.core.speech import ATTEMPTS, FEMALE_VOICE, MALE_VOICE, PAUSES, Speech
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.messenger import Messenger

CHAT = -100123
THREAD = 7561
VOICE_BYTES = b"OggS-voice"


class FakeRun:
    """subprocess.run for Speech: edge-tts writes the MP3, ffmpeg writes the OGG; the first `fails` calls fail."""

    def __init__(self, fails: int = 0) -> None:
        self.fails = fails
        self.calls: list[list[str]] = []
        self.texts: list[str] = []

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        if argv[0] == "edge-tts":
            with open(argv[argv.index("--file") + 1], encoding="utf-8") as handle:
                self.texts.append(handle.read())
            if self.fails:
                self.fails -= 1
                raise subprocess.CalledProcessError(1, argv, stderr=b"No audio was received")
            with open(argv[argv.index("--write-media") + 1], "wb") as handle:
                handle.write(b"mp3")
        else:
            with open(argv[-1], "wb") as handle:
                handle.write(VOICE_BYTES)
        return subprocess.CompletedProcess(argv, 0)


class InlinePool:
    def submit(self, fn, *args) -> None:
        fn(*args)


class DrivingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        base = os.path.realpath(self.home.home)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.db = Database(Paths.from_home(base))
        self.addCleanup(self.db.close)
        self.kv = KvRepo(self.db)
        self.sessions = SessionRepo(self.db, self.clock)
        self.sessions.ensure(THREAD, "accounting", "claude", base, "")
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.uploader = FakeUploader()
        self.run = FakeRun()
        self.mode = DrivingMode(self.kv, self.clock, self.telegram)
        self.voice = DrivingVoice(self.mode, Speech(self.clock, self.logger, os.path.join(base, "speech"), self.run),
                                  self.uploader, self.messenger, self.sessions, InlinePool(), self.clock, self.logger)

    def topics_made(self) -> int:
        return sum(1 for call in self.telegram.calls if call.method == "createForumTopic")


class SpokenTextTests(unittest.TestCase):
    def test_the_last_marked_line_is_spoken(self) -> None:
        answer = "Done.\n\n```\ncode\n```\n🔊 old\nmore\n🔊 ستون کارمزد اضافه شد و تست‌ها رد شدند."
        self.assertEqual(spoken_text(answer), "ستون کارمزد اضافه شد و تست‌ها رد شدند.")

    def test_without_a_mark_code_paths_and_links_are_left_out(self) -> None:
        answer = "تست‌ها **رد** شدند.\n```py\nx = 1\n```\nفایل `core/a.py` عوض شد: https://x.y/z"
        self.assertEqual(spoken_text(answer), "تست‌ها رد شدند. فایل عوض شد:")

    def test_a_long_answer_is_cut_at_a_sentence_end(self) -> None:
        spoken = spoken_text("جمله‌ی کوتاه. " * 100)
        self.assertLessEqual(len(spoken), 450)
        self.assertTrue(spoken.endswith("."))


class ModeTests(DrivingCase):
    def test_on_makes_the_topic_once_and_ends_by_itself(self) -> None:
        self.assertFalse(self.mode.active())
        self.assertEqual(drive_note(self.mode), "")
        thread = self.mode.turn_on(CHAT)
        self.assertEqual(self.mode.turn_on(CHAT), thread)
        self.assertEqual(self.topics_made(), 1)
        self.assertEqual([c.params["name"] for c in self.telegram.calls if c.method == "createForumTopic"],
                         [DRIVE_TOPIC_NAME])
        self.assertTrue(self.mode.active())
        self.assertEqual(drive_note(self.mode), DRIVE_NOTE)
        self.clock.advance(DRIVE_SECONDS + 1)
        self.assertFalse(self.mode.active())

    def test_off_and_the_voice_pick(self) -> None:
        self.mode.turn_on(CHAT)
        self.mode.turn_off()
        self.assertFalse(self.mode.active())
        self.assertEqual(self.mode.voice(), "female")
        self.mode.set_voice("male")
        self.assertEqual(self.mode.voice(), "male")
        with self.assertRaises(ValueError):
            self.mode.set_voice("robot")


class SpeechTests(DrivingCase):
    def test_a_failed_attempt_is_retried_with_a_pause(self) -> None:
        self.run.fails = 2
        speech = Speech(self.clock, self.logger, None, self.run)
        self.assertEqual(speech.synth("سلام", MALE_VOICE), VOICE_BYTES)
        self.assertEqual(self.clock.sleeps, list(PAUSES))
        self.assertIn(MALE_VOICE, self.run.calls[0])

    def test_every_attempt_failing_gives_none(self) -> None:
        self.run.fails = ATTEMPTS
        self.assertIsNone(Speech(self.clock, self.logger, None, self.run).synth("سلام"))


class VoiceTests(DrivingCase):
    def test_nothing_is_spoken_while_the_mode_is_off(self) -> None:
        self.voice.answer(CHAT, THREAD, "🔊 hi")
        self.assertEqual(self.uploader.uploads, ())

    def test_an_answer_becomes_a_voice_note_in_the_driving_topic(self) -> None:
        drive = self.mode.turn_on(CHAT)
        self.voice.answer(CHAT, THREAD, "متن کامل\n🔊 کار تمام شد.")
        [upload] = self.uploader.uploads
        self.assertEqual((upload.thread_id, upload.method, upload.field_name, upload.caption, upload.data),
                         (drive, "sendVoice", "voice", "accounting", VOICE_BYTES))
        self.assertEqual(self.run.texts, ["accounting. کار تمام شد."])
        self.assertIn(FEMALE_VOICE, self.run.calls[0])

    def test_a_voice_queued_before_the_mode_went_off_is_dropped(self) -> None:
        jobs: list[tuple] = []
        pool = type("Pool", (), {"submit": lambda _self, fn, *args: jobs.append((fn, args))})()
        voice = DrivingVoice(self.mode, Speech(self.clock, self.logger, None, self.run), self.uploader, self.messenger,
                             self.sessions, pool, self.clock, self.logger)
        self.mode.turn_on(CHAT)
        voice.answer(CHAT, THREAD, "🔊 hi")
        self.mode.turn_off()
        for fn, args in jobs:
            fn(*args)
        self.assertEqual((self.run.calls, self.uploader.uploads), ([], ()))

    def test_the_driving_topic_itself_is_never_spoken(self) -> None:
        drive = self.mode.turn_on(CHAT)
        self.voice.answer(CHAT, drive, "🔊 hi")
        self.assertEqual(self.uploader.uploads, ())

    def test_a_deleted_driving_topic_is_made_again(self) -> None:
        first = self.mode.turn_on(CHAT)
        self.uploader.fail_next("Bad Request: message thread not found")
        self.voice.answer(CHAT, THREAD, "🔊 hi")
        threads = [upload.thread_id for upload in self.uploader.uploads]
        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[0], first)
        self.assertNotEqual(threads[1], first)
        self.assertEqual(self.topics_made(), 2)

    def test_a_voice_that_never_goes_out_is_told_in_text(self) -> None:
        drive = self.mode.turn_on(CHAT)
        for _ in range(UPLOAD_ATTEMPTS):
            self.uploader.fail_next()
        self.voice.answer(CHAT, THREAD, "🔊 hi")
        self.assertEqual(self.telegram.texts(drive), [NO_VOICE_TEXT.format(topic="accounting")])

    def test_a_voice_that_cannot_be_made_is_told_in_text(self) -> None:
        drive = self.mode.turn_on(CHAT)
        self.run.fails = ATTEMPTS
        self.voice.answer(CHAT, THREAD, "🔊 hi")
        self.assertEqual(self.uploader.uploads, ())
        self.assertEqual(self.telegram.texts(drive), [NO_VOICE_TEXT.format(topic="accounting")])

    def test_cards_are_spoken_as_one_line(self) -> None:
        self.mode.turn_on(CHAT)
        choices = frozenset({Decision.ALLOW_ONCE, Decision.DENY})
        plan = ApprovalRequest(approval_id="p", kind=PLAN_KIND, tool="ExitPlanMode", input={}, summary="",
                               choices=choices)
        tool = ApprovalRequest(approval_id="t", kind="command", tool="Bash", input={}, summary="Bash", choices=choices)
        self.voice.card(CHAT, THREAD, plan)
        self.voice.card(CHAT, THREAD, tool)
        self.assertEqual(self.run.texts, [f"accounting. {PLAN_SPEECH}", f"accounting. {CARD_SPEECH}"])

    def test_the_answer_sender_tells_its_listener_and_survives_it(self) -> None:
        heard: list[tuple] = []
        sender = AnswerSender(self.telegram, self.uploader, self.messenger, self.logger, self.clock,
                              on_answer=lambda *args: heard.append(args))
        sender.send_answer(CHAT, THREAD, "  جواب  ")
        self.assertEqual(heard, [(CHAT, THREAD, "جواب")])

        def broken(*args) -> None:
            raise RuntimeError("boom")

        AnswerSender(self.telegram, self.uploader, self.messenger, self.logger, self.clock,
                     on_answer=broken).send_answer(CHAT, THREAD, "x")
        self.assertIn("x", self.telegram.texts(THREAD))


class FakeTap:
    def __init__(self, action: str) -> None:
        self.args, self.chat_id = (action,), CHAT
        self.toasts: list[str] = []
        self.edits: list[tuple] = []

    def toast(self, text: str) -> None:
        self.toasts.append(text)

    def edit(self, text: str, rows=None) -> None:
        self.edits.append((text, rows))


def labels(rows) -> list[str]:
    return [text for row in rows for text, _ in row]


class DriveCommandTests(DrivingCase):
    def setUp(self) -> None:
        super().setUp()
        self.out = WorkerPool("out", 1, self.logger)
        self.commands = CommandRegistry()
        DriveCommand(self.mode, self.logger).register(self.commands)
        self.deps = CommandDeps(config=None, sessions=self.sessions, turns=None, rules=None, kv=self.kv, live=None,
                                queues=None, backends=None, messenger=self.messenger, out=self.out)

    def drive(self, args: str = "") -> str:
        self.assertTrue(self.commands.dispatch(self.deps, CHAT, THREAD, "accounting", f"/drive {args}".strip()))
        self.assertTrue(self.out.join(5))
        return self.telegram.texts(THREAD)[-1]

    def test_drive_alone_shows_the_state_and_its_buttons(self) -> None:
        self.assertIn("OFF", self.drive())
        self.assertFalse(self.mode.active())
        [keyboard] = self.telegram.buttons(THREAD)
        self.assertEqual([[b["text"] for b in row] for row in keyboard],
                         [[TURN_ON_LABEL], ["👩 female ✓", "👨 male"]])
        self.assertEqual([b["callback_data"] for row in keyboard for b in row], ["dr:on", "dr:female", "dr:male"])

    def test_the_words_still_work(self) -> None:
        self.assertIn("ON", self.drive("روشن"))
        self.assertTrue(self.mode.active())
        self.assertIn("male", self.drive("مرد"))
        self.assertEqual(self.mode.voice(), "male")
        self.assertIn("OFF", self.drive("off"))
        self.assertIn("usage", self.drive("fast"))

    def test_the_panel_buttons_change_the_mode_and_repaint_the_panel(self) -> None:
        taps = DriveTaps(mode=self.mode, logger=self.logger)
        on = FakeTap("on")
        taps._tap(on)
        self.assertTrue(self.mode.active())
        text, rows = on.edits[-1]
        self.assertIn("ON", text)
        self.assertEqual(labels(rows), [TURN_OFF_LABEL, "👩 female ✓", "👨 male"])
        male = FakeTap("male")
        taps._tap(male)
        self.assertEqual(self.mode.voice(), "male")
        self.assertEqual(labels(male.edits[-1][1]), [TURN_OFF_LABEL, "👩 female", "👨 male ✓"])
        off = FakeTap("off")
        taps._tap(off)
        self.assertFalse(self.mode.active())
        bad = FakeTap("fly")
        taps._tap(bad)
        self.assertEqual(bad.edits, [])
        self.assertTrue(bad.toasts)


if __name__ == "__main__":
    unittest.main()
