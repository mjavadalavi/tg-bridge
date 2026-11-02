"""Native images from the queue to the session: a photo that the inbound handler hands to Dispatcher.deliver (or fills
into a reserved placeholder) reaches FakeBackend's send_user as image paths, and a backend without native image input
gets no paths (its '[photo saved for you at ...]' note stays in the text).

Real Dispatcher, real TurnRunner, real SlotPool and Messenger over FakeTelegram; only the backend is a fake.
"""

import dataclasses
import threading
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, ProbingBackend, TurnHarness, wait_until
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.topic_queues import Admission
from tg_bridge.telegram.answers import AnswerSender

SENDER = "42"
PHOTO = "/home/owner/state/inbox/x.jpg"
NOTE = "[photo saved for you at /home/owner/state/inbox/x.jpg]"
OK = TurnResult(TurnStatus.OK, answer="ok:photo", duration_ms=5, num_turns=1, subtype="success")


class _Titles:
    """Stands in for TitleService: the title call is not under test here."""

    def submit(self, *args: object) -> None:
        pass


class PhotoQueueCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.threads: list[threading.Thread] = []
        self.addCleanup(self._join_threads)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def _join_threads(self) -> None:
        for thread in self.threads:
            thread.join(10.0)

    def start(self, capabilities=FAKE_CAPABILITIES) -> None:
        self.backend = ProbingBackend([[OK], [OK]], capabilities=capabilities)
        self.runner = self.h.runner(self.h.registry_for(self.backend))
        self.dispatcher = Dispatcher(
            turns=self.runner, slots=self.h.slots, inbox=None, messenger=self.h.messenger,
            answers=AnswerSender(self.h.tg, FakeUploader(), self.h.messenger, self.h.logger, self.h.clock),
            titles=_Titles(), client=self.h.tg, live=self.h.live, config=self.h.config, logger=self.h.logger,
            start_thread=self._start_thread)

    def _start_thread(self, **kwargs: object) -> threading.Thread:
        thread = threading.Thread(**kwargs)  # type: ignore[arg-type]
        self.threads.append(thread)
        return thread

    def send_user_calls(self) -> list[tuple]:
        return [call for session in self.backend.sessions for call in session.calls if call[0] == "send_user"]

    def wait_for_turn_end(self) -> None:
        self.assertTrue(wait_until(lambda: self.send_user_calls() and not any(t.is_alive() for t in self.threads)),
                        "the turn did not run to its end")


class PhotoReachesSessionTests(PhotoQueueCase):
    def test_photo_without_caption_is_sent_as_native_image(self) -> None:
        # the owner sends a photo with no caption: MediaService gives text='' and images=(path,)
        self.start()
        delivery = self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, None, "", images=(PHOTO,))
        self.assertIs(delivery.admission, Admission.STARTED)
        self.wait_for_turn_end()
        self.assertEqual(self.send_user_calls(), [("send_user", "", (PHOTO,))])

    def test_placeholder_filled_with_text_and_image_keeps_both(self) -> None:
        # a photo reserved while another message waits: the filled placeholder carries the image to the session
        self.start()
        reserved = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, None)
        self.assertIs(reserved.admission, Admission.STARTED)
        filled = self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, None, "what is this?", reserved=reserved.nid,
                                         images=(PHOTO,))
        self.assertIs(filled.admission, Admission.FILLED)
        self.wait_for_turn_end()
        self.assertEqual(self.send_user_calls(), [("send_user", "what is this?", (PHOTO,))])

    def test_backend_without_native_images_gets_no_paths_and_keeps_the_note(self) -> None:
        # the Claude case: capabilities.images is False, so the paths are dropped and the note in the text is the carrier
        self.start(capabilities=dataclasses.replace(FAKE_CAPABILITIES, images=False))
        self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, None, NOTE, images=(PHOTO,))
        self.wait_for_turn_end()
        self.assertEqual(self.send_user_calls(), [("send_user", NOTE, ())])


if __name__ == "__main__":
    unittest.main()
