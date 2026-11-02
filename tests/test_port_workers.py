"""Port of the voice, title and ordering scenarios of fakecli/test_workers.py (labels 1e, 1f, 2b, 5a) that the split
dispatcher and title service carry in their own modules. The other labels of that suite are in test_workers,
test_poller, test_media, test_dispatcher and test_titles (see the 'Ported labels' docstring lines there).

Uses the DispatcherCase harness of test_dispatcher: real inbox, queues and Messenger; a StubTurns for the runner.
"""

import unittest

from tests.test_dispatcher import CHAT, SENDER, THREAD, TOPIC, DispatcherCase, RecordingTitles
from tg_bridge.core.topic_queues import Admission


class OrderedTitles(RecordingTitles):
    """Records each submit together with whether the answer of that turn had been sent when it was submitted."""

    def __init__(self, tg, thread: int) -> None:
        super().__init__()
        self._tg = tg
        self._thread = thread
        self.answer_sent_first: list[bool] = []

    def submit(self, chat_id, thread_id, name, first_msg, first_answer) -> None:
        self.answer_sent_first.append(first_answer in self._tg.texts(self._thread))
        super().submit(chat_id, thread_id, name, first_msg, first_answer)


class PortWorkerScenarios(DispatcherCase):
    def test_plain_text_reaches_the_dispatcher_at_once(self) -> None:
        """1e: text message still reaches the dispatcher: a plain text starts its turn without any placeholder."""
        uid = self.ingest(6001)
        self.assertIs(self.deliver(uid, "plain text").admission, Admission.STARTED)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["plain text"])

    def test_voice_turn_submits_its_title_with_the_transcript(self) -> None:
        """1f: title job of the voice turn: the transcript that filled the voice place is the first message of the
        turn, and the title job is submitted for it once the answer is out.

        Ported labels:
            1f: title job of the voice turn finished
        """
        voice = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, self.ingest(6101))
        self.assertIs(self.deliver(self.ingest(6102), "hello there", reserved=voice.nid).admission, Admission.FILLED)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["hello there"])
        self.assertEqual(self.titles.submits, [(CHAT, THREAD, TOPIC, "hello there", "answer to hello there")])

    def test_answer_is_sent_before_the_title_is_submitted(self) -> None:
        """2b: the answer sendMessage is recorded before the title job is handed over (the title call never delays
        the answer).

        Ported labels:
            2b: answer sendMessage recorded before title_session finished (events=[('send', 'ans:start a title'), ('title_start', 'start a title'), ('title_done', 'start a title')])
        """
        self.titles = OrderedTitles(self.tg, THREAD)
        self.dispatcher = self._make_dispatcher()
        self.deliver(self.ingest(6201), "start a title")
        self.assertTrue(self.idle())
        self.assertEqual(self.titles.answer_sent_first, [True])

    def test_text_after_a_failed_voice_still_runs(self) -> None:
        """5a: text ran after the failed voice: the transcription failure gives the voice place back (unreserve, as
        the media job does), and the text queued behind it runs.

        Ported labels:
            5a: text ran after the failed voice (got ['still runs'])
        """
        voice = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, self.ingest(6301))
        behind = self.deliver(self.ingest(6302), "still runs")
        self.assertIs(behind.admission, Admission.QUEUED)
        self.assertTrue(self.dispatcher.unreserve(THREAD, voice.nid))
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["still runs"])


if __name__ == "__main__":
    unittest.main()
