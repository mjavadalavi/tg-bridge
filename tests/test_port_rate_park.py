"""Port of the poll-thread scenario of fakecli/test_rate_park.py (label D): a command's reply goes through the ordered
out-worker, so handle() returns at once while the send budget is full, and the reply goes out once the send can.

The other rate-park scenarios (A, B, E) are gate and flood tests in test_ratelimit and test_telegram_client; C (idle
time in the header) is in test_feed and test_feed_format. Uses the InboundCase harness of inbound_support.
"""

import threading
import time
import unittest

from tests.inbound_support import THREAD, TOPIC, InboundCase

WAIT = 5.0


class PollThreadNeverWaitsForSends(InboundCase):
    def test_handle_returns_at_once_while_the_send_is_held_and_the_reply_goes_out_later(self) -> None:
        """D: handle_message returned at once while the send budget is full (0.00 s); D: the general-board reply went
        out once a slot freed (1 call(s))."""
        self.name_topic(THREAD, TOPIC)
        gate = threading.Event()
        self.out.submit(gate.wait, WAIT)  # the one out-worker is busy: the reply has to wait for a free slot
        started = time.monotonic()
        self.handler.handle(self.message("/ping"), None)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5, f"handle() waited {elapsed:.2f}s for the send")
        self.assertEqual(self.sent(), [], "nothing is sent while the slot is held")
        gate.set()
        self.settle()
        self.assertEqual(self.sent(), ["pong"])


if __name__ == "__main__":
    unittest.main()
