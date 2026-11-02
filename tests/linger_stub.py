"""StubKeptLinger: the LingerService calls that the bot's stop, status and tap code make, with the answers a test sets.

There is no hold, no self turn and no process here. status() returns the LingerStatus the test gives (None when nothing
is kept), and each request is recorded as (kind, thread) so a test can see what was asked. The real service is covered
by tests/test_linger.py; this stub only stands in for it where the bot's own behaviour is under test.
"""

from tg_bridge.core.linger import LingerStatus


class StubKeptLinger:
    def __init__(self, status: LingerStatus | None = None, release: bool = False, interrupt: bool = False) -> None:
        self.status_value = status
        self.release_answer = release
        self.interrupt_answer = interrupt
        self.requests: list[tuple[str, int]] = []
        self.self_turn_now: tuple[int, int] | None = None  # (hold_id, self_seq) of the self turn running now, if any

    def in_self_turn(self, thread_id: int, hold_id: int, self_seq: int) -> bool:
        return self.self_turn_now == (hold_id, self_seq)

    def status(self, thread_id: int) -> LingerStatus | None:
        return self.status_value

    def request_release(self, thread_id: int) -> bool:
        self.requests.append(("release", thread_id))
        return self.release_answer

    def request_interrupt(self, thread_id: int) -> bool:
        self.requests.append(("interrupt", thread_id))
        return self.interrupt_answer
