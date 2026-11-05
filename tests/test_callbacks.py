"""Callback router and the queue, stop and /status taps (STEP 49), ported from fakecli/test_buttons.py (sections H, I,
J, K, M, N, O, P) plus the router rules of the design: allow-list, unknown prefix logged and ignored, one try/except.

Scenarios: a tap from a non-allowed user is refused and changes nothing; an unknown prefix or picker kind is logged and
ignored; a malformed picker callback is ignored; a failing handler is logged and does not escape; a duplicate prefix
is rejected. now / jump / drop act on the queued message (now also interrupts the turn); st and /stop call the same
SessionCommands.stop_turn method (asserted with a spy); /status shortcuts reuse the commands and send the pickers; the
stop-and-new button waits for the turn and edits its message.
Every test runs on FakeTelegram, FakeClock and a TempHome; the only threads are the ones the code under test starts.
"""

import json
import os
import unittest
from unittest import mock

from tests.callback_support import CHAT, MID, THREAD, TOPIC, U1, CallbackCase
from tests.fakes.backend import FakeBackend, RecordingSink, fake_spec
from tests.linger_stub import StubKeptLinger
from tests.turn_support import wait_until
from tg_bridge.agents.claude.backend import ClaudeRules
from tg_bridge.bot.callbacks.handlers import (
    CHOICE_EXPIRED_TOAST,
    CHOOSE_TOAST,
    DROP_NOTE,
    INTERRUPT_TOAST,
    JUMP_NOTE,
    NOT_QUEUED_TEXT,
    NOW_NOTE,
    NOTHING_RUNNING_TEXT,
    STOP_TOAST,
    StatusTaps,
    StopChoiceTaps,
    StopTap,
)
from tg_bridge.bot.callbacks.handlers import QueueTaps
from tg_bridge.bot.callbacks.picker_taps import PickerTaps
from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.config import Config
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import LingerStatus
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.bot.commands.session import (
    NEW_BUSY_TEXT,
    NEW_DONE_TEXT,
    NEW_TEXT,
    NO_SESSION_TEXT,
    NOTHING_RUNNING_TEXT as STOP_REPLY_NOTHING,
    STOP_NEW_TOAST,
    UPDATE_CLOSED_TEXT,
    UPDATE_GONE_TEXT,
    UPDATE_INTERRUPTED_TEXT,
    SessionCommands,
)
from tg_bridge.bot.pickers import STATUS_STOP, STOP_CHOICE_TEXT
from tg_bridge.core.topic_queues import Admission


class RouterRules(CallbackCase):
    """The router itself: allow-list, prefixes, picker kinds and the single error path."""

    def test_refused_tapper_changes_nothing_and_is_logged(self) -> None:
        """A tap from a user outside allow_user_ids gets 'not allowed' and reaches no handler (fakecli P).

        Ported labels:
        P: a shortcut tap from a non-allowlisted user changes nothing
        """
        session = self.start_turn()
        self.tap(f"st:{THREAD}", tapper=7)
        self.assertEqual(self.answers(), ["not allowed"])
        self.assertEqual(session.stops, 0)
        self.assertTrue(self.logged("callback refused from 7"))

    def test_no_readable_config_refuses_every_tap(self) -> None:
        """Ported labels:
        5: no config readable -> tap refused
        5: test home has no config file
        """
        paths = Paths.from_home(self.base)
        self.assertFalse(os.path.exists(paths.config), "test home has no config file")
        router = CallbackRouter(config=Config.load(paths), client=self.telegram, messenger=self.messenger,
                                out=self.out, logger=self.logger)
        QueueTaps(dispatcher=self.dispatcher, live=self.live).register(router)
        self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, "first")
        second = self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, "second").nid
        router.dispatch(self.query(f"drop:{THREAD}:{second}"))
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["first", "second"])
        self.assertEqual(self.answers(), ["not allowed"])

    def test_unknown_prefix_is_logged_and_ignored(self) -> None:
        """An unknown prefix is logged, answered with nothing and changes nothing (design section 2)."""
        self.tap("zz:whatever")
        self.assertEqual(self.answers(), [])
        self.assertEqual(self.calls("editMessageText"), [])
        self.assertTrue(self.logged("callback with unknown prefix 'zz' ignored"))

    def test_unknown_picker_kind_is_logged_and_ignored(self) -> None:
        """Ported labels:
        F: unknown picker kind is ignored
        """
        self.tap(f"cm:{THREAD}:xx:0")
        self.assertEqual(self.answers(), [])
        self.assertTrue(self.logged("picker kind 'xx' is not handled"))

    def test_malformed_picker_callbacks_are_ignored(self) -> None:
        """A non-numeric thread, a missing arg or a missing kind never reach a handler (fakecli F, malformed cm:)."""
        for data in ("cm:abc:st:0", f"cm:{THREAD}:st", f"cm:{THREAD}", f"cm:{THREAD}:mo:"):
            self.tap(data)
        self.assertEqual(self.answers(), [])
        self.assertTrue(self.logged("malformed picker callback ignored"))

    def test_a_failing_handler_is_logged_and_does_not_escape(self) -> None:
        """One try/except around every handler: the exception is logged, the poll loop gets nothing back."""
        def boom(tap: Tap) -> None:
            raise ValueError("kaput")

        self.router.on("boom", boom)
        self.router.dispatch(self.query("boom:1"))
        self.assertTrue(self.logged("answerCallbackQuery: kaput"))

    def test_registries_reject_duplicates_and_reserved_names(self) -> None:
        self.router.on("probe", lambda tap: None)
        with self.assertRaises(ValueError):
            self.router.on("probe", lambda tap: None)
        with self.assertRaises(ValueError):
            self.router.on("cm", lambda tap: None)  # the picker prefix has its own registry
        with self.assertRaises(ValueError):
            self.router.on("a:b", lambda tap: None)
        self.router.on_picker("probe", lambda tap: None)
        with self.assertRaises(ValueError):
            self.router.on_picker("probe", lambda tap: None)

    def test_the_handler_receives_prefix_args_and_message(self) -> None:
        seen: list[Tap] = []
        self.router.on("probe", seen.append)
        self.router.dispatch(self.query("probe:a:b", mid=MID, text="the card"))
        self.assertEqual(seen[0].prefix, "probe")
        self.assertEqual(seen[0].args, ("a", "b"))
        self.assertEqual(seen[0].message_id, MID)
        self.assertEqual(seen[0].message_text, "the card")
        self.assertEqual(seen[0].chat_id, CHAT)


class LiveModeTapTests(CallbackCase):
    """The mode button during a running turn, through a real PermissionService to a fake running session."""

    def setUp(self) -> None:
        super().setUp()
        self.permissions = PermissionService(
            asks=AskRepo(self.db, self.clock), rules=self.rules,
            cards=ApprovalCards(self.messenger, self.logger, self.clock), messenger=self.messenger,
            workers=self.out, waiting=mock.Mock(), logger=self.logger)
        self.router = CallbackRouter(config=self.config, client=self.telegram, messenger=self.messenger,
                                     out=self.out, logger=self.logger)
        PickerTaps(config=self.config, sessions=self.sessions, live=self.live, backends=self.backends,
                   permissions=self.permissions, launchers=self.launchers, stops=self.stops, rules=self.rules,
                   folders=self.folders, logger=self.logger).register(self.router)

    def test_mode_button_during_a_turn_reaches_the_running_session(self) -> None:
        """Ported labels:
        E2: running session object gets the new mode
        """
        session = FakeBackend().open_session(fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        feed = object()
        self.assertTrue(self.live.begin_turn(THREAD, session, feed))
        binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.project, rules=ClaudeRules(),
                              mode="bypassPermissions")
        self.permissions.bind_turn(binding)
        self.permissions.attach(binding, session)
        self.tap(f"cm:{THREAD}:md:3")  # the claude picker: bypass, acceptEdits, default, plan
        self.assertEqual(self.answers(), ["🛡 mode set to plan"])
        self.assertTrue(wait_until(lambda: ("set_mode", "plan") in session.calls, 5), session.calls)
        self.assertEqual(binding.mode, "plan")
        self.assertEqual(self.sessions.get(THREAD).mode, "plan")
        self.live.end_turn(THREAD, session, feed)


class QueueTapTests(CallbackCase):
    """now / jump / drop on the queued messages of the topic (the Dispatcher's queue, driven directly)."""

    def queue_two(self) -> tuple[str, str]:
        """The first message starts the (idle) topic's runner; the second waits behind it as a queued item."""
        first = self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, "first")
        second = self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, "second")
        self.assertEqual(first.admission, Admission.STARTED)
        self.assertEqual(second.admission, Admission.QUEUED)
        return first.nid or "", second.nid or ""

    def test_now_moves_the_item_to_front_and_interrupts_the_turn(self) -> None:
        """⏭ Now: the item runs next, the running session is stopped on its own thread, the message is edited.

        Ported labels:
        ⏭: queued item moved to front
        """
        _, second = self.queue_two()
        session = self.start_turn()
        self.tap(f"now:{THREAD}:{second}")
        self.assertTrue(session.stopped.wait(5), "the running turn was not stopped")
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["second", "first"])
        self.assertEqual(self.answers(), ["Interrupting — your item runs now ⏭"])
        self.assertEqual(self.edits()[-1]["text"], NOW_NOTE)
        self.assertEqual(self.edits()[-1]["reply_markup"], json.dumps({"inline_keyboard": []}))

    def test_now_for_an_item_no_longer_queued_only_says_so(self) -> None:
        session = self.start_turn()
        self.queue_two()
        self.tap(f"now:{THREAD}:deadbeef00")
        self.assertEqual(self.answers(), [NOT_QUEUED_TEXT])
        self.assertEqual(session.stops, 0)
        self.assertEqual(self.edits(), [])

    def test_jump_moves_the_item_to_front_without_stopping(self) -> None:
        _, second = self.queue_two()
        session = self.start_turn()
        self.tap(f"jump:{THREAD}:{second}")
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["second", "first"])
        self.assertEqual(self.answers(), ["Moved to front ✅"])
        self.assertEqual(self.edits()[-1]["text"], JUMP_NOTE)
        self.assertEqual(session.stops, 0)

    def test_drop_removes_the_item_and_edits_the_message(self) -> None:
        """Ported labels:
        5: owner tap applied
        """
        _, second = self.queue_two()
        self.tap(f"drop:{THREAD}:{second}")
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["first"])
        self.assertEqual(self.answers(), ["Dropped 🗑"])
        self.assertEqual(self.edits()[-1]["text"], DROP_NOTE)

    def test_a_stranger_drop_changes_nothing(self) -> None:
        """Ported labels:
        5: stranger gets answerCallbackQuery "not allowed"
        5: stranger tap changed nothing
        """
        _, second = self.queue_two()
        self.tap(f"drop:{THREAD}:{second}", tapper=7)
        self.assertEqual(self.answers(), ["not allowed"])
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["first", "second"])

    def test_drop_of_an_unknown_item_says_not_queued(self) -> None:
        self.queue_two()
        self.tap(f"drop:{THREAD}:nope")
        self.assertEqual(self.answers(), [NOT_QUEUED_TEXT])
        self.assertEqual(self.edits(), [])

    def test_a_queue_tap_without_an_item_id_is_ignored(self) -> None:
        self.queue_two()
        self.tap(f"now:{THREAD}")
        self.assertEqual(self.answers(), [])

    def queue_five(self) -> list[str]:
        """Five queued messages 'item 0' .. 'item 4' of the topic; the first one starts (idle topic), the rest wait."""
        results = [self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, f"item {n}") for n in range(5)]
        self.assertEqual([r.admission for r in results][1:], [Admission.QUEUED] * 4)
        return [r.nid or "" for r in results]

    def list_lines(self) -> list[str]:
        return self.edits()[-1]["text"].splitlines()

    def list_rows(self) -> list[list[dict]]:
        return json.loads(self.edits()[-1]["reply_markup"])["inline_keyboard"]

    def test_qdrop_on_item_three_edits_the_list_to_four_rows(self) -> None:
        """The /queue list keeps working after one tap: its message becomes the 4-item list, 4 button rows, and the
        other rows keep their own q-kinds (not the per-message now / jump / drop)."""
        nids = self.queue_five()
        self.tap(f"qdrop:{THREAD}:{nids[2]}")
        self.assertEqual([i.text for i in self.dispatcher.queue_snapshot(THREAD)], ["item 0", "item 1", "item 3", "item 4"])
        self.assertEqual(self.answers(), ["Dropped 🗑"])
        self.assertEqual(self.list_lines(), ["📥 4 queued:", "1. item 0", "2. item 1", "3. item 3", "4. item 4"])
        rows = self.list_rows()
        self.assertEqual(len(rows), 4)
        callbacks = [b["callback_data"] for row in rows for b in row]
        self.assertIn(f"qdrop:{THREAD}:{nids[3]}", callbacks)
        self.assertNotIn(f"qdrop:{THREAD}:{nids[2]}", callbacks)
        self.assertTrue(all(c.startswith(("qnow:", "qjump:", "qdrop:")) for c in callbacks), callbacks)

    def test_qjump_and_qnow_rerender_the_list_in_the_new_order(self) -> None:
        nids = self.queue_five()
        session = self.start_turn()
        self.tap(f"qjump:{THREAD}:{nids[3]}")
        self.assertEqual(session.stops, 0)
        self.assertEqual(self.answers()[-1], "Moved to front ✅")
        self.assertEqual(self.list_lines()[1:], ["1. item 3", "2. item 0", "3. item 1", "4. item 2", "5. item 4"])
        self.assertEqual(len(self.list_rows()), 5)

        self.tap(f"qnow:{THREAD}:{nids[4]}")
        self.assertTrue(session.stopped.wait(5), "the running turn was not stopped")
        self.assertEqual(self.answers()[-1], "Interrupting — your item runs now ⏭")
        self.assertEqual(self.list_lines()[1:], ["1. item 4", "2. item 3", "3. item 0", "4. item 1", "5. item 2"])
        self.assertEqual(self.list_lines()[0], "📥 5 queued:")

    def test_the_last_qdrop_edits_the_list_to_the_empty_text(self) -> None:
        nid = self.dispatcher.deliver(THREAD, TOPIC, CHAT, "sender", None, "only").nid or ""
        self.tap(f"qdrop:{THREAD}:{nid}")
        self.assertEqual(self.edits()[-1]["text"], "the queue is empty.")
        self.assertEqual(self.edits()[-1]["reply_markup"], json.dumps({"inline_keyboard": []}))

    def test_a_list_tap_on_an_item_already_gone_rerenders_without_it(self) -> None:
        """An item taken out of the queue by someone else (a running turn took it): the tap says so, and the list is
        re-rendered as it is now, without that item and without changing anything else."""
        nids = self.queue_five()
        self.dispatcher.drop(THREAD, nids[1])
        session = self.start_turn()
        self.tap(f"qnow:{THREAD}:{nids[1]}")
        self.assertEqual(self.answers()[-1], NOT_QUEUED_TEXT)
        self.assertEqual(session.stops, 0)
        self.assertEqual(self.list_lines(), ["📥 4 queued:", "1. item 0", "2. item 2", "3. item 3", "4. item 4"])
        self.assertEqual(len(self.list_rows()), 4)

        self.tap(f"qdrop:{THREAD}:{nids[1]}")
        self.assertEqual(self.answers()[-1], NOT_QUEUED_TEXT)
        self.assertEqual(self.list_lines()[0], "📥 4 queued:")
        self.assertEqual(len(self.list_rows()), 4)


class StopTapTests(CallbackCase):
    """The feed's ⏹ Stop (st:<thread>) and the /status Stop button share SessionCommands.stop_turn with /stop."""

    def test_stop_button_stops_the_running_turn_once_and_sends_nothing(self) -> None:
        """fakecli I: the Stop tap stops the turn, answers 'Stopping ⏹' and posts no new message.

        Ported labels:
        I: Stop tap stops the running turn once (1)
        I: stop is marked as requested on purpose
        I: toast answers the tap
        I: no new message in the topic
        G: no live session left

        Ported labels:
        G: Stop tap stops the live session in the background
        G: thread marked as stopped by the button
        G: toast sent
        """
        session = self.start_turn()
        session.on_stop = lambda: self.end_turn(session)
        self.tap(f"st:{THREAD}")
        self.assertTrue(session.stopped.wait(5))
        self.assertEqual(session.stops, 1)
        self.assertTrue(self.live.is_aborted(THREAD))
        self.assertEqual(self.answers(), [STOP_TOAST])
        self.assertEqual(self.sent_texts(), [])
        self.assertIsNone(self.live.running(THREAD))

    def test_stop_button_with_nothing_running_says_so(self) -> None:
        """Ported labels:
        I: Stop with no turn gets a toast

        Ported labels:
        G: tap with no running turn does nothing
        """
        self.tap(f"st:{THREAD}")
        self.assertEqual(self.answers(), [NOTHING_RUNNING_TEXT])
        self.assertFalse(self.live.is_aborted(THREAD))
        self.assertIsNone(self.live.running(THREAD))

    def test_stop_button_and_slash_stop_call_the_same_stop_turn(self) -> None:
        """Design: the st handler and /stop call the same method (asserted via spy). The turn ends at the first stop,
        so the /stop that follows finds nothing running and says so."""
        session = self.start_turn()
        session.on_stop = lambda: self.end_turn(session)
        with mock.patch.object(self.stops, "stop_turn", wraps=self.stops.stop_turn) as spy:
            self.tap(f"st:{THREAD}")
            self.assertTrue(session.stopped.wait(5))
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(spy.call_args.args, (THREAD,))
            self.assertTrue(self.commands.dispatch(self.deps, CHAT, THREAD, TOPIC, "/stop"))
            self.assertTrue(self.out.join(5))
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(session.stops, 1)
        self.assertIn(STOP_REPLY_NOTHING, self.sent_texts())

    def test_status_stop_button_uses_the_same_method(self) -> None:
        session = self.start_turn()
        with mock.patch.object(self.stops, "stop_turn", wraps=self.stops.stop_turn) as spy:
            self.tap(f"cm:{THREAD}:{STATUS_STOP}:0")
            self.assertTrue(session.stopped.wait(5))
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(self.answers(), [STOP_TOAST])


class StatusTapTests(CallbackCase):
    """The /status shortcut kinds (fakecli H, K, L, M, O) and the stop-and-new button."""

    def test_new_button_runs_slash_new_and_moves_the_id(self) -> None:
        """fakecli J: 🆕 New behaves like /new, the old id moves to prev_uuid, the reply is the /new text.

        Ported labels:
        J: the /new reply is the usual text ('🆕 the next message starts a fresh session. /resume <uuid> brings the old one back.')
        """
        self.sessions.set_uuid(THREAD, U1)
        self.tap(f"cm:{THREAD}:sn:0")
        row = self.sessions.get(THREAD)
        self.assertEqual((row.uuid, row.prev_uuid), ("", U1))
        self.assertEqual(self.sent_texts(), [NEW_TEXT])
        self.assertEqual(self.answers(), ["🆕 fresh session next"])

    def test_new_button_while_running_offers_the_stop_and_new_button(self) -> None:
        """fakecli M: /new is refused during a turn and the reply carries the stop-and-new button.

        Ported labels:
        M: /new while running says so
        M: nothing changed yet
        M: the reply carries the stop-and-fresh button
        """
        self.sessions.set_uuid(THREAD, U1)
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:sn:0")
        self.assertEqual(self.sessions.get(THREAD).uuid, U1)
        self.assertIn(NEW_BUSY_TEXT, self.sent_texts())
        self.assertEqual(self.callbacks_of(self.sent_buttons()[-1]["reply_markup"]), [f"cm:{THREAD}:nw:0"])
        self.assertEqual(session.stops, 0)

    def test_stop_and_new_stops_then_starts_fresh_in_the_same_message(self) -> None:
        """fakecli M and O: nw stops the turn, the wait ends, /new runs and the same message says so.

        Ported labels:
        M: the same message is edited ('🆕 stopped; the next message starts a fresh session')
        M: the button is removed
        M: the new session is started after the stop
        M: the turn had ended before /new ran
        M: toast answers the tap
        """
        self.sessions.set_uuid(THREAD, U1)
        session = self.start_turn()
        session.on_stop = lambda: self.end_turn(session)
        self.tap(f"cm:{THREAD}:nw:0")
        self.assertEqual(self.answers(), [STOP_NEW_TOAST])
        self.assertTrue(wait_until(lambda: any(p.get("text") == NEW_DONE_TEXT for p in self.edits()), 5))
        self.assertEqual(self.sessions.get(THREAD).uuid, "")
        self.assertEqual(self.sessions.get(THREAD).prev_uuid, U1)
        self.assertEqual(self.edits()[-1]["reply_markup"], json.dumps({"inline_keyboard": []}))
        self.assertIsNone(self.live.running(THREAD))
        self.assertEqual(self.answers(), [STOP_NEW_TOAST])

    def test_stop_and_new_with_nothing_running_edits_at_once(self) -> None:
        """Ported labels:
        O: /new applied
        """
        self.sessions.set_uuid(THREAD, U1)
        self.tap(f"cm:{THREAD}:nw:0")
        self.assertTrue(wait_until(lambda: any(p.get("text") == NEW_DONE_TEXT for p in self.edits()), 5))
        self.assertEqual(self.sessions.get(THREAD).uuid, "")

    def test_cost_button_runs_slash_cost(self) -> None:
        """Ported labels:
        K: 💰 Cost sends the /cost text
        """
        self.deps.turns.insert(THREAD, started="", ended="", status="ok", cost=0.25, cost_known=True,
                               in_tok=1200, out_tok=40, backend="claude")
        self.tap(f"cm:{THREAD}:sc:0")
        self.assertEqual(self.answers(), ["💰"])
        self.assertEqual(self.sent_texts()[-1:], ["1 turns · ↑1.2k ↓40 tokens · cost $0.25"])

    def test_mode_button_sends_the_mode_picker_as_a_new_message(self) -> None:
        """fakecli L: 🛡 Mode sends the picker as a new message and marks the current mode.

        Ported labels:
        L: 🛡 Mode sends the mode picker ('🛡 mode: acceptEdits')
        L: the /status message is left as it is
        """
        self.sessions.set_mode(THREAD, "acceptEdits")
        self.tap(f"cm:{THREAD}:sm:0")
        self.assertEqual(self.answers(), ["🛡 mode"])
        sent = self.sent_buttons()[-1]
        self.assertEqual(sent["text"], "🛡 mode: acceptEdits")
        self.assertIn("✓ acceptEdits", self.labels_of(sent["reply_markup"]))
        self.assertEqual(self.edits(), [], "the /status message is left as it is")

    def test_model_button_sends_the_model_picker(self) -> None:
        """Ported labels:
        L: 🧠 Model sends the model picker ('🧠 model: haiku')
        """
        self.sessions.set_model(THREAD, "haiku")
        self.tap(f"cm:{THREAD}:so:0")
        self.assertEqual(self.answers(), ["🧠 model"])
        sent = self.sent_buttons()[-1]
        self.assertEqual(sent["text"], "🧠 model: haiku")
        self.assertIn(f"cm:{THREAD}:mo:qwen3:8b", self.callbacks_of(sent["reply_markup"]))

    def test_model_picker_sent_from_status_persists_and_switches_the_live_session(self) -> None:
        """Ported labels:
        L: the picker sent from /status persists the choice
        L: and switches the running session
        """
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:so:0")
        self.tap(f"cm:{THREAD}:mo:opus")
        self.assertEqual(self.sessions.get(THREAD).model, "opus")
        self.assertTrue(wait_until(lambda: session.models == ["opus"], 5), session.models)
        self.end_turn(session)

    def test_launcher_button_sends_the_launcher_picker(self) -> None:
        self.tap(f"cm:{THREAD}:sl:0")
        self.assertEqual(self.answers(), ["🚀 launcher"])
        self.assertTrue(self.sent_buttons()[-1]["text"].startswith("🚀 launcher: wrap"))

    def test_status_button_for_a_topic_without_session_says_so(self) -> None:
        self.tap("cm:9699:sn:0")
        self.assertEqual(self.answers(), [NO_SESSION_TEXT])
        self.assertEqual(self.sent_texts(), [])

    def test_status_button_from_a_stranger_changes_nothing(self) -> None:
        """fakecli P: a non-allowed user cannot use the shortcut buttons.

        Ported labels:
        P: a shortcut tap from a non-allowlisted user changes nothing
        P: and gets the usual refusal
        """
        self.sessions.set_uuid(THREAD, U1)
        self.tap(f"cm:{THREAD}:sn:0", tapper=7)
        self.assertEqual(self.sessions.get(THREAD).uuid, U1)
        self.assertEqual(self.answers(), ["not allowed"])


class KeptStopTapTests(CallbackCase):
    """The stop buttons of a topic with a kept session (section 5 D2): the feed's and the status ⏹ post the choice card
    during a background update, and the card's two buttons interrupt the update or close the session."""

    def setUp(self) -> None:
        super().setUp()
        self.linger = StubKeptLinger()
        self.stops = SessionCommands(config=self.config, sessions=self.sessions, history=self.history,
                                     live=self.live, backends=self.backends, messenger=self.messenger,
                                     clock=self.clock, logger=self.logger, stop_wait=0.6,
                                     handoff=HandoffService(self.kv, self.backends, self.logger), linger=self.linger)
        self.router = CallbackRouter(config=self.config, client=self.telegram, messenger=self.messenger,
                                     out=self.out, logger=self.logger)
        StopTap(stops=self.stops).register(self.router)
        StopChoiceTaps(stops=self.stops).register(self.router)
        StatusTaps(deps=self.deps, registry=self.commands, stops=self.stops).register(self.router)
        QueueTaps(dispatcher=self.dispatcher, live=self.live, linger=self.linger).register(self.router)

    def test_status_stop_button_closes_a_kept_session(self) -> None:
        """Nothing runs but the session is kept: the status ⏹ asks it to close and answers 'Stopping ⏹'."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=False)
        self.linger.release_answer = True
        self.tap(f"cm:{THREAD}:{STATUS_STOP}:0")
        self.assertEqual(self.answers(), [STOP_TOAST])
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_status_stop_button_during_a_self_turn_posts_the_choice_card(self) -> None:
        """The status ⏹ during a background update posts the same choice card as the feed's ⏹ and stops nothing."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:{STATUS_STOP}:0")
        self.assertEqual(self.answers(), [CHOOSE_TOAST])
        cards = [p for p in self.sent_buttons() if p.get("text") == STOP_CHOICE_TEXT]
        self.assertEqual(len(cards), 1)
        self.assertEqual(session.stops, 0)
        self.assertEqual(self.linger.requests, [])

    def test_feed_stop_during_a_self_turn_posts_the_choice_card(self) -> None:
        """The feed's ⏹ during a background update posts the card and stops nothing before a choice."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        session = self.start_turn()
        self.tap(f"st:{THREAD}")
        self.assertEqual(self.answers(), [CHOOSE_TOAST])
        cards = [p for p in self.sent_buttons() if p.get("text") == STOP_CHOICE_TEXT]
        self.assertEqual(len(cards), 1)
        self.assertEqual(self.callbacks_of(cards[0]["reply_markup"]),
                         [f"cm:{THREAD}:sq:u:3:2", f"cm:{THREAD}:sq:c:3:2"])
        self.assertEqual(session.stops, 0)
        self.assertEqual(self.linger.requests, [])

    def test_choice_update_button_interrupts_the_update_only(self) -> None:
        """'this update only' interrupts the update, edits the card and leaves the session running."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        self.linger.interrupt_answer = True
        self.linger.self_turn_now = (3, 2)
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:sq:u:3:2")
        self.assertEqual(self.answers(), [INTERRUPT_TOAST])
        self.assertEqual(self.edits()[-1]["text"], UPDATE_INTERRUPTED_TEXT)
        self.assertEqual(self.linger.requests, [("interrupt", THREAD)])
        self.assertEqual(session.stops, 0)

    def test_choice_update_button_after_the_update_ended_is_expired(self) -> None:
        """A card tapped after its update ended says the choice expired, edits nothing and interrupts nothing."""
        self.linger.interrupt_answer = True
        self.tap(f"cm:{THREAD}:sq:u:3:2")
        self.assertEqual(self.answers(), [CHOICE_EXPIRED_TOAST])
        self.assertEqual(self.edits(), [])
        self.assertEqual(self.linger.requests, [])

    def test_choice_close_button_stops_the_session(self) -> None:
        """'close the session' stops the update and its session, as the stop did before D2."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        self.linger.self_turn_now = (3, 2)
        session = self.start_turn()
        session.on_stop = lambda: self.end_turn(session)
        self.tap(f"cm:{THREAD}:sq:c:3:2")
        self.assertTrue(session.stopped.wait(5), "the close button did not stop the session")
        self.assertEqual(self.answers(), [STOP_TOAST])
        self.assertEqual(self.edits()[-1]["text"], UPDATE_CLOSED_TEXT)
        self.assertEqual(session.stops, 1)

    def test_choice_close_button_after_the_update_ended_is_expired(self) -> None:
        """A card closed after its update ended says the choice expired and stops nothing, not even a kept session."""
        self.linger.release_answer = True
        self.tap(f"cm:{THREAD}:sq:c:3:2")
        self.assertEqual(self.answers(), [CHOICE_EXPIRED_TOAST])
        self.assertEqual(self.edits(), [])
        self.assertEqual(self.linger.requests, [])

    def test_a_stale_choice_card_expires_and_stops_nothing(self) -> None:
        """A card posted for self turn 2 of hold 3 expires after that turn ended, after a new self turn started, after
        its hold ended and during a user turn. Both buttons answer 'this choice has expired', edit nothing and stop
        nothing, while the running turn goes on."""
        ended = LingerStatus(pending=1, seconds_left=60.0, self_turn=False, hold_id=3, self_seq=2)
        cases = {
            "self turn ended": (ended, None),
            "new self turn started": (LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=3),
                                     (3, 3)),
            "hold ended": (None, None),
            "user turn on the kept session": (ended, None),
        }
        for name, (status, running) in cases.items():
            with self.subTest(name):
                self.linger.status_value = status
                self.linger.self_turn_now = running
                session = self.start_turn()
                for choice in ("u", "c"):
                    self.tap(f"cm:{THREAD}:sq:{choice}:3:2")
                    self.assertEqual(self.answers()[-1], CHOICE_EXPIRED_TOAST)
                self.assertEqual(self.edits(), [])
                self.assertEqual(self.linger.requests, [])
                self.assertEqual(session.stops, 0)
                self.assertFalse(session.stopped.is_set())
                self.end_turn(session)

    def test_a_choice_card_without_hold_numbers_expires(self) -> None:
        """A card in the old format (no hold id or self turn) is expired too, and it stops nothing."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        self.linger.self_turn_now = (3, 2)
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:sq:c")
        self.assertEqual(self.answers(), [CHOICE_EXPIRED_TOAST])
        self.assertEqual(self.linger.requests, [])
        self.assertFalse(session.stopped.is_set())
        self.end_turn(session)

    def test_now_during_a_self_turn_interrupts_the_update_only(self) -> None:
        """⏭ Now during a background update interrupts that update, not the session: the item runs on the same process."""
        session = self.start_turn()
        self.linger.interrupt_answer = True
        with mock.patch.object(self.dispatcher, "to_front", return_value=True):
            self.tap(f"now:{THREAD}:n1")
        self.assertEqual(self.linger.requests, [("interrupt", THREAD)])
        self.assertEqual(session.stops, 0)
        self.assertFalse(session.stopped.is_set())

    def test_now_when_no_update_can_be_interrupted_stops_the_turn_as_before(self) -> None:
        """When the interrupt does not apply (a user turn on a kept process, or no kept session), ⏭ Now takes the old
        path: the turn is aborted and its session is stopped."""
        session = self.start_turn()
        self.linger.interrupt_answer = False
        with mock.patch.object(self.dispatcher, "to_front", return_value=True):
            self.tap(f"now:{THREAD}:n1")
        self.assertEqual(self.linger.requests, [("interrupt", THREAD)])
        self.assertTrue(session.stopped.wait(5), "the old path did not stop the session")
        self.assertEqual(session.stops, 1)
        self.assertTrue(self.live.is_aborted(THREAD))


if __name__ == "__main__":
    unittest.main()
