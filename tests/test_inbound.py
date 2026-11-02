"""inbound: port of the handle_message and _continue_message scenarios of fakecli/test_inbox.py and fakecli/test_modes.py
(C18), with the two new services the step adds: the tmux seats and the media deferral.

Scenario labels follow the live order: allowlist and learning, the General guard, topic resolution (session row, config
topic, spawn alias, default topic, unknown), the awaited 'Other' answer, pane topics, auto on/off (stored mode and a
running turn), bridge commands (never queued), the queue (position notice, Now/Next/Drop buttons, the 5-message limit),
and media (the reserved place, a failed download, a new topic, a pane topic).
Every test runs on a migrated database in a TempHome with FakeTelegram and FakeClock; tmux is a recording stub.
"""

from tg_bridge.bot.commands.info import register_info_commands
from tg_bridge.bot.inbound import AUTO_OFF_NEXT_TEXT, AUTO_OFF_TEXT, AUTO_ON_TEXT, Disposition
from tg_bridge.core.media import MediaOutcome
from tg_bridge.core.permissions import BYPASS_MODE
from tg_bridge.core.topics import GENERAL_TEXT, NO_TOPIC_TEXT
from tests.inbound_support import (
    CHAT,
    CLAUDE_CMD,
    PANE,
    PANE_TOPIC,
    SPAWN_CMD,
    SPAWN_CWD,
    SPAWN_LAUNCHER,
    STRANGER_UID,
    THREAD,
    TIMEOUT,
    TOPIC,
    TUI_CMD,
    TUI_TOPIC,
    WAIT_PANE,
    InboundCase,
)

QUEUE_FULL = "📥 queue full (5) — wait for the current turn to finish."
QUEUED_SECOND = "📥 queued — position 2 after the current turn."
QUEUED_TERMINAL_SECOND = ("📥 queued — position 2; the topic's terminal session (tg-bridge view) is busy "
                          "or has text typed in it.")
SPAWNED = "✅ spawned budget-claude in tmux window 'mytask' ({pane}) — your message is its first prompt."


class GuardTests(InboundCase):
    def test_stranger_is_ignored_and_logged(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("hi", uid=STRANGER_UID)), Disposition.DONE)
        self.assertEqual(self.sent(), [])
        self.assertIsNone(self.row())
        self.assertTrue(self.logged(f"IGNORED message from tester ({STRANGER_UID}) — not allow-listed"))

    def test_topic_created_is_learned_even_from_a_refused_user(self) -> None:
        created = {"forum_topic_created": {"name": "budget.task"}}
        self.assertIs(self.handle(self.message("", uid=STRANGER_UID, **created)), Disposition.DONE)
        self.assertEqual(self.kv.get(f"thread:{THREAD}"), "budget.task")

    def test_topic_created_by_the_owner_asks_for_a_folder_and_nothing_else(self) -> None:
        calls: list[tuple] = []
        self.folders.suggest_for_new_topic = lambda cfg, chat, thread, name: calls.append((chat, thread, name))
        created = {"forum_topic_created": {"name": "side.docs"}}
        self.assertIs(self.handle(self.message("", **created)), Disposition.DONE)
        self.assertEqual(calls, [(CHAT, THREAD, "side.docs")])
        self.assertEqual(self.sent(), [])

    def test_general_topic_gets_the_board_text(self) -> None:
        self.assertIs(self.handle(self.message("hi", thread=1)), Disposition.DONE)
        self.assertEqual(self.sent(1), [GENERAL_TEXT])
        self.assertIsNone(self.sessions.get(1))

    def test_group_message_without_a_topic_gets_the_board_text(self) -> None:
        self.handle(self.message("hi", thread=None))  # a reply without a thread is recorded under 'None'
        self.assertEqual(self.sent(None), [GENERAL_TEXT])

    def test_private_chat_without_a_thread_gets_the_add_me_text(self) -> None:
        self.handle(self.message("hi", thread=None, chat_type="private"))
        self.assertEqual(self.sent(None), [NO_TOPIC_TEXT])

    def test_private_chat_in_a_topic_continues(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("hi", chat_type="private")), Disposition.DEFERRED)
        self.assertEqual(self.row().name, TOPIC)

    def test_empty_text_without_media_is_dropped(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("   ")), Disposition.DONE)
        self.assertIsNone(self.row())
        self.assertEqual(self.starts.started, [])


class RefusedTests(InboundCase):
    def _config_data(self) -> dict:
        return {"allow_user_ids": []}

    def test_empty_allowlist_refuses_and_logs(self) -> None:
        self.assertIs(self.handle(self.message("hi")), Disposition.DONE)
        self.assertTrue(self.logged("REFUSED message from tester (42) — allowlist not provisioned"))
        self.assertIsNone(self.row())


class DefaultTopicTests(InboundCase):
    def _config_data(self) -> dict:
        data = super()._config_data()
        data["default_topic"] = {"launcher": CLAUDE_CMD, "cwd": self.dir_a}
        return data

    def test_default_topic_makes_a_fresh_topic_its_own_session(self) -> None:
        self.name_topic(THREAD, "fresh")
        self.assertIs(self.handle(self.message("hi")), Disposition.DEFERRED)
        row = self.row()
        self.assertEqual((row.name, row.launcher, row.cwd), ("fresh", CLAUDE_CMD, self.dir_a))


class AcceptEditsDefaultTests(InboundCase):
    def _config_data(self) -> dict:
        data = super()._config_data()
        data["default_mode"] = "acceptEdits"
        return data

    def test_new_topic_starts_in_the_configured_default_mode(self) -> None:
        """Ported labels:
        B: new topic 9203 starts in acceptEdits (got 'acceptEdits')
        """
        self.name_topic(THREAD, TOPIC)
        self.handle(self.message("hi"))
        self.assertEqual(self.row().mode, "acceptEdits")

    def test_auto_off_returns_to_the_configured_default(self) -> None:
        """Ported labels:
        G: auto off -> configured default (got 'acceptEdits')
        """
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, BYPASS_MODE)
        self.handle(self.message("auto off"))
        self.assertEqual(self.row().mode, "acceptEdits")


class ResolutionTests(InboundCase):
    def test_unknown_topic_is_told_and_gets_no_row(self) -> None:
        self.name_topic(THREAD, "mystery")
        self.assertIs(self.handle(self.message("hi")), Disposition.DONE)
        self.assertEqual(len(self.sent()), 1)
        self.assertIn('Unknown topic "mystery" (thread 9701).', self.sent()[0])
        self.assertIn("Known: budget, seat, side", self.sent()[0])
        self.assertIsNone(self.row())
        self.assertEqual(self.starts.started, [])

    def test_configured_topic_creates_the_row_and_starts_the_runner(self) -> None:
        """Ported labels:
        B: new topic 9204 starts in bypassPermissions (got 'bypassPermissions')
        """
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("hi")), Disposition.DEFERRED)
        row = self.row()
        self.assertEqual((row.name, row.launcher, row.cwd, row.mode), (TOPIC, CLAUDE_CMD, self.project, BYPASS_MODE))
        self.assertEqual(self.starts.started, [(THREAD, TOPIC)])
        self.assertEqual(self.sent(), [])

    def test_existing_row_wins_over_an_unknown_learned_name(self) -> None:
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project)
        self.name_topic(THREAD, "mystery")
        self.assertIs(self.handle(self.message("hi")), Disposition.DEFERRED)
        self.assertEqual(self.starts.started, [(THREAD, TOPIC)])
        self.assertEqual(self.sent(), [])

    def test_kept_folder_is_applied_to_the_first_message(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.folders.pending_set(THREAD, self.dir_a)
        self.assertIsNone(self.row())  # the first message creates the session row, in the tapped folder
        self.handle(self.message("hi"))
        self.assertEqual(self.row().cwd, self.dir_a)
        self.assertEqual(self.folders.pending_get(THREAD), "")

    def test_spawn_alias_opens_a_seat_and_binds_its_pane(self) -> None:
        self.name_topic(THREAD, "budget.mytask")
        self.assertIs(self.handle(self.message("hello")), Disposition.DONE)
        self.assertEqual(self.seats.spawned, [(SPAWN_CMD, "mytask", "budget.mytask", "Tester", "hello")])
        self.assertEqual(self.kv.get("pane:budget.mytask"), WAIT_PANE)
        row = self.row()
        self.assertEqual((row.name, row.launcher, row.cwd), ("budget.mytask", SPAWN_LAUNCHER, SPAWN_CWD))
        self.assertEqual(self.sent(), [SPAWNED.format(pane=WAIT_PANE)])
        self.assertTrue(self.logged("spawn topic budget.mytask: typed into pane %9"))

    def test_failed_spawn_is_reported_and_leaves_no_row(self) -> None:
        self.seats.spawn_result = (False, "spawn failed: no server running", None)
        self.name_topic(THREAD, "budget.mytask")
        self.handle(self.message("hello"))
        self.assertEqual(self.sent(), ["⚠️ spawn failed: no server running"])
        self.assertIsNone(self.row())
        self.assertIsNone(self.kv.get("pane:budget.mytask"))

    def test_spawn_alias_with_a_blank_task_is_an_unknown_topic(self) -> None:
        self.name_topic(THREAD, "budget.   ")
        self.handle(self.message("hello"))
        self.assertEqual(self.seats.spawned, [])
        self.assertIn('Unknown topic "budget.   "', self.sent()[0])

    def test_awaited_other_answer_comes_before_everything(self) -> None:
        """Ported labels:
        K: nothing queued (None)
        """
        self.permissions.awaited = True
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("my own answer")), Disposition.DONE)
        self.assertEqual(self.permissions.texts, [(THREAD, "my own answer")])
        self.assertIsNone(self.row())
        self.assertEqual(self.sent(), [])


class PaneTests(InboundCase):
    def test_configured_pane_topic_types_the_text(self) -> None:
        self.name_topic(THREAD, PANE_TOPIC)
        self.assertIs(self.handle(self.message("ls -la")), Disposition.DONE)
        self.assertEqual(self.seats.typed, [(PANE, "ls -la")])
        self.assertEqual(self.sent(), ["✅ → seat-label"])
        self.assertEqual(self.queued(), [])

    def test_spawned_pane_binding_types_into_its_pane(self) -> None:
        self.sessions.ensure(THREAD, "budget.mytask", SPAWN_LAUNCHER, SPAWN_CWD)
        self.kv.set("pane:budget.mytask", WAIT_PANE)
        self.name_topic(THREAD, "budget.mytask")
        self.handle(self.message("go"))
        self.assertEqual(self.seats.typed, [(WAIT_PANE, "go")])
        self.assertEqual(self.sent(), [f"✅ → pane {WAIT_PANE}"])

    def test_failed_type_is_reported(self) -> None:
        self.seats.type_result = (False, "pane %7 is not alive")
        self.name_topic(THREAD, PANE_TOPIC)
        self.handle(self.message("ls"))
        self.assertEqual(self.sent(), ["⚠️ pane %7 is not alive"])

    def test_pane_topic_takes_commands_and_auto_words_as_plain_text(self) -> None:
        self.name_topic(THREAD, PANE_TOPIC)
        self.handle(self.message("/ping"))
        self.handle(self.message("auto on"))
        self.assertEqual(self.seats.typed, [(PANE, "/ping"), (PANE, "auto on")])
        self.assertEqual(self.sent(), ["✅ → seat-label", "✅ → seat-label"])
        self.assertEqual(self.row().mode, BYPASS_MODE)


class AutoModeTests(InboundCase):
    def setUp(self) -> None:
        super().setUp()
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "default")

    def test_auto_on_stores_bypass_and_confirms(self) -> None:
        """Ported labels:
        D: auto on stores bypassPermissions
        D: confirmation sent
        D: no control request without a live session
        """
        self.assertIs(self.handle(self.message("auto on")), Disposition.DONE)
        self.assertEqual(self.row().mode, BYPASS_MODE)
        self.assertEqual(self.sent(), [AUTO_ON_TEXT])
        self.assertEqual(self.permissions.live_modes, [])

    def test_auto_off_in_any_case_returns_to_default_when_default_is_bypass(self) -> None:
        self.sessions.set_mode(THREAD, BYPASS_MODE)
        self.handle(self.message("Auto OFF"))
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.sent(), [AUTO_OFF_TEXT])

    def test_auto_on_during_a_live_turn_switches_the_session(self) -> None:
        """Ported labels:
        E: mode stored for later turns
        """
        self.start_turn(THREAD)
        self.handle(self.message("auto on"))
        self.assertTrue(self.permissions.switched.wait(TIMEOUT))
        self.assertEqual(self.permissions.live_modes, [(THREAD, BYPASS_MODE, "auto on")])
        self.assertEqual(self.row().mode, BYPASS_MODE)

    def test_auto_off_during_a_live_turn_switches_back(self) -> None:
        """Ported labels:
        F: mode stored as default
        """
        self.sessions.set_mode(THREAD, BYPASS_MODE)
        self.start_turn(THREAD)
        self.handle(self.message("auto off"))
        self.assertTrue(self.permissions.switched.wait(TIMEOUT))
        self.assertEqual(self.permissions.live_modes, [(THREAD, "default", "auto off")])
        self.assertEqual(self.row().mode, "default")

    def test_auto_with_extra_words_is_an_ordinary_message(self) -> None:
        self.assertIs(self.handle(self.message("auto on please")), Disposition.DEFERRED)
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.sent(), [])


class AutoOffReplyTests(InboundCase):
    """The 'auto off' reply names the next message when the running turn's CLI cannot switch its mode live."""

    def test_tui_turn_running_says_approvals_return_from_the_next_message(self) -> None:
        self.sessions.ensure(THREAD, TUI_TOPIC, TUI_CMD, self.project, "default")
        self.sessions.set_mode(THREAD, BYPASS_MODE)
        self.start_turn(THREAD)
        self.handle(self.message("auto off"))
        self.assertTrue(self.permissions.switched.wait(TIMEOUT))
        self.assertEqual(self.permissions.live_modes, [(THREAD, "default", "auto off")])
        self.assertEqual(self.sent(), [AUTO_OFF_NEXT_TEXT])

    def test_tui_topic_without_a_running_turn_gets_the_plain_reply(self) -> None:
        self.sessions.ensure(THREAD, TUI_TOPIC, TUI_CMD, self.project, "default")
        self.sessions.set_mode(THREAD, BYPASS_MODE)
        self.handle(self.message("auto off"))
        self.assertEqual(self.sent(), [AUTO_OFF_TEXT])

    def test_print_topic_with_a_running_turn_gets_the_plain_reply(self) -> None:
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "default")
        self.sessions.set_mode(THREAD, BYPASS_MODE)
        self.start_turn(THREAD)
        self.handle(self.message("auto off"))
        self.assertTrue(self.permissions.switched.wait(TIMEOUT))
        self.assertEqual(self.sent(), [AUTO_OFF_TEXT])


class CommandTests(InboundCase):
    def test_registered_command_runs_now_and_is_never_queued(self) -> None:
        """Ported labels:
        M: a command is answered without entering the queue
        M: the status answer went out
        """
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("/ping now")), Disposition.DONE)
        self.assertEqual(self.sent(), ["pong now"])
        self.assertEqual(self.queued(), [])
        self.assertEqual(self.starts.started, [])
        register_info_commands(self.commands)  # /status is an info command; the harness starts with /ping only
        self.assertIs(self.handle(self.message("/status")), Disposition.DONE)
        self.settle()
        self.assertIn("topic: side", self.sent()[-1])
        self.assertEqual(self.queued(), [])

    def test_command_word_with_a_bot_name_runs(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.handle(self.message("/PING@MyBot"))
        self.assertEqual(self.sent(), ["pong"])

    def test_unknown_slash_word_reaches_the_agent(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.assertIs(self.handle(self.message("/unknown x")), Disposition.DEFERRED)
        self.assertEqual([item.text for item in self.queued()], ["/unknown x"])


class QueueTests(InboundCase):
    def setUp(self) -> None:
        super().setUp()
        self.name_topic(THREAD, TOPIC)

    def test_first_message_starts_the_runner_without_a_notice(self) -> None:
        self.assertIs(self.handle(self.message("one")), Disposition.DEFERRED)
        self.assertEqual(self.starts.started, [(THREAD, TOPIC)])
        self.assertEqual(self.sent(), [])
        self.assertEqual([item.text for item in self.queued()], ["one"])

    def test_second_message_is_queued_with_now_next_drop_buttons(self) -> None:
        self.handle(self.message("one"))
        self.assertIs(self.handle(self.message("two")), Disposition.DEFERRED)
        self.assertEqual(self.sent(), [QUEUED_SECOND])
        row = self.keyboards()[0][0]
        self.assertEqual([button["text"] for button in row], ["⏭ Now", "⏩ Next", "❌ Drop"])
        nid = self.queued()[1].nid
        self.assertEqual([button["callback_data"] for button in row],
                         [f"now:{THREAD}:{nid}", f"jump:{THREAD}:{nid}", f"drop:{THREAD}:{nid}"])
        self.assertEqual(self.starts.started, [(THREAD, TOPIC)])

    def test_a_message_queued_while_the_topic_waits_for_its_terminal_says_so(self) -> None:
        self.dispatcher.waiting_terminal = lambda thread: True  # the runner is in the terminal wait (see test_terminal_attach)
        self.handle(self.message("one"))
        self.assertIs(self.handle(self.message("two")), Disposition.DEFERRED)
        self.assertEqual(self.sent(), [QUEUED_TERMINAL_SECOND])
        self.assertEqual(len(self.keyboards()[0][0]), 3, "the queued notice lost its Now/Next/Drop buttons")

    def test_messages_keep_their_order(self) -> None:
        for text in ("a", "b", "c"):
            self.handle(self.message(text))
        self.assertEqual([item.text for item in self.queued()], ["a", "b", "c"])

    def test_sixth_message_is_refused_when_the_queue_is_full(self) -> None:
        for index in range(5):
            self.handle(self.message(f"m{index}"))
        self.assertIs(self.handle(self.message("m5")), Disposition.DONE)
        self.assertEqual(self.sent()[-1], QUEUE_FULL)
        self.assertEqual(len(self.queued()), 5)


class MediaTests(InboundCase):
    PHOTO = [{"file_id": "f1"}]

    def setUp(self) -> None:
        super().setUp()
        self.name_topic(THREAD, TOPIC)

    def test_photo_in_a_session_topic_fills_its_reserved_place(self) -> None:
        self.assertIs(self.handle(self.message("", photo=self.PHOTO), update_id=501), Disposition.DEFERRED)
        self.assertEqual(self.media.calls, [("", False)])  # Claude takes photos as a text note, not as images
        self.assertEqual([item.text for item in self.queued()], ["heard you"])
        self.assertEqual(self.starts.started, [(THREAD, TOPIC)])

    def test_photo_on_a_codex_topic_asks_for_native_images(self) -> None:
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project)
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.media.outcome = MediaOutcome(text="look", images=("/tmp/p.jpg",))
        self.handle(self.message("look", photo=self.PHOTO))
        self.assertEqual(self.media.calls, [("look", True)])

    def test_photo_follows_the_launcher_when_the_stored_backend_is_stale(self) -> None:
        """A topic moved back to Claude by its launcher pick keeps the stored backend 'codex': the photo is a text note."""
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project)
        self.sessions.set_launcher_pick(THREAD, "wrap")
        self.sessions.set_backend(THREAD, "codex")
        self.handle(self.message("", photo=self.PHOTO))
        self.assertEqual(self.media.calls, [("", False)])

    def test_failed_download_releases_the_place_and_marks_the_update_done(self) -> None:
        self.inbox.ingest(601, {"update_id": 601})
        self.media.outcome = None
        self.handle(self.message("", photo=self.PHOTO), update_id=601)
        self.assertEqual(self.queued(), [])
        self.assertEqual(self.inbox.get(601).state, "done")

    def test_media_error_releases_the_place(self) -> None:
        self.media.error = RuntimeError("boom")
        self.handle(self.message("", photo=self.PHOTO))
        self.assertEqual(self.queued(), [])
        self.assertTrue(self.logged(f"media for thread {THREAD} failed: boom"))

    def test_media_in_a_new_topic_is_resolved_by_the_worker(self) -> None:
        other = 9750
        self.name_topic(other, TOPIC)
        self.handle(self.message("", thread=other, photo=self.PHOTO))
        self.assertEqual(self.sessions.get(other).name, TOPIC)
        self.assertEqual([item.text for item in self.dispatcher.queue_snapshot(other)], ["heard you"])

    def test_media_queue_full_is_refused_before_any_download(self) -> None:
        for index in range(5):
            self.handle(self.message(f"m{index}"))
        self.handle(self.message("", photo=self.PHOTO))
        self.assertEqual(self.sent()[-1], QUEUE_FULL)
        self.assertEqual(self.media.calls, [])

    def test_media_in_a_pane_topic_is_typed_into_the_pane(self) -> None:
        self.name_topic(THREAD, PANE_TOPIC)
        self.handle(self.message("", photo=self.PHOTO))
        self.assertEqual(self.seats.typed, [(PANE, "heard you")])
        self.assertEqual(self.queued(), [])
        self.assertEqual(self.starts.started, [])  # a pane is typed into, no turn runs
