"""TurnRunner end to end: one real turn through ClaudeBackend + fake_claude and one through CodexBackend + fake_codex.

Both fakes are child processes speaking the real protocols (no network, no real claude or codex binary). The runner is
built the way the App builds it: the default backend registry, a named launcher picked for the topic, the real
repositories, and the same permission service. The feed is a recording double. The clock is the FakeClock, moved by
join_turn() so a dead process is still noticed.
"""

import json
import os
import sys
import unittest

import tests.fakes as fakes_package
from tests.fakes.telegram import FIRST_MESSAGE_ID
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, join_turn, run_async
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.core.turn_listeners import slash_key
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.feed.feed import TurnFeed

FAKES = os.path.dirname(os.path.abspath(fakes_package.__file__))
FAKE_CLAUDE = os.path.join(FAKES, "fake_claude.py")
FAKE_CODEX = os.path.join(FAKES, "fake_codex.py")


def shown_texts(tg, thread_id: int) -> list[str]:
    """What the topic shows: every message sent to it, with its last applied edit, HTML unescaped. The fake numbers
    its messages in send order, so the n-th sendMessage is FIRST_MESSAGE_ID + n (no send fails in these tests)."""
    shown, message_id = [], FIRST_MESSAGE_ID
    for call in tg.calls:
        if call.method != "sendMessage":
            continue
        message_id += 1
        if str(call.params.get("message_thread_id")) != str(thread_id):
            continue
        edits = tg.edits(message_id)
        text = edits[-1] if edits else call.params["text"]
        text = text.replace("<blockquote expandable>", "").replace("</blockquote>", "")
        shown.append(text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    return shown


class ClaudeEndToEndTests(unittest.TestCase):
    """The Claude adapter with the fake CLI: bypass mode, the 'text' scenario (the ask is answered by the bridge)."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        command = f"{sys.executable} -I {FAKE_CLAUDE}"
        self.h = TurnHarness(self.home.home, dict(launchers={"fake": {"cmd": command,
                                                                       "config_dir": self.home.home}}),
                             base_env={"PATH": os.environ.get("PATH", ""), "FAKE_NOWAIT": "1"}, poll_seconds=0.01)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fake")  # a named Claude launcher from config

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def test_usage_is_stored_and_the_answer_is_posted_once_with_a_real_feed(self) -> None:
        # the fake result carries cost and usage, and its answer text comes as an assistant message first. Before the
        # end gate, the interrupt that followed the finished turn made the CLI answer again: the usage was replaced by
        # an empty one (zero tokens, no cost) and the held answer was posted to the topic as a second message.
        self.h.base_env = dict(self.h.base_env, FAKE_TEXT="real answer", FAKE_USAGE="1")
        self.h.feeds = lambda chat, thread, parked: TurnFeed(self.h.tg, self.h.logger, self.h.clock, chat, thread,
                                                             parked=parked)
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.answer, reply.err, reply.outcome), ("real answer", None, TurnOutcome.OK))
        row = self.h.db.query("SELECT status, in_tok, out_tok, cost, cost_known FROM turns WHERE thread_id=?",
                              (THREAD,))[0]
        self.assertEqual((row["status"], row["in_tok"], row["out_tok"], row["cost"]), ("ok", 100, 40, 0.5))
        self.assertTrue(row["cost_known"])
        posted = [text for text in self.h.tg.texts(THREAD) if "real answer" in text]
        self.assertEqual(posted, [], "the answer is returned to the caller, not posted again by the feed")

    def test_stored_input_is_the_fresh_input_and_the_cache_reads_stay_out_of_it(self) -> None:
        # the live file stores the fresh input in in_tok; the cache reads (80000) are counted in the footer only
        self.h.base_env = dict(self.h.base_env, FAKE_TEXT="cached answer", FAKE_CACHE="1")
        self.h.feeds = lambda chat, thread, parked: TurnFeed(self.h.tg, self.h.logger, self.h.clock, chat, thread,
                                                             parked=parked)
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        join_turn(self.h.clock, thread)
        self.assertEqual(box[0].answer, "cached answer")
        row = self.h.db.query("SELECT in_tok, out_tok, cost, cost_known FROM turns WHERE thread_id=?", (THREAD,))[0]
        self.assertEqual((row["in_tok"], row["out_tok"], row["cost"], row["cost_known"]), (12, 40, None, 1))

    def test_the_feed_posts_the_text_then_one_tool_log_with_the_summary_header(self) -> None:
        """Ported labels:
        H: turn answered ('ok:allow', None)
        H: the text is its own message first
        H: the log ends with the summary header
        H: the log lists the tools, the failed read marked
        H: the answer is not echoed here (the dispatcher sends it), subagent text is not shown
        H: no separate footer message
        H: feed and session released after the turn
        """
        base = {key: value for key, value in self.h.base_env.items() if key != "FAKE_NOWAIT"}
        self.h.base_env = dict(base, FAKE_FEED="1")  # the ask is auto-allowed in bypass, so the turn runs to its result
        feeds = []

        def real_feed(chat, thread, parked):
            feeds.append(TurnFeed(self.h.tg, self.h.logger, self.h.clock, chat, thread, parked=parked))
            return feeds[-1]

        self.h.feeds = real_feed
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        join_turn(self.h.clock, thread)
        self.assertEqual((box[0].answer, box[0].err), ("ok:allow", None))
        self.assertFalse(feeds[-1]._thread.is_alive(), "the feed's sender is still running after the turn")
        shown = shown_texts(self.h.tg, THREAD)
        self.assertEqual(shown[0], "💬 Let me check.")
        head, _, body = shown[1].partition("\n")
        self.assertTrue(head.startswith("✅ 3 tools · "), head)
        self.assertEqual(body.split("\n"), ["❌ Read x.txt", "🔧 Bash List files", "  ↳ 🔧 Glob **/*.py"])
        self.assertEqual(len(shown), 2)
        self.assertFalse(any("ok:allow" in text for text in shown))
        self.assertIsNone(self.h.live.running(THREAD))

    def test_slash_commands_from_init_are_stored_for_the_topic(self) -> None:
        """Ported labels:
        L: a real turn with the fake CLI completes (err=None)
        L: slash_commands from init are stored in kv slash:<thread>
        """
        self.h.base_env["FAKE_SLASH"] = json.dumps(["compact", "review"])  # the fake reports these in its init event
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hi", sender="tester")
        join_turn(self.h.clock, thread)
        self.assertIsNone(box[0].err)
        self.assertEqual(json.loads(self.h.kv.get(slash_key(THREAD))), ["compact", "review"])

    def test_one_claude_turn_answers_and_stores_the_session(self) -> None:
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.err, reply.outcome), (None, TurnOutcome.OK))
        self.assertEqual(reply.answer, "ok:unanswered")  # the fake's result line for the 'text' scenario
        stored = self.h.sessions.get(THREAD).uuid
        self.assertTrue(stored)
        self.assertIsNotNone(self.h.history.find(THREAD, stored))
        row = self.h.db.query("SELECT status, backend FROM turns WHERE thread_id=?", (THREAD,))[0]
        self.assertEqual((row["status"], row["backend"]), ("ok", "claude"))
        self.assertTrue(self.h.feeds.feeds[-1].started and self.h.feeds.feeds[-1].closed)
        self.assertIsNone(self.h.live.running(THREAD))


class CodexEndToEndTests(unittest.TestCase):
    """The Codex adapter with the fake app-server: normal mode, one answer, the thread id learned from thread/start."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        command = f"{sys.executable} -I {FAKE_CODEX}"
        self.h = TurnHarness(self.home.home, dict(agent_launchers={"fakecodex": {
                                 "backend": "codex", "command": command, "config_dir": self.home.home}}),
                             base_env={"PATH": os.environ.get("PATH", ""), "FAKE_CODEX_MODE": "normal",
                                       "FAKE_CODEX_ANSWER": "codex answer"}, poll_seconds=0.01)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fakecodex")

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def test_one_codex_turn_answers_and_learns_the_thread_id(self) -> None:
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hi", sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.err, reply.outcome), (None, TurnOutcome.OK))
        self.assertEqual(reply.answer, "codex answer")
        self.assertTrue(self.h.sessions.get(THREAD).uuid, "the thread id is stored on SessionStarted")
        row = self.h.db.query("SELECT status, backend FROM turns WHERE thread_id=?", (THREAD,))[0]
        self.assertEqual((row["status"], row["backend"]), ("ok", "codex"))
        self.assertEqual(self.h.history.find(THREAD, self.h.sessions.get(THREAD).uuid).backend, "codex")


if __name__ == "__main__":
    unittest.main()
