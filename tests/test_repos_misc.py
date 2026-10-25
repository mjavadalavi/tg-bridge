"""Repositories part 2 (STEP 12): TurnRepo, AskRepo, RuleRepo and InboxRepo on a real migrated Database.

Every method is covered under a TempHome with FakeClock. The /cost totals are checked against the SQL the live file
runs (copied verbatim from its /cost branch) on the same rows. The answer and ingest idempotence rules are checked
directly and with four threads racing on one row.
"""

import json
import threading
import unittest

from tests.fakes.clock import DEFAULT_START, FakeClock
from tests.support import TempHome
from tg_bridge.clock import utc_stamp
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRecord, AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import InboxRecord, InboxRepo
from tg_bridge.store.rules import RuleRecord, RuleRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.store.turns import CostTotals, TurnRepo

DAY = 86400
STAMP = utc_stamp(DEFAULT_START)

# The /cost query of the live file (tg-bridge.py, cmd == 'cost'), copied verbatim.
LIVE_COST_SQL = ('SELECT COUNT(*) n, SUM(cost) cost, SUM(in_tok) tin, SUM(out_tok) tout, '
                 'SUM(CASE WHEN cost_known=0 THEN 1 ELSE 0 END) unknown FROM turns WHERE thread_id=?')


class MiscCase(unittest.TestCase):
    """Opens a migrated Database under a TempHome for each test."""

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock()
        self.db = Database(Paths.from_home(self.home))
        self.turns = TurnRepo(self.db)
        self.sessions = SessionRepo(self.db, self.clock)
        self.asks = AskRepo(self.db, self.clock)
        self.rules = RuleRepo(self.db, self.clock)
        self.inbox = InboxRepo(self.db, self.clock)

    def tearDown(self) -> None:
        self.db.close()
        self._home.__exit__(None, None, None)

    def _turn(self, thread_id: int, **overrides: object) -> int:
        fields = {"started": STAMP, "ended": STAMP, "status": "ok"}
        fields.update(overrides)
        return self.turns.insert(thread_id, **fields)

    def _ask(self, nonce: str, thread_id: int = 7, **overrides: object) -> None:
        fields = {"nonce": nonce, "thread_id": thread_id, "chat_id": -100, "request_id": "req-" + nonce,
                  "tool_name": "Bash", "input": {"command": "ls"}, "suggestions": [{"type": "addRules"}]}
        fields.update(overrides)
        self.asks.insert(**fields)


class TurnRepoTests(MiscCase):
    def test_insert_returns_row_id_and_defaults_to_claude_backend(self) -> None:
        first = self._turn(7, cost=0.5, in_tok=10, out_tok=4)
        second = self._turn(7, cost=0.25, in_tok=1, out_tok=2)
        self.assertEqual(second, first + 1)
        row = self.db.query("SELECT * FROM turns WHERE id=?", (first,))[0]
        self.assertEqual((row["backend"], row["cost_known"], row["status"]), ("claude", 1, "ok"))

    def test_insert_stores_unknown_cost_as_null_with_cost_known_zero(self) -> None:
        self._turn(7, cost=None, cost_known=False, in_tok=3, out_tok=5)
        row = self.db.query("SELECT cost, cost_known FROM turns WHERE thread_id=7")[0]
        self.assertEqual((row["cost"], row["cost_known"]), (None, 0))

    def test_insert_stamps_ended_as_last_active_of_its_topic_only(self) -> None:
        self.sessions.ensure(7, "topic 7", "claude --alpha", "")
        self.sessions.ensure(8, "topic 8", "claude --alpha", "")
        ended = "2026-10-24T10:00:00+00:00"
        self._turn(7, started=STAMP, ended=ended)
        rows = self.db.query("SELECT thread_id, last_active FROM sessions ORDER BY thread_id")
        self.assertEqual([(row["thread_id"], row["last_active"]) for row in rows], [(7, ended), (8, "")])

    def test_insert_for_topic_without_session_row_still_stores_the_turn(self) -> None:
        turn_id = self._turn(9)
        self.assertEqual(self.db.query("SELECT COUNT(*) n FROM turns WHERE id=?", (turn_id,))[0]["n"], 1)
        self.assertEqual(self.db.query("SELECT COUNT(*) n FROM sessions")[0]["n"], 0)

    def test_totals_sums_cost_tokens_and_counts_turns(self) -> None:
        self._turn(7, cost=0.5, in_tok=10, out_tok=4)
        self._turn(7, cost=0.25, in_tok=1, out_tok=2)
        self._turn(8, cost=9.0, in_tok=100, out_tok=100)
        self.assertEqual(self.turns.totals(7), CostTotals(turns=2, cost=0.75, tokens_in=11, tokens_out=6, unknown=0))

    def test_totals_counts_unknown_cost_turns_and_keeps_known_cost_sum(self) -> None:
        self._turn(7, cost=0.5, in_tok=10, out_tok=4)
        self._turn(7, cost=None, cost_known=False, in_tok=None, out_tok=None)
        totals = self.turns.totals(7)
        self.assertEqual((totals.turns, totals.cost, totals.tokens_in, totals.tokens_out, totals.unknown),
                         (2, 0.5, 10, 4, 1))

    def test_totals_of_topic_without_turns_is_zero_with_cost_none(self) -> None:
        self.assertEqual(self.turns.totals(99), CostTotals(turns=0, cost=None, tokens_in=0, tokens_out=0, unknown=0))

    def test_totals_all_unknown_costs_reports_cost_none_like_live_cost_command(self) -> None:
        self._turn(7, cost=None, cost_known=False)
        self._turn(7, cost=None, cost_known=False)
        totals = self.turns.totals(7)
        self.assertIsNone(totals.cost)
        self.assertEqual(totals.unknown, 2)

    def test_totals_match_the_live_cost_query_on_the_same_rows(self) -> None:
        self._turn(7, cost=0.5, in_tok=10, out_tok=4)
        self._turn(7, cost=None, cost_known=False, in_tok=7, out_tok=None)
        self._turn(7, cost=0.125, in_tok=None, out_tok=3, backend="codex")
        live = self.db.query(LIVE_COST_SQL, (7,))[0]
        totals = self.turns.totals(7)
        self.assertEqual((totals.turns, totals.unknown), (live["n"], live["unknown"]))
        self.assertAlmostEqual(totals.cost, live["cost"])
        self.assertEqual((totals.tokens_in, totals.tokens_out), (live["tin"], live["tout"]))

    def test_summary_by_backend_splits_claude_and_codex(self) -> None:
        self._turn(7, cost=0.5, in_tok=10, out_tok=4)
        self._turn(7, cost=None, cost_known=False, in_tok=2, out_tok=2, backend="codex")
        self._turn(7, cost=0.125, in_tok=3, out_tok=1, backend="codex")
        self._turn(8, cost=7.0, backend="claude")
        summary = self.turns.summary_by_backend(7)
        self.assertEqual(sorted(summary), ["claude", "codex"])
        self.assertEqual(summary["claude"], CostTotals(turns=1, cost=0.5, tokens_in=10, tokens_out=4, unknown=0))
        self.assertEqual(summary["codex"], CostTotals(turns=2, cost=0.125, tokens_in=5, tokens_out=3, unknown=1))

    def test_summary_by_backend_empty_topic_is_empty_dict(self) -> None:
        self.assertEqual(self.turns.summary_by_backend(42), {})


class AskRepoTests(MiscCase):
    def test_insert_then_by_nonce_round_trips_fields_as_pending(self) -> None:
        self._ask("n1", input={"command": "ls -la"}, suggestions=[{"type": "addRules"}])
        record = self.asks.by_nonce("n1")
        self.assertIsInstance(record, AskRecord)
        self.assertEqual((record.thread_id, record.chat_id, record.request_id, record.tool_name),
                         (7, -100, "req-n1", "Bash"))
        self.assertEqual(record.tool_input(), {"command": "ls -la"})
        self.assertEqual(json.loads(record.suggestions), [{"type": "addRules"}])
        self.assertIsNone(record.message_id)
        self.assertEqual(record.created, STAMP)
        self.assertTrue(record.pending)

    def test_insert_stores_non_ascii_input_unescaped(self) -> None:
        self._ask("n1", input={"command": "echo سلام"})
        self.assertIn("سلام", self.asks.by_nonce("n1").input)

    def test_by_nonce_returns_none_for_unknown_nonce(self) -> None:
        self.assertIsNone(self.asks.by_nonce("missing"))

    def test_tool_input_reads_non_object_or_bad_json_as_empty_dict(self) -> None:
        self._ask("bad", input={})
        self.db.execute("UPDATE asks SET input='not json' WHERE nonce='bad'")
        self.assertEqual(self.asks.by_nonce("bad").tool_input(), {})
        self.db.execute("UPDATE asks SET input='[1, 2]' WHERE nonce='bad'")
        self.assertEqual(self.asks.by_nonce("bad").tool_input(), {})

    def test_set_message_id_stores_card_message(self) -> None:
        self._ask("n1")
        self.asks.set_message_id("n1", 555)
        self.assertEqual(self.asks.by_nonce("n1").message_id, 555)
        self.asks.set_message_id("missing", 1)  # unknown nonce: ignored, no error

    def test_answer_returns_true_once_and_second_answer_returns_false_without_changing_verdict(self) -> None:
        self._ask("n1")
        self.assertTrue(self.asks.answer("n1", "allow"))
        self.assertFalse(self.asks.answer("n1", "deny"))
        record = self.asks.by_nonce("n1")
        self.assertEqual(record.answer, "allow")
        self.assertFalse(record.pending)

    def test_answer_for_unknown_nonce_returns_false(self) -> None:
        self.assertFalse(self.asks.answer("missing", "allow"))

    def test_answer_rejects_empty_or_pending_verdict(self) -> None:
        self._ask("n1")
        with self.assertRaises(ValueError):
            self.asks.answer("n1", "")
        self.assertTrue(self.asks.by_nonce("n1").pending)

    def test_answer_is_exactly_once_under_four_threads(self) -> None:
        self._ask("race")
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(4)

        def tap() -> None:
            barrier.wait()
            outcome = self.asks.answer("race", "allow")
            with results_lock:
                results.append(outcome)

        threads = [threading.Thread(target=tap) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(sorted(results), [False, False, False, True])
        self.assertEqual(self.asks.by_nonce("race").answer, "allow")

    def test_pending_lists_oldest_first_and_filters_by_thread(self) -> None:
        self._ask("a", thread_id=7)
        self.clock.advance(1)
        self._ask("b", thread_id=8)
        self.clock.advance(1)
        self._ask("c", thread_id=7)
        self.asks.answer("c", "deny")
        self.assertEqual([r.nonce for r in self.asks.pending()], ["a", "b"])
        self.assertEqual([r.nonce for r in self.asks.pending(7)], ["a"])
        self.assertEqual(self.asks.pending(9), [])

    def test_void_all_for_one_thread_closes_only_its_pending_asks(self) -> None:
        self._ask("a", thread_id=7)
        self._ask("b", thread_id=8)
        self._ask("c", thread_id=7)
        self.asks.answer("c", "allow")
        self.assertEqual(self.asks.void_all(7), ["a"])
        self.assertEqual(self.asks.by_nonce("a").answer, "void")
        self.assertEqual(self.asks.by_nonce("c").answer, "allow")
        self.assertTrue(self.asks.by_nonce("b").pending)

    def test_void_all_without_thread_closes_every_pending_ask(self) -> None:
        self._ask("a", thread_id=7)
        self._ask("b", thread_id=8)
        self.assertEqual(sorted(self.asks.void_all()), ["a", "b"])
        self.assertEqual(self.asks.pending(), [])
        self.assertEqual(self.asks.void_all(), [])

    def test_answer_after_void_returns_false(self) -> None:
        self._ask("a")
        self.asks.void_all()
        self.assertFalse(self.asks.answer("a", "allow"))
        self.assertEqual(self.asks.by_nonce("a").answer, "void")


class RuleRepoTests(MiscCase):
    def test_add_returns_true_for_new_rule_and_false_for_duplicate(self) -> None:
        self.assertTrue(self.rules.add(7, "Edit"))
        self.assertFalse(self.rules.add(7, "Edit"))
        self.assertTrue(self.rules.add(8, "Edit"))  # the same text in another topic is a different rule
        self.assertEqual(len(self.db.query("SELECT id FROM perm_rules")), 2)

    def test_for_thread_lists_rules_in_added_order_with_zero_hits(self) -> None:
        self.rules.add(7, "Edit")
        self.clock.advance(5)
        self.rules.add(7, "Bash(git status *)")
        self.rules.add(8, "Write")
        records = self.rules.for_thread(7)
        self.assertEqual([r.rule for r in records], ["Edit", "Bash(git status *)"])
        self.assertIsInstance(records[0], RuleRecord)
        self.assertEqual((records[0].hits, records[0].created), (0, STAMP))
        self.assertEqual(self.rules.for_thread(99), [])

    def test_texts_returns_only_rule_strings(self) -> None:
        self.rules.add(7, "Edit")
        self.rules.add(7, "Bash(ls *)")
        self.assertEqual(self.rules.texts(7), ["Edit", "Bash(ls *)"])

    def test_hit_counts_uses_and_ignores_unknown_rules(self) -> None:
        self.rules.add(7, "Edit")
        self.rules.add(8, "Edit")
        self.rules.hit(7, "Edit")
        self.rules.hit(7, "Edit")
        self.rules.hit(7, "Write")  # not a saved rule of this topic
        self.rules.hit(8, "Edit")  # the same rule text in another topic counts on that topic only
        self.assertEqual(self.rules.for_thread(7)[0].hits, 2)
        self.assertEqual(self.rules.for_thread(8)[0].hits, 1)

    def test_delete_removes_one_rule_by_number_and_only_for_its_topic(self) -> None:
        self.rules.add(7, "Edit")
        self.rules.add(7, "Write")
        rule_id = self.rules.for_thread(7)[0].id
        self.assertFalse(self.rules.delete(8, rule_id))  # wrong topic
        self.assertTrue(self.rules.delete(7, rule_id))
        self.assertFalse(self.rules.delete(7, rule_id))  # already gone
        self.assertEqual(self.rules.texts(7), ["Write"])

    def test_delete_all_removes_only_the_topic_rules_and_counts_them(self) -> None:
        self.rules.add(7, "Edit")
        self.rules.add(7, "Write")
        self.rules.add(8, "Edit")
        self.assertEqual(self.rules.delete_all(7), 2)
        self.assertEqual(self.rules.delete_all(7), 0)
        self.assertEqual(self.rules.texts(8), ["Edit"])


class InboxRepoTests(MiscCase):
    def test_ingest_stores_pending_row_with_payload_as_json(self) -> None:
        update = {"update_id": 10, "message": {"text": "hi سلام"}}
        self.assertTrue(self.inbox.ingest(10, update, thread_id=7, chat_id=-100, sender="42"))
        record = self.inbox.get(10)
        self.assertIsInstance(record, InboxRecord)
        self.assertEqual((record.thread_id, record.chat_id, record.sender, record.state),
                         (7, -100, "42", "pending"))
        self.assertEqual(json.loads(record.payload), update)
        self.assertEqual(record.created, STAMP)

    def test_ingest_of_duplicate_update_id_returns_false_and_keeps_first_row(self) -> None:
        self.assertTrue(self.inbox.ingest(10, {"update_id": 10, "message": {"text": "first"}}))
        self.assertFalse(self.inbox.ingest(10, {"update_id": 10, "message": {"text": "second"}}))
        self.assertIn("first", self.inbox.get(10).payload)
        self.assertEqual(len(self.db.query("SELECT id FROM inbox_queue")), 1)

    def test_ingest_accepts_missing_thread_and_chat(self) -> None:
        self.assertTrue(self.inbox.ingest(11, {"update_id": 11}))
        record = self.inbox.get(11)
        self.assertEqual((record.thread_id, record.chat_id, record.sender), (None, None, ""))

    def test_get_returns_none_for_unknown_update(self) -> None:
        self.assertIsNone(self.inbox.get(999))

    def test_pending_lists_pending_rows_in_update_id_order(self) -> None:
        self.inbox.ingest(12, {"update_id": 12})
        self.inbox.ingest(10, {"update_id": 10})
        self.inbox.ingest(11, {"update_id": 11})
        self.assertEqual([r.update_id for r in self.inbox.pending()], [10, 11, 12])

    def test_done_removes_row_from_pending_and_unknown_done_is_ignored(self) -> None:
        self.inbox.ingest(10, {"update_id": 10})
        self.inbox.done(10)
        self.inbox.done(777)
        self.assertEqual(self.inbox.get(10).state, "done")
        self.assertEqual(self.inbox.pending(), [])

    def test_mark_pending_puts_done_row_back_into_pending(self) -> None:
        self.inbox.ingest(10, {"update_id": 10})
        self.inbox.done(10)
        self.inbox.mark_pending(10)
        self.assertEqual([r.update_id for r in self.inbox.pending()], [10])

    def test_prune_deletes_done_rows_older_than_seven_days_only(self) -> None:
        self.inbox.ingest(1, {"update_id": 1})
        self.inbox.done(1)
        self.clock.advance(3 * DAY)
        self.inbox.ingest(2, {"update_id": 2})
        self.inbox.done(2)
        self.clock.advance(5 * DAY)  # row 1 is 8 days old, row 2 is 5 days old
        self.assertEqual(self.inbox.prune(), 1)
        self.assertIsNone(self.inbox.get(1))
        self.assertEqual(self.inbox.get(2).state, "done")

    def test_prune_keeps_a_row_exactly_seven_days_old_and_never_deletes_pending(self) -> None:
        self.inbox.ingest(1, {"update_id": 1})
        self.inbox.done(1)
        self.inbox.ingest(2, {"update_id": 2})  # pending, never done
        self.clock.advance(7 * DAY)
        self.assertEqual(self.inbox.prune(), 0)
        self.clock.advance(1)
        self.assertEqual(self.inbox.prune(), 1)
        self.assertEqual(self.inbox.get(2).state, "pending")


class ConcurrentIngestTests(MiscCase):
    def test_four_threads_ingesting_one_update_store_exactly_one_row(self) -> None:
        barrier = threading.Barrier(4)
        outcomes: list[bool] = []
        outcomes_lock = threading.Lock()

        def deliver() -> None:
            barrier.wait()
            stored = self.inbox.ingest(50, {"update_id": 50})
            with outcomes_lock:
                outcomes.append(stored)

        threads = [threading.Thread(target=deliver) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(sorted(outcomes), [False, False, False, True])
        self.assertEqual(len(self.db.query("SELECT id FROM inbox_queue WHERE update_id=50")), 1)


if __name__ == "__main__":
    unittest.main()
