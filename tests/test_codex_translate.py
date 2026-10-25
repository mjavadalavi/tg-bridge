"""CodexTranslator (agents/codex/translate.py): recorded probe frames, synthetic frames and the fake app-server.

tests/fixtures/codex_run1.jsonl (a command turn that completes) and codex_run2.jsonl (a resumed thread whose command is
declined and then interrupted, plus the start of a second turn) are incoming frames from the 0.162.1 probe, scrubbed of
home paths, account data and MCP startup noise by the probe's own scrub step.

Scenarios: session start and resume; deltas buffered; commentary vs final answer; tool labels and ToolEnd outcomes; a
file change announced once; usage cumulative with cost unknown; willRetry errors (retry and stream_cut, attempts);
compaction; turn results (completed, failed, interrupted, unknown status); resume error; server requests and noise
yield nothing; fake_codex modes retry and resume_missing run through the translator; usage is per turn (the total
minus the baseline at turn/started), so a resumed thread's replayed total is never counted again.
"""

import json
import os
import subprocess
import unittest

from tests.support import TempHome
from tg_bridge.agents.codex.translate import CodexTranslator
from tg_bridge.agents.events import (SessionStarted, StatusEvent, TextEvent, ToolEnd, ToolStart, TurnResult,
                                     TurnStatus, UsageEvent)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
LAUNCHER = os.path.join(HERE, "fakes", "launchers", "codex")
THREAD = "01a16e66-5313-7d21-8e70-bed8d2e619d3"
COMMENTARY_ID = "msg_077eb8fcdc385f0c016adcc64544148196bc916476fa4d3e8d"
FINAL_ID = "msg_077eb8fcdc385f0c016adcc6499e2c81968b1e2947d9e12edb"
EXEC_ID = "exec-61e46054-1060-4ebd-b326-d3c98fbc7862"
DECLINED_ID = "exec-653c133b-2a8a-491c-bf77-b9147d086253"
CWD = "/home/x/proj"


def fixture(name: str) -> list[dict]:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(frames: list[dict], methods: dict[int, str], mode: str = "default", cwd: str | None = None) -> list:
    """Feeds frames in order; methods maps a response id to the request method the caller sent."""
    translator = CodexTranslator(mode=mode, cwd=cwd)
    events: list = []
    for frame in frames:
        events.extend(translator.translate(frame, methods.get(frame.get("id"))))
    return events


def note(method: str, params: dict) -> dict:
    return {"method": method, "params": params}


def item_started(item: dict) -> dict:
    return note("item/started", {"item": item, "threadId": THREAD, "turnId": "t1"})


def item_completed(item: dict) -> dict:
    return note("item/completed", {"item": item, "threadId": THREAD, "turnId": "t1"})


def command(item_id: str, status: str = "inProgress", exit_code: int | None = None, **extra) -> dict:
    item = {"type": "commandExecution", "id": item_id, "command": "/usr/bin/zsh -lc 'ls'", "cwd": CWD,
            "status": status, "commandActions": [{"type": "unknown", "command": "ls"}], "exitCode": exit_code}
    item.update(extra)
    return item


def message(item_id: str, text: str, phase: str) -> dict:
    return {"type": "agentMessage", "id": item_id, "text": text, "phase": phase}


def turn_completed(status: str, items: list | None = None, error: dict | None = None,
                   duration: int | None = 10) -> dict:
    turn = {"id": "t1", "status": status, "items": items or [], "error": error, "durationMs": duration}
    return note("turn/completed", {"threadId": THREAD, "turn": turn})


class FixtureRunTest(unittest.TestCase):
    METHODS = {1: "initialize", 2: "thread/start", 3: "turn/start"}

    def test_run1_command_turn_events_in_order(self) -> None:
        events = replay(fixture("codex_run1.jsonl"), self.METHODS, mode="default")
        self.assertEqual(events, [
            SessionStarted(session_id=THREAD, model="gpt-6.1-sol", mode="default", cwd="/home/x/proj/w1",
                           resumed=False),
            TextEvent(text="I’ll run the command now.\n", final=False, item_id=COMMENTARY_ID),
            ToolStart(tool_id=EXEC_ID, name="commandExecution", label="$ touch probe_ok.txt",
                      input={"command": "touch probe_ok.txt",
                             "cwd": "/home/x/proj/w1"}),
            ToolEnd(tool_id=EXEC_ID, ok=True),
            UsageEvent(input_tokens=19078, output_tokens=44, cached_tokens=0, cost_known=False,
                       context_window=258400),
            TextEvent(text="done", final=True, item_id=FINAL_ID),
            UsageEvent(input_tokens=38257, output_tokens=49, cached_tokens=18944, cost_known=False,
                       context_window=258400),
            TurnResult(status=TurnStatus.OK, answer="done", duration_ms=15061, subtype="completed"),
        ])

    def test_run2_resumed_thread_declined_command_is_interrupted(self) -> None:
        methods = {2: "thread/resume", 3: "turn/start", 4: "turn/start"}
        events = replay(fixture("codex_run2.jsonl"), methods, mode="default")
        self.assertEqual(events[0], SessionStarted(session_id=THREAD, model="gpt-6.1-sol", mode="default",
                                                   cwd="/home/x/proj/w1", resumed=True))
        self.assertIn(ToolStart(tool_id=DECLINED_ID, name="commandExecution", label="$ touch declined.txt",
                                input={"command": "touch declined.txt", "cwd": "/home/x/proj/w1"}), events)
        self.assertIn(ToolEnd(tool_id=DECLINED_ID, ok=False, detail="declined"), events)
        results = [e for e in events if isinstance(e, TurnResult)]
        self.assertEqual(results, [TurnResult(status=TurnStatus.INTERRUPTED, answer=None, duration_ms=29707,
                                              subtype="interrupted")])
        self.assertFalse(any(isinstance(e, TextEvent) and e.final for e in events))

    def test_noise_and_server_requests_yield_nothing(self) -> None:
        frames = [
            note("thread/started", {"thread": {"id": THREAD}}),
            note("thread/status/changed", {"threadId": THREAD, "status": {"type": "idle"}}),
            note("mcpServer/startupStatus/updated", {"name": "x", "status": "ready"}),
            note("hook/started", {"threadId": THREAD}),
            note("account/rateLimits/updated", {"rateLimits": {}}),
            note("turn/plan/updated", {"explanation": "", "plan": []}),
            note("warning", {"message": "careful"}),
            note("thread/goal/cleared", {"threadId": THREAD}),
            note("totally/unknown", {"x": 1}),
            item_started({"type": "reasoning", "id": "r1", "summary": [], "content": []}),
            item_completed({"type": "userMessage", "id": "u1", "content": []}),
            item_started({"type": "plan", "id": "p1"}),
            item_completed({"type": "hookPrompt", "id": "h1"}),
            item_started({"type": "futureThing", "id": "f1"}),
            {"id": 9, "method": "item/tool/requestUserInput", "params": {"questions": []}},
            {"id": 0, "method": "item/commandExecution/requestApproval", "params": {"command": "ls"}},
            {"method": "serverRequest/resolved", "params": {"requestId": 0}},
        ]
        self.assertEqual(replay(frames, {}), [])


class MessageTest(unittest.TestCase):
    def test_deltas_are_buffered_until_the_item_completes(self) -> None:
        translator = CodexTranslator()
        self.assertEqual(translator.translate(item_started(message("m1", "", "final_answer"))), [])
        for piece in ("He", "llo"):
            self.assertEqual(translator.translate(note("item/agentMessage/delta",
                                                       {"itemId": "m1", "delta": piece})), [])
        self.assertEqual(translator.translate(item_completed(message("m1", "", "final_answer"))),
                         [TextEvent(text="Hello", final=True, item_id="m1")])

    def test_completed_text_wins_over_deltas_and_commentary_is_not_final(self) -> None:
        translator = CodexTranslator()
        translator.translate(note("item/agentMessage/delta", {"itemId": "m2", "delta": "partial"}))
        events = translator.translate(item_completed(message("m2", "full text\n", "commentary")))
        self.assertEqual(events, [TextEvent(text="full text\n", final=False, item_id="m2")])

    def test_blank_text_is_dropped(self) -> None:
        self.assertEqual(CodexTranslator().translate(item_completed(message("m3", "  \n", "commentary"))), [])

    def test_final_answer_falls_back_to_the_turn_summary(self) -> None:
        translator = CodexTranslator()
        events = translator.translate(turn_completed("completed", items=[message("m4", "summary answer",
                                                                                  "final_answer")]))
        self.assertEqual(events, [TurnResult(status=TurnStatus.OK, answer="summary answer", duration_ms=10,
                                             subtype="completed")])

    def test_turn_started_clears_the_previous_answer(self) -> None:
        translator = CodexTranslator()
        translator.translate(item_completed(message("m5", "first turn", "final_answer")))
        translator.translate(note("turn/started", {"threadId": THREAD, "turn": {"id": "t2"}}))
        events = translator.translate(turn_completed("completed"))
        self.assertEqual(events[0].answer, None)


class ToolTest(unittest.TestCase):
    def test_command_label_prefers_command_actions_then_command(self) -> None:
        translator = CodexTranslator()
        first = translator.translate(item_started(command("c1")))
        self.assertEqual(first[0].label, "$ ls")
        bare = {"type": "commandExecution", "id": "c2", "command": "pwd", "status": "inProgress"}
        self.assertEqual(translator.translate(item_started(bare))[0].label, "$ pwd")

    def test_command_exit_code_and_status_decide_the_end(self) -> None:
        translator = CodexTranslator()
        translator.translate(item_started(command("c3")))
        self.assertEqual(translator.translate(item_completed(command("c3", "completed", exit_code=2))),
                         [ToolEnd(tool_id="c3", ok=False, detail="exit 2")])
        translator.translate(item_started(command("c4")))
        self.assertEqual(translator.translate(item_completed(command("c4", "completed", exit_code=0))),
                         [ToolEnd(tool_id="c4", ok=True)])

    def test_completed_without_started_announces_the_tool_first(self) -> None:
        events = CodexTranslator().translate(item_completed(command("c5", "completed", exit_code=0)))
        self.assertEqual(events[0].tool_id, "c5")
        self.assertIsInstance(events[0], ToolStart)
        self.assertEqual(events[1], ToolEnd(tool_id="c5", ok=True))

    def test_file_change_waits_for_paths_and_is_announced_once(self) -> None:
        translator = CodexTranslator(cwd=CWD)
        empty = {"type": "fileChange", "id": "f1", "status": "inProgress", "changes": []}
        self.assertEqual(translator.translate(item_started(empty)), [])
        late = note("item/fileChange/patchUpdated", {"itemId": "f1", "changes": [
            {"path": "/home/x/proj/src/a.py", "kind": {"type": "update"}, "diff": ""}]})
        self.assertEqual(translator.translate(late), [ToolStart(tool_id="f1", name="fileChange",
                                                                  label="✏️ src/a.py",
                                                                  input={"paths": ["/home/x/proj/src/a.py"]})])
        self.assertEqual(translator.translate(late), [])
        failed = translator.translate(item_completed({"type": "fileChange", "id": "f1", "status": "failed",
                                                      "changes": []}))
        self.assertEqual(failed, [ToolEnd(tool_id="f1", ok=False, detail="failed")])

    def test_mcp_and_web_search_labels_and_error_detail(self) -> None:
        translator = CodexTranslator()
        mcp = {"type": "mcpToolCall", "id": "x1", "server": "docs", "tool": "search", "status": "inProgress"}
        self.assertEqual(translator.translate(item_started(mcp)),
                         [ToolStart(tool_id="x1", name="mcp:docs.search", label="🔌 docs.search",
                                    input={"server": "docs", "tool": "search"})])
        done = dict(mcp, status="failed", error={"message": "boom\nsecond line"})
        self.assertEqual(translator.translate(item_completed(done)), [ToolEnd(tool_id="x1", ok=False,
                                                                              detail="boom")])
        web = {"type": "webSearch", "id": "w1", "query": "tg bridge"}
        self.assertEqual(translator.translate(item_started(web)),
                         [ToolStart(tool_id="w1", name="webSearch", label="🔎 tg bridge",
                                    input={"query": "tg bridge"})])


class UsageAndStatusTest(unittest.TestCase):
    def test_usage_is_per_turn_and_never_has_a_cost(self) -> None:
        total = {"inputTokens": 100, "outputTokens": 7, "cachedInputTokens": 40, "totalTokens": 107}
        translator = CodexTranslator()
        translator.translate(note("turn/started", {"threadId": THREAD, "turn": {"id": "t1"}}))
        events = translator.translate(note("thread/tokenUsage/updated", {
            "threadId": THREAD, "tokenUsage": {"total": total, "last": total, "modelContextWindow": 1000}}))
        self.assertEqual(events, [UsageEvent(input_tokens=100, output_tokens=7, cached_tokens=40, cost_usd=None,
                                             cost_known=False, context_window=1000)])

    def test_resume_replay_is_a_baseline_not_usage(self) -> None:
        # live probe run4: resume, the thread total replayed before turn/start, then the turn's own update
        resume = {"id": 2, "result": {"thread": {"id": THREAD}, "model": "gpt-6.1-sol", "cwd": CWD}}
        frames = [
            resume,
            note("thread/tokenUsage/updated", {"threadId": THREAD, "tokenUsage": {
                "total": {"inputTokens": 76849, "outputTokens": 1000, "cachedInputTokens": 60000},
                "last": {"inputTokens": 19000, "outputTokens": 40, "cachedInputTokens": 0},
                "modelContextWindow": 258400}}),
            note("turn/started", {"threadId": THREAD, "turn": {"id": "t5"}}),
            note("thread/tokenUsage/updated", {"threadId": THREAD, "tokenUsage": {
                "total": {"inputTokens": 96365, "outputTokens": 1040, "cachedInputTokens": 76000},
                "last": {"inputTokens": 19516, "outputTokens": 40, "cachedInputTokens": 16000},
                "modelContextWindow": 258400}}),
        ]
        events = replay(frames, {2: "thread/resume"})
        self.assertEqual(events, [
            SessionStarted(session_id=THREAD, model="gpt-6.1-sol", mode="default", cwd=CWD, resumed=True),
            UsageEvent(input_tokens=19516, output_tokens=40, cached_tokens=16000, cost_usd=None, cost_known=False,
                       context_window=258400),
        ])

    def test_each_turn_reports_its_own_delta_after_a_resume(self) -> None:
        def usage(input_total: int) -> dict:
            return note("thread/tokenUsage/updated", {"threadId": THREAD, "tokenUsage": {
                "total": {"inputTokens": input_total, "outputTokens": 0, "cachedInputTokens": 0},
                "modelContextWindow": 258400}})

        def started(turn_id: str) -> dict:
            return note("turn/started", {"threadId": THREAD, "turn": {"id": turn_id}})

        translator = CodexTranslator()
        translator.translate(usage(76849))  # replay of the resumed thread
        self.assertEqual(translator.translate(started("t5")), [])
        first = translator.translate(usage(96365))
        self.assertEqual([e.input_tokens for e in first], [19516])
        self.assertEqual(translator.translate(started("t6")), [])
        second = translator.translate(usage(115000))
        self.assertEqual([e.input_tokens for e in second], [18635])

    def test_usage_without_a_total_block_yields_nothing(self) -> None:
        self.assertEqual(CodexTranslator().translate(note("thread/tokenUsage/updated", {"tokenUsage": {}})), [])

    def test_will_retry_errors_count_attempts_and_stream_cuts_are_named(self) -> None:
        translator = CodexTranslator()
        overloaded = {"message": "server busy", "codexErrorInfo": "serverOverloaded"}
        first = translator.translate(note("error", {"error": overloaded, "willRetry": True}))
        self.assertEqual(first, [StatusEvent(kind="retry", detail="server busy", attempt=1)])
        cut = {"message": "connection lost",
               "codexErrorInfo": {"responseStreamConnectionFailed": {"httpStatusCode": None}}}
        second = translator.translate(note("error", {"error": cut, "willRetry": True}))
        self.assertEqual(second, [StatusEvent(kind="stream_cut", detail="connection lost", attempt=2)])

    def test_fatal_error_yields_nothing_because_turn_completed_reports_it(self) -> None:
        self.assertEqual(CodexTranslator().translate(note("error", {"error": {"message": "x"},
                                                                    "willRetry": False})), [])

    def test_compaction_from_notification_and_from_item(self) -> None:
        translator = CodexTranslator()
        self.assertEqual(translator.translate(note("thread/compacted", {"threadId": THREAD})),
                         [StatusEvent(kind="compacting")])
        self.assertEqual(translator.translate(item_started({"type": "contextCompaction", "id": "k1"})),
                         [StatusEvent(kind="compacting")])


class TurnResultTest(unittest.TestCase):
    def test_failed_turn_carries_the_error_message(self) -> None:
        events = CodexTranslator().translate(turn_completed("failed", error={"message": "fake failure"}))
        self.assertEqual(events, [TurnResult(status=TurnStatus.ERROR, answer=None, duration_ms=10,
                                             subtype="failed", error="fake failure")])

    def test_interrupted_turn_has_no_error(self) -> None:
        events = CodexTranslator().translate(turn_completed("interrupted"))
        self.assertIsNone(events[0].error)
        self.assertEqual(events[0].status, TurnStatus.INTERRUPTED)

    def test_unknown_status_is_an_error_with_the_status_named(self) -> None:
        events = CodexTranslator().translate(turn_completed("weird"))
        self.assertEqual(events[0].status, TurnStatus.ERROR)
        self.assertEqual(events[0].error, "unexpected turn status: weird")


class ResponseTest(unittest.TestCase):
    def test_thread_start_result_is_a_fresh_session(self) -> None:
        frame = {"id": 2, "result": {"thread": {"id": "th-1"}, "model": "m1", "cwd": "/w"}}
        events = CodexTranslator(mode="bypassPermissions").translate(frame, "thread/start")
        self.assertEqual(events, [SessionStarted(session_id="th-1", model="m1", mode="bypassPermissions",
                                                 cwd="/w", resumed=False)])

    def test_thread_resume_error_is_resume_failed(self) -> None:
        frame = {"id": 2, "error": {"code": -32600, "message": "no rollout found"}}
        self.assertEqual(CodexTranslator().translate(frame, "thread/resume"),
                         [TurnResult(status=TurnStatus.ERROR, answer=None, error="no rollout found",
                                     resume_failed=True)])

    def test_start_and_turn_start_errors_are_plain_turn_errors(self) -> None:
        translator = CodexTranslator()
        start = translator.translate({"id": 2, "error": {"message": "bad cwd"}}, "thread/start")
        self.assertEqual(start, [TurnResult(status=TurnStatus.ERROR, answer=None, error="bad cwd")])
        turn = translator.translate({"id": 3, "error": {"message": "busy"}}, "turn/start")
        self.assertFalse(turn[0].resume_failed)

    def test_other_responses_and_responses_without_a_method_yield_nothing(self) -> None:
        translator = CodexTranslator()
        self.assertEqual(translator.translate({"id": 1, "error": {"message": "no"}}, "initialize"), [])
        self.assertEqual(translator.translate({"id": 3, "result": {"turn": {"id": "t"}}}, "turn/start"), [])
        self.assertEqual(translator.translate({"id": 2, "result": {"thread": {"id": "x"}}}), [])


class FakeCodexTest(unittest.TestCase):
    """tests/fakes/fake_codex.py run through its launcher: the frames it really writes, translated."""

    def run_fake(self, mode: str, frames: list[dict], home: str) -> tuple[list[dict], dict[int, str]]:
        payload = "".join(json.dumps(frame) + "\n" for frame in frames)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": home, "FAKE_CODEX_MODE": mode}
        done = subprocess.run([LAUNCHER, "app-server"], input=payload, capture_output=True, text=True,
                              env=env, timeout=30, check=False)
        out = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
        methods = {frame["id"]: frame["method"] for frame in frames if "id" in frame and "method" in frame}
        return out, methods

    def setUp(self) -> None:
        self.home = TempHome()
        self.home.__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def requests(self, first: dict) -> list[dict]:
        return [
            {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "t", "version": "0"},
                                                          "capabilities": {}}},
            {"method": "initialized"},
            first,
            {"id": 3, "method": "turn/start", "params": {"threadId": "fake-thread-1",
                                                          "input": [{"type": "text", "text": "hi",
                                                                     "text_elements": []}]}},
        ]

    def test_retry_mode_yields_exactly_one_stream_cut(self) -> None:
        start = {"id": 2, "method": "thread/start", "params": {"cwd": self.home.home, "approvalPolicy": "untrusted",
                                                               "sandbox": "workspace-write"}}
        out, methods = self.run_fake("retry", self.requests(start), self.home.home)
        events = replay(out, methods, mode="default")
        cuts = [e for e in events if isinstance(e, StatusEvent)]
        self.assertEqual(cuts, [StatusEvent(kind="stream_cut", detail="stream disconnected before completion",
                                            attempt=1)])
        self.assertIsInstance(events[0], SessionStarted)
        self.assertEqual(events[-1], TurnResult(status=TurnStatus.OK, answer="done", duration_ms=None,
                                                subtype="completed"))

    def test_resume_missing_mode_yields_one_resume_failed(self) -> None:
        resume = {"id": 2, "method": "thread/resume", "params": {"threadId": "fake-thread-1"}}
        out, methods = self.run_fake("resume_missing", self.requests(resume)[:3], self.home.home)
        events = replay(out, methods, mode="default")
        self.assertEqual(events, [TurnResult(status=TurnStatus.ERROR, answer=None,
                                             error="no rollout found for thread id fake-thread-1",
                                             resume_failed=True)])


if __name__ == "__main__":
    unittest.main()
