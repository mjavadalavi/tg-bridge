"""Shared harness of the launcher and cwd port tests (fakecli/test_launcher.py and test_launcher_cwd.py).

It is the callback harness of tests/callback_support.py with the real LauncherSwitch, the real settings and session
commands, and a CwdService whose folder finder records each search. The launcher names are the ones of the old suite
(wrap = 'wrap run', plain, claude), so the replies can be compared with the old texts. Nothing starts a process:
sessions are StubSession objects, Telegram is FakeTelegram, and every write of a command or tap is awaited with join().
"""

import json
import os
import threading

from tests.callback_support import CHAT, THREAD, TOPIC, CallbackCase
from tg_bridge.bot.callbacks.handlers import QueueTaps, StatusTaps, StopTap
from tg_bridge.bot.callbacks.approvals import ApprovalTaps
from tg_bridge.bot.callbacks.picker_taps import PickerTaps
from tg_bridge.bot.commands.settings import SettingsCommands
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.launcher_switch import LauncherSwitch
from tg_bridge.core.workers import WorkerPool
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.store.inbox import CwdRepo

JOIN_SECONDS = 5
OLD_LAUNCHERS = ("wrap", "plain", "claude")


class RecordingFinder:
    """A folder finder that records each search and returns `result`. When `gate` is set, it waits on it first."""

    def __init__(self, result=(), gate: threading.Event | None = None) -> None:
        self.result = list(result)
        self.gate = gate
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, term: str, roots, deadline: float) -> list[str]:
        with self._lock:
            self.calls.append(term)
        if self.gate is not None:
            self.gate.wait(JOIN_SECONDS)
        return list(self.result)


class LauncherCwdPortCase(CallbackCase):
    """The callback harness with the old suite's launchers, the real LauncherSwitch and the settings commands registered.

    The topic has no configured launcher, so the stored launcher of its row decides, as in the old suite after its D part.

    Folders: wrap and plain keep their transcripts in two config folders of the test home, claude in ~/.claude
    (under the same home), so a change between them copies or looks for a transcript as the live file does.
    """

    def setUp(self) -> None:
        self.finder = RecordingFinder()
        super().setUp()

    def _config_data(self) -> dict:
        data = super()._config_data()
        data["launchers"] = {
            "wrap": {"cmd": "wrap run", "config_dir": os.path.join(self.project, "cfg-a")},
            "plain": {"cmd": "plain", "config_dir": os.path.join(self.project, "cfg-b")},
            "claude": {"cmd": "claude", "config_dir": os.path.join(self.base, ".claude")},
        }
        data["topics"] = {TOPIC: {"cwd": self.project}}  # no launcher of its own: the topic row decides
        return data

    def register_all(self) -> None:
        """The real LauncherSwitch, a CwdService with the recording finder and every tap handler, as the App wires them."""
        self.search = WorkerPool("search", 1, self.logger)
        handoff = HandoffService(self.kv, self.backends, self.logger)
        self.launchers = LauncherSwitch(self.config, self.sessions, self.history, handoff, self.backends)
        self.folders = CwdService(
            sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger, live=self.live,
            search_pool=self.search, out_pool=self.out, logger=self.logger, clock=self.clock, handoff=handoff,
            finder=self.finder)
        settings = SettingsCommands(
            config=self.config, sessions=self.sessions, live=self.live, backends=self.backends,
            permissions=self.permissions, launchers=self.launchers, cwd=self.folders, logger=self.logger)
        settings.register(self.commands)
        QueueTaps(dispatcher=self.dispatcher, live=self.live).register(self.router)
        StopTap(stops=self.stops).register(self.router)
        StatusTaps(deps=self.deps, registry=self.commands, stops=self.stops).register(self.router)
        PickerTaps(config=self.config, sessions=self.sessions, live=self.live, backends=self.backends,
                   permissions=self.permissions, launchers=self.launchers, stops=self.stops, rules=self.rules,
                   folders=self.folders, logger=self.logger).register(self.router)
        ApprovalTaps(permissions=self.permissions).register(self.router)

    # --- driving commands and reading replies

    def run_cmd(self, text: str, thread: int = THREAD) -> bool:
        """Dispatches one typed command and waits for the search worker and the ordered out-worker."""
        handled = self.commands.dispatch(self.deps, CHAT, thread, TOPIC, text)
        self.assertTrue(self.search.join(JOIN_SECONDS), "the folder search did not finish")
        self.assertTrue(self.out.join(JOIN_SECONDS), "the replies did not go out")
        return handled

    def last_text(self, thread: int = THREAD) -> str:
        return self.telegram.texts(thread)[-1]

    def card_buttons(self) -> list[dict]:
        """The buttons of the newest message in the topic that has a keyboard, row by row flattened."""
        markup = self.sent_buttons()[-1]["reply_markup"]
        return [b for row in json.loads(markup)["inline_keyboard"] for b in row]

    def card_labels(self) -> list[str]:
        return [b["text"] for b in self.card_buttons()]

    def card_callbacks(self) -> list[str]:
        return [b["callback_data"] for b in self.card_buttons()]

    def row(self, thread: int = THREAD):
        return self.sessions.get(thread)

    def set_row(self, thread: int = THREAD, **columns) -> None:
        """Sets columns of the topic's session row directly (the old suite's make_row, for the stored values)."""
        for column, value in columns.items():
            self.db.execute(f"UPDATE sessions SET {column}=? WHERE thread_id=?", (value, thread))

    def put_transcript(self, config_dir: str, sid: str, text: str = '{"type": "user"}\n') -> None:
        """A Claude transcript for sid in the project folder of config_dir (where its launcher keeps it)."""
        folder = os.path.join(config_dir, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, sid + ".jsonl"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def note(self, sid: str, launcher: str, cwd: str | None = None, text: str = "earlier question",
             backend: str = "claude", thread: int = THREAD) -> None:
        self.history.note_session(thread, sid, text, cwd or self.project, launcher, backend)

    def history_id(self, sid: str, thread: int = THREAD) -> int:
        return self.history.find(thread, sid).id
