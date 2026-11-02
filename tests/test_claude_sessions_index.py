"""Claude on-disk session listing (agents/claude/sessions_index.py, the list_sessions default and ClaudeBackend override).

Scenarios: the ai-title wins; a summary is used when there is no ai-title; the first real prompt skips meta lines,
command wrappers and tool-result-only content; a file with neither an ai-title nor a real prompt is skipped; sub-folders
and names that are not session ids are ignored; newest first, with the limit counted in usable sessions only; a big file
whose prompt lies beyond the scan window is skipped (the bound holds); a symlinked cwd finds both folder forms, once per
id; the base default and the Claude override behave as the card says. Card F1-index: a transcript recorded for another
cwd that encodes to the same folder is skipped (a); the newer of a symlink pair is listed (b); the candidates come from a
bounded heap and a failing stat skips the entry (c); excluded ids are left out before the limit and the file cap (d).
"""

import json
import os
import unittest
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.base import AgentBackend, Capabilities, DiskSession
from tg_bridge.agents.claude import sessions_index
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger

SID_A = "11111111-aaaa-4bbb-8ccc-000000000001"
SID_B = "22222222-aaaa-4bbb-8ccc-000000000002"
SID_C = "33333333-aaaa-4bbb-8ccc-000000000003"
SID_D = "44444444-aaaa-4bbb-8ccc-000000000004"
SID_E = "55555555-aaaa-4bbb-8ccc-000000000005"
SUB_SID = "66666666-aaaa-4bbb-8ccc-000000000006"


def prompt(text: str) -> dict:
    """A plain user prompt line (message content as a string)."""
    return {"type": "user", "message": {"role": "user", "content": text}}


def block_prompt(text: str) -> dict:
    """A user line whose content is a list with one text block."""
    return {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def tool_result_line() -> dict:
    """A user line that carries only a tool result, as Claude writes after a tool call."""
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def assistant(text: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def ai_title(text: str) -> dict:
    return {"type": "ai-title", "aiTitle": text}


def summary(text: str) -> dict:
    return {"type": "summary", "summary": text}


class SessionsIndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.real_home = os.path.realpath(self.home.home)
        self.cwd = os.path.join(self.real_home, "work", "my proj")
        os.makedirs(self.cwd)
        self.config = os.path.join(self.real_home, "cfg")
        self.folder = os.path.join(self.config, "projects", encoded_cwd(os.path.realpath(self.cwd)))
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))

    def put(self, sid: str, lines: list[dict] | None = None, folder: str | None = None, mtime: float | None = None,
            raw: str | None = None) -> str:
        """Write one transcript (lines dumped as JSON, or raw text) into folder and optionally set its mtime."""
        target = folder or self.folder
        os.makedirs(target, exist_ok=True)
        path = os.path.join(target, sid + ".jsonl")
        text = raw if raw is not None else "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines or [])
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def listed(self, limit: int = 50, cwd: str | None = None) -> list[DiskSession]:
        return list(sessions_index.list_sessions(self.config, cwd or self.cwd, limit))

    def title_of(self, sid: str) -> str:
        found = [s for s in self.listed() if s.uuid == sid]
        self.assertEqual(len(found), 1, f"{sid} should be listed once")
        return found[0].title


class TitleTest(SessionsIndexTestCase):
    def test_ai_title_wins_over_summary_and_prompt(self) -> None:
        self.put(SID_A, [prompt("first question here"), summary("a summary"), assistant("ok"),
                         ai_title("Fix the login bug")])
        self.assertEqual(self.title_of(SID_A), "Fix the login bug")

    def test_summary_used_when_no_ai_title(self) -> None:
        self.put(SID_A, [prompt("hello there"), summary("Parser refactor"), assistant("ok")])
        self.assertEqual(self.title_of(SID_A), "Parser refactor")

    def test_first_real_prompt_skips_meta_wrappers_and_tool_results(self) -> None:
        self.put(SID_A, [
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta caveat text"}},
            prompt("<command-name>/clear</command-name>"),
            block_prompt("<command-message>wrapped</command-message>"),
            tool_result_line(),
            block_prompt("  Add   retry\n to the   uploader  "),
            prompt("a later prompt"),
        ])
        self.assertEqual(self.title_of(SID_A), "Add retry to the uploader")

    def test_title_cut_to_sixty_characters(self) -> None:
        long = "x" * 80
        self.put(SID_A, [prompt(long)])
        self.assertEqual(self.title_of(SID_A), "x" * sessions_index.TITLE_CHARS)

    def test_file_without_real_prompt_or_ai_title_is_skipped(self) -> None:
        self.put(SID_A, [
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta only"}},
            prompt("<command-name>/status</command-name>"),
            tool_result_line(),
            assistant("nothing to title"),
        ])
        self.put(SID_B, raw="")
        self.put(SID_C, [prompt("the real one")])
        self.assertEqual([s.uuid for s in self.listed()], [SID_C])

    def test_ai_title_alone_is_enough_without_prompt(self) -> None:
        self.put(SID_A, [ai_title("Generated only"), assistant("ok")])
        self.assertEqual(self.title_of(SID_A), "Generated only")

    def test_prompt_beyond_line_window_is_skipped(self) -> None:
        self.put(SID_A, [assistant("filler")] * sessions_index.TITLE_SCAN_LINES + [prompt("too late")])
        self.assertEqual(self.listed(), [])

    def test_big_file_prompt_beyond_byte_window_is_skipped(self) -> None:
        # 300 lines: the prompt is past the line window as well, so only the size of the file is at issue here.
        path = self.put(SID_A, [assistant("y" * 1000)] * 300 + [prompt("hidden prompt")])
        self.assertGreater(os.path.getsize(path), sessions_index.TITLE_SCAN_BYTES)
        # 150 lines of 2 KB: the prompt is inside the line window but past the byte window, so the byte bound alone skips it.
        few_long = self.put(SID_C, [assistant("z" * 2000)] * 150 + [prompt("behind the bytes")])
        self.assertLess(150, sessions_index.TITLE_SCAN_LINES)
        self.assertGreater(os.path.getsize(few_long), sessions_index.TITLE_SCAN_BYTES)
        self.put(SID_B, [prompt("early prompt")] + [assistant("y" * 1000)] * 300)
        self.assertEqual([s.uuid for s in self.listed()], [SID_B])
        self.assertEqual(self.title_of(SID_B), "early prompt")


class ListingTest(SessionsIndexTestCase):
    def test_newest_first_and_limit(self) -> None:
        self.put(SID_A, [prompt("oldest")], mtime=1000.0)
        self.put(SID_B, [prompt("newest")], mtime=3000.0)
        self.put(SID_C, [prompt("middle")], mtime=2000.0)
        self.assertEqual([s.uuid for s in self.listed()], [SID_B, SID_C, SID_A])
        self.assertEqual([s.uuid for s in self.listed(limit=2)], [SID_B, SID_C])
        self.assertEqual(self.listed(limit=0), [])

    def test_modified_is_the_file_mtime(self) -> None:
        self.put(SID_A, [prompt("stamped")], mtime=1234.5)
        self.assertEqual(self.listed()[0], DiskSession(uuid=SID_A, title="stamped", modified=1234.5))

    def test_limit_counts_only_usable_sessions(self) -> None:
        self.put(SID_A, [assistant("no prompt")], mtime=3000.0)
        self.put(SID_B, [prompt("usable older")], mtime=2000.0)
        self.put(SID_C, [prompt("usable oldest")], mtime=1000.0)
        self.assertEqual([s.uuid for s in self.listed(limit=1)], [SID_B])

    def test_sub_folders_and_bad_names_are_ignored(self) -> None:
        self.put(SID_A, [prompt("top level")])
        self.put(SUB_SID, [prompt("in a sub-folder")], folder=os.path.join(self.folder, SUB_SID))
        self.put(SID_B, [prompt("subagent")], folder=os.path.join(self.folder, SUB_SID, "subagents"))
        self.put("bad id", [prompt("space in the name")])
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, "notes.txt"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(prompt("not a jsonl")) + "\n")
        self.assertEqual([s.uuid for s in self.listed()], [SID_A])

    def test_missing_folder_gives_empty_tuple(self) -> None:
        self.assertEqual(sessions_index.list_sessions(self.config, self.cwd, 5), ())

    def test_empty_config_dir_or_cwd_gives_empty_tuple(self) -> None:
        self.put(SID_A, [prompt("present")])
        self.assertEqual(sessions_index.list_sessions("", self.cwd, 5), ())
        self.assertEqual(sessions_index.list_sessions(self.config, "", 5), ())


class FileCapTest(SessionsIndexTestCase):
    def test_unusable_files_are_opened_at_most_limit_times_the_factor(self) -> None:
        for n in range(30):  # newer than the usable one, and none has a prompt
            self.put(f"{n:08d}-aaaa-4bbb-8ccc-000000000000", [assistant("no prompt here")], mtime=3000.0 + n)
        self.put(SID_A, [prompt("usable but too old")], mtime=1000.0)
        with mock.patch.object(sessions_index, "_title", wraps=sessions_index._title) as title:
            self.assertEqual(self.listed(limit=2), [])
        self.assertEqual(title.call_count, 2 * sessions_index.MAX_FILES_PER_SESSION)

    def test_usable_session_inside_the_file_cap_is_still_found(self) -> None:
        for n in range(3):
            self.put(f"{n:08d}-aaaa-4bbb-8ccc-000000000000", [assistant("no prompt here")], mtime=3000.0 + n)
        self.put(SID_A, [prompt("found in time")], mtime=1000.0)
        with mock.patch.object(sessions_index, "_title", wraps=sessions_index._title) as title:
            self.assertEqual([s.uuid for s in self.listed(limit=1)], [SID_A])
        self.assertEqual(title.call_count, 4)


class SymlinkCwdTest(SessionsIndexTestCase):
    def test_symlinked_cwd_finds_both_folder_forms_once_per_id(self) -> None:
        link = os.path.join(self.real_home, "link-to-work")
        os.symlink(self.cwd, link)
        link_folder = os.path.join(self.config, "projects", encoded_cwd(link))
        self.put(SID_A, [prompt("under the real folder")], mtime=2000.0)
        self.put(SID_A, [prompt("under the real folder")], folder=link_folder, mtime=2000.0)
        self.put(SID_B, [prompt("under the link folder")], folder=link_folder, mtime=1000.0)
        listed = self.listed(cwd=link)
        self.assertEqual([s.uuid for s in listed], [SID_A, SID_B])
        self.assertEqual(listed[1].title, "under the link folder")


class BackendTest(SessionsIndexTestCase):
    def test_base_default_returns_empty_tuple(self) -> None:
        class Bare(AgentBackend):
            name = "bare"
            capabilities = Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False,
                                        images=False, preassigned_id=False)

            def modes(self) -> tuple:
                return ()

            def models(self, cfg: object) -> tuple:
                return ()

            def open_session(self, spec: object, sink: object) -> object:
                raise NotImplementedError

        spec = LauncherSpec(name="l", backend="bare", command="c", argv=("c",), config_dir=self.config)
        self.assertEqual(Bare().list_sessions(spec, self.cwd, 5), ())

    def test_claude_backend_delegates_with_launcher_config_dir(self) -> None:
        self.put(SID_A, [prompt("delegated")], mtime=5000.0)
        backend = ClaudeBackend(logger=self.logger, clock=FakeClock())
        spec = LauncherSpec(name="l", backend="claude", command="c", argv=("c",), config_dir=self.config)
        self.assertEqual(backend.list_sessions(spec, self.cwd, 5),
                         (DiskSession(uuid=SID_A, title="delegated", modified=5000.0),))


class FlakyEntry:
    """A real directory entry whose stat() fails for one name, as a file removed in the middle of a scan does."""

    def __init__(self, entry: os.DirEntry, fail_name: str) -> None:
        self.name = entry.name
        self.path = entry.path
        self._entry = entry
        self._fail_name = fail_name

    def is_file(self) -> bool:
        return self._entry.is_file()

    def stat(self) -> os.stat_result:
        if self.name == self._fail_name:
            raise FileNotFoundError(self.path)
        return self._entry.stat()


class FlakyListing(list):
    """A list that can be used as the context manager os.scandir returns."""

    def __enter__(self) -> "FlakyListing":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class CwdCollisionTest(SessionsIndexTestCase):
    """(a) /work/my-proj and /work/my proj encode to the same folder; a transcript records which cwd it belongs to."""

    def setUp(self) -> None:
        super().setUp()
        self.other = os.path.join(self.real_home, "work", "my-proj")
        os.makedirs(self.other)
        self.assertEqual(encoded_cwd(self.other), encoded_cwd(self.cwd))

    def test_transcript_recorded_for_another_cwd_is_skipped(self) -> None:
        self.put(SID_A, [{**prompt("from the other folder"), "cwd": self.other}])
        self.put(SID_B, [{**prompt("from this folder"), "cwd": self.cwd}])
        self.assertEqual([s.uuid for s in self.listed()], [SID_B])

    def test_listing_the_other_cwd_keeps_its_own_transcript(self) -> None:
        self.put(SID_A, [{**prompt("from the other folder"), "cwd": self.other}])
        self.assertEqual([s.uuid for s in self.listed(cwd=self.other)], [SID_A])
        self.assertEqual(self.listed(), [])

    def test_recorded_cwd_that_is_a_symlink_alias_is_kept(self) -> None:
        alias = os.path.join(self.real_home, "alias")
        os.symlink(self.cwd, alias)
        self.put(SID_A, [{**prompt("via the alias"), "cwd": alias}])
        self.assertEqual([s.uuid for s in self.listed()], [SID_A])

    def test_no_recorded_cwd_in_the_window_is_kept(self) -> None:
        self.put(SID_A, [prompt("written before the cwd field existed")])
        self.assertEqual([s.uuid for s in self.listed()], [SID_A])

    def test_cwd_check_skips_only_the_mismatched_file(self) -> None:
        self.put(SID_A, [{**prompt("newer, other cwd"), "cwd": self.other}], mtime=3000.0)
        self.put(SID_B, [prompt("older, no cwd")], mtime=1000.0)
        with mock.patch.object(sessions_index, "_title", wraps=sessions_index._title) as title:
            self.assertEqual([s.uuid for s in self.listed(limit=1)], [SID_B])
        self.assertEqual(title.call_count, 2)  # the mismatched file was opened (it counts toward the cap) and skipped


class NewestCopyTest(SessionsIndexTestCase):
    """(b) a session id held by both the realpath and the abspath (symlink) folder is listed from the newer copy."""

    def setUp(self) -> None:
        super().setUp()
        self.link = os.path.join(self.real_home, "link-to-work")
        os.symlink(self.cwd, self.link)
        self.link_folder = os.path.join(self.config, "projects", encoded_cwd(self.link))

    def test_newer_copy_in_the_symlink_folder_is_listed(self) -> None:
        self.put(SID_A, [prompt("old real copy")], mtime=1000.0)
        self.put(SID_A, [prompt("new link copy")], folder=self.link_folder, mtime=2000.0)
        self.assertEqual(self.listed(cwd=self.link),
                         [DiskSession(uuid=SID_A, title="new link copy", modified=2000.0)])

    def test_newer_copy_in_the_realpath_folder_is_listed(self) -> None:
        self.put(SID_A, [prompt("new real copy")], mtime=2000.0)
        self.put(SID_A, [prompt("old link copy")], folder=self.link_folder, mtime=1000.0)
        self.assertEqual(self.listed(cwd=self.link),
                         [DiskSession(uuid=SID_A, title="new real copy", modified=2000.0)])


class BoundedScanTest(SessionsIndexTestCase):
    """(c) the candidates are the newest limit * MAX_FILES_PER_SESSION from a heap, and a failing stat skips the entry."""

    def test_candidates_are_chosen_with_a_bounded_heap(self) -> None:
        self.put(SID_A, [prompt("only one")])
        with mock.patch.object(sessions_index.heapq, "nlargest", wraps=sessions_index.heapq.nlargest) as nlargest:
            self.assertEqual([s.uuid for s in self.listed(limit=3)], [SID_A])
        nlargest.assert_called_once()
        self.assertEqual(nlargest.call_args.args[0], 3 * sessions_index.MAX_FILES_PER_SESSION)

    def test_entry_whose_stat_fails_is_skipped(self) -> None:
        self.put(SID_A, [prompt("stat works")], mtime=1000.0)
        self.put(SID_B, [prompt("stat fails")], mtime=2000.0)
        real_scandir = os.scandir

        def flaky_scandir(path: str) -> FlakyListing:
            with real_scandir(path) as handle:
                return FlakyListing(FlakyEntry(entry, SID_B + sessions_index.TRANSCRIPT_SUFFIX) for entry in handle)

        with mock.patch.object(sessions_index.os, "scandir", side_effect=flaky_scandir):
            self.assertEqual([s.uuid for s in self.listed()], [SID_A])


class ExcludeTest(SessionsIndexTestCase):
    """(d) excluded ids are left out before the limit is applied and do not count toward the file-open cap."""

    def test_excluded_ids_are_left_out_before_the_limit(self) -> None:
        self.put(SID_A, [prompt("newest but excluded")], mtime=3000.0)
        self.put(SID_B, [prompt("older and usable")], mtime=2000.0)
        found = sessions_index.list_sessions(self.config, self.cwd, 1, frozenset({SID_A}))
        self.assertEqual([s.uuid for s in found], [SID_B])

    def test_excluded_files_do_not_use_up_the_file_cap(self) -> None:
        excluded = [f"{n:08d}-aaaa-4bbb-8ccc-000000000000" for n in range(30)]
        for n, sid in enumerate(excluded):  # newer than the usable one, and every one of them is usable
            self.put(sid, [prompt("excluded and usable")], mtime=4000.0 + n)
        self.put(SID_A, [prompt("usable but old")], mtime=1000.0)
        with mock.patch.object(sessions_index, "_title", wraps=sessions_index._title) as title:
            found = sessions_index.list_sessions(self.config, self.cwd, 1, frozenset(excluded))
        self.assertEqual([s.uuid for s in found], [SID_A])
        self.assertEqual(title.call_count, 1)

    def test_default_exclude_leaves_every_session_listed(self) -> None:
        self.put(SID_A, [prompt("first")], mtime=2000.0)
        self.put(SID_B, [prompt("second")], mtime=1000.0)
        self.assertEqual([s.uuid for s in sessions_index.list_sessions(self.config, self.cwd, 5)], [SID_A, SID_B])


class ExcludeBackendTest(SessionsIndexTestCase):
    def test_claude_backend_passes_exclude_through(self) -> None:
        self.put(SID_A, [prompt("excluded")], mtime=5000.0)
        self.put(SID_B, [prompt("kept")], mtime=4000.0)
        backend = ClaudeBackend(logger=self.logger, clock=FakeClock())
        spec = LauncherSpec(name="l", backend="claude", command="c", argv=("c",), config_dir=self.config)
        found = backend.list_sessions(spec, self.cwd, 5, frozenset({SID_A}))
        self.assertEqual([s.uuid for s in found], [SID_B])

    def test_base_default_accepts_exclude(self) -> None:
        class Bare(AgentBackend):
            name = "bare"
            capabilities = Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False,
                                        images=False, preassigned_id=False)

            def modes(self) -> tuple:
                return ()

            def models(self, cfg: object) -> tuple:
                return ()

            def open_session(self, spec: object, sink: object) -> object:
                raise NotImplementedError

        spec = LauncherSpec(name="l", backend="bare", command="c", argv=("c",), config_dir=self.config)
        self.assertEqual(Bare().list_sessions(spec, self.cwd, 5, frozenset({SID_A})), ())


if __name__ == "__main__":
    unittest.main()
