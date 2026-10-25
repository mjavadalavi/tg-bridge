"""Scaffold checks: entry script, symlink entry, package import tree and the TempHome helper."""

import importlib
import os
import pkgutil
import subprocess
import sys
import tempfile
import unittest

import tg_bridge
from tests.support import TempHome

V3_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ENTRY = os.path.join(V3_DIR, "tg-bridge.py")


def _run_env(home: str) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", ""), "HOME": home, "LANG": "C.UTF-8"}


class EntryScriptTests(unittest.TestCase):
    def test_entry_is_executable(self) -> None:
        self.assertTrue(os.access(ENTRY, os.X_OK))

    def test_direct_run_prints_spec(self) -> None:
        with TempHome() as home:
            result = subprocess.run(
                [ENTRY, "help"], env=_run_env(home.home), capture_output=True, text=True, timeout=60
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, tg_bridge.__doc__)

    def test_symlink_run_prints_spec(self) -> None:
        with TempHome() as home, tempfile.TemporaryDirectory() as link_dir:
            link = os.path.join(link_dir, "tg-bridge")
            os.symlink(ENTRY, link)
            result = subprocess.run(
                [sys.executable, "-I", link, "help"],
                env=_run_env(home.home),
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, tg_bridge.__doc__)

    def test_unknown_subcommand_prints_spec(self) -> None:
        with TempHome() as home:
            result = subprocess.run(
                [sys.executable, "-I", ENTRY, "no-such-command"],
                env=_run_env(home.home),
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, tg_bridge.__doc__)


class PackageTests(unittest.TestCase):
    def test_spec_docstring_is_the_live_spec(self) -> None:
        self.assertTrue(tg_bridge.__doc__.startswith("tg-bridge — talk to your Claude sessions"))
        self.assertIn("Subcommands:", tg_bridge.__doc__)
        self.assertEqual(tg_bridge.__version__, "3.0.0")

    def test_every_module_imports_and_has_a_docstring(self) -> None:
        names = []
        for info in pkgutil.walk_packages(tg_bridge.__path__, prefix="tg_bridge."):
            names.append(info.name)
        self.assertIn("tg_bridge.telegram.render", names)
        self.assertIn("tg_bridge.agents.codex.translate", names)
        self.assertIn("tg_bridge.daemon.panes", names)
        for name in names:
            module = importlib.import_module(name)
            self.assertTrue(module.__doc__, f"{name} has no module docstring")

    def test_python_m_entry_prints_spec(self) -> None:
        with TempHome() as home:
            result = subprocess.run(
                [sys.executable, "-m", "tg_bridge", "help"],
                cwd=V3_DIR,
                env={**_run_env(home.home), "PYTHONPATH": V3_DIR},
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, tg_bridge.__doc__)

    def test_no_module_level_mutable_state_in_scaffold(self) -> None:
        for name in ("tg_bridge.cli", "tg_bridge.paths", "tg_bridge.app"):
            module = importlib.import_module(name)
            for attr, value in vars(module).items():
                if attr.startswith("__"):
                    continue
                self.assertNotIsInstance(value, (list, dict, set), f"{name}.{attr} is mutable module state")


class TempHomeTests(unittest.TestCase):
    def test_home_is_switched_and_restored(self) -> None:
        real_home = os.environ.get("HOME")
        with TempHome() as home:
            self.assertEqual(os.environ["HOME"], home.home)
            self.assertEqual(os.path.expanduser("~"), home.home)
            self.assertTrue(os.path.isdir(home.state))
            self.assertEqual(home.state, os.path.join(home.home, ".local", "state", "tg-bridge"))
            self.assertNotEqual(home.home, real_home)
            kept = home.home
        self.assertEqual(os.environ.get("HOME"), real_home)
        self.assertFalse(os.path.exists(kept))

    def test_nested_homes_do_not_leak(self) -> None:
        real_home = os.environ.get("HOME")
        with TempHome() as outer:
            with TempHome() as inner:
                self.assertEqual(os.environ["HOME"], inner.home)
            self.assertEqual(os.environ["HOME"], outer.home)
        self.assertEqual(os.environ.get("HOME"), real_home)


if __name__ == "__main__":
    unittest.main()
