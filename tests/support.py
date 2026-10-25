"""Shared test helpers. TempHome isolates HOME so no test touches the real state directory."""

import os
import shutil
import tempfile


class TempHome:
    """Context manager: HOME points at a fresh temp directory with the STATE directory built.

    On exit the previous HOME is restored and the temp directory is removed.
    """

    def __init__(self) -> None:
        self.home = ""
        self.state = ""
        self._saved_home: str | None = None
        self._dir: str | None = None

    def __enter__(self) -> "TempHome":
        self._saved_home = os.environ.get("HOME")
        self._dir = tempfile.mkdtemp(prefix="tgb-v3-test-")
        self.home = self._dir
        os.environ["HOME"] = self.home
        self.state = os.path.join(self.home, ".local", "state", "tg-bridge")
        os.makedirs(self.state, mode=0o700, exist_ok=True)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if self._saved_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._saved_home
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        return False
