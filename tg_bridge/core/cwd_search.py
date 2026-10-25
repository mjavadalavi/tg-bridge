"""Pure folder-search helpers of the /cwd name search (port of search_roots, find_dirs and _rank_found, C19).

The search walks configured roots and returns the folders whose NAME contains the term (or every word of it). Nothing
here keeps state or talks to Telegram: find_dirs is a plain function, so CwdService can take it as an injected
finder and tests can replace it. The deadline argument is a time.monotonic() value; a caller that uses another clock
passes its own deadline on that clock's monotonic timeline.
"""

import os
import time
from collections.abc import Iterable

CWD_SEARCH_SECONDS = 5.0  # hard time limit of one /cwd name search (and of one folder suggestion)
CWD_SEARCH_DEPTH = 5  # directory levels below each search root
CWD_SEARCH_SKIP = frozenset({"node_modules", ".venv", "venv", "__pycache__", ".git", "site-packages", "dist", "build"})
CWD_SEARCH_MAX = 8  # buttons on one search card


def normalize_term(term: str) -> str:
    """The search term in the form find_dirs compares: lower case, single spaces."""
    return " ".join(term.lower().split())


def resolve_search_roots(configured: Iterable[str]) -> list[str]:
    """The real, existing search roots: repeated roots are kept once, and a root inside another root is dropped,
    so no folder is walked twice."""
    roots: list[str] = []
    for p in configured:
        real = os.path.realpath(os.path.expanduser(str(p)))
        if os.path.isdir(real) and real not in roots:
            roots.append(real)
    return [r for r in roots if not any(r != o and r.startswith(o + os.sep) for o in roots)]


def find_dirs(term: str, roots: Iterable[str], deadline: float) -> list[str]:
    """Directories whose NAME contains the term, or contains every word of it (case-insensitive), under roots, at most
    CWD_SEARCH_DEPTH levels deep. Hidden, build and virtualenv folders are skipped. Stops at the deadline (a
    time.monotonic() value) with what it has. Ranked: exact name first, then the shorter path; at most CWD_SEARCH_MAX."""
    want = normalize_term(term)
    words = want.split()
    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if time.monotonic() > deadline:
                return rank_found(found, want)
            depth = dirpath.count(os.sep) - base  # the root itself is 0
            if depth >= CWD_SEARCH_DEPTH:
                dirnames[:] = []
            else:
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in CWD_SEARCH_SKIP]
            name = os.path.basename(dirpath).lower()
            if dirpath not in seen and (name == want or all(w in name for w in words)):
                seen.add(dirpath)
                found.append(dirpath)
    return rank_found(found, want)


def rank_found(found: list[str], want: str) -> list[str]:
    """Exact name matches first, then the shorter path; at most CWD_SEARCH_MAX. `want` is a normalized term."""
    ranked = sorted(found, key=lambda p: (os.path.basename(p).lower() != want, len(p)))
    return ranked[:CWD_SEARCH_MAX]
