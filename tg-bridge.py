#!/usr/bin/env python3
"""Thin entry point: puts this directory on sys.path and runs tg_bridge.cli.main()."""

import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tg_bridge.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
