"""Module entry: python -m tg_bridge runs the command line."""

import sys

from tg_bridge.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
