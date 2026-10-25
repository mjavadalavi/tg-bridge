#!/bin/sh
# Launcher wrapper for tests: runs the fake Claude CLI next to this folder, the way a topic launcher runs claude.
exec python3 "$(dirname "$0")/../fake_claude.py" "$@"
