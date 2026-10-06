#!/usr/bin/env bash
# Linux/macOS launcher: finds Python 3.11+ and runs start.py.
cd "$(dirname "$0")" || exit 1
for py in .venv/bin/python python3.12 python3.11 python3.13 python3; do
  if command -v "$py" >/dev/null 2>&1 &&
     "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    exec "$py" start.py "$@"
  fi
done
echo "Reveal Cam needs Python 3.11 or newer."
echo "Install it from https://www.python.org/downloads/ (macOS: the official installer),"
echo "then double-click Reveal Cam again."
read -r -p "Press Enter to close this window." _
exit 1
