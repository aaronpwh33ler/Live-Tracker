#!/usr/bin/env bash
# Linux/macOS launcher: finds a supported Python (3.11-3.14) and runs start.py.
cd "$(dirname "$0")" || exit 1
ok='import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)'
for py in .venv/bin/python python3.13 python3.12 python3.14 python3.11 python3 \
          /Library/Frameworks/Python.framework/Versions/3.1[1-4]/bin/python3 \
          /opt/homebrew/bin/python3.1[1-4] /usr/local/bin/python3.1[1-4]; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "$ok" 2>/dev/null; then
    exec "$py" start.py "$@"
  fi
done
found=$(python3 --version 2>/dev/null)
echo "Reveal Cam needs Python 3.11, 3.12, 3.13 or 3.14${found:+ (found $found)}."
echo "Newer pre-release versions such as 3.15 aren't supported yet by the AI engine."
echo "Install Python 3.13 from https://www.python.org/downloads/ (you can keep your"
echo "other Python installed), then start Reveal Cam again."
read -r -p "Press Enter to close this window." _
exit 1
