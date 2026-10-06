#!/usr/bin/env bash
# Update Reveal Cam to the latest version in place. Keeps your .venv, models
# and settings, so there is no big re-download.
set -e
cd "$(dirname "$0")"
URL="https://github.com/aaronpwh33ler/Live-Tracker/archive/refs/heads/claude/reveal-window-face-swap.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Downloading the latest Reveal Cam…"
curl -fsSL -o "$TMP/rc.zip" "$URL"
unzip -q "$TMP/rc.zip" -d "$TMP/x"
cp -R "$TMP"/x/*/. .
chmod +x start.sh update.sh "Reveal Cam.command" 2>/dev/null || true
echo "Updated. Start it with ./start.sh"
