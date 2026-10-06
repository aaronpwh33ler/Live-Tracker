#!/usr/bin/env python3
"""Start Reveal Cam: set everything up on first run, then open the app window.

The double-click launchers (Reveal Cam.command, Reveal Cam.bat, start.sh) run
this with any Python 3.11+. Standard library only.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".reveal_setup"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def setup_fingerprint() -> str:
    """Changes whenever the requirements change, so updates re-run setup."""
    h = hashlib.sha256()
    for f in sorted(ROOT.glob("requirements-*.txt")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def is_ready() -> bool:
    py = venv_python()
    if not py.exists() or not MARKER.exists() or MARKER.read_text().strip() != setup_fingerprint():
        return False
    if not (ROOT / "models" / "inswapper_128_fp16.onnx").exists():
        return False
    return subprocess.run([str(py), "-c", "import PySide6, onnxruntime, cv2, onnx, yaml"],
                          capture_output=True).returncode == 0


def pause(msg: str) -> None:
    print(msg)
    try:
        input("Press Enter to close this window.")
    except EOFError:
        pass


def main() -> int:
    if not is_ready():
        if sys.version_info < (3, 11):
            pause(f"Reveal Cam needs Python 3.11 or newer (found {sys.version.split()[0]}).\n"
                  "Install it from https://www.python.org/downloads/ and try again.")
            return 1
        print("=" * 64)
        print(" Setting up Reveal Cam. This happens once and downloads about")
        print(" 1 GB (Python packages and AI models). It can take a while.")
        print("=" * 64)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "setup.py"), "--from-launcher"], cwd=ROOT)
        if r.returncode != 0:
            pause("\nSetup did not finish. Scroll up for the error; TROUBLESHOOTING.md covers "
                  "the common ones. Double-click again to retry.")
            return 1
        MARKER.write_text(setup_fingerprint())
    print("Opening Reveal Cam…")
    r = subprocess.run([str(venv_python()), str(ROOT / "app.py")], cwd=ROOT)
    if r.returncode != 0:
        pause("\nReveal Cam closed with an error (see above).")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
