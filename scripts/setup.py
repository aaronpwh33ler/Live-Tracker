#!/usr/bin/env python3
"""One-shot setup: check hardware, create .venv, install the matching
onnxruntime build, download models.

    python3.11 scripts/setup.py                 # use the provider check_env picks
    python3.11 scripts/setup.py --provider cpu  # force a path: cuda|coreml|directml|openvino|cpu

Run it with the Python you want the venv to use (3.11+; 3.11-3.13 have wheels
for every provider package).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
REQS = {
    "cuda": "requirements-cuda.txt",
    "coreml": "requirements-coreml.txt",
    "directml": "requirements-directml.txt",
    "openvino": "requirements-openvino.txt",
    "cpu": "requirements-cpu.txt",
}
ORT_PACKAGES = ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino",
                "onnxruntime-silicon", "onnxruntime-coreml"]


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(cmd: list) -> None:
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(REQS), help="override the detected hardware path")
    ap.add_argument("--enhancer", action="store_true", help="also download the GFPGAN enhancer model")
    ap.add_argument("--from-launcher", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit(f"Python {sys.version.split()[0]} is too old: onnxruntime needs 3.11+. "
                         f"Run this script with python3.11 (or newer).")

    run([sys.executable, ROOT / "scripts" / "check_env"])
    report = json.loads((ROOT / ".env_report.json").read_text())
    req = REQS[args.provider] if args.provider else report["requirements_file"]

    if not venv_python().exists():
        print(f"Creating virtual environment in {VENV.name}/")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    # Several onnxruntime packages share the same import name; only one may be installed.
    run([py, "-m", "pip", "uninstall", "-y", *ORT_PACKAGES])
    run([py, "-m", "pip", "install", "-r", ROOT / req])
    run([py, ROOT / "scripts" / "download_models.py", "--buffalo"] + (["--enhancer"] if args.enhancer else []))
    run([py, "-c", "import onnxruntime as o; print('onnxruntime providers:', o.get_available_providers())"])
    run([py, "-c", "import cv2; print('OpenCV', cv2.__version__, 'GUI:', "
                   "[l.split(':')[1].strip() for l in cv2.getBuildInformation().splitlines() if 'GUI:' in l])"])

    if args.from_launcher:
        return 0
    activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    print("\nDone. Next:")
    print(f"  {activate}")
    if req == REQS["cpu"]:
        print("  python reveal_cam.py --input your_clip.mp4 --output out.mp4   # CPU-only: offline mode")
    else:
        print("  python reveal_cam.py --source faces/source.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
