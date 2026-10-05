#!/usr/bin/env python3
"""Step 2 baseline: install the stock Deep-Live-Cam app into third_party/ to
confirm the models and acceleration backend work on this machine.

    python3.11 scripts/setup_baseline.py [--provider cuda|coreml|dml|openvino|cpu]

It clones Deep-Live-Cam into third_party/Deep-Live-Cam, creates its own venv
there (third_party/Deep-Live-Cam/venv), installs its requirements, swaps in the
onnxruntime build for your provider (per its README), and links our models/
into its models folder. Then it prints the commands to run the stock app.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLC = ROOT / "third_party" / "Deep-Live-Cam"
REPO = "https://github.com/hacksider/Deep-Live-Cam.git"

# From the Deep-Live-Cam README's GPU acceleration section.
ORT = {
    "cuda": ["onnxruntime-gpu==1.26.0"],
    "coreml": None,  # requirements.txt already installs the CoreML-capable wheel on Apple Silicon
    "dml": ["onnxruntime-directml==1.21.0"],
    "openvino": ["onnxruntime-openvino==1.21.0"],
    "cpu": None,
}
FLAGS = {"cuda": "cuda", "coreml": "coreml", "dml": "dml", "openvino": "openvino", "cpu": "cpu"}
PROVIDER_KEY = {
    "CUDAExecutionProvider": "cuda", "CoreMLExecutionProvider": "coreml", "DmlExecutionProvider": "dml",
    "OpenVINOExecutionProvider": "openvino", "CPUExecutionProvider": "cpu",
}


def run(cmd, **kw) -> None:
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(ORT))
    args = ap.parse_args()
    if sys.version_info < (3, 11):
        raise SystemExit("Deep-Live-Cam needs Python 3.11+ (its README: onnxruntime dropped 3.10).")

    provider = args.provider
    if not provider:
        run([sys.executable, ROOT / "scripts" / "check_env"])
        provider = PROVIDER_KEY[json.loads((ROOT / ".env_report.json").read_text())["provider"]]

    if not DLC.exists():
        DLC.parent.mkdir(exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO, DLC])
    py = DLC / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        venv.EnvBuilder(with_pip=True).create(DLC / "venv")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", DLC / "requirements.txt"])
    if ORT[provider]:
        run([py, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu",
             "onnxruntime-directml", "onnxruntime-openvino"])
        run([py, "-m", "pip", "install", *ORT[provider]])

    # Reuse our downloaded models instead of fetching them twice.
    for name in ("inswapper_128_fp16.onnx", "gfpgan-1024.onnx"):
        src, dst = ROOT / "models" / name, DLC / "models" / name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:  # Windows without symlink rights
                shutil.copy2(src, dst)

    rel = os.path.relpath(DLC, Path.cwd())
    pyrel = os.path.relpath(py, DLC)
    print("\nBaseline installed. From", rel, "run:")
    print(f"  {pyrel} run.py --execution-provider {FLAGS[provider]}        # GUI: pick a face, click Live")
    print(f"  {pyrel} run.py -s ../../faces/source.jpg -t clip.mp4 -o out.mp4 "
          f"--frame-processor face_swapper --keep-fps --keep-audio --execution-provider {FLAGS[provider]}")
    print("Note: the stock app downloads buffalo_l into ~/.insightface (outside this project).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
