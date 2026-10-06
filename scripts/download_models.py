#!/usr/bin/env python3
"""Download the models into models/ (links from the Deep-Live-Cam README).

    python scripts/download_models.py              # swapper + face analysis
    python scripts/download_models.py --enhancer   # also the GFPGAN enhancer

The InsightFace `buffalo_l` detector/recognizer is fetched into models/buffalo_l/
the first time reveal_cam.py runs; pass --buffalo to fetch it now instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HF = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/"

FILES = {
    "inswapper_128_fp16.onnx": (HF + "inswapper_128_fp16.onnx", 277_680_638),
    "gfpgan-1024.onnx": (HF + "gfpgan-1024.onnx", 365_875_079),
}


def fetch(name: str, url: str, size: int) -> None:
    dest = MODELS / name
    if dest.exists() and dest.stat().st_size == size:
        print(f"[ok] {name} already present")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {name} ({size / 1e6:.0f} MB)")
    sys.path.insert(0, str(ROOT))
    from reveal.insight import urlopen

    with urlopen(url) as r, open(tmp, "wb") as fh:
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            print(f"\r      {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise SystemExit(f"{name}: size mismatch, download incomplete. Try again.")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enhancer", action="store_true", help="also download gfpgan-1024.onnx (~366 MB)")
    ap.add_argument("--buffalo", action="store_true", help="also fetch InsightFace buffalo_l now")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    fetch("inswapper_128_fp16.onnx", *FILES["inswapper_128_fp16.onnx"])
    if args.enhancer:
        fetch("gfpgan-1024.onnx", *FILES["gfpgan-1024.onnx"])
    if args.buffalo:
        sys.path.insert(0, str(ROOT))
        from reveal.insight import ensure_buffalo

        print(f"[ok] buffalo_l in {ensure_buffalo(MODELS).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
