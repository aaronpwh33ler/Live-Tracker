#!/usr/bin/env python3
"""Measure per-stage timings on this machine, for the README's numbers.

    python scripts/benchmark.py --video clip.mp4 [--provider auto] [--frames 60]

Prints detection time at several widths, swap time, enhancer time (if the
model is present), and the resulting pipeline FPS for common settings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reveal.engine import ENHANCER_FILE, FaceEngine  # noqa: E402
from reveal.providers import provider_name, resolve_providers  # noqa: E402


def timeit(fn, n: int) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="clip with a visible face")
    ap.add_argument("--source", default=str(ROOT / "faces" / "source.jpg"))
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--frames", type=int, default=10, help="repetitions per measurement")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frame = None
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        raise SystemExit("could not read video")
    h, w = frame.shape[:2]

    providers = resolve_providers(args.provider)
    has_enh = (ROOT / "models" / ENHANCER_FILE).exists()
    eng = FaceEngine(ROOT / "models", providers, enhancer=has_enh)
    eng.set_source(Path(args.source))
    prov = eng.swapper.session.get_providers()[0]
    print(f"\nprovider {prov} (requested {provider_name(providers)}), frame {w}x{h}")

    det = {}
    for dw in (320, 480, 640):
        eng.det_width = dw
        det[dw] = timeit(lambda: eng.detect(frame), args.frames)
        print(f"detect @ {dw:4d}px : {det[dw] * 1000:7.1f} ms")
    eng.det_width = 640
    faces = eng.detect(frame)
    if not faces:
        raise SystemExit("no face found in the clip's frame 15")
    swap = timeit(lambda: eng.swap(frame, faces), args.frames)
    print(f"swap (1 face)   : {swap * 1000:7.1f} ms")
    enh = 0.0
    if has_enh:
        sw = eng.swap(frame, faces)
        enh = timeit(lambda: eng.enhance(sw.copy(), faces), max(3, args.frames // 3))
        print(f"enhance (GFPGAN): {enh * 1000:7.1f} ms")

    print("\nestimated pipeline FPS (1 face, mask open):")
    for dw, skip in ((640, 1), (480, 1), (320, 1), (320, 3)):
        per = det[dw] / skip + swap
        line = f"  det {dw}, skip-N {skip}: {1 / per:6.1f} FPS"
        if has_enh:
            line += f"   with enhancer: {1 / (per + enh):6.2f} FPS"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
