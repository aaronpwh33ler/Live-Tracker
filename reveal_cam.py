#!/usr/bin/env python3
"""reveal_cam: real-time face swap that only shows through a "reveal window".

Live:     python reveal_cam.py --source faces/source.jpg
Offline:  python reveal_cam.py --source faces/source.jpg --input clip.mp4 --output out.mp4

Only use source faces of fictional (AI-generated) people or of real people who
have given consent. See README.md.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from reveal.config import ANIM_STYLES, DIRECTIONS, MASK_MODES, load_config, resolve_path
from reveal.engine import FaceEngine
from reveal.io import CameraReader, LiveRecorder, FFmpegWriter, VirtualCam, open_camera, remux_audio
from reveal.masks import RevealMask
from reveal.providers import provider_name, resolve_providers

WINDOW = "reveal_cam"

HELP_LINES = [
    "1/2/3/4  mask: rect / quad / face-follow / full",
    "m        cycle mask mode",
    "space    open / close the reveal (animated)",
    "a / d    cycle animation style / direction",
    "[ / ]    feather -/+      b  border on/off",
    "drag     move box / corners (rect, quad); 0 resets",
    "s        swap on/off      e  enhancer on/off",
    "r        record on/off    v  virtual camera on/off",
    "f        fullscreen       i  FPS/info overlay",
    "h        this help        q / Esc  quit",
]


class Pipeline:
    """Per-frame: detect -> swap (+enhance) -> composite through the mask."""

    def __init__(self, cfg: dict, engine: FaceEngine, mask: RevealMask):
        self.cfg = cfg
        self.engine = engine
        self.mask = mask
        self.skip_n = cfg["detection"]["skip_n"]
        self.all_faces = bool(cfg["swap"]["all_faces"])
        self.swap_on = bool(cfg["swap"]["enabled"])
        self.enhance_on = bool(cfg["enhancer"]["enabled"]) and engine.enhancer is not None
        self.faces: list = []
        self.n = 0
        self.timings = {"detect": 0.0, "swap": 0.0, "enhance": 0.0, "composite": 0.0}

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        h, w = frame.shape[:2]
        self.timings["swap"] = self.timings["enhance"] = 0.0
        t0 = time.perf_counter()
        if self.n % self.skip_n == 0:
            self.faces = self.engine.detect(frame, all_faces=self.all_faces)
        self.n += 1
        t1 = time.perf_counter()
        self.mask.update_faces(self.faces, w, h)

        swapped = frame
        region = self.mask.render(w, h)
        if self.swap_on and region is not None and self.faces:
            x0, y0, x1, y1 = region[1]
            visible = [f for f in self.faces
                       if f.bbox[0] < x1 and f.bbox[2] > x0 and f.bbox[1] < y1 and f.bbox[3] > y0]
            if visible:
                swapped = self.engine.swap(frame, visible)
                t2 = time.perf_counter()
                self.timings["swap"] = t2 - t1
                if self.enhance_on:
                    self.engine.enhance(swapped, visible)
                    self.timings["enhance"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        out, poly = self.mask.composite(frame, swapped)
        self.mask.draw_border(out, poly)
        self.mask.step()
        self.timings["detect"] = t1 - t0
        self.timings["composite"] = time.perf_counter() - t3
        return out, poly


def _gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _put(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(img, fps, pipe: Pipeline, provider: str, recording: bool, vcam: bool,
             show_info: bool, show_help: bool, thumb: np.ndarray | None) -> None:
    m = pipe.mask
    if show_info:
        t = pipe.timings
        lines = [
            f"FPS {fps:5.1f}   {provider.replace('ExecutionProvider', '')}",
            f"mask {m.mode}  anim {m.style}/{m.direction}  feather {m.feather}"
            f"  {'OPEN' if m.is_open else 'CLOSED'}",
            f"det {t['detect'] * 1000:4.0f}ms  swap {t['swap'] * 1000:4.0f}ms"
            f"  enh {t['enhance'] * 1000:4.0f}ms  skip-N {pipe.skip_n}",
            f"swap {'on' if pipe.swap_on else 'OFF'}  enhancer {'on' if pipe.enhance_on else 'off'}"
            f"  faces {len(pipe.faces)}",
        ]
        for i, line in enumerate(lines):
            _put(img, line, (10, 22 + 20 * i))
        if thumb is not None:
            th, tw = thumb.shape[:2]
            h, w = img.shape[:2]
            img[h - th - 10:h - 10, w - tw - 10:w - 10] = thumb
    y = img.shape[0] - 14
    if recording:
        cv2.circle(img, (18, y - 4), 7, (0, 0, 255), -1, cv2.LINE_AA)
        _put(img, "REC", (30, y), color=(0, 0, 255))
    if vcam:
        _put(img, "VCAM", (80, y), color=(0, 255, 0))
    if show_help:
        y0 = 110
        cv2.rectangle(img, (6, y0 - 18), (440, y0 + 20 * len(HELP_LINES)), (0, 0, 0), -1)
        for i, line in enumerate(HELP_LINES):
            _put(img, line, (12, y0 + 20 * i), scale=0.5)
    elif show_info:
        _put(img, "h = help", (10, 22 + 20 * 4), scale=0.45, color=(200, 200, 200))
    m.draw_handles(img)


def make_thumb(face: np.ndarray, size: int = 96) -> np.ndarray | None:
    if face is None or face.size == 0:
        return None
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)


class LiveControls:
    """Keyboard state for live mode: mask edits, recording, virtual camera, overlays."""

    def __init__(self, cfg: dict, pipe: Pipeline, w: int, h: int):
        self.cfg, self.pipe, self.w, self.h = cfg, pipe, w, h
        out = cfg["output"]
        self.rec_fps = float(cfg["camera"]["fps"])
        self.recorder: LiveRecorder | None = None
        self.vcam: VirtualCam | None = VirtualCam(w, h, self.rec_fps) if out["virtual_cam"] else None
        self.fullscreen = bool(out["fullscreen"])
        self.show_info = bool(out["show_fps"])
        self.show_help = False
        self.quit = False

    @property
    def vcam_on(self) -> bool:
        return bool(self.vcam and self.vcam.ok)

    def emit(self, frame: np.ndarray) -> None:
        if self.recorder:
            self.recorder.write(frame)
        if self.vcam_on:
            self.vcam.send(frame)

    def toggle_record(self) -> None:
        if self.recorder:
            self.recorder.close()
            print(f"[record] saved {self.recorder.path}")
            self.recorder = None
        else:
            out = self.cfg["output"]
            path = resolve_path(out["record_dir"]) / f"reveal_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            print(f"[record] -> {path}")
            self.recorder = LiveRecorder(path, self.w, self.h, self.rec_fps, int(out["crf"]))

    def toggle_vcam(self) -> None:
        if self.vcam_on:
            self.vcam.close()
            self.vcam = None
        else:
            self.vcam = VirtualCam(self.w, self.h, self.rec_fps)

    def handle_key(self, key: int) -> None:
        if key == 255 or key < 0:
            return
        m, pipe = self.pipe.mask, self.pipe
        c = chr(key) if key < 128 else ""
        if key == 27 or c == "q":
            self.quit = True
        elif c and c in "1234":
            m.set_mode(MASK_MODES[int(c) - 1])
        elif c == "m":
            m.cycle_mode()
        elif c == " ":
            m.toggle()
        elif c == "a":
            m.style = ANIM_STYLES[(ANIM_STYLES.index(m.style) + 1) % len(ANIM_STYLES)]
        elif c == "d":
            m.direction = DIRECTIONS[(DIRECTIONS.index(m.direction) + 1) % len(DIRECTIONS)]
        elif c == "[":
            m.feather = max(0, m.feather - 5)
        elif c == "]":
            m.feather = min(200, m.feather + 5)
        elif c == "b":
            m.border_on = not m.border_on
        elif c == "0":
            m.reset_geometry()
        elif c == "s":
            pipe.swap_on = not pipe.swap_on
        elif c == "e":
            pipe.enhance_on = (not pipe.enhance_on) and pipe.engine.load_enhancer()
        elif c == "r":
            self.toggle_record()
        elif c == "v":
            self.toggle_vcam()
        elif c == "f":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif c == "i":
            self.show_info = not self.show_info
        elif c == "h":
            self.show_help = not self.show_help

    def close(self) -> None:
        if self.recorder:
            self.toggle_record()
        if self.vcam:
            self.vcam.close()


# --------------------------------------------------------------------- live
def run_live(cfg: dict, args, pipe: Pipeline, provider: str, thumb) -> None:
    cam_cfg = cfg["camera"]
    cap = open_camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"])
    is_file = isinstance(cam_cfg["index"], str)
    reader = CameraReader(cap, mirror=bool(cam_cfg["mirror"]),
                          pace_fps=(cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0)
    seq, frame = reader.read(0, timeout=5.0)
    if frame is None:
        reader.stop()
        raise SystemExit("Camera opened but returned no frames.")
    h, w = frame.shape[:2]
    print(f"[live] camera {w}x{h}, provider {provider}. Press h in the window for help.")

    out_cfg = cfg["output"]
    preview = bool(out_cfg["preview"]) and _gui_available()
    if out_cfg["preview"] and not preview:
        print("[live] no GUI available (opencv-python-headless?); running without a preview window.")
    if preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, w, h)
        cv2.setMouseCallback(WINDOW, lambda e, x, y, fl, p: pipe.mask.on_mouse(e, x, y, w, h))
        if out_cfg["fullscreen"]:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ctl = LiveControls(cfg, pipe, w, h)
    if args.record:
        ctl.toggle_record()
    fps, last = 0.0, time.perf_counter()
    frames = 0
    fps_log = []
    t_start = None

    try:
        while not ctl.quit:
            seq, frame = reader.read(seq)
            if frame is None:
                print("[live] camera stopped delivering frames.")
                break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out, _ = pipe.process(frame)
            ctl.emit(out)

            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last)
            last = now
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            frames += 1
            if frames == 10:
                t_start = now  # skip warm-up frames in the summary
            elif frames > 10:
                fps_log.append(inst)

            if preview:
                disp = out.copy()
                draw_hud(disp, fps, pipe, provider, ctl.recorder is not None, ctl.vcam_on,
                         ctl.show_info, ctl.show_help, thumb)
                cv2.imshow(WINDOW, disp)
                ctl.handle_key(cv2.waitKey(1) & 0xFF)
            elif frames % 30 == 0:
                print(f"[live] {frames} frames, {fps:.1f} FPS")
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctl.close()
        reader.stop()
        if preview:
            cv2.destroyAllWindows()
    if fps_log:
        print(f"[live] {len(fps_log) / (last - t_start):.1f} FPS average over {len(fps_log)} frames "
              f"(median {np.median(fps_log):.1f}, 10th percentile {np.percentile(fps_log, 10):.1f})")


# ------------------------------------------------------------------ offline
def run_offline(cfg: dict, args, pipe: Pipeline, provider: str) -> None:
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = Path(args.output).expanduser().resolve() if args.output else \
        in_path.with_name(in_path.stem + "_reveal.mp4")
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mirror = args.mirror is True  # offline: mirror only when asked on the command line

    keyframes = sorted(cfg["animation"]["keyframes"], key=lambda k: float(k["t"]))
    if keyframes:
        # Start in the opposite state of the first keyframe ("open at 2s" => closed before 2s).
        pipe.mask.set_open(keyframes[0]["action"] != "open", instant=True)
    kf_i = 0

    tmp = out_path.with_name(out_path.stem + ".video_only.mp4")
    writer = FFmpegWriter(tmp, w, h, fps, int(cfg["output"]["crf"]))
    preview = args.preview is True and _gui_available()
    print(f"[offline] {in_path.name}: {w}x{h} @ {fps:.2f} fps, {total or '?'} frames, provider {provider}")

    t_start = time.perf_counter()
    last_print = t_start
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            t = idx / fps
            while kf_i < len(keyframes) and t >= float(keyframes[kf_i]["t"]):
                pipe.mask.set_open(keyframes[kf_i]["action"] == "open")
                kf_i += 1
            out, _ = pipe.process(frame)
            writer.write(out)
            idx += 1
            if preview:
                cv2.imshow(WINDOW, out)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("[offline] aborted by user")
                    break
            now = time.perf_counter()
            if now - last_print >= 2.0 or idx == total:
                rate = idx / (now - t_start)
                eta = (total - idx) / rate if total and rate > 0 else 0
                pct = f"{100 * idx / total:5.1f}%" if total else ""
                print(f"[offline] {idx}/{total or '?'} {pct}  {rate:.2f} FPS  ETA {eta:5.0f}s", flush=True)
                last_print = now
            if args.max_frames and idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("[offline] interrupted; finalizing partial output")
    finally:
        cap.release()
        writer.close()
        if preview:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    print(f"[offline] processed {idx} frames in {elapsed:.1f}s ({idx / max(elapsed, 1e-6):.2f} FPS)")
    has_audio = remux_audio(tmp, in_path, out_path)
    print(f"[offline] wrote {out_path}" + (" (with source audio)" if has_audio else ""))


def main(argv: list[str] | None = None) -> int:
    cfg, args = load_config(argv)
    if isinstance(cfg["camera"]["index"], str) and cfg["camera"]["index"].isdigit():
        cfg["camera"]["index"] = int(cfg["camera"]["index"])

    source = resolve_path(cfg["source_face"])
    if source is None or not source.exists():
        raise SystemExit(
            f"Source face not found: {source}\n"
            "Use --source path/to/face.jpg (an AI-generated face or a consenting person)."
        )

    providers = resolve_providers(cfg["execution_provider"])
    provider = provider_name(providers)
    if provider == "CPUExecutionProvider" and not args.input:
        print("[warn] Running live on CPU: expect a low frame rate. Offline mode "
              "(--input/--output) is recommended on this machine.")

    engine = FaceEngine(resolve_path(cfg["models_dir"]), providers,
                        det_width=cfg["detection"]["width"], min_score=float(cfg["detection"]["min_score"]),
                        enhancer=bool(cfg["enhancer"]["enabled"]))
    actual = engine.swapper.session.get_providers()[0]
    if actual != provider:
        print(f"[provider] requested {provider} but onnxruntime is running on {actual}. "
              f"See TROUBLESHOOTING.md -> 'onnxruntime provider conflicts'.")
        provider = actual
    face_crop = engine.set_source(source)
    print(f"[source] embedding computed from {source.name}")
    mask = RevealMask(cfg)
    pipe = Pipeline(cfg, engine, mask)

    if args.input:
        run_offline(cfg, args, pipe, provider)
    else:
        run_live(cfg, args, pipe, provider, make_thumb(face_crop))
    return 0


if __name__ == "__main__":
    sys.exit(main())
