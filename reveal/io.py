"""Camera reader thread, ffmpeg recording, audio remux and virtual camera."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def find_ffmpeg() -> str | None:
    """System ffmpeg first, then the binary bundled with imageio-ffmpeg (project venv)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def open_camera(index: int | str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a webcam by index. A file path or stream URL also works (handy for testing)."""
    import platform

    if isinstance(index, str) and not index.isdigit():
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source: {index}")
        return cap
    index = int(index)
    system = platform.system()
    backends = {"Windows": [cv2.CAP_DSHOW, cv2.CAP_MSMF], "Darwin": [cv2.CAP_AVFOUNDATION],
                "Linux": [cv2.CAP_V4L2]}.get(system, []) + [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise SystemExit(
        f"Could not open camera {index}. Check that it is connected and not used by another app. "
        f"On macOS, allow camera access for your terminal app (see TROUBLESHOOTING.md)."
    )


class CameraReader:
    """Grabs frames on a background thread and keeps only the newest one, so
    slow processing never builds up latency."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool, pace_fps: float = 0.0):
        self.cap = cap
        self.mirror = mirror
        self.pace = 1.0 / pace_fps if pace_fps > 0 else 0.0  # video files: emulate a live camera
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._running = True
        self.failed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        misses = 0
        next_t = time.monotonic()
        while self._running:
            if self.pace:
                next_t += self.pace
                time.sleep(max(0.0, next_t - time.monotonic()))
            ok, frame = self.cap.read()
            if not ok:
                misses += 1
                if misses > 100:
                    self.failed = True
                    break
                time.sleep(0.01)
                continue
            misses = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq: int, timeout: float = 2.0) -> tuple[int, np.ndarray | None]:
        """Wait for a frame newer than `last_seq`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._seq, self._frame
            if self.failed:
                break
            time.sleep(0.001)
        return last_seq, None

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class FFmpegWriter:
    """Pipe BGR frames into ffmpeg -> H.264 MP4. Falls back to OpenCV mp4v if
    ffmpeg is missing."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.proc = None
        self.cv_writer = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("[record] ffmpeg not found; writing mp4v with OpenCV (install ffmpeg for H.264).")
            self.cv_writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self.cv_writer is not None:
            self.cv_writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            if self.proc.returncode != 0:
                print(f"[record] ffmpeg exited with code {self.proc.returncode}")
        if self.cv_writer is not None:
            self.cv_writer.release()


class LiveRecorder:
    """Records at a fixed frame rate, duplicating or dropping frames so the
    file plays back at real-time speed even when processing FPS varies."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int):
        self.fps = fps
        self.writer = FFmpegWriter(path, width, height, fps, crf)
        self.t0 = time.monotonic()
        self.path = self.writer.path

    def write(self, frame: np.ndarray) -> None:
        due = int((time.monotonic() - self.t0) * self.fps) + 1
        n = due - self.writer.frames
        for _ in range(max(0, n)):
            self.writer.write(frame)

    def close(self) -> None:
        self.writer.close()


def remux_audio(video_only: Path, audio_source: Path, out_path: Path) -> bool:
    """Copy the processed video stream and the source's audio into `out_path`."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.move(str(video_only), str(out_path))
        print("[audio] ffmpeg not found; output has no audio.")
        return False
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_only), "-i", str(audio_source),
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[audio] remux failed; writing video without audio.")
        shutil.move(str(video_only), str(out_path))
        return False
    video_only.unlink(missing_ok=True)
    return True


class VirtualCam:
    """Optional pyvirtualcam output. `ok` is False if no virtual camera is installed."""

    def __init__(self, width: int, height: int, fps: float):
        self.cam = None
        try:
            import pyvirtualcam  # type: ignore

            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                           fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[virtualcam] sending to {self.cam.device}")
        except Exception as e:  # ImportError or no backend installed
            print(f"[virtualcam] skipped: {e}")

    @property
    def ok(self) -> bool:
        return self.cam is not None

    def send(self, frame: np.ndarray) -> None:
        if self.cam is not None:
            self.cam.send(frame)

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()
            self.cam = None
