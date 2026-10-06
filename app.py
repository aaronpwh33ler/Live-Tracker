#!/usr/bin/env python3
"""Reveal Cam: the simple desktop window.

Pick a photo, pick a camera, press Start. Started by the double-click
launchers (Reveal Cam.command / Reveal Cam.bat / start.sh) via start.py,
or directly: .venv/bin/python app.py

The camera view itself is reveal_cam.py, run as a child process.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path

import cv2

# opencv-python points Qt at its own (Qt5) plugins; PySide6 needs its own.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PySide6.QtCore import QObject, QProcess, Qt, QTimer, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QDesktopServices, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / ".launcher.json"
PHOTO_FILTER = "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
SHAPES = [("Box", "rect"), ("Tilted box", "quad"), ("Follows my face", "face"), ("Whole picture", "full")]
KEYS_HELP = ("In the camera window:  Space show/hide the swap  ·  1–4 change shape  ·  "
             "drag to move the box  ·  R record  ·  F fullscreen  ·  Q close")

STYLE = """
QWidget { font-size: 14px; }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#muted { font-size: 13px; }
QFrame#card { border: 1px solid palette(mid); border-radius: 10px; }
QLabel#thumb { border: 1px dashed palette(mid); border-radius: 8px; }
QPushButton { padding: 7px 14px; }
QPushButton#primary { font-size: 16px; font-weight: 600; padding: 11px 18px; }
"""


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def to_pixmap(bgr, size: int) -> QPixmap:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def list_cameras() -> list[tuple[str, int]]:
    """(label, index) for each camera the OS reports; falls back to plain indexes."""
    cams: list[tuple[str, int]] = []
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        backend = {"Windows": cv2.CAP_DSHOW, "Darwin": cv2.CAP_AVFOUNDATION,
                   "Linux": cv2.CAP_V4L2}.get(platform.system(), cv2.CAP_ANY)
        seen = set()
        for info in enumerate_cameras(backend):
            idx = info.index - info.backend if info.backend and info.index >= info.backend else info.index
            if idx in seen:
                continue
            seen.add(idx)
            cams.append((info.name or f"Camera {idx}", idx))
    except Exception:
        pass
    if not cams:
        cams = [("Default camera", 0), ("Camera 2", 1), ("Camera 3", 2)]
    return cams


def hardware_note() -> tuple[str, bool]:
    """(sentence about expected speed, is_cpu_only)."""
    report = ROOT / ".env_report.json"
    if not report.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_env"), "--json"], capture_output=True)
    try:
        r = json.loads(report.read_text())
    except (OSError, ValueError):
        return "", False
    p = r.get("provider", "")
    if p == "CUDAExecutionProvider":
        return "NVIDIA graphics card found: the live camera should be smooth.", False
    if p == "CoreMLExecutionProvider":
        return "Apple Silicon Mac: the live camera works, at a modest frame rate.", False
    if p == "DmlExecutionProvider":
        return "Graphics card found: live speed depends on the card.", False
    if p == "OpenVINOExecutionProvider":
        return "Intel graphics: the live camera will be slow. Making a video works well.", False
    return ("This computer has no supported graphics card, so the live camera will be very slow "
            "(about 1 frame a second). “Make a video” works well: it just takes a while."), True


class Signals(QObject):
    photo_checked = Signal(str, object, str)  # path, face crop (or None), message
    cameras_found = Signal(list)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reveal Cam")
        self.setMinimumWidth(560)
        self.settings = load_settings()
        self.signals = Signals()
        self.signals.photo_checked.connect(self._photo_checked)
        self.signals.cameras_found.connect(self._cameras_found)
        self.photo: str | None = None
        self.photo_ok = False
        self.proc: QProcess | None = None
        self.proc_kind = ""
        self.proc_log: list[str] = []
        self.output_path: Path | None = None
        self._analyser = None
        self._analyser_lock = threading.Lock()

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(14)

        title = QLabel("Reveal Cam")
        title.setObjectName("title")
        sub = QLabel("Swap your face with a photo, shown through a window on your camera.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(sub)

        # 1. Photo
        card = self._card(root, "1. Who do you want to become?")
        row = QHBoxLayout()
        self.thumb = QLabel("No photo")
        self.thumb.setObjectName("thumb")
        self.thumb.setFixedSize(120, 120)
        self.thumb.setAlignment(Qt.AlignCenter)
        row.addWidget(self.thumb)
        col = QVBoxLayout()
        self.choose_btn = QPushButton("Choose photo…")
        self.choose_btn.clicked.connect(self.choose_photo)
        self.photo_status = QLabel("A clear photo of their face works best. A full-body photo is OK too.")
        self.photo_status.setWordWrap(True)
        self.photo_status.setMinimumHeight(44)
        self.consent = QCheckBox("This is an AI-generated person, or someone who agreed to this")
        self.consent.toggled.connect(self._update_buttons)
        col.addWidget(self.choose_btn, alignment=Qt.AlignLeft)
        col.addWidget(self.photo_status)
        col.addStretch(1)
        row.addLayout(col, 1)
        card.addLayout(row)
        card.addWidget(self.consent)

        # 2. Camera + shape
        card = self._card(root, "2. Camera and window shape")
        grid = QGridLayout()
        grid.addWidget(QLabel("Camera"), 0, 0)
        self.camera = QComboBox()
        self.camera.addItem("Looking for cameras…", None)
        grid.addWidget(self.camera, 0, 1, 1, 4)
        grid.addWidget(QLabel("Shape"), 1, 0)
        self.shape_group = QButtonGroup(self)
        saved_shape = self.settings.get("shape", "rect")
        for i, (label, key) in enumerate(SHAPES):
            rb = QRadioButton(label)
            rb.setProperty("key", key)
            rb.setChecked(key == saved_shape)
            self.shape_group.addButton(rb, i)
            grid.addWidget(rb, 1, 1 + i)
        self.sharper = QCheckBox("Sharper face (much slower)")
        self.sharper.setChecked(bool(self.settings.get("sharper", False)))
        grid.addWidget(self.sharper, 2, 1, 1, 4)
        card.addLayout(grid)

        # 3. Go
        note, self.cpu_only = hardware_note()
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start camera")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_live)
        self.video_btn = QPushButton("Make a video from a file…")
        self.video_btn.clicked.connect(self.start_video)
        if self.cpu_only:
            self.video_btn.setObjectName("primary")
        btns.addWidget(self.start_btn, 1)
        btns.addWidget(self.video_btn, 1)
        root.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.hide()
        root.addWidget(self.progress)
        self.status = QLabel(note)
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        root.addWidget(self.status)
        self.open_btn = QPushButton("Show the video")
        self.open_btn.clicked.connect(self.open_output)
        self.open_btn.hide()
        root.addWidget(self.open_btn, alignment=Qt.AlignLeft)
        keys = QLabel(KEYS_HELP)
        keys.setWordWrap(True)
        keys.setObjectName("muted")
        root.addWidget(keys)

        self._update_buttons()
        threading.Thread(target=lambda: self.signals.cameras_found.emit(list_cameras()), daemon=True).start()
        last = self.settings.get("photo")
        if last and Path(last).exists():
            self.set_photo(last)

    def _card(self, parent: QVBoxLayout, heading: str) -> QVBoxLayout:
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 14)
        h = QLabel(heading)
        h.setStyleSheet("font-weight: 600;")
        lay.addWidget(h)
        parent.addWidget(frame)
        return lay

    # ------------------------------------------------------------ photo
    def choose_photo(self) -> None:
        start = str(Path(self.photo).parent) if self.photo else str(ROOT / "faces")
        path, _ = QFileDialog.getOpenFileName(self, "Choose a photo", start, PHOTO_FILTER)
        if path:
            self.set_photo(path)

    def set_photo(self, path: str) -> None:
        self.photo, self.photo_ok = path, False
        if path != self.settings.get("photo"):
            self.consent.setChecked(False)
        else:
            self.consent.setChecked(bool(self.settings.get("consent")))
        self.photo_status.setText("Looking for a face…")
        self.thumb.setText("…")
        self._update_buttons()
        threading.Thread(target=self._check_photo, args=(path,), daemon=True).start()

    def _check_photo(self, path: str) -> None:
        try:
            from reveal.engine import SMALL_FACE_PX, face_crop, face_height, find_source_face, \
                load_face_analysis, read_image

            img = read_image(Path(path))
            if img is None:
                self.signals.photo_checked.emit(path, None, "Couldn't open this image. Use a JPG or PNG "
                                                            "(iPhone photos: export or share as JPG).")
                return
            with self._analyser_lock:
                if self._analyser is None:
                    self._analyser = load_face_analysis(ROOT / "models", ["CPUExecutionProvider"])
                face = find_source_face(self._analyser, img)
            if face is None:
                self.signals.photo_checked.emit(path, None, "No face found in this photo. Try one where "
                                                            "the face is clearly visible and facing forward.")
                return
            msg = "Face found."
            if face_height(face) < SMALL_FACE_PX:
                msg = ("Face found, but it's small in this photo. A closer photo of the face "
                       "will look more like them.")
            self.signals.photo_checked.emit(path, face_crop(img, face), msg)
        except Exception as e:  # keep the window alive whatever happens
            self.signals.photo_checked.emit(path, None, f"Couldn't check this photo: {e}")

    def _photo_checked(self, path: str, crop, msg: str) -> None:
        if path != self.photo:
            return  # a newer choice is in flight
        self.photo_ok = crop is not None
        self.photo_status.setText(msg)
        if crop is not None:
            self.thumb.setPixmap(to_pixmap(crop, 116))
        else:
            self.thumb.setText("No face")
        self._update_buttons()

    # ---------------------------------------------------------- cameras
    def _cameras_found(self, cams: list) -> None:
        self.camera.clear()
        for label, idx in cams:
            self.camera.addItem(label, idx)
        saved = self.settings.get("camera")
        i = self.camera.findData(saved)
        if i >= 0:
            self.camera.setCurrentIndex(i)
        self._update_buttons()

    # ------------------------------------------------------------- run
    def _ready(self) -> bool:
        return self.photo_ok and self.consent.isChecked() and self.proc is None

    def _update_buttons(self) -> None:
        running = self.proc is not None
        self.start_btn.setEnabled(self._ready() and self.camera.currentData() is not None
                                  or (running and self.proc_kind == "live"))
        self.start_btn.setText("Stop camera" if running and self.proc_kind == "live" else "Start camera")
        self.video_btn.setEnabled(self._ready() or (running and self.proc_kind == "video"))
        self.video_btn.setText("Cancel" if running and self.proc_kind == "video" else "Make a video from a file…")
        self.choose_btn.setEnabled(not running)

    def _common_args(self) -> list[str]:
        shape = self.shape_group.checkedButton().property("key")
        args = ["--source", self.photo, "--mask", shape]
        args.append("--enhancer" if self.sharper.isChecked() else "--no-enhancer")
        if self.sharper.isChecked() and not (ROOT / "models" / "gfpgan-1024.onnx").exists():
            args = ["__download_enhancer__"] + args
        self.settings.update(photo=self.photo, consent=True, shape=shape, sharper=self.sharper.isChecked(),
                             camera=self.camera.currentData())
        save_settings(self.settings)
        return args

    def start_live(self) -> None:
        if self.proc is not None:
            self.stop_live()
            return
        args = self._common_args() + ["--camera", str(self.camera.currentData()), "--stdin-control"]
        self._run("live", args, "Opening the camera… (a new window will appear)")

    def stop_live(self) -> None:
        p = self.proc
        if p is None:
            return
        p.write(b"quit\n")
        p.closeWriteChannel()
        QTimer.singleShot(5000, lambda: p.kill() if p.state() != QProcess.NotRunning else None)

    def start_video(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            return
        src, _ = QFileDialog.getOpenFileName(self, "Choose a video", self.settings.get("video_dir", ""),
                                             VIDEO_FILTER)
        if not src:
            return
        srcp = Path(src)
        default = srcp.with_name(srcp.stem + "_reveal.mp4")
        out, _ = QFileDialog.getSaveFileName(self, "Save the new video as", str(default), "MP4 video (*.mp4)")
        if not out:
            return
        if not out.lower().endswith(".mp4"):
            out += ".mp4"
        self.settings["video_dir"] = str(srcp.parent)
        self.output_path = Path(out)
        args = self._common_args() + ["--input", src, "--output", out]
        self.progress.setRange(0, 0)
        self.progress.show()
        self._run("video", args, "Getting ready…")

    def _run(self, kind: str, args: list[str], msg: str) -> None:
        self.open_btn.hide()
        self.proc_log = []
        self.proc_kind = kind
        if args and args[0] == "__download_enhancer__":
            # Fetch the enhancer model first, then run the real command.
            args = args[1:]
            self._spawn([str(ROOT / "scripts" / "download_models.py"), "--enhancer"],
                        "Downloading the face sharpening model (about 370 MB)…",
                        then=lambda: self._spawn([str(ROOT / "reveal_cam.py")] + args, msg))
        else:
            self._spawn([str(ROOT / "reveal_cam.py")] + args, msg)
        self._update_buttons()

    def _spawn(self, argv: list[str], msg: str, then=None) -> None:
        self.status.setText(msg)
        p = QProcess(self)
        p.setWorkingDirectory(str(ROOT))
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._read(p))
        p.finished.connect(lambda code, _status: self._finished(p, code, then))
        self.proc = p
        p.start(sys.executable, ["-u"] + argv)

    def _read(self, p: QProcess) -> None:
        text = bytes(p.readAllStandardOutput()).decode(errors="replace")
        for line in re.split(r"[\r\n]+", text):
            line = line.strip()
            if not line:
                continue
            self.proc_log = (self.proc_log + [line])[-40:]
            m = re.search(r"\[offline\] (\d+)/(\d+)\s+[\d.]+%\s+([\d.]+) FPS\s+ETA\s+(\d+)s", line)
            if m:
                done, total, eta = int(m.group(1)), int(m.group(2)), int(m.group(4))
                self.progress.setRange(0, total)
                self.progress.setValue(done)
                mins, secs = divmod(eta, 60)
                self.status.setText(f"Making your video… frame {done} of {total}, about "
                                    f"{f'{mins} min ' if mins else ''}{secs} s left.")
            elif line.startswith("[live] camera"):
                self.status.setText("Camera running. Click the camera window and press H for all keys.")
            elif line.startswith("[record] saved"):
                self.status.setText("Recording saved in the “recordings” folder.")
            m = re.search(r"([\d.]+) / ([\d.]+) MB", line)
            if m:
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(1000 * float(m.group(1)) / float(m.group(2))))
                self.progress.show()

    def _finished(self, p: QProcess, code: int, then) -> None:
        if p is not self.proc:
            return
        self.proc = None
        if code == 0 and then is not None:
            self.progress.hide()
            then()
            self._update_buttons()
            return
        self.progress.hide()
        kind, self.proc_kind = self.proc_kind, ""
        if code == 0 or (kind == "live" and p.exitStatus() == QProcess.CrashExit):
            if kind == "video" and self.output_path and self.output_path.exists():
                self.status.setText(f"Done! Saved {self.output_path.name}")
                self.open_btn.show()
            elif kind == "live":
                self.status.setText("Camera closed.")
        elif kind == "video" and p.exitStatus() == QProcess.CrashExit:
            self.status.setText("Cancelled.")
            if self.output_path:  # partial file from the cancelled run
                self.output_path.with_name(self.output_path.stem + ".video_only.mp4").unlink(missing_ok=True)
        else:
            reason = next((ln for ln in reversed(self.proc_log)
                           if not ln.startswith(("Applied providers", "find model", "model ignore", "set det"))),
                          "Unknown error")
            self.status.setText("Something went wrong. See the details.")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Reveal Cam")
            box.setText(reason if len(reason) < 300 else reason[:300] + "…")
            box.setInformativeText("TROUBLESHOOTING.md in the Reveal Cam folder covers the common problems.")
            box.setDetailedText("\n".join(self.proc_log))
            box.exec()
        self._update_buttons()

    def open_output(self) -> None:
        if self.output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path)))

    def closeEvent(self, event) -> None:
        p = self.proc
        if p is not None:
            if self.proc_kind == "live":
                self.stop_live()
                p.waitForFinished(6000)
            else:
                p.kill()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Reveal Cam")
    app.setStyleSheet(STYLE)
    w = Window()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
