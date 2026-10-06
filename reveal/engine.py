"""Face detection, swapping and optional enhancement.

Detection: InsightFace `buffalo_l` (SCRFD detector + ArcFace recognition).
Swap: `inswapper_128`. Both are loaded through reveal/insight.py, a port of
the parts of insightface we need (insightface itself needs a compiler to install).
Enhance: GFPGAN v1.4 exported to ONNX (`gfpgan-1024.onnx`, from Deep-Live-Cam).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .insight import FaceAnalysis, INSwapper, SafeSession, norm_crop2

# Standard FFHQ 5-point template at 512x512 (left eye, right eye, nose,
# left mouth corner, right mouth corner) used to align faces for GFPGAN.
FFHQ_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32,
)

SWAPPER_FILES = ("inswapper_128_fp16.onnx", "inswapper_128.onnx")
ENHANCER_FILE = "gfpgan-1024.onnx"


@dataclass
class DetectedFace:
    bbox: np.ndarray  # x1, y1, x2, y2 in frame pixels
    kps: np.ndarray  # 5x2 landmarks in frame pixels
    score: float

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))


def _round32(x: float) -> int:
    return max(32, int(round(x / 32.0)) * 32)


def _soft_square_mask(size: int, margin: float, blur: float) -> np.ndarray:
    """A square mask, shrunk by `margin` px and blurred, as float32 in [0, 1]."""
    m = np.zeros((size, size), np.float32)
    a = int(round(margin))
    m[a:size - a, a:size - a] = 1.0
    if blur > 0:
        k = int(blur) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def _paste_roi(frame: np.ndarray, patch: np.ndarray, mask: np.ndarray, M: np.ndarray) -> None:
    """Warp `patch` (and its soft `mask`) from aligned space back into `frame`
    with the inverse of affine `M`, blending only inside the face's bounding box."""
    h, w = frame.shape[:2]
    ps = patch.shape[0]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [ps, 0, 1], [0, ps, 1], [ps, ps, 1]], np.float32) @ IM.T
    x0 = max(0, int(np.floor(corners[:, 0].min())))
    y0 = max(0, int(np.floor(corners[:, 1].min())))
    x1 = min(w, int(np.ceil(corners[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(corners[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    IMr = IM.copy()
    IMr[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(patch, IMr, size, flags=cv2.INTER_LINEAR, borderValue=0)
    wmask = cv2.warpAffine(mask, IMr, size, flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (warped.astype(np.float32) * wmask + roi * (1.0 - wmask)).astype(np.uint8)


SMALL_FACE_PX = 100  # below this the identity embedding gets noticeably weaker


def load_face_analysis(models_dir: Path, providers: list, min_score: float = 0.5) -> FaceAnalysis:
    """buffalo_l detection + recognition. Downloads into <models_dir>/buffalo_l on first use."""
    return FaceAnalysis(Path(models_dir), providers, det_thresh=min_score)


def read_image(path: Path) -> np.ndarray | None:
    """Read an image (EXIF rotation applied). Handles non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def face_height(face) -> float:
    return float(face.bbox[3] - face.bbox[1])


def face_crop(img: np.ndarray, face, margin: float = 0.25) -> np.ndarray:
    x1, y1, x2, y2 = face.bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = img.shape[:2]
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return img[y1:y2, x1:x2].copy()


def find_source_face(app, img: np.ndarray, min_score: float = 0.5):
    """Find the largest face in a source photo. Works for close-ups and for
    full-body photos where the face is small: detection is retried at higher
    resolution, then with padding (very tight crops can also defeat the detector).
    Returns the face (coordinates in `img`) or None."""
    def largest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None

    det = app.det_model
    original = det.input_size
    longest = max(img.shape[:2])
    try:
        for size in (640, 1024, 1600):
            if size > 640 and longest <= size * 0.8:
                break  # the image is already small; a bigger detector input won't help
            det.input_size = (size, size)
            face = largest(app.get(img))
            if face is not None:
                return face
        det.input_size = (640, 640)
        pad = longest // 4
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        face = largest(app.get(padded))
        if face is not None:
            # Report coordinates in the original photo (the embedding is already computed).
            face.bbox = face.bbox - pad
            face.kps = face.kps - pad
        return face
    finally:
        det.input_size = original

class FaceEngine:
    def __init__(self, models_dir: Path, providers: list, det_width: int = 640,
                 min_score: float = 0.5, enhancer: bool = False):
        self.providers = providers
        self.models_dir = Path(models_dir)
        self.min_score = min_score
        self.det_width = det_width

        self.app = load_face_analysis(self.models_dir, providers, min_score)

        swapper_path = next((self.models_dir / f for f in SWAPPER_FILES if (self.models_dir / f).exists()), None)
        if swapper_path is None:
            raise SystemExit(f"No swapper model found in {self.models_dir}. "
                             f"Run: python scripts/download_models.py")
        self.swapper = INSwapper(swapper_path, providers)
        self.swap_size = int(self.swapper.input_size[0])
        self.swap_mask = _soft_square_mask(self.swap_size, margin=self.swap_size * 0.10,
                                           blur=self.swap_size * 0.05)
        self.latent: np.ndarray | None = None

        self.enhancer = None
        self.enh_mask: np.ndarray | None = None
        if enhancer:
            self.load_enhancer()

    # ---------------------------------------------------------------- source
    def set_source(self, image_path: Path) -> np.ndarray:
        """Compute the source embedding once. Returns the source face crop for display."""
        img = read_image(image_path)
        if img is None:
            raise SystemExit(f"Could not read source image: {image_path} (use JPG or PNG)")
        face = find_source_face(self.app, img, self.min_score)
        if face is None:
            raise SystemExit(f"No face found in source image: {image_path}")
        if face_height(face) < SMALL_FACE_PX:
            print(f"[source] the face is only {face_height(face):.0f}px tall in this photo; "
                  f"a closer photo of the face will give a better likeness.")
        latent = face.normed_embedding.reshape((1, -1)).astype(np.float32)
        latent = latent @ self.swapper.emap
        self.latent = (latent / np.linalg.norm(latent)).astype(np.float32)
        return face_crop(img, face)

    # ------------------------------------------------------------- detection
    def detect(self, frame: np.ndarray, all_faces: bool = False) -> list[DetectedFace]:
        """Detect at reduced resolution (`det_width` px wide); SCRFD scales the
        boxes and landmarks back to frame coordinates."""
        h, w = frame.shape[:2]
        dw = _round32(min(self.det_width, w))
        dh = _round32(dw * h / w)
        bboxes, kpss = self.app.det_model.detect(frame, input_size=(dw, dh), max_num=0, metric="default")
        faces = [DetectedFace(b[:4].astype(np.float32), k.astype(np.float32), float(b[4]))
                 for b, k in zip(bboxes, kpss if kpss is not None else [None] * len(bboxes))
                 if k is not None and b[4] >= self.min_score]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces if all_faces else faces[:1]

    # ------------------------------------------------------------------ swap
    def swap(self, frame: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        out = frame.copy()
        if self.latent is None:
            return out
        s = self.swapper
        for f in faces:
            aimg, M = norm_crop2(frame, f.kps, self.swap_size)
            blob = cv2.dnn.blobFromImage(aimg, 1.0 / s.input_std, (self.swap_size, self.swap_size),
                                         (s.input_mean,) * 3, swapRB=True)
            pred = s.session.run(s.output_names, {s.input_names[0]: blob, s.input_names[1]: self.latent})[0]
            fake = np.clip(255 * pred.transpose(0, 2, 3, 1)[0], 0, 255).astype(np.uint8)[:, :, ::-1]
            _paste_roi(out, np.ascontiguousarray(fake), self.swap_mask, M)
        return out

    # --------------------------------------------------------------- enhance
    def load_enhancer(self) -> bool:
        if self.enhancer is not None:
            return True
        path = self.models_dir / ENHANCER_FILE
        if not path.exists():
            print(f"[enhancer] {path.name} not found in {self.models_dir}; enhancer disabled. "
                  f"Run: python scripts/download_models.py --enhancer")
            return False
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enhancer = SafeSession(path, self.providers, sess_options=opts)
        shape = self.enhancer.get_inputs()[0].shape
        self.enh_size = int(shape[2]) if isinstance(shape[2], int) else 512
        self.enh_mask = _soft_square_mask(self.enh_size, margin=self.enh_size * 0.04,
                                          blur=self.enh_size * 0.04)
        return True

    def enhance(self, frame: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        """Run GFPGAN on each face region only, in place."""
        if self.enhancer is None:
            return frame
        size = self.enh_size
        template = FFHQ_TEMPLATE_512 * (size / 512.0)
        inp = self.enhancer.get_inputs()[0].name
        for f in faces:
            M, _ = cv2.estimateAffinePartial2D(f.kps, template, method=cv2.LMEDS)
            if M is None:
                continue
            aligned = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
            x = aligned[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 127.5 - 1.0
            y = self.enhancer.run(None, {inp: x})[0][0]
            face = ((y + 1.0) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
            if face.shape[0] != size:
                face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
            _paste_roi(frame, np.ascontiguousarray(face), self.enh_mask, M)
        return frame
