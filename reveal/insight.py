"""The small part of InsightFace this project uses, without the `insightface`
package itself (it ships as source only and needs a C++ compiler to install).

Ported from insightface 0.7.3 (https://github.com/deepinsight/insightface,
MIT License, Copyright (c) 2018 Jiankang Deng and Jia Guo):
SCRFD detection, ArcFace recognition, 5-point face alignment and the
inswapper model wrapper. The similarity-transform fit replaces the
scikit-image dependency with the same Umeyama algorithm in numpy.

The pretrained models themselves are licensed for non-commercial research
use only.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_FILES = {"det_10g.onnx": "detection", "w600k_r50.onnx": "recognition"}

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def urlopen(url: str):
    """urllib using the system certificates, falling back to certifi's bundle
    when they can't verify the server (python.org builds on macOS ship with no
    system certificates configured). Corporate proxies are usually only trusted
    by the system store, which is why that is tried first."""
    import ssl

    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise e from None
        return urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))


# ------------------------------------------------------------------ alignment
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation)
    mapping src -> dst. Same result as skimage SimilarityTransform.estimate."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    num, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    A = dst_d.T @ src_d / num
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_d.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio, diff_x = image_size / 112.0, 0.0
    else:
        ratio = image_size / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    return umeyama(lmk, dst)[0:2, :]


def norm_crop2(img: np.ndarray, landmark: np.ndarray, image_size: int = 112):
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0), M


# ---------------------------------------------------------------------- types
class Face:
    def __init__(self, bbox, kps, det_score):
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score
        self.embedding: np.ndarray | None = None

    @property
    def normed_embedding(self) -> np.ndarray | None:
        if self.embedding is None:
            return None
        return self.embedding / np.linalg.norm(self.embedding)


def _provider_name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


def fallback_chain(providers: list) -> list[list]:
    """Provider setups to try in order. Hardware acceleration can refuse a
    particular model (CoreML in particular rejects some graphs, sometimes only
    once it runs), so every accelerated setup steps down to plain CPU."""
    first = _provider_name(providers[0])
    if first == "CPUExecutionProvider":
        return [["CPUExecutionProvider"]]
    if first == "CoreMLExecutionProvider":
        return [
            [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
             "CPUExecutionProvider"],
            [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}),
             "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]
    return [list(providers), ["CPUExecutionProvider"]]


def _describe(setup: list) -> str:
    p = setup[0]
    if isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider":
        return "CoreML (GPU only)" if p[1].get("MLComputeUnits") == "CPUAndGPU" else "CoreML"
    return _provider_name(p).replace("ExecutionProvider", "")


def _short(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg if len(msg) < 160 else msg[:160] + "…"


class SafeSession:
    """An onnxruntime session that steps down to the next provider setup in
    `fallback_chain` if loading or running the model fails, instead of crashing."""

    def __init__(self, path: Path, providers: list, sess_options=None):
        self.path = Path(path)
        self.chain = fallback_chain(providers)
        self.sess_options = sess_options
        self.level = -1
        self._open(0)

    def _open(self, start: int) -> None:
        for i in range(start, len(self.chain)):
            try:
                self.session = onnxruntime.InferenceSession(str(self.path), sess_options=self.sess_options,
                                                            providers=self.chain[i])
                self.level = i
                return
            except Exception as e:
                if i == len(self.chain) - 1:
                    raise
                print(f"[provider] {self.path.name}: {_describe(self.chain[i])} could not load it "
                      f"({_short(e)}). Trying {_describe(self.chain[i + 1])}.", flush=True)

    def run(self, output_names, feed):
        while True:
            try:
                return self.session.run(output_names, feed)
            except Exception as e:
                accelerated = _provider_name(self.chain[self.level][0]) != "CPUExecutionProvider"
                if not accelerated or self.level >= len(self.chain) - 1:
                    raise
                print(f"[provider] {self.path.name}: {_describe(self.chain[self.level])} failed while running "
                      f"({_short(e)}). Switching to {_describe(self.chain[self.level + 1])}.", flush=True)
                self._open(self.level + 1)

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def get_providers(self):
        return self.session.get_providers()


def _session(path: Path, providers: list) -> SafeSession:
    return SafeSession(path, providers)


# ------------------------------------------------------------------ detection
class SCRFD:
    def __init__(self, model_file: Path, providers: list):
        self.session = _session(model_file, providers)
        self.center_cache: dict = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = None if isinstance(inp.shape[2], str) else tuple(inp.shape[2:4][::-1])
        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.output_names = [o.name for o in outputs]
        self.input_mean, self.input_std = 127.5, 128.0
        n = len(outputs)
        self.use_kps = n in (9, 15)
        self.fmc = 3 if n in (6, 9) else 5
        self._feat_stride_fpn = [8, 16, 32] if self.fmc == 3 else [8, 16, 32, 64, 128]
        self._num_anchors = 2 if self.fmc == 3 else 1

    def forward(self, img: np.ndarray, threshold: float):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                     (self.input_mean,) * 3, swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height, input_width = blob.shape[2], blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                kps_preds = net_outs[idx + fmc * 2][0] * stride if self.use_kps else None
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                kps_preds = net_outs[idx + fmc * 2] * stride if self.use_kps else None
            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)
            centers = self.center_cache.get(key)
            if centers is None:
                centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    centers = np.stack([centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = centers
            pos = np.where(scores >= threshold)[0]
            x1 = centers[:, 0] - bbox_preds[:, 0]
            y1 = centers[:, 1] - bbox_preds[:, 1]
            x2 = centers[:, 0] + bbox_preds[:, 2]
            y2 = centers[:, 1] + bbox_preds[:, 3]
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)
            scores_list.append(scores[pos])
            bboxes_list.append(bboxes[pos])
            if self.use_kps:
                kps = np.empty((kps_preds.shape[0], kps_preds.shape[1] // 2, 2), np.float32)
                kps[:, :, 0] = centers[:, 0:1] + kps_preds[:, 0::2]
                kps[:, :, 1] = centers[:, 1:2] + kps_preds[:, 1::2]
                kpss_list.append(kps[pos])
        return scores_list, bboxes_list, kpss_list

    def detect(self, img: np.ndarray, input_size=None, max_num: int = 0, metric: str = "default"):
        input_size = self.input_size if input_size is None else input_size
        im_ratio = img.shape[0] / img.shape[1]
        model_ratio = input_size[1] / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = new_height / img.shape[0]
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width] = cv2.resize(img, (new_width, new_height))

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order]
        keep = self.nms(pre_det)
        det = pre_det[keep]
        kpss = None
        if self.use_kps:
            kpss = (np.vstack(kpss_list) / det_scale)[order][keep]
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            off = np.vstack([(det[:, 0] + det[:, 2]) / 2 - cx, (det[:, 1] + det[:, 3]) / 2 - cy])
            values = area if metric == "max" else area - np.sum(off ** 2, 0) * 2.0
            idx = np.argsort(values)[::-1][:max_num]
            det = det[idx]
            if kpss is not None:
                kpss = kpss[idx]
        return det, kpss

    def nms(self, dets: np.ndarray) -> list[int]:
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        return keep


# ---------------------------------------------------------------- recognition
class ArcFace:
    def __init__(self, model_file: Path, providers: list):
        import onnx

        graph = onnx.load(str(model_file)).graph
        names = [n.name for n in graph.node[:8]]
        has_sub = any(n.startswith(("Sub", "_minus")) for n in names)
        has_mul = any(n.startswith(("Mul", "_mul")) for n in names)
        self.input_mean, self.input_std = (0.0, 1.0) if has_sub and has_mul else (127.5, 127.5)
        self.session = _session(model_file, providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = tuple(inp.shape[2:4][::-1])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def get(self, img: np.ndarray, face: Face) -> np.ndarray:
        aimg, _ = norm_crop2(img, face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / self.input_std, self.input_size,
                                      (self.input_mean,) * 3, swapRB=True)
        face.embedding = self.session.run(self.output_names, {self.input_name: blob})[0].flatten()
        return face.embedding


# ------------------------------------------------------------------- analysis
def ensure_buffalo(models_dir: Path) -> Path:
    """Make sure the two buffalo_l models we use are in models/buffalo_l/,
    downloading InsightFace's buffalo_l.zip once if needed."""
    target = Path(models_dir) / "buffalo_l"
    if all((target / f).exists() for f in BUFFALO_FILES):
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[models] downloading face detection models (buffalo_l, 280 MB) from {BUFFALO_URL}")
    with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
        zpath = Path(tmp) / "buffalo_l.zip"
        with urlopen(BUFFALO_URL) as r, open(zpath, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r         {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in BUFFALO_FILES:
                    with z.open(member) as src, open(target / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    missing = [f for f in BUFFALO_FILES if not (target / f).exists()]
    if missing:
        raise SystemExit(f"buffalo_l.zip did not contain {missing}")
    return target


class FaceAnalysis:
    """Detection + identity embedding (the parts of insightface's FaceAnalysis we use)."""

    def __init__(self, models_dir: Path, providers: list, det_thresh: float = 0.5):
        folder = ensure_buffalo(models_dir)
        self.det_model = SCRFD(folder / "det_10g.onnx", providers)
        self.det_model.det_thresh = det_thresh
        self.det_model.input_size = (640, 640)
        self.rec_model = ArcFace(folder / "w600k_r50.onnx", providers)

    def get(self, img: np.ndarray) -> list[Face]:
        bboxes, kpss = self.det_model.detect(img, max_num=0, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bboxes[i, 0:4], kpss[i] if kpss is not None else None, bboxes[i, 4])
            if face.kps is not None:
                self.rec_model.get(img, face)
            faces.append(face)
        return faces


# ---------------------------------------------------------------------- swap
class INSwapper:
    def __init__(self, model_file: Path, providers: list):
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(str(model_file)).graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean, self.input_std = 0.0, 255.0
        self.session = _session(model_file, providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = tuple(self.session.get_inputs()[0].shape[2:4][::-1])
