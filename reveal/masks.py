"""The reveal window: mask shapes, feathering, borders and open/close animation.

Compositing rule:  output = real * (1 - mask) + swapped * mask
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import MASK_MODES

HANDLE_RADIUS = 18  # px; how close a click must be to grab a corner


def _clip_half_plane(poly: np.ndarray, axis: int, limit: float, keep_less: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon against x/y <= limit (or >= limit)."""
    def inside(p):
        return p[axis] <= limit if keep_less else p[axis] >= limit

    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cin, nin = inside(cur), inside(nxt)
        if cin:
            out.append(cur)
        if cin != nin:
            t = (limit - cur[axis]) / (nxt[axis] - cur[axis])
            out.append(cur + t * (nxt - cur))
    return np.array(out, np.float32).reshape(-1, 2)


def _point_in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class RevealMask:
    def __init__(self, cfg: dict):
        m = cfg["mask"]
        a = cfg["animation"]
        self.mode: str = m["mode"]
        self.feather: int = int(m["feather"])
        self.border_on: bool = bool(m["border"]["enabled"])
        self.border_thickness: int = int(m["border"]["thickness"])
        self.border_color = tuple(int(c) for c in m["border"]["color"])
        self._rect0 = list(map(float, m["rect"]))
        self._quad0 = [list(map(float, p)) for p in m["quad"]]
        self.rect = list(self._rect0)  # normalized x1, y1, x2, y2
        self.quad = [list(p) for p in self._quad0]  # normalized 4 x [x, y]
        self.face_padding = float(m["face"]["padding"])
        self.face_smoothing = float(m["face"]["smoothing"])
        self.face_hold = int(m["face"]["hold_frames"])
        self._face_box: np.ndarray | None = None  # smoothed cx, cy, w, h in pixels
        self._face_missing = 0

        self.style: str = a["style"]
        self.direction: str = a["direction"]
        self.anim_frames: int = int(a["frames"])
        self.openness: float = 1.0 if a["start_open"] else 0.0
        self.target: float = self.openness

        self._drag: tuple | None = None
        self._cache_key = None
        self._cache_val = None

    # ----------------------------------------------------------- mode/state
    def cycle_mode(self) -> None:
        self.set_mode(MASK_MODES[(MASK_MODES.index(self.mode) + 1) % len(MASK_MODES)])

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._drag = None

    def reset_geometry(self) -> None:
        self.rect = list(self._rect0)
        self.quad = [list(p) for p in self._quad0]

    def toggle(self) -> None:
        self.target = 0.0 if self.target > 0.5 else 1.0

    def set_open(self, is_open: bool, instant: bool = False) -> None:
        self.target = 1.0 if is_open else 0.0
        if instant:
            self.openness = self.target

    def step(self) -> None:
        """Advance the open/close animation by one frame."""
        d = 1.0 / self.anim_frames
        if self.openness < self.target:
            self.openness = min(self.target, self.openness + d)
        elif self.openness > self.target:
            self.openness = max(self.target, self.openness - d)
        if abs(self.openness - self.target) < 1e-6:
            self.openness = self.target

    @property
    def is_open(self) -> bool:
        return self.target > 0.5

    # ------------------------------------------------------------ face-follow
    def update_faces(self, faces: list, w: int, h: int) -> None:
        if not faces:
            self._face_missing += 1
            if self._face_missing > self.face_hold:
                self._face_box = None
            return
        self._face_missing = 0
        x1, y1, x2, y2 = faces[0].bbox
        box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.float32)
        if self._face_box is None:
            self._face_box = box
        else:
            s = min(0.95, max(0.0, self.face_smoothing))
            self._face_box = s * self._face_box + (1 - s) * box

    # ----------------------------------------------------------------- shapes
    def base_polygon(self, w: int, h: int) -> np.ndarray | None:
        if self.mode == "full":
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        if self.mode == "rect":
            x1, y1, x2, y2 = self.rect
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            return np.array([[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]], np.float32)
        if self.mode == "quad":
            return np.array([[x * w, y * h] for x, y in self.quad], np.float32)
        if self.mode == "face":
            if self._face_box is None:
                return None
            cx, cy, bw, bh = self._face_box
            side_w, side_h = bw * self.face_padding / 2, bh * self.face_padding / 2
            return np.array([[cx - side_w, cy - side_h], [cx + side_w, cy - side_h],
                             [cx + side_w, cy + side_h], [cx - side_w, cy + side_h]], np.float32)
        return None

    def animated_polygon(self, w: int, h: int, amount: float | None = None) -> np.ndarray | None:
        poly = self.base_polygon(w, h)
        if poly is None:
            return None
        a = ease(self.openness if amount is None else amount)
        if self.style == "fade" or a >= 1.0:
            return poly
        if a <= 0.0:
            return None
        if self.style == "scale":
            c = poly.mean(axis=0)
            return c + (poly - c) * a
        if self.style == "slide":
            d = self.direction
            if d == "right":
                return poly - [(1 - a) * poly[:, 0].max(), 0]
            if d == "left":
                return poly + [(1 - a) * (w - poly[:, 0].min()), 0]
            if d == "down":
                return poly - [0, (1 - a) * poly[:, 1].max()]
            return poly + [0, (1 - a) * (h - poly[:, 1].min())]
        # wipe: clip the shape with a moving edge; the feather margin makes the
        # edge start fully outside the shape and end fully past it.
        f = self.feather
        axis = 0 if self.direction in ("left", "right") else 1
        lo, hi = poly[:, axis].min() - f, poly[:, axis].max() + f
        if self.direction in ("right", "down"):
            clipped = _clip_half_plane(poly, axis, lo + a * (hi - lo), keep_less=True)
        else:
            clipped = _clip_half_plane(poly, axis, hi - a * (hi - lo), keep_less=False)
        return clipped if len(clipped) >= 3 else None

    # ----------------------------------------------------------------- render
    def render(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray] | None:
        """Return (mask_roi float32 HxWx1 in [0,1], (x0, y0, x1, y1), polygon) or None if closed."""
        poly = self.animated_polygon(w, h)
        if poly is None:
            return None
        alpha = ease(self.openness) if self.style == "fade" else 1.0
        if alpha <= 0.0:
            return None
        key = (poly.round(1).tobytes(), round(alpha, 3), self.feather, w, h)
        if key == self._cache_key:
            return self._cache_val

        f = self.feather
        x0 = max(0, int(math.floor(poly[:, 0].min())) - f - 1)
        y0 = max(0, int(math.floor(poly[:, 1].min())) - f - 1)
        x1 = min(w, int(math.ceil(poly[:, 0].max())) + f + 1)
        y1 = min(h, int(math.ceil(poly[:, 1].max())) + f + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        hard = np.zeros((y1 - y0, x1 - x0), np.float32)
        pts = np.round((poly - [x0, y0]) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(hard, [pts], 1.0, lineType=cv2.LINE_AA, shift=4)
        if f > 0:
            sigma = f / 4.0
            k = 2 * int(math.ceil(3 * sigma)) + 1
            hard = cv2.GaussianBlur(hard, (k, k), sigma, borderType=cv2.BORDER_CONSTANT)
        if alpha < 1.0:
            hard *= alpha
        val = (hard[..., None], (x0, y0, x1, y1), poly)
        self._cache_key, self._cache_val = key, val
        return val

    def composite(self, real: np.ndarray, swapped: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Blend `swapped` over `real` inside the reveal window."""
        h, w = real.shape[:2]
        r = self.render(w, h)
        if r is None:
            return real.copy(), None
        mask, (x0, y0, x1, y1), poly = r
        out = real.copy()
        a = real[y0:y1, x0:x1].astype(np.float32)
        b = swapped[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (a + (b - a) * mask).astype(np.uint8)
        return out, poly

    def draw_border(self, frame: np.ndarray, poly: np.ndarray | None) -> None:
        if not self.border_on or poly is None or self.mode == "full" or self.border_thickness <= 0:
            return
        pts = np.round(poly * 16).astype(np.int32).reshape(-1, 1, 2)
        if self.style == "fade" and self.openness < 1.0:
            overlay = frame.copy()
            cv2.polylines(overlay, [pts], True, self.border_color, self.border_thickness, cv2.LINE_AA, shift=4)
            a = ease(self.openness)
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, dst=frame)
        else:
            cv2.polylines(frame, [pts], True, self.border_color, self.border_thickness, cv2.LINE_AA, shift=4)

    # ------------------------------------------------------------------ mouse
    def on_mouse(self, event: int, x: int, y: int, w: int, h: int) -> None:
        nx, ny = x / w, y / h
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = self._hit_test(x, y, w, h, nx, ny)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self._apply_drag(nx, ny)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "rect":
                x1, y1, x2, y2 = self.rect
                self.rect = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            self._drag = None

    def _hit_test(self, x, y, w, h, nx, ny):
        if self.mode == "quad":
            pts = np.array(self.quad) * [w, h]
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            i = int(d.argmin())
            if d[i] <= HANDLE_RADIUS * 1.5:
                return ("quad_corner", i)
            if _point_in_poly(pts, x, y):
                return ("quad_move", nx, ny, [list(p) for p in self.quad])
            return None
        if self.mode == "rect":
            x1, y1, x2, y2 = self.rect
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for i, (cx, cy) in enumerate(corners):
                if math.hypot(cx * w - x, cy * h - y) <= HANDLE_RADIUS:
                    return ("rect_corner", i)
            if x1 * w <= x <= x2 * w and y1 * h <= y <= y2 * h:
                return ("rect_move", nx, ny, list(self.rect))
            # Click outside: draw a new box from here.
            self.rect = [nx, ny, nx, ny]
            return ("rect_corner", 2)
        return None

    def _apply_drag(self, nx: float, ny: float) -> None:
        nx, ny = min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))
        kind = self._drag[0]
        if kind == "quad_corner":
            self.quad[self._drag[1]] = [nx, ny]
        elif kind == "quad_move":
            _, sx, sy, orig = self._drag
            self.quad = [[px + nx - sx, py + ny - sy] for px, py in orig]
        elif kind == "rect_corner":
            i = self._drag[1]
            x1, y1, x2, y2 = self.rect
            if i == 0:
                x1, y1 = nx, ny
            elif i == 1:
                x2, y1 = nx, ny
            elif i == 2:
                x2, y2 = nx, ny
            else:
                x1, y2 = nx, ny
            self.rect = [x1, y1, x2, y2]
        elif kind == "rect_move":
            _, sx, sy, (x1, y1, x2, y2) = self._drag
            dx = min(max(nx - sx, -x1), 1 - x2)
            dy = min(max(ny - sy, -y1), 1 - y2)
            self.rect = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def draw_handles(self, frame: np.ndarray) -> None:
        """Corner handles for the editable shapes (preview only)."""
        h, w = frame.shape[:2]
        if self.openness < 1.0:
            return
        if self.mode == "quad":
            pts = [(x * w, y * h) for x, y in self.quad]
        elif self.mode == "rect":
            x1, y1, x2, y2 = self.rect
            pts = [(x1 * w, y1 * h), (x2 * w, y1 * h), (x2 * w, y2 * h), (x1 * w, y2 * h)]
        else:
            return
        for px, py in pts:
            cv2.circle(frame, (int(px), int(py)), 6, (0, 200, 255), 2, cv2.LINE_AA)
