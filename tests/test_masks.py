"""Model-free tests for the reveal mask, compositing and config.

Run: python -m pytest tests  (or: python tests/test_masks.py)
"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reveal.config import DEFAULTS, load_config  # noqa: E402
from reveal.engine import DetectedFace  # noqa: E402
from reveal.masks import RevealMask  # noqa: E402

W, H = 640, 360


def make_mask(**overrides):
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in overrides.items():
        node = cfg
        keys = k.split("__")
        for kk in keys[:-1]:
            node = node[kk]
        node[keys[-1]] = v
    return RevealMask(cfg)


def frames():
    real = np.zeros((H, W, 3), np.uint8)
    swapped = np.full((H, W, 3), 255, np.uint8)
    return real, swapped


def test_rect_only_inside_with_feather():
    m = make_mask(mask__mode="rect", mask__feather=20, mask__rect=[0.25, 0.25, 0.75, 0.75])
    real, swapped = frames()
    out, poly = m.composite(real, swapped)
    assert poly is not None
    assert out[H // 2, W // 2, 0] == 255  # center fully swapped
    assert out[5, 5, 0] == 0  # far outside untouched
    edge = out[H // 2, int(W * 0.25), 0]  # on the edge: half blended
    assert 60 < edge < 200, edge
    inside_near = out[H // 2, int(W * 0.25) + 15, 0]
    outside_near = out[H // 2, int(W * 0.25) - 15, 0]
    assert inside_near > edge > outside_near


def test_full_mode_covers_frame():
    m = make_mask(mask__mode="full", mask__feather=0)
    real, swapped = frames()
    out, _ = m.composite(real, swapped)
    assert out.min() == 255


def test_quad_mode_tilted():
    quad = [[0.2, 0.1], [0.8, 0.3], [0.7, 0.9], [0.1, 0.7]]
    m = make_mask(mask__mode="quad", mask__feather=0, mask__quad=quad)
    real, swapped = frames()
    out, _ = m.composite(real, swapped)
    assert out[int(0.5 * H), int(0.45 * W), 0] == 255
    assert out[int(0.05 * H), int(0.75 * W), 0] == 0  # above the slanted top edge


def test_face_follow_tracks_and_holds():
    m = make_mask(mask__mode="face", mask__feather=0, mask__face={"padding": 1.0, "smoothing": 0.0,
                                                                  "hold_frames": 2})
    face = DetectedFace(np.array([100, 100, 200, 220], np.float32), np.zeros((5, 2), np.float32), 0.9)
    assert m.base_polygon(W, H) is None  # nothing until a face is seen
    m.update_faces([face], W, H)
    poly = m.base_polygon(W, H)
    assert np.allclose(poly.min(axis=0), [100, 100]) and np.allclose(poly.max(axis=0), [200, 220])
    m.update_faces([], W, H)
    m.update_faces([], W, H)
    assert m.base_polygon(W, H) is not None  # held
    m.update_faces([], W, H)
    assert m.base_polygon(W, H) is None  # dropped after hold_frames


def test_face_follow_smoothing():
    m = make_mask(mask__mode="face", mask__face={"padding": 1.0, "smoothing": 0.5, "hold_frames": 5})
    a = DetectedFace(np.array([0, 0, 100, 100], np.float32), np.zeros((5, 2), np.float32), 0.9)
    b = DetectedFace(np.array([100, 0, 200, 100], np.float32), np.zeros((5, 2), np.float32), 0.9)
    m.update_faces([a], W, H)
    m.update_faces([b], W, H)
    cx = m.base_polygon(W, H)[:, 0].mean()
    assert 90 < cx < 110  # halfway between 50 and 150


def _anim_coverage(style, direction="right", steps=10):
    m = make_mask(mask__mode="rect", mask__feather=10, animation__style=style,
                  animation__direction=direction, animation__frames=steps, animation__start_open=False)
    real, swapped = frames()
    cov = []
    out, _ = m.composite(real, swapped)
    cov.append(out[..., 0].mean())
    m.toggle()
    for _ in range(steps):
        m.step()
        out, _ = m.composite(real, swapped)
        cov.append(out[..., 0].mean())
    return m, cov


def test_animations_open_monotonic():
    for style in ("wipe", "slide", "scale", "fade"):
        for d in ("left", "right", "up", "down"):
            m, cov = _anim_coverage(style, d)
            assert cov[0] == 0, (style, d)
            assert m.openness == 1.0
            full = make_mask(mask__mode="rect", mask__feather=10)
            ref, _ = full.composite(*frames())
            assert abs(cov[-1] - ref[..., 0].mean()) < 0.5, (style, d)
            if style in ("wipe", "scale", "fade"):
                assert all(b >= a - 1e-6 for a, b in zip(cov, cov[1:])), (style, d, cov)


def test_wipe_direction():
    m = make_mask(mask__mode="full", mask__feather=0, animation__style="wipe",
                  animation__direction="right", animation__frames=2, animation__start_open=False)
    m.toggle()
    m.step()  # halfway
    out, _ = m.composite(*frames())
    assert out[H // 2, 10, 0] == 255 and out[H // 2, W - 10, 0] == 0


def test_close_animation():
    m = make_mask(animation__frames=4)
    assert m.is_open
    m.toggle()
    for _ in range(4):
        m.step()
    assert m.openness == 0.0
    assert m.render(W, H) is None


def test_mouse_drag_rect_and_quad():
    m = make_mask(mask__mode="rect", mask__rect=[0.25, 0.25, 0.75, 0.75])
    # drag bottom-right corner
    m.on_mouse(cv2.EVENT_LBUTTONDOWN, int(0.75 * W), int(0.75 * H), W, H)
    m.on_mouse(cv2.EVENT_MOUSEMOVE, int(0.9 * W), int(0.9 * H), W, H)
    m.on_mouse(cv2.EVENT_LBUTTONUP, int(0.9 * W), int(0.9 * H), W, H)
    assert abs(m.rect[2] - 0.9) < 0.01 and abs(m.rect[3] - 0.9) < 0.01
    # move the box
    m.on_mouse(cv2.EVENT_LBUTTONDOWN, int(0.5 * W), int(0.5 * H), W, H)
    m.on_mouse(cv2.EVENT_MOUSEMOVE, int(0.45 * W), int(0.5 * H), W, H)
    m.on_mouse(cv2.EVENT_LBUTTONUP, int(0.45 * W), int(0.5 * H), W, H)
    assert abs(m.rect[0] - 0.20) < 0.01
    m.reset_geometry()
    assert m.rect == [0.25, 0.25, 0.75, 0.75]

    m.set_mode("quad")
    x, y = m.quad[0]
    m.on_mouse(cv2.EVENT_LBUTTONDOWN, int(x * W), int(y * H), W, H)
    m.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 40, W, H)
    m.on_mouse(cv2.EVENT_LBUTTONUP, 50, 40, W, H)
    assert abs(m.quad[0][0] - 50 / W) < 1e-6 and abs(m.quad[0][1] - 40 / H) < 1e-6


def test_config_overrides():
    cfg, _ = load_config(["--mask", "quad", "--feather", "7", "--skip-n", "3",
                          "--set", "animation.keyframes=[{t: 2.0, action: open}, {t: 5.5, action: close}]",
                          "--set", "mask.border.enabled=false"])
    assert cfg["mask"]["mode"] == "quad"
    assert cfg["mask"]["feather"] == 7
    assert cfg["detection"]["skip_n"] == 3
    assert cfg["mask"]["border"]["enabled"] is False
    assert cfg["animation"]["keyframes"][1] == {"t": 5.5, "action": "close"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
