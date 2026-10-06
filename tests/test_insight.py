"""Model-free tests for the ported InsightFace alignment math."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reveal.insight import ARCFACE_DST, estimate_norm, umeyama  # noqa: E402


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.random((5, 2)) * 200
    ang, scale, t = 0.3, 1.7, np.array([12.0, -5.0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (scale * (R @ src.T)).T + t
    T = umeyama(src, dst)
    assert np.allclose(T[:2, :2], scale * R, atol=1e-9)
    assert np.allclose(T[:2, 2], t, atol=1e-9)


def test_estimate_norm_maps_template_to_itself():
    M = estimate_norm(ARCFACE_DST.copy(), 112)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-5)
    # inswapper's 128px crop: same scale as the 112 template, shifted 8px right (as in insightface)
    M128 = estimate_norm(ARCFACE_DST.copy(), 128)
    assert np.allclose(M128, [[1, 0, 8], [0, 1, 0]], atol=1e-5)


if __name__ == "__main__":
    test_umeyama_recovers_similarity()
    test_estimate_norm_maps_template_to_itself()
    print("ok")
