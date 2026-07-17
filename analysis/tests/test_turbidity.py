import numpy as np
import pandas as pd
import pytest

from analysis import turbidity as T


def _synthetic_marker(canon_px=96, surround=1.6, black=30.0, white=200.0):
    """A fronto-parallel 7x7 marker on a white sheet, rendered directly in canonical space.

    Returns (img, corners) where corners are the marker's outer quad in img pixels.
    """
    side = int(round(canon_px * surround))
    img = np.full((side, side, 3), white, dtype=np.uint8)
    off = (side - canon_px) // 2
    cell = canon_px / 7.0
    # Black border ring: the outer cell of the 7x7 grid.
    img[off:off + canon_px, off:off + canon_px] = black
    inner = int(round(cell))
    # Data area: white, so only the border ring is black.
    img[off + inner:off + canon_px - inner, off + inner:off + canon_px - inner] = white
    corners = np.array([[off, off], [off + canon_px, off],
                        [off + canon_px, off + canon_px], [off, off + canon_px]],
                       dtype=float)
    return img, corners


def test_patch_means_separate_the_black_ring_from_the_white_sheet():
    img, corners = _synthetic_marker(black=30.0, white=200.0)
    warped = T.warp_marker(img, corners)
    black, white = T.patch_means(warped)
    assert black.shape == (4,) and white.shape == (4,)
    assert np.allclose(black, 30.0, atol=2.0), black
    assert np.allclose(white, 200.0, atol=2.0), white


def test_warp_marker_is_size_invariant():
    """A marker at 2x the apparent size must yield the same patch means."""
    small, cs = _synthetic_marker(canon_px=48)
    big, cb = _synthetic_marker(canon_px=192)
    bs, ws = T.patch_means(T.warp_marker(small, cs))
    bb, wb = T.patch_means(T.warp_marker(big, cb))
    assert np.allclose(bs, bb, atol=2.0)
    assert np.allclose(ws, wb, atol=2.0)


def test_fit_beta_recovers_a_known_decay():
    d = np.linspace(0.5, 4.0, 60)
    beta, c0, r2 = T.fit_beta(d, 120.0 * np.exp(-0.42 * d))
    assert beta == pytest.approx(0.42, abs=1e-6)
    assert c0 == pytest.approx(120.0, rel=1e-6)
    assert r2 > 0.999


def test_fit_beta_ignores_nonpositive_contrast():
    """At extreme range the measured contrast can go negative through noise. log() of it
    is nan, and one nan would otherwise poison the whole fit."""
    d = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    c = np.array([100.0, 78.7, 62.0, 48.8, -3.0])
    beta, _, r2 = T.fit_beta(d, c)
    assert beta == pytest.approx(0.48, abs=0.02)
    assert np.isfinite(r2)


def test_fit_veiling_recovers_a_known_B():
    d = np.linspace(0.5, 4.0, 40)
    B_true, beta = 150.0, 0.33
    black = B_true * (1 - np.exp(-beta * d))
    B, r2 = T.fit_veiling(d, black, beta)
    assert B == pytest.approx(B_true, rel=1e-6)
    assert r2 > 0.999


def test_measure_beta_is_per_channel_and_red_dies_fastest():
    d = np.linspace(0.6, 3.0, 50)
    truth = {"b": 0.33, "g": 0.29, "r": 0.53, "grey": 0.31}
    rows = {"range_m": d}
    for ch, bt in truth.items():
        rows[f"white_{ch}"] = 200.0 * np.exp(-bt * d) + 40.0
        rows[f"black_{ch}"] = 40.0
    out = T.measure_beta(pd.DataFrame(rows)).set_index("channel")
    for ch, bt in truth.items():
        assert out.loc[ch, "beta"] == pytest.approx(bt, abs=1e-6)
        assert out.loc[ch, "n"] == 50
    assert out.loc["r", "beta"] > out.loc["g", "beta"]
