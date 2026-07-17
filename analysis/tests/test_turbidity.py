import numpy as np
import pandas as pd
import pytest

from analysis import geometry as g
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


def test_fit_beta_r2_drops_with_real_scatter():
    """r2 must actually measure fit quality, not just report a number close to 1.

    The two fixtures above put points almost exactly on the true curve, so a stubbed
    _r2 that always returns 1.0 would pass them unnoticed. Here the contrast carries
    real multiplicative noise around the true decay, so a genuine r2 lands well below
    1 and well above 0. The band is wide enough to tolerate a different but reasonable
    RNG draw, and tight enough that a hardcoded 1.0 (or a near-1.0 stub) fails it.
    """
    rng = np.random.default_rng(42)
    d = np.linspace(0.5, 4.0, 40)
    contrast = 120.0 * np.exp(-0.4 * d) * np.exp(rng.normal(0.0, 0.3, size=d.size))
    _, _, r2 = T.fit_beta(d, contrast)
    assert 0.5 < r2 < 0.95, r2


def test_fit_beta_r2_is_near_zero_for_unrelated_data():
    """Pins the low end: contrast with no exponential relationship to range at all
    must not report a high r2. This is the case a stubbed _r2 = 1.0 gets most wrong."""
    rng = np.random.default_rng(1)
    d = np.linspace(0.5, 4.0, 40)
    contrast = rng.uniform(50.0, 150.0, size=d.size)
    _, _, r2 = T.fit_beta(d, contrast)
    assert r2 < 0.2, r2


def test_fit_beta_r2_matches_the_log_space_definition():
    """fit_beta reports r2 in log space (see its docstring), computed on log(contrast)
    against the fitted line, not on contrast itself in linear space. With noise-free
    fixtures the two spaces are indistinguishable because every point sits on the
    curve; here real scatter makes them diverge, so this test can actually tell them
    apart, and pins the function to the log-space definition specifically.
    """
    rng = np.random.default_rng(3)
    d = np.linspace(0.5, 4.0, 40)
    contrast = 120.0 * np.exp(-0.4 * d) * np.exp(rng.normal(0.0, 0.3, size=d.size))
    beta, c0, r2 = T.fit_beta(d, contrast)

    y = np.log(contrast)
    yhat_log = -beta * d + np.log(c0)
    ss_res = np.sum((y - yhat_log) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    expected_log_r2 = 1.0 - ss_res / ss_tot

    c_hat = c0 * np.exp(-beta * d)
    ss_res_lin = np.sum((contrast - c_hat) ** 2)
    ss_tot_lin = np.sum((contrast - np.mean(contrast)) ** 2)
    expected_lin_r2 = 1.0 - ss_res_lin / ss_tot_lin

    # The two spaces must disagree enough here for the comparison below to mean
    # anything; if they converged this assertion catches the fixture, not the code.
    assert abs(expected_log_r2 - expected_lin_r2) > 0.1

    assert r2 == pytest.approx(expected_log_r2, abs=1e-9)


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


def _K_mat(K):
    return np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])


def test_plane_depth_is_the_range_at_the_principal_point_when_fronto_parallel(
    K, layout_true
):
    """Fronto-parallel at 2 m: the centre pixel's ray is the optical axis, so its
    Euclidean range to the plane is exactly 2 m."""
    rv = np.zeros(3)
    tv = np.array([0.0, 0.0, 2.0])
    dm = T.plane_depth_map(layout_true, rv, tv, _K_mat(K), (540, 960))
    # round, not int: cx = 483.78, so the pixel nearest the optical axis is 484.
    # Truncating to 483 lands 0.78 px off-axis, which is 1.1e-6 of extra range and
    # fails this tolerance.
    assert dm[round(K["cy"]), round(K["cx"])] == pytest.approx(2.0, abs=1e-6)


def test_plane_depth_grows_away_from_the_principal_point(K, layout_true):
    """Off-axis rays travel further to reach the same fronto-parallel plane."""
    dm = T.plane_depth_map(layout_true, np.zeros(3), np.array([0.0, 0.0, 2.0]),
                           _K_mat(K), (540, 960))
    centre = dm[round(K["cy"]), round(K["cx"])]
    assert dm[round(K["cy"]), 10] > centre
    assert np.all(dm[np.isfinite(dm)] >= centre - 1e-9)


def test_plane_depth_is_nan_for_a_plane_behind_the_camera(K, layout_true):
    dm = T.plane_depth_map(layout_true, np.zeros(3), np.array([0.0, 0.0, -2.0]),
                           _K_mat(K), (540, 960))
    assert np.all(np.isnan(dm))


def test_board_mask_covers_the_markers_and_not_the_whole_frame(K, layout_true):
    rv = np.zeros(3)
    tv = np.array([-0.13, 0.05, 1.2])
    Km = _K_mat(K)
    mask = T.board_mask(layout_true, rv, tv, Km, (540, 960))
    assert mask.any() and not mask.all()
    # Every projected marker corner must fall inside the mask.
    for mid in g.IDS:
        px = g.project(g.board_pts(layout_true, np.array([g.IDX[mid]])),
                       rv[None], tv[None], np.zeros(1, dtype=int),
                       K["fx"], K["fy"], K["cx"], K["cy"])[0]
        for u, v in px:
            if 0 <= int(v) < 540 and 0 <= int(u) < 960:
                assert mask[int(v), int(u)], f"marker {mid} corner outside mask"


def test_marker_ranges_are_all_near_the_board_range(layout_true):
    rv = np.zeros(3)
    tv = np.array([0.0, 0.0, 2.0])
    r = T.marker_ranges(layout_true, rv, tv)
    assert set(r) == set(g.IDS)
    assert all(1.9 < v < 2.2 for v in r.values()), r
