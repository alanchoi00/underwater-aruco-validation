import cv2
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


def test_plane_depth_matches_an_independent_reference_at_an_oblique_pose(K, layout_true):
    """Pins the rotation convention against an outside authority, not against itself.

    All the fronto-parallel tests above use rv = 0, the one pose where a transposed
    R, a swapped rotation convention, or an R @ x vs x @ R.T mix-up cannot show up:
    with no rotation, R = R.T = I. Here the pose is genuinely tilted (pitch of 0.4
    rad about the board's y axis), and the expected range at each pixel is built by
    hand from cv2.Rodrigues plus the ray/plane intersection n . (s*ray - t) = 0,
    range = |s*ray|, deliberately not via g.rodrigues or any helper plane_depth_map
    itself calls. Agreement here means the implementation's rotation matches an
    independent derivation, not just its own internal convention.
    """
    rv = np.array([0.0, 0.4, 0.0])
    tv = np.array([0.1, -0.05, 2.0])
    Km = _K_mat(K)
    shape = (540, 960)
    dm = T.plane_depth_map(layout_true, rv, tv, Km, shape)

    R, _ = cv2.Rodrigues(rv)
    n = R @ np.array([0.0, 0.0, 1.0])
    Kinv = np.linalg.inv(Km)

    for u, v in [(483, 280), (100, 280), (880, 280), (483, 50), (483, 500)]:
        ray = Kinv @ np.array([float(u), float(v), 1.0])
        s = (n @ tv) / (n @ ray)
        pt = s * ray
        expected = float(np.linalg.norm(pt))
        assert dm[v, u] == pytest.approx(expected, rel=1e-9), (u, v, dm[v, u], expected)


def test_plane_depth_asymmetry_would_reverse_under_a_transposed_rotation(K, layout_true):
    """A property that holds for R but fails for R.T, at the same oblique pose as above.

    Pitching the board by +0.4 rad about y tilts it so the left side of the frame is
    farther from the camera than the right side. R.T is, for this single-axis
    rotation, the same as rotating by -0.4 rad, which tilts the board the other way
    and makes the left side nearer instead. So this ordering is not just non-trivial,
    it is a direct witness of R vs R.T: a transposed rotation, or any convention that
    silently swaps R for its inverse, flips this inequality rather than merely
    perturbing it.
    """
    rv = np.array([0.0, 0.4, 0.0])
    tv = np.array([0.1, -0.05, 2.0])
    Km = _K_mat(K)
    dm = T.plane_depth_map(layout_true, rv, tv, Km, (540, 960))
    left = dm[280, 100]
    centre = dm[280, round(K["cx"])]
    right = dm[280, 880]
    assert left > centre > right, (left, centre, right)


def test_marker_ranges_order_reverses_under_a_transposed_rotation(layout_true):
    """marker_ranges at a tilted pose: markers further along the tilt axis are nearer.

    Pitching by +0.4 rad about the board's y axis brings markers with larger board-x
    closer to the camera (their board-frame x maps partly onto -z in camera coords).
    Marker 201 sits at board x = 0 (the gauge) and 402 at the largest board x, so 402
    must be nearer than 201. Under R.T this same pose is a -0.4 rad pitch, which
    pushes larger-x markers further away instead, reversing the inequality. So this
    ordering is rotation-convention-sensitive with an obvious expected sign, not just
    a plausible-looking number.
    """
    rv = np.array([0.0, 0.4, 0.0])
    tv = np.array([0.0, 0.0, 2.0])
    r = T.marker_ranges(layout_true, rv, tv)
    assert r[402] < r[201] - 0.05, r


def _flat_scene(depth_m=2.0, shape=(40, 60)):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    depth = np.full(shape, float(depth_m))
    mask = np.ones(shape, dtype=bool)
    return img, depth, mask


def test_synthesise_at_zero_added_turbidity_is_the_identity():
    img, depth, mask = _flat_scene()
    out = T.synthesise(img, depth, mask, B=np.array([150.0, 160.0, 60.0]),
                       dbeta=np.zeros(3))
    assert np.array_equal(out, img)


def test_synthesise_scales_contrast_by_exactly_exp_minus_delta_tau():
    """The claim the whole design rests on: any two pixels at the same range have their
    difference scaled by exp(-dbeta*d), with B cancelling."""
    img, depth, mask = _flat_scene(depth_m=2.0)
    img[0, 0] = [200, 200, 200]
    img[0, 1] = [40, 40, 40]
    dbeta = np.array([0.30, 0.30, 0.30])
    out = T.synthesise(img, depth, mask, B=np.array([150.0, 150.0, 150.0]), dbeta=dbeta)
    before = img[0, 0].astype(float) - img[0, 1].astype(float)
    after = out[0, 0].astype(float) - out[0, 1].astype(float)
    assert np.allclose(after, before * np.exp(-0.30 * 2.0), atol=1.0)


def test_synthesise_contrast_is_independent_of_B():
    """B is known to r^2 ~ 0.5. If contrast depended on it, the design would be unsound.
    """
    img, depth, mask = _flat_scene(depth_m=2.0)
    img[0, 0] = [200, 200, 200]
    img[0, 1] = [40, 40, 40]
    dbeta = np.full(3, 0.30)
    a = T.synthesise(img, depth, mask, B=np.full(3, 60.0), dbeta=dbeta)
    b = T.synthesise(img, depth, mask, B=np.full(3, 190.0), dbeta=dbeta)
    ca = a[0, 0].astype(float) - a[0, 1].astype(float)
    cb = b[0, 0].astype(float) - b[0, 1].astype(float)
    assert np.allclose(ca, cb, atol=1.0)
    # And the DC level genuinely does move, so the test above is not vacuous.
    assert not np.allclose(a[0, 0], b[0, 0], atol=1.0)


def test_synthesise_leaves_masked_out_pixels_untouched():
    img, depth, mask = _flat_scene()
    mask[:, :30] = False
    out = T.synthesise(img, depth, mask, B=np.full(3, 150.0), dbeta=np.full(3, 0.5))
    assert np.array_equal(out[:, :30], img[:, :30])
    assert not np.array_equal(out[:, 30:], img[:, 30:])


def test_synthesise_treats_nan_depth_as_off_plane():
    img, depth, mask = _flat_scene()
    depth[:, :30] = np.nan
    out = T.synthesise(img, depth, mask, B=np.full(3, 150.0), dbeta=np.full(3, 0.5))
    assert np.array_equal(out[:, :30], img[:, :30])


def test_synthesise_scales_each_channel_by_its_own_dbeta():
    """Distinct per-channel dbeta must scale each channel by its own factor.

    All the tests above use dbeta=0 or a single value shared across channels, so a
    mutant that reverses the channel order of both B and dbeta (a BGR/RGB mixup)
    passes every one of them. Here B and dbeta are both distinct per channel, and
    each channel's contrast is checked against its own exp(-dbeta[c]*d), which a
    channel-order bug cannot satisfy.
    """
    img, depth, mask = _flat_scene(depth_m=2.0)
    img[0, 0] = [200, 100, 50]
    img[0, 1] = [40, 20, 10]
    B = np.array([150.0, 90.0, 70.0])
    dbeta = np.array([0.6, 0.3, 0.1])
    out = T.synthesise(img, depth, mask, B=B, dbeta=dbeta)
    before = img[0, 0].astype(float) - img[0, 1].astype(float)
    after = out[0, 0].astype(float) - out[0, 1].astype(float)
    expected = before * np.exp(-dbeta * 2.0)
    assert np.allclose(after, expected, atol=1.0)


def test_synthesise_moves_pixels_toward_B_as_turbidity_grows():
    img, depth, mask = _flat_scene(depth_m=3.0)
    img[:] = 20
    out = T.synthesise(img, depth, mask, B=np.full(3, 180.0), dbeta=np.full(3, 3.0))
    assert np.all(out > 100), "heavy turbidity must wash a dark scene toward the veil"


def _blurred_square(sigma, side=120, canvas=240, black=30, white=220):
    img = np.full((canvas, canvas), white, dtype=np.uint8)
    off = (canvas - side) // 2
    img[off:off + side, off:off + side] = black
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    corners = np.array([[off, off], [off + side, off],
                        [off + side, off + side], [off, off + side]], dtype=float)
    return img, corners


def _asymmetric_edge_fixture(canvas=240, side=48, black=30.0, white=220.0, patch=170.0,
                             patch_inset=4, patch_width=2):
    """A black square with a lighter ring a few pixels inward of every edge, mimicking a
    white data cell just past a real marker's black border, while the region right
    outside the square stays clean white sheet.

    The symmetric black-square fixtures used elsewhere in this file are direction
    invariant by construction: abs(t90 - t10) does not care which way the normal points
    when both sides of the boundary look the same. This fixture breaks that symmetry on
    purpose, so a broken outward-normal flip changes the result instead of passing
    unnoticed.
    """
    img = np.full((canvas, canvas), white, dtype=np.uint8)
    off = (canvas - side) // 2
    img[off:off + side, off:off + side] = black
    lo, hi = patch_inset, patch_inset + patch_width
    ring = np.zeros((canvas, canvas), dtype=bool)
    ring[off + lo:off + side - lo, off + lo:off + side - lo] = True
    ring[off + hi:off + side - hi, off + hi:off + side - hi] = False
    img[ring] = patch
    corners = np.array([[off, off], [off + side, off],
                        [off + side, off + side], [off, off + side]], dtype=float)
    return img, corners


def test_edge_width_direction_matters_for_an_asymmetric_boundary():
    """The outward-normal flip is load-bearing: disabling it changes 92.5% of 200 real
    detections, several by 2 to 4 px against a 4 to 8 px signal, but no prior test could
    catch it because the only fixture is a symmetric black square.

    Here the inward side carries structure (a ring standing in for a white data cell)
    and the outward side is clean sheet. A correct outward flip keeps the far (reach_px)
    reach in the clean sheet and the bounded near reach short of the ring, giving a
    small, finite width. Flipping the normal the wrong way instead pushes the far reach
    inward, through the ring and back to black, producing a white-black-white profile
    that every sample rejects, so edge_width comes back nan instead of a number.
    """
    img, corners = _asymmetric_edge_fixture()
    w = T.edge_width(img, corners)
    assert np.isfinite(w), w
    assert w < 3.0, w


def test_edge_width_is_small_for_a_sharp_edge():
    img, corners = _blurred_square(sigma=0.0)
    assert T.edge_width(img, corners) < 2.5


def test_edge_width_grows_with_a_known_gaussian():
    widths = [T.edge_width(*_blurred_square(sigma=s)) for s in (0.0, 1.0, 2.0, 3.0)]
    assert widths == sorted(widths), widths
    assert widths[-1] > widths[0] + 2.0


def test_edge_width_recovers_the_gaussian_scale():
    """For a Gaussian edge the 10-90 rise is about 2.563 * sigma. Recovering that scale
    (not just monotonicity) is what makes a measured trend interpretable as sigma(tau)."""
    for sigma in (1.5, 2.5):
        w = T.edge_width(*_blurred_square(sigma=sigma))
        assert w == pytest.approx(2.563 * sigma, rel=0.25), (sigma, w)


def test_trials_at_tau_keeps_the_trial_set_and_rescores_the_outcome():
    """The denominator must not move when the imagery degrades."""
    pred = pd.DataFrame({
        "frame_idx": [0, 0, 1, 1], "marker_id": [201, 301, 201, 301],
        "size_mm": [149.4, 74.7, 149.4, 74.7], "pred_px": [90.0, 45.0, 80.0, 40.0],
        "present": [1, 1, 1, 1],
    })
    out = T.trials_at_tau(pred, {0: {201}, 1: set()})
    assert len(out) == len(pred), "trial count must be tau-invariant"
    assert list(out.columns) == ["frame_idx", "marker_id", "size_mm", "pred_px",
                                "detected"]
    assert out.detected.tolist() == [1, 0, 0, 0]


def test_trials_at_tau_scores_a_frame_with_no_detections_as_all_misses():
    pred = pd.DataFrame({
        "frame_idx": [7], "marker_id": [201], "size_mm": [149.4], "pred_px": [30.0],
        "present": [1],
    })
    out = T.trials_at_tau(pred, {})
    assert out.detected.tolist() == [0]
    assert len(out) == 1
