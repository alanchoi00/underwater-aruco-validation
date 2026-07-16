import numpy as np
import pandas as pd
import pytest

from analysis import metrics as m


def _det(rows):
    return pd.DataFrame(rows, columns=["frame_idx", "marker_id", "apparent_px"])


def test_pixel_budget_for_the_5x5_aruco_grid():
    assert m.pixel_budget_px(5) == 21


def test_pixel_budget_scales_with_grid_size():
    assert m.pixel_budget_px(4) == 18
    assert m.pixel_budget_px(6) == 24


def test_mis_id_rate_is_zero_for_board_only_detections():
    df = _det([[0, 201, 100.0], [0, 301, 50.0], [1, 201, 99.0]])
    assert m.mis_id_rate(df) == pytest.approx(0.0)


def test_mis_id_rate_counts_off_board_ids():
    df = _det([[0, 201, 100.0], [0, 7, 30.0], [1, 999, 12.0], [1, 202, 90.0]])
    assert m.mis_id_rate(df) == pytest.approx(0.5)


def test_detection_rate_by_px_bin():
    df = _det([[0, 201, 25.0], [1, 201, 26.0], [2, 201, 60.0]])
    out = m.detection_rate_by_px_bin(df, {201: 10}, bins=[20, 40, 80])
    lo = out[(out.marker_id == 201) & (out.px_lo == 20)].iloc[0]
    assert lo.n_detected == 2
    assert lo.rate == pytest.approx(0.2)


def test_furthest_detection_range_takes_the_last_frame_a_marker_appeared_in():
    df = _det([[i, 201, 30.0] for i in range(10)])
    ranges = {i: 1.0 + 0.5 * i for i in range(10)}
    assert m.furthest_detection_range(df, ranges)[201] == pytest.approx(5.5)


def test_furthest_detection_range_ignores_off_board_ids():
    df = _det([[0, 201, 30.0], [1, 7, 30.0]])
    out = m.furthest_detection_range(df, {0: 1.0, 1: 9.0})
    assert set(out) == {201}
    assert out[201] == pytest.approx(1.0)


def test_jitter_std_of_a_constant_is_zero():
    assert m.jitter_std([1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_jitter_std_matches_numpy():
    v = [1.0, 2.0, 3.0, 4.0]
    assert m.jitter_std(v) == pytest.approx(np.std(v, ddof=1))


def test_range_from_apparent_px_inverts_the_pinhole_relation():
    """Z = f * S / s_px. The MEASURED 149.4 mm marker at 100 px, f=797.54 -> ~1.19 m.

    Under the nominal 155.64 mm this would read 1.24 m -- the 4.1% print-scale error.
    """
    z = m.range_from_apparent_px(100.0, 201, 797.54)
    assert z == pytest.approx(797.54 * 0.1494 / 100.0, rel=1e-9)
    assert z == pytest.approx(1.191, abs=1e-3)


def test_range_from_apparent_px_is_inversely_proportional():
    a = m.range_from_apparent_px(50.0, 201, 797.54)
    b = m.range_from_apparent_px(100.0, 201, 797.54)
    assert a == pytest.approx(2 * b)


def test_incidence_angle_is_zero_looking_straight_at_the_board():
    """A board facing the camera (normal along +z, i.e. no rotation) reads 0 deg."""
    assert m.incidence_angle_deg(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_incidence_angle_grows_as_the_board_yaws_away():
    """Rotating the board 40 deg about camera y tilts its normal 40 deg off boresight."""
    import cv2
    rv, _ = cv2.Rodrigues(np.array([[np.cos(np.deg2rad(40)), 0, np.sin(np.deg2rad(40))],
                                     [0, 1, 0],
                                     [-np.sin(np.deg2rad(40)), 0, np.cos(np.deg2rad(40))]]))
    assert m.incidence_angle_deg(rv.ravel()) == pytest.approx(40.0, abs=0.5)


def test_binned_stats_reports_mean_and_std_per_bin():
    df = pd.DataFrame({"r": [1.0, 1.2, 3.0], "e": [0.01, 0.03, 0.5]})
    out = m.binned_stats(df, "e", "r", bins=[0.0, 2.0, 4.0])
    lo = out[out.bin_lo == 0.0].iloc[0]
    assert lo.n == 2
    assert lo["mean"] == pytest.approx(0.02)
    assert lo["std"] == pytest.approx(np.std([0.01, 0.03], ddof=1))


def test_binned_stats_marks_empty_bins_without_crashing():
    df = pd.DataFrame({"r": [1.0], "e": [0.01]})
    out = m.binned_stats(df, "e", "r", bins=[0.0, 2.0, 4.0])
    hi = out[out.bin_lo == 2.0].iloc[0]
    assert hi.n == 0
    assert np.isnan(hi["mean"])


def test_pose_error_vs_reference_is_zero_on_noise_free_synthetic(layout_true, K):
    # A perfect projection must score ~zero against its own board reference.
    from analysis import geometry as gg
    rv = np.array([0.05, -0.2, 0.02])
    tv = np.array([0.03, -0.01, 1.4])
    Xb = gg.board_pts(layout_true, np.arange(9))
    px = gg.project(Xb, rv[None], tv[None], np.zeros(9, dtype=int), **K)
    obs = {(0, 0): {gg.IDS[i]: px[i] for i in range(9)}}
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    out = m.pose_error_vs_reference(obs, layout_true, Km)
    assert len(out) == 9
    assert out.trans_err_m.max() < 5e-3
    assert out.rot_err_deg.max() < 1.0


def test_pose_error_vs_reference_skips_frames_without_a_reference(layout_true, K):
    # One marker alone cannot be scored: excluding it leaves no reference.
    from analysis import geometry as gg
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    Xb = gg.board_pts(layout_true, np.array([0]))
    px = gg.project(Xb, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.2]]),
                    np.zeros(1, dtype=int), **K)
    out = m.pose_error_vs_reference({(0, 0): {201: px[0]}}, layout_true, Km)
    assert len(out) == 0


def test_detection_trials_includes_the_misses(layout_true, K):
    """The whole point: a rate needs negatives, and a detection table only has hits."""
    from analysis import geometry as gg
    rv = np.array([0.0, 0.0, 0.0]); tv = np.array([0.0, 0.0, 1.4])
    Xb = gg.board_pts(layout_true, np.arange(9))
    px = gg.project(Xb, rv[None], tv[None], np.zeros(9, dtype=int), **K)
    # Only 4 of the 9 markers were actually "detected" this frame.
    seen = {gg.IDS[i]: px[i] for i in range(4)}
    obs = {(0, 0): seen}
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    t = m.detection_trials(obs, layout_true, Km, 960, 540)
    assert set(t.columns) == {"frame_idx", "marker_id", "size_mm", "pred_px", "detected"}
    assert t.detected.sum() >= 1          # some hits
    assert (1 - t.detected).sum() >= 1    # and crucially some MISSES
    assert t.detected.isin([0, 1]).all()


def test_detection_trials_predicts_size_for_undetected_markers(layout_true, K):
    """An undetected marker still gets a predicted apparent size from the board pose."""
    from analysis import geometry as gg
    rv = np.array([0.0, 0.0, 0.0]); tv = np.array([0.0, 0.0, 1.4])
    Xb = gg.board_pts(layout_true, np.arange(9))
    px = gg.project(Xb, rv[None], tv[None], np.zeros(9, dtype=int), **K)
    seen = {gg.IDS[i]: px[i] for i in range(4)}
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    t = m.detection_trials({(0, 0): seen}, layout_true, Km, 960, 540)
    missed = t[t.detected == 0]
    assert len(missed) >= 1
    assert (missed.pred_px > 0).all()     # predicted, not NaN


def test_detection_trials_excludes_markers_not_fully_in_frame(layout_true, K):
    """A marker whose corners fall outside the image is not a fair trial."""
    from analysis import geometry as gg
    rv = np.array([0.0, 0.0, 0.0]); tv = np.array([0.0, 0.0, 0.30])   # very close
    Xb = gg.board_pts(layout_true, np.arange(9))
    px = gg.project(Xb, rv[None], tv[None], np.zeros(9, dtype=int), **K)
    seen = {gg.IDS[i]: px[i] for i in range(3)}
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    t = m.detection_trials({(0, 0): seen}, layout_true, Km, 960, 540)
    # At 0.30 m most of the board spills outside 960x540, so trials must be dropped.
    assert len(t) < 9


def test_rate_by_bin_computes_a_true_rate():
    t = pd.DataFrame({"size_mm": [44.4] * 10, "pred_px": [25.0] * 10,
                      "detected": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]})
    out = m.rate_by_bin(t, bins=[21, 28])
    row = out.iloc[0]
    assert row.n == 10
    assert row.n_detected == 3
    assert row.rate == pytest.approx(0.3)


def test_rate_by_bin_pools_same_size_markers():
    """302-305 are all 44.4mm; pooling them multiplies samples per bin."""
    t = pd.DataFrame({"size_mm": [44.4] * 8, "pred_px": [30.0] * 8,
                      "detected": [1] * 8})
    out = m.rate_by_bin(t, bins=[28, 38])
    assert len(out) == 1              # one pooled row, not four per-marker rows
    assert out.iloc[0].n == 8


def test_rate_by_bin_reports_a_confidence_interval():
    t = pd.DataFrame({"size_mm": [44.4] * 100, "pred_px": [30.0] * 100,
                      "detected": [1] * 50 + [0] * 50})
    row = m.rate_by_bin(t, bins=[28, 38]).iloc[0]
    assert row.ci_lo < row.rate < row.ci_hi
    assert 0.0 <= row.ci_lo and row.ci_hi <= 1.0
