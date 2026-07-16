import cv2
import numpy as np
import pytest

from analysis import geometry as g


def test_ids_are_sorted_and_201_is_the_gauge_anchor():
    assert g.IDS[0] == 201
    assert g.IDS == sorted(g.IDS)
    assert g.IDX[201] == 0
    assert len(g.IDS) == 9


def test_sizes_are_the_MEASURED_print_not_the_nominal():
    """The board printed at ~95.9%. Using nominal sizes inflates every range by 4.1%."""
    assert g.SIZES[201] == pytest.approx(0.1494)
    assert g.SIZES[301] == pytest.approx(0.0747)
    assert g.SIZES[302] == pytest.approx(0.0444)
    assert g.SIZES[401] == pytest.approx(0.0356)


def test_print_scale_is_recorded():
    assert g.PRINT_SCALE == pytest.approx(0.9589, abs=1e-3)


def test_measured_sizes_are_consistent_with_one_uniform_print_scale():
    """A printer scales uniformly; a big spread here would mean something else is wrong."""
    nominal = {201: 0.15564, 301: 0.07782, 302: 0.04669, 401: 0.03696}
    ratios = [g.SIZES[m] / n for m, n in nominal.items()]
    assert max(ratios) - min(ratios) < 0.015     # ruler noise on a 44 mm square


def test_local_corners_are_a_centred_square_in_cv2_order():
    c = g.local_corners(201)
    h = 0.1494 / 2                 # MEASURED print, not the nominal 0.15564
    assert c.shape == (4, 3)
    np.testing.assert_allclose(c[0], [-h, h, 0])   # top-left
    np.testing.assert_allclose(c[1], [h, h, 0])    # top-right
    np.testing.assert_allclose(c[2], [h, -h, 0])   # bottom-right
    np.testing.assert_allclose(c[3], [-h, -h, 0])  # bottom-left
    assert np.all(c[:, 2] == 0)


def test_rodrigues_matches_opencv():
    rng = np.random.default_rng(1)
    rv = rng.normal(size=(7, 3))
    ours = g.rodrigues(rv)
    for i in range(len(rv)):
        expected, _ = cv2.Rodrigues(rv[i])
        np.testing.assert_allclose(ours[i], expected, atol=1e-10)


def test_rodrigues_handles_zero_rotation():
    np.testing.assert_allclose(g.rodrigues(np.zeros((1, 3)))[0], np.eye(3), atol=1e-12)


def test_board_pts_gauge_marker_returns_its_local_corners(layout_true):
    pts = g.board_pts(layout_true, np.array([0]))
    np.testing.assert_allclose(pts[0], g.local_corners(201), atol=1e-12)


def test_board_pts_are_coplanar(layout_true):
    pts = g.board_pts(layout_true, np.arange(9))
    assert np.all(np.abs(pts[:, :, 2]) < 1e-12)


def test_board_pts_translates_and_rotates_in_plane(layout_true):
    layout = np.zeros((9, 3))
    layout[1] = [0.5, 0.25, np.pi / 2]
    pts = g.board_pts(layout, np.array([1]))[0]
    h = g.SIZES[202] / 2
    # +90 deg about board z sends local (-h, h) -> (-h, -h), then translate.
    np.testing.assert_allclose(pts[0], [0.5 - h, 0.25 - h, 0.0], atol=1e-12)


def test_project_pnp_roundtrip_recovers_pose(layout_true, K, synth_views):
    """Project a known board at a known pose, then solve PnP back to it."""
    rv, tv = synth_views[0]
    mrk = np.arange(9)
    Xb = g.board_pts(layout_true, mrk)
    fr = np.zeros(9, dtype=int)
    uv = g.project(Xb, rv[None], tv[None], fr, **K)

    Kmat = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    ok, rv_est, tv_est = cv2.solvePnP(
        Xb.reshape(-1, 3).astype(np.float32), uv.reshape(-1, 2).astype(np.float32),
        Kmat, None, flags=cv2.SOLVEPNP_IPPE)
    assert ok
    np.testing.assert_allclose(tv_est.ravel(), tv, atol=1e-6)
    np.testing.assert_allclose(rv_est.ravel(), rv, atol=1e-6)


def test_project_is_sensitive_to_focal_length(layout_true, K, synth_views):
    """Guards the pilot's trap: a frozen f must not silently look correct."""
    rv, tv = synth_views[0]
    Xb = g.board_pts(layout_true, np.arange(9))
    fr = np.zeros(9, dtype=int)
    a = g.project(Xb, rv[None], tv[None], fr, **K)
    b = g.project(Xb, rv[None], tv[None], fr, **{**K, "fx": K["fx"] * 1.1})
    assert np.abs(a - b).max() > 5.0


def test_board_centroid_is_the_mean_of_the_marker_centres(layout_true):
    c = g.board_centroid(layout_true)
    np.testing.assert_allclose(c, layout_true[:, :2].mean(axis=0), atol=1e-12)


def test_board_centroid_is_offset_from_the_201_gauge(layout_true):
    """201 is the bundle's gauge and sits at the board edge, not its middle."""
    c = g.board_centroid(layout_true)
    assert np.linalg.norm(c) > 0.15          # ~208 mm away from marker 201
    np.testing.assert_allclose(layout_true[g.IDX[201]][:2], [0, 0], atol=1e-12)
