import numpy as np
import pytest

from analysis import geometry as g
from analysis import layout as L


def _synth(layout_true, K, poses, noise_px=0.0, seed=0):
    """Project the true board through every pose; return obs_by_frame + flat arrays."""
    rng = np.random.default_rng(seed)
    obs, uv, fr, mrk = {}, [], [], []
    for fi, (rv, tv) in enumerate(poses):
        m = np.arange(9)
        Xb = g.board_pts(layout_true, m)
        px = g.project(Xb, rv[None], tv[None], np.zeros(9, dtype=int), **K)
        px = px + rng.normal(0, noise_px, px.shape)
        obs[(0, fi)] = {g.IDS[i]: px[i] for i in range(9)}
        for i in range(9):
            uv.append(px[i]); fr.append(fi); mrk.append(i)
    return obs, np.array(uv), np.array(fr), np.array(mrk)


def _Kmat(K):
    return np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])


def test_bootstrap_recovers_a_known_layout(layout_true, K, synth_views):
    obs, *_ = _synth(layout_true, K, synth_views)
    est = L.bootstrap_layout(obs, _Kmat(K))
    np.testing.assert_allclose(est[:, :2], layout_true[:, :2], atol=2e-3)


def test_bootstrap_pins_201_as_the_gauge(layout_true, K, synth_views):
    obs, *_ = _synth(layout_true, K, synth_views)
    est = L.bootstrap_layout(obs, _Kmat(K))
    np.testing.assert_allclose(est[0], [0, 0, 0], atol=1e-12)


def test_bootstrap_survives_pixel_noise(layout_true, K, synth_views):
    obs, *_ = _synth(layout_true, K, synth_views, noise_px=0.4, seed=3)
    est = L.bootstrap_layout(obs, _Kmat(K))
    np.testing.assert_allclose(est[:, :2], layout_true[:, :2], atol=1e-2)


def test_focal_profile_has_a_minimum_at_the_true_focal_length(layout_true, K, synth_views):
    """The identifiability check. A flat curve here means f is unidentifiable."""
    obs, uv, fr, mrk = _synth(layout_true, K, synth_views, noise_px=0.2, seed=5)
    prob = L.FixedK(uv, fr, mrk, len(synth_views), K["cx"], K["cy"])
    costs = L.profile_focal(prob, layout_true, _Kmat(K),
                            scales=[0.7, 0.85, 1.0, 1.2, 1.45])
    best = min(costs, key=costs.get)
    assert best == pytest.approx(1.0, abs=0.16)
    # Non-flat: the wrong scales must cost visibly more than the right one.
    assert costs[1.45] > 2 * costs[best]


def test_board_pnp_recovers_the_camera_pose(layout_true, K, synth_views):
    obs, *_ = _synth(layout_true, K, synth_views)
    rv, tv = synth_views[0]
    rv_est, tv_est = L.board_pnp(layout_true, obs[(0, 0)], _Kmat(K))
    np.testing.assert_allclose(tv_est, tv, atol=1e-5)
    np.testing.assert_allclose(rv_est, rv, atol=1e-5)


def test_layout_yaml_roundtrip(tmp_path, layout_true):
    p = tmp_path / "board_layout.yaml"
    L.save_layout(layout_true, str(p))
    np.testing.assert_allclose(L.load_layout(str(p)), layout_true, atol=1e-12)
