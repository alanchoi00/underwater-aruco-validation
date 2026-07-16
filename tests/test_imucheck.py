import numpy as np
import pytest

from analysis import imucheck as ic


def test_gravity_is_normalised():
    v = ic.gravity_in_camera(np.array([-0.721, -0.828, 9.739]))
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_gravity_points_along_the_dominant_axis():
    v = ic.gravity_in_camera(np.array([0.0, 0.0, 9.81]))
    np.testing.assert_allclose(v, [0, 0, 1], atol=1e-9)


def test_board_normal_at_zero_rotation_is_camera_z():
    np.testing.assert_allclose(ic.board_normal_in_camera(np.zeros(3)), [0, 0, 1], atol=1e-12)


def test_tilt_residual_is_zero_for_a_vertical_board_and_level_camera():
    """A vertical board's normal is horizontal, so it is perpendicular to gravity."""
    rv = np.zeros(3)                       # board normal along camera +z
    accel = np.array([0.0, 9.81, 0.0])     # gravity along camera +y => z is horizontal
    assert ic.tilt_residual_deg(rv, accel) == pytest.approx(0.0, abs=1e-6)


def test_tilt_residual_grows_when_the_board_normal_tips_toward_gravity():
    rv = np.array([np.deg2rad(20.0), 0.0, 0.0])   # tip the normal 20 deg about x
    accel = np.array([0.0, 9.81, 0.0])
    assert ic.tilt_residual_deg(rv, accel) == pytest.approx(20.0, abs=0.5)


def test_yaw_from_identity_quaternion_is_zero():
    assert ic.yaw_from_quat(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_yaw_from_quat_recovers_a_known_rotation():
    a = np.deg2rad(35.0)
    q = np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])   # (x,y,z,w)
    assert ic.yaw_from_quat(q) == pytest.approx(a)


def test_integrate_gyro_yaw_recovers_a_constant_rate_turn():
    t = np.arange(0.0, 5.0, 1 / 60.0)
    gz = np.full_like(t, 0.2)               # 0.2 rad/s for ~5 s
    assert ic.integrate_gyro_yaw(t, gz) == pytest.approx(1.0, abs=0.02)


def test_pnp_delta_yaw_matches_a_known_relative_rotation():
    rv0 = np.array([0.0, 0.0, 0.0])
    rv1 = np.array([0.0, np.deg2rad(30.0), 0.0])
    assert abs(ic.pnp_delta_yaw(rv0, rv1)) == pytest.approx(np.deg2rad(30.0), abs=1e-6)
