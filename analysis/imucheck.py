"""Stage 4: validate PnP attitude against the IMU.

The IMU is the ONLY sensor here independent of the camera, so this is the only
non-circular accuracy check the dataset supports. It is nearly free: the ZED's
left_cam_imu_transform is effectively identity (2.3 cm, <0.3 deg), so the IMU frame
is the camera frame and no extrinsic calibration is required.

Two checks:
  - gravity: IMU roll/pitch is absolute and drift-free, so a vertical board's normal
    must stay perpendicular to gravity.
  - yaw turns: over a few seconds gyro integration barely drifts, so PnP delta-yaw
    can be compared against integrated gyro delta-yaw.
"""
import numpy as np

from analysis import geometry as g


def gravity_in_camera(accel):
    """Unit gravity direction in the camera frame (IMU frame == camera frame here)."""
    a = np.asarray(accel, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        raise ValueError("degenerate accelerometer sample")
    return a / n


def board_normal_in_camera(rv):
    """The board plane is z=0, so its normal is the pose's third column."""
    return g.rodrigues(np.asarray(rv, float)[None])[0][:, 2]


def tilt_residual_deg(rv, accel, board_to_world_deg=0.0):
    """Angle by which the board normal departs from perpendicular-to-gravity.

    A vertical board has a horizontal normal, so normal . gravity == 0. Deviation is
    board tilt (fixed for a run, fit once) plus PnP attitude error.
    """
    n = board_normal_in_camera(rv)
    gv = gravity_in_camera(accel)
    dev = np.degrees(np.arcsin(np.clip(abs(n @ gv), -1.0, 1.0)))
    return dev - board_to_world_deg


def yaw_from_quat(q):
    """Yaw (rad) from an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def integrate_gyro_yaw(stamps, gyro_z):
    """Trapezoidal integration of yaw rate over a segment (rad)."""
    return float(np.trapezoid(np.asarray(gyro_z, float), np.asarray(stamps, float)))


def pnp_delta_yaw(rv0, rv1):
    """Relative yaw (rad) between two PnP poses."""
    R0 = g.rodrigues(np.asarray(rv0, float)[None])[0]
    R1 = g.rodrigues(np.asarray(rv1, float)[None])[0]
    Rr = R0.T @ R1
    theta = np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))
    return float(theta)
