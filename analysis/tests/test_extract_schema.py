"""Schema guard. Importable without ROS -- that separation is the point of Stage 0."""
from analysis.extract_bags import FRAMES_COLUMNS, IMU_COLUMNS


def test_frames_schema():
    assert FRAMES_COLUMNS == ["frame_idx", "stamp"]


def test_imu_schema_carries_orientation_and_gyro():
    assert IMU_COLUMNS == [
        "stamp", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"
    ]


def test_module_imports_without_ros():
    """Guards the ROS-free boundary: constants must not drag rclpy in at import time."""
    import analysis.extract_bags as m
    assert m.FRAMES_COLUMNS
