#!/usr/bin/env python3
"""Stage 0: decode a rosbag into a plain PNG + CSV dataset. ROS container ONLY.

This is the only file in the repo that imports ROS. Everything downstream reads the
dataset it writes, so the analysis never needs rclpy and can pin its own OpenCV.

ROS imports are deliberately deferred into main() so the schema constants stay
importable from the ROS-free analysis image (analysis/tests/test_extract_schema.py).

Usage (inside the ROS 2 Jazzy container):
    source /opt/ros/jazzy/setup.bash
    python3 analysis/extract_bags.py /bags/test1/alan1_0.mcap dataset/test1

Bags ship as .mcap.zstd; decompress first:
    zstd -d alan1_0.mcap.zstd -o alan1_0.mcap
"""
import csv
import os
import sys

FRAMES_COLUMNS = ["frame_idx", "stamp"]
IMU_COLUMNS = ["stamp", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"]

IMAGE_TOPIC = "left/image_rect_color"
INFO_TOPIC = "left/camera_info"
IMU_TOPIC = "imu/data"


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(bag, out_dir):
    import cv2
    import numpy as np
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    frames_f = open(os.path.join(out_dir, "frames.csv"), "w", newline="")
    imu_f = open(os.path.join(out_dir, "imu.csv"), "w", newline="")
    frames_w = csv.writer(frames_f); frames_w.writerow(FRAMES_COLUMNS)
    imu_w = csv.writer(imu_f); imu_w.writerow(IMU_COLUMNS)

    idx = 0
    wrote_info = False
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic.endswith(INFO_TOPIC) and not wrote_info:
            m = deserialize_message(data, get_message(types[topic]))
            K = np.array(m.k).reshape(3, 3)
            with open(os.path.join(out_dir, "camera_info.yaml"), "w") as f:
                yaml.safe_dump({
                    "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                    "width": int(m.width), "height": int(m.height),
                    "distortion_model": m.distortion_model,
                    "d": [float(x) for x in m.d],
                }, f, sort_keys=True)
            wrote_info = True
        elif topic.endswith(IMU_TOPIC):
            m = deserialize_message(data, get_message(types[topic]))
            q, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
            imu_w.writerow([_stamp(m.header), q.x, q.y, q.z, q.w,
                            w.x, w.y, w.z, a.x, a.y, a.z])
        elif topic.endswith(IMAGE_TOPIC):
            m = deserialize_message(data, get_message(types[topic]))
            # bgra8 -> bgr; ascontiguousarray because the sliced view is read-only
            # and non-contiguous, which segfaults cv2 downstream.
            img = np.ascontiguousarray(
                np.frombuffer(m.data, np.uint8)
                  .reshape(m.height, m.width, 4)[:, :, :3])
            cv2.imwrite(os.path.join(out_dir, "frames", f"{idx:06d}.png"), img)
            frames_w.writerow([idx, _stamp(m.header)])
            idx += 1

    frames_f.close(); imu_f.close()
    print(f"wrote {idx} frames to {out_dir}")
    if not wrote_info:
        raise SystemExit("no camera_info found -- wrong topic prefix?")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
